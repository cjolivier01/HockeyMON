#include "HmTracker.h"
#include "Utils.h"

#include <torch/torch.h>

#include <unistd.h>

namespace hm {
namespace tracker {

namespace {
at::Tensor make_square_bounding_boxes(
    const at::Tensor& bounding_boxes,
    float side_length) {
  if (bounding_boxes.sizes().size() != 2 || bounding_boxes.size(1) != 4) {
    throw std::invalid_argument(
        "Input tensor must be of shape [N, 4], where N is the number of bounding boxes");
  }

  const float half_side = side_length / 2.0;

  // Calculate centers of the bounding boxes
  at::Tensor x_center = (bounding_boxes.index({at::indexing::Slice(), 0}) +
                         bounding_boxes.index({at::indexing::Slice(), 2})) /
      2.0;
  at::Tensor y_center = (bounding_boxes.index({at::indexing::Slice(), 1}) +
                         bounding_boxes.index({at::indexing::Slice(), 3})) /
      2.0;

  // Calculate new bounding box coordinates
  at::Tensor x_min = x_center - half_side;
  at::Tensor y_min = y_center - half_side;
  at::Tensor x_max = x_center + half_side;
  at::Tensor y_max = y_center + half_side;

  // Concatenate to form the squared bounding boxes
  at::Tensor squared_boxes = at::stack({x_min, y_min, x_max, y_max}, 1);

  return squared_boxes;
}
} // namespace

HmTracker::HmTracker(HmTrackerConfig hm_config, ByteTrackConfig bt_config)
    : BYTETracker(std::move(bt_config)), hm_config_(std::move(hm_config)) {}

HmTracker::HmTracker(HmByteTrackConfig config)
    : BYTETracker(config), hm_config_(config) {}

HmTracker::~HmTracker() = default;

void HmTracker::reset() {
  Super::reset();
  activated_id_mapping_.clear();
}

at::Tensor HmTracker::adjust_detection_boxes(at::Tensor det_bboxes) {
  assert(det_bboxes.ndimension() == 2 && det_bboxes.size(1) == 4);
  // std::cout << det_bboxes << std::endl;
  switch (hm_config_.prediction_mode) {
    case HmTrackerPredictionMode::BoxCenter:
      return make_square_bounding_boxes(det_bboxes, /*side_length=*/20.0);
    case HmTrackerPredictionMode::BoundingBox:
    default:
      break;
  }
  return det_bboxes;
}

std::unordered_map<std::string, at::Tensor> HmTracker::track(
    std::unordered_map<std::string, at::Tensor>&& data) {
  const int64_t frame_id =
      get_map_tensor("data", kFrameId, data).item<int64_t>();
  if (frame_id == 0) {
    reset();
  }
  std::unordered_map<std::string, at::Tensor> results =
      Super::track(std::move(data));
  if (hm_config_.remove_tentative || hm_config_.return_user_ids) {
    const at::Tensor& ids = results.at(kIds);

    std::vector<int64_t> all_active_ids;
    all_active_ids.reserve(activated_id_mapping_.size());
    for (const auto& item : activated_id_mapping_) {
      all_active_ids.push_back(item.first);
    }
    at::Tensor all_active_ids_tensor =
        vector_to_tensor(all_active_ids, ids.device());
    at::Tensor all_active_mask = at::isin(ids, all_active_ids_tensor);

    [[maybe_unused]] const std::size_t id_count = ids.numel();
    for (auto& item : results) {
      at::Tensor& tensor = item.second;
      if (tensor.dim() == 0) {
        // Ignore scalars
        std::cout << "Ignoring scalar: key = " << item.first
                  << ", value: " << to_string(tensor) << std::endl;
        // _PT(tensor);
        continue;
      } else if (item.first == kFrameId) {
        continue;
      }
      assert(tensor.size(0) == static_cast<int64_t>(id_count));
      // std::cout << item.first << " -> " << tensor.device()
      //           << std::endl;
      tensor = bool_mask_select(tensor, all_active_mask);
    }
    if (hm_config_.return_user_ids) {
      at::Tensor& ids = results.at(kIds);
      std::vector<int64_t> all_ids = tensor_to_int64_vector(ids.cpu());
      std::vector<int64_t> return_user_ids;
      std::vector<at::Tensor> return_ages;
      return_user_ids.reserve(all_ids.size());
      return_ages.reserve(all_ids.size());
      for (int64_t id : all_ids) {
        return_user_ids.emplace_back(activated_id_mapping_.at(id));
        if (hm_config_.return_track_age) {
          return_ages.emplace_back(tracks_.at(id).age());
        }
      }
      results.emplace(
          "user_ids", vector_to_tensor(return_user_ids, ids.device()));
      if (hm_config_.return_track_age && !return_ages.empty()) {
        results.emplace("track_ages", at::cat(return_ages, /*dim=*/0));
      }
    }
  }
  return results;
}

void HmTracker::activate_track(int64_t track_id) {
  Super::activate_track(track_id);
  // May be re-activating a lost track that we already know about
  if (!activated_id_mapping_.count(track_id)) {
    activated_id_mapping_.emplace(track_id, ++total_activated_tracks_count_);
  }
}

void HmTracker::pop_track(int64_t track_id) {
  Super::pop_track(track_id);
  activated_id_mapping_.erase(track_id);
}

} // namespace tracker
} // namespace hm
