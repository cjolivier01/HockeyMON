#include "hockeymon/csrc/df/DfTrackerCudaStatic.h"

#include "hockeymon/csrc/pytorch/bytetrack_cuda_ops.h"

#include <ATen/Functions.h>

#include <algorithm>

namespace hm {
namespace tracker {
namespace {

constexpr const char* kFrameIdKey = "frame_id";
constexpr const char* kBBoxesKey = "bboxes";
constexpr const char* kLabelsKey = "labels";
constexpr const char* kScoresKey = "scores";
constexpr const char* kIdsKey = "ids";
constexpr const char* kNumDetectionsKey = "num_detections";
constexpr const char* kNumTracksKey = "num_tracks";
constexpr const char* kReidKey = "reid_features";

constexpr float kInvalidCost = 1e6f;
constexpr float kLabelCost = 1e6f;
constexpr float kReidEps = 1e-6f;

int64_t tensor_length(const at::Tensor& tensor, const char* name) {
  TORCH_CHECK(
      tensor.numel() >= 1,
      name,
      " tensor must contain at least one value");
  return tensor.to(at::kLong).item<int64_t>();
}

template <typename T>
void copy_prefix(at::Tensor& dst, const at::Tensor& src, T count) {
  if (count <= 0) {
    return;
  }
  dst.slice(0, 0, count).copy_(src);
}

} // namespace

DfTrackerCudaStatic::DfTrackerCudaStatic(
    ByteTrackConfig config,
    int64_t max_detections,
    int64_t max_tracks,
    int64_t reid_feature_dim,
    float iou_weight,
    float reid_weight,
    float box_momentum,
    float reid_momentum,
    float min_similarity,
    float lost_track_cost,
    c10::Device device)
    : config_(std::move(config)),
      device_(device),
      max_detections_(max_detections),
      max_tracks_(max_tracks),
      reid_feature_dim_(reid_feature_dim),
      iou_weight_(iou_weight),
      reid_weight_(reid_weight),
      box_momentum_(box_momentum),
      reid_momentum_(reid_momentum),
      min_similarity_(min_similarity),
      lost_track_cost_(lost_track_cost) {
  TORCH_CHECK(device_.is_cuda(), "DfTrackerCudaStatic expects a CUDA/HIP GPU device");
  TORCH_CHECK(max_detections_ > 0, "max_detections must be positive");
  TORCH_CHECK(max_tracks_ > 0, "max_tracks must be positive");
  TORCH_CHECK(reid_feature_dim_ >= 0, "reid_feature_dim must be non-negative");
  TORCH_CHECK(
      box_momentum_ >= 0.0f && box_momentum_ <= 1.0f,
      "box_momentum must be in [0, 1]");
  TORCH_CHECK(
      reid_momentum_ >= 0.0f && reid_momentum_ <= 1.0f,
      "reid_momentum must be in [0, 1]");
  reset();
}

void DfTrackerCudaStatic::reset() {
  active_tracks_ = 0;
  next_track_id_ = 0;
  auto long_opts =
      at::TensorOptions().dtype(at::kLong).device(device_);
  auto float_opts =
      at::TensorOptions().dtype(at::kFloat).device(device_);
  track_ids_ = at::full({max_tracks_}, -1, long_opts);
  track_state_ = at::full(
      {max_tracks_},
      static_cast<int64_t>(TrackState::Inactive),
      long_opts);
  track_labels_ = at::full({max_tracks_}, -1, long_opts);
  track_scores_ = at::zeros({max_tracks_}, float_opts);
  track_last_frame_ = at::zeros({max_tracks_}, long_opts);
  track_hits_ = at::zeros({max_tracks_}, long_opts);
  track_bboxes_ = at::zeros({max_tracks_, 4}, float_opts);
  if (reid_feature_dim_ > 0) {
    track_reid_ = at::zeros(
        {max_tracks_, reid_feature_dim_},
        float_opts);
  } else {
    track_reid_ = at::Tensor();
  }
}

at::Tensor DfTrackerCudaStatic::ensure_vector(
    const at::Tensor& tensor,
    at::ScalarType dtype) const {
  auto out = tensor.to(device_, dtype);
  if (out.dim() == 0) {
    out = out.reshape({1});
  }
  TORCH_CHECK(out.dim() == 1, "Expected 1-D tensor");
  return out;
}

at::Tensor DfTrackerCudaStatic::ensure_bboxes(
    const at::Tensor& tensor) const {
  auto out = tensor.to(device_, at::kFloat);
  TORCH_CHECK(
      out.dim() == 2 && out.size(1) == 4,
      "bboxes tensor must have shape [N, 4]");
  return out;
}

at::Tensor DfTrackerCudaStatic::mask_indices(
    const at::Tensor& mask) const {
  if (!mask.defined() || mask.numel() == 0) {
    return at::empty({0}, mask.options().dtype(at::kLong));
  }
  auto mask_bool = mask.to(at::kBool);
  auto options = at::TensorOptions().dtype(at::kLong).device(mask.device());
  auto indices = at::arange(mask_bool.size(0), options);
  return indices.masked_select(mask_bool);
}

std::unordered_map<std::string, at::Tensor> DfTrackerCudaStatic::track(
    std::unordered_map<std::string, at::Tensor>&& data) {
  auto frame_id_it = data.find(kFrameIdKey);
  TORCH_CHECK(frame_id_it != data.end(), "data must contain 'frame_id'");
  auto frame_id_tensor = frame_id_it->second;
  TORCH_CHECK(
      frame_id_tensor.numel() >= 1,
      "frame_id tensor must contain a value");
  auto frame_id_device =
      frame_id_tensor.to(device_, at::kLong).reshape({-1}).slice(0, 0, 1);
  auto frame_id_full = frame_id_device.expand_as(track_last_frame_);

  auto num_det_it = data.find(kNumDetectionsKey);
  TORCH_CHECK(
      num_det_it != data.end(),
      "data must contain 'num_detections' entry");
  TORCH_CHECK(
      num_det_it->second.device().is_cpu(),
      "num_detections must be a CPU tensor to avoid stream syncs");
  int64_t num_detections =
      tensor_length(num_det_it->second, "num_detections");
  TORCH_CHECK(
      num_detections >= 0 && num_detections <= max_detections_,
      "num_detections (",
      num_detections,
      ") exceeds configured max_detections (",
      max_detections_,
      ")");

  auto det_bboxes = ensure_bboxes(data.at(kBBoxesKey));
  auto det_labels = ensure_vector(data.at(kLabelsKey), at::kLong);
  auto det_scores = ensure_vector(data.at(kScoresKey), at::kFloat);

  TORCH_CHECK(
      det_bboxes.size(0) == max_detections_,
      "bboxes tensor first dimension must equal max_detections");
  TORCH_CHECK(
      det_labels.size(0) == max_detections_,
      "labels tensor first dimension must equal max_detections");
  TORCH_CHECK(
      det_scores.size(0) == max_detections_,
      "scores tensor first dimension must equal max_detections");

  bool has_reid_input = false;
  at::Tensor det_reid;
  if (reid_feature_dim_ > 0) {
    auto reid_it = data.find(kReidKey);
    if (reid_it != data.end()) {
      det_reid = reid_it->second.to(device_, at::kFloat);
      if (det_reid.dim() == 1) {
        det_reid = det_reid.reshape({1, -1});
      }
      TORCH_CHECK(det_reid.dim() == 2, "reid_features must be 2-D");
      TORCH_CHECK(
          det_reid.size(0) == max_detections_,
          "reid_features first dimension must equal max_detections");
      TORCH_CHECK(
          det_reid.size(1) == reid_feature_dim_,
          "reid_features second dimension must equal reid_feature_dim");
      has_reid_input = true;
    }
  }

  auto long_opts =
      at::TensorOptions().dtype(at::kLong).device(device_);
  auto det_indices = at::arange(max_detections_, long_opts);
  auto det_valid_mask = det_indices < num_detections;
  if (config_.obj_score_thrs_low > 0.0f) {
    det_valid_mask = det_valid_mask &
        det_scores.ge(config_.obj_score_thrs_low);
  }

  auto track_valid_mask =
      track_state_.ne(static_cast<int64_t>(TrackState::Inactive));

  auto ious = ops::bbox_iou_cuda(track_bboxes_, det_bboxes);
  if (config_.weight_iou_with_det_scores && det_scores.numel() > 0) {
    ious *= det_scores.unsqueeze(0);
  }

  float iou_weight = std::max(0.0f, iou_weight_);
  float reid_weight =
      (has_reid_input ? std::max(0.0f, reid_weight_) : 0.0f);
  float weight_sum = iou_weight + reid_weight;
  if (weight_sum <= 0.0f) {
    iou_weight = 1.0f;
    weight_sum = 1.0f;
  }

  auto sim = ious * iou_weight;
  if (has_reid_input && reid_weight > 0.0f) {
    auto track_norm =
        track_reid_.pow(2).sum(1, true).sqrt().add_(kReidEps);
    auto det_norm =
        det_reid.pow(2).sum(1, true).sqrt().add_(kReidEps);
    auto track_reid_norm = track_reid_ / track_norm;
    auto det_reid_norm = det_reid / det_norm;
    auto appearance =
        at::matmul(track_reid_norm, det_reid_norm.transpose(0, 1));
    sim = sim + appearance * reid_weight;
  }
  sim = sim / weight_sum;

  auto label_match =
      track_labels_.unsqueeze(1).eq(det_labels.unsqueeze(0));
  auto cost = 1.0f - sim;
  cost = cost + (~label_match).to(cost.dtype()) * kLabelCost;
  auto valid_pairs =
      track_valid_mask.unsqueeze(1) & det_valid_mask.unsqueeze(0);
  cost = cost + (~valid_pairs).to(cost.dtype()) * kInvalidCost;
  if (lost_track_cost_ > 0.0f) {
    auto lost_mask =
        track_state_.eq(static_cast<int64_t>(TrackState::Lost));
    cost = cost + lost_mask.unsqueeze(1).to(cost.dtype()) * lost_track_cost_;
  }

  float min_similarity = min_similarity_;
  if (min_similarity < 0.0f) {
    min_similarity = static_cast<float>(config_.match_iou_thrs_low);
  }
  min_similarity = std::min(std::max(min_similarity, 0.0f), 1.0f);
  float cost_limit = 1.0f - min_similarity;

  auto assignments = ops::hungarian_assign_cuda(
      cost,
      max_tracks_,
      max_detections_,
      cost_limit);
  auto track_to_det = assignments.first.to(device_);
  auto det_to_track = assignments.second.to(device_);

  auto matched = track_to_det.ge(0) & track_valid_mask;
  auto safe_det_idx =
      at::where(matched, track_to_det, at::zeros_like(track_to_det));
  auto det_valid_for_track =
      det_valid_mask.index_select(0, safe_det_idx);
  matched = matched & det_valid_for_track;

  auto matched_expanded = matched.unsqueeze(1).expand_as(track_bboxes_);
  auto det_boxes_for_tracks =
      det_bboxes.index_select(0, safe_det_idx);
  auto updated_boxes =
      track_bboxes_ * (1.0f - box_momentum_) +
      det_boxes_for_tracks * box_momentum_;
  track_bboxes_ =
      at::where(matched_expanded, updated_boxes, track_bboxes_);

  auto det_labels_for_tracks =
      det_labels.index_select(0, safe_det_idx);
  track_labels_ =
      at::where(matched, det_labels_for_tracks, track_labels_);

  auto det_scores_for_tracks =
      det_scores.index_select(0, safe_det_idx);
  track_scores_ =
      at::where(matched, det_scores_for_tracks, track_scores_);

  track_last_frame_ =
      at::where(matched, frame_id_full, track_last_frame_);
  track_hits_ = at::where(matched, track_hits_ + 1, track_hits_);

  track_state_.index_put_(
      {matched},
      static_cast<int64_t>(TrackState::Tracking));

  if (has_reid_input && reid_feature_dim_ > 0) {
    auto det_reid_for_tracks =
        det_reid.index_select(0, safe_det_idx);
    auto updated_reid =
        track_reid_ * (1.0f - reid_momentum_) +
        det_reid_for_tracks * reid_momentum_;
    auto matched_reid =
        matched.unsqueeze(1).expand_as(track_reid_);
    track_reid_ =
        at::where(matched_reid, updated_reid, track_reid_);
  }

  auto unmatched = track_valid_mask & (~matched);
  auto tracking_mask =
      track_state_.eq(static_cast<int64_t>(TrackState::Tracking));
  auto to_lost = tracking_mask & unmatched;
  track_state_.index_put_(
      {to_lost},
      static_cast<int64_t>(TrackState::Lost));

  auto frame_delta = frame_id_full - track_last_frame_;
  auto stale =
      track_state_.eq(static_cast<int64_t>(TrackState::Lost)) &
      frame_delta.ge(
          static_cast<int64_t>(config_.num_frames_to_keep_lost_tracks));
  track_state_.index_put_(
      {stale},
      static_cast<int64_t>(TrackState::Inactive));
  track_ids_.index_put_({stale}, -1);
  track_labels_.index_put_({stale}, -1);
  track_scores_.index_put_({stale}, 0);
  track_last_frame_.index_put_({stale}, 0);
  track_hits_.index_put_({stale}, 0);
  track_bboxes_.index_put_({stale}, 0);
  if (reid_feature_dim_ > 0) {
    track_reid_.index_put_({stale}, 0);
  }

  auto unmatched_det =
      det_to_track.lt(0) &
      det_valid_mask &
      det_scores.ge(config_.init_track_thr);
  auto new_det_idx = mask_indices(unmatched_det);
  auto free_mask =
      track_state_.eq(static_cast<int64_t>(TrackState::Inactive));
  auto free_idx = mask_indices(free_mask);
  int64_t num_new =
      std::min<int64_t>(new_det_idx.size(0), free_idx.size(0));
  if (num_new > 0) {
    auto det_idx = new_det_idx.slice(0, 0, num_new);
    auto track_idx = free_idx.slice(0, 0, num_new);
    auto new_boxes = det_bboxes.index_select(0, det_idx);
    auto new_labels = det_labels.index_select(0, det_idx);
    auto new_scores = det_scores.index_select(0, det_idx);
    auto new_ids =
        at::arange(
            next_track_id_,
            next_track_id_ + num_new,
            long_opts);
    next_track_id_ += num_new;

    track_ids_.index_put_({track_idx}, new_ids);
    track_bboxes_.index_put_({track_idx}, new_boxes);
    track_labels_.index_put_({track_idx}, new_labels);
    track_scores_.index_put_({track_idx}, new_scores);
    track_state_.index_put_(
        {track_idx},
        static_cast<int64_t>(TrackState::Tracking));
    track_hits_.index_put_({track_idx}, 1);
    auto frame_id_init = frame_id_device.expand({num_new});
    track_last_frame_.index_put_({track_idx}, frame_id_init);
    if (has_reid_input && reid_feature_dim_ > 0) {
      track_reid_.index_put_(
          {track_idx},
          det_reid.index_select(0, det_idx));
    }
  }

  auto active_mask =
      track_state_.eq(static_cast<int64_t>(TrackState::Tracking));
  auto active_idx = mask_indices(active_mask);
  int64_t num_tracks = active_idx.size(0);
  active_tracks_ = static_cast<std::size_t>(num_tracks);

  auto ids_out = at::full({max_tracks_}, -1, long_opts);
  auto labels_out = at::full({max_tracks_}, -1, long_opts);
  auto scores_out =
      at::zeros({max_tracks_}, det_scores.options());
  auto bboxes_out =
      at::zeros({max_tracks_, 4}, det_bboxes.options());
  at::Tensor reid_out;
  if (reid_feature_dim_ > 0) {
    reid_out = at::zeros(
        {max_tracks_, reid_feature_dim_},
        det_bboxes.options());
  }

  if (num_tracks > 0) {
    copy_prefix(
        ids_out,
        track_ids_.index_select(0, active_idx),
        num_tracks);
    copy_prefix(
        labels_out,
        track_labels_.index_select(0, active_idx),
        num_tracks);
    copy_prefix(
        scores_out,
        track_scores_.index_select(0, active_idx),
        num_tracks);
    copy_prefix(
        bboxes_out,
        track_bboxes_.index_select(0, active_idx),
        num_tracks);
    if (reid_feature_dim_ > 0) {
      copy_prefix(
          reid_out,
          track_reid_.index_select(0, active_idx),
          num_tracks);
    }
  }

  data[kIdsKey] = ids_out;
  data[kLabelsKey] = labels_out;
  data[kScoresKey] = scores_out;
  data[kBBoxesKey] = bboxes_out;
  data[kNumTracksKey] =
      at::full({1}, num_tracks, long_opts);
  data[kNumDetectionsKey] =
      at::full({1}, num_detections, long_opts);
  if (reid_feature_dim_ > 0) {
    data[kReidKey] = reid_out;
  }

  return data;
}

} // namespace tracker
} // namespace hm
