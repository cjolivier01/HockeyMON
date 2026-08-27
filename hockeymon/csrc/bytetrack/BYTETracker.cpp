#include "BYTETracker.h"
#include "Lapjv.h"
#include "Utils.h"

#include <torch/torch.h>

#include <cstdlib>
#include <limits>
#include <sstream>
#include <unistd.h>

namespace hm {
namespace tracker {

namespace {

at::Tensor _unscalar(at::Tensor&& t) {
  if (t.ndimension() == 0 && t.numel() == 1) {
    // convert scalar into shape [1]
    return t.unsqueeze(0);
  }
  return std::move(t);
}

std::string tensor_summary(const at::Tensor& tensor) {
  std::ostringstream ss;
  ss << "sizes=" << tensor.sizes() << ", dtype=" << tensor.scalar_type()
     << ", device=" << tensor.device() << ", numel=" << tensor.numel();
  return ss.str();
}

int64_t checked_scalar_int64(
    const at::Tensor& tensor,
    const std::string& context) {
  TORCH_CHECK(
      tensor.numel() == 1,
      "Expected scalar tensor in ",
      context,
      ", got ",
      tensor_summary(tensor));
  try {
    return tensor.reshape({-1})[0].item<int64_t>();
  } catch (const c10::Error& exc) {
    TORCH_CHECK(
        false,
        "Failed int64 scalar conversion in ",
        context,
        ": ",
        exc.what_without_backtrace(),
        "; tensor ",
        tensor_summary(tensor));
  }
}

int checked_scalar_int(const at::Tensor& tensor, const std::string& context) {
  const int64_t value = checked_scalar_int64(tensor, context);
  TORCH_CHECK(
      value >= std::numeric_limits<int>::min() &&
          value <= std::numeric_limits<int>::max(),
      "Integer scalar out of int range in ",
      context,
      ": value=",
      value,
      "; tensor ",
      tensor_summary(tensor));
  return static_cast<int>(value);
}

} // namespace

BYTETracker::BYTETracker(ByteTrackConfig config)
    : BaseTracker(
          /*momentums=*/std::nullopt,
          /*num_frames_retain=*/config.track_buffer_size),
      config_(std::move(config)) {}

BYTETracker::~BYTETracker() = default;

void BYTETracker::reset() {
  Super::reset();
  track_calls_since_last_empty_ = 0;
  lost_tentative_count_ = 0;
  reacquired_count_ = 0;
  lost_tracks_.clear();
}

at::Tensor BYTETracker::adjust_detection_boxes(at::Tensor det_bboxes) {
  return det_bboxes;
}

std::tuple<at::Tensor, at::Tensor> BYTETracker::assign_ids(
    const std::vector<long>& ids,
    const at::Tensor& det_bboxes_initial,
    const at::Tensor& det_labels,
    const at::Tensor& det_scores,
    bool weight_iou_with_det_scores,
    float match_iou_thr) {
  // Get track_bboxes
  std::vector<at::Tensor> track_bboxes_tensors;
  track_bboxes_tensors.reserve(ids.size());
  at::Tensor det_bboxes = adjust_detection_boxes(det_bboxes_initial);
  for (int id : ids) {
    const auto& mean = tracks_[id].mean;
    float values[4] = {mean(0, 0), mean(0, 1), mean(0, 2), mean(0, 3)};
    at::Tensor tensor = torch::from_blob(&values, {4}, at::kFloat)
                            .clone()
                            .to(det_bboxes.device());
    track_bboxes_tensors.emplace_back(tensor);
  }
  at::Tensor track_bboxes;
  if (!track_bboxes_tensors.empty()) {
    track_bboxes = at::stack(track_bboxes_tensors);
  } else {
    float fake_data = 0.0;
    track_bboxes = at::from_blob(&fake_data, {0, 4}, at::kFloat)
                       .clone()
                       .to(det_bboxes.device());
  }

  // PT("track_bboxes", track_bboxes);
  track_bboxes = bbox_cxcyah_to_xyxy(track_bboxes);
  //_PT(track_bboxes);

  // assert(track_bboxes.size(0) == det_bboxes.size(0));
  // Compute distance
  at::Tensor ious = bbox_overlaps(track_bboxes, det_bboxes);

  //_PT(ious);

  if (weight_iou_with_det_scores) {
    ious *= det_scores;
  }

  // _PT(ious);

  // Support multi-class association
  vector<long> track_labels_vec;
  track_labels_vec.reserve(ids.size());
  for (int id : ids) {
    std::ostringstream ctx;
    ctx << "BYTETracker::assign_ids track label"
        << " track_id=" << id << " ids_count=" << ids.size()
        << " det_count=" << det_bboxes.size(0);
    track_labels_vec.push_back(checked_scalar_int(tracks_.at(id).labels.back(), ctx.str()));
  }
  at::Tensor track_labels = at::from_blob(
                                track_labels_vec.data(),
                                {static_cast<long>(track_labels_vec.size())},
                                at::kLong)
                                .to(det_bboxes.device());

  at::Tensor cate_match =
      (det_labels.unsqueeze(0) == track_labels.unsqueeze(1)).to(at::kLong);
  at::Tensor cate_cost = (1 - cate_match) * 1e6;

  at::Tensor dists = (1 - ious + cate_cost).cpu();

  auto dists_mat = tensor_to_vector_2d(dists);

  // Bipartite match
  std::vector<int> row, col;
  if (dists_mat.size() > 0) {
    row.resize(ids.size(), 0);
    col.resize(det_bboxes.size(0), 0);
    lapjv::lapjv(dists_mat, row, col, true, 1.0 - match_iou_thr);
  } else {
    row.resize(ids.size(), -1);
    col.resize(det_bboxes.size(0), -1);
  }

  // Convert Eigen results to PyTorch tensors
  at::Tensor row_tensor =
      at::from_blob(row.data(), {static_cast<long>(row.size())}, at::kInt)
          .to(at::kLong);
  at::Tensor col_tensor =
      at::from_blob(col.data(), {static_cast<long>(col.size())}, at::kInt)
          .to(at::kLong);
  //_PT(row_tensor);
  //_PT(col_tensor);
  return make_tuple(row_tensor, col_tensor);
}

inline at::Tensor index_select(
    const at::Tensor& tensor,
    const at::Tensor& indices) {
  return at::index_select(tensor, 0, indices);
}

inline at::Tensor copy_to_dest_mask(
    at::Tensor& dest,
    const at::Tensor& dest_mask,
    const at::Tensor& src) {
  assert(dest_mask.dtype() == at::kBool);
  assert(dest_mask.size(0) == dest.size(0));
  // Boolean flags into dest, src size will be sum or True flags
  return dest.masked_scatter(dest_mask, src);
}

inline at::Tensor copy_valid_with_valid_dest_mask(
    at::Tensor dest,
    const at::Tensor& valid_dest,
    const at::Tensor& src_ids,
    const at::Tensor& src_id_indexes,
    const at::Tensor& valid_src) {
  // valid = tentative_match_det_inds > -1
  // first_unmatch_det_ids[valid] = torch.tensor(self.unconfirmed_ids)[
  //     tentative_match_det_inds[valid]]
  assert(valid_src.dtype() == at::kBool);
  assert(valid_dest.dtype() == at::kBool);
  assert(valid_dest.size(0) == dest.size(0));
  assert(src_ids.dtype() == at::kLong);
  assert(src_id_indexes.dtype() == at::kLong);
  assert(valid_src.size(0) == dest.size(0));
  // std::cout << "------------" << std::endl;
  // PT("ids", src_ids);
  // PT("valid_dest", valid_dest);
  // PT("valid_src", valid_src);
  // PT("src_id_indexes", src_id_indexes);
  assert(valid_src.size(0) == src_id_indexes.size(0));
  at::Tensor valid_indexes = bool_mask_select(src_id_indexes, valid_src);
  // PT("valid_indexes", valid_indexes);
  at::Tensor valid_ids = index_select(src_ids, valid_indexes);
  // PT("valid_ids", valid_ids);
  // PT("dest", dest);
  return copy_to_dest_mask(dest, valid_dest, valid_ids);
}

std::unordered_map<std::string, at::Tensor> BYTETracker::track(
    std::unordered_map<std::string, at::Tensor>&& data) {
  const int64_t frame_id =
      get_map_tensor("data", kFrameId, data).item<int64_t>();
  auto bboxes = get_map_tensor("data", kBBoxes, data);
  auto scores = get_map_tensor("data", kScores, data);
  auto labels = get_map_tensor("data", kLabels, data);
  at::Tensor ids;
  if (empty()) {
    track_calls_since_last_empty_ = 0;
    //_PT(scores);
    auto valid_inds = scores > config_.init_track_thr;
    // std::cout << valid_inds << std::endl;
    scores = bool_mask_select(scores, valid_inds);
    bboxes = bool_mask_select(bboxes, valid_inds);
    labels = bool_mask_select(labels, valid_inds);
    //_PT(scores);
    // std::cout << scores << std::endl;
    // scores = scores[valid_inds];
    // bboxes = bboxes[valid_inds];
    // labels = labels[valid_inds];
    const int64_t num_new_tracks = bboxes.size(0);
    ids = at::arange(num_tracks_, num_tracks_ + num_new_tracks).to(labels);
    num_tracks_ += num_new_tracks;
  } else {
    ++track_calls_since_last_empty_;
    // 0. Init
    ids = at::full(
        {bboxes.size(0)},
        at::Scalar(-1),
        at::TensorOptions().dtype(labels.dtype()).device(labels.device()));
    // Get the detection bboxes for the first association (high scores)
    auto first_det_inds = scores > config_.obj_score_thrs_high;
    // PT("first_det_inds", first_det_inds);
    auto first_det_bboxes = bool_mask_select(bboxes, first_det_inds);
    auto first_det_labels = bool_mask_select(labels, first_det_inds);
    auto first_det_scores = bool_mask_select(scores, first_det_inds);
    auto first_det_ids = bool_mask_select(ids, first_det_inds);
    // _PT(first_det_ids);

    // Get the detection bboxes for the second association (lower scores, but
    // above some threshold)
    assert(first_det_inds.dtype() == at::kBool);
    auto second_det_inds =
        (~first_det_inds) & (scores > config_.obj_score_thrs_low);
    auto second_det_bboxes = bool_mask_select(bboxes, second_det_inds);
    auto second_det_labels = bool_mask_select(labels, second_det_inds);
    auto second_det_scores = bool_mask_select(scores, second_det_inds);
    auto second_det_ids = bool_mask_select(ids, second_det_inds);

    //
    // 1. Use Kalman Filter to predict current location
    //
    // Iterate confirmed ids
    std::vector<long> not_lost_confirmed_tracks;
    std::vector<long> confirmed_ids, unconfirmed_ids;
    not_lost_confirmed_tracks.reserve(tracks_.size());
    confirmed_ids.reserve(tracks_.size());
    unconfirmed_ids.reserve(tracks_.size());
    for (auto& track_item : tracks_) {
      STrack& track = track_item.second;
      const std::size_t id = track_item.first;
      if (track.state == STrack::State::Tentative) {
        unconfirmed_ids.emplace_back(id);
        continue;
      }
      confirmed_ids.emplace_back(id);
      assert(!track.frame_ids.empty());
      int64_t last_frame_id = track.frame_ids.back().item<int64_t>();
      if (last_frame_id != frame_id - 1) {
        assert(lost_tracks_.count(id));
        // std::cout << "Track " << id << " has been lost for "
        //           << (frame_id - last_frame_id) << " frames." << std::end;
        // Skipped at least one frame since this was last tracked
        // if (track.mean[7] != 0.0) {
        //   std::cout << "d;opwqdnhjjqwpdnhqwpd: track id " << id << std::endl;
        // }
        // track.mean[7] = 0.0; // wth is this?
        track.mean[7] /= 2;
        // if (id == 3) {
        //   usleep(0);
        // }
      } else {
        assert(!lost_tracks_.count(id));
        not_lost_confirmed_tracks.emplace_back(id);
      }
      // if (id == 3) {
      //   std::cout << track.mean << std::endl;
      // }
      kalman_filter_.predict(track.mean, track.covariance);
      // if (id == 3) {
      //   std::cout << track.mean << std::endl;
      //   kalman_filter_.predict(track.mean, track.covariance);
      //   std::cout << track.mean << std::endl;
      // }
    }
    // std::cout << "confirmed_ids: " << confirmed_ids << std::endl;
    //
    // 2. first match
    //
    auto first_match_tuple = assign_ids(
        confirmed_ids,
        first_det_bboxes,
        first_det_labels,
        first_det_scores,
        config_.weight_iou_with_det_scores,
        config_.match_iou_thrs_high);
    at::Tensor first_match_track_inds =
        std::move(std::get<0>(first_match_tuple));
    at::Tensor first_match_det_inds = std::move(std::get<1>(first_match_tuple));
    // std::cout << "Pass: " << track_pass_ << std::endl;
    //_PT(first_match_track_inds);
    //_PT(first_match_det_inds);

    // '-1' mean a detection box is not matched with tracklets in
    // previous frame
    // PT("first_match_det_inds", first_match_det_inds);
    at::Tensor valid = first_match_det_inds > -1;
    // PT("first_det_ids", first_det_ids);
    // PT("valid", valid);
    first_det_ids = first_det_ids.cpu();
    first_det_ids = copy_valid_with_valid_dest_mask(
        first_det_ids,
        valid,
        vector_to_tensor(confirmed_ids, first_det_ids.device()),
        first_match_det_inds,
        valid);
    // PT("first_det_ids", first_det_ids);

    // first_det_ids[valid] =
    //     vector_to_tensor(confirmed_ids)[first_match_det_inds[valid]].to(labels);
    auto valid_dev = valid.to(first_det_bboxes.device());
    auto first_match_det_bboxes = bool_mask_select(first_det_bboxes, valid_dev);
    auto first_match_det_labels = bool_mask_select(first_det_labels, valid_dev);
    auto first_match_det_scores = bool_mask_select(first_det_scores, valid_dev);
    auto first_match_det_ids = bool_mask_select(first_det_ids, valid);
    assert((first_match_det_ids > -1).all().item<bool>());

    //_PT(first_match_det_scores);

    auto invalid_dev = ~valid_dev;
    auto invalid = ~valid;
    auto first_unmatch_det_bboxes =
        bool_mask_select(first_det_bboxes, invalid_dev);
    auto first_unmatch_det_labels =
        bool_mask_select(first_det_labels, invalid_dev);
    auto first_unmatch_det_scores =
        bool_mask_select(first_det_scores, invalid_dev);
    // TODO: everything on same device, I think it goes wrong in assign_ids
    auto first_unmatch_det_ids = bool_mask_select(first_det_ids, invalid);
    assert((first_unmatch_det_ids == -1).all().item<bool>());

    //_PT(first_unmatch_det_scores);

    // PT("first_unmatch_det_bboxes", first_unmatch_det_bboxes);
    // PT("first_unmatch_det_labels", first_unmatch_det_labels);
    // PT("first_unmatch_det_scores", first_unmatch_det_scores);
    // PT("first_unmatch_det_ids", first_unmatch_det_ids);

    //
    // 3. use unmatched detection bboxes from the first match to match
    // the unconfirmed tracks
    //
    // std::cout << "unconfirmed_ids=" << unconfirmed_ids << std::endl;
    auto tentative_match_tuple = assign_ids(
        unconfirmed_ids,
        first_unmatch_det_bboxes,
        first_unmatch_det_labels,
        first_unmatch_det_scores,
        config_.weight_iou_with_det_scores,
        config_.match_iou_thrs_tentative);
    at::Tensor tentative_match_track_inds =
        std::move(std::get<0>(tentative_match_tuple));
    at::Tensor tentative_match_det_inds =
        std::move(std::get<1>(tentative_match_tuple));
    //_PT(tentative_match_track_inds);
    //_PT(tentative_match_det_inds);
    valid = tentative_match_det_inds > -1;

    // if (at::sum(valid).item<int64_t>()) {
    //   _PT(tentative_match_track_inds);
    //   _PT(tentative_match_det_inds);
    // }

    // valid = tentative_match_det_inds > -1
    // first_unmatch_det_ids[valid] = torch.tensor(self.unconfirmed_ids)[
    //     tentative_match_det_inds[valid]].to(labels)

    first_unmatch_det_ids = first_unmatch_det_ids.cpu();
    first_unmatch_det_ids = copy_valid_with_valid_dest_mask(
        first_unmatch_det_ids,
        valid,
        vector_to_tensor(unconfirmed_ids, first_unmatch_det_ids.device()),
        tentative_match_det_inds,
        valid);

    // std::cout << first_unmatch_det_ids << std::endl;
    //
    // 4. second match for unmatched tracks from the first match
    //
    std::vector<long> debug_not_lost_in_last_frame;
    std::vector<long> debug_was_first_match;
    std::vector<long> debug_confirmed_was_not_in_first_match;
    debug_not_lost_in_last_frame.reserve(confirmed_ids.size());
    debug_was_first_match.reserve(confirmed_ids.size());
    debug_confirmed_was_not_in_first_match.reserve(confirmed_ids.size());

    std::vector<long> first_unmatch_track_ids;
    first_unmatch_track_ids.reserve(confirmed_ids.size());
    for (std::size_t i = 0, n = confirmed_ids.size(); i < n; ++i) {
      auto id = confirmed_ids[i];
      // tracklet is not matched in the first match
      std::ostringstream ctx;
      ctx << "BYTETracker::track first_match_track_inds"
          << " frame_id=" << frame_id << " index=" << i << " track_id=" << id
          << " confirmed_count=" << confirmed_ids.size()
          << " first_match_shape=" << first_match_track_inds.sizes();
      bool case_1 = checked_scalar_int(first_match_track_inds[i], ctx.str()) == -1;
      // tracklet is not lost in the previous frame
      std::ostringstream frame_ctx;
      frame_ctx << "BYTETracker::track last frame id"
                << " frame_id=" << frame_id << " track_id=" << id;
      bool case_2 =
          checked_scalar_int64(tracks_[id].frame_ids.back(), frame_ctx.str()) ==
          frame_id - 1;
      if (debug_) {
        if (case_1) {
          debug_confirmed_was_not_in_first_match.emplace_back(id);
        } else {
          debug_was_first_match.emplace_back(id);
        }
        if (case_2) {
          debug_not_lost_in_last_frame.emplace_back(id);
        }
      }
      if (case_1 && case_2) {
        first_unmatch_track_ids.emplace_back(id);
      }
    }

    // std::cout << "not_lost_in_last_frame: " << debug_not_lost_in_last_frame
    //           << ", "
    //           << "was_first_match: " << debug_was_first_match << ", "
    //           << "confirmed_was_not_in_first_match: "
    //           << debug_confirmed_was_not_in_first_match << std::endl;

    auto second_match_tuple = assign_ids(
        first_unmatch_track_ids,
        second_det_bboxes,
        second_det_labels,
        second_det_scores,
        /*weight_iou_with_det_scores=*/false,
        config_.match_iou_thrs_low);
    at::Tensor second_match_det_inds = std::get<1>(second_match_tuple);
    valid = second_match_det_inds > -1;

    // second_det_ids[valid] =
    //     vector_to_tensor(
    //         first_unmatch_track_ids,
    //         second_det_ids.device())[second_match_det_inds[valid]]
    //         .to(ids);
    second_det_ids = second_det_ids.cpu();
    second_det_ids = copy_valid_with_valid_dest_mask(
        second_det_ids,
        valid,
        vector_to_tensor(first_unmatch_track_ids, second_det_ids.device()),
        second_match_det_inds,
        valid);

    //
    // 5. gather all matched detection bboxes from step 2-4
    // we only keep matched detection bboxes in second match, which
    // means the id != -1
    //
    valid = second_det_ids > -1;
    valid_dev = valid.to(second_det_bboxes.device());
    bboxes =
        at::cat({first_match_det_bboxes, first_unmatch_det_bboxes}, /*dim=*/0);
    bboxes = at::cat(
        {bboxes, bool_mask_select(second_det_bboxes, valid_dev)}, /*dim=*/0);

    labels =
        at::cat({first_match_det_labels, first_unmatch_det_labels}, /*dim=*/0);
    labels = at::cat(
        {labels, bool_mask_select(second_det_labels, valid_dev)}, /*dim=*/0);

    scores =
        at::cat({first_match_det_scores, first_unmatch_det_scores}, /*dim=*/0);
    scores = at::cat(
        {scores, bool_mask_select(second_det_scores, valid_dev)}, /*dim=*/0);

    ids = at::cat(
        {first_match_det_ids, first_unmatch_det_ids},
        /*dim=*/0);
    ids = at::cat({ids, bool_mask_select(second_det_ids, valid)}, /*dim=*/0);

    //
    // 6. assign new ids
    //
    at::Tensor new_track_inds = ids == -1;
    // ATTN: SYNC POINT
    const int64_t new_track_inds_sum = new_track_inds.sum().item<int64_t>();
    // TODO: fix this to()
    ids = ids.to(labels);

    // _PT(ids);

    new_track_inds = new_track_inds.to(labels.device());
    at::Tensor new_indices = at::arange(
        num_tracks_,
        num_tracks_ + new_track_inds_sum,
        at::TensorOptions().dtype(labels.dtype()).device(labels.device()));

    // _PT(new_track_inds);
    // _PT(new_indices);

    assert(new_track_inds_sum == new_indices.size(0));
    //_PT(ids);
    ids = copy_to_dest_mask(ids, new_track_inds, new_indices);
    // ids[new_track_inds] =
    //     at::arange(num_tracks_, num_tracks_ + new_track_inds_sum).to(labels);
    num_tracks_ += new_track_inds_sum;

    std::vector<int64_t> current_tracks = tensor_to_int64_vector(ids.cpu());
    std::unordered_set<int64_t> current_tracks_set{
        current_tracks.begin(), current_tracks.end()};
    for (auto not_lost_id : not_lost_confirmed_tracks) {
      if (!current_tracks_set.count(not_lost_id)) {
        STrack& track = tracks_.at(not_lost_id);
        // std::cout << "Lost track " << not_lost_id
        //           << " after tracking for at least " <<
        //           track.frame_ids.size()
        //           << " frames." << std::endl;
        assert(track.state == STrack::State::Tracking);
        track.state = STrack::State::Lost;
        lost_tracks_.emplace(not_lost_id);
      }
    }
  }

  ids = _unscalar(std::move(ids));
  scores = _unscalar(std::move(scores));
  labels = _unscalar(std::move(labels));
  bboxes = _unscalar(std::move(bboxes));

  assert(ids.ndimension() > 0);
  assert(scores.ndimension() > 0);
  assert(labels.ndimension() > 0);
  assert(bboxes.ndimension() > 0);

  track_update(ids.cpu(), bboxes.cpu(), labels.cpu(), scores.cpu(), {frame_id});

  // TODO: Return Tuple[bboxes, labels, scores, ids]
  ++track_pass_;
  data[kIds] = std::move(ids);
  data[kScores] = std::move(scores);
  data[kLabels] = std::move(labels);
  data[kBBoxes] = std::move(bboxes);
  return std::move(data);
}

void BYTETracker::track_update(
    const at::Tensor& ids,
    const at::Tensor& bboxes,
    const at::Tensor& labels,
    const at::Tensor& scores,
    const std::vector<int64_t>& frame_ids) {
  at::Tensor frame_id_tensor = vector_to_tensor(frame_ids, kCpuDevice);
  if (frame_id_tensor.size(0) == 1 && ids.size(0) > 1) {
    frame_id_tensor = frame_id_tensor.repeat(ids.size(0));
    assert(frame_id_tensor.size(0) == ids.size(0));
  }
  assert(ids.numel() == bboxes.size(0));
  assert(ids.numel() == labels.numel());
  assert(ids.numel() == scores.numel());
  std::unordered_map<std::string, const at::Tensor*> kwargs{
      {kIds, &ids},
      {kFrameIds, &frame_id_tensor},
      {kBBoxes, &bboxes},
      {kLabels, &labels},
      {kScores, &scores}};
  if (validate_memos_enabled()) {
    validate_label_tensor(
        "before BaseTracker::update",
        frame_ids.empty() ? -1 : frame_ids.front(),
        labels);
  }
  BaseTracker::update(kwargs);
  if (validate_memos_enabled()) {
    validate_track_memos(
        "after BaseTracker::update",
        frame_ids.empty() ? -1 : frame_ids.front());
  }
}

bool BYTETracker::validate_memos_enabled() const {
  const char* value = std::getenv("HM_TRACKER_VALIDATE_MEMOS");
  return value != nullptr && value[0] != '\0' && value[0] != '0';
}

void BYTETracker::validate_label_tensor(
    const char* stage,
    int64_t frame_id,
    const at::Tensor& labels) const {
  at::Tensor labels_cpu = labels.cpu().reshape({-1});
  for (int64_t i = 0, n = labels_cpu.size(0); i < n; ++i) {
    std::ostringstream ctx;
    ctx << "BYTETracker::validate_label_tensor"
        << " stage=" << stage << " frame_id=" << frame_id
        << " index=" << i << " count=" << n;
    checked_scalar_int(labels_cpu[i], ctx.str());
  }
}

void BYTETracker::validate_track_memos(const char* stage, int64_t frame_id)
    const {
  for (const auto& item : tracks_) {
    const int64_t track_id = item.first;
    const STrack& track = item.second;
    std::size_t label_index = 0;
    for (const at::Tensor& label : track.labels) {
      std::ostringstream ctx;
      ctx << "BYTETracker::validate_track_memos label"
          << " stage=" << stage << " frame_id=" << frame_id
          << " track_id=" << track_id << " label_index=" << label_index
          << " label_count=" << track.labels.size();
      checked_scalar_int(label, ctx.str());
      ++label_index;
    }
  }
}

void BYTETracker::activate_track(int64_t id) {
  STrack& track = tracks_.at(id);
  assert(track.state != STrack::State::Tracking);
  auto lost_iter = lost_tracks_.find(id);
  if (lost_iter != lost_tracks_.end()) {
    assert(track.state == STrack::State::Lost);
    assert(track.frame_ids.size() >= 2);
    auto f_iter = track.frame_ids.rbegin();
    ++f_iter;
    // std::cout << "Re-acquired track " << id << " after being lost for "
    //           << (f_iter[-1]->item<int64_t>() - f_iter->item<int64_t>() - 1)
    //           << " frame(s), for a total of "
    //           << reacquired_count_ << " reacquisitions." << std::endl;
    lost_tracks_.erase(lost_iter);
    assert(track.state == STrack::State::Lost);
    track.state = STrack::State::Tracking;
    ++reacquired_count_;
  } else {
    assert(track.state != STrack::State::Lost);
  }
  track.state = STrack::State::Tracking;
}

void BYTETracker::init_track(
    int64_t id,
    const std::unordered_map<std::string, at::Tensor>& memos) {
  BaseTracker::init_track(id, memos);
  STrack& track = tracks_.at(id);
  assert(track.state == STrack::State::Unknown);
  if (track_calls_since_last_empty_ == 0) {
    activate_track(id);
  } else {
    track.state = STrack::State::Tentative;
  }
  auto bbox =
      bbox_xyxy_to_cxcyah(track.bboxes.back().unsqueeze(0)); // size = (1, 4)
  assert(bbox.ndimension() == 2 && bbox.size(0) == 1);
  bbox = adjust_detection_boxes(bbox);
  std::vector<float> sq_bbox = tensor_to_vector(bbox.squeeze(0).cpu());
  byte_kalman::KAL_DATA kal_data =
      kalman_filter_.initiate({sq_bbox[0], sq_bbox[1], sq_bbox[2], sq_bbox[3]});
  track.mean = kal_data.first;
  track.covariance = kal_data.second;
}

void BYTETracker::update_track(
    int64_t id,
    const std::unordered_map<std::string, at::Tensor>& memos) {
  BaseTracker::update_track(id, memos);
  STrack& track = tracks_.at(id);
  if (track.state == STrack::State::Tentative &&
      track.bboxes.size() >= config_.num_tentatives) {
    activate_track(id);
    assert(track.state == STrack::State::Tracking);
  } else if (track.state == STrack::State::Lost) {
    activate_track(id);
  } else {
    assert(!lost_tracks_.count(id));
  }

  // Check label consistency
  assert(
      track.labels.back().item<int64_t>() == memos.at(kLabels).item<int64_t>());

  auto bbox =
      bbox_xyxy_to_cxcyah(track.bboxes.back().unsqueeze(0)); // size = (1, 4)
  assert(bbox.ndimension() == 2 && bbox.size(0) == 1);
  bbox = adjust_detection_boxes(bbox);
  std::vector<float> sq_bbox = tensor_to_vector(bbox.squeeze(0).cpu());
  byte_kalman::KAL_DATA kal_data = kalman_filter_.update(
      track.mean,
      track.covariance,
      {sq_bbox[0], sq_bbox[1], sq_bbox[2], sq_bbox[3]});
  track.mean = kal_data.first;
  track.covariance = kal_data.second;
}

void BYTETracker::pop_track(int64_t track_id) {
  lost_tracks_.erase(track_id);
  Super::pop_track(track_id);
}

void BYTETracker::pop_invalid_tracks(int64_t frame_id) {
  // We add another condition to BaseTracker's version
  std::vector<std::size_t> invalid_ids;
  invalid_ids.reserve(tracks_.size());
  for (const auto& item : tracks_) {
    const STrack& track = item.second;
    const int64_t last_track_frame_id = track.frame_ids.back().item<int64_t>();
    // case1: disappeared frames >= num_frames_to_keep_lost_tracks
    const bool case1 = frame_id - last_track_frame_id >=
        static_cast<int64_t>(config_.num_frames_to_keep_lost_tracks);
    assert(!case1 || track.state == STrack::State::Lost);
    // We never got past the initial tentative state (short new detection)
    const bool case2 = track.state == STrack::State::Tentative &&
        last_track_frame_id != frame_id;
    if (case1 || case2) {
      const int64_t id = item.first;
      // Should be gone from these by now
      // assert(!last_active_tracks_.count(id));
      // assert(!last_inactive_tracks_.count(id));
      invalid_ids.emplace_back(id);
      if (track.state == STrack::State::Tentative) {
        ++lost_tentative_count_;
        // std::cout << "Lost tentative tracking id " << id << " after "
        //           << track.frame_ids.size() << " frames." << std::endl;
      }
    }
  }
  for (std::size_t id : invalid_ids) {
    pop_track(id);
  }
}

} // namespace tracker
} // namespace hm
