#include "BYTETrackerCuda.h"
#include "Utils.h"

#include <vector>

#include <ATen/Functions.h>
#include <ATen/ops/arange.h>
#include <ATen/ops/cat.h>
#include <ATen/ops/diag_embed.h>
#include <ATen/ops/eye.h>
#include <ATen/ops/full.h>
#include <ATen/ops/matmul.h>
#include <ATen/ops/masked_select.h>
#include <ATen/ops/scalar_tensor.h>
#include <ATen/ops/sum.h>
#include <ATen/ops/where.h>

namespace hm {
namespace tracker {

namespace {

constexpr const char* kFrameIdKey = "frame_id";
constexpr const char* kBBoxesKey = "bboxes";
constexpr const char* kLabelsKey = "labels";
constexpr const char* kScoresKey = "scores";
constexpr const char* kIdsKey = "ids";

inline at::Tensor unsqueeze_if_scalar(at::Tensor tensor) {
  if (tensor.ndimension() == 0 && tensor.numel() == 1) {
    return tensor.unsqueeze(0);
  }
  return tensor;
}

inline at::Tensor make_bool_mask(const at::Tensor& tensor) {
  if (!tensor.defined()) {
    return at::Tensor();
  }
  return tensor.to(at::kBool);
}

inline at::Tensor mask_select(const at::Tensor& tensor, const at::Tensor& mask) {
  if (!tensor.defined()) {
    return tensor;
  }
  if (mask.ndimension() == 0 && mask.numel() == 0) {
    return tensor.new_empty({0});
  }
  return tensor.index({mask});
}

inline at::Tensor ensure_int64(const at::Tensor& tensor) {
  return tensor.scalar_type() == at::kLong ? tensor : tensor.to(at::kLong);
}

} // namespace

BYTETrackerCuda::BYTETrackerCuda(ByteTrackConfig config, c10::Device device)
    : config_(std::move(config)), device_(std::move(device)) {
  auto options_f = at::TensorOptions().dtype(at::kFloat).device(device_);
  motion_mat_ = at::eye(8, options_f);
  for (int i = 0; i < 4; ++i) {
    motion_mat_.index_put_({i, 4 + i}, 1.0);
  }
  motion_mat_T_ = motion_mat_.transpose(0, 1).contiguous();

  update_mat_ = at::zeros({4, 8}, options_f);
  update_mat_.slice(1, 0, 4).copy_(at::eye(4, options_f));
  update_mat_T_ = update_mat_.transpose(0, 1).contiguous();

  reset();
}

void BYTETrackerCuda::reset() {
  auto long_opts = at::TensorOptions().dtype(at::kLong).device(device_);
  auto float_opts = at::TensorOptions().dtype(at::kFloat).device(device_);

  track_ids_ = at::empty({0}, long_opts);
  track_states_ = at::empty({0}, long_opts);
  track_labels_ = at::empty({0}, long_opts);
  track_scores_ = at::empty({0}, float_opts);
  track_last_frame_ = at::empty({0}, long_opts);
  track_hits_ = at::empty({0}, long_opts);
  track_mean_ = at::empty({0, 8}, float_opts);
  track_covariance_ = at::empty({0, 8, 8}, float_opts);
  next_track_id_ = 0;
  track_calls_since_last_empty_ = 0;
}

at::Tensor BYTETrackerCuda::ensure_bboxes(const at::Tensor& tensor) const {
  if (tensor.numel() == 0) {
    return tensor.to(device_).reshape({0, 4});
  }
  if (tensor.dim() == 1) {
    TORCH_CHECK(tensor.size(0) == 4, "bbox tensor must have 4 elements");
    return tensor.to(device_).unsqueeze(0);
  }
  TORCH_CHECK(tensor.size(1) == 4, "bbox tensor last dim must be 4");
  return tensor.to(device_);
}

at::Tensor BYTETrackerCuda::ensure_vector(
    const at::Tensor& tensor,
    at::ScalarType dtype) const {
  if (tensor.numel() == 0) {
    return tensor.to(device_, dtype).reshape({0});
  }
  if (tensor.dim() == 0) {
    return tensor.to(device_, dtype).unsqueeze(0);
  }
  return tensor.to(device_, dtype);
}

std::unordered_map<std::string, at::Tensor> BYTETrackerCuda::track(
    std::unordered_map<std::string, at::Tensor>&& data) {
  return run_tracker(std::move(data));
}

std::unordered_map<std::string, at::Tensor> BYTETrackerCuda::run_tracker(
    std::unordered_map<std::string, at::Tensor>&& data) {
  auto frame_id_tensor = data.at(kFrameIdKey);
  TORCH_CHECK(
      frame_id_tensor.numel() >= 1,
      "frame_id tensor must contain a value");
  int64_t frame_id = frame_id_tensor.to(at::kLong).item<int64_t>();

  auto bboxes = ensure_bboxes(data.at(kBBoxesKey));
  auto labels = ensure_vector(data.at(kLabelsKey), at::kLong);
  auto scores = ensure_vector(data.at(kScoresKey), at::kFloat);

  TORCH_CHECK(
      bboxes.size(0) == labels.size(0) && labels.size(0) == scores.size(0),
      "bboxes/labels/scores must have matching length");

  at::Tensor ids;

  if (track_ids_.size(0) == 0) {
    track_calls_since_last_empty_ = 0;
    auto valid = (scores > config_.init_track_thr);
    auto keep = mask_indices(valid);
    if (keep.numel() == 0) {
      data[kIdsKey] = at::empty({0}, labels.options());
      data[kBBoxesKey] = at::empty({0, 4}, bboxes.options());
      data[kLabelsKey] = at::empty({0}, labels.options());
      data[kScoresKey] = at::empty({0}, scores.options());
      return data;
    }
    bboxes = bboxes.index_select(0, keep);
    labels = labels.index_select(0, keep);
    scores = scores.index_select(0, keep);

    auto num = bboxes.size(0);
    ids = at::arange(
              next_track_id_,
              next_track_id_ + num,
              at::TensorOptions().dtype(at::kLong).device(device_))
              .contiguous();
    next_track_id_ += num;
    track_mean_ = at::empty({0, 8}, bboxes.options());
    track_covariance_ = at::empty({0, 8, 8}, bboxes.options());
    init_new_tracks(ids, bboxes, labels, scores, frame_id);
  } else {
    ++track_calls_since_last_empty_;
    if (bboxes.size(0) == 0) {
      ids = at::empty({0}, labels.options());
      mark_unmatched_tracking(at::empty({0}, at::TensorOptions().dtype(at::kLong).device(device_)));
      remove_stale_tracks(frame_id);
      data[kIdsKey] = ids;
      data[kBBoxesKey] = bboxes;
      data[kLabelsKey] = labels;
      data[kScoresKey] = scores;
      return data;
    }

    auto det_count = bboxes.size(0);
    ids = at::full(
        {det_count},
        -1,
        at::TensorOptions().dtype(at::kLong).device(device_));

    auto first_mask = (scores > config_.obj_score_thrs_high);
    auto second_mask =
        (~first_mask) & (scores > config_.obj_score_thrs_low);

    auto first_idx = mask_indices(first_mask);
    auto second_idx = mask_indices(second_mask);

    auto first_bboxes =
        first_idx.numel() ? bboxes.index_select(0, first_idx)
                          : at::empty({0, 4}, bboxes.options());
    auto first_labels =
        first_idx.numel() ? labels.index_select(0, first_idx)
                          : at::empty({0}, labels.options());
    auto first_scores =
        first_idx.numel() ? scores.index_select(0, first_idx)
                          : at::empty({0}, scores.options());
    auto first_det_ids = at::full(
        {first_bboxes.size(0)},
        -1,
        ids.options());
    auto first_track_indices = at::full_like(first_det_ids, -1);

    auto second_bboxes =
        second_idx.numel() ? bboxes.index_select(0, second_idx)
                           : at::empty({0, 4}, bboxes.options());
    auto second_labels =
        second_idx.numel() ? labels.index_select(0, second_idx)
                           : at::empty({0}, labels.options());
    auto second_scores =
        second_idx.numel() ? scores.index_select(0, second_idx)
                           : at::empty({0}, scores.options());
    auto second_det_ids = at::full(
        {second_bboxes.size(0)},
        -1,
        ids.options());
    auto second_track_indices = at::full_like(second_det_ids, -1);

    predict_tracks(frame_id);

    auto states = track_states_;
    auto confirmed_mask =
        states.ne(static_cast<int64_t>(TrackState::Tentative));
    auto unconfirmed_mask =
        states.eq(static_cast<int64_t>(TrackState::Tentative));

    auto confirmed_idx = mask_indices(confirmed_mask);
    auto unconfirmed_idx = mask_indices(unconfirmed_mask);

    MatchResult first_match;
    if (confirmed_idx.numel() > 0 && first_bboxes.size(0) > 0) {
      first_match = assign_tracks(
          confirmed_idx,
          first_bboxes,
          first_labels,
          first_scores,
          config_.weight_iou_with_det_scores,
          config_.match_iou_thrs_high);
      auto valid_det = first_match.det_to_track >= 0;
      auto det_valid_idx = mask_indices(valid_det);
      if (det_valid_idx.numel() > 0) {
        auto matched_tracks_subset =
            first_match.det_to_track.index_select(0, det_valid_idx);
        auto matched_tracks_global =
            confirmed_idx.index_select(0, matched_tracks_subset);
        auto matched_track_ids =
            track_ids_.index_select(0, matched_tracks_global);
        auto matched_det_global = first_idx.index_select(0, det_valid_idx);
        ids.index_put_({matched_det_global}, matched_track_ids);
        first_det_ids.index_put_({det_valid_idx}, matched_track_ids);
        first_track_indices.index_put_({det_valid_idx}, matched_tracks_global);
      }
    }

    // Unmatched detections from first step
    auto unmatched_first_mask =
        first_det_ids.lt(0);
    auto unmatched_first_idx = mask_indices(unmatched_first_mask);
    auto first_unmatch_bboxes =
        unmatched_first_idx.numel()
        ? first_bboxes.index_select(0, unmatched_first_idx)
        : at::empty({0, 4}, bboxes.options());
    auto first_unmatch_labels =
        unmatched_first_idx.numel()
        ? first_labels.index_select(0, unmatched_first_idx)
        : at::empty({0}, labels.options());
    auto first_unmatch_scores =
        unmatched_first_idx.numel()
        ? first_scores.index_select(0, unmatched_first_idx)
        : at::empty({0}, scores.options());
    auto first_unmatch_det_ids =
        unmatched_first_idx.numel()
        ? first_det_ids.index_select(0, unmatched_first_idx).clone()
        : at::empty({0}, ids.options());
    auto first_unmatch_track_indices =
        unmatched_first_idx.numel()
        ? first_track_indices.index_select(0, unmatched_first_idx).clone()
        : at::empty({0}, ids.options());

    if (unconfirmed_idx.numel() > 0 && first_unmatch_bboxes.size(0) > 0) {
      auto tentative_match = assign_tracks(
          unconfirmed_idx,
          first_unmatch_bboxes,
          first_unmatch_labels,
          first_unmatch_scores,
          config_.weight_iou_with_det_scores,
          config_.match_iou_thrs_tentative);
      auto valid_det = tentative_match.det_to_track >= 0;
      auto det_valid_idx = mask_indices(valid_det);
      if (det_valid_idx.numel() > 0) {
        auto matched_tracks_subset =
            tentative_match.det_to_track.index_select(0, det_valid_idx);
        auto matched_tracks_global =
            unconfirmed_idx.index_select(0, matched_tracks_subset);
        auto matched_track_ids =
            track_ids_.index_select(0, matched_tracks_global);
        auto global_det_idx =
            unmatched_first_idx.index_select(0, det_valid_idx);
        ids.index_put_({first_idx.index_select(0, global_det_idx)}, matched_track_ids);
        first_unmatch_det_ids.index_put_({det_valid_idx}, matched_track_ids);
        first_unmatch_track_indices.index_put_(
            {det_valid_idx}, matched_tracks_global);
      }
    }

    // Second match
    at::Tensor second_selected_idx;
    if (first_match.track_to_det.defined()) {
      auto track_unmatched_mask = first_match.track_to_det.lt(0);
      auto recent_mask =
          track_last_frame_.index_select(0, confirmed_idx)
              .eq(frame_id - 1);
      auto selectable_mask = track_unmatched_mask & recent_mask;
      second_selected_idx = mask_indices(selectable_mask);
    } else {
      second_selected_idx = confirmed_idx.new_empty({0});
    }

    if (second_selected_idx.numel() > 0 && second_bboxes.size(0) > 0) {
      auto selected_tracks =
          confirmed_idx.index_select(0, second_selected_idx);
      auto second_match = assign_tracks(
          selected_tracks,
          second_bboxes,
          second_labels,
          second_scores,
          /*weight_with_scores=*/false,
          config_.match_iou_thrs_low);
      auto valid_det = second_match.det_to_track >= 0;
      auto det_valid_idx = mask_indices(valid_det);
      if (det_valid_idx.numel() > 0) {
        auto matched_tracks_subset =
            second_match.det_to_track.index_select(0, det_valid_idx);
        auto matched_tracks_global =
            selected_tracks.index_select(0, matched_tracks_subset);
        auto matched_track_ids =
            track_ids_.index_select(0, matched_tracks_global);
        auto matched_det_global = second_idx.index_select(0, det_valid_idx);
        ids.index_put_({matched_det_global}, matched_track_ids);
        second_det_ids.index_put_({det_valid_idx}, matched_track_ids);
        second_track_indices.index_put_(
            {det_valid_idx}, matched_tracks_global);
      }
    }

    auto first_match_bboxes = first_bboxes;
    auto first_unmatch_bboxes_all = first_unmatch_bboxes;
    auto second_valid_mask = second_det_ids.ge(0);
    auto second_valid_idx = mask_indices(second_valid_mask);

    auto cat_bboxes = std::vector<at::Tensor>{
        first_match_bboxes,
        first_unmatch_bboxes_all};
    auto cat_labels = std::vector<at::Tensor>{first_labels, first_unmatch_labels};
    auto cat_scores = std::vector<at::Tensor>{first_scores, first_unmatch_scores};
    auto cat_ids = std::vector<at::Tensor>{first_det_ids, first_unmatch_det_ids};
    auto cat_track_indices = std::vector<at::Tensor>{
        first_track_indices,
        first_unmatch_track_indices};

    if (second_valid_idx.numel() > 0) {
      cat_bboxes.push_back(second_bboxes.index_select(0, second_valid_idx));
      cat_labels.push_back(second_labels.index_select(0, second_valid_idx));
      cat_scores.push_back(second_scores.index_select(0, second_valid_idx));
      cat_ids.push_back(second_det_ids.index_select(0, second_valid_idx));
      cat_track_indices.push_back(
          second_track_indices.index_select(0, second_valid_idx));
    }

    bboxes = at::cat(cat_bboxes, 0);
    labels = at::cat(cat_labels, 0);
    scores = at::cat(cat_scores, 0);
    auto combined_ids = at::cat(cat_ids, 0);
    auto combined_track_indices = at::cat(cat_track_indices, 0);

    auto new_track_mask = combined_ids.lt(0);
    auto new_track_idx = mask_indices(new_track_mask);
    auto new_track_count = new_track_idx.size(0);
    if (new_track_count > 0) {
      auto new_ids = at::arange(
              next_track_id_,
              next_track_id_ + new_track_count,
              at::TensorOptions().dtype(at::kLong).device(device_));
      next_track_id_ += new_track_count;
      combined_ids.index_put_({new_track_idx}, new_ids);
    }
    ids = combined_ids;

    auto matched_tracks = combined_track_indices.ge(0);
    auto matched_track_idx = mask_indices(matched_tracks);
    at::Tensor matched_global_idx;
    if (matched_track_idx.numel() > 0) {
      matched_global_idx =
          combined_track_indices.index_select(0, matched_track_idx);
    } else {
      matched_global_idx = combined_track_indices.new_empty({0});
    }

    if (matched_global_idx.numel() > 0) {
      mark_unmatched_tracking(matched_global_idx);
    } else {
      mark_unmatched_tracking(at::empty({0}, matched_global_idx.options()));
    }

    auto matched_mask = matched_tracks;
    auto matched_det_idx = matched_track_idx;

    if (matched_det_idx.numel() > 0) {
      auto update_track_indices =
          combined_track_indices.index_select(0, matched_det_idx);
      auto matched_ids =
          ids.index_select(0, matched_det_idx);
      (void)matched_ids;
      update_tracks(
          update_track_indices,
          bboxes.index_select(0, matched_det_idx),
          labels.index_select(0, matched_det_idx),
          scores.index_select(0, matched_det_idx),
          frame_id);
    }

    if (new_track_idx.numel() > 0) {
      init_new_tracks(
          ids.index_select(0, new_track_idx),
          bboxes.index_select(0, new_track_idx),
          labels.index_select(0, new_track_idx),
          scores.index_select(0, new_track_idx),
          frame_id);
    }
  }

  ids = unsqueeze_if_scalar(ids);
  scores = unsqueeze_if_scalar(scores);
  labels = unsqueeze_if_scalar(labels);
  bboxes = unsqueeze_if_scalar(bboxes);

  remove_stale_tracks(frame_id);

  data[kIdsKey] = ids;
  data[kScoresKey] = scores;
  data[kLabelsKey] = labels;
  data[kBBoxesKey] = bboxes;
  return data;
}

void BYTETrackerCuda::predict_tracks(int64_t frame_id) {
  if (track_ids_.size(0) == 0) {
    return;
  }

  auto confirmed_mask =
      track_states_.ne(static_cast<int64_t>(TrackState::Tentative));
  auto not_prev_frame =
      track_last_frame_.ne(frame_id - 1);
  auto degrade_mask = confirmed_mask & not_prev_frame;
  auto degrade_idx = mask_indices(degrade_mask);
  if (degrade_idx.numel() > 0) {
    auto values =
        track_mean_.index({degrade_idx, 7}) / 2.0;
    track_mean_.index_put_({degrade_idx, 7}, values);
  }
  kalman_predict(track_mean_, track_covariance_);
}

void BYTETrackerCuda::remove_stale_tracks(int64_t frame_id) {
  if (track_ids_.size(0) == 0) {
    return;
  }
  auto lost_mask =
      track_states_.eq(static_cast<int64_t>(TrackState::Lost));
  auto frame_delta = at::full_like(track_last_frame_, frame_id) - track_last_frame_;
  auto too_long = lost_mask &
      frame_delta.ge(static_cast<int64_t>(config_.num_frames_to_keep_lost_tracks));
  auto stale_tent =
      track_states_.eq(static_cast<int64_t>(TrackState::Tentative)) &
      track_last_frame_.ne(frame_id);
  auto remove_mask = too_long | stale_tent;
  auto remove_idx = mask_indices(remove_mask);
  if (remove_idx.numel() == 0) {
    return;
  }
  auto keep_idx = mask_indices(~remove_mask);
  track_ids_ = track_ids_.index_select(0, keep_idx);
  track_states_ = track_states_.index_select(0, keep_idx);
  track_labels_ = track_labels_.index_select(0, keep_idx);
  track_scores_ = track_scores_.index_select(0, keep_idx);
  track_last_frame_ = track_last_frame_.index_select(0, keep_idx);
  track_hits_ = track_hits_.index_select(0, keep_idx);
  track_mean_ = track_mean_.index_select(0, keep_idx);
  track_covariance_ = track_covariance_.index_select(0, keep_idx);
  if (track_ids_.size(0) == 0) {
    track_calls_since_last_empty_ = 0;
  }
}

void BYTETrackerCuda::mark_unmatched_tracking(
    const at::Tensor& matched_indices) {
  auto tracking_mask =
      track_states_.eq(static_cast<int64_t>(TrackState::Tracking));
  auto matched_mask = at::zeros_like(track_ids_, at::kBool);
  if (matched_indices.numel() > 0) {
    matched_mask.index_put_({matched_indices}, true);
  }
  auto to_lose_mask = tracking_mask & (~matched_mask);
  auto lose_idx = mask_indices(to_lose_mask);
  if (lose_idx.numel() > 0) {
    track_states_.index_put_(
        {lose_idx}, static_cast<int64_t>(TrackState::Lost));
  }
}

void BYTETrackerCuda::update_tracks(
    const at::Tensor& track_indices,
    const at::Tensor& detections_xyxy,
    const at::Tensor& labels,
    const at::Tensor& scores,
    int64_t frame_id) {
  if (track_indices.numel() == 0) {
    return;
  }
  auto measurements = bbox_xyxy_to_cxcyah(detections_xyxy);
  auto mean_subset = track_mean_.index_select(0, track_indices);
  auto cov_subset = track_covariance_.index_select(0, track_indices);
  auto kalman_pair = kalman_update(mean_subset, cov_subset, measurements);
  auto new_mean = kalman_pair.first;
  auto new_cov = kalman_pair.second;

  track_mean_.index_put_({track_indices}, new_mean);
  track_covariance_.index_put_({track_indices}, new_cov);

  track_scores_.index_put_({track_indices}, scores);
  track_labels_.index_put_({track_indices}, labels);
  track_last_frame_.index_put_(
      {track_indices},
      at::full_like(track_indices, frame_id));

  auto hits = track_hits_.index_select(0, track_indices) +
      at::ones_like(track_indices, track_hits_.scalar_type());
  track_hits_.index_put_({track_indices}, hits);

  auto states = track_states_.index_select(0, track_indices);
  auto tent_mask =
      states.eq(static_cast<int64_t>(TrackState::Tentative)) &
      hits.ge(config_.num_tentatives);
  auto tent_idx = mask_indices(tent_mask);
  if (tent_idx.numel() > 0) {
    auto to_activate = track_indices.index_select(0, tent_idx);
    track_states_.index_put_(
        {to_activate}, static_cast<int64_t>(TrackState::Tracking));
  }
  auto lost_mask =
      states.eq(static_cast<int64_t>(TrackState::Lost));
  auto lost_idx = mask_indices(lost_mask);
  if (lost_idx.numel() > 0) {
    auto to_activate = track_indices.index_select(0, lost_idx);
    track_states_.index_put_(
        {to_activate}, static_cast<int64_t>(TrackState::Tracking));
  }
}

void BYTETrackerCuda::init_new_tracks(
    const at::Tensor& ids,
    const at::Tensor& detections_xyxy,
    const at::Tensor& labels,
    const at::Tensor& scores,
    int64_t frame_id) {
  if (ids.numel() == 0) {
    return;
  }
  auto measurements = bbox_xyxy_to_cxcyah(detections_xyxy);
  auto mean_cov = kalman_initiate(measurements);
  auto new_mean = mean_cov.first;
  auto new_cov = mean_cov.second;

  if (track_ids_.numel() == 0) {
    track_ids_ = ids.clone();
    track_states_ = at::full(
        {ids.size(0)},
        track_calls_since_last_empty_ == 0
            ? static_cast<int64_t>(TrackState::Tracking)
            : static_cast<int64_t>(TrackState::Tentative),
        ids.options());
    track_labels_ = labels.clone();
    track_scores_ = scores.clone();
    track_last_frame_ = at::full_like(ids, frame_id);
    track_hits_ = at::ones_like(ids);
    track_mean_ = new_mean;
    track_covariance_ = new_cov;
    return;
  }

  track_ids_ = at::cat({track_ids_, ids}, 0);
  auto new_state_value = track_calls_since_last_empty_ == 0
      ? static_cast<int64_t>(TrackState::Tracking)
      : static_cast<int64_t>(TrackState::Tentative);
  auto new_states = at::full({ids.size(0)}, new_state_value, ids.options());
  track_states_ = at::cat({track_states_, new_states}, 0);
  track_labels_ = at::cat({track_labels_, labels}, 0);
  track_scores_ = at::cat({track_scores_, scores}, 0);
  auto frame_tensor = at::full_like(ids, frame_id);
  track_last_frame_ = at::cat({track_last_frame_, frame_tensor}, 0);
  auto ones = at::ones_like(ids);
  track_hits_ = at::cat({track_hits_, ones}, 0);
  track_mean_ = at::cat({track_mean_, new_mean}, 0);
  track_covariance_ = at::cat({track_covariance_, new_cov}, 0);
}

std::pair<at::Tensor, at::Tensor> BYTETrackerCuda::kalman_initiate(
    const at::Tensor& measurements_cxcyah) const {
  auto options = measurements_cxcyah.options();
  auto count = measurements_cxcyah.size(0);
  auto mean = at::zeros({count, 8}, options);
  mean.slice(1, 0, 4).copy_(measurements_cxcyah);
  auto h = measurements_cxcyah.select(1, 3);
  auto std_pos = at::stack(
      {2 * std_weight_position_ * h,
       2 * std_weight_position_ * h,
       at::full_like(h, 1e-2),
       2 * std_weight_position_ * h},
      1);
  auto std_vel = at::stack(
      {10 * std_weight_velocity_ * h,
       10 * std_weight_velocity_ * h,
       at::full_like(h, 1e-5),
       10 * std_weight_velocity_ * h},
      1);
  auto std = at::cat({std_pos, std_vel}, 1);
  auto cov = at::diag_embed(std.pow(2));
  return {mean, cov};
}

void BYTETrackerCuda::kalman_predict(
    at::Tensor& mean,
    at::Tensor& covariance) const {
  if (mean.size(0) == 0) {
    return;
  }
  auto h = mean.select(1, 3);
  auto std_pos = at::stack(
      {std_weight_position_ * h,
       std_weight_position_ * h,
       at::full_like(h, 1e-2),
       std_weight_position_ * h},
      1);
  auto std_vel = at::stack(
      {std_weight_velocity_ * h,
       std_weight_velocity_ * h,
       at::full_like(h, 1e-5),
       std_weight_velocity_ * h},
      1);
  auto std = at::cat({std_pos, std_vel}, 1);
  auto motion_cov = at::diag_embed(std.pow(2));

  auto motion_mat = motion_mat_;
  auto motion_T = motion_mat_T_;

  mean = at::matmul(mean, motion_T);
  auto motion_expanded = motion_mat.unsqueeze(0).expand_as(covariance);
  auto motion_T_expanded = motion_T.unsqueeze(0).expand_as(covariance);
  covariance = at::matmul(
      motion_expanded,
      at::matmul(covariance, motion_T_expanded)) +
      motion_cov;
}

std::pair<at::Tensor, at::Tensor> BYTETrackerCuda::kalman_project(
    const at::Tensor& mean,
    const at::Tensor& covariance) const {
  auto measurement_mean = mean.slice(1, 0, 4);
  auto h = mean.select(1, 3);
  auto std = at::stack(
      {std_weight_position_ * h,
       std_weight_position_ * h,
       at::full_like(h, 1e-1),
       std_weight_position_ * h},
      1);
  auto project_cov =
      covariance.narrow(1, 0, 4).narrow(2, 0, 4) +
      at::diag_embed(std.pow(2));
  return {measurement_mean, project_cov};
}

std::pair<at::Tensor, at::Tensor> BYTETrackerCuda::kalman_update(
    const at::Tensor& mean,
    const at::Tensor& covariance,
    const at::Tensor& measurement_cxcyah) const {
  auto proj = kalman_project(mean, covariance);
  auto projected_mean = proj.first;
  auto projected_cov = proj.second;

  auto B = at::matmul(covariance, update_mat_T_);
  auto BT = B.transpose(1, 2);

  // Keep the full Kalman update on the tracker device to avoid
  // host round-trips and implicit stream synchronizations.
  auto chol = at::linalg_cholesky(projected_cov);
  auto sol = at::cholesky_solve(BT, chol);
  auto kalman_gain = sol.transpose(1, 2);

  auto innovation = measurement_cxcyah - projected_mean;
  auto delta = at::matmul(
                   innovation.unsqueeze(1),
                   kalman_gain.transpose(1, 2))
                   .squeeze(1);

  auto new_mean = mean + delta;
  auto temp = at::matmul(projected_cov, kalman_gain.transpose(1, 2));
  auto new_cov =
      covariance - at::matmul(kalman_gain, temp);
  return {new_mean, new_cov};
}

BYTETrackerCuda::MatchResult BYTETrackerCuda::assign_tracks(
    const at::Tensor& track_indices,
    const at::Tensor& det_bboxes,
    const at::Tensor& det_labels,
    const at::Tensor& det_scores,
    bool weight_with_scores,
    float iou_thr) const {
  MatchResult result;
  if (track_indices.numel() == 0 || det_bboxes.size(0) == 0) {
    result.track_to_det = at::full(
        {track_indices.size(0)},
        -1,
        at::TensorOptions().dtype(at::kLong).device(device_));
    result.det_to_track = at::full(
        {det_bboxes.size(0)},
        -1,
        at::TensorOptions().dtype(at::kLong).device(device_));
    return result;
  }

  auto means = track_mean_.index_select(0, track_indices).slice(1, 0, 4);
  auto track_boxes = bbox_cxcyah_to_xyxy(means);
  auto det_boxes = det_bboxes;

  auto ious = ops::bbox_iou_cuda(track_boxes, det_boxes);
  if (weight_with_scores && det_scores.numel() > 0) {
    ious *= det_scores.unsqueeze(0);
  }

  auto track_labels = track_labels_.index_select(0, track_indices);
  auto label_penalty = (track_labels.unsqueeze(1) == det_labels.unsqueeze(0))
      .to(ious.dtype());
  auto cate_cost = (1 - label_penalty) * 1e6;
  auto cost = (1 - ious) + cate_cost;

  auto assignments = ops::hungarian_assign_cuda(
      cost, track_indices.size(0), det_boxes.size(0), 1.0f - iou_thr);
  auto track_to_det = assignments.first.to(device_);
  auto det_to_track = assignments.second.to(device_);

  result.track_to_det = track_to_det;
  result.det_to_track = det_to_track;
  return result;
}

at::Tensor BYTETrackerCuda::mask_indices(const at::Tensor& mask) const {
  if (!mask.defined() || mask.numel() == 0) {
    return at::empty({0}, mask.options().dtype(at::kLong));
  }
  auto mask_bool = mask.to(at::kBool);
  auto options = at::TensorOptions().dtype(at::kLong).device(mask.device());
  auto indices = at::arange(mask_bool.size(0), options);
  return indices.masked_select(mask_bool);
}

} // namespace tracker
} // namespace hm
