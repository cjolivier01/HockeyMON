#pragma once

#include "BYTETracker.h"

#include "hockeymon/csrc/pytorch/bytetrack_cuda_ops.h"

#include <torch/torch.h>
#include <utility>

namespace hm {
namespace tracker {

class BYTETrackerCuda {
 public:
  explicit BYTETrackerCuda(
      ByteTrackConfig config,
      c10::Device device = c10::Device(c10::kCUDA, 0));

  virtual ~BYTETrackerCuda() = default;

  void reset();

  std::size_t num_tracks() const {
    return static_cast<std::size_t>(track_ids_.size(0));
  }

  virtual std::unordered_map<std::string, at::Tensor> track(
      std::unordered_map<std::string, at::Tensor>&& data);

 protected:
  std::unordered_map<std::string, at::Tensor> run_tracker(
      std::unordered_map<std::string, at::Tensor>&& data);

  virtual at::Tensor mask_indices(const at::Tensor& mask) const;

  const c10::Device& device() const {
    return device_;
  }

  enum class TrackState : int64_t { Tentative = 0, Tracking = 1, Lost = 2 };

  at::Tensor ensure_bboxes(const at::Tensor& tensor) const;
  at::Tensor ensure_vector(
      const at::Tensor& tensor,
      at::ScalarType dtype) const;

  void predict_tracks(int64_t frame_id);
  void remove_stale_tracks(int64_t frame_id);
  void mark_unmatched_tracking(const at::Tensor& matched_indices);
  void update_tracks(
      const at::Tensor& track_indices,
      const at::Tensor& detections_xyxy,
      const at::Tensor& labels,
      const at::Tensor& scores,
      int64_t frame_id);
  void init_new_tracks(
      const at::Tensor& ids,
      const at::Tensor& detections_xyxy,
      const at::Tensor& labels,
      const at::Tensor& scores,
      int64_t frame_id);

  std::pair<at::Tensor, at::Tensor> kalman_initiate(
      const at::Tensor& measurements_cxcyah) const;
  void kalman_predict(at::Tensor& mean, at::Tensor& covariance) const;
  std::pair<at::Tensor, at::Tensor> kalman_project(
      const at::Tensor& mean,
      const at::Tensor& covariance) const;
  std::pair<at::Tensor, at::Tensor> kalman_update(
      const at::Tensor& mean,
      const at::Tensor& covariance,
      const at::Tensor& measurement_cxcyah) const;

  struct MatchResult {
    at::Tensor track_to_det;
    at::Tensor det_to_track;
  };

  MatchResult assign_tracks(
      const at::Tensor& track_indices,
      const at::Tensor& det_bboxes,
      const at::Tensor& det_labels,
      const at::Tensor& det_scores,
      bool weight_with_scores,
      float iou_thr) const;

  void append_tensors(
      at::Tensor& base,
      const at::Tensor& values);

  ByteTrackConfig config_;
  c10::Device device_;
  int64_t next_track_id_{0};
  int64_t track_calls_since_last_empty_{0};

  at::Tensor track_ids_;
  at::Tensor track_states_;
  at::Tensor track_labels_;
  at::Tensor track_scores_;
  at::Tensor track_last_frame_;
  at::Tensor track_hits_;
  at::Tensor track_mean_;
  at::Tensor track_covariance_;

  at::Tensor motion_mat_;
  at::Tensor motion_mat_T_;
  at::Tensor update_mat_;
  at::Tensor update_mat_T_;

  const float std_weight_position_{1.f / 20.f};
  const float std_weight_velocity_{1.f / 160.f};
};

} // namespace tracker
} // namespace hm
