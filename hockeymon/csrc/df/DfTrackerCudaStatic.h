#pragma once

#include "hockeymon/csrc/bytetrack/BYTETracker.h"

#include <torch/torch.h>

#include <c10/core/Device.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>

namespace hm {
namespace tracker {

class DfTrackerCudaStatic {
 public:
  explicit DfTrackerCudaStatic(
      ByteTrackConfig config,
      int64_t max_detections,
      int64_t max_tracks,
      int64_t reid_feature_dim = 256,
      float iou_weight = 0.5f,
      float reid_weight = 0.5f,
      float box_momentum = 0.6f,
      float reid_momentum = 0.2f,
      float min_similarity = -1.0f,
      float lost_track_cost = 0.05f,
      c10::Device device = c10::Device(c10::kCUDA, 0));

  ~DfTrackerCudaStatic() = default;

  void reset();

  std::size_t num_tracks() const {
    return active_tracks_;
  }

  std::unordered_map<std::string, at::Tensor> track(
      std::unordered_map<std::string, at::Tensor>&& data);

  int64_t max_detections() const {
    return max_detections_;
  }

  int64_t max_tracks() const {
    return max_tracks_;
  }

  int64_t reid_feature_dim() const {
    return reid_feature_dim_;
  }

 private:
  enum class TrackState : int64_t { Inactive = 0, Tracking = 1, Lost = 2 };

  at::Tensor mask_indices(const at::Tensor& mask) const;
  at::Tensor ensure_vector(
      const at::Tensor& tensor,
      at::ScalarType dtype) const;
  at::Tensor ensure_bboxes(const at::Tensor& tensor) const;

  ByteTrackConfig config_;
  c10::Device device_;
  int64_t max_detections_;
  int64_t max_tracks_;
  int64_t reid_feature_dim_;
  float iou_weight_;
  float reid_weight_;
  float box_momentum_;
  float reid_momentum_;
  float min_similarity_;
  float lost_track_cost_;
  std::size_t active_tracks_{0};
  int64_t next_track_id_{0};

  at::Tensor track_ids_;
  at::Tensor track_state_;
  at::Tensor track_labels_;
  at::Tensor track_scores_;
  at::Tensor track_last_frame_;
  at::Tensor track_hits_;
  at::Tensor track_bboxes_;
  at::Tensor track_reid_;
};

} // namespace tracker
} // namespace hm
