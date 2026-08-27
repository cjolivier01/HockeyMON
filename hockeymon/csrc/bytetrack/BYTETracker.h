#pragma once

// #include <ATen/ATen.h>

#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "BaseTracker.h"

namespace hm {
namespace tracker {

struct ByteTrackConfig {
  float init_track_thr = 0.7;
  float obj_score_thrs_low = 0.1;
  float obj_score_thrs_high = 0.6;
  // IOU distance threshold for matching between two frames.
  //      - high (float): Threshold of the first matching. Defaults to 0.1.
  //      - low (float): Threshold of the second matching. Defaults to 0.5.
  //      - tentative (float): Threshold of the matching for tentative
  //        tracklets. Defaults to 0.3.
  float match_iou_thrs_high = 0.1;
  float match_iou_thrs_low = 0.5;
  float match_iou_thrs_tentative = 0.3;
  std::size_t track_buffer_size = 30;
  std::size_t num_frames_to_keep_lost_tracks = 30;
  // Whether using detection scores to weight IOU which is used for matching.
  // Defaults to True.
  bool weight_iou_with_det_scores = true;
  std::size_t num_tentatives = 3;
  std::optional<std::unordered_map<std::string, float>> momentums;
};

struct IByteTracker : public ITracker {
  virtual ~IByteTracker() = default;
  virtual void activate_track(int64_t id) = 0;
};

class BYTETracker : public BaseTracker, public IByteTracker {
  using Super = BaseTracker;

 public:
  static inline constexpr const char* kFrameId = "frame_id";
  static inline constexpr const char* kBBoxes = "bboxes";
  static inline constexpr const char* kLabels = "labels";
  static inline constexpr const char* kScores = "scores";

  BYTETracker(ByteTrackConfig config);

  ~BYTETracker();

  std::unordered_map<std::string, at::Tensor> track(
      std::unordered_map<std::string, at::Tensor>&& data) override;

  std::size_t num_tracks() const {
    return num_tracks_;
  }

 protected:
  void reset() override;
  virtual at::Tensor adjust_detection_boxes(at::Tensor det_bboxes);
  void pop_invalid_tracks(int64_t frame_id) override;
  void pop_track(int64_t track_id) override;
  void init_track(
      int64_t id,
      const std::unordered_map<std::string, at::Tensor>& memos) override;
  void update_track(
      int64_t id,
      const std::unordered_map<std::string, at::Tensor>& memos) override;

  void activate_track(int64_t id) override;

  constexpr bool is_debug() const {
    return debug_;
  };
  constexpr byte_kalman::KalmanFilter& kalman_filter() {
    return kalman_filter_;
  }
  constexpr const byte_kalman::KalmanFilter& kalman_filter() const {
    return kalman_filter_;
  }

  constexpr const ByteTrackConfig& config() const {
    return config_;
  }

 private:
  void track_update(
      const at::Tensor& ids,
      const at::Tensor& bboxes,
      const at::Tensor& labels,
      const at::Tensor& scores,
      const std::vector<int64_t>& frame_ids);

  bool validate_memos_enabled() const;
  void validate_label_tensor(
      const char* stage,
      int64_t frame_id,
      const at::Tensor& labels) const;
  void validate_track_memos(const char* stage, int64_t frame_id) const;

  std::tuple<at::Tensor, at::Tensor> assign_ids(
      const std::vector<long>& ids,
      const at::Tensor& det_bboxes,
      const at::Tensor& det_labels,
      const at::Tensor& det_scores,
      bool weight_iou_with_det_scores,
      float match_iou_thr);

  ByteTrackConfig config_;

  int64_t num_tracks_{0};
  std::size_t track_calls_since_last_empty_{0};
  bool debug_{true};
  std::unordered_set<int64_t> lost_tracks_;
  std::size_t reacquired_count_{0};
  std::size_t lost_tentative_count_{0};

  std::size_t track_pass_{0};
  byte_kalman::KalmanFilter kalman_filter_;
};

} // namespace tracker
} // namespace hm
