#pragma once

#include "BYTETracker.h"

#include <ATen/ATen.h>

#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace hm {
namespace tracker {

enum class HmTrackerPredictionMode {
  BoundingBox,
  BoxCenter,
  BoxBottom,
  BoxBottomCenter,
};

struct HmTrackerConfig {
  HmTrackerPredictionMode prediction_mode{HmTrackerPredictionMode::BoundingBox};
  float tentative_high_confidence = 0.5;
  std::size_t num_tentative_high_confidence = 2;
  float tentative_low_confidence = 0.2;
  std::size_t num_tentative_low_confidence = 6;
  bool remove_tentative{true};
  bool return_user_ids{false};
  bool return_track_age{false};
};

struct HmByteTrackConfig : public ByteTrackConfig, public HmTrackerConfig {};

class HmTracker : public BYTETracker {
  using Super = BYTETracker;

 public:
  HmTracker(HmTrackerConfig hm_config, ByteTrackConfig bt_config);
  HmTracker(HmByteTrackConfig config);

  ~HmTracker();

  std::unordered_map<std::string, at::Tensor> track(
      std::unordered_map<std::string, at::Tensor>&& data) override;

  std::size_t total_activated_tracks_count() const {
    return total_activated_tracks_count_;
  }

 protected:
  void reset() override;
  at::Tensor adjust_detection_boxes(at::Tensor det_bboxes) override;
  void activate_track(int64_t id) override;
  void pop_track(int64_t track_id) override;

 private:
  HmTrackerConfig hm_config_;
  std::size_t total_activated_tracks_count_{0};
  // tracking id -> activated track id (doesn't need to be sorted)
  std::map<int64_t, int64_t> activated_id_mapping_;
};

} // namespace tracker
} // namespace hm
