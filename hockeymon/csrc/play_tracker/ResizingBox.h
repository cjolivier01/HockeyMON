#pragma once

#include "hockeymon/csrc/play_tracker/LivingBox.h"

#include <cassert>

namespace hm {
namespace play_tracker {

struct ResizingState {
  bool size_is_frozen{false};
  FloatValue current_speed_w{0.0};
  FloatValue current_speed_h{0.0};
  // Per-axis stop-on-direction-change braking for resizing
  std::optional<IntValue> stop_delay_w{0};
  IntValue stop_delay_w_counter{0};
  FloatValue stop_decel_w{0.0};
  FloatValue stop_trigger_dir_w{0.0};
  IntValue cancel_opp_w_count{0};
  bool deadband_stop_w{false};
  IntValue cooldown_w_counter{0};
  std::optional<IntValue> stop_delay_h{0};
  IntValue stop_delay_h_counter{0};
  FloatValue stop_decel_h{0.0};
  FloatValue stop_trigger_dir_h{0.0};
  IntValue cancel_opp_h_count{0};
  bool deadband_stop_h{false};
  IntValue cooldown_h_counter{0};
  bool canceled_stop_w{false};
  bool canceled_stop_h{false};
};

class ResizingBox : virtual public IBasicLivingBox {
 public:
  ResizingBox(ResizingBox&&) = delete;
  ResizingBox(const ResizingConfig& config);

  void set_destination(const BBox& dest_box) override;

  const ResizingState& get_state() const;

  const ResizingConfig& get_config() const;

  GrowShrink get_grow_shrink_wh(const BBox& bbox) const;
  void set_shrink_thresholds(
      FloatValue width_ratio, FloatValue height_ratio);

 protected:
  SizeDiff get_proposed_next_size_change() const;

  WHDims get_min_allowed_width_height() const;

  void clamp_size_scaled();

  // After new position is set, adjust per-axis resize stop delays
  void update_stop_delays();

 private:
  void set_destination_size(
      FloatValue dest_width,
      FloatValue dest_height,
      bool prioritize_width_thresh = true);

  void adjust_size(
      FloatValue accel_w,
      FloatValue accel_h,
      bool use_constraints = true);

  void clamp_resizing();

  void begin_width_deadband_stop();

  void begin_height_deadband_stop();

  void clear_width_stop(bool start_cooldown);

  void clear_height_stop(bool start_cooldown);

  ResizingConfig config_;
  ResizingState state_;
};

} // namespace play_tracker
} // namespace hm
