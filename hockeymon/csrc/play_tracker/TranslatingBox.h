#pragma once

#include "hockeymon/csrc/play_tracker/BoxUtils.h"
#include "hockeymon/csrc/play_tracker/LivingBox.h"

#include <cassert>
#include <optional>

namespace hm {
namespace play_tracker {

struct TranslationState {
  FloatValue current_speed_x{0.0};
  FloatValue current_speed_y{0.0};
  bool translation_is_frozen{false};
  FloatValue last_arena_edge_center_position_scale{0.0};
  // Nonstop stuff
  std::optional<IntValue> nonstop_delay{0};
  IntValue nonstop_delay_counter{0};
  // Per-axis stop-on-direction-change braking (frames + per-frame decel)
  std::optional<IntValue> stop_delay_x{0};
  IntValue stop_delay_x_counter{0};
  FloatValue stop_decel_x{0.0};
  FloatValue stop_trigger_dir_x{0.0};
  IntValue cancel_opp_x_count{0};
  IntValue cooldown_x_counter{0};
  std::optional<IntValue> stop_delay_y{0};
  IntValue stop_delay_y_counter{0};
  FloatValue stop_decel_y{0.0};
  FloatValue stop_trigger_dir_y{0.0};
  IntValue cancel_opp_y_count{0};
  IntValue cooldown_y_counter{0};
  // Visual cue booleans to indicate a cancel occurred this frame
  bool canceled_stop_x{false};
  bool canceled_stop_y{false};
  // Low-pass filtered target center for smooth panning
  std::optional<Point> filtered_target_center{std::nullopt};
};

class TranslatingBox : virtual public IBasicLivingBox {
 public:
  TranslatingBox(const TranslatingBoxConfig& config);

  void set_destination(const BBox& dest_box) override;

  const TranslationState& get_state() const;

  const TranslatingBoxConfig& get_config() const;

  void adjust_speed(
      std::optional<FloatValue> accel_x = std::nullopt,
      std::optional<FloatValue> accel_y = std::nullopt,
      std::optional<FloatValue> scale_constraints = std::nullopt,
      std::optional<IntValue> nonstop_delay = std::nullopt);

  // Begin a per-axis stop delay externally (e.g., overshoot/catch-up cases)
  void begin_stop_delay(
      std::optional<IntValue> delay_x = std::nullopt,
      std::optional<IntValue> delay_y = std::nullopt);

  // Update braking-related configuration at runtime
  void set_braking_params(
      IntValue stop_on_dir_change_delay,
      bool cancel_on_opposite,
      IntValue cancel_hysteresis_frames,
      IntValue stop_delay_cooldown_frames,
      IntValue post_nonstop_stop_delay_count);

  // Update translation constraints (max speeds/accels) at runtime
  void set_translation_constraints(
      FloatValue max_speed_x,
      FloatValue max_speed_y,
      FloatValue max_accel_x,
      FloatValue max_accel_y);

  /**
   * Scale the current speed by the given ratio
   */
  void scale_speed(
      std::optional<FloatValue> ratio_x = std::nullopt,
      std::optional<FloatValue> ratio_y = std::nullopt,
      bool clamp_to_max = false);

  std::tuple<FloatValue, FloatValue> get_sticky_translation_sizes() const;

  // From the current position and size, get the ratio of closeness to the left
  // or right edge, where 1.0 means on the edge and 0.0 means in the middle. The
  // size of the box may be part of the consideration, for example a very large
  // box will cover more horizontal ground and thus be less "completely on the
  // edge", so some of the result will be a heuristic wrt size and position.
  FloatValue get_arena_edge_center_position_scale() const;

  static FloatValue get_arena_edge_point_position_scale(
      const Point& pt,
      const BBox& arena_box,
      const FloatValue field_edge_veritcal_angle);

 protected:
  PointDiff get_proposed_next_position_change() const;

  void stop_translation_if_out_of_arena();

  // After new position is set, adjust the nonstop-delay
  void update_nonstop_delay();

  // After new position is set, adjust per-axis stop delays
  void update_stop_delays();

  void on_new_position();

 private:
  void test_arena_edge_position_scale();

  bool is_nonstop() const;

  void clamp_speed(FloatValue scale);

  FloatValue get_gaussian_y_about_width_center(FloatValue x) const;

 private:
  TranslatingBoxConfig config_;
  TranslationState state_;

  // Calculated clamp values based upon the given arena box (of any)
  std::optional<std::pair<FloatValue, FloatValue>> gasussian_clamp_lr{
      std::nullopt};
};

} // namespace play_tracker
} // namespace hm
