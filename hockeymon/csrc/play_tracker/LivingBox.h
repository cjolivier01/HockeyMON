#pragma once

#include "hockeymon/csrc/play_tracker/BoxUtils.h"

#include <cassert>
#include <cmath>
#include <memory>
#include <optional>
#include <string>
#include <variant>

namespace hm {
namespace play_tracker {

struct AllLivingBoxConfig;
struct LivingState;
struct ResizingState;
struct TranslationState;

struct GrowShrink {
  FloatValue grow_width, grow_height, shrink_width, shrink_height;
};

inline constexpr bool is_zero(const FloatValue& f) {
  // Good enough as zero for opur purposes
  return std::abs(f) < 1e-6;
}

/* clang-format off */
/**
 *  _____ ____              _      _      _         _             ____
 * |_   _|  _ \            (_)    | |    (_)       (_)           |  _ \
 *   | | | |_) | __ _  ___  _  ___| |     _ __   __ _ _ __   __ _| |_) | ___ __  __
 *   | | |  _ < / _` |/ __|| |/ __| |    | |\ \ / /| | '_ \ / _` |  _ < / _ \\ \/ /
 *  _| |_| |_) | (_| |\__ \| | (__| |____| | \ V / | | | | | (_| | |_) | (_) |>  <
 * |_____|____/ \__,_||___/|_|\___|______|_|  \_/  |_|_| |_|\__, |____/ \___//_/\_\
 *                                                           __/ |
 *                                                          |___/
 */
/* clang-format on */
struct IBasicLivingBox {
  virtual ~IBasicLivingBox() = default;

  virtual const std::string& name() const = 0;

  virtual void set_bbox(const BBox& bbox) = 0;

  virtual BBox bounding_box() const = 0;

  virtual void set_destination(const BBox& dest_box) = 0;
};

/**
 *  _____ _      _         _             ____
 * |_   _| |    (_)       (_)           |  _ \
 *   | | | |     _ __   __ _ _ __   __ _| |_) | ___ __  __
 *   | | | |    | |\ \ / /| | '_ \ / _` |  _ < / _ \\ \/ /
 *  _| |_| |____| | \ V / | | | | | (_| | |_) | (_) |>  <
 * |_____|______|_|  \_/  |_|_| |_|\__, |____/ \___//_/\_\
 *                                  __/ |
 *                                 |___/
 */
struct ILivingBox : virtual public IBasicLivingBox {
  virtual void set_dest(std::shared_ptr<IBasicLivingBox>) = 0;
  virtual void set_destination_ex(
      const std::variant<BBox, std::shared_ptr<IBasicLivingBox>>& dest) = 0;
  virtual void set_dest_ex(
      const std::variant<BBox, std::shared_ptr<IBasicLivingBox>>& dest) = 0;
  virtual BBox forward(
      const std::variant<BBox, std::shared_ptr<IBasicLivingBox>>& dest) = 0;
  virtual std::optional<BBox> get_arena_box() const = 0;

  virtual void adjust_speed(
      std::optional<FloatValue> accel_x,
      std::optional<FloatValue> accel_y,
      std::optional<FloatValue> scale_constraints,
      std::optional<IntValue> nonstop_delay) = 0;


  virtual const LivingState& get_live_box_state() const = 0;
  virtual const ResizingState& get_resizing_state() const = 0;
  virtual const TranslationState& get_translation_state() const = 0;

  virtual GrowShrink get_grow_shrink_wh(const BBox& bbox) const = 0;
  virtual std::tuple<FloatValue, FloatValue> get_sticky_translation_sizes() const = 0;

  // virtual const AllLivingBoxConfig& get_config() const = 0;

  /**
   * Scale the current speed by the given ratio
   */
  virtual void scale_speed(
      std::optional<FloatValue> ratio_x,
      std::optional<FloatValue> ratio_y,
      bool clamp_to_max = false) = 0;

  // Begin a per-axis stop delay externally (e.g., overshoot cases)
  virtual void begin_stop_delay(
      std::optional<IntValue> delay_x, std::optional<IntValue> delay_y) = 0;

  // Runtime tuning keeps the current box, velocity, and braking state intact.
  virtual void set_braking(
      IntValue stop_on_dir_change_delay,
      bool cancel_on_opposite,
      IntValue cancel_hysteresis_frames,
      IntValue stop_delay_cooldown_frames,
      IntValue post_nonstop_stop_delay_count,
      IntValue time_to_dest_speed_limit_frames) = 0;

  virtual void set_translation_constraints(
      FloatValue max_speed_x,
      FloatValue max_speed_y,
      FloatValue max_accel_x,
      FloatValue max_accel_y) = 0;

  virtual void set_camera_geometry(
      FloatValue arena_angle_from_vertical,
      FloatValue dynamic_acceleration_scaling) = 0;

  virtual void set_resizing_shrink_thresholds(
      FloatValue width_ratio, FloatValue height_ratio) = 0;
};

/* clang-format off */
/**
 *  _____             _       _              _____              __  _
 * |  __ \           (_)     (_)            / ____|            / _|(_)
 * | |__) | ___  ___  _  ____ _ _ __   __ _| |      ___  _ __ | |_  _  __ _
 * |  _  / / _ \/ __|| ||_  /| | '_ \ / _` | |     / _ \| '_ \|  _|| |/ _` |
 * | | \ \|  __/\__ \| | / / | | | | | (_| | |____| (_) | | | | |  | | (_| |
 * |_|  \_\\___||___/|_|/___||_|_| |_|\__, |\_____|\___/|_| |_|_|  |_|\__, |
 *                                     __/ |                           __/ |
 *                                    |___/                           |___/
 */
/* clang-format on */
struct ResizingConfig {
  bool resizing_enabled{true};
  FloatValue max_speed_w{0.0};
  FloatValue max_speed_h{0.0};
  FloatValue max_accel_w{0.0};
  FloatValue max_accel_h{0.0};
  FloatValue min_width{0.0};
  FloatValue min_height{0.0};
  FloatValue max_width{0.0};
  FloatValue max_height{0.0};
  bool stop_resizing_on_dir_change{true};
  // When >0, instead of damping the input on a direction change,
  // initiate a per-axis stop over this many frames, ignoring new inputs
  // on that axis while decelerating to zero.
  IntValue resizing_stop_on_dir_change_delay{0};
  // If true, while a resize stop-delay is active on an axis,
  // cancel that stop-delay immediately if the input direction flips
  // opposite to the direction that originally triggered the stop.
  bool resizing_cancel_stop_on_opposite_dir{false};
  // Hysteresis for cancel-on-opposite during resize braking (consecutive frames)
  IntValue resizing_stop_cancel_hysteresis_frames{0};
  // Cooldown after a resize stop-delay finishes or is canceled before a new one can start
  IntValue resizing_stop_delay_cooldown_frames{0};
  // When increasing size toward destination, cap the speed so that
  // time-to-go along that axis is at least this many frames. 0 disables.
  IntValue resizing_time_to_dest_speed_limit_frames{10};
  // When time-to-dest limiting is active, snap to zero if speed drops below this
  FloatValue resizing_time_to_dest_stop_speed_threshold{0.0};
  bool sticky_sizing{false};
  //
  // Sticky sizing thresholds
  //
  // Threshold to grow width (ratio of bbox)
  FloatValue size_ratio_thresh_grow_dw{0.05};
  // Threshold to grow height (ratio of bbox)
  FloatValue size_ratio_thresh_grow_dh{0.1};
  // Threshold to shrink width (ratio of bbox)
  FloatValue size_ratio_thresh_shrink_dw{0.08};
  // Threshold to shrink height (ratio of bbox)
  FloatValue size_ratio_thresh_shrink_dh{0.1};
};

/* clang-format off */
/**
 *  _______                     _       _   _              _____              __  _
 * |__   __|                   | |     | | (_)            / ____|            / _|(_)
 *    | |_ __  __ _ _ __   ___ | | __ _| |_ _  ___  _ __ | |      ___  _ __ | |_  _  __ _
 *    | | '__|/ _` | '_ \ / __|| |/ _` | __| |/ _ \| '_ \| |     / _ \| '_ \|  _|| |/ _` |
 *    | | |  | (_| | | | |\__ \| | (_| | |_| | (_) | | | | |____| (_) | | | | |  | | (_| |
 *    |_|_|   \__,_|_| |_||___/|_|\__,_|\__|_|\___/|_| |_|\_____|\___/|_| |_|_|  |_|\__, |
 *                                                                                   __/ |
 *                                                                                  |___/
 */
/* clang-format on */
struct TranslatingBoxConfig {
  bool translation_enabled{true};
  FloatValue max_speed_x{0.0};
  FloatValue max_speed_y{0.0};
  FloatValue max_accel_x{0.0};
  FloatValue max_accel_y{0.0};
  bool stop_translation_on_dir_change{true};
  // When >0, instead of damping the input on a direction change,
  // initiate a per-axis stop over this many frames, ignoring new inputs
  // on that axis while decelerating to zero.
  IntValue stop_translation_on_dir_change_delay{0};
  // If true, while a stop-delay is active on an axis,
  // cancel that stop-delay immediately if the input direction flips
  // opposite to the direction that originally triggered the stop.
  bool cancel_stop_on_opposite_dir{false};
  FloatValue dynamic_acceleration_scaling{0.0};
  FloatValue arena_angle_from_vertical{0.0};
  std::optional<BBox> arena_box{std::nullopt};
  // Sticky Sizing
  bool sticky_translation{false};
  FloatValue sticky_size_ratio_to_frame_width{10.0};
  FloatValue sticky_translation_gaussian_mult{5.0};
  FloatValue unsticky_translation_size_ratio{0.75};
  // Smooth the target center before computing velocity (0 disables)
  // FloatValue pan_smoothing_alpha{0.18};
  // Optional braking after a nonstop (breakaway catch-up) window ends
  IntValue post_nonstop_stop_delay_count{0};
  // Hysteresis for cancel-on-opposite during braking (consecutive frames)
  IntValue cancel_stop_hysteresis_frames{0};
  // Cooldown after a stop-delay finishes or is canceled before a new one can start
  IntValue stop_delay_cooldown_frames{0};
  // When increasing speed toward destination, cap the speed so that
  // time-to-go along that axis is at least this many frames. 0 disables.
  IntValue time_to_dest_speed_limit_frames{10};
  // When time-to-dest limiting is active, snap to zero if speed drops below this
  FloatValue time_to_dest_stop_speed_threshold{0.0};
};

struct LivingBoxConfig {
  FloatValue scale_dest_width{1.0};
  FloatValue scale_dest_height{1.0};
  std::optional<FloatValue> fixed_aspect_ratio{std::nullopt};
  bool clamp_scaled_input_box{true};
};

struct AllLivingBoxConfig : public ResizingConfig,
                            public TranslatingBoxConfig,
                            public LivingBoxConfig {
  std::string name;
};

/**
 *  ____                        _  _             ____
 * |  _ \                      | |(_)           |  _ \
 * | |_) | ___  _   _ _ __   __| | _ _ __   __ _| |_) | ___ __  __
 * |  _ < / _ \| | | | '_ \ / _` || | '_ \ / _` |  _ < / _ \\ \/ /
 * | |_) | (_) | |_| | | | | (_| || | | | | (_| | |_) | (_) |>  <
 * |____/ \___/ \__,_|_| |_|\__,_||_|_| |_|\__, |____/ \___//_/\_\
 *                                          __/ |
 *                                         |___/
 */
class BoundingBox : virtual public IBasicLivingBox {
 public:
  BoundingBox(BBox bbox) : bbox_(bbox.clone()) {}

  void set_bbox(const BBox& bbox) override {
    bbox_ = bbox.clone();
  }

  BBox bounding_box() const override {
    return bbox_.clone();
  }

 protected:
  BBox bbox_;
};

} // namespace play_tracker
} // namespace hm
