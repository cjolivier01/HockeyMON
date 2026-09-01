#include "hockeymon/csrc/play_tracker/TranslatingBox.h"
#include "hockeymon/csrc/play_tracker/LogCapture.h"

#include <cassert>
#include <csignal>
#include <iostream>

#include <unistd.h>

namespace hm {
namespace play_tracker {

namespace {
constexpr FloatValue kSpeedDiffDirectionAssumeStoppedMaxRatio = 6.0;
constexpr FloatValue kMaxSpeedDiffDirectionCutRateRatio = 2.0;
constexpr FloatValue kDestinationDistanceToArenaWidthRatioToIgnoreScalingSpeed =
    1.0f / 4;

constexpr FloatValue kEpsilon = 1e-4f;

bool is_close(
    const FloatValue& f1,
    const FloatValue& f2,
    const FloatValue epsilon = kEpsilon) {
  return std::abs(f2 - f1) < epsilon;
}

} // namespace

TranslatingBox::TranslatingBox(const TranslatingBoxConfig& config)
    : config_(config) {
  if (config_.arena_box.has_value()) {
    gasussian_clamp_lr =
        std::make_pair(config_.arena_box->left, config_.arena_box->right);
  }
}

void TranslatingBox::set_destination(const BBox& dest_box) {
  if (!config_.translation_enabled) {
    return;
  }
  // raise(SIGTRAP);
  BBox bbox = bounding_box();
  Point center_current = bbox.center();
  Point center_dest = dest_box.center();
  // Apply simple low-pass filtering to the target to avoid jerky pans.
  // if (config_.pan_smoothing_alpha > 0.0f) {
  //   if (!state_.filtered_target_center.has_value()) {
  //     state_.filtered_target_center = center_dest;
  //   } else {
  //     Point f = *state_.filtered_target_center;
  //     const FloatValue a = clamp(config_.pan_smoothing_alpha, 0.0f, 1.0f);
  //     f.x = f.x + a * (center_dest.x - f.x);
  //     f.y = f.y + a * (center_dest.y - f.y);
  //     state_.filtered_target_center = f;
  //   }
  //   center_dest = *state_.filtered_target_center;
  // }
  PointDiff total_diff = center_dest - center_current;
  // Reset single-frame cancel flash indicators
  state_.canceled_stop_x = false;
  state_.canceled_stop_y = false;

  std::optional<FloatValue> x_gaussian;
  if (!is_zero(config_.dynamic_acceleration_scaling)) {
    assert(config_.sticky_translation);
    static size_t test_pass_counter = 0;
    if (!test_pass_counter++) {
      test_arena_edge_position_scale();
    }
    // Only do this if we aren't super off so that massive movements are still
    // possible in desperate situations
    // TODO: We can cache this computation
    if (std::abs(total_diff.dx) < config_.arena_box->width() *
            kDestinationDistanceToArenaWidthRatioToIgnoreScalingSpeed) {
      x_gaussian = 1.0 -
          std::abs(config_.dynamic_acceleration_scaling *
                   get_arena_edge_center_position_scale());
    } else {
      static size_t wayoff_count = 0;
      ++wayoff_count;
      hm_log_warning(
          std::to_string(wayoff_count) +
          ": We are way off, ignoring any position scale");
      constexpr FloatValue kEmergencyPanFixScaleConstraintRatio = 4.0;
      x_gaussian = kEmergencyPanFixScaleConstraintRatio;
    }
  } else {
    x_gaussian = 1.0;
  }
  state_.last_arena_edge_center_position_scale = x_gaussian ? *x_gaussian : 1.0;
  // std::cout << name() << ": total_diff: " << total_diff << std::endl;

  // If both the dest box and our current box are on an edge, we zero-out
  // the magnitude in the direction of that edge so that the size
  // differences of the box don't keep us in the un-stuck mode,
  // even though we can't move anymore in that direction
  if (config_.arena_box.has_value()) {
    // slightly deflated box
    BBox inflated_box = config_.arena_box->inflate(1, 1, -1, -1);
    std::tuple<bool, bool> x_y_on_edge = check_for_box_overshoot(
        bbox,
        inflated_box,
        /*moving_directions=*/total_diff,
        /*epsilon=*/0.1);
    state_.current_speed_x *= !std::get<0>(x_y_on_edge);
    state_.current_speed_y *= !std::get<1>(x_y_on_edge);
    total_diff.dx *= !std::get<0>(x_y_on_edge);
    total_diff.dy *= !std::get<1>(x_y_on_edge);
  } else {
    assert(false);
  }

  if (config_.sticky_translation && !is_nonstop()) {
    //
    // BEGIN Sticky Translation
    //
    const FloatValue diff_magnitude = norm(total_diff);

    // See if we are breaking the sticky or unsticky threshold
    const auto sticky_unsticky = get_sticky_translation_sizes();
    const FloatValue sticky_size = std::get<0>(sticky_unsticky);
    const FloatValue unsticky_size = std::get<1>(sticky_unsticky);
    if (!state_.translation_is_frozen) {
      if (diff_magnitude <= sticky_size) {
        state_.translation_is_frozen = true;
        state_.current_speed_x = 0.0;
        state_.current_speed_y = 0.0;
      }
    } else {
      if (diff_magnitude >= unsticky_size) {
        state_.translation_is_frozen = false;
        // Unstick at zero velocity
        state_.current_speed_x = 0.0;
        state_.current_speed_y = 0.0;
      }
    }
    //
    // END Sticky Translation
    //
  }

  // Build per-axis accel, allowing stop-delay braking to override input
  std::optional<FloatValue> accel_x = total_diff.dx;
  std::optional<FloatValue> accel_y = total_diff.dy;

  if (!is_nonstop()) {
    // Only consider triggering new stop-delays if not already braking on axis
    if ((!state_.stop_delay_x || *state_.stop_delay_x == 0) && state_.cooldown_x_counter == 0 &&
        different_directions(total_diff.dx, state_.current_speed_x)) {
      const bool moving_enough_x =
          std::abs(state_.current_speed_x) >=
          config_.max_speed_x / kSpeedDiffDirectionAssumeStoppedMaxRatio;
      if (config_.stop_translation_on_dir_change_delay > 0 && moving_enough_x) {
        state_.stop_delay_x = config_.stop_translation_on_dir_change_delay;
        state_.stop_delay_x_counter = 0;
        state_.stop_decel_x = -state_.current_speed_x /
            static_cast<FloatValue>(config_.stop_translation_on_dir_change_delay);
        state_.stop_trigger_dir_x = sign(total_diff.dx);
      } else {
        if (std::abs(state_.current_speed_x) <
            config_.max_speed_x / kSpeedDiffDirectionAssumeStoppedMaxRatio) {
          state_.current_speed_x = 0.0;
        } else {
          state_.current_speed_x /= kMaxSpeedDiffDirectionCutRateRatio;
        }
        if (config_.stop_translation_on_dir_change) {
          accel_x = total_diff.dx * 0.25f; // soften abrupt direction reversals
        }
      }
    }

    if ((!state_.stop_delay_y || *state_.stop_delay_y == 0) && state_.cooldown_y_counter == 0 &&
        different_directions(total_diff.dy, state_.current_speed_y)) {
      const bool moving_enough_y =
          std::abs(state_.current_speed_y) >=
          config_.max_speed_y / kSpeedDiffDirectionAssumeStoppedMaxRatio;
      if (config_.stop_translation_on_dir_change_delay > 0 && moving_enough_y) {
        state_.stop_delay_y = config_.stop_translation_on_dir_change_delay;
        state_.stop_delay_y_counter = 0;
        state_.stop_decel_y = -state_.current_speed_y /
            static_cast<FloatValue>(config_.stop_translation_on_dir_change_delay);
        state_.stop_trigger_dir_y = sign(total_diff.dy);
      } else {
        if (std::abs(state_.current_speed_y) <
            config_.max_speed_y / kSpeedDiffDirectionAssumeStoppedMaxRatio) {
          state_.current_speed_y = 0.0;
        } else {
          state_.current_speed_y /= kMaxSpeedDiffDirectionCutRateRatio;
        }
        if (config_.stop_translation_on_dir_change) {
          accel_y = total_diff.dy * 0.25f;
        }
      }
    }
  } // end of is_nonstop()

  // If braking is active, ignore input-derived accel for that axis
  if (state_.stop_delay_x && *state_.stop_delay_x != 0) {
    // Optional: cancel braking if input flips opposite of the trigger direction
    if (config_.cancel_stop_on_opposite_dir && sign(total_diff.dx) != 0.0f &&
        sign(total_diff.dx) == -state_.stop_trigger_dir_x) {
      // Hysteresis: require N consecutive frames before cancel
      if (config_.cancel_stop_hysteresis_frames > 0) {
        state_.cancel_opp_x_count += 1;
        if (state_.cancel_opp_x_count >= config_.cancel_stop_hysteresis_frames) {
          state_.stop_delay_x = zero();
          state_.stop_delay_x_counter = zero();
          state_.stop_decel_x = 0.0f;
          state_.canceled_stop_x = true;
          state_.cancel_opp_x_count = zero_int();
          state_.cooldown_x_counter = config_.stop_delay_cooldown_frames;
        }
      } else {
        state_.stop_delay_x = zero();
        state_.stop_delay_x_counter = zero();
        state_.stop_decel_x = 0.0f;
        state_.canceled_stop_x = true;
        state_.cooldown_x_counter = config_.stop_delay_cooldown_frames;
      }
    } else {
      state_.cancel_opp_x_count = zero_int();
      accel_x = state_.stop_decel_x;
    }
  }
  if (state_.stop_delay_y && *state_.stop_delay_y != 0) {
    if (config_.cancel_stop_on_opposite_dir && sign(total_diff.dy) != 0.0f &&
        sign(total_diff.dy) == -state_.stop_trigger_dir_y) {
      if (config_.cancel_stop_hysteresis_frames > 0) {
        state_.cancel_opp_y_count += 1;
        if (state_.cancel_opp_y_count >= config_.cancel_stop_hysteresis_frames) {
          state_.stop_delay_y = zero();
          state_.stop_delay_y_counter = zero();
          state_.stop_decel_y = 0.0f;
          state_.canceled_stop_y = true;
          state_.cancel_opp_y_count = zero_int();
          state_.cooldown_y_counter = config_.stop_delay_cooldown_frames;
        }
      } else {
        state_.stop_delay_y = zero();
        state_.stop_delay_y_counter = zero();
        state_.stop_decel_y = 0.0f;
        state_.canceled_stop_y = true;
        state_.cooldown_y_counter = config_.stop_delay_cooldown_frames;
      }
    } else {
      state_.cancel_opp_y_count = zero_int();
      accel_y = state_.stop_decel_y;
    }
  }

  adjust_speed(accel_x, accel_y, /*scale_constraints=*/x_gaussian);

  // Time-to-destination speed limiting (per-axis)
  auto limit_speed_ttg = [&](FloatValue& v, const FloatValue dist) {
    if (config_.time_to_dest_speed_limit_frames > 0) {
      const FloatValue sgn = sign(dist);
      if (sgn != 0.0f) {
        const FloatValue new_sgn = sign(v);
        if (new_sgn == sgn) {
          const FloatValue limit = std::abs(dist) /
              static_cast<FloatValue>(config_.time_to_dest_speed_limit_frames);
          // Clamp magnitude to at most limit
          const FloatValue vmax = limit;
          auto v1 = clamp(v, -vmax, vmax);
          if (v1 != v) {
            // std::cout << "FPS-clamped speed from " << v << " to " << v1 << " over dist "
            //           << dist << " in " << config_.time_to_dest_speed_limit_frames
            //           << " frames.\n";
          }
          v = v1;
          if (config_.time_to_dest_stop_speed_threshold > 0.0f) {
            const FloatValue thresh = config_.time_to_dest_stop_speed_threshold;
            if (std::abs(dist) <=
                    thresh * static_cast<FloatValue>(config_.time_to_dest_speed_limit_frames) &&
                std::abs(v) <= thresh) {
              v = 0.0f;
            }
          }
        }
      }
    }
  };
  limit_speed_ttg(state_.current_speed_x, total_diff.dx);
  limit_speed_ttg(state_.current_speed_y, total_diff.dy);

  // Clamp overshoot during braking to avoid reversing direction
  if (state_.stop_delay_x && *state_.stop_delay_x != 0) {
    const FloatValue next_speed_x = state_.current_speed_x;
    if (std::abs(next_speed_x) < std::abs(state_.stop_decel_x) + kEpsilon) {
      state_.current_speed_x = 0.0f;
    }
  }
  if (state_.stop_delay_y && *state_.stop_delay_y != 0) {
    const FloatValue next_speed_y = state_.current_speed_y;
    if (std::abs(next_speed_y) < std::abs(state_.stop_decel_y) + kEpsilon) {
      state_.current_speed_y = 0.0f;
    }
  }
}

const TranslationState& TranslatingBox::get_state() const {
  return state_;
}

PointDiff TranslatingBox::get_proposed_next_position_change() const {
  if (state_.translation_is_frozen) {
    return PointDiff{.dx = zero(), .dy = zero()};
  }
  return PointDiff{.dx = state_.current_speed_x, .dy = state_.current_speed_y};
}

void TranslatingBox::stop_translation_if_out_of_arena() {
  if (!config_.arena_box.has_value()) {
    return;
  }
  std::tuple<bool, bool> x_y_on_edge = check_for_box_overshoot(
      bounding_box(),
      config_.arena_box->inflate(1, 1, -1, -1),
      /*moving_directions=*/
      PointDiff{.dx = state_.current_speed_x, .dy = state_.current_speed_y},
      /*epsilon=*/0.1);
  state_.current_speed_x *= !std::get<0>(x_y_on_edge);
  state_.current_speed_y *= !std::get<1>(x_y_on_edge);
}

// After new position is set, adjust the nonstop-delay
void TranslatingBox::update_nonstop_delay() {
  if (state_.nonstop_delay != zero()) {
    state_.nonstop_delay_counter += 1;
    if (state_.nonstop_delay_counter > state_.nonstop_delay) {
      state_.nonstop_delay = zero();
      state_.nonstop_delay_counter = zero();
      // Optional braking after nonstop (breakaway catch-up) completes
      if (config_.post_nonstop_stop_delay_count > 0) {
        begin_stop_delay(
            /*delay_x=*/config_.post_nonstop_stop_delay_count,
            /*delay_y=*/std::nullopt);
      }
    }
  }
}

void TranslatingBox::update_stop_delays() {
  // X-axis stop delay
  if (state_.stop_delay_x != zero()) {
    state_.stop_delay_x_counter += 1;
    if (state_.stop_delay_x_counter >= state_.stop_delay_x) {
      state_.stop_delay_x = zero();
      state_.stop_delay_x_counter = zero();
      state_.stop_decel_x = 0.0f;
      state_.current_speed_x = 0.0f; // ensure fully stopped
      if (config_.stop_delay_cooldown_frames > 0) {
        state_.cooldown_x_counter = config_.stop_delay_cooldown_frames;
      }
    }
  }
  // Y-axis stop delay
  if (state_.stop_delay_y != zero()) {
    state_.stop_delay_y_counter += 1;
    if (state_.stop_delay_y_counter >= state_.stop_delay_y) {
      state_.stop_delay_y = zero();
      state_.stop_delay_y_counter = zero();
      state_.stop_decel_y = 0.0f;
      state_.current_speed_y = 0.0f;
      if (config_.stop_delay_cooldown_frames > 0) {
        state_.cooldown_y_counter = config_.stop_delay_cooldown_frames;
      }
    }
  }
  // Decrement cooldowns
  if (state_.cooldown_x_counter > 0) {
    state_.cooldown_x_counter -= 1;
  }
  if (state_.cooldown_y_counter > 0) {
    state_.cooldown_y_counter -= 1;
  }
}

const TranslatingBoxConfig& TranslatingBox::get_config() const {
  return config_;
}

void TranslatingBox::on_new_position() {
  if (!config_.arena_box.has_value()) {
    return;
  }
  const ShiftResult shift_result =
      shift_box_to_edge(bounding_box(), *config_.arena_box);
  if (shift_result.was_shifted_x) {
    // Abrupt slow-down at the edge: keep immediate halving rather than braking
    state_.current_speed_x /= kMaxSpeedDiffDirectionCutRateRatio;
  }
  if (shift_result.was_shifted_y) {
    // Abrupt slow-down at the edge: keep immediate halving rather than braking
    state_.current_speed_y /= kMaxSpeedDiffDirectionCutRateRatio;
  }
  if (shift_result.was_shifted_x || shift_result.was_shifted_y) {
    set_bbox(shift_result.bbox);
  }
}

bool TranslatingBox::is_nonstop() const {
  return state_.nonstop_delay != 0;
}

void TranslatingBox::clamp_speed(FloatValue scale) {
  state_.current_speed_x = clamp(
      state_.current_speed_x,
      -config_.max_speed_x * scale,
      config_.max_speed_x * scale);
  state_.current_speed_y = clamp(
      state_.current_speed_y,
      -config_.max_speed_y * scale,
      config_.max_speed_y * scale);
}

void TranslatingBox::adjust_speed(
    std::optional<FloatValue> accel_x,
    std::optional<FloatValue> accel_y,
    std::optional<FloatValue> scale_constraints,
    std::optional<IntValue> nonstop_delay) {
  if (scale_constraints.has_value()) {
    const FloatValue mult = *scale_constraints;
    if (accel_x.has_value()) {
      accel_x = clamp(
          *accel_x, -config_.max_accel_x * mult, config_.max_accel_x * mult);
    }
    if (accel_y.has_value()) {
      accel_y = clamp(
          *accel_y, -config_.max_accel_y * mult, config_.max_accel_y * mult);
    }
  } else {
    assert(false); // always, for now
  }

  if (accel_x.has_value()) {
    state_.current_speed_x += *accel_x;
  }

  if (accel_y.has_value()) {
    state_.current_speed_y += *accel_y;
  }

  if (scale_constraints.has_value()) {
    clamp_speed(*scale_constraints);
  }
  if (nonstop_delay.has_value()) {
    state_.nonstop_delay = std::move(nonstop_delay);
    state_.nonstop_delay_counter = 0;
  }
}

void TranslatingBox::begin_stop_delay(
    std::optional<IntValue> delay_x, std::optional<IntValue> delay_y) {
  // A sustained overshoot can request braking on every frame. Keep the
  // original linear deceleration deadline instead of restarting it into an
  // asymptotic drift away from the target.
  const bool braking_x = state_.stop_delay_x && *state_.stop_delay_x != 0;
  if (delay_x.has_value() && *delay_x > 0 && !braking_x) {
    state_.stop_delay_x = *delay_x;
    state_.stop_delay_x_counter = 0;
    // Decelerate linearly to zero over N frames
    state_.stop_decel_x = -state_.current_speed_x /
        static_cast<FloatValue>(*delay_x);
    state_.stop_trigger_dir_x = sign(state_.current_speed_x);
  }
  const bool braking_y = state_.stop_delay_y && *state_.stop_delay_y != 0;
  if (delay_y.has_value() && *delay_y > 0 && !braking_y) {
    state_.stop_delay_y = *delay_y;
    state_.stop_delay_y_counter = 0;
    state_.stop_decel_y = -state_.current_speed_y /
        static_cast<FloatValue>(*delay_y);
    state_.stop_trigger_dir_y = sign(state_.current_speed_y);
  }
}

void TranslatingBox::set_braking_params(
    IntValue stop_on_dir_change_delay,
    bool cancel_on_opposite,
    IntValue cancel_hysteresis_frames,
    IntValue stop_delay_cooldown_frames,
    IntValue post_nonstop_stop_delay_count,
    IntValue time_to_dest_speed_limit_frames) {
  config_.stop_translation_on_dir_change_delay = stop_on_dir_change_delay;
  config_.cancel_stop_on_opposite_dir = cancel_on_opposite;
  config_.cancel_stop_hysteresis_frames = cancel_hysteresis_frames;
  config_.stop_delay_cooldown_frames = stop_delay_cooldown_frames;
  config_.post_nonstop_stop_delay_count = post_nonstop_stop_delay_count;
  config_.time_to_dest_speed_limit_frames = time_to_dest_speed_limit_frames;
}

void TranslatingBox::set_translation_constraints(
    FloatValue max_speed_x,
    FloatValue max_speed_y,
    FloatValue max_accel_x,
    FloatValue max_accel_y) {
  config_.max_speed_x = max_speed_x;
  config_.max_speed_y = max_speed_y;
  config_.max_accel_x = max_accel_x;
  config_.max_accel_y = max_accel_y;
  // Clamp current speeds to new limits
  clamp_speed(1.0);
}

void TranslatingBox::set_camera_geometry(
    FloatValue arena_angle_from_vertical,
    FloatValue dynamic_acceleration_scaling) {
  config_.arena_angle_from_vertical = arena_angle_from_vertical;
  config_.dynamic_acceleration_scaling = dynamic_acceleration_scaling;
}

/**
 * Scale the current speed by the given ratio
 */
void TranslatingBox::scale_speed(
    std::optional<FloatValue> ratio_x,
    std::optional<FloatValue> ratio_y,
    bool clamp_to_max) {
  if (ratio_x.has_value()) {
    state_.current_speed_x *= *ratio_x;
  }

  if (ratio_y.has_value()) {
    state_.current_speed_y *= *ratio_y;
  }

  if (clamp_to_max) {
    clamp_speed(1.0);
  }
}

FloatValue TranslatingBox::get_gaussian_y_about_width_center(
    FloatValue x) const {
  // Different than python
  if (!config_.arena_box.has_value()) {
    return 1.0;
  }
  // return 1.0;
  x = clamp(x, gasussian_clamp_lr->first, gasussian_clamp_lr->second);
  const FloatValue center_x = config_.arena_box->center().x;
  if (x < center_x) {
    x = -(center_x - x);
  } else if (x > center_x) {
    x = x - center_x;
  } else {
    return 1.0;
  }
  // Just simple linear scaling
  return 1 - x / center_x;
}

std::tuple<FloatValue, FloatValue> TranslatingBox::
    get_sticky_translation_sizes() const {
  const BBox bbox = bounding_box();
  const FloatValue gaussian_factor =
      1.0 - get_gaussian_y_about_width_center(bbox.center().x);
  constexpr FloatValue kGaussianMult = 6.0;
  const FloatValue gaussian_add = gaussian_factor * kGaussianMult;

  const FloatValue max_sticky_size =
      config_.max_speed_x * config_.sticky_translation_gaussian_mult +
      gaussian_add;
  FloatValue sticky_size =
      bbox.width() / config_.sticky_size_ratio_to_frame_width;
  sticky_size = std::min(sticky_size, max_sticky_size);

  // Ensure proper hysteresis: unsticky threshold should be >= sticky.
  FloatValue ratio = config_.unsticky_translation_size_ratio;
  if (ratio < 1.0f) {
    ratio = 1.0f / std::max(ratio, 1e-3f);
  }
  FloatValue unsticky_size = sticky_size * ratio;
  return std::make_tuple(sticky_size, unsticky_size);
}

static FloatValue adjusted_horizontal_distance_from_edge(
    FloatValue x,
    FloatValue y,
    const BBox& arena_box,
    const FloatValue field_edge_veritcal_angle) {
  const FloatValue half_height = arena_box.height() / 2;
  const FloatValue half_width = arena_box.width() / 2;
  // On the edges, the perspective will have only the top half or so with
  // ice/field, with the center of the field warping down to the bottom in the
  // middle
  FloatValue percent_y =
      (std::min(half_height, y) - arena_box.top) / half_height;
  FloatValue max_x_adjusted_distance =
      std::sin(field_edge_veritcal_angle) * half_height;
  FloatValue x_adjusted_distance = max_x_adjusted_distance * (1.0 - percent_y);

  FloatValue x_adjusted_width = half_width - x_adjusted_distance;
  assert(x_adjusted_width > 0);
  // Scale back x_adjusted_distance based on how close to center (linearly
  // interpolation)
  FloatValue dist_from_center = std::abs(x_adjusted_width - x);
  FloatValue dist_from_center_ratio = dist_from_center / x_adjusted_width;
  x_adjusted_distance *= dist_from_center_ratio;

  FloatValue adjusted_x_distance_from_edge =
      std::max(x - x_adjusted_distance, 0.0f);
  return std::min(adjusted_x_distance_from_edge, half_width);
}

void TranslatingBox::test_arena_edge_position_scale() {
  const BBox arena_box = *config_.arena_box;
  const Point arena_center = arena_box.center();
  assert(
      config_.arena_angle_from_vertical !=
      0); // should have been set to something, please.
  const FloatValue kDegreesEdgePerspecive = config_.arena_angle_from_vertical;
  FloatValue full_left = get_arena_edge_point_position_scale(
      Point{.x = 0, .y = arena_center.y}, arena_box, kDegreesEdgePerspecive);
  FloatValue arena_left = get_arena_edge_point_position_scale(
      Point{.x = arena_box.left, .y = arena_center.y},
      arena_box,
      kDegreesEdgePerspecive);
  FloatValue full_center = get_arena_edge_point_position_scale(
      Point{.x = arena_center.x, .y = arena_center.y},
      arena_box,
      kDegreesEdgePerspecive);
  FloatValue full_right = get_arena_edge_point_position_scale(
      Point{.x = arena_box.right * 2, .y = arena_center.y},
      arena_box,
      kDegreesEdgePerspecive);
  FloatValue arena_right = get_arena_edge_point_position_scale(
      Point{.x = arena_box.right, .y = arena_center.y},
      arena_box,
      kDegreesEdgePerspecive);
  // The edge cases, far edges = 1, dead center = 0
  assert(is_close(full_left, -1.0f));
  assert(is_close(full_left, arena_left));
  assert(is_close(full_right, 1.0f));
  assert(is_close(full_right, arena_right));
  assert(is_close(full_center, 0.0f));

  const FloatValue kSmallTestDistance = arena_box.width() / 25;

  // Now check that the calculation is continuous near those edge points
  // (doesn't jump to some crazy value)
  FloatValue far_left = get_arena_edge_point_position_scale(
      Point{.x = arena_box.left + kSmallTestDistance, .y = arena_center.y},
      arena_box,
      kDegreesEdgePerspecive);
  FloatValue far_right = get_arena_edge_point_position_scale(
      Point{.x = arena_box.right - kSmallTestDistance, .y = arena_center.y},
      arena_box,
      kDegreesEdgePerspecive);
  FloatValue left_of_center = get_arena_edge_point_position_scale(
      Point{.x = arena_center.x - kSmallTestDistance, .y = arena_center.y},
      arena_box,
      kDegreesEdgePerspecive);
  FloatValue right_of_center = get_arena_edge_point_position_scale(
      Point{.x = arena_center.x + kSmallTestDistance, .y = arena_center.y},
      arena_box,
      kDegreesEdgePerspecive);

  // std::cout << far_left << ", " << left_of_center << ", " << right_of_center
  //           << ", " << far_right << std::endl;

  assert(is_close(-far_left, far_right));
  assert(is_close(-left_of_center, right_of_center));
  assert(far_left < left_of_center && far_right > right_of_center);
  // Don't want to restrict/define the algorithm in this test, so just make sure
  // it's some reasonable value.
  assert(std::fabs(1.0 - std::abs(far_left)) < kSmallTestDistance / 10);
  assert(std::abs(left_of_center) < kSmallTestDistance / 10);

  // Now check difference in Y changes
  FloatValue adjusted_x_last = 0, first_adjusted_x = 0.0;
  for (FloatValue y = arena_box.bottom; y >= 0.0f; y -= 100) {
    FloatValue left_x = arena_box.center().x - arena_box.left / 2;
    FloatValue adjusted_x = adjusted_horizontal_distance_from_edge(
        left_x, y, arena_box, kDegreesEdgePerspecive);
    if (adjusted_x_last) {
      assert(adjusted_x <= adjusted_x_last);
    } else {
      first_adjusted_x = adjusted_x;
    }
    adjusted_x_last = adjusted_x;
  }
  // It should have decreased with decreasing y
  assert(adjusted_x_last <= first_adjusted_x);
}

FloatValue TranslatingBox::get_arena_edge_point_position_scale(
    const Point& pt,
    const BBox& arena_box,
    const FloatValue field_edge_veritcal_angle) {
  const FloatValue arena_center_x = arena_box.center().x;
  if (pt.x == arena_center_x) {
    return 0.0;
  }
  FloatValue x_eff;
  if (pt.x < arena_center_x) {
    x_eff = std::max(0.0f, pt.x - arena_box.left);
  } else {
    x_eff = std::max(0.0f, arena_box.right - pt.x);
  }
  FloatValue x_eff_adjusted = adjusted_horizontal_distance_from_edge(
      x_eff, pt.y, arena_box, field_edge_veritcal_angle);
  const FloatValue half_arena_width = arena_box.width() / 2;
  FloatValue ratio = 1.0f - (x_eff_adjusted / half_arena_width);
  // Negative just means we're to the left
  return pt.x < arena_center_x ? -ratio : ratio;
}

FloatValue TranslatingBox::get_arena_edge_center_position_scale() const {
  assert(config_.arena_box.has_value());
  const BBox bbox = bounding_box();
  ;
  FloatValue ratio = get_arena_edge_point_position_scale(
      bbox.center(), *config_.arena_box, config_.arena_angle_from_vertical);
  // std::cout << "X Scale Ratio: " << ratio << "\n";
  return ratio;
}

} // namespace play_tracker
} // namespace hm
