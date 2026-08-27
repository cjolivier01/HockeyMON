#include "hockeymon/csrc/play_tracker/ResizingBox.h"

#include <cassert>
#include <iostream>

#include <unistd.h>

namespace hm {
namespace play_tracker {

namespace {
constexpr FloatValue kMaxWidthHeightDiffDirectionAssumeStoppedMaxRatio = 6.0;
constexpr FloatValue kMaxWidthHeightDiffDirectionCutRateRatio = 2.0;
constexpr FloatValue kResizeLargerScaleDifference = 2.0;
constexpr FloatValue kEpsilon = 1e-4f;
} // namespace

ResizingBox::ResizingBox(const ResizingConfig& config) : config_(config) {}

void ResizingBox::set_destination(const BBox& dest_box) {
  if (!config_.resizing_enabled) {
    return;
  }
  set_destination_size(dest_box.width(), dest_box.height());
}

const ResizingConfig& ResizingBox::get_config() const {
  return config_;
}

const ResizingState& ResizingBox::get_state() const {
  return state_;
}

SizeDiff ResizingBox::get_proposed_next_size_change() const {
  // std::cout << name() << ": current_speed_w = " << state_.current_speed_w
  //           << ", current_speed_h = " << state_.current_speed_h << std::endl;
  if (state_.size_is_frozen) {
    return SizeDiff{.dw = zero(), .dh = zero()};
  }
  return SizeDiff{
      .dw = state_.current_speed_w / 2, .dh = state_.current_speed_h / 2};
}

WHDims ResizingBox::get_min_allowed_width_height() const {
  return WHDims{.width = config_.min_width, .height = config_.min_height};
}

void ResizingBox::clamp_size_scaled() {
  const BBox bbox = bounding_box();
  float w = bbox.width();
  float h = bbox.height();
  float wscale = zero(), hscale = zero();
  if (w > config_.max_width) {
    wscale = config_.max_width / w;
  }
  if (h > config_.max_height) {
    hscale = config_.max_height / h;
  }
  float final_scale = std::max(wscale, hscale);
  if (!isZero(final_scale)) {
    w *= final_scale;
    h *= final_scale;
    clamp_if_close(w, 0.0, config_.max_width);
    clamp_if_close(h, 0.0, config_.max_height);
    set_bbox(BBox(bbox.center(), WHDims{.width = w, .height = h}));
  }
}

void ResizingBox::set_destination_size(
    FloatValue dest_width,
    FloatValue dest_height,
    bool prioritize_width_thresh) {
  BBox bbox = bounding_box();
  auto current_w = bbox.width();
  auto current_h = bbox.height();

  auto dw = dest_width - current_w;
  auto dh = dest_height - current_h;

  if (config_.sticky_sizing) {
    //
    // Begin size threshholding stuff
    //
    GrowShrink resize_rates = get_grow_shrink_wh(bbox);

    // dw
    bool dw_thresh = dw < 0 && dw < -resize_rates.shrink_width;
    const bool want_bigger_w = dw > 0 && dw > resize_rates.grow_width;
    dw_thresh |= want_bigger_w;
    if (!dw_thresh) {
      dw = 0.0f;
    }

    // dh
    bool dh_thresh = dh < 0 && dh < -resize_rates.shrink_height;
    const bool want_bigger_h = dh > 0 && dh > resize_rates.grow_height;
    dh_thresh |= want_bigger_h;
    if (!dh_thresh) {
      dh = 0.0f;
    }

    bool freeze_size = !dw_thresh && !dh_thresh;

    bool both_thresh = dw_thresh && dh_thresh;
    if (prioritize_width_thresh) {
      both_thresh |= dw_thresh;
    }
    const bool any_thresh = dw_thresh || dh_thresh;
    const bool want_bigger = (want_bigger_w || want_bigger_h) && any_thresh;

    // bool was_frozen = state_.size_is_frozen;
    state_.size_is_frozen = freeze_size || !(both_thresh || want_bigger);
    // if (was_frozen && !state_.size_is_frozen) {
    //   std::cout << "Unfreezing size" << std::endl;
    // }
    // if (state_.size_is_frozen) {
    //   usleep(0);
    // }

    //
    // End size threshholding stuff
    //
  }

  state_.canceled_stop_w = false;
  state_.canceled_stop_h = false;

  FloatValue accel_w = dw;
  FloatValue accel_h = dh;

  // Only consider triggering new stop-delays if not already braking on axis
  if ((!state_.stop_delay_w || *state_.stop_delay_w == 0) &&
      state_.cooldown_w_counter == 0 &&
      different_directions(dw, state_.current_speed_w)) {
    const bool moving_enough_w =
        std::abs(state_.current_speed_w) >=
        (config_.max_speed_w /
         kMaxWidthHeightDiffDirectionAssumeStoppedMaxRatio);
    if (config_.resizing_stop_on_dir_change_delay > 0 && moving_enough_w) {
      state_.stop_delay_w = config_.resizing_stop_on_dir_change_delay;
      state_.stop_delay_w_counter = 0;
      state_.stop_decel_w = -state_.current_speed_w /
          static_cast<FloatValue>(config_.resizing_stop_on_dir_change_delay);
      state_.stop_trigger_dir_w = sign(dw);
    } else {
      if (std::abs(state_.current_speed_w) <
          (config_.max_speed_w /
           kMaxWidthHeightDiffDirectionAssumeStoppedMaxRatio)) {
        state_.current_speed_w = 0.0f;
      } else {
        state_.current_speed_w /= kMaxWidthHeightDiffDirectionCutRateRatio;
      }
      if (config_.stop_resizing_on_dir_change) {
        accel_w = dw * 0.25f;
      }
    }
  }

  if ((!state_.stop_delay_h || *state_.stop_delay_h == 0) &&
      state_.cooldown_h_counter == 0 &&
      different_directions(dh, state_.current_speed_h)) {
    const bool moving_enough_h =
        std::abs(state_.current_speed_h) >=
        (config_.max_speed_h /
         kMaxWidthHeightDiffDirectionAssumeStoppedMaxRatio);
    if (config_.resizing_stop_on_dir_change_delay > 0 && moving_enough_h) {
      state_.stop_delay_h = config_.resizing_stop_on_dir_change_delay;
      state_.stop_delay_h_counter = 0;
      state_.stop_decel_h = -state_.current_speed_h /
          static_cast<FloatValue>(config_.resizing_stop_on_dir_change_delay);
      state_.stop_trigger_dir_h = sign(dh);
    } else {
      if (std::abs(state_.current_speed_h) <
          (config_.max_speed_h /
           kMaxWidthHeightDiffDirectionAssumeStoppedMaxRatio)) {
        state_.current_speed_h = 0.0f;
      } else {
        state_.current_speed_h /= kMaxWidthHeightDiffDirectionCutRateRatio;
      }
      if (config_.stop_resizing_on_dir_change) {
        accel_h = dh * 0.25f;
      }
    }
  }

  // If braking is active, ignore input-derived accel for that axis
  if (state_.stop_delay_w && *state_.stop_delay_w != 0) {
    if (config_.resizing_cancel_stop_on_opposite_dir && sign(dw) != 0.0f &&
        sign(dw) == -state_.stop_trigger_dir_w) {
      if (config_.resizing_stop_cancel_hysteresis_frames > 0) {
        state_.cancel_opp_w_count += 1;
        if (state_.cancel_opp_w_count >=
            config_.resizing_stop_cancel_hysteresis_frames) {
          state_.stop_delay_w = zero();
          state_.stop_delay_w_counter = zero();
          state_.stop_decel_w = 0.0f;
          state_.canceled_stop_w = true;
          state_.cancel_opp_w_count = zero_int();
          state_.cooldown_w_counter =
              config_.resizing_stop_delay_cooldown_frames;
        }
      } else {
        state_.stop_delay_w = zero();
        state_.stop_delay_w_counter = zero();
        state_.stop_decel_w = 0.0f;
        state_.canceled_stop_w = true;
        state_.cooldown_w_counter =
            config_.resizing_stop_delay_cooldown_frames;
      }
    } else {
      state_.cancel_opp_w_count = zero_int();
      accel_w = state_.stop_decel_w;
    }
  }
  if (state_.stop_delay_h && *state_.stop_delay_h != 0) {
    if (config_.resizing_cancel_stop_on_opposite_dir && sign(dh) != 0.0f &&
        sign(dh) == -state_.stop_trigger_dir_h) {
      if (config_.resizing_stop_cancel_hysteresis_frames > 0) {
        state_.cancel_opp_h_count += 1;
        if (state_.cancel_opp_h_count >=
            config_.resizing_stop_cancel_hysteresis_frames) {
          state_.stop_delay_h = zero();
          state_.stop_delay_h_counter = zero();
          state_.stop_decel_h = 0.0f;
          state_.canceled_stop_h = true;
          state_.cancel_opp_h_count = zero_int();
          state_.cooldown_h_counter =
              config_.resizing_stop_delay_cooldown_frames;
        }
      } else {
        state_.stop_delay_h = zero();
        state_.stop_delay_h_counter = zero();
        state_.stop_decel_h = 0.0f;
        state_.canceled_stop_h = true;
        state_.cooldown_h_counter =
            config_.resizing_stop_delay_cooldown_frames;
      }
    } else {
      state_.cancel_opp_h_count = zero_int();
      accel_h = state_.stop_decel_h;
    }
  }

  const FloatValue prev_speed_w = state_.current_speed_w;
  const FloatValue prev_speed_h = state_.current_speed_h;

  adjust_size(/*accel_w=*/accel_w, /*accel_h=*/accel_h);

  // Time-to-destination speed limiting (per-axis)
  auto limit_speed_ttg = [&](FloatValue& v, const FloatValue dist, const FloatValue prev_v) {
    if (config_.resizing_time_to_dest_speed_limit_frames > 0) {
      const FloatValue sgn = sign(dist);
      if (sgn != 0.0f) {
        const FloatValue new_sgn = sign(v);
        const bool increasing = std::abs(v) > std::abs(prev_v);
        if (new_sgn == sgn && increasing) {
          const FloatValue limit = std::abs(dist) /
              static_cast<FloatValue>(config_.resizing_time_to_dest_speed_limit_frames);
          const FloatValue vmax = limit;
          v = clamp(v, -vmax, vmax);
          if (config_.resizing_time_to_dest_stop_speed_threshold > 0.0f) {
            const FloatValue thresh = config_.resizing_time_to_dest_stop_speed_threshold;
            if (std::abs(dist) <=
                    thresh *
                        static_cast<FloatValue>(config_.resizing_time_to_dest_speed_limit_frames) &&
                std::abs(v) <= thresh) {
              v = 0.0f;
            }
          }
        }
      }
    }
  };
  limit_speed_ttg(state_.current_speed_w, dw, prev_speed_w);
  limit_speed_ttg(state_.current_speed_h, dh, prev_speed_h);

  // Clamp overshoot during braking to avoid reversing direction
  if (state_.stop_delay_w && *state_.stop_delay_w != 0) {
    const FloatValue next_speed_w = state_.current_speed_w;
    if (std::abs(next_speed_w) < std::abs(state_.stop_decel_w) + kEpsilon) {
      state_.current_speed_w = 0.0f;
    }
  }
  if (state_.stop_delay_h && *state_.stop_delay_h != 0) {
    const FloatValue next_speed_h = state_.current_speed_h;
    if (std::abs(next_speed_h) < std::abs(state_.stop_decel_h) + kEpsilon) {
      state_.current_speed_h = 0.0f;
    }
  }
}

void ResizingBox::adjust_size(
    FloatValue accel_w,
    FloatValue accel_h,
    bool use_constraints) {
  if (state_.size_is_frozen) {
    return;
  }

  if (use_constraints) {
    // Growing is allowed at a higher rate than shrinking
    const FloatValue max_accel_w = accel_w > 0
        ? (config_.max_accel_w * kResizeLargerScaleDifference)
        : config_.max_accel_w;
    const FloatValue max_accel_h = accel_h > 0
        ? (config_.max_accel_h * kResizeLargerScaleDifference)
        : config_.max_accel_h;

    accel_w = clamp(accel_w, -max_accel_w, max_accel_w);
    accel_h = clamp(accel_h, -max_accel_h, max_accel_h);
  }

  state_.current_speed_w += accel_w;
  state_.current_speed_h += accel_h;

  if (use_constraints) {
    clamp_resizing();
  }
}

void ResizingBox::clamp_resizing() {
  state_.current_speed_w =
      clamp(state_.current_speed_w, -config_.max_speed_w, config_.max_speed_w);
  state_.current_speed_h =
      clamp(state_.current_speed_h, -config_.max_speed_h, config_.max_speed_h);
}

void ResizingBox::update_stop_delays() {
  if (state_.stop_delay_w != zero()) {
    state_.stop_delay_w_counter += 1;
    if (state_.stop_delay_w_counter >= state_.stop_delay_w) {
      state_.stop_delay_w = zero();
      state_.stop_delay_w_counter = zero();
      state_.stop_decel_w = 0.0f;
      state_.current_speed_w = 0.0f;
      if (config_.resizing_stop_delay_cooldown_frames > 0) {
        state_.cooldown_w_counter =
            config_.resizing_stop_delay_cooldown_frames;
      }
    }
  }
  if (state_.stop_delay_h != zero()) {
    state_.stop_delay_h_counter += 1;
    if (state_.stop_delay_h_counter >= state_.stop_delay_h) {
      state_.stop_delay_h = zero();
      state_.stop_delay_h_counter = zero();
      state_.stop_decel_h = 0.0f;
      state_.current_speed_h = 0.0f;
      if (config_.resizing_stop_delay_cooldown_frames > 0) {
        state_.cooldown_h_counter =
            config_.resizing_stop_delay_cooldown_frames;
      }
    }
  }
  if (state_.cooldown_w_counter > 0) {
    state_.cooldown_w_counter -= 1;
  }
  if (state_.cooldown_h_counter > 0) {
    state_.cooldown_h_counter -= 1;
  }
}

GrowShrink ResizingBox::get_grow_shrink_wh(const BBox& bbox) const {
  auto ww = bbox.width(), hh = bbox.height();
  return GrowShrink{
      .grow_width = ww * config_.size_ratio_thresh_grow_dw,
      .grow_height = hh * config_.size_ratio_thresh_grow_dh,
      .shrink_width = ww * config_.size_ratio_thresh_shrink_dw,
      .shrink_height = hh * config_.size_ratio_thresh_shrink_dh,
  };
}

} // namespace play_tracker
} // namespace hm
