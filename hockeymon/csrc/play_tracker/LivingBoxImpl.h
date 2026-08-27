#pragma once

#include "hockeymon/csrc/play_tracker/BoxUtils.h"
#include "hockeymon/csrc/play_tracker/LivingBox.h"
#include "hockeymon/csrc/play_tracker/ResizingBox.h"
#include "hockeymon/csrc/play_tracker/TranslatingBox.h"

namespace hm {
namespace play_tracker {

struct LivingState {
  bool was_size_constrained{false};
};

class LivingBox : public ILivingBox,
                  public BoundingBox,
                  public ResizingBox,
                  public TranslatingBox {
 public:
  LivingBox(std::string label, BBox bbox, const AllLivingBoxConfig& config);

  // -ILivingBox
  void set_destination_ex(
      const std::variant<BBox, std::shared_ptr<IBasicLivingBox>>& dest)
      override;
  void set_dest(std::shared_ptr<IBasicLivingBox>) override {}
  void set_dest_ex(const std::variant<BBox, std::shared_ptr<IBasicLivingBox>>&
                       dest) override {}
  std::optional<BBox> get_arena_box() const override;
  // ILivingBox-

  WHDims get_size_scale() const;

  // -IBasicLivingBox
  void set_bbox(const BBox& bbox) override;

  BBox bounding_box() const override;

  void set_destination(const BBox& dest_box) override;
  // IBasicLivingBox-

  BBox forward(const std::variant<BBox, std::shared_ptr<IBasicLivingBox>>& dest)
      override;

  const std::string& name() const override {
    return label_;
  }

  const LivingBoxConfig& config() const {
    return config_;
  }
  const LivingState& get_live_box_state() const override {
    return state_;
  }

  const ResizingState& get_resizing_state() const override {
    return ResizingBox::get_state();
  }

  const TranslationState& get_translation_state() const override {
    return TranslatingBox::get_state();
  }

  GrowShrink get_grow_shrink_wh(const BBox& bbox) const override {
    return ResizingBox::get_grow_shrink_wh(bbox);
  }

  virtual std::tuple<FloatValue, FloatValue> get_sticky_translation_sizes()
      const override {
    return TranslatingBox::get_sticky_translation_sizes();
  }

  void adjust_speed(
      std::optional<FloatValue> accel_x,
      std::optional<FloatValue> accel_y,
      std::optional<FloatValue> scale_constraints,
      std::optional<IntValue> nonstop_delay) override {
    TranslatingBox::adjust_speed(
        accel_x, accel_y, scale_constraints, nonstop_delay);
  }

  void scale_speed(
      std::optional<FloatValue> ratio_x,
      std::optional<FloatValue> ratio_y,
      bool clamp_to_max = false) override {
    TranslatingBox::scale_speed(ratio_x, ratio_y, clamp_to_max);
  }

  void begin_stop_delay(
      std::optional<IntValue> delay_x, std::optional<IntValue> delay_y) override {
    TranslatingBox::begin_stop_delay(delay_x, delay_y);
  }

  void set_braking(
      IntValue stop_on_dir_change_delay,
      bool cancel_on_opposite,
      IntValue cancel_hysteresis_frames,
      IntValue stop_delay_cooldown_frames,
      IntValue post_nonstop_stop_delay_count) {
    TranslatingBox::set_braking_params(
        stop_on_dir_change_delay,
        cancel_on_opposite,
        cancel_hysteresis_frames,
        stop_delay_cooldown_frames,
        post_nonstop_stop_delay_count);
  }

  void set_translation_constraints(
      FloatValue max_speed_x,
      FloatValue max_speed_y,
      FloatValue max_accel_x,
      FloatValue max_accel_y) {
    TranslatingBox::set_translation_constraints(
        max_speed_x, max_speed_y, max_accel_x, max_accel_y);
  }

 protected:
  BBox next_position();

  void clamp_to_arena();

 private:
  const std::string label_;
  const LivingBoxConfig config_;
  // Copnstraints when "clamping to arena box" that we can calculate once
  // beforehand
  FloatValue max_eff_width_{0}, max_eff_height_{0};
  // Flag to show we were size-constrained on the last update
  // (debugging/visualization only)
  LivingState state_;
  std::size_t forward_counter_{0};
};

} // namespace play_tracker
} // namespace hm
