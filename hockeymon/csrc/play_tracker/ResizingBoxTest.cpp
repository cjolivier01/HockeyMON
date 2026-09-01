#include "hockeymon/csrc/play_tracker/LivingBoxImpl.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>

namespace {

bool near(float actual, float expected, float tolerance = 0.0001f) {
  return std::abs(actual - expected) < tolerance;
}

bool expect(bool condition, const char* message) {
  if (!condition)
    std::cerr << "FAIL: " << message << '\n';
  return condition;
}

bool expect_near(float actual, float expected, const char* message) {
  if (near(actual, expected))
    return true;
  std::cerr << "FAIL: " << message << " (actual=" << actual
            << ", expected=" << expected << ")\n";
  return false;
}

hm::BBox centered_box(float width, float height) {
  return hm::BBox(hm::Point{500.0f, 500.0f}, hm::WHDims{width, height});
}

hm::play_tracker::AllLivingBoxConfig base_config() {
  hm::play_tracker::AllLivingBoxConfig config;
  config.arena_box = hm::BBox(0, 0, 1000, 1000);
  config.translation_enabled = false;
  config.max_width = 1000;
  config.max_height = 1000;
  config.max_speed_w = 100;
  config.max_speed_h = 100;
  config.max_accel_w = 10;
  config.max_accel_h = 10;
  config.sticky_sizing = true;
  config.size_ratio_thresh_grow_dw = 0.1f;
  config.size_ratio_thresh_grow_dh = 0.1f;
  config.size_ratio_thresh_shrink_dw = 0.034f;
  config.size_ratio_thresh_shrink_dh = 0.034f;
  config.resizing_stop_on_dir_change_delay = 4;
  config.resizing_stop_delay_cooldown_frames = 2;
  config.resizing_time_to_dest_speed_limit_frames = 10;
  config.resizing_time_to_dest_stop_speed_threshold = 0.25f;
  return config;
}

bool test_deadband_brakes_and_clears_state() {
  auto config = base_config();
  hm::play_tracker::LivingBox box("deadband", centered_box(100, 100), config);

  box.forward(centered_box(300, 300));
  bool ok = true;
  const float initial_height_speed = box.get_resizing_state().current_speed_h;
  ok &= expect(
      initial_height_speed > 0.0f,
      "The initial resize should build height velocity");

  const float current_height = box.bounding_box().height();
  const hm::BBox width_only_destination = centered_box(300, current_height);
  box.forward(width_only_destination);
  ok &= expect(
      box.get_resizing_state().deadband_stop_h &&
          box.get_resizing_state().current_speed_h > 0.0f &&
          box.get_resizing_state().current_speed_h < initial_height_speed,
      "Entering the height deadband should brake rather than preserve resize velocity");

  for (int frame = 0; frame < 4; ++frame) {
    box.forward(centered_box(300, box.bounding_box().height()));
  }
  const auto& state = box.get_resizing_state();
  ok &= expect_near(
      state.current_speed_h,
      0.0f,
      "Deadband braking should reach zero in finite time");
  ok &= expect(
      !state.deadband_stop_h && state.stop_delay_h == 0 &&
          state.stop_delay_h_counter == 0 && near(state.stop_decel_h, 0.0f) &&
          near(state.stop_trigger_dir_h, 0.0f) && state.cancel_opp_h_count == 0,
      "Stopping should clear stale per-axis braking state");
  return ok;
}

bool test_speed_limit_uses_deadband_edge() {
  auto config = base_config();
  hm::play_tracker::LivingBox box("edge", centered_box(100, 100), config);
  box.forward(centered_box(300, 300));

  const float width = box.bounding_box().width();
  const float height = box.bounding_box().height();
  constexpr float kDistancePastEdge = 10.0f;
  const hm::BBox destination = centered_box(
      width + width * config.size_ratio_thresh_grow_dw + kDistancePastEdge,
      height + height * config.size_ratio_thresh_grow_dh + kDistancePastEdge);
  box.forward(destination);

  const float expected_limit = kDistancePastEdge /
      static_cast<float>(config.resizing_time_to_dest_speed_limit_frames);
  const auto& state = box.get_resizing_state();
  return expect(
      near(state.current_speed_w, expected_limit) &&
          near(state.current_speed_h, expected_limit),
      "Sticky resize speed should be limited by distance to the deadband edge");
}

bool test_steady_fixed_aspect_target_does_not_oscillate() {
  auto config = base_config();
  config.arena_box = hm::BBox(0, 0, 3469, 1138);
  config.max_width = 3469;
  config.max_height = 1138;
  config.max_speed_w = 9.64f;
  config.max_speed_h = 6.67f;
  config.max_accel_w = 1.0f;
  config.max_accel_h = 1.0f;
  config.fixed_aspect_ratio = 16.0f / 9.0f;

  hm::play_tracker::LivingBox box(
      "follower",
      hm::BBox(hm::Point{1734.5f, 569.0f}, hm::WHDims{1600, 900}),
      config);
  const hm::BBox narrow_tall_target(
      hm::Point{1734.5f, 569.0f}, hm::WHDims{1000, 850});
  float min_tail_width = std::numeric_limits<float>::max();
  float max_tail_width = 0.0f;
  for (int frame = 0; frame < 600; ++frame) {
    const hm::BBox output = box.forward(narrow_tall_target);
    if (frame >= 480) {
      min_tail_width = std::min(min_tail_width, output.width());
      max_tail_width = std::max(max_tail_width, output.width());
    }
  }

  const auto& state = box.get_resizing_state();
  bool ok = true;
  ok &= expect(
      max_tail_width - min_tail_width < 0.01f,
      "A steady narrow/tall target must not create a fixed-aspect zoom limit cycle");
  ok &= expect_near(
      state.current_speed_h,
      0.0f,
      "The steady follower should finish with no latent height velocity");
  ok &= expect(
      !state.deadband_stop_h && state.stop_delay_h == 0,
      "The steady follower should clear height braking state after settling");

  const float settled_width = box.bounding_box().width();
  for (int frame = 0; frame < 20; ++frame)
    box.forward(hm::BBox(hm::Point{1734.5f, 569.0f}, hm::WHDims{600, 500}));
  ok &= expect(
      box.bounding_box().width() < settled_width - 1.0f,
      "An active resize outside the deadband should continue responding");
  return ok;
}

} // namespace

int main() {
  bool ok = true;
  ok &= test_deadband_brakes_and_clears_state();
  ok &= test_speed_limit_uses_deadband_edge();
  ok &= test_steady_fixed_aspect_target_does_not_oscillate();
  return ok ? 0 : 1;
}
