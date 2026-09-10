#include "hockeymon/csrc/play_tracker/PlayTrackerSnapshot.h"

#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <locale>
#include <stdexcept>
#include <string>

namespace {

using hm::BBox;
using namespace hm::play_tracker;

void expect(bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

void expect_invalid(const std::function<void()>& operation) {
  bool rejected = false;
  try {
    operation();
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  expect(rejected, "invalid checkpoint was accepted");
}

PlayTrackerConfig config() {
  PlayTrackerConfig config;
  config.no_wide_start = true;
  config.ignore_largest_bbox = false;
  config.ignore_outlier_players = true;
  config.play_detector.max_positions = 24;
  config.play_detector.max_velocity_positions = 8;
  config.play_detector.overshoot_stop_delay_count = 4;
  config.play_detector.min_considered_group_velocity = 0.2f;
  for (int i = 0; i < 2; ++i) {
    AllLivingBoxConfig c;
    c.name = i == 0 ? "fast" : "follower";
    c.arena_box = BBox(0, 0, 2000, 1000);
    c.min_width = 300;
    c.min_height = 180;
    c.max_width = 2000;
    c.max_height = 1000;
    c.max_speed_x = i == 0 ? 18 : 8;
    c.max_speed_y = 3;
    c.max_accel_x = i == 0 ? 2.5f : 0.75f;
    c.max_accel_y = 0.5f;
    c.max_speed_w = 10;
    c.max_speed_h = 8;
    c.max_accel_w = 0.4f;
    c.max_accel_h = 0.4f;
    c.sticky_translation = true;
    c.sticky_size_ratio_to_frame_width = 20;
    c.stop_translation_on_dir_change_delay = 5;
    c.cancel_stop_on_opposite_dir = true;
    c.cancel_stop_hysteresis_frames = 2;
    c.stop_delay_cooldown_frames = 3;
    c.post_nonstop_stop_delay_count = 4;
    c.sticky_sizing = true;
    c.resizing_stop_on_dir_change_delay = 4;
    c.resizing_stop_delay_cooldown_frames = 2;
    config.living_boxes.push_back(c);
  }
  return config;
}

PlayTrackerResults advance(PlayTracker& tracker, size_t frame) {
  std::vector<size_t> ids;
  std::vector<BBox> boxes;
  // Sweeps, direction reversals, changing spread, missing players and complete
  // observation gaps exercise velocity/history and pan/zoom state together.
  if (frame % 150 < 144) {
    float x = 650.0f + 420.0f * std::sin(frame * 0.028f);
    float spread = 30.0f + 16.0f * std::sin(frame * 0.051f);
    for (size_t i = 0; i < 15; ++i) {
      if ((i + frame) % 29 == 0) {
        continue;
      }
      ids.push_back(i + 10);
      const float left = x + spread * i;
      const float top = 340.0f + 12.0f * (i % 4);
      boxes.emplace_back(left, top, left + 25, top + 95);
    }
  }
  return tracker.forward(ids, boxes);
}

bool equal_box(const BBox& a, const BBox& b) {
  return a.left == b.left && a.top == b.top && a.right == b.right &&
      a.bottom == b.bottom;
}

void compare_results(const PlayTrackerResults& a, const PlayTrackerResults& b) {
  expect(a.tracking_boxes.size() == b.tracking_boxes.size(), "box count");
  for (size_t i = 0; i < a.tracking_boxes.size(); ++i) {
    expect(equal_box(a.tracking_boxes[i], b.tracking_boxes[i]), "camera drift");
  }
  expect(equal_box(a.final_cluster_box, b.final_cluster_box), "cluster drift");
  expect(
      a.removed_cluster_outlier_box.size() ==
          b.removed_cluster_outlier_box.size(),
      "outlier results drift");
  for (const auto& [id, box] : a.removed_cluster_outlier_box) {
    expect(
        equal_box(box, b.removed_cluster_outlier_box.at(id)),
        "outlier geometry drift");
  }
}

void continuation_matches_uninterrupted() {
  PlayTracker original(BBox(0, 0, 2000, 1000), config());
  bool saw_translation = false;
  bool saw_resizing = false;
  bool saw_braking = false;
  bool saw_frozen = false;
  bool saw_history = false;
  std::unique_ptr<PlayTracker> replay;
  for (size_t frame = 0; frame < 520; ++frame) {
    if (frame == 45) {
      original.set_player_size_filter(1, true, 175);
      original.set_breakaway_braking(7, 0.63f);
      original.get_live_box(0)->set_translation_constraints(12, 4, 1.3f, 0.7f);
      original.get_live_box(1)->set_braking(7, true, 3, 4, 5, 11);
      original.get_live_box(1)->set_resizing_shrink_thresholds(0.041f, 0.067f);
      original.get_live_box(1)->set_camera_geometry(34.2f, 0.5f);
      replay.reset();
    }
    if (!replay || frame % 17 == 0) {
      auto before = original.snapshot();
      const std::string saved = serialize_snapshot(before);
      replay = PlayTracker::from_snapshot(deserialize_snapshot(saved));
      expect(
          serialize_snapshot(replay->snapshot()) == saved,
          "snapshot did not restore every field");
    }
    compare_results(advance(original, frame), advance(*replay, frame));
    const auto after = original.snapshot();
    expect(
        serialize_snapshot(after) == serialize_snapshot(replay->snapshot()),
        "hidden state drift after continuation at frame " +
            std::to_string(frame));
    for (const auto& box : after.living_boxes) {
      saw_translation |= box.translation.current_speed_x != 0;
      saw_resizing |= box.resizing.current_speed_w != 0;
      saw_braking |= box.translation.stop_delay_x.value_or(0) != 0 ||
          box.resizing.stop_delay_w.value_or(0) != 0;
      saw_frozen |=
          box.translation.translation_is_frozen || box.resizing.size_is_frozen;
    }
    for (const auto& track : after.detector.tracks) {
      saw_history |= track.position_diffs.size() >= 4;
    }
  }
  expect(
      saw_translation && saw_resizing && saw_braking && saw_frozen &&
          saw_history,
      "continuation fixture did not exercise all required native state");
}

PlayTrackerSnapshot moving_snapshot() {
  PlayTracker tracker(BBox(0, 0, 2000, 1000), config());
  for (size_t i = 0; i < 65; ++i) {
    advance(tracker, i);
  }
  // Public control methods can start braking at the exact fork boundary.
  tracker.get_live_box(0)->adjust_speed(1.0f, -0.25f, 1.0f, 3);
  tracker.get_live_box(0)->begin_stop_delay(6, 7);
  return tracker.snapshot();
}

void repeated_trials_are_independent() {
  const auto start = moving_snapshot();
  const auto saved = serialize_snapshot(start);
  auto original = PlayTracker::from_snapshot(start);
  auto first = PlayTracker::from_snapshot(start);
  auto second = PlayTracker::from_snapshot(deserialize_snapshot(saved));
  for (auto* trial : {first.get(), second.get()}) {
    trial->get_live_box(0)->set_translation_constraints(2, 2, 0.1f, 0.1f);
    trial->get_live_box(1)->set_translation_constraints(1, 1, 0.1f, 0.1f);
    trial->set_breakaway_braking(3, 0.2f);
  }
  bool changed = false;
  for (size_t frame = 65; frame < 120; ++frame) {
    const auto reference = advance(*original, frame);
    const auto a = advance(*first, frame);
    const auto b = advance(*second, frame);
    compare_results(a, b);
    changed |=
        !equal_box(reference.tracking_boxes.back(), a.tracking_boxes.back());
  }
  expect(changed, "trial constraints did not change the camera trajectory");
  expect(serialize_snapshot(start) == saved, "trial changed immutable state");
  expect(
      first->get_live_box(0).get() != second->get_live_box(0).get(),
      "trials shared a living box");
}

void rejects_malformed_state() {
  const auto valid = moving_snapshot();
  auto check =
      [&valid](const std::function<void(PlayTrackerSnapshot&)>& mutate) {
        auto altered = valid;
        mutate(altered);
        expect_invalid([&] { PlayTracker::from_snapshot(altered); });
        expect_invalid([&] { serialize_snapshot(altered); });
      };
  check([](auto& s) { ++s.schema_version; });
  check([](auto& s) { s.implementation = "different policy"; });
  check([](auto& s) { s.living_boxes.pop_back(); });
  check([](auto& s) { s.living_boxes[0].bbox.right = -1; });
  check([](auto& s) {
    s.living_boxes[0].translation.current_speed_x =
        std::numeric_limits<float>::quiet_NaN();
  });
  check([](auto& s) { s.living_boxes[0].resizing.cooldown_w_counter = -1; });
  check([](auto& s) { s.config.play_detector.max_velocity_positions = 0; });
  check(
      [](auto& s) { s.detector.tracks.push_back(s.detector.tracks.front()); });
  check([](auto& s) {
    s.detector.tracks[0].last_frame_id = s.state.tick_count_;
  });
  check([](auto& s) { s.detector.tracks[0].centers.clear(); });
  check([](auto& s) { s.config.living_boxes[0].arena_box.reset(); });
  const auto encoded = serialize_snapshot(valid);
  expect_invalid([&] { deserialize_snapshot(encoded + "extra"); });
  expect_invalid(
      [&] { deserialize_snapshot(encoded.substr(0, encoded.size() / 2)); });
  expect_invalid(
      [&] { deserialize_snapshot(std::string(16 * 1024 * 1024 + 1, ' ')); });
  auto wrong_field = encoded;
  wrong_field.replace(wrong_field.find("schema_version"), 14, "unknown_field_");
  expect_invalid([&] { deserialize_snapshot(wrong_field); });
  auto bad_count = encoded;
  const auto count = bad_count.find("living_boxes ");
  bad_count.replace(
      count, std::string("living_boxes 2").size(), "living_boxes 99999999");
  expect_invalid([&] { deserialize_snapshot(bad_count); });
  // Rejections must not affect the original checkpoint or a later valid load.
  expect(
      serialize_snapshot(PlayTracker::from_snapshot(valid)->snapshot()) ==
          encoded,
      "failed restore changed valid state");
}

struct CommaDecimal : std::numpunct<char> {
  char do_decimal_point() const override {
    return ',';
  }
};

void serialization_is_lossless_and_locale_independent() {
  auto snapshot = moving_snapshot();
  snapshot.living_boxes[0].translation.filtered_target_center =
      hm::Point{std::numeric_limits<float>::denorm_min(), -0.0f};
  const auto encoded = serialize_snapshot(snapshot);
  const auto old = std::locale();
  std::locale::global(std::locale(old, new CommaDecimal));
  try {
    const auto restored = deserialize_snapshot(encoded);
    expect(serialize_snapshot(restored) == encoded, "locale changed encoding");
    expect(
        std::signbit(
            restored.living_boxes[0].translation.filtered_target_center->y),
        "negative zero was lost");
  } catch (...) {
    std::locale::global(old);
    throw;
  }
  std::locale::global(old);
}

void full_history_capacity_restores() {
  auto settings = config();
  settings.play_detector.max_positions = 4;
  settings.play_detector.max_velocity_positions = 4;
  PlayTracker tracker(BBox(0, 0, 2000, 1000), settings);
  for (size_t i = 0; i < 12; ++i) {
    advance(tracker, i);
  }
  auto copy = PlayTracker::from_snapshot(
      deserialize_snapshot(serialize_snapshot(tracker.snapshot())));
  compare_results(advance(tracker, 12), advance(*copy, 12));
}

void initial_scaled_arena_restores_before_first_observation() {
  auto settings = config();
  settings.living_boxes[0].scale_dest_width = 1.5f;
  settings.living_boxes[1].scale_dest_height = 1.2f;
  PlayTracker tracker(BBox(0, 0, 2000, 1000), settings);
  const auto initial = tracker.snapshot();
  expect(
      initial.living_boxes[0].bbox.left < 0, "fixture needs initial overscan");
  auto restored = PlayTracker::from_snapshot(
      deserialize_snapshot(serialize_snapshot(initial)));
  compare_results(advance(tracker, 0), advance(*restored, 0));
}

} // namespace

int main() {
  try {
    continuation_matches_uninterrupted();
    repeated_trials_are_independent();
    rejects_malformed_state();
    serialization_is_lossless_and_locale_independent();
    full_history_capacity_restores();
    initial_scaled_arena_restores_before_first_observation();
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
  std::cout << "play-tracker snapshot tests passed\n";
  return 0;
}
