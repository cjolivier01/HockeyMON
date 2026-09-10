#pragma once

#include "hockeymon/csrc/play_tracker/LivingBoxImpl.h"
#include "hockeymon/csrc/play_tracker/PlayTracker.h"

#include <cstdint>
#include <deque>
#include <string>
#include <vector>

namespace hm {
namespace play_tracker {

// Increment when the camera-policy implementation or state interpretation
// changes incompatibly. Persistence is explicit fields, never object bytes.
inline constexpr uint32_t kPlayTrackerSnapshotSchema = 1;
inline constexpr const char* kPlayTrackerSnapshotImplementation =
    "hockeymon-play-tracker-2026-09-replay-v1";

struct LivingBoxSnapshot {
  AllLivingBoxConfig config;
  BBox bbox;
  LivingState living;
  TranslationState translation;
  ResizingState resizing;
  uint64_t forward_counter{0};
};

struct PlayerTrackSnapshot {
  uint64_t tracking_id{0};
  std::deque<Point> positions;
  std::deque<Point> centers;
  std::deque<PointDiff> position_diffs;
  PointDiff sum_position_diffs{0.0f, 0.0f};
  uint64_t last_frame_id{0};
};

struct PlayDetectorSnapshot {
  std::optional<int> overshoot_stop_delay_override;
  std::optional<float> overshoot_scale_override;
  // Sorted by tracking ID for stable serialization. Iteration order of the
  // detector's lookup table is not used to compute player velocities.
  std::vector<PlayerTrackSnapshot> tracks;
};

struct PlayTrackerSnapshot {
  uint32_t schema_version{kPlayTrackerSnapshotSchema};
  std::string implementation{kPlayTrackerSnapshotImplementation};
  // Original tracker config, including the player filter's current settings.
  // Living-box runtime overrides reside in each box's effective config.
  PlayTrackerConfig config;
  PlayTrackerState state;
  std::vector<LivingBoxSnapshot> living_boxes;
  PlayDetectorSnapshot detector;
};

// All entry points reject malformed/incompatible input with invalid_argument.
// Limits cap persisted input to 16 MiB, 64 boxes, 4096 tracked players, and
// 4096 history entries per player. Restore creates an independent tracker.
void validate_snapshot(const PlayTrackerSnapshot& snapshot);
std::string serialize_snapshot(const PlayTrackerSnapshot& snapshot);
PlayTrackerSnapshot deserialize_snapshot(const std::string& contents);

} // namespace play_tracker
} // namespace hm
