#pragma once

#include "hockeymon/csrc/play_tracker/BoxUtils.h"
#include "hockeymon/csrc/play_tracker/PlayerSTrack.h"

#include <optional>
#include <set>
#include <unordered_map>
#include <vector>

namespace hm {
namespace play_tracker {

struct IBreakawayAdjuster {
  virtual ~IBreakawayAdjuster() = default;

  virtual void adjust_speed(
      std::optional<FloatValue> accel_x,
      std::optional<FloatValue> accel_y,
      std::optional<FloatValue> scale_constraints,
      std::optional<IntValue> nonstop_delay) = 0;

  /**
   * Scale the current speed by the given ratio
   */
  virtual void scale_speed(
      std::optional<FloatValue> ratio_x,
      std::optional<FloatValue> ratio_y,
      bool clamp_to_max = false) = 0;

  // Begin a per-axis stop delay (frames) for braking
  virtual void begin_stop_delay(
      std::optional<IntValue> delay_x,
      std::optional<IntValue> delay_y) = 0;
};

struct PlayDetectorConfig {
  // Trajectory/velocity config
  size_t max_positions{30};
  size_t max_velocity_positions{10};
  size_t frame_step{1};

  // Different fps have different speed scaling
  // Default is 1.0 for 29.97 fps (~30fps)
  float fps_speed_scale{1.0};

  // Breakaway detection
  // Group velocities less than this are ignored
  float min_considered_group_velocity = 3.0;
  // What ratio of tracked players must be all reacting
  // to a breakaway?
  float group_ratio_threshold = 0.5;
  // What ratio of the group speed should we try to
  // apply within a single frame's speed adjustment?
  float group_velocity_speed_ratio = 0.3;
  // During a breakaway, what scale do we adjust the
  // maximum camera speed to allow for faster-than-normal tracking?
  float scale_speed_constraints = 2.0;
  // The number of frame steps to not allow any camera-stop conditions to
  // trigger such as cluster(s) direction change or "sticky translation" rules.
  // This is because the clusters may not have changed enough at the
  // beginning of a breakaway and may simply stop the camera movement
  // due to the cluster tracking rules.
  size_t nonstop_delay_count = 2;
  // When over-shooting the breakaway players, what scale do we apply
  // to the current speed each frame in order to slow it down?
  float overshoot_scale_speed_ratio = 0.7;
  // If >0, instead of multiplicative speed scaling on overshoot,
  // initiate a stop-delay on X to brake over N frames while ignoring inputs.
  int overshoot_stop_delay_count{0};
};

struct PlayDetectorResults {
  // Center of the object that is farthest forward in the breakaway (logically
  // the one that we think most likely has the puck/ball)
  std::optional<Point> breakaway_edge_center;
  std::optional<BBox> breakaway_target_bbox;
};

class PlayDetector {
  using Velocity = PointDiff;

 public:
  PlayDetector(const PlayDetectorConfig& config, IBreakawayAdjuster* adjuster);

  // Live tuning setters (overrides)
  void set_overshoot_stop_delay_count(int v);
  void set_overshoot_scale_speed_ratio(float v);

  PlayDetectorResults forward(
      size_t frame_id,
      const BBox& current_target_bbox,
      const Point& current_roi_center,
      std::vector<size_t>& tracking_ids,
      std::vector<BBox>& tracking_boxes,
      const std::set<size_t>& disregard_tracking_ids);

  void reset();

 private:
  const PlayDetectorConfig config_;
  IBreakawayAdjuster* adjuster_;

  // Overrides (if set) for live tuning
  std::optional<int> overshoot_stop_delay_override_;
  std::optional<float> overshoot_scale_override_;

  struct TrackStateInfo {
    std::unordered_map<size_t, Velocity> track_velocity;
    Velocity cumulative_velocity;
  };

  TrackStateInfo update_tracks(
      size_t frame_id,
      std::vector<size_t>& tracking_ids,
      std::vector<BBox>& tracking_boxes,
      const std::set<size_t>& disregard_tracking_ids);

  // Function to compute the average velocity of the top N fastest-moving
  // velocities with the same dx sign
  struct GroupMovementInfo {
    Velocity group_velocity{0.0, 0.0};
    std::optional<Point> leftmost_center;
    std::optional<Point> rightmost_center;
  };

  std::optional<GroupMovementInfo> get_group_velocity(
      const std::unordered_map<size_t, Velocity>& track_velocities);

  std::optional<std::tuple<BBox, Point>> detect_breakaway(
      const BBox& current_box,
      const Point& current_roi_center,
      const TrackStateInfo& track_state_info,
      bool average_boxes = true);

  using TrackingMap = std::unordered_map</*tracking_id=*/size_t, PlayerSTrack>;

  TrackingMap tracks_;
};

} // namespace play_tracker
} // namespace hm
