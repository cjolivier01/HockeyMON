#pragma once

#include "hockeymon/csrc/play_tracker/LivingBox.h"
#include "hockeymon/csrc/play_tracker/LogCapture.h"
#include "hockeymon/csrc/play_tracker/PlayDetector.h"

#include <array>
#include <limits>
#include <memory>
#include <unordered_map>
#include <vector>

namespace hm {
namespace play_tracker {

constexpr size_t kBadIdOrIndex = std::numeric_limits<size_t>::max();

struct PlayTrackerConfig {
  int no_wide_start{false};
  // For less than this, we just move towards the arena box
  size_t min_tracked_players{4};
  std::vector<AllLivingBoxConfig> living_boxes;
  // After this number of ticks, "lost" tracks are discarded
  size_t max_lost_track_age{30};
  bool ignore_largest_bbox{true};
  bool ignore_left_and_right_extremes{false};
  // Ignore a cluster item that's very far away from the next person on that
  // side
  size_t ignore_outlier_players{true};
  float ignore_outlier_players_dist_ratio{1.0};
  PlayDetectorConfig play_detector;
};

struct PlayTrackerState {
  // Just keep track of the number of iterations
  size_t tick_count_{0};
  size_t tracked_player_count{0};
};

struct Track {
  size_t tracking_id{kBadIdOrIndex};
  BBox bbox;
};

struct PlayTrackerResults {
  std::unordered_map<size_t, BBox> cluster_boxes;
  std::unordered_map<size_t, BBox> removed_cluster_outlier_box;
  BBox final_cluster_box;
  std::vector<BBox> tracking_boxes;
  std::optional<PlayDetectorResults> play_detection;
  // TODO: combine these ignored items
  std::optional<Track> largest_tracking_bbox;
  std::optional<Track> leftmost_tracking_bbox;
  std::optional<Track> rightmost_tracking_bbox;
  std::vector<HmLogMessage> log_messages;
};

class PlayTracker : public IBreakawayAdjuster {
  using Velocity = PointDiff;

  struct ClusterBoxes {
    std::unordered_map<size_t, BBox> cluster_boxes;
    std::unordered_map<size_t, BBox> removed_cluster_outlier_box;
    BBox final_cluster_box;
  };

 public:
  PlayTracker(const BBox& initial_box, const PlayTrackerConfig& config);
  virtual ~PlayTracker() = default;

  PlayTrackerResults forward(
      std::vector<size_t>& tracking_ids,
      std::vector<BBox>& tracking_boxes,
      bool debug_to_stdout = false);

  void set_bboxes(const std::vector<BBox>& bboxes);
  // TODO: this can be done via living box's get_size_scale()
  void set_bboxes_scaled(BBox bbox, float scale_step);

  std::shared_ptr<ILivingBox> get_live_box(size_t index) const;

  // Live tuning: breakaway braking
  void set_breakaway_braking(int overshoot_delay_count, float overshoot_scale_ratio) {
    play_detector_.set_overshoot_stop_delay_count(overshoot_delay_count);
    play_detector_.set_overshoot_scale_speed_ratio(overshoot_scale_ratio);
  }

 private:
  void create_boxes(const BBox& initial_box);
  ClusterBoxes get_cluster_boxes(const std::vector<BBox>& tracking_boxes) const;

  BBox get_play_box() const;

  // BEGIN IBreakawayAdjuster
  void adjust_speed(
      std::optional<FloatValue> accel_x,
      std::optional<FloatValue> accel_y,
      std::optional<FloatValue> scale_constraints,
      std::optional<IntValue> nonstop_delay) override {
    living_boxes_.at(0)->adjust_speed(
        accel_x, accel_y, scale_constraints, nonstop_delay);
  }

  void scale_speed(
      std::optional<FloatValue> ratio_x,
      std::optional<FloatValue> ratio_y,
      bool clamp_to_max = false) override {
    living_boxes_.at(0)->scale_speed(ratio_x, ratio_y, clamp_to_max);
  }

  void begin_stop_delay(
      std::optional<IntValue> delay_x, std::optional<IntValue> delay_y) override {
    living_boxes_.at(0)->begin_stop_delay(delay_x, delay_y);
  }
  // END IBreakawayAdjuster

  //
  // DATA
  //

  // Config
  const PlayTrackerConfig config_;

  // Cluster stuff
  const std::array<size_t, 2> cluster_sizes_{2, 3};

  // Tracking stuff
  std::vector<std::shared_ptr<ILivingBox>> living_boxes_;

  // Should be close to last since it has a pointer to this class
  PlayDetector play_detector_;

  PlayTrackerState state_;
};

} // namespace play_tracker
} // namespace hm
