#include "hockeymon/csrc/play_tracker/PlayerSizeFilter.h"
#include "hockeymon/csrc/play_tracker/PlayTracker.h"

#include <iostream>
#include <limits>

namespace {
bool expect(bool condition, const char* message) {
  if (!condition)
    std::cerr << "FAIL: " << message << '\n';
  return condition;
}
std::vector<hm::BBox> boxes(std::initializer_list<int> areas) {
  std::vector<hm::BBox> result;
  for (int area : areas)
    result.emplace_back(0, 0, area, 1);
  return result;
}
} // namespace

int main() {
  using hm::play_tracker::player_size_exclusions;
  bool ok = true;
  const auto players = boxes({100, 100, 100, 100, 300, 1000});
  ok &= expect(
      player_size_exclusions(players, 0, false, 100).empty(),
      "Disabled filters retain every player");
  ok &= expect(
      player_size_exclusions(players, 1, false, 100) == std::vector<size_t>{5},
      "Default excludes one largest");
  ok &= expect(
      player_size_exclusions(players, 2, false, 100) ==
          std::vector<size_t>({5, 4}),
      "Count excludes N largest");
  ok &= expect(
      player_size_exclusions(players, 100000, false, 100) ==
          std::vector<size_t>({5, 4, 0}),
      "Large counts preserve three, ties retain input order");
  ok &= expect(
      player_size_exclusions(players, 1, true, 100) ==
          std::vector<size_t>({5, 4}),
      "Mean excludes count-pruned players and candidate itself");
  ok &= expect(
      player_size_exclusions(boxes({100, 100, 100, 201}), 0, true, 100) ==
          std::vector<size_t>{3},
      "Strictly more than twice the other players is oversized");
  ok &= expect(
      player_size_exclusions(boxes({100, 100, 100, 200}), 0, true, 100).empty(),
      "Exactly twice the mean is retained");
  ok &= expect(
      player_size_exclusions(players, 0, true, 100) == std::vector<size_t>{5},
      "Threshold decisions use a single snapshot, without cascading");
  ok &= expect(
      player_size_exclusions(boxes({0, 0, 0, 100}), 0, true, 100) ==
          std::vector<size_t>{3},
      "Zero peer area avoids division by zero");
  for (size_t size = 0; size <= 3; ++size)
    ok &= expect(
        player_size_exclusions(
            std::vector<hm::BBox>(players.begin(), players.begin() + size),
            5,
            true,
            0)
            .empty(),
        "Sparse tracks are never pruned");
  for (double percent :
       {-1.0,
        std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::quiet_NaN()}) {
    bool rejected = false;
    try {
      player_size_exclusions(players, 1, false, percent);
    } catch (const std::invalid_argument&) {
      rejected = true;
    }
    ok &= expect(rejected, "Invalid percentage is rejected even when disabled");
  }
  bool rejected = false;
  try {
    player_size_exclusions(players, -1, false, 100);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  ok &= expect(rejected, "Negative counts are rejected");

  hm::play_tracker::PlayTrackerConfig config;
  hm::play_tracker::AllLivingBoxConfig live;
  live.arena_box = hm::BBox(0, 0, 2000, 1000);
  live.max_width = 2000;
  live.max_height = 1000;
  config.living_boxes = {live};
  config.ignore_outlier_players = false;
  config.ignore_largest_bbox_count = 2;
  hm::play_tracker::PlayTracker tracker(*live.arena_box, config);
  auto tracked = boxes({100, 100, 100, 300, 1000});
  std::vector<size_t> ids{10, 11, 12, 13, 14};
  auto result = tracker.forward(ids, tracked);
  ok &= expect(
      result.size_ignored_tracking_boxes.size() == 2 &&
          result.size_ignored_tracking_boxes[0].tracking_id == 14 &&
          result.size_ignored_tracking_boxes[1].tracking_id == 13,
      "Results preserve ignored tracking identities");
  ok &= expect(
      result.final_cluster_box.right == 100,
      "Fallback for three players excludes oversized players from play framing");
  tracker.set_player_size_filter(0, false, 100);
  ok &= expect(
      tracker.forward(ids, tracked).size_ignored_tracking_boxes.empty(),
      "Live disable takes effect without reconstructing tracker");
  tracker.set_player_size_filter(1, true, 100);
  ok &= expect(
      tracker.forward(ids, tracked).size_ignored_tracking_boxes.size() == 2,
      "Live count and percentage updates apply together");
  return ok ? 0 : 1;
}
