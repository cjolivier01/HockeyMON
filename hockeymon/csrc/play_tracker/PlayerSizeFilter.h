#pragma once

#include "hockeymon/csrc/play_tracker/BoxUtils.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace hm::play_tracker {

inline void validate_player_size_filter(
    int largest_count,
    double oversized_percent) {
  if (largest_count < 0)
    throw std::invalid_argument(
        "ignore_largest_bbox_count must be nonnegative");
  if (!std::isfinite(oversized_percent) || oversized_percent < 0)
    throw std::invalid_argument(
        "oversized_bbox_percent must be finite and nonnegative");
}

// Remove the N largest areas first, then compare each remaining area to the
// mean of the OTHER remaining players. Percentage decisions use one snapshot,
// not a cascading average. Preserve at least three players, as the legacy
// single-largest filter did. Equal areas retain their input order.
inline std::vector<size_t> player_size_exclusions(
    const std::vector<BBox>& boxes,
    int largest_count,
    bool ignore_oversized,
    double oversized_percent) {
  validate_player_size_filter(largest_count, oversized_percent);
  constexpr size_t kMinimumPlayers = 3;
  if (boxes.size() <= kMinimumPlayers || (!largest_count && !ignore_oversized))
    return {};
  std::vector<size_t> order(boxes.size());
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
    return boxes[a].area() > boxes[b].area();
  });
  const size_t max_removed = boxes.size() - kMinimumPlayers;
  const size_t count =
      std::min(static_cast<size_t>(largest_count), max_removed);
  std::vector<size_t> removed(order.begin(), order.begin() + count);
  if (!ignore_oversized || count == max_removed)
    return removed;
  double total_area = 0;
  for (size_t i = count; i < order.size(); ++i)
    total_area += boxes[order[i]].area();
  const double other_count = static_cast<double>(boxes.size() - count - 1);
  const double ratio = 1.0 + oversized_percent / 100.0;
  for (size_t i = count; i < order.size() && removed.size() < max_removed;
       ++i) {
    const double area = boxes[order[i]].area();
    const double other_average = (total_area - area) / other_count;
    if (area > ratio * other_average)
      removed.push_back(order[i]);
  }
  return removed;
}

} // namespace hm::play_tracker
