#pragma once

#include "hockeymon/csrc/play_tracker/BoxUtils.h"

#include <cassert>
#include <deque>

namespace hm {
namespace play_tracker {

class PlayerSTrack {
 public:
  PlayerSTrack(
      size_t max_positions,
      size_t max_velocity_positions,
      size_t frame_step = 1);

  void add_position(
      size_t frame_id,
      const Point& position,
      const Point& center);

  size_t age(size_t frame_id) const;

  void reset();

  PointDiff velocity() const;

  bool empty() const {
    return !positions_.empty();
  }

  const Point& position() const {
    return positions_.back();
  }

  const Point& center() const {
    return centers_.back();
  }

 private:
  const size_t max_positions_;
  const size_t max_velocity_positions_;
  const size_t frame_step_;
  std::deque<Point> positions_;
  std::deque<Point> centers_;
  std::deque<PointDiff> position_diffs_;
  PointDiff sum_position_diffs_{0.0f, 0.0f};
  size_t last_frame_id_{0};
};

} // namespace play_tracker
} // namespace hm
