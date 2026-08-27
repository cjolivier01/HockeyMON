
#include "hockeymon/csrc/play_tracker/PlayDetector.h"

#include <cassert>
#include <iterator>
#include <limits>

namespace hm {
namespace play_tracker {

namespace {

using Velocity = PointDiff;

constexpr size_t kMinPlayersForBreakawayDetection = 4;

// Helper function to compute the magnitude of a velocity
template <typename MapType>
class ValueIterator {
  using IteratorType = typename MapType::iterator;

 public:
  using iterator_category = std::input_iterator_tag; // Iterator category
  using value_type = typename MapType::mapped_type; // Type of the values
  using difference_type = std::ptrdiff_t; // Difference type
  using pointer = value_type*; // Pointer type
  using reference = value_type&; // Reference type

  explicit ValueIterator(IteratorType it) : mapIt(it) {}

  // Dereference operators
  reference operator*() {
    return mapIt->second;
  }
  pointer operator->() {
    return &(mapIt->second);
  }

  // Pre-increment operator
  ValueIterator& operator++() {
    ++mapIt;
    return *this;
  }

  // Post-increment operator
  ValueIterator operator++(int) {
    ValueIterator temp = *this;
    ++mapIt;
    return temp;
  }

  // Equality operators
  bool operator==(const ValueIterator& other) const {
    return mapIt == other.mapIt;
  }
  bool operator!=(const ValueIterator& other) const {
    return mapIt != other.mapIt;
  }

 private:
  IteratorType mapIt;
};

// Helper function to create value iterators
template <typename MapType>
auto make_value_iterator(MapType& map) {
  return std::pair<ValueIterator<MapType>, ValueIterator<MapType>>(
      ValueIterator<MapType>(map.begin()), ValueIterator<MapType>(map.end()));
}

} // namespace

PlayDetector::PlayDetector(
    const PlayDetectorConfig& config,
    IBreakawayAdjuster* adjuster)
    : config_(config), adjuster_(adjuster), overshoot_stop_delay_override_(std::nullopt), overshoot_scale_override_(std::nullopt) {}

void PlayDetector::set_overshoot_stop_delay_count(int v) {
  overshoot_stop_delay_override_ = v;
}

void PlayDetector::set_overshoot_scale_speed_ratio(float v) {
  overshoot_scale_override_ = v;
}

void PlayDetector::reset() {
  tracks_.clear();
}

//
// Update the tracks
//
PlayDetector::TrackStateInfo PlayDetector::update_tracks(
    size_t frame_id,
    std::vector<size_t>& tracking_ids,
    std::vector<BBox>& tracking_boxes,
    const std::set<size_t>& disregard_tracking_ids) {
  TrackStateInfo ts_info;
  ts_info.track_velocity.reserve(tracking_ids.size());

  for (size_t i = 0, n = tracking_ids.size(); i < n; ++i) {
    const size_t track_id = tracking_ids[i];
    auto found_track = tracks_.find(track_id);
    if (found_track == tracks_.end()) {
      found_track = tracks_
                        .emplace(
                            track_id,
                            PlayerSTrack(
                                config_.max_positions,
                                config_.max_velocity_positions,
                                config_.frame_step))
                        .first;
    }
    // We don't record velocities of "disregarded" items (such as a ref),
    // but it's ok to keep trackl of them in our overall tracks, since
    // they may not always be "disregarded".
    PlayerSTrack& track = found_track->second;
    const BBox& tracking_box = tracking_boxes.at(i);
    track.add_position(
        frame_id, tracking_box.anchor_point(), tracking_box.center());
    auto track_velocity = track.velocity() * config_.fps_speed_scale;
    if (!disregard_tracking_ids.count(track_id)) {
      ts_info.track_velocity.emplace(track_id, track_velocity);
      ts_info.cumulative_velocity =
          ts_info.cumulative_velocity + track_velocity;
    }
  }

  //
  // Dump any lost, stale tracks
  //
  std::vector<TrackingMap::iterator> remove_tracks;
  remove_tracks.reserve(tracks_.size());
  for (auto track_iter = tracks_.begin(), e_iter = tracks_.end();
       track_iter != e_iter;
       ++track_iter) {
    if (track_iter->second.age(frame_id) >= config_.max_positions) {
      // Lost track is now stale
      remove_tracks.emplace_back(track_iter);
    }
  }
  std::for_each(remove_tracks.begin(), remove_tracks.end(), [this](auto& iter) {
    tracks_.erase(iter);
  });

  return ts_info;
}

PlayDetectorResults PlayDetector::forward(
    size_t frame_id,
    const BBox& current_target_bbox,
    const Point& current_roi_center,
    std::vector<size_t>& tracking_ids,
    std::vector<BBox>& tracking_boxes,
    const std::set<size_t>& disregard_tracking_ids) {
  PlayDetectorResults result;
  assert(tracking_ids.size() == tracking_boxes.size());

  //
  // Update the tracks
  //
  TrackStateInfo track_state_info = update_tracks(
      frame_id, tracking_ids, tracking_boxes, disregard_tracking_ids);

  //
  // Ok, now analyze what's going on...
  //
  auto br_result = detect_breakaway(
      current_target_bbox, current_roi_center, track_state_info);
  if (br_result.has_value()) {
    result.breakaway_target_bbox = std::get<0>(*br_result);
    result.breakaway_edge_center = std::get<1>(*br_result);
  }
  return result;
}

std::optional<std::tuple<BBox, Point>> PlayDetector::detect_breakaway(
    const BBox& current_target_bbox,
    const Point& current_roi_center,
    const TrackStateInfo& track_state_info,
    bool average_boxes) {
  if (track_state_info.track_velocity.size() <
      kMinPlayersForBreakawayDetection) {
    return std::nullopt;
  }
  std::optional<GroupMovementInfo> group_info =
      get_group_velocity(track_state_info.track_velocity);
  if (!group_info.has_value()) {
    return std::nullopt;
  }
  BBox new_dest_bbox;
  const std::optional<Point> edge_center = group_info->group_velocity.dx < 0
      ? group_info->leftmost_center
      : group_info->rightmost_center;
  assert(edge_center.has_value());
  if (average_boxes) {
    Point cb_center = current_target_bbox.center();
    Point avg_center{
        .x = float((double(cb_center.x) + edge_center->x) / 2.0f),
        .y = float((double(cb_center.y) + edge_center->y) / 2.0f),
    };
    new_dest_bbox = current_target_bbox.at_center(avg_center);
  } else {
    new_dest_bbox = current_target_bbox.at_center(*edge_center);
  }

  if (adjuster_) {
    // Maybe adjust some speeds
    const bool should_adjust_speed = (group_info->group_velocity.dx > 0 &&
                                      current_roi_center.x < edge_center->x) ||
        (group_info->group_velocity.dx < 0 &&
         current_roi_center.x > edge_center->x);
    if (should_adjust_speed) {
      adjuster_->adjust_speed(
          /*accel_x=*/group_info->group_velocity.dx *
              config_.group_velocity_speed_ratio,
          /*accel_y=*/std::nullopt,
          /*scale_constraints=*/config_.scale_speed_constraints,
          /*nonstop_delay=*/config_.nonstop_delay_count);
    } else {
      // Overshoot case: either multiplicative damping or begin a stop-delay
      const int osd = overshoot_stop_delay_override_.has_value()
          ? *overshoot_stop_delay_override_
          : config_.overshoot_stop_delay_count;
      const float osc = overshoot_scale_override_.has_value()
          ? *overshoot_scale_override_
          : config_.overshoot_scale_speed_ratio;
      if (osd > 0) {
        adjuster_->begin_stop_delay(/*delay_x=*/osd, /*delay_y=*/std::nullopt);
      } else {
        // Cut the speed quickly due to overshoot
        adjuster_->scale_speed(
            /*ratio_x=*/osc,
            /*ratio_y=*/std::nullopt,
            /*clamp_to_max=*/false);
      }
    }
  }
  return std::make_tuple(new_dest_bbox, *edge_center);
}

std::optional<PlayDetector::GroupMovementInfo> PlayDetector::get_group_velocity(
    const std::unordered_map<size_t, Velocity>& track_velocities) {
  GroupMovementInfo group_info;
  std::optional<Point> leftmost_fast;
  std::optional<Point> rightmost_fast;
  size_t pos_count = 0, neg_count = 0;
  Velocity pos_sum{0.0, 0.0}, neg_sum{0.0, 0.0};
  for (const auto& track_velocity_item : track_velocities) {
    const Velocity& velocity = track_velocity_item.second;
    const float x_speed = velocity.dx;
    if (x_speed >= config_.min_considered_group_velocity) {
      ++pos_count;
      pos_sum = pos_sum + velocity;
      const size_t track_id = track_velocity_item.first;
      const Point cc = tracks_.at(track_id).center();
      if (!rightmost_fast.has_value() || cc.x > rightmost_fast->x) {
        rightmost_fast = cc;
      }
    } else if (x_speed <= -config_.min_considered_group_velocity) {
      ++neg_count;
      neg_sum = neg_sum + velocity;
      const size_t track_id = track_velocity_item.first;
      const Point cc = tracks_.at(track_id).center();
      if (!leftmost_fast.has_value() || cc.x < leftmost_fast->x) {
        leftmost_fast = cc;
      }
    }
  }
  const size_t total_ids = track_velocities.size();
  if (float(pos_count) / total_ids > config_.group_ratio_threshold) {
    // Make sure the opposite isn't also true, or else we need to chose the max
    assert(!(float(neg_count) / total_ids > config_.group_ratio_threshold));
    group_info.leftmost_center = leftmost_fast;
    assert(rightmost_fast.has_value());
    group_info.rightmost_center = rightmost_fast;
    Velocity average_speed = pos_sum / pos_count;
    group_info.group_velocity = average_speed;
    return group_info;
  } else if (float(neg_count) / total_ids > config_.group_ratio_threshold) {
    // Make sure the opposite isn't also true, or else we need to chose the max
    assert(!(float(pos_count) / total_ids > config_.group_ratio_threshold));
    assert(leftmost_fast.has_value());
    group_info.leftmost_center = leftmost_fast;
    Velocity average_speed = neg_sum / neg_count;
    group_info.group_velocity = average_speed;
    return group_info;
  }

  return std::nullopt;
}

} // namespace play_tracker
} // namespace hm
