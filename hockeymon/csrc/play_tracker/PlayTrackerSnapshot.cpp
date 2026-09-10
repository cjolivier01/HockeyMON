#include "hockeymon/csrc/play_tracker/PlayTrackerSnapshot.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <iomanip>
#include <limits>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <unordered_set>

namespace hm {
namespace play_tracker {
namespace {

constexpr size_t kMaxSnapshotBytes = 16 * 1024 * 1024;
constexpr size_t kMaxBoxes = 64;
constexpr size_t kMaxPlayers = 4096;
constexpr size_t kMaxHistory = 4096;
constexpr size_t kMaxContainer = 4096;
constexpr size_t kMaxStringBytes = 1024;
constexpr size_t kMaxValues = 1024 * 1024;

void require(bool condition, const std::string& message) {
  if (!condition) {
    throw std::invalid_argument("invalid play-tracker snapshot: " + message);
  }
}

// Shared field lists keep validation and both archive directions in agreement.
// These names and their order are part of the versioned persistence format.
// A policy/state change must update the schema or implementation identifier.

template <typename Archive>
void fields(Archive& archive, BBox& value) {
  archive.field("left", value.left);
  archive.field("top", value.top);
  archive.field("right", value.right);
  archive.field("bottom", value.bottom);
}

template <typename Archive>
void fields(Archive& archive, Point& value) {
  archive.field("x", value.x);
  archive.field("y", value.y);
}

template <typename Archive>
void fields(Archive& archive, PointDiff& value) {
  archive.field("dx", value.dx);
  archive.field("dy", value.dy);
}

template <typename Archive>
void fields(Archive& archive, ResizingConfig& value) {
  archive.field("resizing_enabled", value.resizing_enabled);
  archive.field("max_speed_w", value.max_speed_w);
  archive.field("max_speed_h", value.max_speed_h);
  archive.field("max_accel_w", value.max_accel_w);
  archive.field("max_accel_h", value.max_accel_h);
  archive.field("min_width", value.min_width);
  archive.field("min_height", value.min_height);
  archive.field("max_width", value.max_width);
  archive.field("max_height", value.max_height);
  archive.field(
      "stop_resizing_on_dir_change", value.stop_resizing_on_dir_change);
  archive.field(
      "resizing_stop_on_dir_change_delay",
      value.resizing_stop_on_dir_change_delay);
  archive.field(
      "resizing_cancel_stop_on_opposite_dir",
      value.resizing_cancel_stop_on_opposite_dir);
  archive.field(
      "resizing_stop_cancel_hysteresis_frames",
      value.resizing_stop_cancel_hysteresis_frames);
  archive.field(
      "resizing_stop_delay_cooldown_frames",
      value.resizing_stop_delay_cooldown_frames);
  archive.field(
      "resizing_time_to_dest_speed_limit_frames",
      value.resizing_time_to_dest_speed_limit_frames);
  archive.field(
      "resizing_time_to_dest_stop_speed_threshold",
      value.resizing_time_to_dest_stop_speed_threshold);
  archive.field("sticky_sizing", value.sticky_sizing);
  archive.field("size_ratio_thresh_grow_dw", value.size_ratio_thresh_grow_dw);
  archive.field("size_ratio_thresh_grow_dh", value.size_ratio_thresh_grow_dh);
  archive.field(
      "size_ratio_thresh_shrink_dw", value.size_ratio_thresh_shrink_dw);
  archive.field(
      "size_ratio_thresh_shrink_dh", value.size_ratio_thresh_shrink_dh);
}

template <typename Archive>
void fields(Archive& archive, TranslatingBoxConfig& value) {
  archive.field("translation_enabled", value.translation_enabled);
  archive.field("max_speed_x", value.max_speed_x);
  archive.field("max_speed_y", value.max_speed_y);
  archive.field("max_accel_x", value.max_accel_x);
  archive.field("max_accel_y", value.max_accel_y);
  archive.field(
      "stop_translation_on_dir_change", value.stop_translation_on_dir_change);
  archive.field(
      "stop_translation_on_dir_change_delay",
      value.stop_translation_on_dir_change_delay);
  archive.field(
      "cancel_stop_on_opposite_dir", value.cancel_stop_on_opposite_dir);
  archive.field(
      "dynamic_acceleration_scaling", value.dynamic_acceleration_scaling);
  archive.field("arena_angle_from_vertical", value.arena_angle_from_vertical);
  archive.field("arena_box", value.arena_box);
  archive.field("sticky_translation", value.sticky_translation);
  archive.field(
      "sticky_size_ratio_to_frame_width",
      value.sticky_size_ratio_to_frame_width);
  archive.field(
      "sticky_translation_gaussian_mult",
      value.sticky_translation_gaussian_mult);
  archive.field(
      "unsticky_translation_size_ratio", value.unsticky_translation_size_ratio);
  archive.field(
      "post_nonstop_stop_delay_count", value.post_nonstop_stop_delay_count);
  archive.field(
      "cancel_stop_hysteresis_frames", value.cancel_stop_hysteresis_frames);
  archive.field("stop_delay_cooldown_frames", value.stop_delay_cooldown_frames);
  archive.field(
      "time_to_dest_speed_limit_frames", value.time_to_dest_speed_limit_frames);
  archive.field(
      "time_to_dest_stop_speed_threshold",
      value.time_to_dest_stop_speed_threshold);
}

template <typename Archive>
void fields(Archive& archive, LivingBoxConfig& value) {
  archive.field("scale_dest_width", value.scale_dest_width);
  archive.field("scale_dest_height", value.scale_dest_height);
  archive.field("fixed_aspect_ratio", value.fixed_aspect_ratio);
  archive.field("clamp_scaled_input_box", value.clamp_scaled_input_box);
}

template <typename Archive>
void fields(Archive& archive, AllLivingBoxConfig& value) {
  fields(archive, static_cast<ResizingConfig&>(value));
  fields(archive, static_cast<TranslatingBoxConfig&>(value));
  fields(archive, static_cast<LivingBoxConfig&>(value));
  archive.field("name", value.name);
}

template <typename Archive>
void fields(Archive& archive, PlayDetectorConfig& value) {
  archive.field("max_positions", value.max_positions);
  archive.field("max_velocity_positions", value.max_velocity_positions);
  archive.field("frame_step", value.frame_step);
  archive.field("fps_speed_scale", value.fps_speed_scale);
  archive.field(
      "min_considered_group_velocity", value.min_considered_group_velocity);
  archive.field("group_ratio_threshold", value.group_ratio_threshold);
  archive.field("group_velocity_speed_ratio", value.group_velocity_speed_ratio);
  archive.field("scale_speed_constraints", value.scale_speed_constraints);
  archive.field("nonstop_delay_count", value.nonstop_delay_count);
  archive.field(
      "overshoot_scale_speed_ratio", value.overshoot_scale_speed_ratio);
  archive.field("overshoot_stop_delay_count", value.overshoot_stop_delay_count);
}

template <typename Archive>
void fields(Archive& archive, PlayTrackerConfig& value) {
  archive.field("no_wide_start", value.no_wide_start);
  archive.field("min_tracked_players", value.min_tracked_players);
  archive.field("living_boxes", value.living_boxes);
  archive.field("max_lost_track_age", value.max_lost_track_age);
  archive.field("ignore_largest_bbox", value.ignore_largest_bbox);
  archive.field("ignore_largest_bbox_count", value.ignore_largest_bbox_count);
  archive.field("ignore_oversized_bboxes", value.ignore_oversized_bboxes);
  archive.field("oversized_bbox_percent", value.oversized_bbox_percent);
  archive.field(
      "ignore_left_and_right_extremes", value.ignore_left_and_right_extremes);
  archive.field("ignore_outlier_players", value.ignore_outlier_players);
  archive.field(
      "ignore_outlier_players_dist_ratio",
      value.ignore_outlier_players_dist_ratio);
  archive.field("play_detector", value.play_detector);
}

template <typename Archive>
void fields(Archive& archive, PlayTrackerState& value) {
  archive.field("tick_count_", value.tick_count_);
  archive.field("tracked_player_count", value.tracked_player_count);
}

template <typename Archive>
void fields(Archive& archive, LivingState& value) {
  archive.field("was_size_constrained", value.was_size_constrained);
}

template <typename Archive>
void fields(Archive& archive, TranslationState& value) {
  archive.field("current_speed_x", value.current_speed_x);
  archive.field("current_speed_y", value.current_speed_y);
  archive.field("translation_is_frozen", value.translation_is_frozen);
  archive.field(
      "last_arena_edge_center_position_scale",
      value.last_arena_edge_center_position_scale);
  archive.field("nonstop_delay", value.nonstop_delay);
  archive.field("nonstop_delay_counter", value.nonstop_delay_counter);
  archive.field("stop_delay_x", value.stop_delay_x);
  archive.field("stop_delay_x_counter", value.stop_delay_x_counter);
  archive.field("stop_decel_x", value.stop_decel_x);
  archive.field("stop_trigger_dir_x", value.stop_trigger_dir_x);
  archive.field("cancel_opp_x_count", value.cancel_opp_x_count);
  archive.field("cooldown_x_counter", value.cooldown_x_counter);
  archive.field("stop_delay_y", value.stop_delay_y);
  archive.field("stop_delay_y_counter", value.stop_delay_y_counter);
  archive.field("stop_decel_y", value.stop_decel_y);
  archive.field("stop_trigger_dir_y", value.stop_trigger_dir_y);
  archive.field("cancel_opp_y_count", value.cancel_opp_y_count);
  archive.field("cooldown_y_counter", value.cooldown_y_counter);
  archive.field("canceled_stop_x", value.canceled_stop_x);
  archive.field("canceled_stop_y", value.canceled_stop_y);
  archive.field("filtered_target_center", value.filtered_target_center);
}

template <typename Archive>
void fields(Archive& archive, ResizingState& value) {
  archive.field("size_is_frozen", value.size_is_frozen);
  archive.field("current_speed_w", value.current_speed_w);
  archive.field("current_speed_h", value.current_speed_h);
  archive.field("stop_delay_w", value.stop_delay_w);
  archive.field("stop_delay_w_counter", value.stop_delay_w_counter);
  archive.field("stop_decel_w", value.stop_decel_w);
  archive.field("stop_trigger_dir_w", value.stop_trigger_dir_w);
  archive.field("cancel_opp_w_count", value.cancel_opp_w_count);
  archive.field("deadband_stop_w", value.deadband_stop_w);
  archive.field("cooldown_w_counter", value.cooldown_w_counter);
  archive.field("stop_delay_h", value.stop_delay_h);
  archive.field("stop_delay_h_counter", value.stop_delay_h_counter);
  archive.field("stop_decel_h", value.stop_decel_h);
  archive.field("stop_trigger_dir_h", value.stop_trigger_dir_h);
  archive.field("cancel_opp_h_count", value.cancel_opp_h_count);
  archive.field("deadband_stop_h", value.deadband_stop_h);
  archive.field("cooldown_h_counter", value.cooldown_h_counter);
  archive.field("canceled_stop_w", value.canceled_stop_w);
  archive.field("canceled_stop_h", value.canceled_stop_h);
}

template <typename Archive>
void fields(Archive& archive, LivingBoxSnapshot& value) {
  archive.field("config", value.config);
  archive.field("bbox", value.bbox);
  archive.field("living", value.living);
  archive.field("translation", value.translation);
  archive.field("resizing", value.resizing);
  archive.field("forward_counter", value.forward_counter);
}

template <typename Archive>
void fields(Archive& archive, PlayerTrackSnapshot& value) {
  archive.field("tracking_id", value.tracking_id);
  archive.field("positions", value.positions);
  archive.field("centers", value.centers);
  archive.field("position_diffs", value.position_diffs);
  archive.field("sum_position_diffs", value.sum_position_diffs);
  archive.field("last_frame_id", value.last_frame_id);
}

template <typename Archive>
void fields(Archive& archive, PlayDetectorSnapshot& value) {
  archive.field(
      "overshoot_stop_delay_override", value.overshoot_stop_delay_override);
  archive.field("overshoot_scale_override", value.overshoot_scale_override);
  archive.field("tracks", value.tracks);
}

template <typename Archive>
void fields(Archive& archive, PlayTrackerSnapshot& value) {
  archive.field("schema_version", value.schema_version);
  archive.field("implementation", value.implementation);
  archive.field("config", value.config);
  archive.field("state", value.state);
  archive.field("living_boxes", value.living_boxes);
  archive.field("detector", value.detector);
}

class ValueBudget {
 protected:
  void count() {
    require(++values_ <= kMaxValues, "too many values");
  }

 private:
  size_t values_{0};
};

class Validator : private ValueBudget {
 public:
  template <typename T>
  void field(const char*, T& value) {
    check(value);
  }

  template <typename T>
  void check(T& value) {
    count();
    if constexpr (std::is_floating_point_v<T>) {
      require(std::isfinite(value), "non-finite number");
    } else if constexpr (!std::is_arithmetic_v<T>) {
      fields(*this, value);
    }
  }

  void check(std::string& value) {
    count();
    require(value.size() <= kMaxStringBytes, "string too long");
  }

  template <typename T>
  void check(std::optional<T>& value) {
    count();
    if (value) {
      check(*value);
    }
  }

  template <typename T>
  void check(std::vector<T>& values) {
    sequence(values);
  }

  template <typename T>
  void check(std::deque<T>& values) {
    sequence(values);
  }

 private:
  template <typename T>
  void sequence(T& values) {
    count();
    require(values.size() <= kMaxContainer, "container too large");
    for (auto& value : values) {
      check(value);
    }
  }
};

class Writer {
 public:
  Writer() {
    out_.imbue(std::locale::classic());
    out_ << std::setprecision(std::numeric_limits<double>::max_digits10);
  }

  template <typename T>
  void field(const char* name, T& value) {
    out_ << name << ' ';
    write(value);
    out_ << '\n';
  }

  template <typename T>
  void write(T& value) {
    if constexpr (std::is_arithmetic_v<T>) {
      out_ << value << ' ';
    } else {
      out_ << "{\n";
      fields(*this, value);
      out_ << "} ";
    }
  }

  void write(std::string& value) {
    out_ << std::quoted(value) << ' ';
  }

  template <typename T>
  void write(std::optional<T>& value) {
    out_ << (value ? 1 : 0) << ' ';
    if (value) {
      write(*value);
    }
  }

  template <typename T>
  void write(std::vector<T>& values) {
    sequence(values);
  }

  template <typename T>
  void write(std::deque<T>& values) {
    sequence(values);
  }

  std::string contents() const {
    auto result = out_.str();
    require(result.size() <= kMaxSnapshotBytes, "payload too large");
    return result;
  }

 private:
  template <typename T>
  void sequence(T& values) {
    out_ << values.size() << ' ';
    for (auto& value : values) {
      write(value);
    }
  }

  std::ostringstream out_;
};

class Reader : private ValueBudget {
 public:
  explicit Reader(const std::string& contents) : in_(contents) {
    in_.imbue(std::locale::classic());
  }

  template <typename T>
  void field(const char* name, T& value) {
    expect(name);
    read(value);
  }

  template <typename T>
  void read(T& value) {
    count();
    if constexpr (std::is_same_v<T, bool>) {
      auto token = next();
      require(token == "0" || token == "1", "invalid boolean");
      value = token == "1";
    } else if constexpr (std::is_integral_v<T>) {
      auto token = next();
      const auto parsed =
          std::from_chars(token.data(), token.data() + token.size(), value);
      require(
          parsed.ec == std::errc{} && parsed.ptr == token.data() + token.size(),
          "invalid integer");
    } else if constexpr (std::is_floating_point_v<T>) {
      auto token = next();
      std::istringstream number(token);
      number.imbue(std::locale::classic());
      number >> value;
      require(
          !number.fail() && number.eof() && std::isfinite(value),
          "invalid floating-point number");
    } else {
      expect("{");
      fields(*this, value);
      expect("}");
    }
  }

  void read(std::string& value) {
    count();
    in_ >> std::ws;
    require(in_.peek() == '"', "expected quoted string");
    in_ >> std::quoted(value);
    require(
        !in_.fail() && value.size() <= kMaxStringBytes,
        "invalid or oversized string");
  }

  template <typename T>
  void read(std::optional<T>& value) {
    bool present = false;
    read(present);
    if (present) {
      T item{};
      read(item);
      value = std::move(item);
    } else {
      value.reset();
    }
  }

  template <typename T>
  void read(std::vector<T>& values) {
    sequence(values);
  }

  template <typename T>
  void read(std::deque<T>& values) {
    sequence(values);
  }

  void finish() {
    in_ >> std::ws;
    require(in_.eof(), "trailing content");
  }

 private:
  std::string next() {
    std::string result;
    require(static_cast<bool>(in_ >> result), "truncated payload");
    require(result.size() <= kMaxStringBytes, "oversized token");
    return result;
  }

  void expect(const std::string& expected) {
    require(next() == expected, "expected field or delimiter " + expected);
  }

  template <typename T>
  void sequence(T& values) {
    size_t size = 0;
    read(size);
    require(size <= kMaxContainer, "container too large");
    values.clear();
    // Parse before appending; a malicious count cannot allocate all entries
    // before the input is checked and the cumulative value budget is applied.
    for (size_t i = 0; i < size; ++i) {
      typename T::value_type value{};
      read(value);
      values.push_back(std::move(value));
    }
  }

  std::istringstream in_;
};

bool positive_box(const BBox& box) {
  return box.right > box.left && box.bottom > box.top;
}

bool safe_counter(IntValue value) {
  return value >= 0 && value < std::numeric_limits<IntValue>::max();
}

void validate_box_config(const AllLivingBoxConfig& c) {
  require(c.arena_box && positive_box(*c.arena_box), "missing/empty arena");
  require(
      c.max_width >= 0 && c.max_height >= 0 && c.min_width >= 0 &&
          c.min_height >= 0 && c.max_width <= c.arena_box->width() &&
          c.max_height <= c.arena_box->height(),
      "invalid box size constraints");
  require(
      (!c.max_width || c.min_width <= c.max_width) &&
          (!c.max_height || c.min_height <= c.max_height),
      "minimum box size exceeds maximum");
  require(
      c.max_speed_x >= 0 && c.max_speed_y >= 0 && c.max_speed_w >= 0 &&
          c.max_speed_h >= 0 && c.max_accel_x >= 0 && c.max_accel_y >= 0 &&
          c.max_accel_w >= 0 && c.max_accel_h >= 0,
      "negative motion constraint");
  require(
      c.scale_dest_width > 0 && c.scale_dest_height > 0 &&
          (!c.fixed_aspect_ratio || *c.fixed_aspect_ratio > 0),
      "invalid box scale/aspect ratio");
  require(
      !c.sticky_translation || c.sticky_size_ratio_to_frame_width > 0,
      "invalid sticky translation divisor");
  require(
      c.dynamic_acceleration_scaling == 0 || c.sticky_translation,
      "dynamic acceleration requires sticky translation");
  require(
      safe_counter(c.stop_translation_on_dir_change_delay) &&
          safe_counter(c.cancel_stop_hysteresis_frames) &&
          safe_counter(c.stop_delay_cooldown_frames) &&
          safe_counter(c.post_nonstop_stop_delay_count) &&
          safe_counter(c.time_to_dest_speed_limit_frames) &&
          safe_counter(c.resizing_stop_on_dir_change_delay) &&
          safe_counter(c.resizing_stop_cancel_hysteresis_frames) &&
          safe_counter(c.resizing_stop_delay_cooldown_frames) &&
          safe_counter(c.resizing_time_to_dest_speed_limit_frames),
      "invalid braking duration");
}

void validate_delay(const std::optional<IntValue>& duration, IntValue counter) {
  require(
      safe_counter(duration.value_or(0)) && safe_counter(counter) &&
          counter <= duration.value_or(0),
      "invalid active delay or elapsed counter");
}

void validate_translation(const TranslationState& s) {
  validate_delay(s.nonstop_delay, s.nonstop_delay_counter);
  validate_delay(s.stop_delay_x, s.stop_delay_x_counter);
  validate_delay(s.stop_delay_y, s.stop_delay_y_counter);
  require(
      safe_counter(s.cancel_opp_x_count) &&
          safe_counter(s.cooldown_x_counter) &&
          safe_counter(s.cancel_opp_y_count) &&
          safe_counter(s.cooldown_y_counter),
      "invalid translation counter");
}

void validate_resizing(const ResizingState& s) {
  validate_delay(s.stop_delay_w, s.stop_delay_w_counter);
  validate_delay(s.stop_delay_h, s.stop_delay_h_counter);
  require(
      safe_counter(s.cancel_opp_w_count) &&
          safe_counter(s.cooldown_w_counter) &&
          safe_counter(s.cancel_opp_h_count) &&
          safe_counter(s.cooldown_h_counter),
      "invalid resizing counter");
}

} // namespace

PlayTrackerSnapshot PlayTracker::snapshot() const {
  PlayTrackerSnapshot result;
  result.config = config_;
  result.state = state_;
  for (const auto& box : living_boxes_) {
    const auto* living = dynamic_cast<const LivingBox*>(box.get());
    require(living != nullptr, "unsupported living-box implementation");
    LivingBoxSnapshot item;
    static_cast<LivingBoxConfig&>(item.config) = living->config_;
    static_cast<TranslatingBoxConfig&>(item.config) =
        living->TranslatingBox::config_;
    static_cast<ResizingConfig&>(item.config) = living->ResizingBox::config_;
    item.config.name = config_.living_boxes[result.living_boxes.size()].name;
    item.bbox = living->bounding_box();
    item.living = living->state_;
    item.translation = living->TranslatingBox::state_;
    item.resizing = living->ResizingBox::state_;
    item.forward_counter = living->forward_counter_;
    result.living_boxes.push_back(std::move(item));
  }
  result.detector.overshoot_stop_delay_override =
      play_detector_.overshoot_stop_delay_override_;
  result.detector.overshoot_scale_override =
      play_detector_.overshoot_scale_override_;
  for (const auto& [id, track] : play_detector_.tracks_) {
    PlayerTrackSnapshot item;
    item.tracking_id = id;
    item.positions = track.positions_;
    item.centers = track.centers_;
    item.position_diffs = track.position_diffs_;
    item.sum_position_diffs = track.sum_position_diffs_;
    item.last_frame_id = track.last_frame_id_;
    result.detector.tracks.push_back(std::move(item));
  }
  std::sort(
      result.detector.tracks.begin(),
      result.detector.tracks.end(),
      [](const auto& a, const auto& b) {
        return a.tracking_id < b.tracking_id;
      });
  return result;
}

std::unique_ptr<PlayTracker> PlayTracker::from_snapshot(
    const PlayTrackerSnapshot& snapshot) {
  validate_snapshot(snapshot);
  // Construction binds PlayDetector::adjuster_ to this fresh object. Neither
  // state restoration nor a later caller can alias the original living boxes.
  auto tracker = std::make_unique<PlayTracker>(
      *snapshot.config.living_boxes.back().arena_box, snapshot.config);
  tracker->living_boxes_.clear();
  for (size_t i = 0; i < snapshot.living_boxes.size(); ++i) {
    const auto& item = snapshot.living_boxes[i];
    auto box = std::make_shared<LivingBox>(
        std::to_string(i + 1), *item.config.arena_box, item.config);
    // Effective size clamps and gaussian arena bounds are derived only from
    // this effective config by the constructor. set_bbox does not reset state.
    box->set_bbox(item.bbox);
    box->state_ = item.living;
    box->TranslatingBox::state_ = item.translation;
    box->ResizingBox::state_ = item.resizing;
    box->forward_counter_ = item.forward_counter;
    tracker->living_boxes_.push_back(std::move(box));
  }
  tracker->state_ = snapshot.state;
  tracker->play_detector_.overshoot_stop_delay_override_ =
      snapshot.detector.overshoot_stop_delay_override;
  tracker->play_detector_.overshoot_scale_override_ =
      snapshot.detector.overshoot_scale_override;
  const auto& c = snapshot.config.play_detector;
  for (const auto& item : snapshot.detector.tracks) {
    auto inserted = tracker->play_detector_.tracks_.emplace(
        item.tracking_id,
        PlayerSTrack(c.max_positions, c.max_velocity_positions, c.frame_step));
    auto& track = inserted.first->second;
    track.positions_ = item.positions;
    track.centers_ = item.centers;
    track.position_diffs_ = item.position_diffs;
    track.sum_position_diffs_ = item.sum_position_diffs;
    track.last_frame_id_ = item.last_frame_id;
  }
  return tracker;
}

void validate_snapshot(const PlayTrackerSnapshot& snapshot) {
  require(
      snapshot.schema_version == kPlayTrackerSnapshotSchema,
      "unsupported schema version");
  require(
      snapshot.implementation == kPlayTrackerSnapshotImplementation,
      "incompatible policy implementation");
  require(
      !snapshot.living_boxes.empty() &&
          snapshot.living_boxes.size() <= kMaxBoxes &&
          snapshot.config.living_boxes.size() == snapshot.living_boxes.size(),
      "invalid living-box topology");
  require(snapshot.detector.tracks.size() <= kMaxPlayers, "too many players");
  // The visitors are read-only for Validator. A bounded defensive copy avoids
  // exposing writable state to generic archive code or casting away constness.
  size_t history_values = 0;
  for (const auto& t : snapshot.detector.tracks) {
    require(
        t.positions.size() <= kMaxHistory && t.centers.size() <= kMaxHistory &&
            t.position_diffs.size() <= kMaxHistory,
        "player history too large");
    history_values +=
        t.positions.size() + t.centers.size() + t.position_diffs.size();
    require(history_values <= kMaxValues / 4, "total player history too large");
  }
  PlayTrackerSnapshot checked = snapshot;
  Validator validator;
  validator.check(checked);
  const auto& c = snapshot.config;
  require(
      c.ignore_largest_bbox_count >= 0 && c.oversized_bbox_percent > 0 &&
          c.min_tracked_players <= kMaxPlayers,
      "invalid player filter");
  const auto& detector = c.play_detector;
  require(
      detector.max_positions > 0 && detector.max_positions <= kMaxHistory &&
          detector.max_velocity_positions > 0 &&
          detector.max_velocity_positions <= detector.max_positions &&
          detector.frame_step > 0 && detector.frame_step <= kMaxHistory &&
          detector.overshoot_stop_delay_count >= 0,
      "invalid player-history configuration");
  require(
      detector.group_ratio_threshold >= 0.5f &&
          detector.group_ratio_threshold <= 1.0f &&
          detector.min_considered_group_velocity > 0 &&
          detector.fps_speed_scale > 0 &&
          detector.group_velocity_speed_ratio >= 0 &&
          detector.scale_speed_constraints >= 0 &&
          detector.overshoot_scale_speed_ratio >= 0 &&
          detector.nonstop_delay_count <
              static_cast<uint64_t>(std::numeric_limits<IntValue>::max()),
      "invalid detector policy thresholds or delay");
  require(
      snapshot.detector.overshoot_stop_delay_override.value_or(0) >= 0 &&
          snapshot.detector.overshoot_scale_override.value_or(0) >= 0,
      "invalid detector override");
  require(
      snapshot.state.tick_count_ < std::numeric_limits<size_t>::max() &&
          snapshot.state.tracked_player_count <= kMaxPlayers,
      "invalid tracker counter");
  for (size_t i = 0; i < snapshot.living_boxes.size(); ++i) {
    validate_box_config(c.living_boxes[i]);
    const auto& box = snapshot.living_boxes[i];
    validate_box_config(box.config);
    require(positive_box(box.bbox), "empty/reversed camera box");
    // Native construction scales the initial arena before the first forward
    // step clamps it. A scale above one legitimately starts beyond the canvas.
    require(
        box.forward_counter < std::numeric_limits<size_t>::max(),
        "invalid camera forward counter");
    require(
        box.config.name == c.living_boxes[i].name,
        "living-box names differ from topology");
    validate_translation(box.translation);
    validate_resizing(box.resizing);
  }
  std::unordered_set<uint64_t> ids;
  for (const auto& track : snapshot.detector.tracks) {
    require(
        track.tracking_id < std::numeric_limits<size_t>::max() &&
            ids.insert(track.tracking_id).second,
        "invalid/duplicate tracked-player ID");
    require(
        !track.positions.empty() &&
            track.positions.size() == track.centers.size() &&
            track.positions.size() <= detector.max_positions &&
            track.position_diffs.size() <= detector.max_velocity_positions &&
            track.position_diffs.size() <= track.positions.size() &&
            track.last_frame_id < snapshot.state.tick_count_,
        "invalid player-history shape or timestamp");
  }
}

std::string serialize_snapshot(const PlayTrackerSnapshot& snapshot) {
  validate_snapshot(snapshot);
  auto value = snapshot;
  Writer writer;
  writer.write(value);
  return writer.contents();
}

PlayTrackerSnapshot deserialize_snapshot(const std::string& contents) {
  require(contents.size() <= kMaxSnapshotBytes, "payload too large");
  PlayTrackerSnapshot snapshot;
  Reader reader(contents);
  reader.read(snapshot);
  reader.finish();
  validate_snapshot(snapshot);
  return snapshot;
}

} // namespace play_tracker
} // namespace hm
