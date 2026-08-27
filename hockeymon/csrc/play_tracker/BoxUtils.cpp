#include "hockeymon/csrc/play_tracker/BoxUtils.h"

#include <iomanip>

namespace hm {

std::ostream& operator<<(std::ostream& os, const WHDims& dims) {
  os << "WHDims(width=" << std::setw(10) << std::fixed << std::setprecision(6)
     << dims.width << ", height=" << std::setw(10) << std::fixed
     << std::setprecision(6) << dims.height << ")";
  return os;
}

std::ostream& operator<<(std::ostream& os, const PointDiff& diff) {
  os << "PointDiff(dx=" << std::setw(10) << std::fixed << std::setprecision(6)
     << diff.dx << ", dy=" << std::setw(10) << std::fixed
     << std::setprecision(6) << diff.dy << ")";
  return os;
}

std::ostream& operator<<(std::ostream& os, const SizeDiff& diff) {
  os << "SizeDiff(dw=" << std::setw(10) << std::fixed << std::setprecision(6)
     << diff.dw << ", dh=" << std::setw(10) << std::fixed
     << std::setprecision(6) << diff.dh << ")";
  return os;
}

std::ostream& operator<<(std::ostream& os, const Point& pt) {
  os << "Point(x=" << std::setw(10) << std::fixed << std::setprecision(6)
     << pt.x << ", y=" << std::setw(10) << std::fixed << std::setprecision(6)
     << pt.y << ")";
  return os;
}

std::ostream& operator<<(std::ostream& os, const BBox& bbox) {
  os << "BBox(l=" << std::setw(10) << std::fixed << std::setprecision(6)
     << bbox.left << ", t=" << std::setw(10) << std::fixed
     << std::setprecision(6) << bbox.top << ", r=" << std::setw(10)
     << std::fixed << std::setprecision(6) << bbox.right
     << ", b=" << std::setw(10) << std::fixed << std::setprecision(6)
     << bbox.bottom << ")";
  return os;
}
// namespace play_tracker {

bool isClose(float a, float b, float rel_tol, float abs_tol) {
  // Compute the absolute difference
  float abs_diff = std::fabs(a - b);

  // Check if the numbers are close enough considering absolute tolerance
  if (abs_diff <= abs_tol) {
    return true;
  }

  // Compute the relative tolerance component
  float max_abs = std::max(std::fabs(a), std::fabs(b));
  if (abs_diff <= max_abs * rel_tol) {
    return true;
  }

  // If neither absolute nor relative tolerance conditions are met, return false
  return false;
}

void clamp_if_close(
    FloatValue& var,
    const FloatValue& min,
    const FloatValue& max,
    FloatValue epsilon) {
  assert(min <= max);
  if (var < min) {
    assert(min - var <= epsilon);
    var = min;
  } else if (var > max) {
    assert(var - max < epsilon);
    var = max;
  }
}

BBox clamp_box(BBox box, const BBox& clamp_box) {
  clamp_box.validate();
  box.validate();
  box.left = std::clamp(box.left, clamp_box.left, clamp_box.right);
  box.right = std::clamp(box.right, clamp_box.left, clamp_box.right);
  box.top = std::clamp(box.top, clamp_box.top, clamp_box.bottom);
  box.bottom = std::clamp(box.bottom, clamp_box.top, clamp_box.bottom);
  box.validate();
  return box;
}

BBox set_box_aspect_ratio(
    BBox setting_box,
    FloatValue aspect_ratio,
    std::optional<BBox> clamp_to_box) {
  if (clamp_to_box.has_value()) {
    setting_box = clamp_box(setting_box, *clamp_to_box);
  }
  const float w = setting_box.width(), h = setting_box.height();
  float new_h, new_w;
  if (w / h < aspect_ratio) {
    new_h = h;
    new_w = new_h * aspect_ratio;
  } else {
    new_w = w;
    new_h = new_w / aspect_ratio;
  }
  assert(new_w >= w); // should always grow to accomodate
  // TODO: Should clamp again to arena box, and still maintain aspect ratio

  return BBox(setting_box.center(), WHDims{.width = new_w, .height = new_h});
}

ShiftResult shift_box_to_edge(const BBox& box, const BBox& bounding_box) {
  ShiftResult result{
      .bbox = box, .was_shifted_x = false, .was_shifted_y = false};
  // FloatValue xw = bounding_box.width(), xh = bounding_box.height();
  // TODO: Make top-left of bounding box not need to be zero
  // assert(isZero(bounding_box.left) && isZero(bounding_box.top));
#if 1
  // Should clamped somewhere along the way before now
  // assert(box.width() <= bounding_box.width());
  // assert(box.height() <= bounding_box.height());
  FloatValue min_x = bounding_box.left, max_x = bounding_box.right,
             min_y = bounding_box.top, max_y = bounding_box.bottom;
  if (result.bbox.left < min_x) {
    FloatValue offset = result.bbox.left - min_x;
    result.bbox.right -= offset;
    result.bbox.left -= offset;
    result.was_shifted_x = true;
  } else if (result.bbox.right >= max_x) {
    FloatValue offset = result.bbox.right - max_x;
    result.bbox.left -= offset;
    result.bbox.right -= offset;
    result.was_shifted_x = true;
  }
  if (result.bbox.top < min_y) {
    FloatValue offset = result.bbox.top - min_y;
    result.bbox.bottom -= offset;
    result.bbox.top -= offset;
    result.was_shifted_y = true;
  } else if (result.bbox.bottom >= max_y) {
    FloatValue offset = result.bbox.bottom - max_y;
    result.bbox.top -= offset;
    result.bbox.bottom -= offset;
    result.was_shifted_y = true;
  }
#else
  if (result.bbox.left < 0) {
    result.bbox.right -= result.bbox.left;
    result.bbox.left -= result.bbox.left;
    result.was_shifted_x = true;
  } else if (result.bbox.right >= xw) {
    FloatValue offset = result.bbox.right - xw;
    result.bbox.left -= offset;
    result.bbox.right -= offset;
    result.was_shifted_x = true;
  }
  if (result.bbox.top < 0) {
    result.bbox.bottom -= result.bbox.top;
    result.bbox.top -= result.bbox.top;
    result.was_shifted_y = true;
  } else if (result.bbox.bottom >= xh) {
    FloatValue offset = result.bbox.bottom - xh;
    result.bbox.top -= offset;
    result.bbox.bottom -= offset;
    result.was_shifted_y = true;
  }
#endif
  clamp_if_close(result.bbox.left, bounding_box.left, bounding_box.right);
  clamp_if_close(result.bbox.right, bounding_box.left, bounding_box.right);
  clamp_if_close(result.bbox.top, bounding_box.top, bounding_box.bottom);
  clamp_if_close(result.bbox.bottom, bounding_box.top, bounding_box.bottom);
  result.bbox.validate();
  return result;
}

std::tuple<bool, bool, bool, bool> is_box_edge_on_or_outside_other_box_edge(
    const BBox& box,
    const BBox& bounding_box) {
  return std::make_tuple(
      box.left <= bounding_box.left,
      box.top <= bounding_box.top,
      box.right >= bounding_box.right,
      box.bottom >= bounding_box.bottom);
}

std::tuple<bool, bool> check_for_box_overshoot(
    const BBox& box,
    const BBox& bounding_box,
    const PointDiff& moving_directions,
    FloatValue epsilon) {
  const auto any_on_edge =
      is_box_edge_on_or_outside_other_box_edge(box, bounding_box);

  const bool left_on_edge =
      std::get<0>(any_on_edge) && moving_directions.dx < epsilon;
  const bool right_on_edge =
      std::get<2>(any_on_edge) && moving_directions.dx > -epsilon;
  const bool x_on_edge = left_on_edge || right_on_edge;

  const bool top_on_edge =
      std::get<1>(any_on_edge) && moving_directions.dy < epsilon;
  const bool bottom_on_edge =
      std::get<3>(any_on_edge) && moving_directions.dy > -epsilon;
  const bool y_on_edge = top_on_edge || bottom_on_edge;

  return std::make_tuple(x_on_edge, y_on_edge);
}

WHDims get_box_size_necessary_for_rotations(
    const WHDims& src_size,
    const WHDims& viewport_size) {
  FloatValue min_width_per_size = std::sqrt(
      (viewport_size.width * viewport_size.width) +
      (viewport_size.height * viewport_size.height)) / 2.0;
  FloatValue w = std::min(src_size.width, min_width_per_size * 2);
  FloatValue h = std::min(src_size.height, min_width_per_size * 2);
  return WHDims{.width = w, .height = h};
}

} // namespace hm
