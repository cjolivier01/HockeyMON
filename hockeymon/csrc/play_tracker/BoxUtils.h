#pragma once

#include <algorithm>
#include <cassert>
#include <cfloat> // For FLT_EPSILON and DBL_EPSILON
#include <cmath>
#include <optional>
#include <ostream>
#include <tuple>

namespace hm {

using FloatValue = float;
using IntValue = int64_t;

/**
 * __          __ _    _ _____  _
 * \ \        / /| |  | |  __ \(_)
 *  \ \  /\  / / | |__| | |  | |_ _ __ ___   ___
 *   \ \/  \/ /  |  __  | |  | | | '_ ` _ \ / __|
 *    \  /\  /   | |  | | |__| | | | | | | |\__ \
 *     \/  \/    |_|  |_|_____/|_|_| |_| |_||___/
 *
 */
struct WHDims {
  FloatValue width;
  FloatValue height;
};

/**
 *  _____       _       _   _____  _  __  __
 * |  __ \     (_)     | | |  __ \(_)/ _|/ _|
 * | |__) |___  _ _ __ | |_| |  | |_| |_| |_
 * |  ___// _ \| | '_ \| __| |  | | |  _|  _|
 * | |   | (_) | | | | | |_| |__| | | | | |
 * |_|    \___/|_|_| |_|\__|_____/|_|_| |_|
 *
 */
struct PointDiff {
  FloatValue dx;
  FloatValue dy;
};

/**
 *   _____  _            _____  _  __  __
 *  / ____|(_)          |  __ \(_)/ _|/ _|
 * | (___   _  ____ ___ | |  | |_| |_| |_
 *  \___ \ | ||_  // _ \| |  | | |  _|  _|
 *  ____) || | / /|  __/| |__| | | | | |
 * |_____/ |_|/___|\___||_____/|_|_| |_|
 *
 */
struct SizeDiff {
  FloatValue dw;
  FloatValue dh;
};

/**
 *  _____       _       _
 * |  __ \     (_)     | |
 * | |__) |___  _ _ __ | |_
 * |  ___// _ \| | '_ \| __|
 * | |   | (_) | | | | | |_
 * |_|    \___/|_|_| |_|\__|
 *
 */
struct Point {
  FloatValue x;
  FloatValue y;

  PointDiff operator-(const Point& pt) const {
    return PointDiff{.dx = x - pt.x, .dy = y - pt.y};
  }
};

/**
 *  ____  ____
 * |  _ \|  _ \
 * | |_) | |_) | ___ __  __
 * |  _ <|  _ < / _ \\ \/ /
 * | |_) | |_) | (_) |>  <
 * |____/|____/ \___//_/\_\
 *
 * @brief Bounding Box
 */
struct BBox {
  BBox() = default;
  BBox(FloatValue l, FloatValue t, FloatValue r, FloatValue b)
      : left(l), top(t), right(r), bottom(b) {}
  BBox(const Point& center, const WHDims& dims) {
    auto half_w = dims.width / 2;
    auto half_h = dims.height / 2;
    left = center.x - half_w;
    right = center.x + half_w;
    top = center.y - half_h;
    bottom = center.y + half_h;
  }
  constexpr FloatValue width() const {
    assert(right >= left);
    return right - left;
  }
  constexpr FloatValue height() const {
    assert(bottom >= top);
    return bottom - top;
  }
  constexpr FloatValue aspect_ratio() const {
    return width() / height();
  }
  BBox clone() const {
    return *this;
  }
  Point center() const {
    return Point{.x = (left + right) / 2, .y = (top + bottom) / 2};
  }
  // Anchor point is bottom center atm
  Point anchor_point() const {
    return Point{.x = (left + right) / 2, .y = bottom};
  }
  WHDims size() const {
    return WHDims{.width = width(), .height = height()};
  }
  BBox make_scaled(FloatValue scale_width, FloatValue scale_height) const {
    return BBox(
        center(),
        WHDims{
            .width = width() * scale_width, .height = height() * scale_height});
  }
  BBox make_canvas_scaled(FloatValue scale_x, FloatValue scale_y) const {
    return BBox(left * scale_x, top * scale_y, right * scale_x, bottom * scale_y);
  }
  BBox inflate(
      FloatValue dleft,
      FloatValue dtop,
      FloatValue dright,
      FloatValue dbottom) const {
    return BBox(left + dleft, top + dtop, right + dright, bottom + dbottom);
  }
  BBox inflate(FloatValue dx, FloatValue dy) const {
    return BBox(left - dx, top - dy, right + dx, bottom + dy);
  }
  BBox deflate(FloatValue dx, FloatValue dy) const {
    return BBox(left + dx, top + dy, right - dx, bottom - dy);
  }
  BBox at_center(const Point& center) const {
    double half_w = width() / 2.0, half_h = height() / 2.0;
    return BBox(
        center.x - half_w,
        center.y - half_h,
        center.x + half_w,
        center.y + half_h);
  }
  BBox normalize(bool reversible) const {
    if (reversible) {
      auto min_x = std::min(left, right);
      auto max_x = std::max(left, right);
      auto min_y = std::min(top, bottom);
      auto max_y = std::max(top, bottom);
      return BBox(min_x, min_y, max_x, max_y);
    } else {
      BBox box = *this;
      if (box.left > box.right) {
        box.left = box.right;
      }
      if (box.top > box.bottom) {
        box.bottom = box.top;
      }
      return box;
    }
  }
  void validate() const {
#ifndef NDEBUG
    assert(right >= left);
    assert(bottom >= top);
#endif
  }
  float area() const {
    return width() * height();
  }
  bool empty() const {
    validate();
    return left == right || top == bottom;
  }
  // The four bbox values
  FloatValue left{0.0};
  FloatValue top{0.0};
  FloatValue right{0.0};
  FloatValue bottom{0.0};
};

#ifndef __CUDACC__

/**
 *  _____       _  _
 * |_   _|     | |(_)
 *   | |  _ __ | | _ _ __   ___  ___
 *   | | | '_ \| || | '_ \ / _ \/ __|
 *  _| |_| | | | || | | | |  __/\__ \
 * |_____|_| |_|_||_|_| |_|\___||___/
 *
 */
inline bool isZero(const float& value, float epsilon = FLT_EPSILON) {
  return std::abs(value) < epsilon;
}

inline bool isZero(const double& value, double epsilon = DBL_EPSILON) {
  return std::abs(value) < epsilon;
}

bool isClose(float a, float b, float rel_tol = 1e-6f, float abs_tol = 1e-8f);

inline constexpr FloatValue one() {
  return 1.0;
}

inline constexpr FloatValue zero() {
  return 0.0;
}

inline constexpr IntValue zero_int() {
  return 0;
}

inline FloatValue norm(const PointDiff& diff) {
  return std::sqrt(diff.dx * diff.dx + diff.dy * diff.dy);
}

inline FloatValue sign(const FloatValue value) {
  if (value > 0) {
    return 1.0f;
  } else if (value < 0) {
    return -1.0f;
  } else {
    return 0.0f;
  }
}

template <typename T>
inline T clamp(const T& value, const T& min, const T& max) {
  return std::clamp(value, min, max);
}

inline bool different_directions(const FloatValue v1, const FloatValue v2) {
  return sign(v1) * sign(v2) < 0.0;
}

// inline Point operator+(const Point& lhs, const Point& rhs) {
//   return Point{.x = lhs.x + rhs.x, .y = lhs.y + rhs.y};
// }

// inline Point operator-(const Point& lhs, const Point& rhs) {
//   return Point{.x = lhs.x - rhs.x, .y = lhs.y - rhs.y};
// }

// inline Point operator*(const Point& lhs, float scalar) {
//   return Point{.x = lhs.x * scalar, .y = lhs.y * scalar};
// }

// inline Point operator/(const Point& lhs, float scalar) {
//   return Point{.x = lhs.x / scalar, .y = lhs.y / scalar};
// }

inline PointDiff operator+(const PointDiff& lhs, const PointDiff& rhs) {
  return PointDiff{.dx = lhs.dx + rhs.dx, .dy = lhs.dy + rhs.dy};
}

inline PointDiff operator-(const PointDiff& lhs, const PointDiff& rhs) {
  return PointDiff{.dx = lhs.dx - rhs.dx, .dy = lhs.dy - rhs.dy};
}

inline PointDiff operator*(const PointDiff& lhs, float scalar) {
  return PointDiff{.dx = lhs.dx * scalar, .dy = lhs.dy * scalar};
}

inline PointDiff operator/(const PointDiff& lhs, float scalar) {
  return PointDiff{.dx = lhs.dx / scalar, .dy = lhs.dy / scalar};
}

/**
 *  ______                _   _
 * |  ____|              | | (_)
 * | |__ _   _ _ __   ___| |_ _  ___  _ __   ___
 * |  __| | | | '_ \ / __| __| |/ _ \| '_ \ / __|
 * | |  | |_| | | | | (__| |_| | (_) | | | |\__ \
 * |_|   \__,_|_| |_|\___|\__|_|\___/|_| |_||___/
 *
 */
BBox clamp_box(BBox box, const BBox& clamp_box);

BBox set_box_aspect_ratio(
    BBox setting_box,
    FloatValue aspect_ratio,
    std::optional<BBox> clamp_to_box = std::nullopt);

struct ShiftResult {
  BBox bbox;
  bool was_shifted_x;
  bool was_shifted_y;
};

ShiftResult shift_box_to_edge(const BBox& box, const BBox& bounding_box);

std::tuple<bool, bool, bool, bool> is_box_edge_on_or_outside_other_box_edge(
    const BBox& box,
    const BBox& bounding_box);

std::tuple<bool, bool> check_for_box_overshoot(
    const BBox& box,
    const BBox& bounding_box,
    const PointDiff& moving_directions,
    FloatValue epsilon = 0.01);

void clamp_if_close(
    FloatValue& var,
    const FloatValue& min,
    const FloatValue& max,
    FloatValue epsilon = 0.01);

/**
 * @brief Given some box (or image) of size 'src_size', and a final viewport
 * size of '' (i.e. the final video frame size such that we want to minimize
 * pixel loss within a box that size), calculate the minimum width and height we
 * need to fill the viewport with all available pixels after an arbitrary
 * rotation (i.e. so that pixels may rotate into view that aren't within the
 * original 'viewport_size' box).
 */
WHDims get_box_size_necessary_for_rotations(
    const WHDims& src_size,
    const WHDims& viewport_size);

std::ostream& operator<<(std::ostream& os, const WHDims& dims);
std::ostream& operator<<(std::ostream& os, const PointDiff& diff);
std::ostream& operator<<(std::ostream& os, const SizeDiff& diff);
std::ostream& operator<<(std::ostream& os, const Point& point);
std::ostream& operator<<(std::ostream& os, const BBox& bbox);

#endif // __CUDACC__

} // namespace hm
