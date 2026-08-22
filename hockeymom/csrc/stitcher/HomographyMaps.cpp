#include "hockeymom/csrc/stitcher/HomographyMaps.h"

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace hm::stitcher {
namespace {

constexpr uint16_t kInvalidCoordinate = std::numeric_limits<uint16_t>::max();
constexpr int kMaximumCoordinate = static_cast<int>(kInvalidCoordinate) - 1;
constexpr int64_t kMaximumCanvasPixels = 100000000;
constexpr double kIntegerBoundsTolerance = 1e-5;

void validate_image_size(int width, int height, const char* label) {
  if (width <= 0 || height <= 0) {
    throw std::invalid_argument(
        std::string(label) + " image dimensions must be positive");
  }
  if (width > kMaximumCoordinate || height > kMaximumCoordinate) {
    throw std::invalid_argument(
        std::string(label) +
        " image dimensions exceed the 16-bit coordinate-map limit");
  }
}

void validate_max_output_dimension(int max_output_dimension) {
  if (max_output_dimension < 0 || max_output_dimension > kMaximumCoordinate) {
    throw std::invalid_argument(
        "Maximum output dimension must be zero or between 1 and " +
        std::to_string(kMaximumCoordinate));
  }
}

std::vector<cv::Point2d> to_cv_points(
    const std::vector<std::array<double, 2>>& points,
    const char* label) {
  std::vector<cv::Point2d> result;
  result.reserve(points.size());
  for (const auto& point : points) {
    if (!std::isfinite(point[0]) || !std::isfinite(point[1])) {
      throw std::invalid_argument(
          std::string(label) + " control points must be finite");
    }
    result.emplace_back(point[0], point[1]);
  }
  return result;
}

std::array<double, 9> to_array(const cv::Matx33d& matrix) {
  std::array<double, 9> result{};
  std::copy(matrix.val, matrix.val + result.size(), result.begin());
  return result;
}

std::array<cv::Point2d, 4> transformed_corners(
    const cv::Matx33d& homography,
    int width,
    int height) {
  std::vector<cv::Point2d> source = {
      {0.0, 0.0},
      {static_cast<double>(width), 0.0},
      {static_cast<double>(width), static_cast<double>(height)},
      {0.0, static_cast<double>(height)},
  };
  std::array<double, 4> denominators{};
  double maximum_denominator = 0.0;
  for (size_t index = 0; index < source.size(); ++index) {
    const auto& point = source[index];
    const double denominator = homography(2, 0) * point.x +
        homography(2, 1) * point.y + homography(2, 2);
    if (!std::isfinite(denominator)) {
      throw std::runtime_error(
          "Estimated transform produced a non-finite projective denominator");
    }
    denominators[index] = denominator;
    maximum_denominator = std::max(maximum_denominator, std::abs(denominator));
  }
  const double denominator_tolerance = maximum_denominator * 1e-12;
  bool has_positive_denominator = false;
  bool has_negative_denominator = false;
  for (const double denominator : denominators) {
    if (std::abs(denominator) <= denominator_tolerance) {
      throw std::runtime_error(
          "Estimated transform has a projective pole on the image boundary");
    }
    has_positive_denominator |= denominator > 0.0;
    has_negative_denominator |= denominator < 0.0;
  }
  if (has_positive_denominator && has_negative_denominator) {
    throw std::runtime_error(
        "Estimated transform has a projective pole within the image bounds");
  }

  std::vector<cv::Point2d> destination;
  cv::perspectiveTransform(source, destination, cv::Mat(homography));
  if (destination.size() != 4) {
    throw std::runtime_error("OpenCV failed to transform image corners");
  }
  std::array<cv::Point2d, 4> result{};
  std::copy(destination.begin(), destination.end(), result.begin());
  for (const auto& point : result) {
    if (!std::isfinite(point.x) || !std::isfinite(point.y)) {
      throw std::runtime_error(
          "Estimated transform produced non-finite image bounds");
    }
  }
  return result;
}

struct Bounds {
  double min_x{std::numeric_limits<double>::infinity()};
  double min_y{std::numeric_limits<double>::infinity()};
  double max_x{-std::numeric_limits<double>::infinity()};
  double max_y{-std::numeric_limits<double>::infinity()};

  void include(const std::array<cv::Point2d, 4>& points) {
    for (const auto& point : points) {
      min_x = std::min(min_x, point.x);
      min_y = std::min(min_y, point.y);
      max_x = std::max(max_x, point.x);
      max_y = std::max(max_y, point.y);
    }
  }
};

HomographyImageMap make_image_map(
    const cv::Matx33d& image_to_canvas,
    int source_width,
    int source_height,
    int canvas_width,
    int canvas_height) {
  const auto corners =
      transformed_corners(image_to_canvas, source_width, source_height);
  Bounds bounds;
  bounds.include(corners);

  const int x0 = std::clamp(
      static_cast<int>(std::floor(bounds.min_x + kIntegerBoundsTolerance)),
      0,
      canvas_width);
  const int y0 = std::clamp(
      static_cast<int>(std::floor(bounds.min_y + kIntegerBoundsTolerance)),
      0,
      canvas_height);
  const int x1 = std::clamp(
      static_cast<int>(std::ceil(bounds.max_x - kIntegerBoundsTolerance)),
      0,
      canvas_width);
  const int y1 = std::clamp(
      static_cast<int>(std::ceil(bounds.max_y - kIntegerBoundsTolerance)),
      0,
      canvas_height);
  if (x1 <= x0 || y1 <= y0) {
    throw std::runtime_error(
        "Estimated transform produced an empty image mapping");
  }

  HomographyImageMap result;
  result.x_position = x0;
  result.y_position = y0;
  result.width = x1 - x0;
  result.height = y1 - y0;
  const auto pixel_count =
      static_cast<size_t>(result.width) * static_cast<size_t>(result.height);
  result.x_map.assign(pixel_count, kInvalidCoordinate);
  result.y_map.assign(pixel_count, kInvalidCoordinate);

  bool inverse_ok = false;
  const cv::Matx33d canvas_to_image =
      image_to_canvas.inv(cv::DECOMP_LU, &inverse_ok);
  if (!inverse_ok ||
      !std::all_of(
          canvas_to_image.val, canvas_to_image.val + 9, [](double value) {
            return std::isfinite(value);
          })) {
    throw std::runtime_error(
        "Estimated transform produced a singular image mapping");
  }
  cv::parallel_for_(cv::Range(0, result.height), [&](const cv::Range& range) {
    for (int row = range.start; row < range.end; ++row) {
      const double canvas_y = static_cast<double>(y0 + row);
      for (int column = 0; column < result.width; ++column) {
        const double canvas_x = static_cast<double>(x0 + column);
        const cv::Vec3d source_h =
            canvas_to_image * cv::Vec3d(canvas_x, canvas_y, 1.0);
        if (std::abs(source_h[2]) < 1e-12) {
          continue;
        }
        const double source_x = source_h[0] / source_h[2];
        const double source_y = source_h[1] / source_h[2];
        if (!std::isfinite(source_x) || !std::isfinite(source_y) ||
            source_x <= -0.5 ||
            source_x >= static_cast<double>(source_width) - 0.5 ||
            source_y <= -0.5 ||
            source_y >= static_cast<double>(source_height) - 0.5) {
          continue;
        }
        const int rounded_x = static_cast<int>(std::llround(source_x));
        const int rounded_y = static_cast<int>(std::llround(source_y));
        if (rounded_x < 0 || rounded_x >= source_width || rounded_y < 0 ||
            rounded_y >= source_height) {
          continue;
        }
        const auto index = static_cast<size_t>(row) * result.width + column;
        result.x_map[index] = static_cast<uint16_t>(rounded_x);
        result.y_map[index] = static_cast<uint16_t>(rounded_y);
      }
    }
  });
  return result;
}

void validate_estimation_inputs(
    const std::vector<std::array<double, 2>>& left_points,
    const std::vector<std::array<double, 2>>& right_points,
    int left_width,
    int left_height,
    int right_width,
    int right_height,
    double reprojection_threshold,
    double confidence,
    int max_iterations,
    size_t minimum_points,
    const char* estimator_name) {
  validate_image_size(left_width, left_height, "Left");
  validate_image_size(right_width, right_height, "Right");
  if (left_points.size() != right_points.size()) {
    throw std::invalid_argument(
        "Left and right control-point counts must match");
  }
  if (left_points.size() < minimum_points) {
    throw std::invalid_argument(
        "At least " + std::to_string(minimum_points) +
        " control-point pairs are required for " + estimator_name);
  }
  if (!std::isfinite(reprojection_threshold) || reprojection_threshold <= 0.0) {
    throw std::invalid_argument("Reprojection threshold must be positive");
  }
  if (!std::isfinite(confidence) || confidence <= 0.0 || confidence >= 1.0) {
    throw std::invalid_argument(
        std::string(estimator_name) + " confidence must be between 0 and 1");
  }
  if (max_iterations <= 0) {
    throw std::invalid_argument(
        std::string(estimator_name) + " iterations must be positive");
  }
}

HomographyMapResult build_map_result(
    const cv::Matx33d& right_to_left,
    const cv::Mat& inlier_mask,
    int left_width,
    int left_height,
    int right_width,
    int right_height,
    int max_output_dimension,
    size_t minimum_inliers,
    const char* estimator_name) {
  HomographyMapResult result;
  result.inlier_mask.reserve(inlier_mask.total());
  for (size_t index = 0; index < inlier_mask.total(); ++index) {
    result.inlier_mask.push_back(inlier_mask.ptr<uint8_t>()[index]);
  }
  const auto inlier_count = std::count_if(
      result.inlier_mask.begin(), result.inlier_mask.end(), [](uint8_t value) {
        return value != 0;
      });
  if (inlier_count < static_cast<decltype(inlier_count)>(minimum_inliers)) {
    throw std::runtime_error(
        std::string("OpenCV ") + estimator_name + " found fewer than " +
        std::to_string(minimum_inliers) + " transform inliers");
  }

  const cv::Matx33d identity = cv::Matx33d::eye();
  Bounds unscaled_bounds;
  unscaled_bounds.include(
      transformed_corners(identity, left_width, left_height));
  unscaled_bounds.include(
      transformed_corners(right_to_left, right_width, right_height));
  const double unscaled_width = unscaled_bounds.max_x - unscaled_bounds.min_x;
  const double unscaled_height = unscaled_bounds.max_y - unscaled_bounds.min_y;
  if (!std::isfinite(unscaled_width) || !std::isfinite(unscaled_height) ||
      unscaled_width <= 0.0 || unscaled_height <= 0.0) {
    throw std::runtime_error(
        std::string("OpenCV ") + estimator_name +
        " produced invalid panorama bounds");
  }

  double output_scale = 1.0;
  if (max_output_dimension > 0) {
    output_scale = std::min(
        1.0,
        static_cast<double>(max_output_dimension) /
            std::max(unscaled_width, unscaled_height));
  }
  const double canvas_width_value =
      std::ceil(unscaled_width * output_scale - kIntegerBoundsTolerance);
  const double canvas_height_value =
      std::ceil(unscaled_height * output_scale - kIntegerBoundsTolerance);
  if (!std::isfinite(canvas_width_value) ||
      !std::isfinite(canvas_height_value) || canvas_width_value <= 0.0 ||
      canvas_height_value <= 0.0 || canvas_width_value > kMaximumCoordinate ||
      canvas_height_value > kMaximumCoordinate ||
      canvas_width_value * canvas_height_value > kMaximumCanvasPixels) {
    throw std::runtime_error(
        std::string("OpenCV ") + estimator_name +
        " panorama exceeds supported coordinate-map dimensions; set a smaller "
        "maximum output dimension");
  }
  const int canvas_width = static_cast<int>(canvas_width_value);
  const int canvas_height = static_cast<int>(canvas_height_value);
  const cv::Matx33d left_to_canvas(
      output_scale,
      0.0,
      -unscaled_bounds.min_x * output_scale,
      0.0,
      output_scale,
      -unscaled_bounds.min_y * output_scale,
      0.0,
      0.0,
      1.0);
  const cv::Matx33d right_to_canvas = left_to_canvas * right_to_left;

  result.canvas_width = canvas_width;
  result.canvas_height = canvas_height;
  result.output_scale = output_scale;
  result.right_to_left_homography = to_array(right_to_left);
  result.left_to_canvas_homography = to_array(left_to_canvas);
  result.right_to_canvas_homography = to_array(right_to_canvas);
  result.image_maps[0] = make_image_map(
      left_to_canvas, left_width, left_height, canvas_width, canvas_height);
  result.image_maps[1] = make_image_map(
      right_to_canvas, right_width, right_height, canvas_width, canvas_height);
  return result;
}

} // namespace

HomographyMapResult create_homography_maps(
    const std::vector<std::array<double, 2>>& left_points,
    const std::vector<std::array<double, 2>>& right_points,
    int left_width,
    int left_height,
    int right_width,
    int right_height,
    double reprojection_threshold,
    double confidence,
    int max_iterations,
    int max_output_dimension) {
  validate_max_output_dimension(max_output_dimension);
  validate_estimation_inputs(
      left_points,
      right_points,
      left_width,
      left_height,
      right_width,
      right_height,
      reprojection_threshold,
      confidence,
      max_iterations,
      4,
      "MAGSAC++ homography");

  const auto left_cv = to_cv_points(left_points, "Left");
  const auto right_cv = to_cv_points(right_points, "Right");
  cv::Mat inlier_mask;
  const cv::Mat homography = cv::findHomography(
      right_cv,
      left_cv,
      cv::USAC_MAGSAC,
      reprojection_threshold,
      inlier_mask,
      max_iterations,
      confidence);
  if (homography.empty()) {
    throw std::runtime_error("OpenCV MAGSAC++ failed to estimate a homography");
  }
  cv::Mat homography_64;
  homography.convertTo(homography_64, CV_64F);
  cv::Matx33d right_to_left;
  std::copy(
      homography_64.ptr<double>(),
      homography_64.ptr<double>() + 9,
      right_to_left.val);
  return build_map_result(
      right_to_left,
      inlier_mask,
      left_width,
      left_height,
      right_width,
      right_height,
      max_output_dimension,
      4,
      "MAGSAC++ homography");
}

HomographyMapResult create_affine_ransac_maps(
    const std::vector<std::array<double, 2>>& left_points,
    const std::vector<std::array<double, 2>>& right_points,
    int left_width,
    int left_height,
    int right_width,
    int right_height,
    double reprojection_threshold,
    double confidence,
    int max_iterations,
    int refine_iterations,
    int max_output_dimension) {
  validate_max_output_dimension(max_output_dimension);
  validate_estimation_inputs(
      left_points,
      right_points,
      left_width,
      left_height,
      right_width,
      right_height,
      reprojection_threshold,
      confidence,
      max_iterations,
      3,
      "affine RANSAC");
  if (refine_iterations < 0) {
    throw std::invalid_argument(
        "Affine RANSAC refinement iterations cannot be negative");
  }

  const auto left_cv = to_cv_points(left_points, "Left");
  const auto right_cv = to_cv_points(right_points, "Right");
  cv::Mat inlier_mask;
  const cv::Mat affine = cv::estimateAffine2D(
      right_cv,
      left_cv,
      inlier_mask,
      cv::RANSAC,
      reprojection_threshold,
      static_cast<size_t>(max_iterations),
      confidence,
      static_cast<size_t>(refine_iterations));
  if (affine.empty()) {
    throw std::runtime_error(
        "OpenCV affine RANSAC failed to estimate a transform");
  }
  cv::Mat affine_64;
  affine.convertTo(affine_64, CV_64F);
  cv::Matx33d right_to_left = cv::Matx33d::eye();
  for (int row = 0; row < 2; ++row) {
    for (int column = 0; column < 3; ++column) {
      right_to_left(row, column) = affine_64.at<double>(row, column);
    }
  }
  return build_map_result(
      right_to_left,
      inlier_mask,
      left_width,
      left_height,
      right_width,
      right_height,
      max_output_dimension,
      3,
      "affine RANSAC");
}

} // namespace hm::stitcher
