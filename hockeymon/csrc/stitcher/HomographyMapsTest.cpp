#include "hockeymon/csrc/stitcher/HomographyMaps.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

int main() {
  const std::vector<std::array<double, 2>> right_points = {
      {5.0, 5.0},
      {75.0, 5.0},
      {75.0, 55.0},
      {5.0, 55.0},
      {30.0, 20.0},
      {50.0, 40.0},
      {20.0, 50.0},
  };
  std::vector<std::array<double, 2>> left_points;
  left_points.reserve(right_points.size());
  for (const auto& point : right_points) {
    left_points.push_back({point[0] + 20.0, point[1] + 10.0});
  }
  left_points.back() = {90.0, 5.0};

  const auto result = hm::stitcher::create_homography_maps(
      left_points, right_points, 100, 80, 100, 80, 1.0, 0.999, 10000, 0);
  assert(result.canvas_width == 120);
  assert(result.canvas_height == 90);
  assert(std::abs(result.right_to_left_homography[2] - 20.0) < 1e-4);
  assert(std::abs(result.right_to_left_homography[5] - 10.0) < 1e-4);
  assert(result.image_maps[0].x_position == 0);
  assert(result.image_maps[0].y_position == 0);
  assert(result.image_maps[1].x_position == 20);
  assert(result.image_maps[1].y_position == 10);
  assert(result.image_maps[0].x_map.front() == 0);
  assert(result.image_maps[0].y_map.front() == 0);
  assert(result.image_maps[1].x_map.front() == 0);
  assert(result.image_maps[1].y_map.front() == 0);
  assert(result.inlier_mask.size() == right_points.size());
  assert(result.inlier_mask.back() == 0);

  const auto scaled = hm::stitcher::create_homography_maps(
      left_points, right_points, 100, 80, 100, 80, 1.0, 0.999, 10000, 60);
  assert(scaled.canvas_width == 60);
  assert(scaled.canvas_height == 45);
  assert(std::abs(scaled.output_scale - 0.5) < 1e-6);

  std::vector<std::array<double, 2>> affine_left_points;
  affine_left_points.reserve(right_points.size());
  for (const auto& point : right_points) {
    affine_left_points.push_back(
        {1.02 * point[0] - 0.12 * point[1] + 18.0,
         0.08 * point[0] + 0.98 * point[1] - 4.0});
  }
  affine_left_points.back() = {-100.0, 200.0};

  const auto affine = hm::stitcher::create_affine_ransac_maps(
      affine_left_points,
      right_points,
      100,
      80,
      100,
      80,
      0.1,
      0.999,
      10000,
      10,
      0);
  assert(std::abs(affine.right_to_left_homography[0] - 1.02) < 1e-4);
  assert(std::abs(affine.right_to_left_homography[1] + 0.12) < 1e-4);
  assert(std::abs(affine.right_to_left_homography[2] - 18.0) < 1e-4);
  assert(std::abs(affine.right_to_left_homography[3] - 0.08) < 1e-4);
  assert(std::abs(affine.right_to_left_homography[4] - 0.98) < 1e-4);
  assert(std::abs(affine.right_to_left_homography[5] + 4.0) < 1e-4);
  assert(affine.right_to_left_homography[6] == 0.0);
  assert(affine.right_to_left_homography[7] == 0.0);
  assert(affine.right_to_left_homography[8] == 1.0);
  assert(affine.canvas_width >= 119 && affine.canvas_width <= 121);
  assert(affine.canvas_height >= 86 && affine.canvas_height <= 88);
  assert(affine.inlier_mask.size() == right_points.size());
  assert(affine.inlier_mask.back() == 0);

  const std::vector<std::array<double, 2>> pole_right_points = {
      {5.0, 5.0},
      {20.0, 10.0},
      {35.0, 30.0},
      {50.0, 15.0},
      {65.0, 45.0},
      {75.0, 60.0},
      {10.0, 70.0},
      {45.0, 55.0},
  };
  std::vector<std::array<double, 2>> pole_left_points;
  pole_left_points.reserve(pole_right_points.size());
  for (const auto& point : pole_right_points) {
    const double denominator = 1.0 - 0.011 * point[0];
    pole_left_points.push_back(
        {point[0] / denominator, point[1] / denominator});
  }
  bool rejected_projective_pole = false;
  try {
    static_cast<void>(hm::stitcher::create_homography_maps(
        pole_left_points,
        pole_right_points,
        100,
        80,
        100,
        80,
        0.1,
        0.999,
        10000,
        0));
  } catch (const std::runtime_error& error) {
    rejected_projective_pole =
        std::string(error.what()).find("projective pole") != std::string::npos;
  }
  assert(rejected_projective_pole);

  bool rejected_non_finite_confidence = false;
  try {
    static_cast<void>(hm::stitcher::create_affine_ransac_maps(
        affine_left_points,
        right_points,
        100,
        80,
        100,
        80,
        0.1,
        std::numeric_limits<double>::quiet_NaN(),
        10000,
        10,
        0));
  } catch (const std::invalid_argument&) {
    rejected_non_finite_confidence = true;
  }
  assert(rejected_non_finite_confidence);

  bool rejected_invalid_maximum_dimension = false;
  try {
    static_cast<void>(hm::stitcher::create_affine_ransac_maps(
        affine_left_points,
        right_points,
        100,
        80,
        100,
        80,
        0.1,
        0.999,
        10000,
        10,
        -1));
  } catch (const std::invalid_argument&) {
    rejected_invalid_maximum_dimension = true;
  }
  assert(rejected_invalid_maximum_dimension);

  return 0;
}
