#pragma once

#include <array>
#include <cstdint>
#include <vector>

namespace hm::stitcher {

struct HomographyImageMap {
  int x_position{0};
  int y_position{0};
  int width{0};
  int height{0};
  std::vector<uint16_t> x_map;
  std::vector<uint16_t> y_map;
};

struct HomographyMapResult {
  int canvas_width{0};
  int canvas_height{0};
  double output_scale{1.0};
  std::array<double, 9> right_to_left_homography{};
  std::array<double, 9> left_to_canvas_homography{};
  std::array<double, 9> right_to_canvas_homography{};
  std::vector<uint8_t> inlier_mask;
  std::array<HomographyImageMap, 2> image_maps;
};

/**
 * Estimate right-to-left geometry with OpenCV's MAGSAC++ implementation and
 * build inverse coordinate maps for a two-image panorama.
 */
HomographyMapResult create_homography_maps(
    const std::vector<std::array<double, 2>>& left_points,
    const std::vector<std::array<double, 2>>& right_points,
    int left_width,
    int left_height,
    int right_width,
    int right_height,
    double reprojection_threshold = 3.0,
    double confidence = 0.999,
    int max_iterations = 10000,
    int max_output_dimension = 0);

/**
 * Estimate a right-to-left affine transform with OpenCV RANSAC and build
 * inverse coordinate maps for a two-image panorama.
 */
HomographyMapResult create_affine_ransac_maps(
    const std::vector<std::array<double, 2>>& left_points,
    const std::vector<std::array<double, 2>>& right_points,
    int left_width,
    int left_height,
    int right_width,
    int right_height,
    double reprojection_threshold = 10.0,
    double confidence = 0.999,
    int max_iterations = 10000,
    int refine_iterations = 10,
    int max_output_dimension = 0);

} // namespace hm::stitcher
