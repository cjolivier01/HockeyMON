#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include <yaml-cpp/yaml.h>

namespace hm {

/*
RINK_CONFIG = {
    "vallco": {
        "fixed_edge_scaling_factor": 0.8,
    },
    "dublin": {
        "fixed_edge_scaling_factor": 0.8,
    },
    "yerba_buena": {
        "fixed_edge_scaling_factor": 2.0,
    }
}
*/

struct HMPostprocessConfig {
  static constexpr bool BASIC_DEBUGGING = false;

  // Display the image every frame (slow)
  bool show_image = false || BASIC_DEBUGGING;

  // Draw individual player boxes, tracking ids, speed and history trails
  bool plot_individual_player_tracking = false;

  // Draw all detection boxes (even if not tracking the detection)
  bool plot_all_detections = false;

  // Draw intermediate boxes which are used to compute the final camera box
  bool plot_cluster_tracking = false || BASIC_DEBUGGING;

  bool plot_camera_tracking = false || BASIC_DEBUGGING;

  // Plot frame ID and speed/velocity in upper-left corner
  bool plot_speed = false;

  // Use a differenmt algorithm when fitting to the proper aspect ratio,
  // such that the box calculated is much larger and often takes
  // the entire height.  The drawback is there's not much zooming.
  bool max_in_aspec_ratio = true;

  // Only apply zoom when the camera box is against
  // either the left or right edge of the video
  bool no_max_in_aspec_ratio_at_edges = false;

  // Zooming is fixed based upon the horizonal position's distance from center
  bool apply_fixed_edge_scaling = true;

  float fixed_edge_scaling_factor =
      0.8; // RINK_CONFIG[rink]["fixed_edge_scaling_factor"]

  bool fixed_edge_rotation = false;

  //float fixed_edge_rotation_angle = 35.0;

  // Use "sticky" panning, where panning occurs in less frequent,
  // but possibly faster, pans rather than a constant
  // pan (which may appear tpo "wiggle")
  bool sticky_pan = true;

  // Plot the component shapes directly related to camera stickiness
  bool plot_sticky_camera = false || BASIC_DEBUGGING;

  // Skip some number of frames before post-processing. Useful for debugging a
  // particular section of video and being able to reach
  // that portiuon of the video more quickly
  std::size_t skip_frame_count = 0;

  // Moving right-to-left
  // bool skip_frame_count = 450

  // Stop at the given frame and (presumably) output the final video.
  // Useful for debugging a
  // particular section of video and being able to reach
  // that portiuon of the video more quickly
  std::size_t stop_at_frame = 0;

  // Crop the final image to the camera window (possibly zoomed)
  bool crop_output_image = true && !BASIC_DEBUGGING;

  // Use cuda for final image resizing (if possible)
  bool use_cuda = false;

  // Draw watermark on the image
  bool use_watermark = true;
  // bool use_watermark = false;

  std::string to_string() const;
};

class ImagePostProcessor {
 public:
  ImagePostProcessor(
      std::shared_ptr<HMPostprocessConfig> postprocess_config,
      std::string config_file);

 private:
  std::shared_ptr<HMPostprocessConfig> postprocess_config_;
  YAML::Node config_;
};

} // namespace hm
