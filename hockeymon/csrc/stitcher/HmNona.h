#pragma once

#include "hockeymon/csrc/common/MatrixRGB.h"
#include "hockeymon/csrc/stitcher/FileRemapper.h"
#include "hockeymon/csrc/stitcher/HmStitcher.h"

#include "nona/StitcherOptions.h"
#include "panodata/Panorama.h"

#include <mutex>
#include <string>
#include <vector>

namespace hm {

/**
 *  _    _           _   _
 * | |  | |         | \ | |
 * | |__| |_ __ ___ |  \| | ___  _ __   __ _
 * |  __  | '_ ` _ \| . ` |/ _ \| '_ \ / _` |
 * | |  | | | | | | | |\  | (_) | | | | (_| |
 * |_|  |_|_| |_| |_|_| \_|\___/|_| |_|\__,_|
 *
 *
 */
class HmNona {
  using ImageType = vigra::BRGBImage;

 public:
  HmNona(std::string project_file);
  ~HmNona();
  bool load_project(const std::string& project_file);

  std::vector<std::unique_ptr<hm::MatrixRGB>> remap_images(
      std::shared_ptr<hm::MatrixRGB> image1,
      std::shared_ptr<hm::MatrixRGB> image2);

  std::vector<std::tuple<std::tuple<float, float>, std::tuple<float, float>>>
  get_control_points() const;

 private:
  void set_ideal_output_size();

  std::string project_file_;
  HuginBase::PanoramaOptions opts_;
  HuginBase::Nona::AdvancedOptions adv_options_;
  HuginBase::Panorama pano_;
  HmFileRemapper<ImageType, vigra::BImage> file_remapper_;
  std::size_t image_pair_pass_count_{0};
  std::unique_ptr<AppBase::DummyProgressDisplay> pdisp_;
  std::unique_ptr<HmMultiImageRemapper<ImageType, vigra::BImage>> stitcher_;

  std::mutex nona_init_mu_;
  static inline std::mutex gpu_thread_pool_mu_;
  static inline std::unique_ptr<Eigen::ThreadPool> gpu_thread_pool_;
  static std::size_t nona_count_;
};

} // namespace hm
