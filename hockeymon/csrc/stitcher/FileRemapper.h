#pragma once

#include "hockeymon/csrc/stitcher/HmRemappedPanoImage.h"

#include "nona/StitcherOptions.h"
#include "panodata/Panorama.h"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace hm {

inline std::vector<std::unique_ptr<ManualResetGate>> make_gates(
    std::size_t count,
    bool initiallySet = false) {
  std::vector<std::unique_ptr<ManualResetGate>> results;
  results.resize(count);
  for (auto& g : results) {
    g = std::make_unique<ManualResetGate>(initiallySet);
  }
  return results;
}

inline void wait_for_all(std::vector<std::unique_ptr<ManualResetGate>>& gates) {
  std::for_each(gates.begin(), gates.end(), [](auto& g) { g->wait(); });
}

/** functor to create a remapped image */
template <typename ImageType, typename AlphaType>
class HmSingleImageRemapper {
 public:
  HmSingleImageRemapper() = default;

  /** create a remapped pano image.
   *
   *  The image ownership is transferred to the caller.
   */
  virtual std::unique_ptr<HmRemappedPanoImage<ImageType, AlphaType>> getRemapped(
      const HuginBase::PanoramaData& pano,
      const HuginBase::PanoramaOptions& opts,
      unsigned int imgNr,
      const std::shared_ptr<hm::MatrixRGB> image,
      vigra::Rect2D outputROI,
      Eigen::ThreadPool& gpu_thread_pool,
      AppBase::ProgressDisplay* progress) = 0;

  virtual ~HmSingleImageRemapper() = default;

  void setAdvancedOptions(
      const HuginBase::Nona::AdvancedOptions advancedOptions) {
    std::unique_lock<std::mutex> lk(mu_);
    m_advancedOptions = advancedOptions;
  }
  constexpr const HuginBase::Nona::AdvancedOptions& get_advanced_options()
      const {
    return m_advancedOptions;
  }

 protected:
  std::size_t pass_{0};

 private:
  std::mutex mu_;
  HuginBase::Nona::AdvancedOptions m_advancedOptions;
};

/** functor to create a remapped image, loads image from disk */
template <typename ImageType, typename AlphaType>
class HmFileRemapper : public HmSingleImageRemapper<ImageType, AlphaType> {
 public:
  HmFileRemapper() : HmSingleImageRemapper<ImageType, AlphaType>() {}

  virtual ~HmFileRemapper(){};

  typedef std::vector<float> LUT;

 public:
  void loadImage(
      const HuginBase::PanoramaOptions& opts,
      vigra::ImageImportInfo& info,
      ImageType& srcImg,
      AlphaType& srcAlpha) {}

  std::unique_ptr<HmRemappedPanoImage<ImageType, AlphaType>> getRemapped(
      const HuginBase::PanoramaData& pano,
      const HuginBase::PanoramaOptions& opts,
      unsigned int imgNr,
      const std::shared_ptr<hm::MatrixRGB> image,
      vigra::Rect2D outputROI,
      Eigen::ThreadPool& gpu_thread_pool,
      AppBase::ProgressDisplay* progress) override;

 protected:
  AlphaType srcAlpha_;
 std::mutex image_import_infos_mu_;
  std::vector<std::unique_ptr<vigra::ImageImportInfo>> image_import_infos_;
};

} // namespace hm
