#pragma once

#include "hockeymon/csrc/common/Gpu.h"
#include "hockeymon/csrc/common/JobRunner.h"
#include "hugin/src/hugin_base/nona/RemappedPanoImage.h"
#include "hugin/src/hugin_base/nona/Stitcher.h"

#include "unsupported/Eigen/CXX11/ThreadPool"

#include <mutex>

namespace hm {

template <class RemapImage, class AlphaImage>
class HmRemappedPanoImage
    : public HuginBase::Nona::RemappedPanoImage<RemapImage, AlphaImage> {
  typedef vigra_ext::ROIImage<RemapImage, AlphaImage> Base;
  using Super = HuginBase::Nona::RemappedPanoImage<RemapImage, AlphaImage>;
  using AdvancedOptions = HuginBase::Nona::AdvancedOptions;
  using PanoramaOptions = HuginBase::PanoramaOptions;

 public:
  /** create a remapped pano image
   *
   *  the actual remapping is done by the remapImage() function.
   */
  HmRemappedPanoImage() = default;

 public:
  // void setPanoImage(
  //     const HuginBase::SrcPanoImage& src,
  //     const PanoramaOptions& dest,
  //     vigra::Rect2D roi);

  // void setAdvancedOptions(const AdvancedOptions& advancedOptions) {
  //   m_advancedOptions = advancedOptions;
  // };

 public:
  /** calculate distance map. pixels contain distance from image center
   *
   *  setPanoImage() has to be called before!
   */
  // template<class DistImgType>
  //     void calcSrcCoordImgs(DistImgType & imgX, DistImgType & imgY);

  // /** calculate only the alpha channel.
  //  *  works for arbitrary transforms, with holes and so on,
  //  *  but is very crude and slow (remapps all image pixels...)
  //  *
  //  *  better transform all images, and get the alpha channel for free!4
  //  *
  //  *  setPanoImage() should be called before.
  //  */
  // void calcAlpha();

  /** remap a image without alpha channel*/
  template <class ImgIter, class ImgAccessor>
  void remapImage(
      vigra::triple<ImgIter, ImgIter, ImgAccessor> srcImg,
      vigra_ext::Interpolator interpol,
      Eigen::ThreadPool& gpu_thread_pool,
      AppBase::ProgressDisplay* progress,
      bool singleThreaded = false);

  /** remap a image, with alpha channel */
  // template <
  //     class ImgIter,
  //     class ImgAccessor,
  //     class AlphaIter,
  //     class AlphaAccessor>
  // void remapImage(
  //     vigra::triple<ImgIter, ImgIter, ImgAccessor> srcImg,
  //     std::pair<AlphaIter, AlphaAccessor> alphaImg,
  //     vigra_ext::Interpolator interp,
  //     AppBase::ProgressDisplay* progress,
  //     bool singleThreaded = false);

  // public:
  //     ///
  //     vigra::ImageImportInfo::ICCProfile m_ICCProfile;

 protected:
  //     SrcPanoImage m_srcImg;
  //     PanoramaOptions m_destImg;
  //     PTools::Transform m_transf;
  //     AdvancedOptions m_advancedOptions;

  using Super::m_advancedOptions;
  using Super::m_destImg;
  using Super::m_srcImg;
  using Super::m_transf;
  static inline std::mutex gpu_mutex_;
};

/** remap a image without alpha channel*/
template <class RemapImage, class AlphaImage>
template <class ImgIter, class ImgAccessor>
void HmRemappedPanoImage<RemapImage, AlphaImage>::remapImage(
    vigra::triple<ImgIter, ImgIter, ImgAccessor> srcImg,
    vigra_ext::Interpolator interpol,
    Eigen::ThreadPool& gpu_thread_pool,
    AppBase::ProgressDisplay* progress,
    bool singleThreaded) {
  assert(!singleThreaded);
  using namespace HuginBase;
  using namespace HuginBase::Nona;
  using HuginBase::SrcPanoImage;

  const bool useGPU = m_destImg.remapUsingGPU;

  if (Base::boundingBox().isEmpty())
    return;

  vigra::Diff2D srcImgSize = srcImg.second - srcImg.first;

  vigra::Size2D expectedSize = m_srcImg.getSize();
  if (useGPU) {
    const int r = expectedSize.width() % 8;
    if (r != 0)
      expectedSize += vigra::Diff2D(8 - r, 0);
  }

  DEBUG_DEBUG(
      "srcImgSize: " << srcImgSize << " m_srcImgSize: " << m_srcImg.getSize());
  vigra_precondition(
      srcImgSize == expectedSize,
      "RemappedPanoImage<RemapImage,AlphaImage>::remapImage(): image unexpectedly changed dimensions.");

  typedef typename ImgAccessor::value_type input_value_type;
  typedef typename vigra_ext::ValueTypeTraits<input_value_type>::value_type
      input_component_type;

  // setup photometric transform for this image type
  // this corrects for response curve, white balance, exposure and
  // radial vignetting
  Photometric::InvResponseTransform<input_component_type, double> invResponse(
      m_srcImg);
  invResponse.enforceMonotonicity();
  if (m_destImg.outputMode == PanoramaOptions::OUTPUT_LDR) {
    // select exposure and response curve for LDR output
    std::vector<double> outLut;
    if (!m_destImg.outputEMoRParams.empty()) {
      vigra_ext::EMoR::createEMoRLUT(m_destImg.outputEMoRParams, outLut);
    };
    double maxVal = vigra_ext::LUTTraits<input_value_type>::max();
    if (!m_destImg.outputPixelType.empty()) {
      maxVal = vigra_ext::getMaxValForPixelType(m_destImg.outputPixelType);
    }

    invResponse.setOutput(
        1.0 / pow(2.0, m_destImg.outputExposureValue),
        outLut,
        maxVal,
        m_destImg.outputRangeCompression);
  } else {
    invResponse.setHDROutput(
        true, 1.0 / pow(2.0, m_destImg.outputExposureValue));
  }

  if ((m_srcImg.hasActiveMasks()) ||
      (m_srcImg.getCropMode() != SrcPanoImage::NO_CROP) ||
      Nona::GetAdvancedOption(m_advancedOptions, "maskClipExposure", false)) {
    // need to create and additional alpha image for the crop mask...
    // not very efficient during the remapping phase, but works.
    vigra::BImage alpha(srcImgSize.x, srcImgSize.y);

    switch (m_srcImg.getCropMode()) {
      case SrcPanoImage::NO_CROP: {
        if (useGPU) {
          if (srcImgSize != m_srcImg.getSize()) {
            // src image with was increased for alignment reasons.
            // Need to make an alpha image to mask off the extended region.
            initImage(vigra::destImageRange(alpha), 0);
            initImage(
                alpha.upperLeft(),
                alpha.upperLeft() + m_srcImg.getSize(),
                alpha.accessor(),
                255);
          } else
            initImage(vigra::destImageRange(alpha), 255);
        } else
          initImage(vigra::destImageRange(alpha), 255);
        break;
      }
      case SrcPanoImage::CROP_CIRCLE: {
        vigra::Rect2D cR = m_srcImg.getCropRect();
        hugin_utils::FDiff2D m(
            (cR.left() + cR.width() / 2.0), (cR.top() + cR.height() / 2.0));

        double radius = std::min(cR.width(), cR.height()) / 2.0;
        // Default the entire alpha channel to opaque..
        initImage(vigra::destImageRange(alpha), 255);
        //..crop everything outside the circle
        vigra_ext::circularCrop(vigra::destImageRange(alpha), m, radius);
        break;
      }
      case SrcPanoImage::CROP_RECTANGLE: {
        vigra::Rect2D cR = m_srcImg.getCropRect();
        // Default the entire alpha channel to transparent..
        initImage(vigra::destImageRange(alpha), 0);
        // Make sure crop is inside the image..
        cR &= vigra::Rect2D(0, 0, srcImgSize.x, srcImgSize.y);
        // Opaque only the area within the crop rectangle..
        initImage(
            alpha.upperLeft() + cR.upperLeft(),
            alpha.upperLeft() + cR.lowerRight(),
            alpha.accessor(),
            255);
        break;
      }
      default:
        break;
    }
    if (m_srcImg.hasActiveMasks())
      vigra_ext::applyMask(
          vigra::destImageRange(alpha), m_srcImg.getActiveMasks());
    if (Nona::GetAdvancedOption(m_advancedOptions, "maskClipExposure", false)) {
      const float lowerCutoff = Nona::GetAdvancedOption(
          m_advancedOptions,
          "maskClipExposureLowerCutoff",
          NONA_DEFAULT_EXPOSURE_LOWER_CUTOFF);
      const float upperCutoff = Nona::GetAdvancedOption(
          m_advancedOptions,
          "maskClipExposureUpperCutoff",
          NONA_DEFAULT_EXPOSURE_UPPER_CUTOFF);
      vigra_ext::applyExposureClipMask(
          srcImg, vigra::destImageRange(alpha), lowerCutoff, upperCutoff);
    };
    if (useGPU) {
      ManualResetGate gate;
      gpu_thread_pool.Schedule([&]() {
        assert(check_cuda_opengl());
        // auto start = get_tick_count_ms();
        transformImageAlphaGPU(
            srcImg,
            vigra::srcImage(alpha),
            destImageRange(Base::m_image),
            destImage(Base::m_mask),
            Base::boundingBox().upperLeft(),
            m_transf,
            invResponse,
            m_srcImg.horizontalWarpNeeded(),
            interpol,
            progress);
        // auto end = get_tick_count_ms();
        // std::cout << "GPU transform took " << (end - start) << " ms"
        //           << std::endl;
        gate.signal();
      });
      gate.wait();
      if (Base::boundingBox().right() > m_destImg.getROI().right()) {
        // dest image was enlarged for GPU alignment issue
        // delete the pixels outside
        vigra::Rect2D newBoundingBox = Base::boundingBox() & m_destImg.getROI();
        Base::m_image = CopyImageNewSize(Base::m_image, newBoundingBox.size());
        Base::m_mask = CopyImageNewSize(Base::m_mask, newBoundingBox.size());
        Base::m_region = newBoundingBox;
      };
    } else {
      //auto start = get_tick_count_ms();
      transformImageAlpha(
          srcImg,
          vigra::srcImage(alpha),
          destImageRange(Base::m_image),
          destImage(Base::m_mask),
          Base::boundingBox().upperLeft(),
          m_transf,
          invResponse,
          m_srcImg.horizontalWarpNeeded(),
          interpol,
          progress,
          singleThreaded);
      //auto end = get_tick_count_ms();
      //std::cout << "CPU transform took " << (end - start) << " ms" << std::endl;
    }
  } else {
    if (useGPU) {
      if (srcImgSize != m_srcImg.getSize()) {
        // src image with was increased for alignment reasons.
        // Need to make an alpha image to mask off the extended region.
        vigra::BImage alpha(srcImgSize.x, srcImgSize.y, vigra::UInt8(0));
        initImage(
            alpha.upperLeft(),
            alpha.upperLeft() + m_srcImg.getSize(),
            alpha.accessor(),
            255);

        ManualResetGate gate;
        gpu_thread_pool.Schedule([&]() {
          assert(check_cuda_opengl());
          // std::unique_lock<std::mutex> lk(gpu_mutex_);
          // auto start = get_tick_count_ms();
          transformImageAlphaGPU(
              srcImg,
              vigra::srcImage(alpha),
              destImageRange(Base::m_image),
              destImage(Base::m_mask),
              Base::boundingBox().upperLeft(),
              m_transf,
              invResponse,
              m_srcImg.horizontalWarpNeeded(),
              interpol,
              progress);
          // auto end = get_tick_count_ms();
          // std::cout << "GPU transform took " << (end - start) << " ms" <<
          // std::endl;
          gate.signal();
        });
        gate.wait();
      } else {
        ManualResetGate gate;
        gpu_thread_pool.Schedule([&]() {
          assert(check_cuda_opengl());
          // std::unique_lock<std::mutex> lk(gpu_mutex_);
          // auto start = get_tick_count_ms();
          auto& base_image = Base::m_image;
          transformImageGPU(
              srcImg,
              destImageRange(Base::m_image),
              destImage(Base::m_mask),
              Base::boundingBox().upperLeft(),
              m_transf,
              invResponse,
              m_srcImg.horizontalWarpNeeded(),
              interpol,
              progress);
          // auto end = get_tick_count_ms();
          // std::cout << "GPU transform took " << (end - start) << " ms"
          //           << std::endl;
          gate.signal();
        });
        gate.wait();
      }
      if (Base::boundingBox().right() > m_destImg.getROI().right()) {
        // dest image was enlarged for GPU alignment issue
        // delete the pixels outside
        vigra::Rect2D newBoundingBox = Base::boundingBox() & m_destImg.getROI();
        Base::m_image = CopyImageNewSize(Base::m_image, newBoundingBox.size());
        Base::m_mask = CopyImageNewSize(Base::m_mask, newBoundingBox.size());
        Base::m_region = newBoundingBox;
      };
    } else {
      //auto start = get_tick_count_ms();
      transformImage(
          srcImg,
          destImageRange(Base::m_image),
          destImage(Base::m_mask),
          Base::boundingBox().upperLeft(),
          m_transf,
          invResponse,
          m_srcImg.horizontalWarpNeeded(),
          interpol,
          progress,
          singleThreaded);
      //auto end = get_tick_count_ms();
      //std::cout << "CPU transform took " << (end - start) << " ms" << std::endl;
    }
  }
}
#if 1
/** remap a single image
 */
template <
    class SrcImgType,
    class FlatImgType,
    class DestImgType,
    class MaskImgType>
void remapImage(
    SrcImgType& srcImg,
    const MaskImgType& srcAlpha,
    const FlatImgType& srcFlat,
    const HuginBase::SrcPanoImage& src,
    const HuginBase::PanoramaOptions& dest,
    vigra::Rect2D outputROI,
    HmRemappedPanoImage<DestImgType, MaskImgType>& remapped,
    Eigen::ThreadPool& gpu_thread_pool,
    AppBase::ProgressDisplay* progress) {
#ifdef DEBUG_REMAP
  {
    vigra::ImageExportInfo exi(DEBUG_FILE_PREFIX "hugin03_BeforeRemap.tif");
    vigra::exportImage(vigra::srcImageRange(srcImg), exi);
  }
  {
    if (srcAlpha.width() > 0) {
      vigra::ImageExportInfo exi(DEBUG_FILE_PREFIX
                                 "hugin04_BeforeRemapAlpha.tif");
      vigra::exportImage(vigra::srcImageRange(srcAlpha), exi);
    }
  }
#endif
  progress->setMessage("remapping", hugin_utils::stripPath(src.getFilename()));
  // set pano image
  DEBUG_DEBUG("setting src image with size: " << src.getSize());
  remapped.setPanoImage(src, dest, outputROI);
  // TODO: add provide support for flatfield images.
  if (srcAlpha.size().x > 0) {
    assert(false);
    // remapped.Base::remapImage(
    //     vigra::srcImageRange(srcImg),
    //     vigra::srcImage(srcAlpha),
    //     dest.interpolator,
    //     progress);
  } else {
    remapped.remapImage(
        vigra::srcImageRange(srcImg), dest.interpolator, gpu_thread_pool, progress);
  }

#ifdef DEBUG_REMAP
  {
    vigra::ImageExportInfo exi(DEBUG_FILE_PREFIX "hugin04_AfterRemap.tif");
    vigra::exportImage(vigra::srcImageRange(remapped.m_image), exi);
  }
  {
    vigra::ImageExportInfo exi(DEBUG_FILE_PREFIX "hugin04_AfterRemapAlpha.tif");
    vigra::exportImage(vigra::srcImageRange(remapped.m_mask), exi);
  }
#endif
}
#endif
} // namespace hm
