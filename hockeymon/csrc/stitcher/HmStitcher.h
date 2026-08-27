#pragma once

#include "hockeymon/csrc/common/MatrixRGB.h"
#include "hockeymon/csrc/stitcher/FileRemapper.h"
#include "hockeymon/csrc/stitcher/HmRemappedPanoImage.h"

#include "hugin/src/hugin_base/nona/Stitcher.h"
#include "hugin/src/hugin_base/nona/StitcherOptions.h"

#include "unsupported/Eigen/CXX11/ThreadPool"

#include <memory>
#include <string>

namespace hm {

/** remap a set of images, and store the individual remapped files. */
template <typename ImageType, typename AlphaType>
class HmMultiImageRemapper
    : public HuginBase::Nona::MultiImageRemapper<ImageType, AlphaType> {
 public:
  using Base = HuginBase::Nona::MultiImageRemapper<ImageType, AlphaType>;
  using MultiImageRemapper =
      HuginBase::Nona::MultiImageRemapper<ImageType, AlphaType>;
  using PanoramaData = HuginBase::PanoramaData;
  using PanoramaOptions = HuginBase::PanoramaOptions;
  using AdvancedOptions = HuginBase::Nona::AdvancedOptions;
  using UIntSet = HuginBase::UIntSet;
  using Base::m_basename;

  HmMultiImageRemapper(
      const PanoramaData& pano,
      AppBase::ProgressDisplay* progress)
      : MultiImageRemapper(pano, progress),
        thread_pool_(std::make_unique<Eigen::ThreadPool>(2)) {}

  void set_input_images(
      std::shared_ptr<MatrixRGB> image1,
      std::shared_ptr<MatrixRGB> image2) {
    images_ = std::vector<std::shared_ptr<MatrixRGB>>{
        std::move(image1), std::move(image2)};
    output_images_.clear();
    output_images_.resize(images_.size());
  }

  std::vector<std::unique_ptr<MatrixRGB>> consume_output_images() {
    std::size_t sz = output_images_.size();
    auto result = std::move(output_images_);
    output_images_.resize(sz);
    return result;
  }

  std::vector<std::unique_ptr<MatrixRGB>> stitch(
      const PanoramaOptions& opts,
      UIntSet& images,
      const std::string& basename,
      HmSingleImageRemapper<ImageType, AlphaType>& remapper,
      const AdvancedOptions& advOptions,
      Eigen::ThreadPool& gpu_thread_pool) {
    ++pass_;
    std::vector<std::unique_ptr<MatrixRGB>> results;
    // Skip over direct base class stitch call
    if (pass_ == 1) {
      // Base::Base::stitch(opts, images, basename, remapper);
      Base::Base::m_images = images;
      Base::calcOutputROIS(opts, images);
    }
    DEBUG_ASSERT(
        opts.outputFormat == PanoramaOptions::TIFF_multilayer ||
        opts.outputFormat == PanoramaOptions::TIFF_m ||
        opts.outputFormat == PanoramaOptions::JPEG_m ||
        opts.outputFormat == PanoramaOptions::PNG_m ||
        opts.outputFormat == PanoramaOptions::HDR_m ||
        opts.outputFormat == PanoramaOptions::EXR_m);

    m_basename = basename;

    // setup the output.
    prepareOutputFile(opts, advOptions);

    // remap each image and save
    if (pass_ == 1) {
      mod_options_.clear();
      for (UIntSet::const_iterator it = images.begin(); it != images.end();
           ++it) {
        // get a remapped image.
        PanoramaOptions modOptions(opts);
        if (HuginBase::Nona::GetAdvancedOption(
                advOptions, "ignoreExposure", false)) {
          modOptions.outputExposureValue =
              Base::m_pano.getImage(*it).getExposureValue();
          modOptions.outputRangeCompression = 0.0;
        };
        mod_options_.emplace_back(std::move(modOptions));
      }
    }
    auto gates = make_gates(images.size());
    std::vector<std::size_t> img_indexes{images.begin(), images.end()};
    int img_count = img_indexes.size();
    for (int i = 0; i < img_count; ++i) {
      thread_pool_->Schedule([this,
                              &gates,
                              i,
                              &remapper,
                              &img_indexes,
                              &opts,
                              &advOptions,
                              &gpu_thread_pool]() {
        // get a remapped image.
        std::unique_ptr<HmRemappedPanoImage<ImageType, AlphaType>> remapped =
            remapper.getRemapped(
                Base::m_pano,
                mod_options_.at(i),
                img_indexes[i],
                images_.at(i),
                Base::m_rois[i],
                gpu_thread_pool,
                Base::m_progress);
        try {
          saveRemapped(
              *remapped,
              img_indexes[i],
              Base::m_pano.getNrOfImages(),
              opts,
              advOptions);
        } catch (vigra::PreconditionViolation& e) {
          // this can be thrown, if an image
          // is completely out of the pano
          std::cerr << e.what();
        }
        gates.at(i)->signal();
      });
    }
    wait_for_all(gates);
    results = consume_output_images();

    finalizeOutputFile(opts);
    if (Base::m_progress) {
      Base::m_progress->taskFinished();
    }
    return results;
  }

  /** prepare the output file (setup file structures etc.) */
  virtual void prepareOutputFile(
      const PanoramaOptions& opts,
      const AdvancedOptions& advOptions) {
    // Base::m_progress->setMessage("Multiple images output");
  }

  template <
      class ImageIterator,
      class ImageAccessor,
      class AlphaIterator,
      class AlphaAccessor>
  static void exportImageAlpha(
      VIGRA_UNIQUE_PTR<vigra::Encoder>& encoder,
      ImageIterator image_upper_left,
      ImageIterator image_lower_right,
      ImageAccessor image_accessor,
      AlphaIterator alpha_upper_left,
      AlphaAccessor alpha_accessor,
      const vigra::ImageExportInfo& export_info,
      /* isScalar? */ vigra::VigraFalseType) {
    using namespace vigra;
    using namespace vigra::detail;
    typedef typename AlphaAccessor::value_type AlphaValueType;

    if (!encoder) {
      encoder = vigra::encoder(export_info);
    }

    const std::string pixel_type(export_info.getPixelType());
    const pixel_t type(pixel_t_of_string(pixel_type));
    encoder->setPixelType(pixel_type);
    vigra_precondition(
        isBandNumberSupported(
            encoder->getFileType(), image_accessor.size(image_upper_left) + 1U),
        "exportImageAlpha(): file format does not support requested number of bands (color channels)");

    // TM: no explicit downcast, when needed this should be done by specialed
    // code before calling exportImageAlpha
    const range_t alpha_source_range(
        vigra_ext::LUTTraits<AlphaValueType>::min(),
        vigra_ext::LUTTraits<AlphaValueType>::max());
    const range_t mask_destination_range(
        0.0f, vigra_ext::getMaxValForPixelType(pixel_type));

    // check if alpha channel matches
    if (alpha_source_range.first != mask_destination_range.first ||
        alpha_source_range.second != mask_destination_range.second) {
      const linear_transform alpha_rescaler(
          alpha_source_range, mask_destination_range);
      switch (type) {
        case UNSIGNED_INT_8:
          write_image_bands_and_alpha<UInt8>(
              encoder.get(),
              image_upper_left,
              image_lower_right,
              image_accessor,
              identity(),
              alpha_upper_left,
              alpha_accessor,
              alpha_rescaler);
          break;
        case UNSIGNED_INT_16:
          write_image_bands_and_alpha<UInt16>(
              encoder.get(),
              image_upper_left,
              image_lower_right,
              image_accessor,
              identity(),
              alpha_upper_left,
              alpha_accessor,
              alpha_rescaler);
          break;
        case UNSIGNED_INT_32:
          write_image_bands_and_alpha<UInt32>(
              encoder.get(),
              image_upper_left,
              image_lower_right,
              image_accessor,
              identity(),
              alpha_upper_left,
              alpha_accessor,
              alpha_rescaler);
          break;
        case SIGNED_INT_16:
          write_image_bands_and_alpha<Int16>(
              encoder.get(),
              image_upper_left,
              image_lower_right,
              image_accessor,
              identity(),
              alpha_upper_left,
              alpha_accessor,
              alpha_rescaler);
          break;
        case SIGNED_INT_32:
          write_image_bands_and_alpha<Int32>(
              encoder.get(),
              image_upper_left,
              image_lower_right,
              image_accessor,
              identity(),
              alpha_upper_left,
              alpha_accessor,
              alpha_rescaler);
          break;
        case IEEE_FLOAT_32:
          write_image_bands_and_alpha<float>(
              encoder.get(),
              image_upper_left,
              image_lower_right,
              image_accessor,
              identity(),
              alpha_upper_left,
              alpha_accessor,
              alpha_rescaler);
          break;
        case IEEE_FLOAT_64:
          write_image_bands_and_alpha<double>(
              encoder.get(),
              image_upper_left,
              image_lower_right,
              image_accessor,
              identity(),
              alpha_upper_left,
              alpha_accessor,
              alpha_rescaler);
          break;
        default:
          vigra_fail("vigra::detail::exportImageAlpha<scalar>: not reached");
      }
    } else {
      switch (type) {
        case UNSIGNED_INT_8:
          write_image_bands_and_alpha<UInt8>(
              encoder.get(),
              image_upper_left,
              image_lower_right,
              image_accessor,
              identity(),
              alpha_upper_left,
              alpha_accessor,
              identity());
          break;
        case UNSIGNED_INT_16:
          write_image_bands_and_alpha<UInt16>(
              encoder.get(),
              image_upper_left,
              image_lower_right,
              image_accessor,
              identity(),
              alpha_upper_left,
              alpha_accessor,
              identity());
          break;
        case UNSIGNED_INT_32:
          write_image_bands_and_alpha<UInt32>(
              encoder.get(),
              image_upper_left,
              image_lower_right,
              image_accessor,
              identity(),
              alpha_upper_left,
              alpha_accessor,
              identity());
          break;
        case SIGNED_INT_16:
          write_image_bands_and_alpha<Int16>(
              encoder.get(),
              image_upper_left,
              image_lower_right,
              image_accessor,
              identity(),
              alpha_upper_left,
              alpha_accessor,
              identity());
          break;
        case SIGNED_INT_32:
          write_image_bands_and_alpha<Int32>(
              encoder.get(),
              image_upper_left,
              image_lower_right,
              image_accessor,
              identity(),
              alpha_upper_left,
              alpha_accessor,
              identity());
          break;
        case IEEE_FLOAT_32:
          write_image_bands_and_alpha<float>(
              encoder.get(),
              image_upper_left,
              image_lower_right,
              image_accessor,
              identity(),
              alpha_upper_left,
              alpha_accessor,
              identity());
          break;
        case IEEE_FLOAT_64:
          write_image_bands_and_alpha<double>(
              encoder.get(),
              image_upper_left,
              image_lower_right,
              image_accessor,
              identity(),
              alpha_upper_left,
              alpha_accessor,
              identity());
          break;
        default:
          vigra_fail("exportImageAlpha<non-scalar>: not reached");
      }
    }

    encoder->close();
  }

  template <
      class ImageIterator,
      class ImageAccessor,
      class AlphaIterator,
      class AlphaAccessor>
  static inline void exportImageAlpha(
      VIGRA_UNIQUE_PTR<vigra::Encoder>& encoder,
      ImageIterator image_upper_left,
      ImageIterator image_lower_right,
      ImageAccessor image_accessor,
      AlphaIterator alpha_upper_left,
      AlphaAccessor alpha_accessor,
      const vigra::ImageExportInfo& export_info) {
    typedef typename ImageAccessor::value_type ImageValueType;
    typedef typename vigra::NumericTraits<ImageValueType>::isScalar is_scalar;

    try {
      exportImageAlpha(
          encoder,
          image_upper_left,
          image_lower_right,
          image_accessor,
          alpha_upper_left,
          alpha_accessor,
          export_info,
          is_scalar());
    } catch (vigra::Encoder::TIFFCompressionException&) {
      vigra::ImageExportInfo info(export_info);

      info.setCompression("");
      exportImageAlpha(
          encoder,
          image_upper_left,
          image_lower_right,
          image_accessor,
          alpha_upper_left,
          alpha_accessor,
          info,
          is_scalar());
    }
  }

  template <
      class ImageIterator,
      class ImageAccessor,
      class AlphaIterator,
      class AlphaAccessor>
  static inline void exportImageAlpha(
      VIGRA_UNIQUE_PTR<vigra::Encoder>& encoder,
      vigra::triple<ImageIterator, ImageIterator, ImageAccessor> image,
      std::pair<AlphaIterator, AlphaAccessor> alpha,
      vigra::ImageExportInfo const& export_info) {
    exportImageAlpha(
        encoder,
        image.first,
        image.second,
        image.third,
        alpha.first,
        alpha.second,
        export_info);
  }

  // template<typename ImageType, typename AlphaType>
  void hmSaveRemapped(
      HmRemappedPanoImage<ImageType, AlphaType>& remapped,
      unsigned int imgNr,
      unsigned int /*nImg*/,
      const PanoramaOptions& opts,
      bool save_as_file,
      const std::string& basename,
      const bool useBigTIFF,
      AppBase::ProgressDisplay* progress) {
    ImageType* final_img = 0;
    AlphaType* alpha_img = 0;
    ImageType complete;
    vigra::BImage alpha;

    assert(!output_images_.at(imgNr));

    if (remapped.boundingBox().isEmpty())
      // do not save empty files...
      // TODO: need to tell other parts (enblend etc.) about it too!
      return;

    if (opts.outputMode == PanoramaOptions::OUTPUT_HDR) {
      // export alpha channel as gray channel (original pixel weights)
      std::ostringstream greyname;
      greyname << basename << std::setfill('0') << std::setw(4) << imgNr
               << "_gray.pgm";
      vigra::ImageExportInfo exinfo1(greyname.str().c_str());
      if (!opts.tiff_saveROI) {
        alpha.resize(opts.getROI().size());
        vigra::Rect2D newOutRect = remapped.boundingBox() & opts.getROI();
        vigra::Rect2D newOutArea(newOutRect);
        newOutRect.moveBy(-opts.getROI().upperLeft());
        vigra::copyImage(
            vigra_ext::applyRect(newOutArea, vigra_ext::srcMaskRange(remapped)),
            vigra_ext::applyRect(newOutRect, destImage(alpha)));
        vigra::exportImage(srcImageRange(alpha), exinfo1);
      } else {
        exinfo1.setPosition(remapped.boundingBox().upperLeft());
        exinfo1.setCanvasSize(vigra::Size2D(opts.getWidth(), opts.getHeight()));
        vigra::exportImage(srcImageRange(remapped.m_mask), exinfo1);
      }

      // calculate real alpha for saving with the image
      progress->setMessage("Calculating mask");
      remapped.calcAlpha();
    }

    if (!opts.tiff_saveROI) {
      // FIXME: this is stupid. Should not require space for full image...
      // but this would need a lower level interface in vigra impex
      complete.resize(opts.getROI().size());
      alpha.resize(opts.getROI().size());
      vigra::Rect2D newOutRect = remapped.boundingBox() & opts.getROI();
      vigra::Rect2D newOutArea(newOutRect);
      newOutRect.moveBy(-opts.getROI().upperLeft());
      vigra::copyImage(
          vigra_ext::applyRect(newOutArea, vigra_ext::srcImageRange(remapped)),
          vigra_ext::applyRect(newOutRect, destImage(complete)));

      vigra::copyImage(
          vigra_ext::applyRect(newOutArea, vigra_ext::srcMaskRange(remapped)),
          vigra_ext::applyRect(newOutRect, destImage(alpha)));
      final_img = &complete;
      alpha_img = &alpha;
    } else {
      final_img = &remapped.m_image;
      alpha_img = &remapped.m_mask;
    }

    std::string ext = opts.getOutputExtension();
    std::ostringstream filename;
    filename << basename << std::setfill('0') << std::setw(4) << imgNr
             << "." + ext;

    progress->setMessage("saving", hugin_utils::stripPath(filename.str()));

    vigra::ImageExportInfo exinfo(
        filename.str().c_str(), useBigTIFF ? "w8" : "w");
    exinfo.setXResolution(150);
    exinfo.setYResolution(150);
    exinfo.setICCProfile(remapped.m_ICCProfile);
    if (opts.tiff_saveROI) {
      exinfo.setPosition(remapped.boundingBox().upperLeft());
      exinfo.setCanvasSize(vigra::Size2D(opts.getWidth(), opts.getHeight()));
    } else {
      exinfo.setPosition(opts.getROI().upperLeft());
      exinfo.setCanvasSize(vigra::Size2D(opts.getWidth(), opts.getHeight()));
    }
    if (!opts.outputPixelType.empty()) {
      exinfo.setPixelType(opts.outputPixelType.c_str());
    }
    bool supportsAlpha = true;
    if (ext == "tif") {
      exinfo.setCompression(opts.tiffCompression.c_str());
    } else {
      if (ext == "jpg") {
        std::ostringstream quality;
        quality << "JPEG QUALITY=" << opts.quality;
        exinfo.setCompression(quality.str().c_str());
        supportsAlpha = false;
      };
    }

    if (supportsAlpha) {
      if (save_as_file) {
        VIGRA_UNIQUE_PTR<vigra::Encoder> encoder{nullptr};
        exportImageAlpha(
            encoder, srcImageRange(*final_img), srcImage(*alpha_img), exinfo);
      }
      VIGRA_UNIQUE_PTR<vigra::Encoder> encoder =
          std::make_unique<MatrixEncoderRGBA>();
      exportImageAlpha(
          encoder, srcImageRange(*final_img), srcImage(*alpha_img), exinfo);
      MatrixEncoderRGBA* matric_encoder_ptr =
          static_cast<MatrixEncoderRGBA*>(encoder.get());
      auto matrix_rgb = matric_encoder_ptr->consume();
      auto ul = remapped.boundingBox().upperLeft();
      matrix_rgb->set_xy_pos(ul.x, ul.y);
      output_images_.at(imgNr) = std::move(matrix_rgb);
    } else {
      vigra::exportImage(srcImageRange(*final_img), exinfo);
    };
  };

  /** save a remapped image, or layer */
  virtual void saveRemapped(
      HmRemappedPanoImage<ImageType, AlphaType>& remapped,
      unsigned int imgNr,
      unsigned int nImg,
      const PanoramaOptions& opts,
      const AdvancedOptions& advOptions,
      bool save_as_file = false) {
    hmSaveRemapped(
        remapped,
        imgNr,
        nImg,
        opts,
        save_as_file,
        m_basename,
        HuginBase::Nona::GetAdvancedOption(advOptions, "useBigTIFF", false),
        Base::m_progress);

    if (opts.saveCoordImgs) {
      vigra::UInt16Image xImg;
      vigra::UInt16Image yImg;

      Base::m_progress->setMessage("creating coordinate images");

      remapped.calcSrcCoordImgs(xImg, yImg);
      vigra::UInt16Image dist;
      if (!opts.tiff_saveROI) {
        dist.resize(opts.getWidth(), opts.getHeight());
        vigra::copyImage(
            srcImageRange(xImg),
            vigra_ext::applyRect(remapped.boundingBox(), destImage(dist)));
        std::ostringstream filename2;
        filename2 << m_basename << std::setfill('0') << std::setw(4) << imgNr
                  << "_x.tif";

        vigra::ImageExportInfo exinfo(filename2.str().c_str());
        exinfo.setXResolution(150);
        exinfo.setYResolution(150);
        exinfo.setCompression(opts.tiffCompression.c_str());
        vigra::exportImage(srcImageRange(dist), exinfo);

        vigra::copyImage(
            srcImageRange(yImg),
            vigra_ext::applyRect(remapped.boundingBox(), destImage(dist)));
        std::ostringstream filename3;
        filename3 << m_basename << std::setfill('0') << std::setw(4) << imgNr
                  << "_y.tif";

        vigra::ImageExportInfo exinfo2(filename3.str().c_str());
        exinfo2.setXResolution(150);
        exinfo2.setYResolution(150);
        exinfo2.setCompression(opts.tiffCompression.c_str());
        vigra::exportImage(srcImageRange(dist), exinfo2);
      } else {
        std::ostringstream filename2;
        filename2 << m_basename << std::setfill('0') << std::setw(4) << imgNr
                  << "_x.tif";
        vigra::ImageExportInfo exinfo(filename2.str().c_str());
        exinfo.setXResolution(150);
        exinfo.setYResolution(150);
        exinfo.setCompression(opts.tiffCompression.c_str());
        vigra::exportImage(srcImageRange(xImg), exinfo);
        std::ostringstream filename3;
        filename3 << m_basename << std::setfill('0') << std::setw(4) << imgNr
                  << "_y.tif";
        vigra::ImageExportInfo exinfo2(filename3.str().c_str());
        exinfo2.setXResolution(150);
        exinfo2.setYResolution(150);
        exinfo2.setCompression(opts.tiffCompression.c_str());
        vigra::exportImage(srcImageRange(yImg), exinfo2);
      }
    }
  }

  virtual void finalizeOutputFile(const PanoramaOptions& opts) {
    Base::m_progress->taskFinished();
  }

  // protected:
  //     std::string m_basename;
 private:
  std::vector<std::shared_ptr<MatrixRGB>> images_;
  std::vector<std::unique_ptr<MatrixRGB>> output_images_;
  std::vector<PanoramaOptions> mod_options_;
  std::size_t pass_{0};
  std::unique_ptr<Eigen::ThreadPool> thread_pool_;
};

template <typename ImageType, typename AlphaType>
std::unique_ptr<HmRemappedPanoImage<ImageType, AlphaType>> HmFileRemapper<
    ImageType,
    AlphaType>::
    getRemapped(
        const HuginBase::PanoramaData& pano,
        const HuginBase::PanoramaOptions& opts,
        unsigned int imgNr,
        const std::shared_ptr<MatrixRGB> image,
        vigra::Rect2D outputROI,
        Eigen::ThreadPool& gpu_thread_pool,
        AppBase::ProgressDisplay* progress) {
  typedef typename ImageType::value_type PixelType;

  // typedef typename vigra::NumericTraits<PixelType>::RealPromote RPixelType;
  //         typedef typename vigra::BasicImage<RPixelType> RImportImageType;
  typedef typename vigra::BasicImage<float> FlatImgType;

  // FlatImgType ffImg;
  // AlphaType srcAlpha;

  // choose image type...
  const HuginBase::SrcPanoImage& img = pano.getImage(imgNr);

  // vigra::Size2D destSize(opts.getWidth(), opts.getHeight());
  std::vector<std::unique_ptr<HmRemappedPanoImage<ImageType, AlphaType>>>
      m_remapped;
  if (m_remapped.size() <= imgNr) {
    m_remapped.resize(imgNr + 1);
  }
  m_remapped.at(imgNr) =
      std::make_unique<HmRemappedPanoImage<ImageType, AlphaType>>();

  // load image if necessary
  vigra::ImageImportInfo* info_ptr{nullptr};
  {
    std::unique_lock<std::mutex> lk(image_import_infos_mu_);
    assert(imgNr < 2);
    if (this->image_import_infos_.empty()) {
      this->image_import_infos_.resize(2);
    }
    if (!this->image_import_infos_.at(imgNr)) {
      //std::cout << "Reading file: " << img.getFilename() << std::endl;
      this->image_import_infos_[imgNr] =
          std::make_unique<vigra::ImageImportInfo>(img.getFilename().c_str());
      assert(this->image_import_infos_[imgNr]);
    }
    info_ptr = this->image_import_infos_.at(imgNr).get();
    assert(info_ptr);
  }
  vigra::ImageImportInfo& info = *info_ptr;

  int width = info.width();
  int height = info.height();

  if (opts.remapUsingGPU) {
    // Extend image width to multiple of 8 for fast GPU transfers.
    const int r = width % 8;
    if (r != 0)
      width += 8 - r;
  }

  std::unique_ptr<ImageType> src_img_ptr;
  bool has_vigra_image = false;
  if (image) {
    assert(width == image->cols());
    assert(height == image->rows());
    src_img_ptr = image->to_vigra_image();
    has_vigra_image = true;
  } else {
    src_img_ptr = std::make_unique<ImageType>(width, height);
  }
  ImageType& srcImg = *src_img_ptr;

  m_remapped.at(imgNr)->m_ICCProfile = info.getICCProfile();

  if (info.numExtraBands() > 0) {
    srcAlpha_.resize(width, height);
  }
  // int nb = info.numBands() - info.numExtraBands();
  bool alpha = info.numExtraBands() > 0;
  // std::string type = info.getPixelType();

  HuginBase::SrcPanoImage src = pano.getSrcImage(imgNr);

  // import the image
  progress->setMessage("loading", hugin_utils::stripPath(img.getFilename()));

  if (!image) {
    if (alpha) {
      vigra::importImageAlpha(
          info, vigra::destImage(srcImg), vigra::destImage(srcAlpha_));
    } else {
      vigra::importImage(info, vigra::destImage(srcImg));
    }
  }

  // check if the image needs to be scaled to 0 .. 1,
  // this only works for int -> float, since the image
  // has already been loaded into the output container
  double maxv = vigra_ext::getMaxValForPixelType(info.getPixelType());
  if (maxv != vigra_ext::LUTTraits<PixelType>::max()) {
    assert(false); // colivier: assumign not the case for quick remap
    double scale = ((double)vigra_ext::LUTTraits<PixelType>::max()) / maxv;
    // std::cout << "Scaling input image (pixel type: " << info.getPixelType()
    // << " with: " << scale << std::endl;
    transformImage(
        vigra::srcImageRange(srcImg),
        destImage(srcImg),
        vigra::functor::Arg1() * vigra::functor::Param(scale));
  }

  FlatImgType ffImg;
  // load flatfield, if needed.
  if (img.getVigCorrMode() & HuginBase::SrcPanoImage::VIGCORR_FLATFIELD) {
    assert(false); // colivier: assumign not the case for quick remap
    // load flatfield image.
    vigra::ImageImportInfo ffInfo(img.getFlatfieldFilename().c_str());
    progress->setMessage(
        "flatfield vignetting correction",
        hugin_utils::stripPath(img.getFilename()));
    vigra_precondition(
        (ffInfo.numBands() == 1),
        "flatfield vignetting correction: "
        "Only single channel flatfield images are supported\n");
    ffImg.resize(ffInfo.width(), ffInfo.height());
    vigra::importImage(ffInfo, vigra::destImage(ffImg));
  }
  m_remapped.at(imgNr)->setAdvancedOptions(
      HmSingleImageRemapper<ImageType, AlphaType>::get_advanced_options());
  // remap the image
  remapImage(
      srcImg,
      srcAlpha_,
      ffImg,
      pano.getSrcImage(imgNr),
      opts,
      outputROI,
      *m_remapped.at(imgNr),
      gpu_thread_pool,
      progress);
  return std::move(m_remapped.at(imgNr));
}

} // namespace hm
