#include "hockeymon/csrc/pytorch/smart_image_stitch.h"

namespace hm {

namespace ops {

SmartImageStitcher::SmartImageStitcher(
    std::size_t batch_size,
    std::vector<RemapImageInfo> remap_image_info,
    ImageBlender::Mode blender_mode,
    bool half,
    std::size_t levels,
    bool minimize_blend,
    at::Tensor seam,
    at::Tensor xor_map,
    bool lazy_init,
    std::optional<std::string> interpolation)
    : stitcher_(std::make_unique<ImageStitcher>(
          batch_size,
          std::move(remap_image_info),
          std::move(blender_mode),
          half,
          levels,
          std::move(seam),
          std::move(xor_map),
          lazy_init,
          std::move(interpolation))),
      minimize_blend_(minimize_blend) {}

at::Tensor SmartImageStitcher::forward(std::vector<StitchImageInfo> inputs) {
  return stitcher_->forward(std::move(inputs));
}

} // namespace ops
} // namespace hm
