#include "hockeymon/csrc/pytorch/image_stitch.h"

#include <torch/torch.h>

namespace hm {
namespace ops {

ImageStitcher::ImageStitcher(
    std::size_t batch_size,
    std::vector<RemapImageInfo> remap_image_info,
    ImageBlender::Mode blender_mode,
    bool half,
    std::size_t levels,
    at::Tensor seam,
    at::Tensor xor_map,
    bool lazy_init,
    std::optional<std::string> interpolation)
    : dtype_(half ? at::ScalarType::Half : at::ScalarType::Float),
      remap_image_infos_(std::move(remap_image_info)) {
  for (const RemapImageInfo& remap_info : remap_image_infos_) {
    remappers_.emplace_back(std::make_unique<ImageRemapper>(
        remap_info.src_width,
        remap_info.src_height,
        remap_info.col_map,
        remap_info.row_map,
        dtype_,
        remap_info.add_alpha_channel,
        remap_info.pad_value,
        interpolation));
    (*remappers_.rbegin())->init(batch_size);
  }
  blender_ = std::make_unique<ImageBlender>(
      blender_mode,
      half,
      levels,
      seam,
      xor_map,
      /*lazy_init=*/true,
      interpolation);
}

void ImageStitcher::to(at::Device device) {
  for (auto& m : remappers_) {
    m->to(device);
  }
  blender_->to(device);
}

at::Tensor ImageStitcher::forward(std::vector<StitchImageInfo> inputs) {
  if (!initialized_) {
    int batch_size = inputs.at(0).image.size(0);
    for (auto& r : remappers_) {
      r->init(batch_size);
    }
  }
  for (auto& t : inputs) {
    TORCH_CHECK(
        torch::is_floating_point(t.image), "Inputs must be floating point");
  }
  std::vector<at::Tensor> blend_inputs;
  blend_inputs.reserve(remappers_.size());
  for (std::size_t i = 0, n = remappers_.size(); i < n; ++i) {
    StitchImageInfo& img_info = inputs.at(i);
    blend_inputs.emplace_back(remappers_.at(i)->forward(img_info.image));
  }
  at::Tensor stitched_tensor = blender_->forward(
      std::move(blend_inputs.at(0)),
      inputs.at(0).xy_pos,
      std::move(blend_inputs.at(1)),
      inputs.at(1).xy_pos);

  return stitched_tensor;
}

} // namespace ops
} // namespace hm
