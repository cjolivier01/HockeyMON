#pragma once

#include "hockeymon/csrc/pytorch/image_blend.h"
#include "hockeymon/csrc/pytorch/image_remap.h"
#include "hockeymon/csrc/pytorch/image_stitch.h"

#include <ATen/ATen.h>
#include <torch/torch.h>

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace hm {

namespace ops {

struct SmartImageStitcherConfig : public ImageStitcherConfig {
  bool minimize_blend{true};
};

class SmartImageStitcher {
  using SizeRef = at::IntArrayRef;

 public:
  // levels=0 = quick, hard seam
  SmartImageStitcher(
      std::size_t batch_size,
      std::vector<RemapImageInfo> remap_image_info,
      ImageBlender::Mode blender_mode,
      bool half,
      std::size_t levels,
      bool minimize_blend,
      at::Tensor seam,
      at::Tensor xor_map,
      bool lazy_init,
      std::optional<std::string> interpolation);
  void to(at::Device device);
  at::Tensor forward(std::vector<StitchImageInfo> inputs);

 private:
  std::unique_ptr<ImageStitcher> stitcher_;
  const bool minimize_blend_;
};

} // namespace ops
} // namespace hm
