#pragma once

#include "hockeymon/csrc/pytorch/image_blend.h"
#include "hockeymon/csrc/pytorch/image_remap.h"

#include <ATen/ATen.h>
#include <torch/torch.h>

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace hm {

namespace ops {

struct RemapImageInfo {
  std::size_t src_width{0};
  std::size_t src_height{0};
  at::ScalarType dtype;
  at::Tensor col_map;
  at::Tensor row_map;
  bool add_alpha_channel{false};
  std::size_t pad_value{0};
};

struct ImageStitcherConfig {
  std::size_t batch_size{1};
  std::vector<RemapImageInfo> remap_image_info;
  ImageBlender::Mode blender_mode{ImageBlender::Mode::Laplacian};
  bool half{false};
  std::size_t levels{6};
  at::Tensor seam;
  at::Tensor xor_map;
  bool lazy_init{false};
  std::optional<std::string> interpolation{std::nullopt};
};

struct StitchImageInfo {
  at::Tensor image;
  std::vector<int> xy_pos;
};

class ImageStitcher {
  using SizeRef = at::IntArrayRef;

 public:
  // levels=0 = quick, hard seam
  ImageStitcher(
      std::size_t batch_size,
      std::vector<RemapImageInfo> remap_image_info,
      ImageBlender::Mode blender_mode,
      bool half,
      std::size_t levels,
      at::Tensor seam,
      at::Tensor xor_map,
      bool lazy_init,
      std::optional<std::string> interpolation);
  void to(at::Device device);
  at::Tensor forward(std::vector<StitchImageInfo> inputs);

 private:
  at::ScalarType dtype_;
  std::vector<RemapImageInfo> remap_image_infos_;
  std::vector<std::unique_ptr<ImageRemapper>> remappers_;
  std::unique_ptr<ImageBlender> blender_;
  bool initialized_{false};
};

} // namespace ops
} // namespace hm
