#pragma once

#include <ATen/ATen.h>
#include <torch/nn/functional.h>
#include <torch/torch.h>

#include <optional>
#include <string>

namespace hm {

namespace ops {

class ImageBlender {
  using SizeRef = at::IntArrayRef;

 public:
  enum class Mode { HardSeam, Laplacian };

  // levels=0 = quick, hard seam
  ImageBlender(
      Mode mode,
      bool half,
      std::size_t levels,
      at::Tensor seam,
      at::Tensor xor_map,
      bool lazy_init,
      std::optional<std::string> interpolation);
  void to(at::Device device);
  at::Tensor forward(
      at::Tensor&& image_1,
      const std::vector<int>& xy_pos_1,
      at::Tensor&& image_2,
      const std::vector<int>& xy_pos_2);

 private:
  void init();

  std::pair<at::Tensor, at::Tensor> make_full(
      const at::Tensor& image_1,
      const at::Tensor& image_2,
      std::size_t level) const;

  at::Tensor hard_seam_blend(
      at::Tensor&& image_1,
      const std::vector<int>& xy_pos_1,
      at::Tensor&& image_2,
      const std::vector<int>& xy_pos_2) const;

  at::Tensor laplacian_pyramid_blend(
      at::Tensor&& image_1,
      const std::vector<int>& xy_pos_1,
      at::Tensor&& image_2,
      const std::vector<int>& xy_pos_2);

  at::Tensor downsample(const at::Tensor& x);
  at::Tensor upsample(at::Tensor x, const SizeRef size) const;
  std::vector<at::Tensor> create_laplacian_pyramid(
      at::Tensor& x,
      at::Tensor& kernel);
  at::Tensor one_level_gaussian_pyramid(at::Tensor& x, at::Tensor& kernel);
  void create_masks();
  void build_coordinate_system(
      const at::Tensor& image_1,
      const std::vector<int>& xy_pos_1,
      const at::Tensor& image_2,
      const std::vector<int>& xy_pos_2);

  at::Tensor blend(
      at::Tensor&& left_small_gaussian_blurred,
      at::Tensor&& right_small_gaussian_blurred,
      int level);

  const Mode mode_;
  at::ScalarType dtype_;
  std::size_t levels_;
  const std::size_t num_images_{2};

  at::Tensor seam_;
  at::Tensor xor_map_;
  at::Tensor left_seam_value_;
  at::Tensor condition_left_;
  at::Tensor right_seam_value_;
  at::Tensor condition_right_;
  std::vector<at::Tensor> seam_masks_;
  std::string interpolation_;

  struct ImageSize {
    std::int64_t w{0};
    std::int64_t h{0};
    ImageSize operator/(std::int64_t q) {
      return ImageSize{
          .w = w / q,
          .h = h / q,
      };
    }
  };
  struct AInfo {
    std::int64_t h{0}, w{0}, x{0}, y{0};
    AInfo operator/(std::int64_t q) {
      return AInfo{
          .h = h / q,
          .w = w / q,
          .x = x / q,
          .y = y / q,
      };
    }
  };
  std::vector<std::vector<AInfo>> ainfos_;
  std::vector<ImageSize> level_canvas_dims_;
  bool make_all_full_first_{true};

  // Laplacian pyramid persistent tensors
  at::Tensor gussian_kernel_;
  at::Tensor mask_gussian_kernel_;
  torch::nn::AvgPool2d avg_pooling_;
  std::vector<at::Tensor> mask_small_gaussian_blurred_;

  bool lazy_init_;
  bool initialized_{false};
};

} // namespace ops
} // namespace hm
