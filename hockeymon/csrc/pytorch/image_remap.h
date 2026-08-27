#pragma once

#include <ATen/ATen.h>

#include <optional>
#include <string>

namespace hm {

// template <typename T>
// using Optional = std::optional<T>;

namespace ops {

struct RemapperConfig {
  std::size_t src_width{0};
  std::size_t src_height{0};
  int x_pos{0};
  int y_pos{0};
  at::Tensor col_map;
  at::Tensor row_map;
  at::ScalarType dtype{at::ScalarType::Byte};
  bool add_alpha_channel{false};
  std::uint8_t pad_value{0};
  std::string interpolation;
  std::size_t batch_size{1};
  std::string device{"cpu"};
};

class ImageRemapper {
 public:
  ImageRemapper(
      std::size_t src_width,
      std::size_t src_height,
      at::Tensor col_map,
      at::Tensor row_map,
      at::ScalarType dtype,
      bool add_alpha_channel,
      std::size_t pad_value,
      std::optional<std::string> interpolation);
  void init(std::size_t batch_size);
  void to(at::Device device);
  bool is_initialized() const {
    return initialized_;
  }
  at::Tensor forward(at::Tensor source_tensor) const;

 private:
  bool initialized_{false};
  std::size_t src_width_;
  std::size_t src_height_;
  std::size_t dest_width_;
  std::size_t dest_height_;
  std::size_t working_width_{0};
  std::size_t working_height_{0};
  std::size_t pad_value_{0};
  at::Tensor col_map_;
  at::Tensor row_map_;
  at::Tensor mask_;
  at::Tensor grid_;
  at::Tensor alpha_channel_;
  at::ScalarType dtype_;
  bool add_alpha_channel_;
  std::string interpolation_;
};

} // namespace ops
} // namespace hm
