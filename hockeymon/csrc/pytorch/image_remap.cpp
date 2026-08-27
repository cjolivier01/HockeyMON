#include "hockeymon/csrc/pytorch/image_remap.h"

#include <torch/nn/functional.h>

#include <iostream>

namespace hm {
namespace ops {

namespace {

// #define DO_EXTRA_PADDING

constexpr std::int64_t kUnmappedPixelValue = 65535;
#ifdef DO_EXTRA_PADDING
template <typename VALUE_TYPE>
at::Tensor pad_tensor_to_size(
    at::Tensor& tensor,
    std::size_t target_width,
    std::size_t target_height,
    const VALUE_TYPE& pad_value) {
  std::size_t pad_width, pad_height;
  if (tensor.sizes().size() == 2) {
    pad_height = target_height - tensor.size(0);
    pad_width = target_width - tensor.size(1);
  } else {
    assert(tensor.sizes().size() == 3);
    pad_height = target_height - tensor.size(1);
    pad_width = target_width - tensor.size(2);
  }
  pad_height = std::max(0UL, pad_height);
  pad_width = std::max(0UL, pad_width);
  if (pad_width || pad_height) {
    return at::constant_pad_nd(
        tensor, {0, (int)pad_width, 0, (int)pad_height}, pad_value);
  }
  return tensor;
}

template <typename VALUE_TYPE>
at::Tensor pad_tensor_to_size_batched(
    at::Tensor& tensor,
    std::size_t target_width,
    std::size_t target_height,
    const VALUE_TYPE& pad_value) {
  std::size_t pad_width, pad_height;
  if (tensor.sizes().size() == 3) {
    pad_height = target_height - tensor.size(1);
    pad_width = target_width - tensor.size(2);
  } else {
    assert(tensor.sizes().size() == 4);
    pad_height = target_height - tensor.size(2);
    pad_width = target_width - tensor.size(3);
  }
  pad_height = std::max(0UL, pad_height);
  pad_width = std::max(0UL, pad_width);
  if (pad_width || pad_height) {
    return at::constant_pad_nd(
        tensor, {0, (int)pad_width, 0, (int)pad_height}, pad_value);
  }
  return tensor;
}
#endif
} // namespace

ImageRemapper::ImageRemapper(
    std::size_t src_width,
    std::size_t src_height,
    at::Tensor col_map,
    at::Tensor row_map,
    at::ScalarType dtype,
    bool add_alpha_channel,
    std::size_t pad_value,
    std::optional<std::string> interpolation)
    : src_width_(src_width),
      src_height_(src_height),
      dest_width_(col_map.size(1)),
      dest_height_(col_map.size(0)),
      col_map_(col_map),
      row_map_(row_map),
      dtype_(dtype),
      add_alpha_channel_(add_alpha_channel),
      pad_value_(pad_value),
      interpolation_(interpolation ? *interpolation : "") {
  working_width_ = src_width_;
  working_height_ = src_height_;
}

void ImageRemapper::init(std::size_t batch_size) {
  if (initialized_) {
    return;
  }
  assert(!initialized_);
  initialized_ = false;
#ifdef DO_EXTRA_PADDING
  // std::cout << "Padding tensors to size: " << working_width_ << " x "
  //           << working_height_ << std::endl;
  auto col_map = pad_tensor_to_size(
      col_map_, working_width_, working_height_, kUnmappedPixelValue);
  auto row_map = pad_tensor_to_size(
      row_map_, working_width_, working_height_, kUnmappedPixelValue);
#else
  auto col_map = col_map_;
  auto row_map = row_map_;
#endif
  at::Tensor mask = at::logical_or(
      col_map == kUnmappedPixelValue, row_map == kUnmappedPixelValue);
  col_map.index_put_({mask}, 0);
  row_map.index_put_({mask}, 0);
  if (!interpolation_.empty()) {
    // Normalize to [-1, 1]
    at::Tensor row_map_normalized =
        (2.0 * row_map / ((float)working_height_ - 1.0)) - 1.0;
    at::Tensor col_map_normalized =
        (2.0 * col_map / ((float)working_width_ - 1.0)) - 1.0;
    // Create the grid for grid_sample
    auto grid = at::stack({col_map_normalized, row_map_normalized}, /*dim=*/-1);
    assert(grid.sizes().size() == 3);
    grid = grid.expand(
        {(int)batch_size, grid.size(0), grid.size(1), grid.size(2)});
    grid_ = grid.to(at::TensorOptions(dtype_)).contiguous();
  }
  col_map_ = col_map.contiguous();
  row_map_ = row_map.contiguous();
  mask_ = mask.contiguous();
  if (add_alpha_channel_) {
    at::TensorOptions options;
    options = options.dtype(dtype_);
    alpha_channel_ = at::empty(
        {(int)batch_size, 1, (int)working_height_, (int)working_width_},
        options);
    if (torch::is_floating_point(alpha_channel_)) {
      alpha_channel_.fill_(1.0);
    } else {
      alpha_channel_.fill_(255);
    }
    alpha_channel_.index_put_(
        {at::indexing::Slice(), at::indexing::Slice(), mask_}, (uint8_t)0);
  }
  initialized_ = true;
}

void ImageRemapper::to(at::Device device) {
  assert(initialized_);
  col_map_ = col_map_.to(device);
  row_map_ = row_map_.to(device);
  mask_ = mask_.to(device);
  if (grid_.defined()) {
    grid_ = grid_.to(device);
  }
  if (alpha_channel_.defined()) {
    alpha_channel_ = alpha_channel_.to(device);
  }
}

at::Tensor ImageRemapper::forward(at::Tensor source_tensor) const {
  assert(initialized_);
  TORCH_CHECK(
      source_tensor.dtype() == dtype_,
      "Current remapper dtype is ",
      dtype_,
      " but the source tensor dtype is ",
      source_tensor.dtype());
  TORCH_CHECK(
      source_tensor.size(1) == 3 || source_tensor.size(2) == 4,
      "Tensor format should be [batch, channels, height, width");
#ifdef DO_EXTRA_PADDING
  source_tensor = pad_tensor_to_size_batched(
      source_tensor, working_width_, working_height_, 0);
#endif
  // batch + 3 channels
  TORCH_CHECK(source_tensor.sizes().size() == 4, "Expected a 4D tensor");
  at::Tensor destination_tensor;
  if (interpolation_.empty()) {
    destination_tensor = at::empty_like(source_tensor);
    destination_tensor.index_put_(
        {at::indexing::Slice(), at::indexing::Slice()},
        source_tensor.index(
            {at::indexing::Slice(),
             at::indexing::Slice(),
             row_map_,
             col_map_}));
  } else {
    torch::nn::functional::GridSampleFuncOptions options;
    assert(interpolation_ == "bilinear"); // TODO: implement enum switch
    auto mode = torch::kBilinear;
    options =
        options.padding_mode(torch::kZeros).align_corners(false).mode(mode);
    if (!torch::is_floating_point(source_tensor)) {
      source_tensor = source_tensor.to(
          at::TensorOptions().dtype(dtype_), /*non_blocking=*/true);
    }
    destination_tensor =
        torch::nn::functional::grid_sample(source_tensor, grid_, options);
    if (destination_tensor.dtype() != dtype_) {
      if (dtype_ == at::ScalarType::Byte) {
        destination_tensor = destination_tensor.clamp(0, 255.0).to(
            torch::kByte, /*non_blocking=*/true);
      } else {
        destination_tensor = destination_tensor.to(
            at::TensorOptions().dtype(dtype_), /*non_blocking=*/true);
      }
    }
  }
  destination_tensor.index_put_(
      {torch::indexing::Slice(), torch::indexing::Slice(), mask_}, (std::uint8_t)pad_value_);
  // Add alpha channel if necessary
  if (add_alpha_channel_) {
    destination_tensor =
        at::cat({destination_tensor, alpha_channel_}, /*dim=*/1);
  }
#ifdef DO_EXTRA_PADDING
  // Clip to the original size that was specified
  destination_tensor = destination_tensor.index(
      {torch::indexing::Slice(),
       torch::indexing::Slice(),
       torch::indexing::Slice(0, dest_height_),
       torch::indexing::Slice(0, dest_width_)});
#endif
  return destination_tensor;
}

} // namespace ops
} // namespace hm
