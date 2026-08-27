#include "hockeymon/csrc/pytorch/image_utils.h"

#include <ATen/Functions.h>

#ifdef WITH_OPENCV
#include <opencv2/opencv.hpp>
#include <opencv2/core/opengl.hpp>
#else
#include <iostream>
#endif

namespace hm {

namespace ops {

constexpr bool is_channel(int c) {
  return c == 3 || c == 4;
}

std::int64_t image_width(const at::Tensor& t) {
  const int ndims = t.ndimension();
  TORCH_CHECK(
      ndims == 2 || (ndims == 4 && is_channel(t.size(1))),
      "Wrong numebr of dims to determine width, or not channels-first: ",
      ndims);
  return t.size(-1);
}

std::int64_t image_height(const at::Tensor& t) {
  const int ndims = t.ndimension();
  TORCH_CHECK(
      ndims == 2 || (ndims == 4 && is_channel(t.size(1))),
      "Wrong numebr of dims to determine height, or not channels-first: ",
      ndims);
  return t.size(-2);
}

std::array<std::int64_t, 2> image_size(const at::Tensor& t) {
  const int ndims = t.ndimension();
  TORCH_CHECK(
      ndims == 2 || (ndims == 4 && is_channel(t.size(1))),
      "Wrong numebr of dims to determine height, or not channels-first: ",
      ndims);
  return std::array<std::int64_t, 2>{t.size(-2), t.size(-1)};
}

// void show_gpu_tensor(const at::Tensor& gpu_tensor) {
//   // Create OpenGL window
//   cv::namedWindow("GPU Tensor", cv::WINDOW_OPENGL);

//   // Create OpenGL buffer
//   cv::ogl::Texture2D tex;
//   tex.create(gpu_tensor.size(2), gpu_tensor.size(1), CV_32FC3);

//   // Convert tensor format if needed (C,H,W -> H,W,C)
//   auto display_tensor = gpu_tensor.permute({1, 2, 0});

//   // Copy GPU tensor directly to OpenGL texture
//   cv::cuda::GpuMat gpu_mat(
//       display_tensor.size(0),
//       display_tensor.size(1),
//       CV_32FC3,
//       display_tensor.data_ptr());

//   tex.copyFrom(gpu_mat);

//   // Display
//   cv::ogl::render(tex);
//   cv::waitKey(1); // Non-blocking display
// }

at::Tensor make_channels_last(at::Tensor t) {
  switch (t.ndimension()) {
    case 3: {
      int cl = t.size(0), cr = t.size(2);
      if (is_channel(cl) && !is_channel(cr)) {
        t = t.permute({1, 2, 0});
      }
    } break;
    case 4: {
      int cl = t.size(1), cr = t.size(3);
      if (is_channel(cl) && !is_channel(cr)) {
        t = t.permute({0, 2, 3, 1});
      }
    } break;
    default:
      break;
  }
  return t;
}

// Scale tensor while maintaining aspect ratio
at::Tensor simple_scale_image(const at::Tensor& img, float scale) {
  int orig_h = image_height(img);
  int orig_w = image_width(img);

  int new_h = static_cast<int>(orig_h * scale);
  int new_w = static_cast<int>(orig_w * scale);
  assert(img.ndimension() == 4); // has batch dim
  if (!new_h || !new_w) {
    return img;
  }
  if (new_h < orig_h) {
    return at::adaptive_avg_pool2d(img, {new_h, new_w});
  } else {
    return at::upsample_bilinear2d(img, {new_h, new_w}, false);
  }
}

#ifdef NDEBUG
void show_image(const at::Tensor& tensor, bool wait = false) {}
#else
#ifdef WITH_OPENCV
void show_image(
    const std::string& label,
    at::Tensor tensor,
    bool wait,
    std::optional<float> scale) {
  if (scale.has_value() && scale.value() != 00 && scale.value() != 1.0) {
    tensor = simple_scale_image(tensor, scale.value());
  }
  // Ensure tensor is on CPU and correct format
  // std::cout << tensor.sizes() << std::endl;
  at::Tensor t = make_channels_last(tensor.detach());
  // std::cout << t.sizes() << std::endl;
  if (t.ndimension() == 4) {
    // assert batch size of one
    assert(t.size(0) == 1);
    t = t.squeeze(0);
  }
  if (at::is_floating_point(t)) {
    t = t.clamp(0.0, 255.0).to(at::TensorOptions().dtype(at::ScalarType::Byte));
  }
  t = t.contiguous().cpu();

  std::cout << t.sizes() << std::endl;

  // Convert to OpenCV Mat
  cv::Mat image(t.size(0), t.size(1), CV_8UC3);
  std::memcpy(image.data, t.data_ptr(), sizeof(uint8_t) * t.numel());

  // Convert RGB to BGR (OpenCV default)
  // cv::cvtColor(image, image, cv::COLOR_RGB2BGR);

  // Display
  cv::imshow(label.c_str(), image);
  cv::waitKey(wait ? 0 : 1);
}
#else
void show_image(const at::Tensor& tensor, bool wait = false) {
  std::cout << "Can't show image when OpenCV is disabled" << std::endl;
}
#endif
#endif

} // namespace ops
} // namespace hm
