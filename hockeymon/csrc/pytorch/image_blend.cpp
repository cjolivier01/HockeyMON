#include "hockeymon/csrc/pytorch/image_blend.h"
#include "hockeymon/csrc/pytorch/image_utils.h"

#include <torch/nn/functional.h>

namespace hm {
namespace ops {

namespace {

bool verbose = false;

constexpr std::size_t kMinPixelSizeForImage = 16;

void prettyPrintTensor(const torch::Tensor& tensor) {
  auto sizes = tensor.sizes();
  int ndim = tensor.dim();

  // Check if the tensor is empty
  if (ndim == 0) {
    std::cout << "Tensor is empty" << std::endl;
    return;
  }

  // Check if the tensor is a scalar
  if (ndim == 1 && sizes[0] == 1) {
    std::cout << tensor.item<float>() << std::endl;
    return;
  }

  // Check if the tensor is a 1D vector
  if (ndim == 1) {
    std::cout << "[ ";
    for (int i = 0; i < sizes[0]; ++i) {
      std::cout << tensor[i].item<float>();
      if (i < sizes[0] - 1)
        std::cout << ", ";
    }
    std::cout << " ]" << std::endl;
    return;
  }

  // For tensors with more dimensions
  for (int i = 0; i < ndim; ++i) {
    std::cout << "Dimension " << i << ": " << sizes[i] << " elements"
              << std::endl;
    int sz_at_dim = sizes[i];
    for (int j = 0; j < sz_at_dim; ++j) {
      std::cout << "  ";
      prettyPrintTensor(tensor[j]);
    }
  }
}

torch::Tensor gaussian_conv2d(torch::Tensor x, const torch::Tensor& g_kernel) {
  // Check if x has a dtype different from torch::kUInt8
  if (x.dtype() == torch::kUInt8) {
    throw std::runtime_error("Input tensor cannot have dtype torch::kUInt8");
  }

  // Infer depth automatically based on the shape of g_kernel
  int64_t channels = g_kernel.size(0);
  int64_t padding =
      g_kernel.size(-1) / 2; // Kernel size needs to be an odd number

  // Check if x has the expected shape (batch, depth, height, width)
  if (x.dim() != 4) {
    throw std::runtime_error(
        "Expected input tensor to be of shape: (batch, depth, height, width)");
  }

  // Perform the convolution operation
  torch::Tensor y = torch::conv2d(
      /*input=*/x,
      /*weight=*/g_kernel,
      /*bias=*/c10::nullopt,
      /*stride=*/1,
      /*padding=*/padding,
      /*dilation=*/1,
      /*groups=*/channels);

  return y;
}

int constrain_index(int max_index, int calculated_index) {
  TORCH_CHECK(calculated_index <= max_index, "Calculated index is too large");
  return calculated_index;
}

inline at::Tensor scalar_float(const float& val, at::ScalarType dtype) {
  return torch::tensor({val}, torch::TensorOptions().dtype(dtype));
}

at::Tensor create_gaussian_kernel(
    int kernel_size,
    int channels,
    float sigma,
    at::ScalarType dtype) {
  // Create Gaussian Kernel. In Numpy
  at::Tensor ax = at::linspace(
      -double(kernel_size - 1) / 2.0,
      double(kernel_size - 1) / 2.0,
      kernel_size,
      torch::TensorOptions().dtype(at::ScalarType::Float));
  auto xx_and_yy = at::meshgrid({ax, ax}, /*indexing=*/"ij");
  at::Tensor& xx = xx_and_yy.at(0);
  at::Tensor& yy = xx_and_yy.at(1);
  at::Tensor kernel_tensor = at::exp(
      -0.5 * (at::square(xx) + at::square(yy)) /
      at::square(scalar_float(sigma, at::ScalarType::Float)));
  kernel_tensor /= at::sum(kernel_tensor);
  // # Reshapes to (channels, 1, size, size)
  kernel_tensor = kernel_tensor.repeat({channels, 1, 1, 1});
  if (kernel_tensor.dtype() != dtype) {
    kernel_tensor = kernel_tensor.to(torch::TensorOptions(dtype));
  }
  return kernel_tensor;
}

} // namespace

ImageBlender::ImageBlender(
    Mode mode,
    bool half,
    std::size_t levels,
    at::Tensor seam,
    at::Tensor xor_map,
    bool lazy_init,
    std::optional<std::string> interpolation)
    : mode_(mode),
      dtype_(half ? at::ScalarType::Half : at::ScalarType::Float),
      levels_(levels),
      seam_(seam),
      xor_map_(xor_map),
      interpolation_(interpolation ? *interpolation : ""),
      avg_pooling_(torch::nn::AvgPool2dOptions(/*kernel_size=*/2)),
      lazy_init_(lazy_init) {
  if (!lazy_init_) {
    init();
  }
}

void ImageBlender::init() {
  initialized_ = false;
  TORCH_CHECK(
      seam_.ndimension() == 2,
      "Seam tensor should be two dimensions only (h, w)");
  auto unique_results = at::_unique(seam_, /*sorted=*/true);
  torch::Tensor unique_elements = std::get<0>(unique_results);
  at::Device current_device = unique_elements.device();
  assert(unique_elements.size(0) == 2);
  assert(unique_elements.dim() == 1);
  left_seam_value_ = unique_elements[0];
  right_seam_value_ = unique_elements[1];

  gussian_kernel_ = create_gaussian_kernel(
                        /*kernel_size=*/5,
                        /*channels=*/3,
                        /*sigma=*/1.0,
                        /*dtype=*/dtype_)
                        .to(seam_.device());
  mask_gussian_kernel_ = create_gaussian_kernel(
                             /*kernel_size=*/5,
                             /*channels=*/1,
                             /*sigma=*/1.0,
                             /*dtype=*/dtype_)
                             .to(seam_.device());

  if (mode_ == Mode::Laplacian) {
    create_masks();
  } else {
    condition_left_ = at::eq(seam_, left_seam_value_);
    condition_right_ = at::eq(seam_, right_seam_value_);
  }

  initialized_ = true;
}

void ImageBlender::create_masks() {
  int canvas_h = seam_.size(0);
  int canvas_w = seam_.size(1);
  // The canvas width pyramid
  level_canvas_dims_.reserve(levels_);
  level_canvas_dims_.clear();
  level_canvas_dims_.emplace_back(ImageSize{.w = canvas_w, .h = canvas_h});
  for (std::size_t l = 0; l < levels_; l++) {
    ImageSize next_dim = *level_canvas_dims_.rbegin() / 2;
    if (next_dim.w < kMinPixelSizeForImage ||
        next_dim.h < kMinPixelSizeForImage) {
      levels_ = l;
      break; // should it be levels_ - 1?
    }
    level_canvas_dims_.emplace_back(next_dim);
  }

  // Now on to the masks...
  at::Tensor mask = seam_.unsqueeze(0).unsqueeze(0);

  at::Tensor condition_left =
      at::eq(seam_, left_seam_value_).unsqueeze(0).unsqueeze(0);
  at::Tensor condition_right =
      at::eq(seam_, right_seam_value_).unsqueeze(0).unsqueeze(0);

  mask.index_put_({condition_left}, 1);
  mask.index_put_({condition_right}, 0);
  mask = mask.to(dtype_);
  at::Tensor mask_img = mask.clone();
  mask_small_gaussian_blurred_ = {mask.squeeze(0).squeeze(0)};
  for (int l = 0; l < levels_ + 1; ++l) {
    mask_img = one_level_gaussian_pyramid(mask_img, mask_gussian_kernel_);
    mask_small_gaussian_blurred_.emplace_back(mask_img.squeeze(0).squeeze(0));
  }
  for (int i = 0; i < mask_small_gaussian_blurred_.size(); ++i) {
    at::Tensor max_mask_val = at::max(mask_small_gaussian_blurred_[i]);
    // std::cout << "mask[" << i
    //           << "] min = " <<
    //           at::min(mask_small_gaussian_blurred_[i]).item()
    //           << ", max = " <<
    //           at::max(mask_small_gaussian_blurred_[i]).item()
    //           << std::endl;
    mask_small_gaussian_blurred_[i] =
        mask_small_gaussian_blurred_[i] / max_mask_val;
    // std::cout << "mask[" << i
    //           << "] min = " <<
    //           at::min(mask_small_gaussian_blurred_[i]).item()
    //           << ", max = " <<
    //           at::max(mask_small_gaussian_blurred_[i]).item()
    //           << std::endl;
  }
}

void ImageBlender::to(at::Device device) {
  seam_ = seam_.to(device);
  xor_map_ = xor_map_.to(device);
  if (initialized_) {
    left_seam_value_ = left_seam_value_.to(device);
    right_seam_value_ = right_seam_value_.to(device);
    gussian_kernel_ = gussian_kernel_.to(device);
    mask_gussian_kernel_ = mask_gussian_kernel_.to(device);
    if (mode_ == Mode::Laplacian) {
      avg_pooling_->to(device);
      for (std::size_t i = 0, n = mask_small_gaussian_blurred_.size(); i < n;
           ++i) {
        mask_small_gaussian_blurred_[i] =
            mask_small_gaussian_blurred_[i].to(device);
      }
    } else {
      condition_left_ = condition_left_.to(device);
      condition_right_ = condition_right_.to(device);
    }
  }
}

at::Tensor ImageBlender::downsample(const at::Tensor& x) {
  return avg_pooling_->forward(x);
}

at::Tensor ImageBlender::upsample(at::Tensor x, const SizeRef size) const {
  return torch::upsample_bilinear2d(x, size, /*align_corners=*/false);
}

std::vector<at::Tensor> ImageBlender::create_laplacian_pyramid(
    at::Tensor& x,
    at::Tensor& kernel) {
  std::vector<at::Tensor> pyramids;
  at::Tensor current_x = x;
  for (int level = 0; level < levels_; ++level) {
    at::Tensor gauss_filtered_x = gaussian_conv2d(current_x, kernel);
    at::Tensor down = downsample(gauss_filtered_x);
    at::Tensor laplacian = current_x -
        upsample(down,
                 {gauss_filtered_x.size(gauss_filtered_x.dim() - 2),
                  gauss_filtered_x.size(gauss_filtered_x.dim() - 1)});
    pyramids.emplace_back(laplacian);
    current_x = down;
  }
  pyramids.emplace_back(current_x);
  return pyramids;
}

at::Tensor ImageBlender::one_level_gaussian_pyramid(
    at::Tensor& x,
    at::Tensor& kernel) {
  at::Tensor gauss_filtered_x;
  gauss_filtered_x = gaussian_conv2d(x, kernel);
  // std::cout << "NEW gauss_filtered_x: min=" <<
  // at::min(gauss_filtered_x).item()
  //           << ", max=" << at::max(gauss_filtered_x) << std::endl;
  return downsample(gauss_filtered_x);
}

std::pair<at::Tensor, at::Tensor> ImageBlender::make_full(
    const at::Tensor& image_1,
    const at::Tensor& image_2,
    std::size_t level) const {
  assert(image_1.dim() == 4);
  assert(image_1.size(1) == 3 || image_1.size(0) == 4);

  const AInfo& ainfo_1 = ainfos_.at(level).at(0);
  const AInfo& ainfo_2 = ainfos_.at(level).at(1);

  int h1 = ainfo_1.h;
  int w1 = ainfo_1.w;
  int x1 = ainfo_1.x;
  int y1 = ainfo_1.y;
  int h2 = ainfo_2.h;
  int w2 = ainfo_2.w;
  int x2 = ainfo_2.x;
  int y2 = ainfo_2.y;

  const auto& canvas_level = level_canvas_dims_.at(level);
  int canvas_w = canvas_level.w;
  int canvas_h = canvas_level.h;

  if (verbose) {
    std::cout << "\nCanvas size=[" << canvas_h << ", " << canvas_w << "]"
              << std::endl;
    std::cout << "image_1 size=" << image_1.sizes()
              << "\nimage_2 size=" << image_2.sizes() << std::endl;
  }

  TORCH_CHECK(x1 == 0 || x2 == 0, "Images not aligned to left edge of canvas");
  TORCH_CHECK(y1 == 0 || y2 == 0, "Images not aligned to top edge of canvas");
  TORCH_CHECK(x1 + w1 <= canvas_w, "First image overflows the canvas width");
  TORCH_CHECK(y1 + h1 <= canvas_h, "First image overflows the canvas height");
  TORCH_CHECK(x2 + w2 <= canvas_w, "Second image overflows the canvas width");
  TORCH_CHECK(y2 + h2 <= canvas_h, "Second image overflows the canvas height");

  TORCH_CHECK(x1 <= w1, "Invalid x1: " + std::to_string(x1));
  TORCH_CHECK(x1 <= h1, "Invalid y1: " + std::to_string(y1));
  TORCH_CHECK(x2 <= w2, "Invalid x2:" + std::to_string(x2));
  TORCH_CHECK(y2 <= h2, "Invalid y2: " + std::to_string(y2));

#if 0
  // This way is slower
  at::Tensor full_left = at::zeros(
      {image_1.size(0), image_1.size(1), canvas_h, canvas_w},
      at::TensorOptions().dtype(image_1.dtype()).device(image_1.device()));

  full_left.index_put_(
      {torch::indexing::Slice(),
       torch::indexing::Slice(),
       torch::indexing::Slice(y1, y1 + h1 + y1),
       torch::indexing::Slice(x1, x1 + w1)},
      image_1);

  at::Tensor full_right = at::zeros(
      {image_1.size(0), image_1.size(1), canvas_h, canvas_w},
      at::TensorOptions().dtype(image_1.dtype()).device(image_1.device()));

  full_right.index_put_(
      {torch::indexing::Slice(),
       torch::indexing::Slice(),
       torch::indexing::Slice(y2, y2 + h2 + y2),
       torch::indexing::Slice(x2, x2 + w2)},
      image_2);

#else
  at::Tensor full_left = at::constant_pad_nd(
      image_1,
      {
          x1,
          constrain_index(canvas_w, canvas_w - (w1 + x1)),
          y1,
          constrain_index(canvas_h, canvas_h - (h1 + y1)),
      },
      0.0);

  at::Tensor full_right = at::constant_pad_nd(
      image_2,
      {
          x2,
          constrain_index(canvas_w, canvas_w - (w2 + x2)),
          y2,
          constrain_index(canvas_h, canvas_h - (h2 + y2)),
      },
      0.0);
#endif
  if (verbose) {
    std::cout << "full_left size=" << full_left.sizes()
              << "\nfull_right size=" << full_right.sizes() << std::endl;
  }
  TORCH_CHECK(full_left.size(2) == canvas_h);
  TORCH_CHECK(full_left.size(3) == canvas_w);
  TORCH_CHECK(
      full_left.sizes() == full_right.sizes(),
      "Full left and right sizes must be the same");

  return {std::move(full_left), std::move(full_right)};
}

at::Tensor ImageBlender::hard_seam_blend(
    at::Tensor&& image_1,
    const std::vector<int>& xy_pos_1,
    at::Tensor&& image_2,
    const std::vector<int>& xy_pos_2) const {
  auto [full_left, full_right] = make_full(image_1, image_2, /*level=*/0);

  int channels = image_1.size(1);
  assert(channels == 3 || channels == 4);
  at::TensorOptions options;
  options = options.dtype(image_1.dtype()).device(image_1.device());
  at::Tensor canvas = at::empty(
      {image_1.size(0), // batch size
       channels,
       seam_.size(0),
       seam_.size(1)},
      options);

  // std::cout << "seam size=" << seam_.sizes() << std::endl;
  // std::cout << "canvas size=" << canvas.sizes() << std::endl;

  // std::cout << seam_.sizes() << ", " << seam_.dtype() << std::endl;
  // std::cout << seam_.sizes() << ", " << seam_.dtype() << std::endl;
  // std::cout << left_seam_value_.sizes() << ", " << left_seam_value_.dtype()
  //           << std::endl;

  canvas.index_put_(
      {torch::indexing::Slice(), torch::indexing::Slice(), condition_left_},
      full_left.index(
          {torch::indexing::Slice(),
           torch::indexing::Slice(),
           condition_left_}));
  canvas.index_put_(
      {torch::indexing::Slice(), torch::indexing::Slice(), condition_right_},
      full_right.index(
          {torch::indexing::Slice(),
           torch::indexing::Slice(),
           condition_right_}));

  // TODO: fix xor map stuff
  // canvas = at::where(xor_map_ > 0, ())

  return canvas;
}

void ImageBlender::build_coordinate_system(
    const at::Tensor& image_1,
    const std::vector<int>& xy_pos_1,
    const at::Tensor& image_2,
    const std::vector<int>& xy_pos_2) {
  // first pass, fill in the size/pos data for each level, even if we aren;t
  // blending, since we need the full-size item anyway got a call to make_full

  // verify channels first
  assert(image_1.size(1) == 3);
  assert(image_2.size(1) == 3);

  int h1 = image_1.size(2);
  int w1 = image_1.size(3);
  int x1 = xy_pos_1.at(0);
  int y1 = xy_pos_1.at(1);
  int h2 = image_2.size(2);
  int w2 = image_2.size(3);
  int x2 = xy_pos_2.at(0);
  int y2 = xy_pos_2.at(1);

  // align with top right of canvas
  if (y1 <= y2) {
    y2 -= y1;
    y1 = 0;
  } else {
    y1 -= y2;
    y2 = 0;
  }
  if (x1 <= x2) {
    x2 -= x1;
    x1 = 0;
  } else {
    x1 -= x2;
    x2 = 0;
  }

  ainfos_.clear();
  // From full size, then levels_ half-sized each time
  ainfos_.emplace_back(std::vector<AInfo>{
      AInfo{
          .h = h1,
          .w = w1,
          .x = x1,
          .y = y1,
      },
      AInfo{
          .h = h2,
          .w = w2,
          .x = x2,
          .y = y2,
      }});
  for (std::size_t l = 0; l < levels_; l++) {
    ainfos_.emplace_back(std::vector<AInfo>{
        (*ainfos_.rbegin())[0] / 2, (*ainfos_.rbegin())[1] / 2});
  }
}

at::Tensor ImageBlender::forward(
    at::Tensor&& image_1,
    const std::vector<int>& xy_pos_1,
    at::Tensor&& image_2,
    const std::vector<int>& xy_pos_2) {
  if (lazy_init_ && !initialized_) {
    init();
  }
  assert(initialized_);

  if (ainfos_.empty()) {
    build_coordinate_system(image_1, xy_pos_1, image_2, xy_pos_2);
  }

  if (mode_ == Mode::HardSeam) {
    return hard_seam_blend(
        std::move(image_1), xy_pos_1, std::move(image_2), xy_pos_2);
  }
  return laplacian_pyramid_blend(
      std::move(image_1), xy_pos_1, std::move(image_2), xy_pos_2);
}

at::Tensor ImageBlender::blend(
    at::Tensor&& left_small_gaussian_blurred,
    at::Tensor&& right_small_gaussian_blurred,
    int level) {
  at::Tensor mask_1d = mask_small_gaussian_blurred_.at(level);

  at::Tensor mask_left = mask_1d;
  at::Tensor mask_right = 1 - mask_1d;

  if (verbose) {
    std::cout << "Level: " << level << ", mask: " << mask_1d.sizes()
              << ", left: " << left_small_gaussian_blurred.sizes()
              << ", right: " << right_small_gaussian_blurred.sizes()
              << std::endl;
  }

  TORCH_CHECK(
      image_size(mask_left) == image_size(left_small_gaussian_blurred),
      "Left image does not match mask size");
  TORCH_CHECK(
      image_size(mask_right) == image_size(right_small_gaussian_blurred),
      "Left image does not match mask size");
#if 1 // in-place
  left_small_gaussian_blurred *= mask_left;
  right_small_gaussian_blurred *= mask_right;
  return left_small_gaussian_blurred + right_small_gaussian_blurred;
#else
  at::Tensor F_2 = left_small_gaussian_blurred * mask_left +
      right_small_gaussian_blurred * mask_right;
  return F_2;
#endif
}

at::Tensor ImageBlender::laplacian_pyramid_blend(
    at::Tensor&& image_1,
    const std::vector<int>& xy_pos_1,
    at::Tensor&& image_2,
    const std::vector<int>& xy_pos_2) {
  at::Tensor image_left, image_right;
  if (make_all_full_first_) {
    auto res = make_full(image_1, image_2, /*level=*/0);
    image_left = res.first;
    image_right = res.second;
  } else {
    image_left = std::move(image_1);
    image_right = std::move(image_2);
  }
  if (image_left.dtype() != dtype_) {
    image_left = image_left.to(dtype_, /*non_blocking=*/true);
  }
  if (image_right.dtype() != dtype_) {
    image_right = image_right.to(dtype_, /*non_blocking=*/true);
  }

  if (verbose) {
    std::cout << "image_left size=" << image_left.sizes()
              << "\nimage_right size=" << image_right.sizes() << std::endl;
  }

  std::vector<at::Tensor> left_laplacian =
      create_laplacian_pyramid(image_left, gussian_kernel_);
  std::vector<at::Tensor> right_laplacian =
      create_laplacian_pyramid(image_right, gussian_kernel_);

  at::Tensor left_small_gaussian_blurred = *left_laplacian.rbegin();
  at::Tensor right_small_gaussian_blurred = *right_laplacian.rbegin();

  if (verbose) {
    std::cout << "left_small_gaussian_blurred size="
              << left_small_gaussian_blurred.sizes()
              << "\nright_small_gaussian_blurred size="
              << right_small_gaussian_blurred.sizes() << std::endl;
  }

  if (!make_all_full_first_) {
    auto res = make_full(
        left_small_gaussian_blurred, right_small_gaussian_blurred, levels_);
    left_small_gaussian_blurred = res.first;
    right_small_gaussian_blurred = res.second;
  }

  if (verbose) {
    std::cout << "left_small_gaussian_blurred size="
              << left_small_gaussian_blurred.sizes()
              << "\nright_small_gaussian_blurred size="
              << right_small_gaussian_blurred.sizes() << std::endl;
  }

  at::Tensor F_2 = blend(
      std::move(left_small_gaussian_blurred),
      std::move(right_small_gaussian_blurred),
      levels_);

  for (int this_level = levels_ - 1; this_level >= 0; this_level--) {
    const ImageSize& canvas_dims = level_canvas_dims_.at(this_level);

    at::Tensor L_left;
    at::Tensor L_right;

    if (!make_all_full_first_) {
      auto res = make_full(
          left_laplacian.at(this_level),
          right_laplacian.at(this_level),
          this_level);
      L_left = res.first;
      L_right = res.second;
    } else {
      L_left = left_laplacian.at(this_level);
      L_right = right_laplacian.at(this_level);
    }

    at::Tensor L_c = blend(std::move(L_left), std::move(L_right), this_level);

    at::Tensor F_1 = upsample(std::move(F_2), {canvas_dims.h, canvas_dims.w});
    at::Tensor upsampled_F1 = gaussian_conv2d(std::move(F_1), gussian_kernel_);

    F_2 = L_c;
    F_2 += upsampled_F1;
  }
  // show_image("blended output", F_2, /*wait=*/false, /*scale=*/0.1);
  return F_2;
}

} // namespace ops
} // namespace hm
