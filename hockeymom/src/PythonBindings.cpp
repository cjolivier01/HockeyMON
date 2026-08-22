#include <ATen/ATen.h>
#include <pybind11/numpy.h>
#include <torch/extension.h>
#include <torch/torch.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <deque>
#include <memory>
#include <mutex>
#include <random>
#include <stdexcept>
#include <thread>
#include <vector>

#include "cudaYUV.h"

#include "cupano/pano/cudaPano.h"
#include "cupano/pano/cudaPano3.h"
#include "cupano/pano/cudaPanoN.h"
#include "hockeymom/csrc/bytetrack/BYTETracker.h"
#include "hockeymom/csrc/bytetrack/BYTETrackerCuda.h"
#include "hockeymom/csrc/bytetrack/BYTETrackerCudaStatic.h"
#include "hockeymom/csrc/bytetrack/HmTracker.h"
#include "hockeymom/csrc/df/DfTrackerCudaStatic.h"
#include "hockeymom/csrc/kmeans/kmeans.h"
#include "hockeymom/csrc/play_tracker/BoxUtils.h"
#include "hockeymom/csrc/play_tracker/LivingBoxImpl.h"
#include "hockeymom/csrc/play_tracker/PlayTracker.h"
#include "hockeymom/csrc/play_tracker/ResizingBox.h"
#include "hockeymom/csrc/play_tracker/TranslatingBox.h"
#include "hockeymom/csrc/pytorch/image_blend.h"
#include "hockeymom/csrc/pytorch/image_remap.h"
#include "hockeymom/csrc/pytorch/image_stitch.h"
#include "hockeymom/csrc/pytorch/torch_cuda_compat.h"
#include "hockeymom/csrc/stitcher/HomographyMaps.h"
#include "hockeymom/csrc/ui/HmRenderSet.h"

#ifndef NO_CPP_BLENDING
#include "hockeymom/csrc/mblend/mblend.h"
#endif

PYBIND11_MAKE_OPAQUE(std::map<std::string, std::complex<double>>);
PYBIND11_MAKE_OPAQUE(std::vector<std::pair<std::string, double>>);

namespace py = pybind11;
using namespace hm;
using namespace hm::play_tracker;

namespace hm {
// TODO: REMOVE

// Only used in python now, so don't need it here
struct BlenderConfig {
  static constexpr const char* kBlendModeGpuHardSeam = "gpu-hard-seam";
  static constexpr const char* kBlendModeGpuLaplacian = "laplacian";
  /**
   * @brief Modes: multiblend, hard_seam, laplacian
   */
  std::string mode = std::string(kBlendModeGpuLaplacian);
  int levels{0};
  at::Tensor seam;
  at::Tensor xor_map;
  bool lazy_init{false};
  std::string interpolation;
  std::string device = std::string("cpu");
};

class AspenGraphSampler {
 public:
  AspenGraphSampler(
      size_t max_samples,
      int min_interval_ms,
      int max_interval_ms)
      : max_samples_(max_samples),
        min_interval_ms_(min_interval_ms),
        max_interval_ms_(max_interval_ms),
        rng_(std::random_device{}()) {
    if (max_samples_ == 0) {
      max_samples_ = 1;
    }
    if (min_interval_ms_ < 1) {
      min_interval_ms_ = 1;
    }
    if (max_interval_ms_ < min_interval_ms_) {
      max_interval_ms_ = min_interval_ms_;
    }
  }

  ~AspenGraphSampler() {
    stop();
  }

  void configure_graph(
      const std::vector<std::string>& names,
      const std::vector<int>& degrees,
      const std::vector<std::pair<int, int>>& edges) {
    if (running_.load(std::memory_order_relaxed)) {
      throw std::runtime_error(
          "AspenGraphSampler configure_graph while running");
    }
    names_ = names;
    degrees_ = degrees;
    edges_ = edges;
    active_count_size_ = names_.size();
    if (active_count_size_ == 0) {
      active_counts_.reset();
      return;
    }
    active_counts_ = std::make_unique<std::atomic<int>[]>(active_count_size_);
    for (size_t i = 0; i < active_count_size_; ++i) {
      active_counts_[i].store(0, std::memory_order_relaxed);
    }
  }

  void start() {
    bool expected = false;
    if (!running_.compare_exchange_strong(expected, true)) {
      return;
    }
    worker_ = std::thread(&AspenGraphSampler::run, this);
  }

  void stop() {
    if (!running_.exchange(false)) {
      return;
    }
    if (worker_.joinable()) {
      worker_.join();
    }
  }

  void enter_index(int idx) {
    if (idx < 0 || static_cast<size_t>(idx) >= active_count_size_) {
      return;
    }
    active_counts_[idx].fetch_add(1, std::memory_order_relaxed);
  }

  void exit_index(int idx) {
    if (idx < 0 || static_cast<size_t>(idx) >= active_count_size_) {
      return;
    }
    int prev = active_counts_[idx].fetch_sub(1, std::memory_order_relaxed);
    if (prev <= 0) {
      active_counts_[idx].store(0, std::memory_order_relaxed);
    }
  }

  py::list pop_samples(size_t max_items) {
    std::deque<Sample> out;
    {
      std::lock_guard<std::mutex> lock(samples_mu_);
      if (samples_.empty()) {
        return py::list();
      }
      size_t count = std::min(max_items, samples_.size());
      auto start = samples_.size() - count;
      for (size_t i = start; i < samples_.size(); ++i) {
        out.push_back(samples_[i]);
      }
      samples_.clear();
    }
    py::list result;
    for (const auto& sample : out) {
      py::dict entry;
      entry["timestamp"] = sample.timestamp;
      py::list active;
      for (auto value : sample.active) {
        active.append(static_cast<bool>(value));
      }
      entry["active"] = active;
      result.append(entry);
    }
    return result;
  }

 private:
  struct Sample {
    double timestamp{0.0};
    std::vector<uint8_t> active;
  };

  void run() {
    std::uniform_int_distribution<int> dist(min_interval_ms_, max_interval_ms_);
    while (running_.load(std::memory_order_relaxed)) {
      int sleep_ms = dist(rng_);
      std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms));
      if (!running_.load(std::memory_order_relaxed)) {
        break;
      }
      Sample sample;
      sample.timestamp =
          std::chrono::duration<double>(
              std::chrono::steady_clock::now().time_since_epoch())
              .count();
      sample.active.resize(active_count_size_);
      for (size_t i = 0; i < active_count_size_; ++i) {
        sample.active[i] =
            active_counts_[i].load(std::memory_order_relaxed) > 0 ? 1 : 0;
      }
      {
        std::lock_guard<std::mutex> lock(samples_mu_);
        samples_.push_back(std::move(sample));
        while (samples_.size() > max_samples_) {
          samples_.pop_front();
        }
      }
    }
  }

  [[maybe_unused]] std::vector<std::string> names_;
  [[maybe_unused]] std::vector<int> degrees_;
  [[maybe_unused]] std::vector<std::pair<int, int>> edges_;
  std::unique_ptr<std::atomic<int>[]> active_counts_;
  size_t active_count_size_{0};
  std::atomic<bool> running_{false};
  std::thread worker_;
  std::mutex samples_mu_;
  std::deque<Sample> samples_;
  size_t max_samples_;
  int min_interval_ms_;
  int max_interval_ms_;
  std::mt19937 rng_;
};

using hm::pano::cuda::CudaStitchPano;
// TODO: make templated and name CudaStitchPanoU8 and CudaStitchPanoF16 python
// classes
template <typename T, typename T_compute>
class PyCudaStitchPano : public CudaStitchPano<T, T_compute> {
  using Super = CudaStitchPano<T, T_compute>;

 public:
  PyCudaStitchPano(
      std::string game_dir,
      int batch_size,
      int num_levels,
      WHDims input1_size,
      WHDims input2_size,
      bool minimize_blend,
      int max_output_width)
      : CudaStitchPano<T, T_compute>(
            batch_size,
            num_levels,
            hm::pano::ControlMasks(std::move(game_dir)),
            /*quiet=*/false,
            /*minimize_blend=*/minimize_blend,
            /*max_output_width=*/max_output_width),
        input1_size_(input1_size),
        input2_size_(input2_size) {
    if (!Super::status().ok()) {
      std::string ss;
      throw std::runtime_error(Super::status().message());
    }
  }
  void process(
      void* d_input1,
      void* d_input2,
      void* d_canvas,
      cudaStream_t stream) {
    const int bs = Super::batch_size();
    auto canvas = std::make_unique<hm::CudaMat<T>>(
        static_cast<T*>(d_canvas),
        bs,
        Super::canvas_width(),
        Super::canvas_height());
    hm::CudaMat<T> i1(
        static_cast<T*>(d_input1), bs, input1_size_.width, input1_size_.height);
    hm::CudaMat<T> i2(
        static_cast<T*>(d_input2), bs, input2_size_.width, input2_size_.height);
    auto result = Super::process(i1, i2, stream, std::move(canvas));
    if (!result.ok()) {
      throw std::runtime_error(result.status().message());
    }
  }

 private:
  const WHDims input1_size_;
  const WHDims input2_size_;
};

using hm::pano::cuda::CudaStitchPano3;
// TODO: make templated and name CudaStitchPanoU8 and CudaStitchPanoF16 python
// classes
template <typename T, typename T_compute>
class PyCudaStitchPano3 : public CudaStitchPano3<T, T_compute> {
  using Super = CudaStitchPano3<T, T_compute>;

 public:
  PyCudaStitchPano3(
      std::string game_dir,
      int batch_size,
      int num_levels,
      std::vector<WHDims> input_sizes)
      : CudaStitchPano3<T, T_compute>(
            batch_size,
            num_levels,
            hm::pano::ControlMasks3(std::move(game_dir))),
        input_sizes_(std::move(input_sizes)) {
    if (!Super::status().ok()) {
      std::string ss;
      throw std::runtime_error(Super::status().message());
    }
  }
  void process(
      const std::vector<void*>& d_inputs,
      void* d_canvas,
      cudaStream_t stream) {
    const int bs = Super::batch_size();
    auto canvas = std::make_unique<hm::CudaMat<T>>(
        static_cast<T*>(d_canvas),
        bs,
        Super::canvas_width(),
        Super::canvas_height());
    hm::CudaMat<T> i1(
        static_cast<T*>(d_inputs[0]),
        bs,
        input_sizes_[0].width,
        input_sizes_[0].height);
    hm::CudaMat<T> i2(
        static_cast<T*>(d_inputs[1]),
        bs,
        input_sizes_[1].width,
        input_sizes_[1].height);
    hm::CudaMat<T> i3(
        static_cast<T*>(d_inputs[2]),
        bs,
        input_sizes_[2].width,
        input_sizes_[2].height);
    auto result = Super::process(i1, i2, i3, stream, std::move(canvas));
    if (!result.ok()) {
      throw std::runtime_error(result.status().message());
    }
  }

 private:
  const std::vector<WHDims> input_sizes_;
};

using hm::pano::cuda::CudaStitchPanoN;
// TODO: make templated and name CudaStitchPanoNU8 and CudaStitchPanoNF32 python
// classes
template <typename T, typename T_compute>
class PyCudaStitchPanoN : public CudaStitchPanoN<T, T_compute> {
  using Super = CudaStitchPanoN<T, T_compute>;

 public:
  PyCudaStitchPanoN(
      std::string game_dir,
      int batch_size,
      int num_levels,
      std::vector<WHDims> input_sizes,
      bool minimize_blend,
      bool quiet)
      : CudaStitchPanoN<T, T_compute>(
            batch_size,
            num_levels,
            hm::pano::ControlMasksN(
                std::move(game_dir),
                static_cast<int>(input_sizes.size())),
            minimize_blend,
            quiet),
        input_sizes_(std::move(input_sizes)) {
    if (!Super::status().ok()) {
      throw std::runtime_error(Super::status().message());
    }
  }

  void process(
      const std::vector<void*>& d_inputs,
      void* d_canvas,
      cudaStream_t stream) {
    const int bs = Super::batch_size();
    if (d_inputs.size() != input_sizes_.size()) {
      throw std::runtime_error("inputs.size() must match stitcher N");
    }

    auto canvas = std::make_unique<hm::CudaMat<T>>(
        static_cast<T*>(d_canvas),
        bs,
        Super::canvas_width(),
        Super::canvas_height());

    std::vector<std::unique_ptr<hm::CudaMat<T>>> inputs;
    inputs.reserve(d_inputs.size());
    std::vector<const hm::CudaMat<T>*> input_ptrs;
    input_ptrs.reserve(d_inputs.size());
    for (size_t i = 0; i < d_inputs.size(); ++i) {
      const WHDims& sz = input_sizes_[i];
      inputs.emplace_back(std::make_unique<hm::CudaMat<T>>(
          static_cast<T*>(d_inputs[i]), bs, sz.width, sz.height));
      input_ptrs.emplace_back(inputs.back().get());
    }

    auto result = Super::process(input_ptrs, stream, std::move(canvas));
    if (!result.ok()) {
      throw std::runtime_error(result.status().message());
    }
  }

 private:
  const std::vector<WHDims> input_sizes_;
};

at::Tensor bgr_to_rgb(const at::Tensor& bgr_hwc) {
  auto idx =
      torch::tensor(
          {2, 1, 0},
          at::TensorOptions().dtype(torch::kLong).device(bgr_hwc.device()))
          .to(bgr_hwc.device(), /*non_blocking=*/true);

  // move channels to front, swap, then move back:
  at::Tensor chw = bgr_hwc.permute({2, 0, 1}); // [3, H, W]
  at::Tensor rgb_chw = chw.index_select(0, idx); // swap
  at::Tensor rgb_hwc = rgb_chw.permute({1, 2, 0}); // [H, W, 3]
  return rgb_hwc;
}

at::Tensor bgr_to_i420_cuda(const at::Tensor& bgr_hwc) {
  TORCH_CHECK(bgr_hwc.device().is_cuda(), "bgr_hwc must be a CUDA tensor");
  TORCH_CHECK(
      bgr_hwc.scalar_type() == at::kByte,
      "bgr_to_i420_cuda expects uint8 tensor");
  TORCH_CHECK(
      bgr_hwc.dim() == 3 && bgr_hwc.size(2) == 3,
      "bgr_to_i420_cuda expects shape [H, W, 3]");

  hm::torch_cuda_compat::DeviceGuard device_guard(bgr_hwc.device());

  at::Tensor rgb_hwc =
      bgr_to_rgb(bgr_hwc.is_contiguous() ? bgr_hwc : bgr_hwc.contiguous());

  const int64_t height = rgb_hwc.size(0);
  const int64_t width = rgb_hwc.size(1);

  TORCH_CHECK(
      (height % 2 == 0) && (width % 2 == 0),
      "bgr_to_i420_cuda requires even width/height (got ",
      width,
      "x",
      height,
      ")");

  at::Tensor rgb_contig = rgb_hwc.contiguous();

  at::Tensor i420 =
      at::empty({height * 3 / 2, width}, rgb_contig.options().dtype(at::kByte));

  cudaStream_t stream =
      hm::torch_cuda_compat::get_current_stream(bgr_hwc.device().index());

  auto* input_ptr = reinterpret_cast<uchar3*>(rgb_contig.data_ptr<uint8_t>());
  void* output_ptr = static_cast<void*>(i420.data_ptr<uint8_t>());

  const cudaError_t err = cudaRGBToI420(
      input_ptr,
      output_ptr,
      static_cast<size_t>(width),
      static_cast<size_t>(height),
      stream);

  TORCH_CHECK(
      err == cudaSuccess,
      "cudaRGBToI420 failed with error: ",
      cudaGetErrorString(err));

  return i420;
}

void show_cuda_tensor_impl(
    const std::string& label,
    at::Tensor img_cuda,
    bool wait,
    cudaStream_t stream) {
  // Expecting a CUDA tensor of shape [H, W, C] and dtype uint8
  TORCH_CHECK(img_cuda.device().is_cuda(), "Tensor must be on CUDA");
  TORCH_CHECK(img_cuda.scalar_type() == at::kByte, "Tensor must be uint8");
  TORCH_CHECK(
      img_cuda.dim() == 3 && img_cuda.size(2) == 3, "Tensor must be H×W×3");

  hm::torch_cuda_compat::DeviceGuard device_guard(img_cuda.device());

  cudaStream_t render_stream = stream;

  // Wrap external streams so ATen conversion work and display interop are
  // queued on the same stream.
  std::unique_ptr<hm::torch_cuda_compat::Stream> cuda_stream;
  std::unique_ptr<hm::torch_cuda_compat::StreamGuard> stream_guard;
  if (render_stream) {
    cuda_stream = std::make_unique<hm::torch_cuda_compat::Stream>(
        hm::torch_cuda_compat::get_stream_from_external(
            render_stream, img_cuda.device().index()));
    stream_guard =
        std::make_unique<hm::torch_cuda_compat::StreamGuard>(*cuda_stream);
  } else {
    render_stream =
        hm::torch_cuda_compat::get_current_stream(img_cuda.device().index());
  }

  img_cuda = bgr_to_rgb(std::move(img_cuda));

  // Make sure it’s contiguous in memory
  auto img = img_cuda.is_contiguous() ? img_cuda : img_cuda.contiguous();

  // 1) Dimensions
  int height = img.size(0); // number of rows
  int width = img.size(1); // number of columns
  int channels = img.size(2); // should be 3

  TORCH_CHECK(
      channels == 3 || channels == 4,
      "Tensor must have three or four channels");

  // 2) GPU data pointer
  //    data_ptr<T>() gives you a raw device pointer to the tensor’s storage.
  uint8_t* d_ptr = img.data_ptr<uint8_t>();

  // size_t pitch = img.stride(0);
  size_t pitch = width * img.element_size() * channels;

  hm::display::DisplaySurface surface(d_ptr, width, height, pitch, channels);
  auto render_set = hm::display::get_or_create_global_render_set().lock();
  if (!render_set) {
    throw std::runtime_error("Render set has been destroyed");
  }
  TORCH_CHECK(
      render_set->render(label, surface, render_stream),
      "show_cuda_tensor failed to render frame");
  if (wait) {
    const cudaError_t err = cudaStreamSynchronize(render_stream);
    TORCH_CHECK(
        err == cudaSuccess,
        "show_cuda_tensor failed while waiting for render stream: ",
        cudaGetErrorString(err));
  }
}

void on_python_exit() {
  hm::display::destroy_global_render_set();
}

} // namespace hm

void init_stitching(::pybind11::module_& m) {
  // hm::init_stack_trace();

  // py::class_<hm::HMPostprocessConfig,
  // std::shared_ptr<hm::HMPostprocessConfig>>(
  //     m, "HMPostprocessConfig")
  //     .def(py::init<>())
  //     .def("to_string", &hm::HMPostprocessConfig::to_string)
  //     .def_readwrite("show_image", &hm::HMPostprocessConfig::show_image)
  //     .def_readwrite(
  //         "plot_individual_player_tracking",
  //         &hm::HMPostprocessConfig::plot_individual_player_tracking)
  //     .def_readwrite(
  //         "plot_cluster_tracking",
  //         &hm::HMPostprocessConfig::plot_cluster_tracking)
  //     .def_readwrite(
  //         "plot_camera_tracking",
  //         &hm::HMPostprocessConfig::plot_camera_tracking)
  //     .def_readwrite("plot_speed", &hm::HMPostprocessConfig::plot_speed)
  //     .def_readwrite(
  //         "max_in_aspec_ratio", &hm::HMPostprocessConfig::max_in_aspec_ratio)
  //     .def_readwrite(
  //         "no_max_in_aspec_ratio_at_edges",
  //         &hm::HMPostprocessConfig::no_max_in_aspec_ratio_at_edges)
  //     .def_readwrite(
  //         "apply_fixed_edge_scaling",
  //         &hm::HMPostprocessConfig::apply_fixed_edge_scaling)
  //     .def_readwrite(
  //         "fixed_edge_scaling_factor",
  //         &hm::HMPostprocessConfig::fixed_edge_scaling_factor)
  //     .def_readwrite(
  //         "fixed_edge_rotation",
  //         &hm::HMPostprocessConfig::fixed_edge_rotation)
  //     .def_readwrite(
  //         "plot_sticky_camera", &hm::HMPostprocessConfig::plot_sticky_camera)
  //     .def_readwrite(
  //         "crop_output_image", &hm::HMPostprocessConfig::crop_output_image)
  //     .def_readwrite("use_cuda", &hm::HMPostprocessConfig::use_cuda)
  //     .def_readwrite("use_watermark",
  //     &hm::HMPostprocessConfig::use_watermark);

  // py::class_<hm::ImagePostProcessor,
  // std::shared_ptr<hm::ImagePostProcessor>>(
  //     m, "ImagePostProcessor")
  //     .def(py::init<std::shared_ptr<hm::HMPostprocessConfig>,
  //     std::string>());

  py::class_<hm::ops::RemapperConfig, std::shared_ptr<hm::ops::RemapperConfig>>(
      m, "RemapperConfig")
      .def(py::init<>())
      .def_readwrite("src_width", &hm::ops::RemapperConfig::src_width)
      .def_readwrite("src_height", &hm::ops::RemapperConfig::src_height)
      .def_readwrite("x_pos", &hm::ops::RemapperConfig::x_pos)
      .def_readwrite("y_pos", &hm::ops::RemapperConfig::y_pos)
      .def_readwrite("col_map", &hm::ops::RemapperConfig::col_map)
      .def_readwrite("row_map", &hm::ops::RemapperConfig::row_map)
      .def_readwrite("dtype", &hm::ops::RemapperConfig::dtype)
      .def_readwrite(
          "add_alpha_channel", &hm::ops::RemapperConfig::add_alpha_channel)
      .def_readwrite("interpolation", &hm::ops::RemapperConfig::interpolation)
      .def_readwrite("batch_size", &hm::ops::RemapperConfig::batch_size)
      .def_readwrite("device", &hm::ops::RemapperConfig::device);

  auto homography_result_to_dict = [](const hm::stitcher::HomographyMapResult&
                                          result) {
    auto matrix_array = [](const std::array<double, 9>& values) {
      py::array_t<double> array({3, 3});
      std::copy(values.begin(), values.end(), array.mutable_data());
      return array;
    };
    py::dict output;
    output["canvas_width"] = result.canvas_width;
    output["canvas_height"] = result.canvas_height;
    output["output_scale"] = result.output_scale;
    output["right_to_left_homography"] =
        matrix_array(result.right_to_left_homography);
    output["left_to_canvas_homography"] =
        matrix_array(result.left_to_canvas_homography);
    output["right_to_canvas_homography"] =
        matrix_array(result.right_to_canvas_homography);

    py::array_t<uint8_t> inlier_mask(result.inlier_mask.size());
    std::copy(
        result.inlier_mask.begin(),
        result.inlier_mask.end(),
        inlier_mask.mutable_data());
    output["inlier_mask"] = std::move(inlier_mask);

    py::list image_maps;
    for (const auto& image_map : result.image_maps) {
      py::dict item;
      item["x_position"] = image_map.x_position;
      item["y_position"] = image_map.y_position;
      py::array_t<uint16_t> x_map({image_map.height, image_map.width});
      py::array_t<uint16_t> y_map({image_map.height, image_map.width});
      std::copy(
          image_map.x_map.begin(), image_map.x_map.end(), x_map.mutable_data());
      std::copy(
          image_map.y_map.begin(), image_map.y_map.end(), y_map.mutable_data());
      item["x_map"] = std::move(x_map);
      item["y_map"] = std::move(y_map);
      image_maps.append(std::move(item));
    }
    output["image_maps"] = std::move(image_maps);
    return output;
  };

  m.def(
      "create_homography_maps",
      [homography_result_to_dict](
          const std::vector<std::array<double, 2>>& left_points,
          const std::vector<std::array<double, 2>>& right_points,
          int left_width,
          int left_height,
          int right_width,
          int right_height,
          double reprojection_threshold,
          double confidence,
          int max_iterations,
          int max_output_dimension) {
        hm::stitcher::HomographyMapResult result;
        {
          py::gil_scoped_release release;
          result = hm::stitcher::create_homography_maps(
              left_points,
              right_points,
              left_width,
              left_height,
              right_width,
              right_height,
              reprojection_threshold,
              confidence,
              max_iterations,
              max_output_dimension);
        }
        return homography_result_to_dict(result);
      },
      py::arg("left_points"),
      py::arg("right_points"),
      py::arg("left_width"),
      py::arg("left_height"),
      py::arg("right_width"),
      py::arg("right_height"),
      py::arg("reprojection_threshold") = 3.0,
      py::arg("confidence") = 0.999,
      py::arg("max_iterations") = 10000,
      py::arg("max_output_dimension") = 0);

  m.def(
      "create_affine_ransac_maps",
      [homography_result_to_dict](
          const std::vector<std::array<double, 2>>& left_points,
          const std::vector<std::array<double, 2>>& right_points,
          int left_width,
          int left_height,
          int right_width,
          int right_height,
          double reprojection_threshold,
          double confidence,
          int max_iterations,
          int refine_iterations,
          int max_output_dimension) {
        hm::stitcher::HomographyMapResult result;
        {
          py::gil_scoped_release release;
          result = hm::stitcher::create_affine_ransac_maps(
              left_points,
              right_points,
              left_width,
              left_height,
              right_width,
              right_height,
              reprojection_threshold,
              confidence,
              max_iterations,
              refine_iterations,
              max_output_dimension);
        }
        return homography_result_to_dict(result);
      },
      py::arg("left_points"),
      py::arg("right_points"),
      py::arg("left_width"),
      py::arg("left_height"),
      py::arg("right_width"),
      py::arg("right_height"),
      py::arg("reprojection_threshold") = 10.0,
      py::arg("confidence") = 0.999,
      py::arg("max_iterations") = 10000,
      py::arg("refine_iterations") = 10,
      py::arg("max_output_dimension") = 0);

  py::class_<hm::BlenderConfig, std::shared_ptr<hm::BlenderConfig>>(
      m, "BlenderConfig")
      .def(py::init<>())
      .def_readwrite("mode", &hm::BlenderConfig::mode)
      .def_readwrite("levels", &hm::BlenderConfig::levels)
      .def_readwrite("seam", &hm::BlenderConfig::seam)
      .def_readwrite("xor_map", &hm::BlenderConfig::xor_map)
      .def_readwrite("lazy_init", &hm::BlenderConfig::lazy_init)
      .def_readwrite("interpolation", &hm::BlenderConfig::interpolation)
      .def_readwrite("device", &hm::BlenderConfig::device);

  // py::class_<hm::StitchingDataLoader,
  // std::shared_ptr<hm::StitchingDataLoader>>(
  //     m, "StitchingDataLoader")
  //     .def(py::init<
  //          at::ScalarType,
  //          std::size_t,
  //          std::string,
  //          std::string,
  //          std::string,
  //          bool,
  //          std::size_t,
  //          std::size_t,
  //          std::size_t>())
  //     .def("fps", &hm::StitchingDataLoader::fps)
  //     .def(
  //         "configure_remapper",
  //         &hm::StitchingDataLoader::configure_remapper,
  //         py::arg("remapper_config"),
  //         py::call_guard<py::gil_scoped_release>())
  //     .def(
  //         "configure_blender",
  //         &hm::StitchingDataLoader::configure_blender,
  //         py::arg("blender_config"),
  //         py::call_guard<py::gil_scoped_release>())
  //     .def(
  //         "add_frame",
  //         [](std::shared_ptr<hm::StitchingDataLoader> data_loader,
  //            std::size_t frame_id,
  //            std::optional<py::array_t<uint8_t>> image1,
  //            std::optional<py::array_t<uint8_t>> image2) {
  //           // We expect a three-channel RGB image here
  //           if (!image1 && !image2) {
  //             // Exiting
  //             py::gil_scoped_release release_gil;
  //             data_loader->add_frame(frame_id, {});
  //             return frame_id;
  //           }
  //           assert(image1->ndim() == 3);
  //           assert(image2->ndim() == 3);
  //           auto m1 = std::make_shared<hm::MatrixRGB>(*image1, 0, 0);
  //           auto m2 = std::make_shared<hm::MatrixRGB>(*image2, 0, 0);
  //           {
  //             py::gil_scoped_release release_gil;
  //             data_loader->add_frame(frame_id, {std::move(m1),
  //             std::move(m2)});
  //           }
  //           return frame_id;
  //         })
  //     .def(
  //         "add_torch_frame",
  //         &hm::StitchingDataLoader::add_torch_frame,
  //         py::arg("frame_id"),
  //         py::arg("image_1"),
  //         py::arg("image_2"),
  //         py::call_guard<py::gil_scoped_release>())
  //     .def(
  //         "add_remapped_frame",
  //         [](std::shared_ptr<hm::StitchingDataLoader> data_loader,
  //            std::size_t frame_id,
  //            std::optional<py::array_t<uint8_t>> image1,
  //            const std::vector<std::size_t>& xy_pos_1,
  //            std::optional<py::array_t<uint8_t>> image2,
  //            const std::vector<std::size_t>& xy_pos_2) {
  //           // We expect a three-channel RGB image here
  //           if (!image1 && !image2) {
  //             // Exiting
  //             py::gil_scoped_release release_gil;
  //             data_loader->add_frame(frame_id, {});
  //             return frame_id;
  //           }
  //           assert(image1->ndim() == 3);
  //           assert(image2->ndim() == 3);
  //           auto m1 = std::make_shared<hm::MatrixRGB>(
  //               *image1, xy_pos_1.at(0), xy_pos_1.at(1));
  //           auto m2 = std::make_shared<hm::MatrixRGB>(
  //               *image2, xy_pos_2.at(0), xy_pos_2.at(1));
  //           {
  //             py::gil_scoped_release release_gil;
  //             data_loader->add_remapped_frame(
  //                 frame_id, {std::move(m1), std::move(m2)});
  //           }
  //           return frame_id;
  //         })
  //     .def(
  //         "get_stitched_frame",
  //         [](std::shared_ptr<hm::StitchingDataLoader> data_loader,
  //            std::size_t frame_id) -> std::optional<at::Tensor> {
  //           std::shared_ptr<hm::MatrixRGB> stitched_image;
  //           {
  //             py::gil_scoped_release release_gil;
  //             stitched_image = data_loader->get_stitched_frame(frame_id);
  //             if (!stitched_image) {
  //               return std::nullopt;
  //             }
  //             return stitched_image->to_tensor();
  //           }
  //         })
  //     .def(
  //         "get_stitched_pytorch_frame",
  //         [](std::shared_ptr<hm::StitchingDataLoader> data_loader,
  //            std::size_t frame_id) -> std::optional<at::Tensor> {
  //           py::gil_scoped_release release_gil;
  //           at::Tensor stitched_image =
  //               data_loader->get_stitched_pytorch_frame(frame_id);
  //           if (!stitched_image.defined()) {
  //             return std::nullopt;
  //           }
  //           return stitched_image;
  //         },
  //         py::arg("frame_id"));

  // using SortedPyArrayUin8Queue =
  //     hm::SortedQueue<std::size_t,
  //     std::unique_ptr<py::array_t<std::uint8_t>>>;
  // py::class_<SortedPyArrayUin8Queue,
  // std::shared_ptr<SortedPyArrayUin8Queue>>(
  //     m, "SortedPyArrayUin8Queue")
  //     .def(py::init<>())
  //     .def(
  //         "enqueue",
  //         [](const std::shared_ptr<SortedPyArrayUin8Queue>& sq,
  //            std::size_t key,
  //            py::array_t<std::uint8_t> array) -> void {
  //           sq->enqueue(
  //               key,
  //               std::make_unique<py::array_t<std::uint8_t>>(std::move(array)));
  //         })
  //     .def(
  //         "dequeue_key",
  //         [](const std::shared_ptr<SortedPyArrayUin8Queue>& sq,
  //            std::size_t key) -> py::array_t<std::uint8_t> {
  //           std::unique_ptr<py::array_t<std::uint8_t>> result;
  //           {
  //             py::gil_scoped_release release;
  //             result = sq->dequeue_key(key);
  //           }
  //           return std::move(*result);
  //         })
  //     .def(
  //         "dequeue_smallest_key",
  //         [](const std::shared_ptr<SortedPyArrayUin8Queue>& sq) {
  //           std::size_t key = ~0;
  //           std::unique_ptr<py::array_t<std::uint8_t>> result;
  //           {
  //             py::gil_scoped_release release;
  //             result = sq->dequeue_smallest_key(&key);
  //           }
  //           return std::make_tuple(key, std::move(*result));
  //         });

  // using SortedRGBImageQueue =
  //     hm::SortedQueue<std::size_t, std::unique_ptr<hm::MatrixRGB>>;

  // py::class_<SortedRGBImageQueue, std::shared_ptr<SortedRGBImageQueue>>(
  //     m, "SortedRGBImageQueue")
  //     .def(py::init<>())
  //     .def(
  //         "identity",
  //         [](const std::shared_ptr<SortedRGBImageQueue>& sq,
  //            py::array_t<std::uint8_t>& array,
  //            bool copy_data) -> py::array_t<std::uint8_t> {
  //           auto matrix =
  //               std::make_unique<hm::MatrixRGB>(array, 0, 0, copy_data);
  //           {
  //             // Unlock the GIL in order to let python muck with the input
  //             array
  //             // if it wants to
  //             py::gil_scoped_release release;
  //           }
  //           return matrix->to_py_array();
  //         })
  //     .def(
  //         "enqueue",
  //         [](const std::shared_ptr<SortedRGBImageQueue>& sq,
  //            std::size_t key,
  //            py::array_t<std::uint8_t> array,
  //            bool copy_data) -> void {
  //           auto matrix =
  //               std::make_unique<hm::MatrixRGB>(array, 0, 0, copy_data);
  //           {
  //             py::gil_scoped_release release;
  //             sq->enqueue(key, std::move(matrix));
  //           }
  //         })
  //     .def(
  //         "dequeue_key",
  //         [](const std::shared_ptr<SortedRGBImageQueue>& sq,
  //            std::size_t key) -> py::array_t<std::uint8_t> {
  //           std::unique_ptr<hm::MatrixRGB> matrix;
  //           {
  //             py::gil_scoped_release release;
  //             matrix = sq->dequeue_key(key);
  //           }
  //           return matrix->to_py_array();
  //         })
  //     .def(
  //         "dequeue_smallest_key",
  //         [](const std::shared_ptr<SortedRGBImageQueue>& sq) {
  //           std::size_t key = ~0;
  //           std::unique_ptr<hm::MatrixRGB> matrix;
  //           {
  //             py::gil_scoped_release release;
  //             matrix = sq->dequeue_smallest_key(&key);
  //           }
  //           return std::make_tuple(key, matrix->to_py_array());
  //         });

  // using SortedTensorQueue = hm::SortedQueue<std::size_t, at::Tensor>;
  // py::class_<SortedTensorQueue, std::shared_ptr<SortedTensorQueue>>(
  //     m, "SortedTensorQueue")
  //     .def(py::init<>())
  //     .def(
  //         "enqueue",
  //         [](const std::shared_ptr<SortedTensorQueue>& sq,
  //            std::size_t key,
  //            at::Tensor tensor) -> void {
  //           sq->enqueue(key, std::move(tensor));
  //         })
  //     .def(
  //         "dequeue_key",
  //         [](const std::shared_ptr<SortedTensorQueue>& sq,
  //            std::size_t key) -> at::Tensor {
  //           py::gil_scoped_release release;
  //           return sq->dequeue_key(key);
  //         })
  //     .def(
  //         "dequeue_smallest_key",
  //         [](const std::shared_ptr<SortedTensorQueue>& sq)
  //             -> std::tuple<std::size_t, at::Tensor> {
  //           std::size_t key = ~0;
  //           std::unique_ptr<py::array_t<std::uint8_t>> result;
  //           py::gil_scoped_release release;
  //           auto tensor = sq->dequeue_smallest_key(&key);
  //           return std::make_tuple(key, tensor);
  //         });
#ifndef NO_CPP_BLENDING
  py::class_<hm::enblend::EnBlender, std::shared_ptr<hm::enblend::EnBlender>>(
      m, "EnBlender")
      .def(
          py::init<std::vector<std::string>>(),
          py::arg("args") = std::vector<std::string>{})
      .def(
          "blend_images",
          [](std::shared_ptr<hm::enblend::EnBlender> blender,
             py::array_t<std::uint8_t>& image1,
             const std::vector<std::size_t>& xy_pos_1,
             py::array_t<std::uint8_t>& image2,
             const std::vector<std::size_t>& xy_pos_2)
              -> py::array_t<std::uint8_t> {
            assert(image1.ndim() == 3);
            assert(image2.ndim() == 3);
            auto m1 = std::make_shared<hm::MatrixRGB>(
                image1, xy_pos_1.at(0), xy_pos_1.at(1), /*copy_data=*/true);
            auto m2 = std::make_shared<hm::MatrixRGB>(
                image2, xy_pos_2.at(0), xy_pos_2.at(1), /*copy_data=*/true);
            std::unique_ptr<hm::MatrixRGB> blended_image;
            {
              py::gil_scoped_release release_gil;
              blended_image = blender->blend_images(
                  std::vector<std::shared_ptr<hm::MatrixRGB>>{m1, m2});
            }
            py::array_t<std::uint8_t> result;
            result = blended_image->to_py_array();
            return result;
          },
          // They don't really have to be left/right here, just the proper order
          // image-1 and image-2, but just for the sake of consistency, lets
          // call it left and right
          py::arg("left_image"),
          py::arg("left_xy_pos"),
          py::arg("right_image"),
          py::arg("right_xy_pos"));
#endif // NO_CPP_BLENDING
  // m.def(
  //     "_add_to_stitching_data_loader",
  //     [](std::shared_ptr<hm::StitchingDataLoader> data_loader,
  //        std::size_t frame_id,
  //        std::optional<py::array_t<uint8_t>> image1,
  //        std::optional<py::array_t<uint8_t>> image2) {
  //       // We expect a three-channel RGB image here
  //       if (!image1 && !image2) {
  //         py::gil_scoped_release release_gil;
  //         data_loader->add_frame(frame_id, {});
  //         return frame_id;
  //       }
  //       assert(image1->ndim() == 3);
  //       assert(image2->ndim() == 3);
  //       auto m1 = std::make_shared<hm::MatrixRGB>(*image1, 0, 0);
  //       auto m2 = std::make_shared<hm::MatrixRGB>(*image2, 0, 0);
  //       {
  //         py::gil_scoped_release release_gil;
  //         data_loader->add_frame(frame_id, {std::move(m1), std::move(m2)});
  //       }
  //       return frame_id;
  //     });

  // m.def("_hello_world", []() {
  //   py::gil_scoped_release release_gil;
  //   std::cout << "Hello, world!" << std::endl;
  // });

  // m.def(
  //     "_enblend",
  //     [](std::string output_image,
  //        std::vector<std::string> input_files) -> int {
  //       py::gil_scoped_release release_gil;
  //       return hm::enblend::enblend_main(
  //           std::move(output_image), std::move(input_files));
  //     });

  // py::class_<hm::HmNona, std::shared_ptr<hm::HmNona>>(m, "HmNona")
  //     .def(py::init<std::string>())
  //     .def("load_project", &hm::HmNona::load_project)
  //     .def("get_control_points", [](const std::shared_ptr<hm::HmNona>& nona)
  //     {
  //       auto results = nona->get_control_points();
  //       return results;
  //     });

  // m.def(
  //     "_nona_process_images",
  //     [](std::shared_ptr<hm::HmNona> nona,
  //        py::array_t<uint8_t>& image1,
  //        py::array_t<uint8_t>& image2) -> std::vector<py::array_t<uint8_t>> {
  //       // We expect a three-channel RGB image here
  //       assert(image1.ndim() == 3);
  //       assert(image2.ndim() == 3);
  //       auto m1 = std::make_shared<hm::MatrixRGB>(image1, 0, 0);
  //       auto m2 = std::make_shared<hm::MatrixRGB>(image2, 0, 0);
  //       // Just remap (no blend)
  //       std::vector<py::array_t<uint8_t>> results;
  //       std::vector<std::unique_ptr<hm::MatrixRGB>> result_matrices;
  //       {
  //         py::gil_scoped_release release_gil;
  //         result_matrices = nona->remap_images(std::move(m1), std::move(m2));
  //       }
  //       results.reserve(result_matrices.size());
  //       for (auto& m : result_matrices) {
  //         if (m) {
  //           results.emplace_back(m->to_py_array());
  //         }
  //       }
  //       return results;
  //     });

  /**
   *   ____                         _
   *  / __ \                       | |
   * | |  | |_ __   ___  _ __  __ _| |_  ___  _ __  ___
   * | |  | | '_ \ / _ \| '__|/ _` | __|/ _ \| '__|/ __|
   * | |__| | |_) |  __/| |  | (_| | |_| (_) | |   \__ \
   *  \____/| .__/ \___||_|   \__,_|\__|\___/|_|   |___/
   *        | |
   *        |_|
   */
  py::class_<hm::ops::ImageRemapper, std::shared_ptr<hm::ops::ImageRemapper>>(
      m, "ImageRemapper")
      .def(
          py::init<
              std::size_t,
              std::size_t,
              at::Tensor,
              at::Tensor,
              at::ScalarType,
              bool,
              std::size_t,
              std::optional<std::string>>(),
          py::arg("src_width"),
          py::arg("src_height"),
          py::arg("col_map"),
          py::arg("row_map"),
          py::arg("dtype"),
          py::arg("add_alpha_channel"),
          py::arg("pad_value"),
          py::arg("interpolation"),
          py::call_guard<py::gil_scoped_release>())
      .def(
          "init",
          &hm::ops::ImageRemapper::init,
          py::arg("batch_size"),
          py::call_guard<py::gil_scoped_release>())
      .def(
          "to",
          &hm::ops::ImageRemapper::to,
          py::arg("device"),
          py::call_guard<py::gil_scoped_release>())
      .def(
          "is_initialized",
          &hm::ops::ImageRemapper::is_initialized,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "forward",
          &hm::ops::ImageRemapper::forward,
          py::arg("source_tensor"),
          py::call_guard<py::gil_scoped_release>());

  py::enum_<hm::ops::ImageBlender::Mode>(m, "ImageBlenderMode")
      .value("HardSeam", hm::ops::ImageBlender::Mode::HardSeam)
      .value("Laplacian", hm::ops::ImageBlender::Mode::Laplacian)
      .export_values();

  py::class_<hm::ops::ImageBlender, std::shared_ptr<hm::ops::ImageBlender>>(
      m, "ImageBlender")
      .def(
          py::init<
              hm::ops::ImageBlender::Mode,
              bool,
              std::size_t,
              at::Tensor,
              at::Tensor,
              bool,
              std::optional<std::string>>(),
          py::arg("mode"),
          py::arg("half"),
          py::arg("levels"),
          py::arg("seam"),
          py::arg("xor_map"),
          py::arg("lazy_init"),
          py::arg("interpolation"),
          py::call_guard<py::gil_scoped_release>())
      .def(
          "to",
          &hm::ops::ImageBlender::to,
          py::arg("device"),
          py::call_guard<py::gil_scoped_release>())
      .def(
          "make_full",
          &hm::ops::ImageBlender::forward,
          py::arg("image_1"),
          py::arg("xy_pos_1"),
          py::arg("image_2"),
          py::arg("xy_pos_2"),
          py::call_guard<py::gil_scoped_release>())
      .def(
          "forward",
          &hm::ops::ImageBlender::forward,
          py::arg("image_1"),
          py::arg("xy_pos_1"),
          py::arg("image_2"),
          py::arg("xy_pos_2"),
          py::call_guard<py::gil_scoped_release>());

  py::class_<
      hm::ops::StitchImageInfo,
      std::shared_ptr<hm::ops::StitchImageInfo>>(m, "StitchImageInfo")
      .def(py::init<>())
      .def_readwrite("image", &hm::ops::StitchImageInfo::image)
      .def_readwrite("xy_pos", &hm::ops::StitchImageInfo::xy_pos);

  py::class_<hm::ops::RemapImageInfo, std::shared_ptr<hm::ops::RemapImageInfo>>(
      m, "RemapImageInfo")
      .def(py::init<>())
      .def_readwrite("src_width", &hm::ops::RemapImageInfo::src_width)
      .def_readwrite("src_height", &hm::ops::RemapImageInfo::src_height)
      .def_readwrite("dtype", &hm::ops::RemapImageInfo::dtype)
      .def_readwrite("col_map", &hm::ops::RemapImageInfo::col_map)
      .def_readwrite("row_map", &hm::ops::RemapImageInfo::row_map)
      .def_readwrite(
          "add_alpha_channel", &hm::ops::RemapImageInfo::add_alpha_channel);

  py::class_<hm::ops::ImageStitcher, std::shared_ptr<hm::ops::ImageStitcher>>(
      m, "ImageStitcher")
      .def(
          py::init<
              std::size_t,
              std::vector<hm::ops::RemapImageInfo>,
              hm::ops::ImageBlender::Mode,
              bool,
              std::size_t,
              at::Tensor,
              at::Tensor,
              bool,
              std::optional<std::string>>(),
          py::arg("batch_size"),
          py::arg("remap_image_info"),
          py::arg("blender_mode"),
          py::arg("half"),
          py::arg("levels"),
          py::arg("seam"),
          py::arg("xor_map"),
          py::arg("lazy_init"),
          py::arg("interpolation") = "bilinear",
          py::call_guard<py::gil_scoped_release>())
      .def(
          "to",
          &hm::ops::ImageStitcher::to,
          py::arg("device"),
          py::call_guard<py::gil_scoped_release>())
      .def(
          "forward",
          &hm::ops::ImageStitcher::forward,
          py::arg("inputs"),
          py::call_guard<py::gil_scoped_release>());
}

void init_tracking(::pybind11::module_& m) {
  /**
   *  ____        _      _______              _
   * |  _ \      | |    |__   __|            | |
   * | |_) |_   _| |_  ___ | |_ __  __ _  ___| | __ ___  _ __
   * |  _ <| | | | __|/ _ \| | '__|/ _` |/ __| |/ // _ \| '__|
   * | |_) | |_| | |_|  __/| | |  | (_| | (__|   <|  __/| |
   * |____/ \__, |\__|\___||_|_|   \__,_|\___|_|\_\\___||_|
   *         __/ |
   *        |___/
   */

  py::class_<hm::tracker::ByteTrackConfig>(m, "ByteTrackConfig")
      .def(py::init<>())
      .def_readwrite(
          "init_track_thr", &hm::tracker::ByteTrackConfig::init_track_thr)
      .def_readwrite(
          "obj_score_thrs_low",
          &hm::tracker::ByteTrackConfig::obj_score_thrs_low)
      .def_readwrite(
          "obj_score_thrs_high",
          &hm::tracker::ByteTrackConfig::obj_score_thrs_high)
      .def_readwrite(
          "match_iou_thrs_high",
          &hm::tracker::ByteTrackConfig::match_iou_thrs_high)
      .def_readwrite(
          "match_iou_thrs_low",
          &hm::tracker::ByteTrackConfig::match_iou_thrs_low)
      .def_readwrite(
          "match_iou_thrs_tentative",
          &hm::tracker::ByteTrackConfig::match_iou_thrs_tentative)
      .def_readwrite(
          "track_buffer_size", &hm::tracker::ByteTrackConfig::track_buffer_size)
      .def_readwrite(
          "num_frames_to_keep_lost_tracks",
          &hm::tracker::ByteTrackConfig::num_frames_to_keep_lost_tracks)

      .def_readwrite(
          "weight_iou_with_det_scores",
          &hm::tracker::ByteTrackConfig::weight_iou_with_det_scores)
      .def_readwrite(
          "num_tentatives", &hm::tracker::ByteTrackConfig::num_tentatives)
      .def_readwrite("momentums", &hm::tracker::ByteTrackConfig::momentums);

  py::class_<
      hm::tracker::BYTETracker,
      std::shared_ptr<hm::tracker::BYTETracker>>(m, "HmByteTracker")
      .def("num_tracks", &hm::tracker::BYTETracker::num_tracks)
      .def(
          "track",
          &hm::tracker::BYTETracker::track,
          py::arg("data"),
          py::call_guard<py::gil_scoped_release>());

  py::class_<
      hm::tracker::BYTETrackerCuda,
      std::shared_ptr<hm::tracker::BYTETrackerCuda>>(m, "HmByteTrackerCuda")
      .def(
          py::init([](hm::tracker::ByteTrackConfig config,
                      const std::string& device) {
            return std::make_shared<hm::tracker::BYTETrackerCuda>(
                std::move(config), c10::Device(device));
          }),
          py::arg("config") = hm::tracker::ByteTrackConfig(),
          py::arg("device") = std::string("cuda:0"))
      .def("num_tracks", &hm::tracker::BYTETrackerCuda::num_tracks)
      .def(
          "track",
          &hm::tracker::BYTETrackerCuda::track,
          py::arg("data"),
          py::call_guard<py::gil_scoped_release>());

  py::class_<
      hm::tracker::BYTETrackerCudaStatic,
      std::shared_ptr<hm::tracker::BYTETrackerCudaStatic>>(
      m, "HmByteTrackerCudaStatic")
      .def(
          py::init([](hm::tracker::ByteTrackConfig config,
                      int64_t max_detections,
                      int64_t max_tracks,
                      const std::string& device) {
            return std::make_shared<hm::tracker::BYTETrackerCudaStatic>(
                std::move(config),
                max_detections,
                max_tracks,
                c10::Device(device));
          }),
          py::arg("config") = hm::tracker::ByteTrackConfig(),
          py::arg("max_detections") = 256,
          py::arg("max_tracks") = 256,
          py::arg("device") = std::string("cuda:0"))
      .def("num_tracks", &hm::tracker::BYTETrackerCudaStatic::num_tracks)
      .def(
          "track",
          &hm::tracker::BYTETrackerCudaStatic::track,
          py::arg("data"),
          py::call_guard<py::gil_scoped_release>())
      .def_property_readonly(
          "max_detections", &hm::tracker::BYTETrackerCudaStatic::max_detections)
      .def_property_readonly(
          "max_tracks", &hm::tracker::BYTETrackerCudaStatic::max_tracks);

  py::class_<
      hm::tracker::DfTrackerCudaStatic,
      std::shared_ptr<hm::tracker::DfTrackerCudaStatic>>(
      m, "HmDcfTrackerCudaStatic")
      .def(
          py::init([](hm::tracker::ByteTrackConfig config,
                      int64_t max_detections,
                      int64_t max_tracks,
                      int64_t reid_feature_dim,
                      float iou_weight,
                      float reid_weight,
                      float box_momentum,
                      float reid_momentum,
                      float min_similarity,
                      float lost_track_cost,
                      const std::string& device) {
            return std::make_shared<hm::tracker::DfTrackerCudaStatic>(
                std::move(config),
                max_detections,
                max_tracks,
                reid_feature_dim,
                iou_weight,
                reid_weight,
                box_momentum,
                reid_momentum,
                min_similarity,
                lost_track_cost,
                c10::Device(device));
          }),
          py::arg("config") = hm::tracker::ByteTrackConfig(),
          py::arg("max_detections") = 256,
          py::arg("max_tracks") = 256,
          py::arg("reid_feature_dim") = 256,
          py::arg("iou_weight") = 0.5f,
          py::arg("reid_weight") = 0.5f,
          py::arg("box_momentum") = 0.6f,
          py::arg("reid_momentum") = 0.2f,
          py::arg("min_similarity") = -1.0f,
          py::arg("lost_track_cost") = 0.05f,
          py::arg("device") = std::string("cuda:0"))
      .def("num_tracks", &hm::tracker::DfTrackerCudaStatic::num_tracks)
      .def(
          "track",
          &hm::tracker::DfTrackerCudaStatic::track,
          py::arg("data"),
          py::call_guard<py::gil_scoped_release>())
      .def_property_readonly(
          "max_detections", &hm::tracker::DfTrackerCudaStatic::max_detections)
      .def_property_readonly(
          "max_tracks", &hm::tracker::DfTrackerCudaStatic::max_tracks)
      .def_property_readonly(
          "reid_feature_dim",
          &hm::tracker::DfTrackerCudaStatic::reid_feature_dim);

  /**
   *  _    _        _______              _
   * | |  | |      |__   __|            | |
   * | |__| |_ __ ___ | |_ __  __ _  ___| | __ ___  _ __
   * |  __  | '_ ` _ \| | '__|/ _` |/ __| |/ // _ \| '__|
   * | |  | | | | | | | | |  | (_| | (__|   <|  __/| |
   * |_|  |_|_| |_| |_|_|_|   \__,_|\___|_|\_\\___||_|
   *
   *
   */

  py::enum_<hm::tracker::HmTrackerPredictionMode>(m, "HmTrackerPredictionMode")
      .value("BoundingBox", hm::tracker::HmTrackerPredictionMode::BoundingBox)
      .value("BoxCenter", hm::tracker::HmTrackerPredictionMode::BoxCenter)
      .value("BoxBottom", hm::tracker::HmTrackerPredictionMode::BoxBottom)
      .value(
          "BoxBottomCenter",
          hm::tracker::HmTrackerPredictionMode::BoxBottomCenter);

  py::class_<hm::tracker::HmTrackerConfig>(m, "HmTrackerConfig")
      .def(py::init<>())
      .def_readwrite(
          "prediction_mode", &hm::tracker::HmTrackerConfig::prediction_mode)
      .def_readwrite(
          "tentative_high_confidence",
          &hm::tracker::HmTrackerConfig::tentative_high_confidence)
      .def_readwrite(
          "num_tentative_high_confidence",
          &hm::tracker::HmTrackerConfig::num_tentative_high_confidence)
      .def_readwrite(
          "tentative_low_confidence",
          &hm::tracker::HmTrackerConfig::tentative_low_confidence)
      .def_readwrite(
          "num_tentative_low_confidence",
          &hm::tracker::HmTrackerConfig::num_tentative_low_confidence)
      .def_readwrite(
          "remove_tentative", &hm::tracker::HmTrackerConfig::remove_tentative)
      .def_readwrite(
          "return_user_ids", &hm::tracker::HmTrackerConfig::return_user_ids)
      .def_readwrite(
          "return_track_age", &hm::tracker::HmTrackerConfig::return_track_age);

  py::class_<
      hm::tracker::HmByteTrackConfig,
      hm::tracker::ByteTrackConfig,
      hm::tracker::HmTrackerConfig>(m, "HmByteTrackConfig")
      .def(py::init<>());

  py::class_<
      hm::tracker::HmTracker,
      hm::tracker::BYTETracker,
      std::shared_ptr<hm::tracker::HmTracker>>(m, "HmTracker")
      .def(
          py::init<hm::tracker::HmByteTrackConfig>(),
          py::arg("config") = hm::tracker::HmByteTrackConfig())
      .def(
          "total_activated_tracks_count",
          &hm::tracker::HmTracker::total_activated_tracks_count);
}

void init_box_structures(::pybind11::module_& m) {
  //
  // Box structures
  //
  py::class_<WHDims>(m, "WHDims")
      .def(py::init<FloatValue, FloatValue>())
      .def_readwrite("width", &WHDims::width)
      .def_readwrite("height", &WHDims::height);

  py::class_<PointDiff>(m, "PointDiff")
      .def(py::init<FloatValue, FloatValue>())
      .def_readwrite("dx", &PointDiff::dx)
      .def_readwrite("dy", &PointDiff::dy);

  py::class_<SizeDiff>(m, "SizeDiff")
      .def(py::init<FloatValue, FloatValue>())
      .def_readwrite("dw", &SizeDiff::dw)
      .def_readwrite("dh", &SizeDiff::dh);

  py::class_<Point>(m, "Point")
      .def(py::init<FloatValue, FloatValue>())
      .def_readwrite("x", &Point::x)
      .def_readwrite("y", &Point::y)
      .def("__sub__", &Point::operator-);

  py::class_<BBox>(m, "BBox")
      .def(py::init<>())
      .def(py::init<FloatValue, FloatValue, FloatValue, FloatValue>())
      .def(py::init<const Point&, const WHDims&>())
      .def_readwrite("left", &BBox::left)
      .def_readwrite("top", &BBox::top)
      .def_readwrite("right", &BBox::right)
      .def_readwrite("bottom", &BBox::bottom)
      .def("width", &BBox::width)
      .def("height", &BBox::height)
      .def("aspect_ratio", &BBox::aspect_ratio)
      .def("clone", &BBox::clone)
      .def("center", &BBox::center)
      .def("make_scaled", &BBox::make_scaled)
      .def("validate", &BBox::validate)
      .def("empty", &BBox::empty);

  //
  // LivingBox Stuff
  //
}

void init_living_boxes(::pybind11::module_& m) {
  py::class_<ResizingConfig>(m, "ResizingConfig")
      .def(py::init<>())
      .def_readwrite("max_speed_w", &ResizingConfig::max_speed_w)
      .def_readwrite("max_speed_h", &ResizingConfig::max_speed_h)
      .def_readwrite("max_accel_w", &ResizingConfig::max_accel_w)
      .def_readwrite("max_accel_h", &ResizingConfig::max_accel_h)
      .def_readwrite("min_width", &ResizingConfig::min_width)
      .def_readwrite("min_height", &ResizingConfig::min_height)
      .def_readwrite("max_width", &ResizingConfig::max_width)
      .def_readwrite("max_height", &ResizingConfig::max_height)
      .def_readwrite(
          "stop_resizing_on_dir_change",
          &ResizingConfig::stop_resizing_on_dir_change)
      .def_readwrite(
          "resizing_stop_on_dir_change_delay",
          &ResizingConfig::resizing_stop_on_dir_change_delay)
      .def_readwrite(
          "resizing_cancel_stop_on_opposite_dir",
          &ResizingConfig::resizing_cancel_stop_on_opposite_dir)
      .def_readwrite(
          "resizing_stop_cancel_hysteresis_frames",
          &ResizingConfig::resizing_stop_cancel_hysteresis_frames)
      .def_readwrite(
          "resizing_stop_delay_cooldown_frames",
          &ResizingConfig::resizing_stop_delay_cooldown_frames)
      .def_readwrite(
          "resizing_time_to_dest_speed_limit_frames",
          &ResizingConfig::resizing_time_to_dest_speed_limit_frames)
      .def_readwrite(
          "resizing_time_to_dest_stop_speed_threshold",
          &ResizingConfig::resizing_time_to_dest_stop_speed_threshold)
      .def_readwrite("sticky_sizing", &ResizingConfig::sticky_sizing)
      .def_readwrite(
          "size_ratio_thresh_grow_dw",
          &ResizingConfig::size_ratio_thresh_grow_dw)
      .def_readwrite(
          "size_ratio_thresh_grow_dh",
          &ResizingConfig::size_ratio_thresh_grow_dh)
      .def_readwrite(
          "size_ratio_thresh_shrink_dw",
          &ResizingConfig::size_ratio_thresh_shrink_dw)
      .def_readwrite(
          "size_ratio_thresh_shrink_dh",
          &ResizingConfig::size_ratio_thresh_shrink_dh);

  py::class_<ResizingState>(m, "ResizingState")
      .def(py::init<>())
      .def_readonly("size_is_frozen", &ResizingState::size_is_frozen)
      .def_readonly("current_speed_w", &ResizingState::current_speed_w)
      .def_readonly("current_speed_h", &ResizingState::current_speed_h)
      // Resize stop-on-direction-change braking state
      .def_readonly("stop_delay_w", &ResizingState::stop_delay_w)
      .def_readonly(
          "stop_delay_w_counter", &ResizingState::stop_delay_w_counter)
      .def_readonly("stop_decel_w", &ResizingState::stop_decel_w)
      .def_readonly("stop_trigger_dir_w", &ResizingState::stop_trigger_dir_w)
      .def_readonly("cancel_opp_w_count", &ResizingState::cancel_opp_w_count)
      .def_readonly("cooldown_w_counter", &ResizingState::cooldown_w_counter)
      .def_readonly("stop_delay_h", &ResizingState::stop_delay_h)
      .def_readonly(
          "stop_delay_h_counter", &ResizingState::stop_delay_h_counter)
      .def_readonly("stop_decel_h", &ResizingState::stop_decel_h)
      .def_readonly("stop_trigger_dir_h", &ResizingState::stop_trigger_dir_h)
      .def_readonly("cancel_opp_h_count", &ResizingState::cancel_opp_h_count)
      .def_readonly("cooldown_h_counter", &ResizingState::cooldown_h_counter)
      .def_readonly("canceled_stop_w", &ResizingState::canceled_stop_w)
      .def_readonly("canceled_stop_h", &ResizingState::canceled_stop_h);

  py::class_<TranslatingBoxConfig>(m, "TranslatingBoxConfig")
      .def(py::init<>())
      .def_readwrite("max_speed_x", &TranslatingBoxConfig::max_speed_x)
      .def_readwrite("max_speed_y", &TranslatingBoxConfig::max_speed_y)
      .def_readwrite("max_accel_x", &TranslatingBoxConfig::max_accel_x)
      .def_readwrite("max_accel_y", &TranslatingBoxConfig::max_accel_y)
      .def_readwrite(
          "stop_translation_on_dir_change",
          &TranslatingBoxConfig::stop_translation_on_dir_change)
      .def_readwrite(
          "stop_translation_on_dir_change_delay",
          &TranslatingBoxConfig::stop_translation_on_dir_change_delay)
      .def_readwrite(
          "cancel_stop_on_opposite_dir",
          &TranslatingBoxConfig::cancel_stop_on_opposite_dir)
      .def_readwrite(
          "cancel_stop_hysteresis_frames",
          &TranslatingBoxConfig::cancel_stop_hysteresis_frames)
      .def_readwrite(
          "stop_delay_cooldown_frames",
          &TranslatingBoxConfig::stop_delay_cooldown_frames)
      .def_readwrite(
          "time_to_dest_speed_limit_frames",
          &TranslatingBoxConfig::time_to_dest_speed_limit_frames)
      .def_readwrite(
          "time_to_dest_stop_speed_threshold",
          &TranslatingBoxConfig::time_to_dest_stop_speed_threshold)
      .def_readwrite(
          "dynamic_acceleration_scaling",
          &TranslatingBoxConfig::dynamic_acceleration_scaling)
      .def_readwrite(
          "arena_angle_from_vertical",
          &TranslatingBoxConfig::arena_angle_from_vertical)
      .def_readwrite("arena_box", &TranslatingBoxConfig::arena_box)
      .def_readwrite(
          "sticky_translation", &TranslatingBoxConfig::sticky_translation)
      .def_readwrite(
          "sticky_size_ratio_to_frame_width",
          &TranslatingBoxConfig::sticky_size_ratio_to_frame_width)
      .def_readwrite(
          "sticky_translation_gaussian_mult",
          &TranslatingBoxConfig::sticky_translation_gaussian_mult)
      .def_readwrite(
          "unsticky_translation_size_ratio",
          &TranslatingBoxConfig::unsticky_translation_size_ratio)
      // .def_readwrite(
      //     "pan_smoothing_alpha",
      //     &TranslatingBoxConfig::pan_smoothing_alpha)
      .def_readwrite(
          "post_nonstop_stop_delay_count",
          &TranslatingBoxConfig::post_nonstop_stop_delay_count);

  py::class_<TranslationState>(m, "TranslationState")
      .def(py::init<>())
      .def_readonly("current_speed_x", &TranslationState::current_speed_x)
      .def_readonly("current_speed_y", &TranslationState::current_speed_y)
      .def_readonly(
          "translation_is_frozen", &TranslationState::translation_is_frozen)
      .def_readonly(
          "last_arena_edge_center_position_scale",
          &TranslationState::last_arena_edge_center_position_scale)
      .def_readonly("nonstop_delay", &TranslationState::nonstop_delay)
      .def_readonly(
          "nonstop_delay_counter", &TranslationState::nonstop_delay_counter)
      // Stop-on-direction-change braking state
      .def_readonly("stop_delay_x", &TranslationState::stop_delay_x)
      .def_readonly(
          "stop_delay_x_counter", &TranslationState::stop_delay_x_counter)
      .def_readonly("stop_decel_x", &TranslationState::stop_decel_x)
      .def_readonly("stop_trigger_dir_x", &TranslationState::stop_trigger_dir_x)
      .def_readonly("cancel_opp_x_count", &TranslationState::cancel_opp_x_count)
      .def_readonly("cooldown_x_counter", &TranslationState::cooldown_x_counter)
      .def_readonly("stop_delay_y", &TranslationState::stop_delay_y)
      .def_readonly(
          "stop_delay_y_counter", &TranslationState::stop_delay_y_counter)
      .def_readonly("stop_decel_y", &TranslationState::stop_decel_y)
      .def_readonly("stop_trigger_dir_y", &TranslationState::stop_trigger_dir_y)
      .def_readonly("cancel_opp_y_count", &TranslationState::cancel_opp_y_count)
      .def_readonly("cooldown_y_counter", &TranslationState::cooldown_y_counter)
      .def_readonly("canceled_stop_x", &TranslationState::canceled_stop_x)
      .def_readonly("canceled_stop_y", &TranslationState::canceled_stop_y);

  py::class_<LivingBoxConfig>(m, "LivingBoxConfig")
      .def(py::init<>())
      .def_readwrite("scale_dest_width", &LivingBoxConfig::scale_dest_width)
      .def_readwrite("scale_dest_height", &LivingBoxConfig::scale_dest_height)
      .def_readwrite("fixed_aspect_ratio", &LivingBoxConfig::fixed_aspect_ratio)
      .def_readwrite(
          "clamp_scaled_input_box", &LivingBoxConfig::clamp_scaled_input_box);

  py::class_<LivingState>(m, "LivingState")
      .def(py::init<>())
      .def_readwrite(
          "was_size_constrained", &LivingState::was_size_constrained);

  py::class_<
      AllLivingBoxConfig,
      ResizingConfig,
      TranslatingBoxConfig,
      LivingBoxConfig>(m, "AllLivingBoxConfig")
      .def(py::init<>());

#define PY_PURE_VIRTUAL_FUNCTION(_class$, _fn_name$, ...)              \
  _fn_name$, [](_class$& self, __VA_ARGS__) {                          \
    throw std::runtime_error("Pure virtual function called: " #_class$ \
                             "::" _fn_name$);                          \
  }

  py::class_<GrowShrink>(m, "GrowShrink")
      .def(py::init<>())
      .def_readonly("grow_width", &GrowShrink::grow_width)
      .def_readonly("grow_height", &GrowShrink::grow_height)
      .def_readonly("shrink_width", &GrowShrink::shrink_width)
      .def_readonly("shrink_height", &GrowShrink::shrink_height);

  py::class_<IBasicLivingBox, std::shared_ptr<IBasicLivingBox>>(
      m, "IBasicLivingBox")
      //.def(py::init<>())
      .def(PY_PURE_VIRTUAL_FUNCTION(IBasicLivingBox, "set_destination", BBox));

  py::class_<ILivingBox, IBasicLivingBox, std::shared_ptr<ILivingBox>>(
      m, "ILivingBox")
      .def(PY_PURE_VIRTUAL_FUNCTION(
          ILivingBox,
          "set_destination",
          const std::variant<BBox, std::shared_ptr<IBasicLivingBox>>&));

  py::class_<
      LivingBox,
      std::shared_ptr<LivingBox>
      //,IBasicLivingBox,
      // ILivingBox
      >(m, "LivingBox")
      .def(py::init<std::string, BBox, AllLivingBoxConfig>())
      .def("name", &LivingBox::name)
      .def("get_size_scale", &LivingBox::get_size_scale)
      .def("set_bbox", &LivingBox::set_bbox)
      .def("bounding_box", &LivingBox::bounding_box)
      .def(
          "forward",
          [](const std::shared_ptr<LivingBox>& self,
             const std::variant<BBox, std::shared_ptr<IBasicLivingBox>>& dest)
              -> BBox { return self->forward(dest); },
          py::call_guard<py::gil_scoped_release>())
      .def(
          "adjust_speed",
          &LivingBox::adjust_speed,
          py::arg("accel_x") = py::none(),
          py::arg("accel_y") = py::none(),
          py::arg("scale_constraints") = py::none(),
          py::arg("nonstop_delay") = py::none())
      .def(
          "scale_speed",
          &LivingBox::scale_speed,
          py::arg("ratio_x") = py::none(),
          py::arg("ratio_y") = py::none(),
          py::arg("clamp_to_max") = false)
      .def(
          "begin_stop_delay",
          &LivingBox::begin_stop_delay,
          py::arg("delay_x") = py::none(),
          py::arg("delay_y") = py::none())
      .def(
          "set_braking",
          &LivingBox::set_braking,
          py::arg("stop_on_dir_change_delay"),
          py::arg("cancel_on_opposite"),
          py::arg("cancel_hysteresis_frames"),
          py::arg("stop_delay_cooldown_frames"),
          py::arg("post_nonstop_stop_delay_count"))
      .def(
          "set_translation_constraints",
          &LivingBox::set_translation_constraints,
          py::arg("max_speed_x"),
          py::arg("max_speed_y"),
          py::arg("max_accel_x"),
          py::arg("max_accel_y"))
      .def("resizing_state", &LivingBox::ResizingBox::get_state)
      .def("resizing_config", &LivingBox::ResizingBox::get_config)
      .def("translation_state", &LivingBox::TranslatingBox::get_state)
      .def("translation_config", &LivingBox::TranslatingBox::get_config)
      .def("living_config", &LivingBox::config)
      .def("living_state", &LivingBox::get_live_box_state)
      .def("get_grow_shrink_wh", &LivingBox::get_grow_shrink_wh)
      .def(
          "get_sticky_translation_sizes",
          &LivingBox::get_sticky_translation_sizes);
}

void init_play_tracker(::pybind11::module_& m) {
  /**
   *  _  __      __  __
   * | |/ /     |  \/  |
   * | ' /______| \  / | ___   __ _ _ __   ___
   * |  <|______| |\/| |/ _ \ / _` | '_ \ / __|
   * | . \      | |  | |  __/| (_| | | | |\__ \
   * |_|\_\     |_|  |_|\___| \__,_|_| |_||___/
   *
   *
   */
  m.def(
      "compute_kmeans_clusters",
      [](const std::vector<float>& points,
         size_t num_clusters,
         size_t dim,
         size_t num_iterations)
          -> std::tuple<
              std::vector<int>,
              std::unordered_map<size_t, std::vector<size_t>>> {
        std::vector<int> assignments;
        compute_kmeans(
            points,
            num_clusters,
            dim,
            num_iterations,
            assignments,
            hm::kmeans::KMEANS_TYPE::KM_SEQ);
        assert(assignments.size() == points.size() / dim);
        std::unordered_map<size_t, std::vector<size_t>> assignment_map;
        for (std::size_t i = 0, n = assignments.size(); i < n; ++i) {
          size_t cluster_id = assignments[i];
          assignment_map[cluster_id].push_back(i);
        }
        return std::make_tuple(assignments, assignment_map);
      },
      py::arg("points"),
      py::arg("num_clusters"),
      py::arg("dim"),
      py::arg("num_iterations") = 6,
      py::call_guard<py::gil_scoped_release>());

  /**
   *  _____  _               _______              _
   * |  __ \| |             |__   __|            | |
   * | |__) | | __ _ _   _     | |_ __  __ _  ___| | __ ___  _ __
   * |  ___/| |/ _` | | | |    | | '__|/ _` |/ __| |/ // _ \| '__|
   * | |    | | (_| | |_| |    | | |  | (_| | (__|   <|  __/| |
   * |_|    |_|\__,_|\__, |    |_|_|   \__,_|\___|_|\_\\___||_|
   *                  __/ |
   *                 |___/
   */

  py::class_<Track>(m, "Track")
      .def(py::init<>())
      .def_readwrite("tracking_id", &Track::tracking_id)
      .def_readwrite("bbox", &Track::bbox);

  py::enum_<HmLogLevel>(m, "HmLogLevel")
      .value("DEBUG", HmLogLevel::kDebug)
      .value("INFO", HmLogLevel::kInfo)
      .value("WARNING", HmLogLevel::kWarning)
      .value("ERROR", HmLogLevel::kError)
      .export_values();

  py::class_<HmLogMessage>(m, "HmLogMessage")
      .def(py::init<>())
      .def_readwrite("level", &HmLogMessage::level)
      .def_readwrite("message", &HmLogMessage::message);

  py::class_<PlayDetectorConfig>(m, "PlayDetectorConfig")
      .def(py::init<>())
      .def_readwrite("fps_speed_scale", &PlayDetectorConfig::fps_speed_scale)
      .def_readwrite("max_positions", &PlayDetectorConfig::max_positions)
      .def_readwrite(
          "max_velocity_positions", &PlayDetectorConfig::max_velocity_positions)
      .def_readwrite("frame_step", &PlayDetectorConfig::frame_step)
      .def_readwrite(
          "min_considered_group_velocity",
          &PlayDetectorConfig::min_considered_group_velocity)
      .def_readwrite(
          "group_ratio_threshold", &PlayDetectorConfig::group_ratio_threshold)
      .def_readwrite(
          "group_velocity_speed_ratio",
          &PlayDetectorConfig::group_velocity_speed_ratio)
      .def_readwrite(
          "scale_speed_constraints",
          &PlayDetectorConfig::scale_speed_constraints)
      .def_readwrite(
          "nonstop_delay_count", &PlayDetectorConfig::nonstop_delay_count)
      .def_readwrite(
          "overshoot_scale_speed_ratio",
          &PlayDetectorConfig::overshoot_scale_speed_ratio)
      .def_readwrite(
          "overshoot_stop_delay_count",
          &PlayDetectorConfig::overshoot_stop_delay_count);

  py::class_<PlayTrackerConfig>(m, "PlayTrackerConfig")
      .def(py::init<>())
      .def_readwrite("no_wide_start", &PlayTrackerConfig::no_wide_start)
      .def_readwrite(
          "min_tracked_players", &PlayTrackerConfig::min_tracked_players)
      .def_readwrite("living_boxes", &PlayTrackerConfig::living_boxes)
      .def_readwrite(
          "ignore_largest_bbox", &PlayTrackerConfig::ignore_largest_bbox)
      .def_readwrite(
          "ignore_left_and_right_extremes",
          &PlayTrackerConfig::ignore_left_and_right_extremes)

      .def_readwrite(
          "ignore_outlier_players", &PlayTrackerConfig::ignore_outlier_players)
      .def_readwrite("play_detector", &PlayTrackerConfig::play_detector);

  py::class_<PlayDetectorResults>(m, "PlayDetectorResults")
      .def(py::init<>())
      .def_readonly(
          "breakaway_edge_center", &PlayDetectorResults::breakaway_edge_center);

  py::class_<PlayTrackerResults>(m, "PlayTrackerResults")
      .def(py::init<>())
      .def_readonly("cluster_boxes", &PlayTrackerResults::cluster_boxes)
      .def_readonly(
          "removed_cluster_outlier_box",
          &PlayTrackerResults::removed_cluster_outlier_box)
      .def_readonly("final_cluster_box", &PlayTrackerResults::final_cluster_box)
      .def_readonly("tracking_boxes", &PlayTrackerResults::tracking_boxes)
      .def_readonly("play_detection", &PlayTrackerResults::play_detection)
      .def_readonly(
          "largest_tracking_bbox", &PlayTrackerResults::largest_tracking_bbox)
      .def_readonly(
          "leftmost_tracking_bbox", &PlayTrackerResults::leftmost_tracking_bbox)
      .def_readonly(
          "rightmost_tracking_bbox",
          &PlayTrackerResults::rightmost_tracking_bbox)
      .def_readonly("log_messages", &PlayTrackerResults::log_messages);

  py::class_<PlayTracker, std::shared_ptr<PlayTracker>>(m, "PlayTracker")
      .def(
          py::init<BBox, PlayTrackerConfig>(),
          py::arg("initial_box"),
          py::arg("config"),
          py::call_guard<py::gil_scoped_release>())
      .def(
          "forward",
          &PlayTracker::forward,
          py::arg("tracking_ids"),
          py::arg("tracking_boxes"),
          py::arg("debug_to_stdout") = false,
          py::call_guard<py::gil_scoped_release>())
      .def("get_live_box", &PlayTracker::get_live_box, py::arg("index"))
      .def("set_bboxes", &PlayTracker::set_bboxes, py::arg("bboxes"))
      .def(
          "set_bboxes_scaled",
          &PlayTracker::set_bboxes_scaled,
          py::arg("bbox"),
          py::arg("scale_step"))
      .def(
          "set_breakaway_braking",
          &PlayTracker::set_breakaway_braking,
          py::arg("overshoot_delay_count"),
          py::arg("overshoot_scale_ratio"));
}

template <typename T_compute>
void init_cuda_pano(::pybind11::module_& m) {
  /**
   *   _____           _       _____
   *  / ____|         | |     |  __ \
   * | |     _   _  __| | __ _| |__) |__ _ _ __   ___
   * | |    | | | |/ _` |/ _` |  ___// _` | '_ \ / _ \
   * | |____| |_| | (_| | (_| | |   | (_| | | | | (_) |
   *  \_____|\__,_|\__,_|\__,_|_|    \__,_|_| |_|\___/
   *
   *
   */
  py::class_<
      PyCudaStitchPano<uchar4, T_compute>,
      std::shared_ptr<PyCudaStitchPano<uchar4, T_compute>>>(
      m, "CudaStitchPanoU8")
      .def(
          py::init<std::string, int, int, WHDims, WHDims, bool, int>(),
          py::arg("game_dir"),
          py::arg("batch_size"),
          py::arg("num_levels"),
          py::arg("input1"),
          py::arg("input2"),
          py::arg("minimize_blend"),
          py::arg("max_output_width") = 0)
      .def("canvas_width", &PyCudaStitchPano<uchar4, T_compute>::canvas_width)
      .def("canvas_height", &PyCudaStitchPano<uchar4, T_compute>::canvas_height)
      .def(
          "process",
          [](std::shared_ptr<PyCudaStitchPano<uchar4, T_compute>> self,
             at::Tensor& i1,
             at::Tensor& i2,
             at::Tensor& canvas,
             ptrdiff_t stream) {
            if (!i1.is_contiguous() || !i2.is_contiguous()) {
              throw std::runtime_error("Inputs should be contiguous");
            }
            if (!canvas.is_contiguous()) {
              throw std::runtime_error("Output should be contiguous");
            }
            if (!i1.device().is_cuda() || !i2.device().is_cuda() ||
                !canvas.device().is_cuda()) {
              throw std::runtime_error("All tensors must be Cuda tensors");
            }
            if (i1.dim() != 4 || i2.dim() != 4 || canvas.dim() != 4) {
              throw std::runtime_error(
                  "All tensors must have four dimensions (B, H, W, C)");
            }
            if (i1.size(3) != 4 || i2.size(3) != 4) {
              throw std::runtime_error("Tensors must have four channels");
            }
            if (canvas.size(3) != 4) {
              throw std::runtime_error("Canvas must have four channels");
            }
            self->process(
                i1.data_ptr(),
                i2.data_ptr(),
                canvas.data_ptr(),
                (cudaStream_t)stream);
          });
  py::class_<
      PyCudaStitchPano3<uchar4, T_compute>,
      std::shared_ptr<PyCudaStitchPano3<uchar4, T_compute>>>(
      m, "CudaStitchPano3U8")
      .def(py::init<std::string, int, int, std::vector<WHDims>>())
      .def("canvas_width", &PyCudaStitchPano3<uchar4, T_compute>::canvas_width)
      .def(
          "canvas_height", &PyCudaStitchPano3<uchar4, T_compute>::canvas_height)
      .def(
          "process",
          [](std::shared_ptr<PyCudaStitchPano3<uchar4, T_compute>> self,
             const std::vector<at::Tensor>& inputs,
             at::Tensor& canvas,
             ptrdiff_t stream) {
            if (!canvas.is_contiguous()) {
              throw std::runtime_error("Output should be contiguous");
            }
            if (!canvas.device().is_cuda()) {
              throw std::runtime_error("All tensors must be Cuda tensors");
            }
            if (canvas.dim() != 4) {
              throw std::runtime_error("Canvas must have four dimensions");
            }
            if (canvas.size(3) != 4) {
              throw std::runtime_error("Canvas must have four channels");
            }
            if (inputs.size() != 3) {
              throw std::runtime_error("Expected 3 input tensors");
            }
            std::vector<void*> data_ptrs;
            data_ptrs.reserve(inputs.size());
            for (const auto& t : inputs) {
              if (!t.is_contiguous()) {
                throw std::runtime_error("Inputs should be contiguous");
              }
              if (!t.device().is_cuda()) {
                throw std::runtime_error("All tensors must be Cuda tensors");
              }
              if (t.dim() != 4) {
                throw std::runtime_error(
                    "Input tensors must have four dimensions (B, H, W, C)");
              }
              if (t.size(3) != 4) {
                throw std::runtime_error("Tensors must have four channels");
              }
              data_ptrs.emplace_back(t.data_ptr());
            }
            self->process(data_ptrs, canvas.data_ptr(), (cudaStream_t)stream);
          });

  py::class_<
      PyCudaStitchPanoN<uchar4, T_compute>,
      std::shared_ptr<PyCudaStitchPanoN<uchar4, T_compute>>>(
      m, "CudaStitchPanoNU8")
      .def(
          py::init<std::string, int, int, std::vector<WHDims>, bool, bool>(),
          py::arg("game_dir"),
          py::arg("batch_size"),
          py::arg("num_levels"),
          py::arg("input_sizes"),
          py::arg("minimize_blend"),
          py::arg("quiet") = false)
      .def("canvas_width", &PyCudaStitchPanoN<uchar4, T_compute>::canvas_width)
      .def(
          "canvas_height", &PyCudaStitchPanoN<uchar4, T_compute>::canvas_height)
      .def(
          "process",
          [](std::shared_ptr<PyCudaStitchPanoN<uchar4, T_compute>> self,
             const std::vector<at::Tensor>& inputs,
             at::Tensor& canvas,
             ptrdiff_t stream) {
            if (!canvas.is_contiguous()) {
              throw std::runtime_error("Output should be contiguous");
            }
            if (!canvas.device().is_cuda()) {
              throw std::runtime_error("All tensors must be Cuda tensors");
            }
            if (canvas.dim() != 4) {
              throw std::runtime_error(
                  "Canvas must have four dimensions (B, H, W, C)");
            }
            if (canvas.size(3) != 4) {
              throw std::runtime_error("Canvas must have four channels");
            }
            std::vector<void*> data_ptrs;
            data_ptrs.reserve(inputs.size());
            for (const auto& t : inputs) {
              if (!t.is_contiguous()) {
                throw std::runtime_error("Inputs should be contiguous");
              }
              if (!t.device().is_cuda()) {
                throw std::runtime_error("All tensors must be Cuda tensors");
              }
              if (t.dim() != 4) {
                throw std::runtime_error(
                    "Input tensors must have four dimensions (B, H, W, C)");
              }
              if (t.size(3) != 4) {
                throw std::runtime_error("Tensors must have four channels");
              }
              data_ptrs.emplace_back(t.data_ptr());
            }
            self->process(data_ptrs, canvas.data_ptr(), (cudaStream_t)stream);
          });
  py::class_<
      PyCudaStitchPano<float4, T_compute>,
      std::shared_ptr<PyCudaStitchPano<float4, T_compute>>>(
      m, "CudaStitchPanoF32")
      .def(
          py::init<std::string, int, int, WHDims, WHDims, bool, int>(),
          py::arg("game_dir"),
          py::arg("batch_size"),
          py::arg("num_levels"),
          py::arg("input1"),
          py::arg("input2"),
          py::arg("minimize_blend"),
          py::arg("max_output_width") = 0)
      .def("canvas_width", &PyCudaStitchPano<float4, T_compute>::canvas_width)
      .def("canvas_height", &PyCudaStitchPano<float4, T_compute>::canvas_height)
      .def(
          "process",
          [](std::shared_ptr<PyCudaStitchPano<float4, T_compute>> self,
             at::Tensor& i1,
             at::Tensor& i2,
             at::Tensor& canvas,
             ptrdiff_t stream) {
            if (!i1.is_contiguous() || !i2.is_contiguous()) {
              throw std::runtime_error("Inputs should be contiguous");
            }
            if (!canvas.is_contiguous()) {
              throw std::runtime_error("Output should be contiguous");
            }
            if (!i1.device().is_cuda() || !i2.device().is_cuda() ||
                !canvas.device().is_cuda()) {
              throw std::runtime_error("All tensors must be Cuda tensors");
            }
            if (i1.dim() != 4 || i2.dim() != 4 || canvas.dim() != 4) {
              throw std::runtime_error(
                  "All tensors must have four dimensions (B, H, W, C)");
            }
            if (i1.size(3) != 4 || i2.size(3) != 4) {
              throw std::runtime_error("Tensors must have four channels");
            }
            if (canvas.size(3) != 4) {
              throw std::runtime_error("Canvas must have four channels");
            }
            self->process(
                i1.data_ptr(),
                i2.data_ptr(),
                canvas.data_ptr(),
                (cudaStream_t)stream);
          });
  py::class_<
      PyCudaStitchPano3<float4, T_compute>,
      std::shared_ptr<PyCudaStitchPano3<float4, T_compute>>>(
      m, "CudaStitchPano3F32")
      .def(py::init<std::string, int, int, std::vector<WHDims>>())
      .def("canvas_width", &PyCudaStitchPano3<float4, T_compute>::canvas_width)
      .def(
          "canvas_height", &PyCudaStitchPano3<float4, T_compute>::canvas_height)
      .def(
          "process",
          [](std::shared_ptr<PyCudaStitchPano3<float4, T_compute>> self,
             const std::vector<at::Tensor>& inputs,
             at::Tensor& canvas,
             ptrdiff_t stream) {
            if (!canvas.is_contiguous()) {
              throw std::runtime_error("Output should be contiguous");
            }
            if (!canvas.device().is_cuda()) {
              throw std::runtime_error("All tensors must be Cuda tensors");
            }
            if (canvas.dim() != 4) {
              throw std::runtime_error("Canvas must have four dimensions");
            }
            if (canvas.size(3) != 4) {
              throw std::runtime_error("Canvas must have four channels");
            }
            if (inputs.size() != 3) {
              throw std::runtime_error("Expected 3 input tensors");
            }
            std::vector<void*> data_ptrs;
            data_ptrs.reserve(inputs.size());
            for (const auto& t : inputs) {
              if (!t.is_contiguous()) {
                throw std::runtime_error("Inputs should be contiguous");
              }
              if (!t.device().is_cuda()) {
                throw std::runtime_error("All tensors must be Cuda tensors");
              }
              if (t.dim() != 4) {
                throw std::runtime_error(
                    "Input tensors must have four dimensions (B, H, W, C)");
              }
              if (t.size(3) != 4) {
                throw std::runtime_error("Tensors must have four channels");
              }
              data_ptrs.emplace_back(t.data_ptr());
            }
            self->process(data_ptrs, canvas.data_ptr(), (cudaStream_t)stream);
          });

  py::class_<
      PyCudaStitchPanoN<float4, T_compute>,
      std::shared_ptr<PyCudaStitchPanoN<float4, T_compute>>>(
      m, "CudaStitchPanoNF32")
      .def(
          py::init<std::string, int, int, std::vector<WHDims>, bool, bool>(),
          py::arg("game_dir"),
          py::arg("batch_size"),
          py::arg("num_levels"),
          py::arg("input_sizes"),
          py::arg("minimize_blend"),
          py::arg("quiet") = false)
      .def("canvas_width", &PyCudaStitchPanoN<float4, T_compute>::canvas_width)
      .def(
          "canvas_height", &PyCudaStitchPanoN<float4, T_compute>::canvas_height)
      .def(
          "process",
          [](std::shared_ptr<PyCudaStitchPanoN<float4, T_compute>> self,
             const std::vector<at::Tensor>& inputs,
             at::Tensor& canvas,
             ptrdiff_t stream) {
            if (!canvas.is_contiguous()) {
              throw std::runtime_error("Output should be contiguous");
            }
            if (!canvas.device().is_cuda()) {
              throw std::runtime_error("All tensors must be Cuda tensors");
            }
            if (canvas.dim() != 4) {
              throw std::runtime_error(
                  "Canvas must have four dimensions (B, H, W, C)");
            }
            if (canvas.size(3) != 4) {
              throw std::runtime_error("Canvas must have four channels");
            }
            std::vector<void*> data_ptrs;
            data_ptrs.reserve(inputs.size());
            for (const auto& t : inputs) {
              if (!t.is_contiguous()) {
                throw std::runtime_error("Inputs should be contiguous");
              }
              if (!t.device().is_cuda()) {
                throw std::runtime_error("All tensors must be Cuda tensors");
              }
              if (t.dim() != 4) {
                throw std::runtime_error(
                    "Input tensors must have four dimensions (B, H, W, C)");
              }
              if (t.size(3) != 4) {
                throw std::runtime_error("Tensors must have four channels");
              }
              data_ptrs.emplace_back(t.data_ptr());
            }
            self->process(data_ptrs, canvas.data_ptr(), (cudaStream_t)stream);
          });

  m.def(
      "show_cuda_tensor",
      [](const std::string& label,
         at::Tensor img_cuda,
         bool wait,
         std::optional<size_t> stream) -> void {
        cudaStream_t strm =
            stream.has_value() ? (cudaStream_t)*stream : (cudaStream_t) nullptr;
        show_cuda_tensor_impl(label, img_cuda, wait, strm);
      },
      py::arg("label"),
      py::arg("img"),
      py::arg("wait") = false,
      py::arg("stream") = py::none(),
      py::call_guard<py::gil_scoped_release>());

  m.def(
      "bgr_to_i420_cuda",
      [](at::Tensor bgr_hwc) { return hm::bgr_to_i420_cuda(bgr_hwc); },
      py::arg("bgr_hwc"),
      py::call_guard<py::gil_scoped_release>());

  m.def(
      "destroy_cuda_windows",
      []() { hm::display::destroy_global_render_set(); },
      py::call_guard<py::gil_scoped_release>());

  py::module_ atexit = py::module_::import("atexit");
  atexit.attr("register")(py::cpp_function(&hm::on_python_exit));
}

void init_aspen_graph_sampler(::pybind11::module_& m) {
  py::class_<hm::AspenGraphSampler>(m, "AspenGraphSampler")
      .def(
          py::init<size_t, int, int>(),
          py::arg("max_samples") = 24,
          py::arg("min_interval_ms") = 12,
          py::arg("max_interval_ms") = 40)
      .def(
          "configure_graph",
          &hm::AspenGraphSampler::configure_graph,
          py::arg("names"),
          py::arg("degrees"),
          py::arg("edges"))
      .def(
          "start",
          &hm::AspenGraphSampler::start,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "stop",
          &hm::AspenGraphSampler::stop,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "enter_index",
          &hm::AspenGraphSampler::enter_index,
          py::arg("index"),
          py::call_guard<py::gil_scoped_release>())
      .def(
          "exit_index",
          &hm::AspenGraphSampler::exit_index,
          py::arg("index"),
          py::call_guard<py::gil_scoped_release>())
      .def(
          "pop_samples",
          &hm::AspenGraphSampler::pop_samples,
          py::arg("max_items") = 1);
}

PYBIND11_MODULE(_hockeymom, m) {
  init_stitching(m);
  init_tracking(m);
  init_box_structures(m);
  init_living_boxes(m);
  init_play_tracker(m);
  init_cuda_pano<float4>(m);
  init_aspen_graph_sampler(m);
}
