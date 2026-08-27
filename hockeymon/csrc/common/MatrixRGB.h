#pragma once

#include <vigra/impex.hxx>
#include <vigra/multi_array.hxx>

// TODO: remove pybind dependency int his file/class

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <vector>

#include <ATen/ATen.h>
#include <torch/csrc/autograd/generated/variable_factories.h>

namespace py = pybind11;

namespace hm {
/**
 * @brief Class to carry python array for RGB data into the blend function
 *        (in-place)
 */
// template <std::size_t CHANNELS>
class __attribute__((visibility("default"))) MatrixImage {
 public:
  static constexpr std::size_t kPixelSampleSize = sizeof(std::uint8_t);

  MatrixImage() {}
  MatrixImage(const MatrixImage&) = delete;
  MatrixImage(MatrixImage&& other) = default;

  MatrixImage(
      py::array_t<uint8_t>& input_image,
      std::size_t xpos,
      std::size_t ypos,
      bool copy_data = false) {
    // Access the data and information from the NumPy array
    std::size_t ndims = input_image.ndim();
    auto dtype = input_image.dtype();
    // Check if the input is a 3D array with dtype uint8 (RGB image)
    if (ndims != 3 /*|| !dtype.is(py::dtype::of<uint8_t>())*/) {
      throw std::runtime_error("Input must be a 3D uint8 RGB image array");
    }

    auto py_buffer_info = input_image.request();

    // Get the dimensions
    rows_ = py_buffer_info.shape[0];
    cols_ = py_buffer_info.shape[1];
    channels_ = py_buffer_info.shape[2];
    assert(channels_ == 3 || channels_ == 4);
    strides_ = {py_buffer_info.strides.begin(), py_buffer_info.strides.end()};
    x_pos_ = xpos;
    y_pos_ = ypos;
    // need to fiogure out how to get the proper byte size
    assert(storage_channel_count() == channels());
    // Is it ever read-only?
    assert(!py_buffer_info.readonly);
    if ((copy_data || py_buffer_info.readonly)) {
      std::cout << "Warning: copying tensor data for MatrixImage.\n";
      // assert(false);
      std::size_t image_bytes =
          sizeof(std::uint8_t) * rows_ * cols_ * storage_channel_count();
      // std::cout << "image_bytes=" << image_bytes << std::endl;
      data_ = new std::uint8_t[image_bytes];
      memcpy(data_, py_buffer_info.ptr, image_bytes);
      m_own_data = true;
      is_python_data = false;
    } else {
      data_ = static_cast<uint8_t*>(py_buffer_info.ptr);
      input_image.release();
      m_own_data = true;
      is_python_data = true;
    }
  }

  MatrixImage(
      at::Tensor tensor,
      std::size_t xpos,
      std::size_t ypos,
      bool copy_data = false) {
    // Access the data and information from the NumPy array
    std::size_t ndims = tensor.sizes().size();
    auto dtype = tensor.dtype();
    // Check if the input is a 3D array with dtype uint8 (RGB image)
    if (ndims != 3 /*|| !dtype.is(py::dtype::of<uint8_t>())*/) {
      throw std::runtime_error("Input must be a 3D uint8 RGB image array");
    }

    // auto py_buffer_info = input_image.request();

    // Get the dimensions
    rows_ = tensor.size(0);
    cols_ = tensor.size(1);
    channels_ = tensor.size(2);
    auto strides = tensor.strides();
    strides_ = {strides.begin(), strides.end()};
    x_pos_ = xpos;
    y_pos_ = ypos;
    // need to fiogure out how to get the proper byte size
    assert(storage_channel_count() == channels());
    // Is it ever read-only?
    if (copy_data) {
      assert(false);
      std::size_t image_bytes =
          sizeof(std::uint8_t) * rows_ * cols_ * storage_channel_count();
      data_ = new std::uint8_t[image_bytes];
      assert(image_bytes == tensor.nbytes());
      memcpy(data_, tensor.data_ptr<uint8_t>(), image_bytes);
      m_own_data = true;
      is_python_data = false;
    } else {
      m_own_data = false;
      is_python_data = false;
      tensor_ = tensor.cpu().clone();
      at_tensor_storage_ = tensor_.storage();
      at_tensor_storage_offset_ = tensor_.storage_offset();
      data_ = tensor_.data_ptr<uint8_t>();
      assert(
          data_ ==
          static_cast<std::uint8_t*>(tensor_.data_ptr<uint8_t>()) +
              at_tensor_storage_offset_);
      assert(
          at_tensor_storage_.nbytes() ==
          rows() * cols() * channels() * kPixelSampleSize);
      has_tensor_ = true;
    }
  }

  MatrixImage(
      size_t rows,
      size_t cols,
      size_t channels,
      std::vector<std::size_t> strides = {})
      : rows_(rows), cols_(cols), channels_(channels) {
    data_ = new std::uint8_t[rows * cols * channels_ * kPixelSampleSize];
    if (!strides.empty()) {
      strides_ = strides;
    } else {
      strides_ = {
          channels_ * kPixelSampleSize *
              cols_ /* Strides (in bytes) for each index */,
          channels_ * kPixelSampleSize,
          kPixelSampleSize};
    }
    m_own_data = true;
    is_python_data = false;
  }

  MatrixImage(
      size_t rows,
      size_t cols,
      size_t channels,
      std::uint8_t* consume_data)
      : rows_(rows), cols_(cols), channels_(channels) {
    data_ = consume_data;
    m_own_data = true;
    strides_ = {
        channels_ * kPixelSampleSize *
            cols_ /* Strides (in bytes) for each index */,
        channels_ * kPixelSampleSize,
        kPixelSampleSize};
  }
  virtual ~MatrixImage() {
    if (data_ && m_own_data) {
      if (!is_python_data) {
        delete[] data_;
      } else {
        free(data_);
      }
    }
  }
  std::vector<std::size_t> xy_pos() const {
    return {x_pos_, y_pos_};
  }
  void set_xy_pos(std::size_t xpos, std::size_t ypos) {
    x_pos_ = xpos;
    y_pos_ = ypos;
  }
  constexpr std::uint8_t* data() {
    return data_;
  }
  constexpr size_t rows() const {
    return rows_;
  }
  constexpr size_t cols() const {
    return cols_;
  }
  constexpr size_t channels() {
    return channels_;
  }

  std::size_t storage_channel_count() const {
    return strides_.empty() ? channels_ : std::max(channels_, strides_.at(1));
  }

  py::array_t<std::uint8_t> to_py_array() {
    auto capsule = py::capsule(data_, [](void* data) {
      assert(data);
      delete[] reinterpret_cast<std::uint8_t*>(data);
    });
    if (!data_ || !m_own_data) {
      std::cout << "rh270377f7yh23f" << std::endl;
    }
    assert(data_ && m_own_data);
    assert(!strides_.empty());

    py::array_t<std::uint8_t> result(
        {rows(),
         cols(),
         channels() * kPixelSampleSize} /* total buffer size in bytes */,
        std::vector<long>{strides_.begin(), strides_.end()},
        data_,
        std::move(capsule));
    m_own_data = false;
    is_python_data = false;
    data_ = nullptr;
    return result;
  }

  at::Tensor to_tensor() {
    assert(data_);
    assert(!strides_.empty());

    at::Tensor result;
    if (m_own_data) {
      std::function<void(void*)> deleter = [](void* p) {
        assert(p);
        delete[] reinterpret_cast<std::uint8_t*>(p);
      };
      result = torch::from_blob(
          data_,
          {(int)rows(),
           (int)cols(),
           (int)channels()} /* total buffer size in bytes */,
          std::vector<long>{strides_.begin(), strides_.end()},
          deleter,
          at::TensorOptions().dtype(at::ScalarType::Byte));
      m_own_data = false;
    } else {
      if (has_tensor_) {
        std::size_t use_count = at_tensor_storage_.use_count();
        assert(at_tensor_storage_.data());
        assert(
            at_tensor_storage_.nbytes() ==
            rows() * cols() * channels() * kPixelSampleSize);
        try {
          result = tensor_.set_(
              at_tensor_storage_,
              at_tensor_storage_offset_,
              {(int)rows(),
               (int)cols(),
               (int)channels()} /* total buffer size in bytes */,
              std::vector<long>{strides_.begin(), strides_.end()});
        } catch (std::exception& e) {
          std::cerr << e.what();
        }
        use_count = at_tensor_storage_.use_count();
        at_tensor_storage_ = at::Storage();
      } else {
        assert(false);
        result = torch::from_blob(
            data_,
            {(int)rows(),
             (int)cols(),
             (int)channels()} /* total buffer size in bytes */,
            std::vector<long>{strides_.begin(), strides_.end()},
            at::TensorOptions().dtype(at::ScalarType::Byte));
      }
    }
    is_python_data = false;
    data_ = nullptr;
    return result;
  }

  std::unique_ptr<vigra::BRGBImage> to_vigra_image() {
    assert(
        channels() ==
        storage_channel_count()); // we dont handle stride-hidden alpha channel
    if (channels() == 3) {
      const vigra::RGBValue<unsigned char>* rgb_data =
          reinterpret_cast<vigra::RGBValue<unsigned char>*>(data());
      return std::make_unique<vigra::BRGBImage>(cols(), rows(), rgb_data);
    } else if (channels() == 4) {
      assert(false);
      const vigra::RGBValue<unsigned char>* rgba_data =
          reinterpret_cast<vigra::RGBValue<unsigned char>*>(data());
      return std::make_unique<vigra::BRGBImage>(cols(), rows(), rgba_data);
    } else {
      assert(false);
    }
    return nullptr;
  }

  constexpr const std::vector<std::size_t>& strides() const {
    return strides_;
  }

 private:
  // PyTorch
  at::Tensor tensor_;
  at::Storage at_tensor_storage_;
  std::size_t at_tensor_storage_offset_{0};
  bool has_tensor_{false};
  // Numpy
  // py::buffer_info buffer_info_;
  // bool has_buffer_info_{false};
  // State info
  bool m_own_data{false};
  bool is_python_data{false};
  size_t rows_{0}, cols_{0}, channels_{0};
  std::vector<std::size_t> strides_;
  std::uint8_t* data_{nullptr};
  std::size_t x_pos_{0};
  std::size_t y_pos_{0};
};

using MatrixRGB = MatrixImage;

template <std::size_t CHANNELS>
struct MatrixEncoder : public vigra::Encoder {
 public:
  virtual ~MatrixEncoder() = default;
  virtual void init(const std::string&) {}

  // initialize with file access mode. For codecs that do not support this
  // feature, the standard init is called.
  virtual void init(const std::string& fileName, const std::string&) {
    init(fileName);
  }

  virtual void close() {}
  virtual void abort() {
    assert(false);
  }

  virtual std::string getFileType() const {
    return "TIFF"; // not really TIFF but tell it we are so we get some more
                   // interesting info
  }
  virtual unsigned int getOffset() const {
    return CHANNELS;
  }

  virtual void setWidth(unsigned int width) {
    width_ = width;
  }
  virtual void setHeight(unsigned int height) {
    height_ = height;
  }
  virtual void setNumBands(unsigned int num_bands) {
    num_bands_ = num_bands;
  }
  virtual void setCompressionType(const std::string& type, int = -1) {}
  virtual void setPixelType(const std::string& pixel_type) {
    assert(pixel_type == "UINT8");
  }
  constexpr std::size_t width() const {
    return width_;
  }
  constexpr std::size_t height() const {
    return height_;
  }
  virtual void finalizeSettings() {
    std::size_t num_channels = CHANNELS;
    std::size_t total_image_size =
        MatrixRGB::kPixelSampleSize * width() * height() * num_channels;
    assert(total_image_size);
    data_ = std::make_unique<std::uint8_t[]>(total_image_size);
    scanlines_.resize(CHANNELS);
    for (std::size_t i = 0; i < CHANNELS; ++i) {
      scanlines_.at(i) = data_.get() + (MatrixRGB::kPixelSampleSize * i);
    }
  }
  virtual void setPosition(const vigra::Diff2D& pos) {
    position_ = pos;
  }
  virtual void setCanvasSize(const vigra::Size2D& size) {
    canvas_size_ = size;
  }

  virtual void setXResolution(float xres) {
    x_res_ = xres;
  }
  virtual void setYResolution(float yres) {
    y_res_ = yres;
  }

  typedef vigra::ArrayVector<unsigned char> ICCProfile;

  virtual void setICCProfile(const ICCProfile& data) {
    icc_profile_ = data;
  }

  virtual void* currentScanlineOfBand(unsigned int band) {
    return scanlines_[band];
  }

  virtual void nextScanline() {
    for (auto& ptr : scanlines_) {
      ptr += sizeof(std::uint8_t) * width_ * CHANNELS;
    }
  }
  std::unique_ptr<MatrixRGB> consume() {
    auto matrix =
        std::make_unique<MatrixRGB>(height_, width_, CHANNELS, data_.release());
    matrix->set_xy_pos(position_.x, position_.y);
    return matrix;
  }

 private:
  std::size_t width_{0}, height_{0};
  float x_res_{0}, y_res_{0};
  int num_bands_{0};
  ICCProfile icc_profile_;
  vigra::Diff2D position_;
  vigra::Size2D canvas_size_;
  std::unique_ptr<std::uint8_t[]> data_;
  std::unique_ptr<MatrixRGB> matrix_rgb_{nullptr};

 protected:
  std::vector<std::uint8_t*> scanlines_;
};

using MatrixEncoderRGB = MatrixEncoder<3>;
using MatrixEncoderRGBA = MatrixEncoder<4>;

} // namespace hm
