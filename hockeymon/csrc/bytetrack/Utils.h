#pragma once

#include <torch/torch.h>

#include <iostream>
#include <sstream>
#include <vector>

namespace hm {
namespace tracker {

constexpr const char* kBBoxes = "bboxes";
constexpr const char* kLabels = "labels";
constexpr const char* kScores = "scores";

template <typename MAP_T>
at::Tensor& get_map_tensor(
    const char* label,
    const std::string& key,
    MAP_T& map) {
  auto found = map.find(key);
  if (found == map.end()) {
    std::stringstream ss;
    ss << "Could not find the required key \"" << key << "\" in the given "
       << label << " map";
    std::cerr << ss.str() << std::endl;
    throw std::runtime_error(ss.str());
  }
  return found->second;
}

at::Tensor float_to_scalar_tensor(
    float value,
    at::ScalarType dtype = at::kFloat);

std::vector<float> tensor_to_vector(const at::Tensor& tensor);
std::vector<int64_t> tensor_to_int64_vector(const at::Tensor& tensor);

std::vector<int64_t> non_negative(const at::Tensor& tensor);

std::vector<float> to_float_vect(const at::Tensor& tensor);

std::vector<float> to_float_vect(const std::list<at::Tensor>& tensor_list);

at::Tensor bool_mask_select(const at::Tensor& tensor, const at::Tensor& mask);

template <typename T, typename CONTAINER_T>
std::vector<T> scalar_tensor_list_to_vector(const CONTAINER_T& tensor_list) {
  std::vector<T> items;
  items.reserve(tensor_list.size());
  for (const at::Tensor& t : tensor_list) {
    assert(t.numel() == 1 && (t.ndimension() == 0 || t.ndimension() == 1));
    items.emplace_back(t.item<T>());
  }
  return items;
}

extern const at::Device kCpuDevice;

// Function to convert tensor to a string representation with square brackets
std::string to_string(
    const at::Tensor& tensor,
    int depth = 0,
    int max_depth = -1);

void PT(const std::string& label, const at::Tensor& t);

#define _PT($tensor) PT(#$tensor, $tensor)

template <typename T>
at::Tensor vector_to_tensor(const std::vector<T>& vec, at::Device device) {
  // Create a tensor from the vector and infer the correct scalar type
  auto tensor =
      at::tensor(vec, at::dtype(torch::CppTypeToScalarType<T>::value));
  if (tensor.device() != device) {
    return tensor.to(device);
  }
  return tensor;
}

std::vector<float> tensor_to_vector_1d(const at::Tensor& tensor);
std::vector<std::vector<float>> tensor_to_vector_2d(const at::Tensor& tensor);
at::Tensor bbox_cxcyah_to_xyxy(const at::Tensor& bboxes);
at::Tensor bbox_xyxy_to_cxcyah(const at::Tensor& bboxes);
at::Tensor bbox_overlaps(
    at::Tensor bboxes1,
    at::Tensor bboxes2,
    const std::string& mode = "iou",
    bool is_aligned = false,
    float eps = 1e-6);
} // namespace tracker
} // namespace hm
