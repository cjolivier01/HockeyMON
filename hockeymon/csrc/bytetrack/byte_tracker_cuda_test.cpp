#include "BYTETracker.h"
#include "BYTETrackerCuda.h"

#include <torch/torch.h>

using hm::tracker::BYTETracker;
using hm::tracker::BYTETrackerCuda;
using hm::tracker::ByteTrackConfig;

namespace {

constexpr const char* kIds = "ids";
constexpr const char* kBBoxes = "bboxes";
constexpr const char* kScores = "scores";

std::unordered_map<std::string, at::Tensor> make_data(
    int64_t frame_id,
    const std::vector<std::array<float, 4>>& boxes,
    const std::vector<int64_t>& labels,
    const std::vector<float>& scores,
    const c10::Device& device) {
  auto options_f = at::TensorOptions().dtype(at::kFloat).device(device);
  auto options_l = at::TensorOptions().dtype(at::kLong).device(device);
  auto options_s = at::TensorOptions().dtype(at::kFloat).device(device);

  std::vector<float> flat_boxes;
  flat_boxes.reserve(boxes.size() * 4);
  for (const auto& b : boxes) {
    flat_boxes.insert(flat_boxes.end(), b.begin(), b.end());
  }

  auto bboxes = at::from_blob(flat_boxes.data(), {static_cast<long>(boxes.size()), 4}).clone().to(options_f);
  auto labels_t = at::from_blob(
                      const_cast<int64_t*>(labels.data()),
                      {static_cast<long>(labels.size())},
                      at::TensorOptions().dtype(at::kLong))
                      .clone()
                      .to(options_l);
  auto scores_t = at::from_blob(
                      const_cast<float*>(scores.data()),
                      {static_cast<long>(scores.size())},
                      at::TensorOptions().dtype(at::kFloat))
                      .clone()
                      .to(options_s);

  std::unordered_map<std::string, at::Tensor> data;
  data[BYTETracker::kFrameId] =
      at::tensor({frame_id}, options_l);
  data[BYTETracker::kBBoxes] = bboxes;
  data[BYTETracker::kLabels] = labels_t;
  data[BYTETracker::kScores] = scores_t;
  return data;
}

void expect_close(
    const at::Tensor& lhs,
    const at::Tensor& rhs,
    float tol = 1e-4f) {
  TORCH_CHECK(
      lhs.sizes() == rhs.sizes(),
      "Tensor shape mismatch: ",
      lhs.sizes(),
      " vs ",
      rhs.sizes());
  TORCH_CHECK(
      at::allclose(lhs, rhs, tol, tol),
      "Tensor mismatch");
}

} // namespace

int main() {
  if (!torch::cuda::is_available()) {
    return 0;
  }

  ByteTrackConfig config;
  BYTETracker cpu_tracker(config);
  BYTETrackerCuda gpu_tracker(config, c10::Device("cuda:0"));

  std::vector<std::array<float, 4>> frame0_boxes = {
      {10.f, 10.f, 30.f, 40.f},
      {100.f, 100.f, 140.f, 160.f},
  };
  std::vector<int64_t> frame0_labels = {1, 1};
  std::vector<float> frame0_scores = {0.9f, 0.85f};

  auto cpu_data0 = make_data(0, frame0_boxes, frame0_labels, frame0_scores, c10::Device(c10::kCPU));
  auto gpu_data0 = make_data(0, frame0_boxes, frame0_labels, frame0_scores, c10::Device(c10::kCUDA));

  auto cpu_res0 = cpu_tracker.track(std::move(cpu_data0));
  auto gpu_res0 = gpu_tracker.track(std::move(gpu_data0));

  expect_close(cpu_res0[kIds], gpu_res0[kIds].cpu().to(at::kLong));
  expect_close(cpu_res0[kBBoxes], gpu_res0[kBBoxes].cpu());

  std::vector<std::array<float, 4>> frame1_boxes = {
      {12.f, 12.f, 32.f, 42.f},
      {102.f, 102.f, 142.f, 162.f},
  };
  std::vector<int64_t> frame1_labels = {1, 1};
  std::vector<float> frame1_scores = {0.92f, 0.8f};

  auto cpu_data1 = make_data(1, frame1_boxes, frame1_labels, frame1_scores, c10::Device(c10::kCPU));
  auto gpu_data1 = make_data(1, frame1_boxes, frame1_labels, frame1_scores, c10::Device(c10::kCUDA));

  auto cpu_res1 = cpu_tracker.track(std::move(cpu_data1));
  auto gpu_res1 = gpu_tracker.track(std::move(gpu_data1));

  expect_close(cpu_res1[kIds], gpu_res1[kIds].cpu().to(at::kLong));
  expect_close(cpu_res1[kBBoxes], gpu_res1[kBBoxes].cpu());

  return 0;
}
