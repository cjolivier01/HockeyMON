#pragma once

#include "BYTETrackerCuda.h"

#include <torch/torch.h>

namespace hm {
namespace tracker {

class BYTETrackerCudaStatic : public BYTETrackerCuda {
 public:
  explicit BYTETrackerCudaStatic(
      ByteTrackConfig config,
      int64_t max_detections,
      int64_t max_tracks,
      c10::Device device = c10::Device(c10::kCUDA, 0));

  ~BYTETrackerCudaStatic() = default;

  std::size_t num_tracks() const {
    return BYTETrackerCuda::num_tracks();
  }

  std::unordered_map<std::string, at::Tensor> track(
      std::unordered_map<std::string, at::Tensor>&& data) override;

  int64_t max_detections() const {
    return max_detections_;
  }

  int64_t max_tracks() const {
    return max_tracks_;
  }

 protected:
  at::Tensor mask_indices(const at::Tensor& mask) const override;

 private:
  int64_t max_detections_;
  int64_t max_tracks_;
};

} // namespace tracker
} // namespace hm
