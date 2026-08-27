#pragma once

#include "cuda_runtime_compat.h"

#ifdef JETSON_USE_HIP
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#else
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#endif

namespace hm {
namespace torch_cuda_compat {

#ifdef JETSON_USE_HIP
using DeviceGuard = c10::hip::HIPGuardMasqueradingAsCUDA;
using Stream = c10::hip::HIPStreamMasqueradingAsCUDA;
using StreamGuard = c10::hip::HIPStreamGuardMasqueradingAsCUDA;

inline Stream get_current_torch_stream(c10::DeviceIndex device_index = -1) {
  return c10::hip::getCurrentHIPStreamMasqueradingAsCUDA(device_index);
}

inline Stream get_stream_from_external(
    cudaStream_t stream,
    c10::DeviceIndex device_index) {
  return c10::hip::getStreamFromExternalMasqueradingAsCUDA(
      stream, device_index);
}
#else
using DeviceGuard = c10::cuda::CUDAGuard;
using Stream = c10::cuda::CUDAStream;
using StreamGuard = c10::cuda::CUDAStreamGuard;

inline Stream get_current_torch_stream(c10::DeviceIndex device_index = -1) {
  return c10::cuda::getCurrentCUDAStream(device_index);
}

inline Stream get_stream_from_external(
    cudaStream_t stream,
    c10::DeviceIndex device_index) {
  return c10::cuda::getStreamFromExternal(stream, device_index);
}
#endif

inline cudaStream_t get_current_stream(c10::DeviceIndex device_index = -1) {
  return get_current_torch_stream(device_index).stream();
}

} // namespace torch_cuda_compat
} // namespace hm
