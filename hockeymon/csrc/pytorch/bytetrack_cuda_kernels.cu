#include <ATen/Dispatch.h>
#include <ATen/Tensor.h>

#include "hockeymon/csrc/pytorch/torch_cuda_compat.h"

namespace hm {
namespace ops {
namespace {

template <typename scalar_t>
__device__ inline scalar_t intersection_area(
    const scalar_t* a,
    const scalar_t* b) {
  const scalar_t x1 = max(a[0], b[0]);
  const scalar_t y1 = max(a[1], b[1]);
  const scalar_t x2 = min(a[2], b[2]);
  const scalar_t y2 = min(a[3], b[3]);
  const scalar_t w = max(static_cast<scalar_t>(0), x2 - x1);
  const scalar_t h = max(static_cast<scalar_t>(0), y2 - y1);
  return w * h;
}

template <typename scalar_t>
__global__ void IoUKernel(
    const scalar_t* __restrict__ boxes1,
    const scalar_t* __restrict__ boxes2,
    scalar_t* __restrict__ output,
    int64_t n,
    int64_t m) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row >= n || col >= m) {
    return;
  }
  const scalar_t* box1 = boxes1 + row * 4;
  const scalar_t* box2 = boxes2 + col * 4;
  const scalar_t area1 = max(static_cast<scalar_t>(0), box1[2] - box1[0]) *
      max(static_cast<scalar_t>(0), box1[3] - box1[1]);
  const scalar_t area2 = max(static_cast<scalar_t>(0), box2[2] - box2[0]) *
      max(static_cast<scalar_t>(0), box2[3] - box2[1]);
  const scalar_t inter = intersection_area(box1, box2);
  const scalar_t denom =
      max(static_cast<scalar_t>(1e-6), area1 + area2 - inter);
  output[row * m + col] = inter / denom;
}

constexpr int kHungarianBlockDim = 32;

__global__ void HungarianKernel(
    const float* __restrict__ cost,
    int64_t padded_n,
    int64_t num_rows,
    int64_t num_cols,
    float cost_limit,
    int* __restrict__ row_solution,
    int* __restrict__ col_solution) {
  if (threadIdx.x != 0 || blockIdx.x != 0) {
    return;
  }
  const int n = static_cast<int>(padded_n);
  const float INF = 1e12f;

  extern __shared__ unsigned char shared[];
  float* u = reinterpret_cast<float*>(shared);
  float* v = u + (n + 1);
  float* minv = v + (n + 1);
  int* p = reinterpret_cast<int*>(minv + (n + 1));
  int* way = p + (n + 1);
  unsigned char* used = reinterpret_cast<unsigned char*>(way + (n + 1));

  for (int i = 0; i <= n; ++i) {
    u[i] = 0.0f;
    v[i] = 0.0f;
    p[i] = 0;
    way[i] = 0;
  }

  for (int i = 1; i <= n; ++i) {
    p[0] = i;
    int j0 = 0;
    for (int j = 0; j <= n; ++j) {
      minv[j] = INF;
      used[j] = 0;
    }
    do {
      used[j0] = 1;
      int i0 = p[j0];
      float delta = INF;
      int j1 = 0;
      for (int j = 1; j <= n; ++j) {
        if (used[j]) {
          continue;
        }
        float cur = cost[(i0 - 1) * n + (j - 1)] - u[i0] - v[j];
        if (cur < minv[j]) {
          minv[j] = cur;
          way[j] = j0;
        }
        if (minv[j] < delta) {
          delta = minv[j];
          j1 = j;
        }
      }
      for (int j = 0; j <= n; ++j) {
        if (used[j]) {
          u[p[j]] += delta;
          v[j] -= delta;
        } else {
          minv[j] -= delta;
        }
      }
      j0 = j1;
    } while (p[j0] != 0);
    do {
      int j1 = way[j0];
      p[j0] = p[j1];
      j0 = j1;
    } while (j0);
  }

  for (int r = 0; r < num_rows; ++r) {
    row_solution[r] = -1;
  }
  for (int c = 0; c < num_cols; ++c) {
    col_solution[c] = -1;
  }

  for (int j = 1; j <= n; ++j) {
    if (p[j] <= 0) {
      continue;
    }
    int row = p[j] - 1;
    int col = j - 1;
    if (row < num_rows && col < num_cols) {
      row_solution[row] = col;
      col_solution[col] = row;
    } else if (row < num_rows) {
      row_solution[row] = -1;
    } else if (col < num_cols) {
      col_solution[col] = -1;
    }
  }
}

} // namespace

void launch_bbox_iou_cuda(
    const at::Tensor& boxes1,
    const at::Tensor& boxes2,
    at::Tensor& output) {
  const auto n = boxes1.size(0);
  const auto m = boxes2.size(0);

  const dim3 threads(16, 16);
  const dim3 blocks(
      (m + threads.x - 1) / threads.x, (n + threads.y - 1) / threads.y);

  AT_DISPATCH_FLOATING_TYPES(
      boxes1.scalar_type(), "bytetrack_bbox_iou_cuda", [&] {
        IoUKernel<scalar_t>
            <<<blocks,
               threads,
               0,
               hm::torch_cuda_compat::get_current_stream()>>>(
                boxes1.data_ptr<scalar_t>(),
                boxes2.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                n,
                m);
      });
}

void launch_hungarian_cuda(
    const at::Tensor& cost_square,
    int64_t padded_size,
    int64_t num_rows,
    int64_t num_cols,
    float cost_limit,
    at::Tensor& row_solution,
    at::Tensor& col_solution) {
  const int64_t shared_bytes = (3 * (padded_size + 1) * sizeof(float)) +
      (2 * (padded_size + 1) * sizeof(int)) +
      ((padded_size + 1) * sizeof(unsigned char));
  HungarianKernel<<<
      1,
      1,
      shared_bytes,
      hm::torch_cuda_compat::get_current_stream()>>>(
      cost_square.data_ptr<float>(),
      padded_size,
      num_rows,
      num_cols,
      cost_limit,
      row_solution.data_ptr<int>(),
      col_solution.data_ptr<int>());
}

} // namespace ops
} // namespace hm
