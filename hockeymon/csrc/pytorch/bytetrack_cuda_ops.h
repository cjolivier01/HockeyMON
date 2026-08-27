#pragma once

#include <torch/torch.h>

namespace hm {
namespace ops {

// Compute IoU matrix between two sets of boxes on CUDA.
// boxes1: [N, 4] (x1, y1, x2, y2)
// boxes2: [M, 4]
// Returns [N, M] float32 tensor.
at::Tensor bbox_iou_cuda(
    const at::Tensor& boxes1,
    const at::Tensor& boxes2);

// Hungarian assignment on CUDA.
// cost: [n_rows, n_cols] float tensor.
// Returns pair (row_assignment, col_assignment) where:
//   row_assignment: len = n_rows, values are column indices or -1.
//   col_assignment: len = n_cols, values are row indices或 -1.
std::pair<at::Tensor, at::Tensor> hungarian_assign_cuda(
    const at::Tensor& cost,
    int64_t num_rows,
    int64_t num_cols,
    double cost_limit);

} // namespace ops
} // namespace hm
