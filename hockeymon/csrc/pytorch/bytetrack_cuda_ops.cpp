#include "hockeymon/csrc/pytorch/bytetrack_cuda_ops.h"

#include <torch/library.h>
#include <algorithm>

namespace hm {
namespace ops {

void launch_bbox_iou_cuda(
    const at::Tensor& boxes1,
    const at::Tensor& boxes2,
    at::Tensor& output);

void launch_hungarian_cuda(
    const at::Tensor& cost_square,
    int64_t padded_size,
    int64_t num_rows,
    int64_t num_cols,
    float cost_limit,
    at::Tensor& row_solution,
    at::Tensor& col_solution);

at::Tensor bbox_iou_cuda(const at::Tensor& boxes1, const at::Tensor& boxes2) {
  TORCH_CHECK(boxes1.device().is_cuda(), "boxes1 must be CUDA tensor");
  TORCH_CHECK(boxes2.device().is_cuda(), "boxes2 must be CUDA tensor");
  TORCH_CHECK(
      boxes1.size(-1) == 4 && boxes2.size(-1) == 4,
      "boxes must have shape (N, 4)");
  auto n = boxes1.size(0);
  auto m = boxes2.size(0);
  auto options = boxes1.options();
  auto output = at::empty({n, m}, options);
  if (n == 0 || m == 0) {
    return output.zero_();
  }
  launch_bbox_iou_cuda(boxes1.contiguous(), boxes2.contiguous(), output);
  return output;
}

std::pair<at::Tensor, at::Tensor> hungarian_assign_cuda(
    const at::Tensor& cost,
    int64_t num_rows,
    int64_t num_cols,
    double cost_limit) {
  TORCH_CHECK(cost.device().is_cuda(), "cost must be CUDA tensor");
  TORCH_CHECK(cost.dim() == 2, "cost matrix must be 2D");
  TORCH_CHECK(cost.scalar_type() == at::kFloat, "cost matrix must be float32");
  if (num_rows == 0 || num_cols == 0) {
    auto empty_rows = at::full({num_rows}, -1, cost.options().dtype(at::kLong));
    auto empty_cols = at::full({num_cols}, -1, cost.options().dtype(at::kLong));
    return {empty_rows, empty_cols};
  }

  auto padded = std::max<int64_t>(num_rows, num_cols);
  auto options_int = cost.options().dtype(at::kLong);
  auto options_int32 = cost.options().dtype(at::kInt);

  // Build padded square matrix
  const double limit = cost_limit;
  auto fill_value = (limit < 1e30) ? static_cast<float>(limit * 0.5) : 1e6f;
  auto cost_square = at::full({padded, padded}, fill_value, cost.options());
  cost_square.slice(/*dim=*/0, 0, num_rows)
      .slice(/*dim=*/1, 0, num_cols)
      .copy_(cost);
  // When both row/col padded portions overlap, set to zero like CPU version.
  if (padded > num_rows && padded > num_cols) {
    cost_square.slice(0, num_rows, padded).slice(1, num_cols, padded).zero_();
  }

  auto row_solution_int = at::full({num_rows}, -1, options_int32);
  auto col_solution_int = at::full({num_cols}, -1, options_int32);
  launch_hungarian_cuda(
      cost_square.contiguous(),
      padded,
      num_rows,
      num_cols,
      static_cast<float>(limit),
      row_solution_int,
      col_solution_int);
  auto row_solution = row_solution_int.to(options_int.dtype(at::kLong));
  auto col_solution = col_solution_int.to(options_int.dtype(at::kLong));

  if (limit < 1e30) {
    auto valid_rows = (row_solution >= 0).nonzero().squeeze(-1);
    if (valid_rows.numel() > 0) {
      auto det_cols = row_solution.index_select(0, valid_rows);
      auto gathered = cost.index({valid_rows, det_cols}).reshape({-1});
      auto invalid = (gathered > cost_limit).nonzero().squeeze(-1);
      if (invalid.numel() > 0) {
        auto bad_rows = valid_rows.index_select(0, invalid);
        row_solution.index_put_({bad_rows}, -1);
        auto bad_cols = det_cols.index_select(0, invalid);
        col_solution.index_put_({bad_cols}, -1);
      }
    }
  }

  return {row_solution, col_solution};
}

} // namespace ops
} // namespace hm
