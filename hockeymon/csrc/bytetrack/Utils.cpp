#include "Utils.h"

namespace hm {
namespace tracker {

at::Tensor float_to_scalar_tensor(float value, at::ScalarType dtype) {
  return at::tensor(value, dtype);
}

std::vector<float> tensor_to_vector(const at::Tensor& tensor) {
  // Ensure tensor is a CPU tensor and is 1D
  assert(tensor.device().is_cpu());
  assert(tensor.dim() == 1);

  // Ensure the tensor is of type float
  assert(tensor.scalar_type() == at::kFloat);

  // Get a pointer to the tensor data
  auto data_ptr = tensor.data_ptr<float>();

  // Create a vector and copy the data
  std::vector<float> result(data_ptr, data_ptr + tensor.numel());

  return result;
}

std::vector<int64_t> tensor_to_int64_vector(const at::Tensor& tensor) {
  // Ensure tensor is a CPU tensor and is 1D
  assert(tensor.device().is_cpu());
  assert(tensor.dim() == 1);

  // Ensure the tensor is of type float
  assert(tensor.scalar_type() == at::kLong);

  // Get a pointer to the tensor data
  auto data_ptr = tensor.data_ptr<int64_t>();

  // Create a vector and copy the data
  std::vector<int64_t> result(data_ptr, data_ptr + tensor.numel());

  return result;
}

const at::Device kCpuDevice("cpu");

// Templated function to handle different element types of tensors
template <typename T>
void appendTensorElementsToStringStream(
    const at::Tensor& tensor,
    std::ostringstream& oss) {
  oss << "[";
  for (int64_t i = 0; i < tensor.size(0); ++i) {
    oss << tensor[i].item<T>();
    if (i < tensor.size(0) - 1) {
      oss << ", ";
    }
  }
  oss << "]";
}

// Function to convert tensor to a string representation with square brackets
std::string to_string(const at::Tensor& tensor, int depth, int max_depth) {
  std::ostringstream oss;
  if (max_depth == -1) {
    max_depth = tensor.dim();
  }
  if (tensor.dim() == 1) {
    // Handle 1D tensor based on its scalar type
    if (tensor.scalar_type() == at::ScalarType::Float) {
      appendTensorElementsToStringStream<float>(tensor, oss);
    } else if (tensor.scalar_type() == at::ScalarType::Bool) {
      appendTensorElementsToStringStream<bool>(tensor, oss);
    } else if (tensor.scalar_type() == at::ScalarType::Double) {
      appendTensorElementsToStringStream<double>(tensor, oss);
    } else if (tensor.scalar_type() == at::ScalarType::Int) {
      appendTensorElementsToStringStream<int>(tensor, oss);
    } else if (tensor.scalar_type() == at::ScalarType::Long) {
      appendTensorElementsToStringStream<int64_t>(tensor, oss);
    }
    // Add more types as needed
    else {
      oss << "Unsupported tensor type for conversion to string.\n";
    }
  } else {
    // Handle higher-dimensional tensors
    oss << "[";
    for (int64_t i = 0; i < tensor.size(0); ++i) {
      oss << to_string(tensor[i], depth + 1, max_depth);
      if (i < tensor.size(0) - 1) {
        if (tensor.dim() != max_depth) {
          // oss << ", ";
          oss << " ";
        } else {
          oss << "\n";
        }
      }
    }
    oss << "]";
  }

  return oss.str();
}

void PT(const std::string& label, const at::Tensor& t) {
  std::stringstream ss;
  ss << label << ": " << t.sizes() << ": ";
  if (t.dim() > 1) {
    ss << "\n";
  }
  ss << to_string(t);
  std::cout << ss.str() << std::endl;
}

std::vector<int64_t> non_negative(const at::Tensor& tensor) {
  assert(tensor.dtype() == at::kLong);
  assert(tensor.ndimension() == 1);
  std::vector<int64_t> vals;
  vals.reserve(tensor.size(0));
  for (std::size_t i = 0, n = tensor.size(0); i < n; ++i) {
    int64_t element = tensor[i].item<int64_t>();
    if (element < 0) {
      continue;
    }
    vals.emplace_back(element);
  }
  return vals;
}

std::vector<float> to_float_vect(const at::Tensor& tensor) {
  assert(tensor.dtype() == at::kFloat);
  assert(tensor.ndimension() == 1);
  std::vector<float> vals;
  vals.reserve(tensor.size(0));
  for (std::size_t i = 0, n = tensor.size(0); i < n; ++i) {
    float element = tensor[i].item<float>();
    vals.emplace_back(element);
  }
  return vals;
}

std::vector<float> to_float_vect(const std::list<at::Tensor>& tensor_list) {
  std::vector<float> vals;
  for (const at::Tensor& tensor : tensor_list) {
    assert(tensor.dtype() == at::kFloat);
    if (tensor.ndimension() == 0) {
      // scalar
      vals.emplace_back(tensor.item<float>());
      continue;
    }
    assert(tensor.ndimension() == 1);
    assert(tensor.size(0) == 1); // length of 1 only
    vals.reserve(tensor.size(0));
    for (std::size_t i = 0, n = tensor.size(0); i < n; ++i) {
      float element = tensor[i].item<float>();
      vals.emplace_back(element);
    }
  }
  return vals;
}
std::vector<float> tensor_to_vector_1d(const at::Tensor& tensor) {
  // Ensure the tensor is 1-dimensional and of type float
  assert(tensor.dim() == 1 && tensor.scalar_type() == at::kFloat);

  // Get the size of the tensor
  auto size = tensor.size(0);

  // Copy data from the tensor to a vector
  std::vector<float> vec(
      tensor.data_ptr<float>(), tensor.data_ptr<float>() + size);

  return vec;
}

std::vector<std::vector<float>> tensor_to_vector_2d(const at::Tensor& tensor) {
  // Ensure the tensor is a 2D tensor and is of type float
  assert(tensor.dim() == 2 && tensor.dtype() == at::kFloat);

  // Get the number of rows and columns
  int64_t rows = tensor.size(0);
  int64_t cols = tensor.size(1);

  // Convert tensor to a vector of vectors
  std::vector<std::vector<float>> result(rows, std::vector<float>(cols));
  auto tensor_data = tensor.accessor<float, 2>();

  for (int64_t i = 0; i < rows; ++i) {
    for (int64_t j = 0; j < cols; ++j) {
      result[i][j] = tensor_data[i][j];
    }
  }
  return result;
}

at::Tensor bbox_cxcyah_to_xyxy(const at::Tensor& bboxes) {
  // Split the input tensor into components cx, cy, ratio, h
  auto cx = bboxes.select(-1, 0).unsqueeze(-1);
  auto cy = bboxes.select(-1, 1).unsqueeze(-1);
  auto ratio = bboxes.select(-1, 2).unsqueeze(-1);
  auto h = bboxes.select(-1, 3).unsqueeze(-1);

  // Calculate width w from ratio and h
  auto w = ratio * h;

  // Calculate the bounding box corners (x1, y1, x2, y2)
  auto x1 = cx - w / 2.0;
  auto y1 = cy - h / 2.0;
  auto x2 = cx + w / 2.0;
  auto y2 = cy + h / 2.0;

  // Concatenate the results along the last dimension
  return at::cat({x1, y1, x2, y2}, -1);
}

at::Tensor bbox_xyxy_to_cxcyah(const at::Tensor& bboxes) {
  // Ensure the input tensor is on the CPU and is 2D with the second dimension
  // being 4
  assert(bboxes.dim() == 2 && bboxes.size(1) == 4);

  // Calculate center x, center y, width, and height
  auto x1 = bboxes.select(1, 0);
  auto y1 = bboxes.select(1, 1);
  auto x2 = bboxes.select(1, 2);
  auto y2 = bboxes.select(1, 3);

  auto w = x2 - x1;
  auto h = y2 - y1;
  auto cx = x1 + 0.5 * w;
  auto cy = y1 + 0.5 * h;
  auto aspect_ratio = w / h;

  // Concatenate these into a new tensor
  auto result = torch::stack({cx, cy, aspect_ratio, h}, 1);

  return result;
}

/*
    """Calculate overlap between two set of bboxes.

    FP16 Contributed by https://github.com/open-mmlab/mmdetection/pull/4889
    Note:
        Assume bboxes1 is M x 4, bboxes2 is N x 4, when mode is 'iou',
        there are some new generated variable when calculating IOU
        using bbox_overlaps function:

        1) is_aligned is False
            area1: M x 1
            area2: N x 1
            lt: M x N x 2
            rb: M x N x 2
            wh: M x N x 2
            overlap: M x N x 1
            union: M x N x 1
            ious: M x N x 1

            Total memory:
                S = (9 x N x M + N + M) * 4 Byte,

            When using FP16, we can reduce:
                R = (9 x N x M + N + M) * 4 / 2 Byte
                R large than (N + M) * 4 * 2 is always true when N and M >= 1.
                Obviously, N + M <= N * M < 3 * N * M, when N >=2 and M >=2,
                           N + 1 < 3 * N, when N or M is 1.

            Given M = 40 (ground truth), N = 400000 (three anchor boxes
            in per grid, FPN, R-CNNs),
                R = 275 MB (one times)

            A special case (dense detection), M = 512 (ground truth),
                R = 3516 MB = 3.43 GB

            When the batch size is B, reduce:
                B x R

            Therefore, CUDA memory runs out frequently.

            Experiments on GeForce RTX 2080Ti (11019 MiB):

            |   dtype   |   M   |   N   |   Use    |   Real   |   Ideal   |
            |:----:|:----:|:----:|:----:|:----:|:----:|
            |   FP32   |   512 | 400000 | 8020 MiB |   --   |   --   |
            |   FP16   |   512 | 400000 |   4504 MiB | 3516 MiB | 3516 MiB |
            |   FP32   |   40 | 400000 |   1540 MiB |   --   |   --   |
            |   FP16   |   40 | 400000 |   1264 MiB |   276MiB   | 275 MiB |

        2) is_aligned is True
            area1: N x 1
            area2: N x 1
            lt: N x 2
            rb: N x 2
            wh: N x 2
            overlap: N x 1
            union: N x 1
            ious: N x 1

            Total memory:
                S = 11 x N * 4 Byte

            When using FP16, we can reduce:
                R = 11 x N * 4 / 2 Byte

        So do the 'giou' (large than 'iou').

        Time-wise, FP16 is generally faster than FP32.

        When gpu_assign_thr is not -1, it takes more time on cpu
        but not reduce memory.
        There, we can reduce half the memory and keep the speed.

    If ``is_aligned`` is ``False``, then calculate the overlaps between each
    bbox of bboxes1 and bboxes2, otherwise the overlaps between each aligned
    pair of bboxes1 and bboxes2.

    Args:
        bboxes1 (Tensor): shape (B, m, 4) in <x1, y1, x2, y2> format or empty.
        bboxes2 (Tensor): shape (B, n, 4) in <x1, y1, x2, y2> format or empty.
            B indicates the batch dim, in shape (B1, B2, ..., Bn).
            If ``is_aligned`` is ``True``, then m and n must be equal.
        mode (str): "iou" (intersection over union), "iof" (intersection over
            foreground) or "giou" (generalized intersection over union).
            Default "iou".
        is_aligned (bool, optional): If True, then m and n must be equal.
            Default False.
        eps (float, optional): A value added to the denominator for numerical
            stability. Default 1e-6.

    Returns:
        Tensor: shape (m, n) if ``is_aligned`` is False else shape (m,)

    Example:
        >>> bboxes1 = torch.FloatTensor([
        >>>     [0, 0, 10, 10],
        >>>     [10, 10, 20, 20],
        >>>     [32, 32, 38, 42],
        >>> ])
        >>> bboxes2 = torch.FloatTensor([
        >>>     [0, 0, 10, 20],
        >>>     [0, 10, 10, 19],
        >>>     [10, 10, 20, 20],
        >>> ])
        >>> overlaps = bbox_overlaps(bboxes1, bboxes2)
        >>> assert overlaps.shape == (3, 3)
        >>> overlaps = bbox_overlaps(bboxes1, bboxes2, is_aligned=True)
        >>> assert overlaps.shape == (3, )

    Example:
        >>> empty = torch.empty(0, 4)
        >>> nonempty = torch.FloatTensor([[0, 0, 10, 9]])
        >>> assert tuple(bbox_overlaps(empty, nonempty).shape) == (0, 1)
        >>> assert tuple(bbox_overlaps(nonempty, empty).shape) == (1, 0)
        >>> assert tuple(bbox_overlaps(empty, empty).shape) == (0, 0)
    """
*/
at::Tensor bbox_overlaps(
    at::Tensor bboxes1,
    at::Tensor bboxes2,
    const std::string& mode,
    bool is_aligned,
    float eps) {
  assert(
      (mode == "iou" || mode == "iof" || mode == "giou") && "Unsupported mode");
  assert(
      (bboxes1.size(-1) == 4 || bboxes1.size(0) == 0) &&
      (bboxes2.size(-1) == 4 || bboxes2.size(0) == 0));
  assert(
      bboxes1.sizes().slice(0, bboxes1.dim() - 2) ==
          bboxes2.sizes().slice(0, bboxes2.dim() - 2) &&
      "Batch dimensions must match");

  auto batch_shape = bboxes1.sizes().slice(0, bboxes1.dim() - 2).vec();
  int64_t rows = bboxes1.size(-2);
  int64_t cols = bboxes2.size(-2);

  if (is_aligned) {
    assert(rows == cols && "If aligned, rows and cols must match");
  }

  if (rows * cols == 0) {
    if (is_aligned) {
      return bboxes1.new_empty(batch_shape).expand({rows});
    } else {
      return bboxes1.new_empty(batch_shape).expand({rows, cols});
    }
  }

  auto area1 = (bboxes1.select(-1, 2) - bboxes1.select(-1, 0)) *
      (bboxes1.select(-1, 3) - bboxes1.select(-1, 1));
  auto area2 = (bboxes2.select(-1, 2) - bboxes2.select(-1, 0)) *
      (bboxes2.select(-1, 3) - bboxes2.select(-1, 1));

  at::Tensor lt, rb, wh, overlap, union_area, ious, enclosed_lt, enclosed_rb,
      enclose_wh, enclose_area;
  if (is_aligned) {
    lt = at::max(bboxes1.slice(-1, 0, 2), bboxes2.slice(-1, 0, 2));
    rb = at::min(bboxes1.slice(-1, 2, 4), bboxes2.slice(-1, 2, 4));
    wh = at::clamp(rb - lt, /*min=*/0);
    overlap = wh.select(-1, 0) * wh.select(-1, 1);

    if (mode == "iou" || mode == "giou") {
      union_area = area1 + area2 - overlap;
    } else {
      union_area = area1;
    }

    if (mode == "giou") {
      enclosed_lt = at::min(bboxes1.slice(-1, 0, 2), bboxes2.slice(-1, 0, 2));
      enclosed_rb = at::max(bboxes1.slice(-1, 2, 4), bboxes2.slice(-1, 2, 4));
    }
  } else {
    // PT("bboxes1", bboxes1);
    // PT("bboxes2", bboxes2);
    auto i1 = bboxes1.index(
        {at::indexing::Slice(), at::indexing::None, at::indexing::Slice(0, 2)});
    auto i2 = bboxes2.index(
        {at::indexing::None, at::indexing::Slice(), at::indexing::Slice(0, 2)});
    // PT("i1", i1);
    // PT("i2", i2);
    lt = at::max(
        bboxes1.index(
            {at::indexing::Slice(),
             at::indexing::None,
             at::indexing::Slice(at::indexing::None, 2)}),
        bboxes2.index(
            {at::indexing::None,
             at::indexing::Slice(),
             at::indexing::Slice(at::indexing::None, 2)}));
    rb = at::min(
        bboxes1.index(
            {at::indexing::Slice(),
             at::indexing::None,
             at::indexing::Slice(2, at::indexing::None)}),
        bboxes2.index(
            {at::indexing::None,
             at::indexing::Slice(),
             at::indexing::Slice(2, at::indexing::None)}));
    wh = at::clamp(rb - lt, /*min=*/0);
    overlap = wh.select(-1, 0) * wh.select(-1, 1);

    if (mode == "iou" || mode == "giou") {
      union_area = area1.unsqueeze(-1) + area2.unsqueeze(-2) - overlap;
    } else {
      union_area = area1.unsqueeze(-1);
    }

    if (mode == "giou") {
      enclosed_lt = at::min(
          bboxes1.index(
              {at::indexing::Slice(),
               at::indexing::None,
               at::indexing::Slice(at::indexing::None, 2)}),
          bboxes2.index(
              {at::indexing::None,
               at::indexing::Slice(),
               at::indexing::Slice(at::indexing::None, 2)}));
      enclosed_rb = at::max(
          bboxes1.index(
              {at::indexing::Slice(),
               at::indexing::None,
               at::indexing::Slice(2, at::indexing::None)}),
          bboxes2.index(
              {at::indexing::None,
               at::indexing::Slice(),
               at::indexing::Slice(2, at::indexing::None)}));
    }
  }

  // auto eps_tensor = union_area.new_tensor({eps});
  auto eps_tensor = at::empty_like(union_area).fill_(eps);
  union_area = at::max(union_area, eps_tensor);
  ious = overlap / union_area;

  if (mode == "iou" || mode == "iof") {
    return ious;
  }

  // Calculate GIoU if required
  enclose_wh = at::clamp(enclosed_rb - enclosed_lt, /*min=*/0);
  enclose_area = enclose_wh.select(-1, 0) * enclose_wh.select(-1, 1);
  enclose_area = at::max(enclose_area, eps_tensor);
  auto gious = ious - (enclose_area - union_area) / enclose_area;

  return gious;
}

at::Tensor bool_mask_select(const at::Tensor& tensor, const at::Tensor& mask) {
  assert(mask.dtype() == at::kBool);
  at::Tensor indices =
      at::nonzero(mask).squeeze(); // Squeezing is necessary to convert it into
                                   // a 1D tensor of indices
  return at::index_select(tensor, 0, indices);
}

} // namespace tracker
} // namespace hm
