#pragma once

#include <ATen/ATen.h>

#include <array>
#include <optional>
#include <string>

namespace hm {

namespace ops {

std::int64_t image_width(const at::Tensor& t);
std::int64_t image_height(const at::Tensor& t);
std::array<std::int64_t, 2> image_size(const at::Tensor& t);

void show_image(
    const std::string& label,
    at::Tensor tensor,
    bool wait = false,
    std::optional<float> scale = std::nullopt);

}
} // namespace hm
