#pragma once

#include "hugin_base/hugin_utils/utils.h"

#include <string>

namespace hm {

void init_stack_trace();

bool check_cuda_opengl();

std::size_t get_tick_count_ms();

void set_thread_name(const std::string& thread_name, int index = -1);


struct GpuContext {
  GpuContext();
  ~GpuContext();
  bool is_valid() const;
};

}  // namespace hm
