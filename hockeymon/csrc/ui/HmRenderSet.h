#pragma once

#include "cuda_runtime_compat.h"

#include "jetson-utils/display/glDisplay.h"

#include <cassert>
#include <map>
#include <memory>
#include <mutex>
#include <string>

namespace hm {
namespace display {

struct DisplaySurface {
  DisplaySurface(void* d, int w, int h, int p, int c)
      : d_ptr(d), width(w), height(h), pitch(p), channels(c) {}

  void* d_ptr;
  int width;
  int height;
  int pitch;
  int channels;

  int pitch_width() const {
    assert(pitch % (channels * width) == 0);
    return pitch / channels;
  }
};

class HmRenderSet {
 public:
  HmRenderSet();
  ~HmRenderSet();

  bool render(
      const std::string& name,
      const DisplaySurface& surface,
      cudaStream_t stream);

 private:
  struct DisplayWorker;

  static std::unique_ptr<glDisplay> create_video_output(
    const std::string& name,
    const DisplaySurface& surface);

  std::shared_ptr<DisplayWorker> get_or_create_worker(
    const std::string& name,
    const DisplaySurface& surface);

  std::mutex workers_mu_;
  std::map<std::string, std::shared_ptr<DisplayWorker>> workers_;
};

std::weak_ptr<HmRenderSet> get_or_create_global_render_set();
void destroy_global_render_set();

} // namespace display
} // namespace hm
