#include "hockeymon/csrc/ui/HmRenderSet.h"

#include <condition_variable>
#include <functional>
#include <future>
#include <queue>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace hm {
namespace display {

namespace {
imageFormat get_image_format(int channels) {
  switch (channels) {
    case 4:
      return imageFormat::IMAGE_RGBA8;
    case 3:
      return imageFormat::IMAGE_RGB8;
    default:
      assert(false);
      return imageFormat::IMAGE_UNKNOWN;
  }
}

std::mutex grs_mu;
std::shared_ptr<HmRenderSet> global_render_set;

} // namespace

struct HmRenderSet::DisplayWorker {
  DisplayWorker(std::string name, const DisplaySurface& surface)
      : name_(std::move(name)),
        surface_width_(surface.width),
        surface_height_(surface.height),
        surface_pitch_(surface.pitch),
        surface_channels_(surface.channels),
        shutting_down_(false),
        worker_thread_(&DisplayWorker::thread_main, this) {}

  ~DisplayWorker() {
    shutdown();
  }

  template <typename Func>
  auto invoke(Func&& func) -> decltype(func(std::declval<glDisplay&>())) {
    using ReturnT = decltype(func(std::declval<glDisplay&>()));
    auto task_ptr = std::make_shared<std::packaged_task<ReturnT()>>(
        [this, func = std::forward<Func>(func)]() mutable {
          if (!display_) {
            throw std::runtime_error("display not initialized");
          }
          return func(*display_);
        });

    auto future = task_ptr->get_future();

    {
      std::unique_lock<std::mutex> lock(queue_mu_);
      if (shutting_down_) {
        throw std::runtime_error("Display worker is shutting down");
      }

      tasks_.emplace([task_ptr]() { (*task_ptr)(); });
    }

    queue_cv_.notify_one();
    return future.get();
  }

  void shutdown() {
    {
      std::unique_lock<std::mutex> lock(queue_mu_);
      if (shutting_down_)
        return;

      shutting_down_ = true;
    }

    queue_cv_.notify_all();

    if (worker_thread_.joinable()) {
      worker_thread_.join();
    }
  }

 private:
  void thread_main() {
    DisplaySurface surface(
        nullptr,
        surface_width_,
        surface_height_,
        surface_pitch_,
        surface_channels_);
    display_ = HmRenderSet::create_video_output(name_, surface);

    while (true) {
      std::function<void()> task;

      {
        std::unique_lock<std::mutex> lock(queue_mu_);
        queue_cv_.wait(
            lock,
            [this]() { return shutting_down_ || !tasks_.empty(); });

        if (shutting_down_ && tasks_.empty())
          break;

        task = std::move(tasks_.front());
        tasks_.pop();
      }

      task();
    }
    // Destroy the display in this thread
    display_.reset();
  }

  std::string name_;
  int surface_width_;
  int surface_height_;
  int surface_pitch_;
  int surface_channels_;
  std::mutex queue_mu_;
  std::condition_variable queue_cv_;
  std::queue<std::function<void()>> tasks_;
  std::atomic<bool> shutting_down_;
  std::thread worker_thread_;
  std::unique_ptr<glDisplay> display_;
};

HmRenderSet::HmRenderSet() = default;

HmRenderSet::~HmRenderSet() {
  std::vector<std::shared_ptr<DisplayWorker>> workers;

  {
    std::unique_lock<std::mutex> lock(workers_mu_);
    for (auto& entry : workers_)
      workers.push_back(entry.second);

    workers_.clear();
  }

  for (auto& worker : workers)
    worker->shutdown();
}

std::weak_ptr<HmRenderSet> get_or_create_global_render_set() {
  std::unique_lock lk(grs_mu);
  if (!global_render_set) {
    global_render_set = std::make_shared<HmRenderSet>();
  }
  return global_render_set;
}

void destroy_global_render_set() {
  std::unique_lock lk(grs_mu);
  global_render_set.reset();
}

std::unique_ptr<glDisplay> HmRenderSet::create_video_output(
    const std::string& name,
    const DisplaySurface& surface) {
  videoOptions vo;
  vo.width = (int)surface.pitch_width();
  vo.height = (int)surface.height;
  auto video_output = std::unique_ptr<glDisplay>(glDisplay::Create(vo));
  video_output->SetTitle(name.c_str());
  return video_output;
}

std::shared_ptr<HmRenderSet::DisplayWorker> HmRenderSet::get_or_create_worker(
    const std::string& name,
    const DisplaySurface& surface) {
  std::unique_lock<std::mutex> lock(workers_mu_);
  auto found = workers_.find(name);

  if (found != workers_.end())
    return found->second;

  auto worker = std::make_shared<DisplayWorker>(name, surface);
  workers_.emplace(name, worker);
  return worker;
}

bool HmRenderSet::render(
    const std::string& name,
    const DisplaySurface& surface,
    cudaStream_t stream) {
  DisplaySurface surface_copy = surface;
  auto worker = get_or_create_worker(name, surface_copy);

  return worker->invoke([surface_copy, stream](glDisplay& display) mutable {
    return display.Render(
        surface_copy.d_ptr,
        surface_copy.pitch_width(),
        surface_copy.height,
        get_image_format(surface_copy.channels),
        stream);
  });
}

} // namespace display
} // namespace hm
