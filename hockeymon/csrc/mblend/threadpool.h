#pragma once

#include "unsupported/Eigen/CXX11/ThreadPool"

#include <atomic>
#include <functional>
#include <unordered_set>

#include "absl/base/thread_annotations.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"

namespace hm {

class HmThreadPool {
 public:
  HmThreadPool(Eigen::ThreadPool* thread_pool);
  HmThreadPool(const HmThreadPool& hm_thread_pool);

  ~HmThreadPool();

  HmThreadPool& operator=(Eigen::ThreadPool* thread_pool);
  std::size_t Schedule(std::function<void()>&& thread_fn);
  void join_all();
  constexpr HmThreadPool* operator->() {
    return this;
  }
  constexpr const HmThreadPool* const operator->() const {
    return this;
  }

  std::size_t GetNThreads() const {
    return thread_pool_->NumThreads();
  }

  static constexpr absl::Duration ms(std::size_t milliseconds) {
    return absl::Milliseconds(milliseconds);
  }

  static Eigen::ThreadPool* get_base_thread_pool();

 private:
  Eigen::ThreadPool* get_internal_thread_pool() const {
    return thread_pool_;
  }

  Eigen::ThreadPool* thread_pool_;

  absl::Mutex schedule_mutex_ ABSL_ACQUIRED_BEFORE(mu_);

  /**
   * @brief Mutex prootecting the controller variables
   */
  absl::Mutex mu_ ABSL_ACQUIRED_AFTER(schedule_mutex_);

  std::unordered_set<std::size_t> job_ids_ ABSL_GUARDED_BY(mu_){};

  static std::atomic<std::size_t> next_job_id_;
};

using ThreadPool = HmThreadPool;

} // namespace hm
