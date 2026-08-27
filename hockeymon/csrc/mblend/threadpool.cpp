#pragma once

#include <cassert>
#include "threadpool.h"

namespace hm {

std::atomic<std::size_t> HmThreadPool::next_job_id_{1};

HmThreadPool::HmThreadPool(Eigen::ThreadPool* thread_pool)
    : thread_pool_(thread_pool) {}

HmThreadPool::HmThreadPool(const HmThreadPool& hm_thread_pool)
    : thread_pool_(hm_thread_pool.get_internal_thread_pool()) {}

HmThreadPool& HmThreadPool::operator=(Eigen::ThreadPool* thread_pool) {
  thread_pool_ = thread_pool;
  return *this;
}

HmThreadPool::~HmThreadPool() {
  join_all();
}

void HmThreadPool::join_all() {
  // Block anything getting scheduled while we perform a join-all
  absl::MutexLock lk(&schedule_mutex_);
  absl::MutexLock lk_inner(&mu_);
  mu_.Await(absl::Condition(
      +[](HmThreadPool* this_ptr) ABSL_EXCLUSIVE_LOCKS_REQUIRED(
           this_ptr->mu_) { return this_ptr->job_ids_.empty(); },
      this));
}

std::size_t HmThreadPool::Schedule(std::function<void()>&& thread_fn) {
  std::size_t new_id = next_job_id_++;

  absl::MutexLock lk_sched(&schedule_mutex_);
  {
    absl::MutexLock lk(&mu_);
    job_ids_.emplace(new_id);
  }
  thread_pool_->Schedule([this, new_id, fn = std::move(thread_fn)]() {
    fn();
    absl::MutexLock lk(&mu_);
    job_ids_.erase(new_id);
  });
  return new_id;
}


} // namespace hm
