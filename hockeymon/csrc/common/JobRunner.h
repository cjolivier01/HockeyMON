#pragma once

#include <atomic>
#include <cassert>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

class ManualResetGate {
 public:
  ManualResetGate(bool initiallySet = false) : is_open_(initiallySet) {}

  void wait() {
    std::unique_lock<std::mutex> lock(mutex_);
    conditionVariable_.wait(lock, [this] { return is_open_.load(); });
  }

  void signal() {
    std::lock_guard<std::mutex> lock(mutex_);
    is_open_ = true;
    conditionVariable_.notify_all();
  }

  void reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    is_open_ = false;
  }

 private:
  std::mutex mutex_;
  std::condition_variable conditionVariable_;
  std::atomic<bool> is_open_;
};

namespace hm {
/**
 *   _____            _             _  ____
 *  / ____|          | |           | |/ __ \
 * | (___   ___  _ __| |_  ___   __| | |  | |_   _  ___  _   _  ___
 *  \___ \ / _ \| '__| __|/ _ \ / _` | |  | | | | |/ _ \| | | |/ _ \
 *  ____) | (_) | |  | |_|  __/| (_| | |__| | |_| |  __/| |_| |  __/
 * |_____/ \___/|_|   \__|\___| \__,_|\___\_\\__,_|\___| \__,_|\___|
 *
 */
template <typename KEY_TYPE, typename ITEM_TYPE>
class SortedQueue {
 public:
  using key_type = KEY_TYPE;
  using value_type = ITEM_TYPE;

  SortedQueue(/*std::size_t max_queue_size = 0*/)
  /*: max_queue_size_(max_queue_size)*/ {}

  void enqueue(const KEY_TYPE& key, ITEM_TYPE&& item) {
    std::unique_lock<std::mutex> lk(items_mu_);
    // if (max_queue_size_) {
    //   cu_max_.wait(lk, [this] { return items_.size() < max_queue_size_; });
    // }
    if (!items_.emplace(key, std::move(item)).second) {
      throw std::runtime_error("Duplciate key in sorted queue");
    }
    cu_.notify_one();
  }

  ITEM_TYPE dequeue_smallest_key(KEY_TYPE* key_type_ptr = nullptr) {
    std::unique_lock<std::mutex> lk(items_mu_);
    cu_.wait(lk, [this] { return !items_.empty(); });
    auto iter = items_.begin();
    auto value = std::move(iter->second);
    if (key_type_ptr) {
      *key_type_ptr = iter->first;
    }
    items_.erase(iter);
    return value;
  }

  ITEM_TYPE dequeue_key(const KEY_TYPE& key) {
    std::unique_lock<std::mutex> lk(items_mu_);
    cu_.wait(lk, [this, &key] { return items_.find(key) != items_.end(); });
    auto iter = items_.find(key);
    assert(iter != items_.end());
    auto value = std::move(iter->second);
    items_.erase(iter);
    return value;
  }

  // constexpr std::size_t max_queue_size() const {
  //   return max_queue_size_;
  // }

 private:
  std::size_t max_queue_size_{0};
  std::mutex items_mu_;
  std::condition_variable cu_;
  std::condition_variable cu_max_;
  std::map<KEY_TYPE, ITEM_TYPE> items_;
};

/**
 *       _       _     _____
 *      | |     | |   |  __ \
 *      | | ___ | |__ | |__) |_   _ _ __  _ __   ___  _ __
 *  _   | |/ _ \| '_ \|  _  /| | | | '_ \| '_ \ / _ \| '__|
 * | |__| | (_) | |_) | | \ \| |_| | | | | | | |  __/| |
 *  \____/ \___/|_.__/|_|  \_\\__,_|_| |_|_| |_|\___||_|
 *
 */
template <typename INPUT_TYPE, typename OUTPUT_TYPE>
class JobRunner {
 public:
  using KeyType = std::size_t;
  using InputQueue = SortedQueue<KeyType, INPUT_TYPE>;
  using OutputQueue = SortedQueue<KeyType, INPUT_TYPE>;

  JobRunner(
      std::shared_ptr<InputQueue> input_queue,
      std::function<OUTPUT_TYPE(std::size_t worker_index, INPUT_TYPE&&)>
          worker_fn)
      : input_queue_(std::move(input_queue)),
        output_queue_(std::make_shared<OutputQueue>()),
        worker_fn_(std::move(worker_fn)) {
    assert(input_queue_);
  }
  ~JobRunner() {
    stop();
  }

  void start(std::size_t thread_count) {
    stop();
    threads_.reserve(thread_count);
    for (std::size_t i = 0; i < thread_count; ++i) {
      threads_.emplace_back(std::make_unique<std::thread>(
          [this, worker_index = i] { this->run(worker_index); }));
    }
  }

  void stop() {
    for (std::size_t i = 0, n = threads_.size(); i < n; ++i) {
      INPUT_TYPE null_input;
      assert(!null_input);
      input_queue_->enqueue(
          std::numeric_limits<KeyType>::max() - i - 1, std::move(null_input));
    }
    for (auto& t : threads_) {
      t->join();
    }
    threads_.clear();
  }

  const std::shared_ptr<InputQueue>& inputs() {
    assert(input_queue_);
    return input_queue_;
  }
  const std::shared_ptr<OutputQueue>& outputs() {
    assert(output_queue_);
    return output_queue_;
  }

 private:
  void run(std::size_t worker_index) {
    do {
      KeyType key{std::numeric_limits<KeyType>::max()};
      INPUT_TYPE input = input_queue_->dequeue_smallest_key(&key);
      if (!input) {
        break;
      }
      assert(output_queue_ != input_queue_);
      output_queue_->enqueue(key, worker_fn_(worker_index, std::move(input)));
    } while (true);
  }
  std::shared_ptr<InputQueue> input_queue_;
  std::shared_ptr<OutputQueue> output_queue_;
  std::vector<std::unique_ptr<std::thread>> threads_;
  std::function<OUTPUT_TYPE(std::size_t worker_index, INPUT_TYPE&&)> worker_fn_;
};

} // namespace hm
