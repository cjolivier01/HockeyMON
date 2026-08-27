#pragma once

#include <cassert>
#include <chrono>
#include <iostream>

namespace hm {

class Timer {
 public:
  Timer(std::string name = "", std::size_t batch_size = 1)
      : name_(std::move(name)), batch_size_(batch_size) {}

  constexpr std::size_t count() const {
    return count_;
  }
  constexpr const std::string& name() const {
    return name_;
  }
  void tic() {
    assert(!in_tic_);
    start_time_ = std::chrono::high_resolution_clock::now();
    in_tic_ = true;
  }
  void toc() {
    auto end_time = std::chrono::high_resolution_clock::now();
    total_microseconds_ +=
        std::chrono::duration_cast<std::chrono::microseconds>(
            end_time - start_time_)
            .count();
    ++count_;
    in_tic_ = false;
  }
  double average_time() const {
    if (!count_) {
      return 0.0;
    }
    return double(total_microseconds_) / double(count_) / 1000000.0;
  }
  double fps() const {
    if (!count_) {
      return 0.0;
    }
    auto average_t = average_time();
    return 1.0 * double(batch_size_) / average_t;
  }
  void reset() {
    count_ = 0;
    total_microseconds_ = 0;
    in_tic_ = false;
  }
  friend std::ostream& operator<<(std::ostream& os, const Timer& timer);

 private:
  std::string name_;
  std::size_t count_{0};
  std::size_t total_microseconds_{0};
  std::chrono::high_resolution_clock::time_point start_time_;
  bool in_tic_{false};
  std::size_t batch_size_;
};

inline std::ostream& operator<<(std::ostream& os, const Timer& timer) {
  return os << timer.name() << ": " << (timer.count() * timer.batch_size_)
     << " items at " << timer.fps() << " fps";
}

} // namespace hm