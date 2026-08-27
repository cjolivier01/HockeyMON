#pragma once

#include <ATen/ATen.h>
#include "KalmanFilter.h"

#include <list>
#include <unordered_map>

using namespace std;

namespace hm {
namespace tracker {

struct STrack {
  enum class State { Unknown, Tentative, Tracking, Lost };

  STrack() {
    if (lookup_.empty()) {
      lookup_ = {
          {"ids",
           reinterpret_cast<ptrdiff_t>(&this->ids) -
               reinterpret_cast<ptrdiff_t>(this)},
          {"bboxes",
           reinterpret_cast<ptrdiff_t>(&this->bboxes) -
               reinterpret_cast<ptrdiff_t>(this)},
          {"scores",
           reinterpret_cast<ptrdiff_t>(&this->scores) -
               reinterpret_cast<ptrdiff_t>(this)},
          {"labels",
           reinterpret_cast<ptrdiff_t>(&this->labels) -
               reinterpret_cast<ptrdiff_t>(this)},
          {"frame_ids",
           reinterpret_cast<ptrdiff_t>(&this->frame_ids) -
               reinterpret_cast<ptrdiff_t>(this)},
      };
    }
  }
  std::list<at::Tensor>& get_ref(const std::string& key) {
    ptrdiff_t offset = lookup_.at(key);
    return *reinterpret_cast<std::list<at::Tensor>*>(
        reinterpret_cast<ptrdiff_t>(this) + offset);
  }
  const std::list<at::Tensor>& get_ref(const std::string& key) const {
    ptrdiff_t offset = lookup_.at(key);
    return *reinterpret_cast<const std::list<at::Tensor>*>(
        reinterpret_cast<ptrdiff_t>(this) + offset);
  }
  float& ref_momentum(const std::string& key) {
    return momentums.at(key);
  }
  const float& ref_momentum(const std::string& key) const {
    return momentums.at(key);
  }

  at::Tensor age() const {
    if (frame_ids.empty()) {
      return at::Tensor();
    }
    return *frame_ids.rbegin() - *frame_ids.begin() + 1;
  }

  State state{State::Unknown};
  std::list<at::Tensor> ids;
  std::list<at::Tensor> bboxes;
  std::list<at::Tensor> scores;
  std::list<at::Tensor> labels;
  std::list<at::Tensor> frame_ids;
  std::unordered_map<std::string, float> momentums;
  byte_kalman::KAL_MEAN mean;
  byte_kalman::KAL_COVA covariance;

  // HmTracker stuff
  // float max_tentative_score{0.0};

 private:
  static std::unordered_map<std::string, ptrdiff_t> lookup_;
};

} // namespace tracker
} // namespace hm
