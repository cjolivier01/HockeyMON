#pragma once

#include "STrack.h"
#include "Utils.h"

#include <ATen/ATen.h>

#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace hm {
namespace tracker {

struct ITracker {
  virtual ~ITracker() = default;

  virtual void reset() = 0;

  virtual std::unordered_map<std::string, at::Tensor> track(
      std::unordered_map<std::string, at::Tensor>&& data) = 0;

  virtual void init_track(
      int64_t id,
      const std::unordered_map<std::string, at::Tensor>& memos) = 0;

  virtual void update_track(
      int64_t id,
      const std::unordered_map<std::string, at::Tensor>& memos) = 0;

  virtual void pop_invalid_tracks(int64_t frame_id) = 0;
  virtual void pop_track(int64_t tracking_id) = 0;
};

class BaseTracker : public ITracker {
 public:
  // Constructor with default values
  BaseTracker(
      std::optional<std::unordered_map<std::string, float>> momentums =
          std::nullopt,
      int num_frames_retain = 10,
      bool validate = true);

  virtual ~BaseTracker() = default;

  // Reset the buffer of the tracker
  void reset() override;

  // Check if the buffer is empty
  bool empty() const;

  // Return all ids in the tracker
  std::vector<int> ids() const;

  // Check if re-identification model is present
  bool with_reid() const;

  // Update the tracker with new information
  void update( // TODO: Make the key an enum and just let that enum grow as
               // needed
      const std::unordered_map<std::string, const at::Tensor*>& kwargs);

  void init_track(
      int64_t id,
      const std::unordered_map<std::string, at::Tensor>& memos) override;

  void update_track(
      int64_t id,
      const std::unordered_map<std::string, at::Tensor>& memos) override;

  void pop_invalid_tracks(int64_t frame_id) override;

  void pop_track(int64_t tracking_id) override;

#if 0
  // Method to get memo (buffer) of specific item
  std::unordered_map<std::string, at::Tensor> memo();
#endif

 protected:
  static inline constexpr const char* kIds = "ids";
  static inline constexpr const char* kFrameIds = "frame_ids";
  std::map</*track_id=*/std::size_t, STrack> tracks_;

  const std::set<std::string>& memo_items() const {
    return memo_items_;
  }

  constexpr int num_frames_retain() const {
    return num_frames_retain_;
  }

 private:
  bool validate_;
  std::optional<std::unordered_map<std::string, float>> momentums_;
  int num_frames_retain_;
  int num_tracks_;
  std::set<std::string> memo_items_;
  std::unique_ptr<std::unordered_map<std::string, at::Tensor>> reid_;
};

} // namespace tracker
} // namespace hm
