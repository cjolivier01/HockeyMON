#include "BaseTracker.h"

namespace hm {
namespace tracker {

BaseTracker::BaseTracker(
    std::optional<std::unordered_map<std::string, float>> momentums,
    int num_frames_retain,
    bool validate)
    : validate_(validate),
      momentums_(std::move(momentums)),
      num_frames_retain_(num_frames_retain),
      num_tracks_(0) {
  reset();
}

// Reset the buffer of the tracker
void BaseTracker::reset() {
  num_tracks_ = 0;
  tracks_.clear();
}

// Check if the buffer is empty
bool BaseTracker::empty() const {
  return tracks_.empty();
}

// Return all ids in the tracker
std::vector<int> BaseTracker::ids() const {
  assert(false); // slow
  std::vector<int> id_list;
  id_list.reserve(tracks_.size());
  for (const auto& item : tracks_) {
    id_list.emplace_back(item.first);
  }
  return id_list;
}

// Check if re-identification model is present
bool BaseTracker::with_reid() const {
  return reid_ != nullptr;
}

// bool is_single_value(const at::Tensor& t) {

// }

// Update the tracker with new information
void BaseTracker::update( // TODO: Make the key an enum and just let that enum
                          // grow as needed
    const std::unordered_map<std::string, const at::Tensor*>& kwargs) {
  // Implement the logic to update the tracker as in the Python `update`
  // function
  bool initial_memo = memo_items_.empty();
  const at::Tensor& ids = *kwargs.at(kIds);
  const std::size_t num_objs = ids.size(0);
  const at::Tensor& frame_ids = *kwargs.at(kFrameIds);
  std::unordered_map<std::string, const at::Tensor*> these_memo_items;
  these_memo_items.reserve(kwargs.size());
  for (const auto& item : kwargs) {
    const std::string& key = item.first;
    const at::Tensor* val = item.second;
    if (val) {
      if (initial_memo) {
        memo_items_.emplace(key);
      } else if (validate_) {
        assert(memo_items_.count(key));
      }
      these_memo_items.emplace(key, val);
    } else if (validate_) {
      assert(!memo_items_.count(key));
    }
  }
  std::unordered_map<std::string, at::Tensor> obj_memo_items;
  std::unordered_set<int64_t> seen_frame_ids;
  obj_memo_items.reserve(kwargs.size());
  seen_frame_ids.reserve(num_objs);
  for (std::size_t i = 0; i < num_objs; ++i) {
    for (const auto& obj_item : these_memo_items) {
      // Track history outlives the per-frame tensors passed into update().
      obj_memo_items[obj_item.first] = (*obj_item.second)[i].clone();
    }
    const int64_t id = ids[i].item().to<int64_t>();
    if (tracks_.count(id)) {
      update_track(id, obj_memo_items);
    } else {
      init_track(id, obj_memo_items);
    }
    seen_frame_ids.emplace(frame_ids[i].item().to<int64_t>());
  }
  for (int64_t f_id : seen_frame_ids) {
    pop_invalid_tracks(f_id);
  }
}

void BaseTracker::init_track(
    int64_t id,
    const std::unordered_map<std::string, at::Tensor>& memos) {
  STrack track;
  for (const auto& item : memos) {
    const auto& key = item.first;
    if (momentums_.has_value()) {
      auto& mm = momentums_.value();
      auto iter = mm.find(key);
      if (iter != mm.end()) {
        float v = item.second.item<float>();
        track.ref_momentum(key) = v;
        continue;
      }
    }
    // regular
    track.get_ref(key).emplace_back(item.second);
  }
  const bool inserted = tracks_.emplace(id, std::move(track)).second;
  assert(inserted);
}

void BaseTracker::update_track(
    int64_t id,
    const std::unordered_map<std::string, at::Tensor>& memos) {
  STrack& track = tracks_.at(id);
  for (const auto& item : memos) {
    const auto& key = item.first;
    if (momentums_.has_value()) {
      auto& mm = momentums_.value();
      auto iter = mm.find(key);
      if (iter != mm.end()) {
        auto m = iter->second;
        float v = item.second.item<float>();
        float& track_m = track.ref_momentum(key);
        track_m = (1.0 - m) * track_m + m * v;
        continue;
      }
    }
    // regular
    auto& val_list = track.get_ref(key);
    val_list.emplace_back(item.second);
    while (val_list.size() > static_cast<size_t>(num_frames_retain_)) {
      val_list.pop_front();
    }
  }
}

void BaseTracker::pop_invalid_tracks(int64_t frame_id) {
  std::vector<std::size_t> invalid_ids;
  invalid_ids.reserve(tracks_.size());
  for (const auto& item : tracks_) {
    const STrack& track = item.second;
    int64_t last_frame_id = track.get_ref(kFrameIds).back().item<int64_t>();
    if (frame_id - last_frame_id >= num_frames_retain_) {
      invalid_ids.emplace_back(item.first);
    }
  }
  for (std::size_t id : invalid_ids) {
    pop_track(id);
  }
}

void BaseTracker::pop_track(int64_t track_id) {
  tracks_.erase(track_id);
}

#if 0
// Method to get memo (buffer) of specific item
std::unordered_map<std::string, at::Tensor> BaseTracker::memo() {
  std::unordered_map<std::string, std::vector<at::Tensor>> outs;
  for (const auto& k : memo_items_) {
    outs[k] = std::vector<at::Tensor>();
  }

  for (const auto& [id, objs] : tracks_) {
    for (const auto& [k, v] : objs) {
      if (outs.find(k) != outs.end()) {
        outs[k].push_back(v.back());
      }
    }
  }

  std::unordered_map<std::string, at::Tensor> result;
  for (const auto& [k, v] : outs) {
    result[k] = at::cat(v, 0);
  }
  return result;
}
#endif

} // namespace tracker
} // namespace hm
