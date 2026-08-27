#pragma once

#include "STrack.h"

#include <ATen/ATen.h>

#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <Eigen/Dense>

namespace hm {
namespace tracker {
#if 0
class CenterPointTracker {
 public:
  struct Track {
    int id;
    at::Tensor bbox;
    int age;
    int hits;
    int time_since_update;
    Eigen::Vector4f state; // [x, y, vx, vy]
    Eigen::Matrix4f covariance;
  };

  CenterPointTracker(float iou_threshold, int max_age, int min_hits)
      : iou_threshold_(iou_threshold),
        max_age_(max_age),
        min_hits_(min_hits),
        next_id_(0) {}

  void update(const at::Tensor& detections) {
    // Predict the next position for each track using a Kalman filter
    for (auto& track : tracks_) {
      predict(track);
    }

    // Step 1: Match detections to existing tracks
    std::vector<int> unmatched_detections;
    at::Tensor iou_matrix = compute_iou_matrix(detections);
    auto [matches, unmatched_tracks, unmatched_detections_indices] =
        match_detections(iou_matrix);

    // Update existing tracks with matched detections
    for (const auto& match : matches) {
      auto& track = tracks_[match.first];
      update_kalman_filter(track, detections[match.second]);
      track.hits += 1;
      track.time_since_update = 0;
    }

    // Create new tracks for unmatched detections
    for (int idx : unmatched_detections_indices) {
      Eigen::Vector4f initial_state;
      initial_state << detections[idx][0].item<float>(),
          detections[idx][1].item<float>(), 0.0, 0.0;
      Eigen::Matrix4f initial_covariance =
          Eigen::Matrix4f::Identity() * 1000.0f;

      tracks_.emplace_back(Track{
          next_id_++,
          detections[idx],
          0,
          1,
          0,
          initial_state,
          initial_covariance});
    }

    // Mark unmatched tracks and remove old ones
    for (int track_idx : unmatched_tracks) {
      auto& track = tracks_[track_idx];
      track.time_since_update += 1;
    }

    tracks_.erase(
        std::remove_if(
            tracks_.begin(),
            tracks_.end(),
            [&](const Track& t) { return t.time_since_update > max_age_; }),
        tracks_.end());
  }

  std::vector<Track> get_tracks() const {
    return tracks_;
  }

 private:
  float iou_threshold_;
  int max_age_;
  int min_hits_;
  int next_id_;
  std::vector<Track> tracks_;

  void predict(Track& track) {
    // Kalman filter prediction step
    Eigen::Matrix4f F;
    F << 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1;

    Eigen::Matrix4f Q = Eigen::Matrix4f::Identity() * 0.01f;

    track.state = F * track.state;
    track.covariance = F * track.covariance * F.transpose() + Q;

    // Update bbox based on the predicted state
    float half_width =
        (track.bbox[2].item<float>() - track.bbox[0].item<float>()) / 2.0f;
    float half_height =
        (track.bbox[3].item<float>() - track.bbox[1].item<float>()) / 2.0f;
    track.bbox[0] = track.state[0] - half_width;
    track.bbox[1] = track.state[1] - half_height;
    track.bbox[2] = track.state[0] + half_width;
    track.bbox[3] = track.state[1] + half_height;
  }

  void update_kalman_filter(Track& track, const at::Tensor& detection) {
    // Kalman filter update step
    Eigen::Vector2f z;
    z << detection[0].item<float>(), detection[1].item<float>();

    Eigen::Matrix<float, 2, 4> H;
    H << 1, 0, 0, 0, 0, 1, 0, 0;

    Eigen::Matrix2f R = Eigen::Matrix2f::Identity() * 0.1f;

    Eigen::Vector2f y = z - H * track.state;
    Eigen::Matrix2f S = H * track.covariance * H.transpose() + R;
    Eigen::Matrix<float, 4, 2> K =
        track.covariance * H.transpose() * S.inverse();

    track.state = track.state + K * y;
    track.covariance = (Eigen::Matrix4f::Identity() - K * H) * track.covariance;

    // Update bbox based on the updated state
    float half_width =
        (detection[2].item<float>() - detection[0].item<float>()) / 2.0f;
    float half_height =
        (detection[3].item<float>() - detection[1].item<float>()) / 2.0f;
    track.bbox[0] = track.state[0] - half_width;
    track.bbox[1] = track.state[1] - half_height;
    track.bbox[2] = track.state[0] + half_width;
    track.bbox[3] = track.state[1] + half_height;
  }

  at::Tensor compute_iou_matrix(const at::Tensor& detections) {
    // Compute IOU matrix between detections and tracks
    if (tracks_.empty()) {
      return at::zeros({detections.size(0), 0});
    }

    at::Tensor track_boxes = at::stack(at::TensorList{tracks_.size()});
    for (size_t i = 0; i < tracks_.size(); ++i) {
      track_boxes[i] = tracks_[i].bbox;
    }

    at::Tensor x1_det =
        detections.index({at::indexing::Slice(), 0}).unsqueeze(1);
    at::Tensor y1_det =
        detections.index({at::indexing::Slice(), 1}).unsqueeze(1);
    at::Tensor x2_det =
        detections.index({at::indexing::Slice(), 2}).unsqueeze(1);
    at::Tensor y2_det =
        detections.index({at::indexing::Slice(), 3}).unsqueeze(1);

    at::Tensor x1_track =
        track_boxes.index({at::indexing::Slice(), 0}).unsqueeze(0);
    at::Tensor y1_track =
        track_boxes.index({at::indexing::Slice(), 1}).unsqueeze(0);
    at::Tensor x2_track =
        track_boxes.index({at::indexing::Slice(), 2}).unsqueeze(0);
    at::Tensor y2_track =
        track_boxes.index({at::indexing::Slice(), 3}).unsqueeze(0);

    at::Tensor inter_x1 = at::max(x1_det, x1_track);
    at::Tensor inter_y1 = at::max(y1_det, y1_track);
    at::Tensor inter_x2 = at::min(x2_det, x2_track);
    at::Tensor inter_y2 = at::min(y2_det, y2_track);

    at::Tensor inter_area =
        at::clamp(inter_x2 - inter_x1, 0) * at::clamp(inter_y2 - inter_y1, 0);

    at::Tensor det_area = (x2_det - x1_det) * (y2_det - y1_det);
    at::Tensor track_area = (x2_track - x1_track) * (y2_track - y1_track);

    at::Tensor union_area = det_area + track_area - inter_area;

    at::Tensor iou_matrix = inter_area / at::clamp(union_area, 1e-6);

    return iou_matrix;
  }

  std::tuple<
      std::vector<std::pair<int, int>>,
      std::vector<int>,
      std::vector<int>>
  match_detections(const at::Tensor& iou_matrix) {
    std::vector<std::pair<int, int>> matches;
    std::vector<int> unmatched_tracks;
    std::vector<int> unmatched_detections;

    if (iou_matrix.size(1) == 0) {
      // No tracks to match
      unmatched_detections.resize(iou_matrix.size(0));
      std::iota(unmatched_detections.begin(), unmatched_detections.end(), 0);
      return {matches, unmatched_tracks, unmatched_detections};
    }

    // Flatten the IOU matrix and sort in descending order
    at::Tensor flattened_indices =
        std::get<1>(iou_matrix.flatten().sort(/*dim=*/0, /*descending=*/true));
    at::Tensor flat_iou_values =
        iou_matrix.flatten().index({flattened_indices});

    // Track matched/unmatched detections and tracks
    std::vector<bool> detection_matched(iou_matrix.size(0), false);
    std::vector<bool> track_matched(iou_matrix.size(1), false);

    for (int i = 0; i < flattened_indices.size(0); ++i) {
      int flat_index = flattened_indices[i].item<int>();
      int det_idx = flat_index / iou_matrix.size(1);
      int track_idx = flat_index % iou_matrix.size(1);

      if (flat_iou_values[i].item<float>() < iou_threshold_) {
        break; // Stop if IOU is below threshold
      }

      if (!detection_matched[det_idx] && !track_matched[track_idx]) {
        matches.emplace_back(track_idx, det_idx);
        detection_matched[det_idx] = true;
        track_matched[track_idx] = true;
      }
    }

    // Find unmatched detections
    for (int i = 0; i < detection_matched.size(); ++i) {
      if (!detection_matched[i]) {
        unmatched_detections.push_back(i);
      }
    }

    // Find unmatched tracks
    for (int i = 0; i < track_matched.size(); ++i) {
      if (!track_matched[i]) {
        unmatched_tracks.push_back(i);
      }
    }

    return {matches, unmatched_tracks, unmatched_detections};
  }
};
#endif
} // namespace tracker
} // namespace hm
