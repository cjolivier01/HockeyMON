#include "hockeymon/csrc/play_tracker/PlayTracker.h"
#include "hockeymon/csrc//kmeans/kmeans.h"
#include "hockeymon/csrc/play_tracker/LivingBoxImpl.h"

#include <algorithm>
#include <cassert>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace hm {
namespace play_tracker {

namespace {

constexpr size_t kMinTracksBeforePruning = 3;
// Arbitrarily large jump in bbox center that would be a bug
constexpr size_t kMaxJumpAssertionValue = 300;
#if 0
std::tuple</*index_removed=*/size_t, std::vector<size_t>, std::vector<BBox>>
remove_largest(std::vector<size_t> ids, std::vector<BBox> bboxes) {
  double largest_area = 0.0;
  size_t largest_index = kBadIdOrIndex;
  for (size_t i = 0, n = ids.size(); i < n; ++i) {
    double this_area = bboxes[i].area();
    if (this_area > largest_area) {
      largest_area = this_area;
      largest_index = i;
    }
  }
  // Making a new vector will be more expensive than just removing a vector item
  // because to create a new vector, you are guaranteed to traverse the entire
  // vector except one item, plus construction and allocation overhead, etc.
  // So, simply erase the element.
  if (largest_index != kBadIdOrIndex) {
    ids.erase(ids.begin() + largest_index);
    bboxes.erase(bboxes.begin() + largest_index);
  }
  return std::make_tuple(largest_index, std::move(ids), std::move(bboxes));
}
#endif

struct PruneResults {
  size_t largest_area_index{kBadIdOrIndex};
  size_t leftmost_index{kBadIdOrIndex};
  size_t rightmost_index{kBadIdOrIndex};
};

PruneResults remove_extremes(
    const std::vector<size_t>& ids,
    const std::vector<BBox>& bboxes,
    bool ignore_largest,
    bool ignore_lr_extremes,
    const size_t min_boxes = 10) {
  if (!ignore_largest && !ignore_lr_extremes) {
    return PruneResults();
  }
  PruneResults results;
  double largest_area = 0.0;

  // size_t leftmost_index = kBadIdOrIndex;
  int64_t leftmost_x = std::numeric_limits<int64_t>::max();

  // size_t rightmost_index = kBadIdOrIndex;
  int64_t rightmost_x = std::numeric_limits<int64_t>::min();

  const bool remove_lr = (bboxes.size() >= min_boxes) && ignore_lr_extremes;

  // Single loop: compute largest area as well as leftmost/rightmost center x.
  const size_t n = ids.size();
  for (size_t i = 0; i < n; ++i) {
    const auto& box = bboxes[i];

    if (ignore_largest) {
      // Compute area.
      double area = box.area();
      if (area > largest_area) {
        largest_area = area;
        results.largest_area_index = i;
      }
    }

    if (remove_lr) {
      // Compute center x.
      int64_t center_x = (box.left + box.right) / 2;
      if (center_x < leftmost_x) {
        leftmost_x = center_x;
        results.leftmost_index = i;
      }
      if (center_x > rightmost_x) {
        rightmost_x = center_x;
        results.rightmost_index = i;
      }
    }
  }

  return results;
}

std::vector<BBox> prune_bboxes(
    std::vector<BBox> bboxes,
    const std::set<size_t>& remove_indices) {
  if (remove_indices.empty()) {
    return bboxes;
  }
  std::vector<BBox> new_bboxes;
  new_bboxes.reserve(bboxes.size());
  for (size_t i = 0, n = bboxes.size(); i < n; ++i) {
    if (!remove_indices.count(i)) {
      new_bboxes.emplace_back(bboxes[i]);
    }
  }
  return new_bboxes;
}

std::vector<size_t> get_largest_cluster_item_indexes(
    const std::vector<float> points,
    size_t num_clusters,
    int dim) {
  std::size_t item_count = points.size() / dim;
  std::vector<int> assignments;
  assignments.reserve(item_count);
  hm::kmeans::compute_kmeans(
      points,
      num_clusters,
      /*dim=*/2,
      /*numIterations=*/6,
      assignments,
      hm::kmeans::KMEANS_TYPE::KM_OMP);
  std::vector<std::vector<size_t>> cluster_to_indexes(num_clusters);
  std::for_each(
      cluster_to_indexes.begin(),
      cluster_to_indexes.end(),
      [item_count](auto& v) { v.reserve(item_count); });
  std::vector<size_t> frequency(num_clusters, 0);
  for (size_t item_index = 0; item_index < item_count; ++item_index) {
    int cluster_nr = assignments.at(item_index);
    cluster_to_indexes[cluster_nr].push_back(item_index);
    ++frequency[cluster_nr];
  }
  return cluster_to_indexes.at(std::distance(
      frequency.begin(), std::max_element(frequency.begin(), frequency.end())));
}

BBox get_union_bounding_box(const std::vector<BBox>& boxes) {
  if (boxes.empty()) {
    return BBox();
  }

  float minTop = std::numeric_limits<float>::max();
  float minLeft = std::numeric_limits<float>::max();
  float maxBottom = std::numeric_limits<float>::min();
  float maxRight = std::numeric_limits<float>::min();

  for (const auto& box : boxes) {
    minTop = std::min(minTop, box.top);
    minLeft = std::min(minLeft, box.left);
    maxBottom = std::max(maxBottom, box.bottom);
    maxRight = std::max(maxRight, box.right);
  }

  return BBox(/*l=*/minLeft, /*t=*/minTop, /*r=*/maxRight, /*b=*/maxBottom);
}

int find_outlier_index(const std::vector<BBox>& boxes, float r = 1.0) {
  int numBoxes = boxes.size();
  if (numBoxes < 2)
    return -1; // need at least 2 boxes for comparison

  // Precompute centers and global extremes for union bounding box.
  std::vector<float> centers(numBoxes);

  // For union bounding box, we need the global min left and max right, plus the
  // second min and second max.
  float globalMinLeft = std::numeric_limits<float>::max();
  float secondMinLeft = std::numeric_limits<float>::max();
  int globalMinLeftIndex = -1;

  float globalMaxRight = std::numeric_limits<float>::lowest();
  float secondMaxRight = std::numeric_limits<float>::lowest();
  int globalMaxRightIndex = -1;

  // Also track the leftmost and rightmost candidate based on center.
  int leftCandidate = 0;
  int rightCandidate = 0;
  float leftCandidateCenter = std::numeric_limits<float>::max();
  float rightCandidateCenter = std::numeric_limits<float>::lowest();

  for (int i = 0; i < numBoxes; i++) {
    // Compute center for the i-th box.
    centers[i] = (boxes[i].left + boxes[i].right) * 0.5f;

    // Update leftmost candidate (smallest center).
    if (centers[i] < leftCandidateCenter) {
      leftCandidateCenter = centers[i];
      leftCandidate = i;
    }
    // Update rightmost candidate (largest center).
    if (centers[i] > rightCandidateCenter) {
      rightCandidateCenter = centers[i];
      rightCandidate = i;
    }

    // Update global minimum left and second minimum left.
    if (boxes[i].left < globalMinLeft) {
      secondMinLeft = globalMinLeft;
      globalMinLeft = boxes[i].left;
      globalMinLeftIndex = i;
    } else if (boxes[i].left < secondMinLeft) {
      secondMinLeft = boxes[i].left;
    }

    // Update global maximum right and second maximum right.
    if (boxes[i].right > globalMaxRight) {
      secondMaxRight = globalMaxRight;
      globalMaxRight = boxes[i].right;
      globalMaxRightIndex = i;
    } else if (boxes[i].right > secondMaxRight) {
      secondMaxRight = boxes[i].right;
    }
  }

  // --- Check leftCandidate (box with smallest center) ---
  // Find the second left candidate center: the smallest center >
  // leftCandidateCenter.
  float secondLeftCandidateCenter = std::numeric_limits<float>::max();
  bool foundLeftNeighbor = false;
  for (int i = 0; i < numBoxes; i++) {
    if (i == leftCandidate)
      continue;
    if (centers[i] > leftCandidateCenter &&
        centers[i] < secondLeftCandidateCenter) {
      secondLeftCandidateCenter = centers[i];
      foundLeftNeighbor = true;
    }
  }

  // Compute union bounding box excluding the leftCandidate.
  float unionLeft =
      (leftCandidate == globalMinLeftIndex) ? secondMinLeft : globalMinLeft;
  float unionRight =
      (leftCandidate == globalMaxRightIndex) ? secondMaxRight : globalMaxRight;
  float unionWidth = unionRight - unionLeft;

  // If the gap between leftCandidate and its neighbor exceeds r * unionWidth,
  // return leftCandidate.
  if (foundLeftNeighbor) {
    float gap = secondLeftCandidateCenter - leftCandidateCenter;
    if (gap > r * unionWidth)
      return leftCandidate;
  }

  // --- Check rightCandidate (box with largest center) ---
  // Find the second right candidate center: the largest center <
  // rightCandidateCenter.
  float secondRightCandidateCenter = std::numeric_limits<float>::lowest();
  bool foundRightNeighbor = false;
  for (int i = 0; i < numBoxes; i++) {
    if (i == rightCandidate)
      continue;
    if (centers[i] < rightCandidateCenter &&
        centers[i] > secondRightCandidateCenter) {
      secondRightCandidateCenter = centers[i];
      foundRightNeighbor = true;
    }
  }

  // Compute union bounding box excluding the rightCandidate.
  unionLeft =
      (rightCandidate == globalMinLeftIndex) ? secondMinLeft : globalMinLeft;
  unionRight =
      (rightCandidate == globalMaxRightIndex) ? secondMaxRight : globalMaxRight;
  unionWidth = unionRight - unionLeft;

  // If the gap between rightCandidate and its neighbor exceeds r * unionWidth,
  // return rightCandidate.
  if (foundRightNeighbor) {
    float gap = rightCandidateCenter - secondRightCandidateCenter;
    if (gap > r * unionWidth)
      return rightCandidate;
  }

  // If neither candidate meets the criterion, return -1.
  return -1;
}

} // namespace

PlayTracker::PlayTracker(
    const BBox& initial_box,
    const PlayTrackerConfig& config)
    : config_(config), play_detector_(config.play_detector, this) {
  create_boxes(initial_box);
}

void PlayTracker::create_boxes(const BBox& initial_box) {
  for (std::size_t i = 0; i < config_.living_boxes.size(); ++i) {
    living_boxes_.emplace_back(std::make_unique<LivingBox>(
        std::to_string(i + 1), initial_box, config_.living_boxes[i]));
  }
}

PlayTracker::ClusterBoxes PlayTracker::get_cluster_boxes(
    const std::vector<BBox>& tracking_boxes) const {
  ClusterBoxes cluster_boxes_result;

  std::vector<float> points;
  points.reserve(tracking_boxes.size() << 1);

  for (const auto& box : tracking_boxes) {
    const Point c = box.center();
    points.push_back(c.x);
    points.push_back(c.y);
  }

  // Restrict toi cluster sizes that have some meaning based on how many
  // tracking objects we have (or are valid, for that matter)
  std::vector<size_t> cluster_sizes;
  cluster_sizes.reserve(cluster_sizes_.size());
  for (size_t cluster_size : cluster_sizes_) {
    // Equal or less clustering is meaningless (also +1)
    if (tracking_boxes.size() > cluster_size + 1) {
      cluster_sizes.emplace_back(cluster_size);
    }
  }

  const size_t cluster_count = cluster_sizes.size();
  std::vector<std::vector<size_t>> cluster_item_indexes(cluster_count);
  std::vector<BBox> cluster_bboxes(cluster_count);

#pragma omp parallel for num_threads(cluster_count)
  for (size_t cluster_id = 0; cluster_id < cluster_count; ++cluster_id) {
    auto& this_cluster_item_indexes = cluster_item_indexes.at(cluster_id);
    this_cluster_item_indexes = get_largest_cluster_item_indexes(
        points,
        cluster_sizes.at(cluster_id),
        /*dim=*/2);
    assert(!this_cluster_item_indexes.empty());
    std::vector<BBox> bboxes;
    bboxes.reserve(this_cluster_item_indexes.size());
    for (size_t idx : this_cluster_item_indexes) {
      bboxes.emplace_back(tracking_boxes.at(idx));
    }

    if (config_.ignore_outlier_players && bboxes.size() > 3) {
      int outlier_index = find_outlier_index(
          bboxes, /*r=*/config_.ignore_outlier_players_dist_ratio);
      if (outlier_index >= 0) {
        auto outlier_iter = bboxes.begin() + outlier_index;
        // FIXME(maybe): Removing this is slow :(
        // std::cout << "Removing outlier box: " << *outlier_iter << std::endl;
        cluster_boxes_result.removed_cluster_outlier_box.emplace(
            cluster_sizes.at(cluster_id), bboxes[outlier_index]);
        bboxes.erase(outlier_iter);
      }
    }
    cluster_bboxes.at(cluster_id) = get_union_bounding_box(bboxes);
  }

  for (size_t i = 0; i < cluster_sizes.size(); ++i) {
    const auto& this_cluster_box = cluster_bboxes.at(i);
    cluster_boxes_result.cluster_boxes[cluster_sizes[i]] = this_cluster_box;
  }

  cluster_boxes_result.final_cluster_box =
      get_union_bounding_box(cluster_bboxes);

  return cluster_boxes_result;
}

void PlayTracker::set_bboxes(const std::vector<BBox>& bboxes) {
  if (bboxes.size() != 1 && bboxes.size() != living_boxes_.size()) {
    throw std::runtime_error(
        "Number of bounding boxes should be one or the same as the number of living boxes");
  }
  for (size_t i = 0, n = living_boxes_.size(); i < n; ++i) {
    living_boxes_[i]->set_bbox(bboxes.size() == 1 ? bboxes[0] : bboxes.at(i));
  }
}

void PlayTracker::set_bboxes_scaled(BBox bbox, float scale_step) {
  const BBox arena_box = get_play_box();
  if (bbox.empty()) {
    bbox = arena_box;
  }
  for (size_t i = 0, n = living_boxes_.size(); i < n; ++i) {
    bbox = clamp_box(bbox, arena_box);
    living_boxes_[i]->set_bbox(bbox);
    BBox new_bbox =
        clamp_box(bbox.make_scaled(scale_step, scale_step), arena_box);
    // We don't allow it to go an empty box (w==0 or h==0)
    if (!new_bbox.empty()) {
      bbox = new_bbox;
    }
  }
}

std::shared_ptr<ILivingBox> PlayTracker::get_live_box(size_t index) const {
  return living_boxes_.at(index);
}

BBox PlayTracker::get_play_box() const {
  auto arena_box = living_boxes_.at(living_boxes_.size() - 1)->get_arena_box();
  assert(arena_box.has_value());
  return *arena_box;
}

PlayTrackerResults PlayTracker::forward(
    std::vector<size_t>& tracking_ids,
    std::vector<BBox>& tracking_boxes,
    bool debug_to_stdout) {
  assert(tracking_ids.size() == tracking_boxes.size());

  PlayTrackerResults results;
  ScopedLogCapture log_capture(&results.log_messages, debug_to_stdout);

  std::set<size_t> ignore_tracking_ids;
  PruneResults prune_results;
  std::vector<BBox>* p_cluster_bboxes = &tracking_boxes;
  std::vector<BBox> repl_bboxes;
  if ((config_.ignore_largest_bbox || config_.ignore_left_and_right_extremes) &&
      tracking_ids.size() > kMinTracksBeforePruning) {
    // prune_results = remove_largest(tracking_ids, tracking_boxes);
    prune_results = remove_extremes(
        tracking_ids,
        *p_cluster_bboxes,
        config_.ignore_largest_bbox,
        config_.ignore_left_and_right_extremes);
    if (prune_results.largest_area_index != kBadIdOrIndex) {
      ignore_tracking_ids.emplace(
          tracking_ids.at(prune_results.largest_area_index));
      results.largest_tracking_bbox = Track{
          .tracking_id = tracking_ids.at(prune_results.largest_area_index),
          .bbox = tracking_boxes.at(prune_results.largest_area_index),
      };
    }
    if (prune_results.leftmost_index != kBadIdOrIndex) {
      ignore_tracking_ids.emplace(
          tracking_ids.at(prune_results.leftmost_index));
      results.leftmost_tracking_bbox = Track{
          .tracking_id = tracking_ids.at(prune_results.leftmost_index),
          .bbox = tracking_boxes.at(prune_results.leftmost_index),
      };
    }
    if (prune_results.rightmost_index != kBadIdOrIndex) {
      ignore_tracking_ids.emplace(
          tracking_ids.at(prune_results.rightmost_index));
      results.rightmost_tracking_bbox = Track{
          .tracking_id = tracking_ids.at(prune_results.rightmost_index),
          .bbox = tracking_boxes.at(prune_results.rightmost_index),
      };
    }
    std::set<size_t> remove_indices{
        prune_results.largest_area_index,
        prune_results.leftmost_index,
        prune_results.rightmost_index};
    remove_indices.erase(kBadIdOrIndex);
    if (!remove_indices.empty()) {
      repl_bboxes = prune_bboxes(*p_cluster_bboxes, remove_indices);
      p_cluster_bboxes = &repl_bboxes;
    }
  }

  //
  // Compute the next box
  //
  const BBox arena_box = get_play_box();
  {
    ClusterBoxes cluster_boxes_result = get_cluster_boxes(*p_cluster_bboxes);
    results.cluster_boxes = std::move(cluster_boxes_result.cluster_boxes);
    results.removed_cluster_outlier_box =
        std::move(cluster_boxes_result.removed_cluster_outlier_box);
    results.final_cluster_box = cluster_boxes_result.final_cluster_box;
  }

  BBox start_bbox = results.final_cluster_box;
  // Special cases
  if (start_bbox.empty()) {
    // probably not enough tracks
    if (!tracking_boxes.empty()) {
      // Just union all tracking boxes
      start_bbox = get_union_bounding_box(tracking_boxes);
    } else {
      // Last resort, the entire arena area
      start_bbox = arena_box;
    }
  }
  clamp_box(start_bbox, arena_box);

  if (!state_.tick_count_ && config_.no_wide_start) {
    // We start at our first detected box
    set_bboxes_scaled(start_bbox, 1.2);
  }
  BBox current_box = start_bbox;

  results.play_detection = play_detector_.forward(
      /*frame_id=*/state_.tick_count_,
      /*current_target_bbox=*/current_box,
      /*current_roi_center=*/get_live_box(0)->bounding_box().center(),
      tracking_ids,
      tracking_boxes,
      ignore_tracking_ids);

  if (results.play_detection->breakaway_target_bbox.has_value()) {
    current_box = *results.play_detection->breakaway_target_bbox;
  }

  // BEGIN Enforce minimum player count
  // After we did all that work, let's check an exception that there aren't
  // enough players to allow a smaller box than the arena
  // This should also help to see a penalty shot.
  if (tracking_ids.size() < config_.min_tracked_players) {
    current_box = arena_box;
    if (state_.tracked_player_count >= config_.min_tracked_players) {
      // We weren't tracking on the last tick
      // std::cout << "Too few players tracked (" << tracking_ids.size()
      //           << "), expanding play box players, so resuming play tracking"
      //           << std::endl;
    }
  } else if (
      state_.tracked_player_count < config_.min_tracked_players &&
      state_.tick_count_) {
    // std::cout << "Now tracking " << tracking_ids.size()
    //           << " players, so resuming play tracking" << std::endl;
  }
  state_.tracked_player_count = tracking_ids.size();
  // END Enforce minimum player count

  for (std::size_t i = 0; i < living_boxes_.size(); ++i) {
    auto& living_box = living_boxes_[i];
    auto last_center = living_box->bounding_box().center();
    current_box = living_box->forward(current_box);
    auto new_center = living_box->bounding_box().center();
    if (norm(new_center - last_center) > kMaxJumpAssertionValue) {
      hm_log_warning(
          "Detected large jump at tick count " + std::to_string(state_.tick_count_));
    }
    results.tracking_boxes.emplace_back(current_box);
  }
  ++state_.tick_count_;
  return results;
}

} // namespace play_tracker
} // namespace hm
