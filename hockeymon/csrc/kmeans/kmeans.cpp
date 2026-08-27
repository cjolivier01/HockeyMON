#include "kmeans.h"
#include <memory.h>

#include <cassert>
#include <cstdlib>

namespace hm {
namespace kmeans {

/* return an array of cluster centers of size [numClusters][numCoords]       */
// float** seq_kmeans(
//     float** objects, /* in: [numObjs][numCoords] */
//     int numCoords, /* no. features */
//     int numObjs, /* no. objects */
//     int numClusters, /* no. clusters */
//     float threshold, /* % objects change membership */
//     int* membership, /* out: [numObjs] */
//     int* loop_iterations) {

// 0.001f,                           // less than 0.1% of the samples are reassigned in the end
// 0.1f,                             // activate Yinyang refinement with 0.1 threshold

// float** omp_kmeans(
//     int is_perform_atomic, /* in: */
//     float** objects, /* in: [numObjs][numCoords] */
//     int numCoords, /* no. coordinates */
//     int numObjs, /* no. objects */
//     int numClusters, /* no. clusters */
//     float threshold, /* % objects change membership */
//     int* membership) /* out: [numObjs] */
// {

void compute_kmeans(
    const std::vector<float>& points,
    int numClusters,
    int dim,
    int numIterations,
    std::vector<int>& assignments,
    KMEANS_TYPE kmeans_type) {
  const size_t object_count = points.size() / 2;
  assert((size_t)numClusters <= object_count);
  assignments.resize(object_count);

  std::vector<std::vector<float>> point_data;
  point_data.reserve(dim);
  size_t point_indexer = 0;
  for (std::size_t i = 0, n = points.size() / dim; i < n; ++i) {
    std::vector<float>& pt = point_data.emplace_back();
    pt.reserve(dim);
    for (int pn = 0; pn < dim; pn++) {
      pt.emplace_back(points[point_indexer++]);
    }
  }

  // Create an array of pointers
  std::vector<const float*> ptrs;
  ptrs.reserve(object_count);
  for (const auto& row : point_data) {
    ptrs.push_back(row.data());
  }

  float** results = nullptr;
  switch (kmeans_type) {
    case KMEANS_TYPE::KM_SEQ:
      results = seq_kmeans(ptrs.data(), dim, object_count, numClusters, 0.001f, assignments.data(), &numIterations);
      break;
    case KMEANS_TYPE::KM_OMP:
      results = omp_kmeans(
          /*is_perform_atomic=*/false, ptrs.data(), dim, object_count, numClusters, 0.001f, assignments.data());
      break;
    case KMEANS_TYPE::KM_CUDA:
    default:
      assert(false);
      break;
  }
  if (results) {
    ::free(results);
  }
}

} // namespace kmeans
} // namespace hm
