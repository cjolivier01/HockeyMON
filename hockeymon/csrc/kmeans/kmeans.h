#pragma once

#include <vector>

#include <assert.h>

/*
// #define msg(format, ...)                    \
//   do {                                      \
//     fprintf(stderr, format, ##__VA_ARGS__); \
//   } while (0)
// #define err(format, ...)                    \
//   do {                                      \
//     fprintf(stderr, format, ##__VA_ARGS__); \
//     exit(1);                                \
//   } while (0)
*/
#define malloc2D(name, xDim, yDim, type)                 \
  do {                                                   \
    name = (type**)malloc(xDim * sizeof(type*));         \
    assert(name != NULL);                                \
    name[0] = (type*)malloc(xDim * yDim * sizeof(type)); \
    assert(name[0] != NULL);                             \
    for (size_t i = 1; i < xDim; i++)                    \
      name[i] = name[i - 1] + yDim;                      \
  } while (0)

#ifdef __CUDACC__
inline void checkCuda(cudaError_t e) {
  if (e != cudaSuccess) {
    // cudaGetErrorString() isn't always very helpful. Look up the error
    // number in the cudaError enum in driver_types.h in the CUDA includes
    // directory for a better explanation.
    err("CUDA Error %d: %s\n", e, cudaGetErrorString(e));
  }
}

inline void checkLastCudaError() {
  checkCuda(cudaGetLastError());
}
#endif

namespace hm {
namespace kmeans {

/* return an array of cluster centers of size [numClusters][numCoords]       */
float** omp_kmeans(int, const float**, int, int, int, float, int*);
float** seq_kmeans(const float**, int, int, int, float, int*, int*);
float** cuda_kmeans(float**, int, int, int, float, int*, int*);

float** file_read(int, char*, int*, int*);
int file_write(char*, int, int, int, float**, int*);

double wtime(void);

enum class KMEANS_TYPE {
  KM_SEQ,
  KM_OMP,
  KM_CUDA, // Cuda will be much slower on small datasets
};

void compute_kmeans(
    const std::vector<float>& points,
    int numClusters,
    int dim,
    int numIterations,
    std::vector<int>& assignments,
    KMEANS_TYPE kmeans_type = KMEANS_TYPE::KM_SEQ);

extern int _debug;
} // namespace kmeans
} // namespace hm
