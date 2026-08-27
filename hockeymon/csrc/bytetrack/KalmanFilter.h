#pragma once

#include <Eigen/Cholesky>
#include "Eigen/Core"

namespace hm {
namespace tracker {

namespace byte_kalman {

typedef Eigen::Matrix<float, 1, 4, Eigen::RowMajor> DETECTBOX;
typedef Eigen::Matrix<float, -1, 4, Eigen::RowMajor> DETECTBOXSS;

// Kalmanfilter
typedef Eigen::Matrix<float, 1, 8, Eigen::RowMajor> KAL_MEAN;
typedef Eigen::Matrix<float, 8, 8, Eigen::RowMajor> KAL_COVA;
typedef Eigen::Matrix<float, 1, 4, Eigen::RowMajor> KAL_HMEAN;
typedef Eigen::Matrix<float, 4, 4, Eigen::RowMajor> KAL_HCOVA;
using KAL_DATA = std::pair<KAL_MEAN, KAL_COVA>;
using KAL_HDATA = std::pair<KAL_HMEAN, KAL_HCOVA>;

class KalmanFilter {
 public:
  static const double chi2inv95[10];

  KalmanFilter();

  KAL_DATA initiate(const DETECTBOX& measurement);

  void predict(KAL_MEAN& mean, KAL_COVA& covariance);

  KAL_HDATA project(const KAL_MEAN& mean, const KAL_COVA& covariance);

  KAL_DATA update(
      const KAL_MEAN& mean,
      const KAL_COVA& covariance,
      const DETECTBOX& measurement);

  Eigen::Matrix<float, 1, -1> gating_distance(
      const KAL_MEAN& mean,
      const KAL_COVA& covariance,
      const std::vector<DETECTBOX>& measurements,
      bool only_position = false);

 private:
  Eigen::Matrix<float, 8, 8, Eigen::RowMajor> _motion_mat;
  Eigen::Matrix<float, 4, 8, Eigen::RowMajor> _update_mat;
  float _std_weight_position;
  float _std_weight_velocity;
};
} // namespace byte_kalman
} // namespace tracker

} // namespace hm
