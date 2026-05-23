#include "st_slam/frontend/st_nsp.h"
#include <Eigen/Eigenvalues>
#include <cmath>
#include <algorithm>
#include <iostream>

namespace st_slam {

STNSP::STNSP(double beta, double tau1, double degeneracy_threshold)
  : beta_(beta), tau1_(tau1),
    degeneracy_threshold_(degeneracy_threshold),
    degeneracy_measure_(0.0),
    eigenvalues_(Vec6::Zero()) {}

DegeneracyState STNSP::AnalyzeHessian(const Mat6& H_geo, double& condition_number) {
  Eigen::SelfAdjointEigenSolver<Mat6> eigs(H_geo);
  eigenvalues_ = eigs.eigenvalues();

  double max_eig = eigenvalues_.maxCoeff();
  double min_eig = eigenvalues_.minCoeff();
  condition_number = (min_eig > 1e-12) ? max_eig / min_eig
                                       : std::numeric_limits<double>::max();

  int degenerate_dirs = 0;
  for (int i = 0; i < 6; ++i) {
    if (std::abs(eigenvalues_(i)) < degeneracy_threshold_) {
      degenerate_dirs++;
    }
  }

  degeneracy_measure_ = static_cast<double>(degenerate_dirs) / 6.0;

  if (degenerate_dirs >= 5) {
    return DegeneracyState::FULL_DEGENERATE;
  } else if (degenerate_dirs >= 2) {
    return DegeneracyState::PARTIAL;
  }
  return DegeneracyState::NONE;
}

Mat6 STNSP::ComputeSafeProjection(const Mat6& H_geo, DegeneracyState state) {
  if (state == DegeneracyState::NONE) {
    return Mat6::Identity();
  }

  Eigen::SelfAdjointEigenSolver<Mat6> eigs(H_geo);
  const auto& eigenvalues = eigs.eigenvalues();
  const auto& eigenvectors = eigs.eigenvectors();

  Mat6 P_safe = Mat6::Identity();

  for (int i = 0; i < 6; ++i) {
    double s_i = math::sigmoid_centered(eigenvalues(i), beta_, tau1_);

    P_safe -= s_i * eigenvectors.col(i) * eigenvectors.col(i).transpose();
  }

  return P_safe;
}

Vec6 STNSP::ProjectIncrement(const Vec6& raw_increment,
                              const Mat6& P_safe,
                              DegeneracyState state) {
  if (state == DegeneracyState::NONE) {
    return raw_increment;
  }
  return P_safe * raw_increment;
}

} // namespace st_slam
