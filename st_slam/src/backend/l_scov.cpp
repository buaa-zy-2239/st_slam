#include "st_slam/backend/l_scov.h"
#include "st_slam/core/math_utils.h"
#include <Eigen/LU>
#include <cmath>
#include <algorithm>
#include <iostream>

namespace st_slam {

LSCOV::LSCOV(int num_eigenvectors, double spectral_noise_std,
              double hamming_threshold)
  : num_eigenvectors_(num_eigenvectors),
    spectral_noise_std_(spectral_noise_std),
    hamming_threshold_(hamming_threshold) {}

RLOMatrix LSCOV::ComputeRLOM(const Eigen::MatrixXd& eigenvectors,
                               int vec_idx) const {
  int n = eigenvectors.rows();
  const auto& phi = eigenvectors.col(vec_idx);

  RLOMatrix rlom;
  rlom.matrix = Eigen::MatrixXi::Zero(n, n);
  rlom.num_nodes = n;

  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      if (i == j) continue;
      double diff = phi(i) - phi(j);
      if (std::abs(diff) > spectral_noise_std_) {
        rlom.matrix(i, j) = (diff > 0) ? 1 : -1;
      } else {
        rlom.matrix(i, j) = 0;
      }
    }
  }

  return rlom;
}

double LSCOV::CompareRLOM(const RLOMatrix& rlom1,
                           const RLOMatrix& rlom2) const {
  if (rlom1.matrix.size() != rlom2.matrix.size()) return 1.0;

  int n = rlom1.matrix.rows();
  int diff_count = 0;
  int total_count = 0;

  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      if (i == j) continue;
      if (rlom1.matrix(i, j) != 0 || rlom2.matrix(i, j) != 0) {
        total_count++;
        if (rlom1.matrix(i, j) != rlom2.matrix(i, j)) {
          diff_count++;
        }
      }
    }
  }

  return total_count > 0 ? static_cast<double>(diff_count) / total_count : 0.0;
}

double LSCOV::ComputeSpectralGapBound(const Eigen::VectorXd& lap_eigenvalues,
                                        int eigen_idx) const {
  if (eigen_idx + 1 >= lap_eigenvalues.size()) {
    return std::numeric_limits<double>::infinity();
  }

  double gamma_k = lap_eigenvalues(eigen_idx + 1) - lap_eigenvalues(eigen_idx);
  return gamma_k;
}

double LSCOV::EstimatePerturbationBound(const Eigen::MatrixXd& L1,
                                          const Eigen::MatrixXd& L2,
                                          int eigen_idx) const {
  Eigen::MatrixXd delta_L = L1 - L2;
  double frob_delta = delta_L.norm();

  double gamma_k = 1.0;
  auto eigs1 = math::partial_eigen_decomposition(L1, eigen_idx + 2);
  auto eigs2 = math::partial_eigen_decomposition(L2, eigen_idx + 2);

  if (eigen_idx + 1 < eigs1.first.size() && eigen_idx + 1 < eigs2.first.size()) {
    double g1 = eigs1.first(eigen_idx + 1) - eigs1.first(eigen_idx);
    double g2 = eigs2.first(eigen_idx + 1) - eigs2.first(eigen_idx);
    gamma_k = std::min(g1, g2);
  }

  return frob_delta / std::max(gamma_k, 1e-10);
}

bool LSCOV::VerifyLoopCandidate(const SpectralHeatKernel& graph1,
                                 const SpectralHeatKernel& graph2,
                                 double& spectral_distance,
                                 double& hamming_distance) const {
  const auto& eigvecs1 = graph1.GetEigenvectors();
  const auto& eigvecs2 = graph2.GetEigenvectors();

  if (eigvecs1.cols() < num_eigenvectors_ || eigvecs2.cols() < num_eigenvectors_) {
    return false;
  }

  std::vector<RLOMatrix> rloms1, rloms2;

  for (int k = 0; k < num_eigenvectors_; ++k) {
    rloms1.push_back(ComputeRLOM(eigvecs1, k));
    rloms2.push_back(ComputeRLOM(eigvecs2, k));
  }

  double total_hamming = 0.0;
  for (int k = 0; k < num_eigenvectors_; ++k) {
    total_hamming += CompareRLOM(rloms1[k], rloms2[k]);
  }
  hamming_distance = total_hamming / num_eigenvectors_;

  const auto& eval1 = graph1.GetEigenvalues();
  const auto& eval2 = graph2.GetEigenvalues();
  spectral_distance = (eval1.head(num_eigenvectors_) -
                       eval2.head(num_eigenvectors_)).norm();

  bool is_valid = (hamming_distance < hamming_threshold_);

  if (!is_valid) {
    double perturb_bound = EstimatePerturbationBound(
      graph1.GetLaplacian(), graph2.GetLaplacian(), 0);
    double gamma_0 = ComputeSpectralGapBound(eval1, 0);

    if (perturb_bound < 0.5 * gamma_0) {
      is_valid = true;
    }
  }

  return is_valid;
}

bool LSCOV::IsValidSpectralMatch(
  const std::vector<RLOMatrix>& rloms_q,
  const std::vector<RLOMatrix>& rloms_m) const {
  if (rloms_q.size() != rloms_m.size()) return false;

  double total_hamming = 0.0;
  for (size_t k = 0; k < rloms_q.size(); ++k) {
    total_hamming += CompareRLOM(rloms_q[k], rloms_m[k]);
  }

  double avg_hamming = total_hamming / rloms_q.size();
  return avg_hamming < hamming_threshold_;
}

} // namespace st_slam
