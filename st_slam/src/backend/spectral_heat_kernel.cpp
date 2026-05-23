#include "st_slam/backend/spectral_heat_kernel.h"
#include <Eigen/Eigenvalues>
#include <cmath>
#include <iostream>

namespace st_slam {

SpectralHeatKernel::SpectralHeatKernel(double ln_t_min, double ln_t_max,
                                         int num_moments, int num_eigenvalues)
  : ln_t_min_(ln_t_min), ln_t_max_(ln_t_max),
    num_moments_(num_moments), num_eigenvalues_(num_eigenvalues) {}

void SpectralHeatKernel::BuildGraph(const std::vector<Vec3>& pose_positions,
                                     double connection_radius) {
  int n = static_cast<int>(pose_positions.size());
  if (n < 2) return;

  W_ = Eigen::MatrixXd::Zero(n, n);

  for (int i = 0; i < n; ++i) {
    for (int j = i + 1; j < n; ++j) {
      double dist = (pose_positions[i] - pose_positions[j]).norm();
      if (dist < connection_radius) {
        double weight = std::exp(-dist * dist / (2.0 * 0.5 * 0.5));
        W_(i, j) = weight;
        W_(j, i) = weight;
      }
    }
  }

  L_ = math::normalize_laplacian(W_);
}

void SpectralHeatKernel::BuildGraphFromAdjacency(const Eigen::MatrixXd& W) {
  W_ = W;
  L_ = math::normalize_laplacian(W_);
}

void SpectralHeatKernel::ComputeEigenDecomposition() {
  if (L_.rows() == 0) return;

  int n = L_.rows();
  int k = std::min(num_eigenvalues_, n);

  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigs(L_);
  eigenvalues_ = eigs.eigenvalues();
  eigenvectors_ = eigs.eigenvectors();
}

Eigen::VectorXd SpectralHeatKernel::ComputeHeatKernelSignature() {
  ComputeEigenDecomposition();
  return ComputeAbsoluteMoments();
}

Eigen::VectorXd SpectralHeatKernel::ComputeAbsoluteMoments() {
  if (eigenvalues_.size() == 0) return Eigen::VectorXd::Zero(num_moments_);

  int n = L_.rows();
  double rho = (n > 1) ? std::log(static_cast<double>(n)) / 1.0 : 1.0;

  moment_vector_ = Eigen::VectorXd::Zero(num_moments_);

  int num_steps = 100;
  double d_ln_t = (ln_t_max_ - ln_t_min_) / num_steps;

  for (int m = 0; m < num_moments_; ++m) {
    double integral = 0.0;

    for (int step = 0; step < num_steps; ++step) {
      double ln_t = ln_t_min_ + (step + 0.5) * d_ln_t;
      double t = std::exp(ln_t);

      double trace = 0.0;
      for (int i = 0; i < eigenvalues_.size(); ++i) {
        trace += std::exp(-t * eigenvalues_(i));
      }

      double integrand = std::pow(ln_t, m) * trace / (n * rho);
      integral += integrand * d_ln_t;
    }

    moment_vector_(m) = integral;
  }

  return moment_vector_;
}

double SpectralHeatKernel::CompareSignatures(const Eigen::VectorXd& sig1,
                                              const Eigen::VectorXd& sig2) const {
  if (sig1.size() != sig2.size()) return std::numeric_limits<double>::max();

  double diff = (sig1 - sig2).norm();
  double norm = std::max(sig1.norm(), sig2.norm());
  if (norm < 1e-10) return 1.0;

  return diff / norm;
}

} // namespace st_slam
