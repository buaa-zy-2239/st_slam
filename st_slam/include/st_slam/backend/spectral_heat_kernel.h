#ifndef ST_SLAM_SPECTRAL_HEAT_KERNEL_H
#define ST_SLAM_SPECTRAL_HEAT_KERNEL_H

#include "st_slam/core/types.h"
#include "st_slam/core/math_utils.h"
#include <Eigen/Dense>
#include <vector>

namespace st_slam {

class SpectralHeatKernel {
public:
  explicit SpectralHeatKernel(double ln_t_min = -5.0,
                               double ln_t_max = 5.0,
                               int num_moments = 3,
                               int num_eigenvalues = 30);

  void BuildGraph(const std::vector<Vec3>& pose_positions,
                   double connection_radius = 2.0);

  void BuildGraphFromAdjacency(const Eigen::MatrixXd& W);

  Eigen::VectorXd ComputeHeatKernelSignature();

  Eigen::VectorXd ComputeAbsoluteMoments();

  double CompareSignatures(const Eigen::VectorXd& sig1,
                           const Eigen::VectorXd& sig2) const;

  Eigen::VectorXd GetMomentVector() const { return moment_vector_; }

  Eigen::MatrixXd GetLaplacian() const { return L_; }
  Eigen::VectorXd GetEigenvalues() const { return eigenvalues_; }
  Eigen::MatrixXd GetEigenvectors() const { return eigenvectors_; }

private:
  double ln_t_min_, ln_t_max_;
  int num_moments_, num_eigenvalues_;

  Eigen::MatrixXd W_;
  Eigen::MatrixXd L_;
  Eigen::VectorXd eigenvalues_;
  Eigen::MatrixXd eigenvectors_;
  Eigen::VectorXd moment_vector_;

  void ComputeEigenDecomposition();

  void ComputeLogTimeMoments();
};

} // namespace st_slam

#endif
