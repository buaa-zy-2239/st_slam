#ifndef ST_SLAM_L_SCOV_H
#define ST_SLAM_L_SCOV_H

#include "st_slam/core/types.h"
#include "st_slam/backend/spectral_heat_kernel.h"
#include <Eigen/Dense>
#include <unordered_map>
#include <vector>

namespace st_slam {

struct RLOMatrix {
  Eigen::MatrixXi matrix;
  int num_nodes;
};

class LSCOV {
public:
  explicit LSCOV(int num_eigenvectors = 4,
                  double spectral_noise_std = 0.05,
                  double hamming_threshold = 0.2);

  RLOMatrix ComputeRLOM(const Eigen::MatrixXd& eigenvectors, int vec_idx) const;

  double CompareRLOM(const RLOMatrix& rlom1, const RLOMatrix& rlom2) const;

  bool VerifyLoopCandidate(
    const SpectralHeatKernel& graph1,
    const SpectralHeatKernel& graph2,
    double& spectral_distance,
    double& hamming_distance) const;

  double ComputeSpectralGapBound(
    const Eigen::VectorXd& lap_eigenvalues,
    int eigen_idx) const;

  double EstimatePerturbationBound(
    const Eigen::MatrixXd& L1,
    const Eigen::MatrixXd& L2,
    int eigen_idx) const;

  void SetThreshold(double t) { hamming_threshold_ = t; }

private:
  int num_eigenvectors_;
  double spectral_noise_std_;
  double hamming_threshold_;

  bool IsValidSpectralMatch(
    const std::vector<RLOMatrix>& rloms_q,
    const std::vector<RLOMatrix>& rloms_m) const;
};

} // namespace st_slam

#endif
