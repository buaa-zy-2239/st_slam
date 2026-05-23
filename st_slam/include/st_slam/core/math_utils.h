#ifndef ST_SLAM_MATH_UTILS_H
#define ST_SLAM_MATH_UTILS_H

#include "types.h"
#include <Eigen/Dense>
#include <Eigen/SVD>
#include <cmath>
#include <vector>
#include <algorithm>

namespace st_slam {
namespace math {

inline double sigmoid(double x, double beta) {
  return 1.0 / (1.0 + std::exp(-beta * x));
}

inline double sigmoid_centered(double x, double beta, double threshold) {
  return 1.0 / (1.0 + std::exp(beta * (x - threshold)));
}

inline double cauchy_loss(double r, double c) {
  return c * c * std::log(1.0 + (r * r) / (c * c));
}

inline double cauchy_derivative(double r, double c) {
  return 2.0 * r / (1.0 + (r * r) / (c * c));
}

inline double cauchy_weight(double r, double c) {
  return 1.0 / (1.0 + (r * r) / (c * c));
}

inline double huber_weight(double r, double delta) {
  double abs_r = std::abs(r);
  if (abs_r <= delta) return 1.0;
  return delta / abs_r;
}

Vec6 se3_log(const SE3& pose);

SE3 se3_exp(const Vec6& twist);

Mat6 adjoint_matrix(const SE3& pose);

Mat3 skew_symmetric(const Vec3& v);

double frobenius_norm(const MatX& M);

double condition_number(const MatX& M);

MatX pseudo_inverse(const MatX& M, double epsilon = 1e-8);

Vec3 compute_normal(const std::vector<Vec3>& points);

double total_variation_derivative(const std::vector<double>& signal, int order = 1);

template<typename T>
int signum(T val) {
  return (T(0) < val) - (val < T(0));
}

Mat4 delta_pose_error(const SE3& est, const SE3& gt);

SE3 umeyama_se3_alignment(
  const std::vector<Vec3>& source,
  const std::vector<Vec3>& target);

void align_trajectory(
  std::vector<SE3>& est_poses,
  const std::vector<SE3>& gt_poses);

double compute_ate_rmse(
  const std::vector<SE3>& est_poses_aligned,
  const std::vector<SE3>& gt_poses);

double compute_ate_mean(
  const std::vector<SE3>& est_poses_aligned,
  const std::vector<SE3>& gt_poses);

double compute_ate_median(
  const std::vector<SE3>& est_poses_aligned,
  const std::vector<SE3>& gt_poses);

double compute_ate_std(
  const std::vector<SE3>& est_poses_aligned,
  const std::vector<SE3>& gt_poses);

struct RPEMetrics {
  double rmse_trans;
  double mean_trans;
  double median_trans;
  double std_trans;
  double rmse_rot_deg;
  double mean_rot_deg;
};

RPEMetrics compute_rpe_full(
  const std::vector<SE3>& est_poses,
  const std::vector<SE3>& gt_poses,
  const std::vector<double>& timestamps);

double compute_rpe_rmse(const std::vector<SE3>& est_poses,
                         const std::vector<SE3>& gt_poses,
                         const std::vector<double>& timestamps);

Eigen::MatrixXd compute_laplacian(const Eigen::MatrixXd& W);

Eigen::MatrixXd normalize_laplacian(const Eigen::MatrixXd& W);

std::pair<Eigen::VectorXd, Eigen::MatrixXd>
partial_eigen_decomposition(const Eigen::MatrixXd& L, int num_eigenvalues);

} // namespace math
} // namespace st_slam

#endif
