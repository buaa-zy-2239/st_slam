#include "st_slam/core/math_utils.h"
#include <Eigen/Eigenvalues>
#include <limits>
#include <iostream>
#include <algorithm>

namespace st_slam {
namespace math {

Vec6 se3_log(const SE3& pose) {
  AngleAxis aa(pose.rot);
  Vec3 rot_vec = aa.axis() * aa.angle();
  Vec3 trans = pose.trans;
  Vec6 twist;
  twist << rot_vec, trans;
  return twist;
}

SE3 se3_exp(const Vec6& twist) {
  Vec3 rot_vec = twist.head<3>();
  Vec3 trans = twist.tail<3>();
  double angle = rot_vec.norm();
  if (angle < 1e-8) {
    return SE3(Quat::Identity(), trans);
  }
  Vec3 axis = rot_vec / angle;
  Quat rot(AngleAxis(angle, axis));
  return SE3(rot, trans);
}

Mat6 adjoint_matrix(const SE3& pose) {
  Mat6 adj = Mat6::Zero();
  Mat3 R = pose.rot.toRotationMatrix();
  Mat3 t_skew = skew_symmetric(pose.trans);
  adj.block<3,3>(0,0) = R;
  adj.block<3,3>(0,3) = t_skew * R;
  adj.block<3,3>(3,3) = R;
  return adj;
}

Mat3 skew_symmetric(const Vec3& v) {
  Mat3 m;
  m << 0, -v(2), v(1),
       v(2), 0, -v(0),
      -v(1), v(0), 0;
  return m;
}

double frobenius_norm(const MatX& M) {
  return M.norm();
}

double condition_number(const MatX& M) {
  Eigen::JacobiSVD<MatX> svd(M, Eigen::ComputeThinU | Eigen::ComputeThinV);
  double max_sv = svd.singularValues()(0);
  double min_sv = svd.singularValues()(svd.singularValues().size() - 1);
  if (min_sv < 1e-12) return std::numeric_limits<double>::max();
  return max_sv / min_sv;
}

MatX pseudo_inverse(const MatX& M, double epsilon) {
  Eigen::JacobiSVD<MatX> svd(M, Eigen::ComputeThinU | Eigen::ComputeThinV);
  const auto& sv = svd.singularValues();
  VecX inv_sv(sv.size());
  for (int i = 0; i < sv.size(); ++i) {
    inv_sv(i) = (sv(i) > epsilon) ? 1.0 / sv(i) : 0.0;
  }
  return svd.matrixV() * inv_sv.asDiagonal() * svd.matrixU().transpose();
}

Vec3 compute_normal(const std::vector<Vec3>& points) {
  if (points.size() < 3) return Vec3::UnitZ();
  Vec3 centroid = Vec3::Zero();
  for (const auto& p : points) centroid += p;
  centroid /= points.size();

  Mat3 cov = Mat3::Zero();
  for (const auto& p : points) {
    Vec3 d = p - centroid;
    cov += d * d.transpose();
  }
  Eigen::SelfAdjointEigenSolver<Mat3> eigs(cov);
  return eigs.eigenvectors().col(0);
}

double total_variation_derivative(const std::vector<double>& signal, int order) {
  if (signal.size() < 2) return 0.0;
  if (order == 0) {
    double tv = 0.0;
    for (size_t i = 1; i < signal.size(); ++i) {
      tv += std::abs(signal[i] - signal[i-1]);
    }
    return tv;
  }
  std::vector<double> diff(signal.size() - 1);
  for (size_t i = 0; i < diff.size(); ++i) {
    diff[i] = signal[i+1] - signal[i];
  }
  return total_variation_derivative(diff, order - 1);
}

Mat4 delta_pose_error(const SE3& est, const SE3& gt) {
  return (est.inverse() * gt).toMatrix();
}

SE3 umeyama_se3_alignment(
  const std::vector<Vec3>& source,
  const std::vector<Vec3>& target) {
  if (source.size() != target.size() || source.size() < 3) {
    return SE3::Identity();
  }
  size_t n = source.size();
  Vec3 mu_src = Vec3::Zero(), mu_tgt = Vec3::Zero();
  for (size_t i = 0; i < n; ++i) {
    mu_src += source[i];
    mu_tgt += target[i];
  }
  mu_src /= n;
  mu_tgt /= n;

  Mat3 H = Mat3::Zero();
  for (size_t i = 0; i < n; ++i) {
    H += (target[i] - mu_tgt) * (source[i] - mu_src).transpose();
  }

  Eigen::JacobiSVD<Mat3> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Mat3 R = svd.matrixV() * svd.matrixU().transpose();
  if (R.determinant() < 0) {
    Mat3 V = svd.matrixV();
    V.col(2) *= -1;
    R = V * svd.matrixU().transpose();
  }
  Vec3 t = mu_tgt - R * mu_src;
  return SE3(Quat(R), t);
}

void align_trajectory(
  std::vector<SE3>& est_poses,
  const std::vector<SE3>& gt_poses) {
  size_t n = std::min(est_poses.size(), gt_poses.size());
  if (n < 3) return;
  std::vector<Vec3> src_pts(n), tgt_pts(n);
  for (size_t i = 0; i < n; ++i) {
    src_pts[i] = est_poses[i].trans;
    tgt_pts[i] = gt_poses[i].trans;
  }
  SE3 T = umeyama_se3_alignment(src_pts, tgt_pts);
  for (size_t i = 0; i < est_poses.size(); ++i) {
    est_poses[i] = T * est_poses[i];
  }
}

double compute_ate_rmse(
  const std::vector<SE3>& est_poses_aligned,
  const std::vector<SE3>& gt_poses) {
  size_t n = std::min(est_poses_aligned.size(), gt_poses.size());
  if (n == 0) return -1.0;
  double sum_sq = 0.0;
  for (size_t i = 0; i < n; ++i) {
    double err = (est_poses_aligned[i].trans - gt_poses[i].trans).norm();
    sum_sq += err * err;
  }
  return std::sqrt(sum_sq / n);
}

double compute_ate_mean(
  const std::vector<SE3>& est_poses_aligned,
  const std::vector<SE3>& gt_poses) {
  size_t n = std::min(est_poses_aligned.size(), gt_poses.size());
  if (n == 0) return -1.0;
  double sum = 0.0;
  for (size_t i = 0; i < n; ++i) {
    sum += (est_poses_aligned[i].trans - gt_poses[i].trans).norm();
  }
  return sum / n;
}

double compute_ate_median(
  const std::vector<SE3>& est_poses_aligned,
  const std::vector<SE3>& gt_poses) {
  size_t n = std::min(est_poses_aligned.size(), gt_poses.size());
  if (n == 0) return -1.0;
  std::vector<double> errors(n);
  for (size_t i = 0; i < n; ++i) {
    errors[i] = (est_poses_aligned[i].trans - gt_poses[i].trans).norm();
  }
  std::nth_element(errors.begin(), errors.begin() + n / 2, errors.end());
  return errors[n / 2];
}

double compute_ate_std(
  const std::vector<SE3>& est_poses_aligned,
  const std::vector<SE3>& gt_poses) {
  size_t n = std::min(est_poses_aligned.size(), gt_poses.size());
  if (n == 0) return -1.0;
  double mean = compute_ate_mean(est_poses_aligned, gt_poses);
  double sum_sq = 0.0;
  for (size_t i = 0; i < n; ++i) {
    double err = (est_poses_aligned[i].trans - gt_poses[i].trans).norm();
    sum_sq += (err - mean) * (err - mean);
  }
  return std::sqrt(sum_sq / n);
}

RPEMetrics compute_rpe_full(
  const std::vector<SE3>& est_poses,
  const std::vector<SE3>& gt_poses,
  const std::vector<double>& timestamps) {
  RPEMetrics m;
  m.rmse_trans = -1.0; m.mean_trans = -1.0;
  m.median_trans = -1.0; m.std_trans = -1.0;
  m.rmse_rot_deg = -1.0; m.mean_rot_deg = -1.0;

  size_t n = std::min({est_poses.size(), gt_poses.size()});
  if (n < 2) return m;

  std::vector<double> trans_errors;
  std::vector<double> rot_errors;
  trans_errors.reserve(n - 1);
  rot_errors.reserve(n - 1);

  for (size_t i = 1; i < n; ++i) {
    SE3 delta_est = est_poses[i-1].inverse() * est_poses[i];
    SE3 delta_gt = gt_poses[i-1].inverse() * gt_poses[i];
    SE3 error = delta_est.inverse() * delta_gt;
    trans_errors.push_back(error.trans.norm());
    AngleAxis aa(error.rot);
    rot_errors.push_back(aa.angle() * 180.0 / M_PI);
  }

  double sum_t = 0.0, sum_sq_t = 0.0;
  double sum_r = 0.0, sum_sq_r = 0.0;
  for (size_t i = 0; i < trans_errors.size(); ++i) {
    sum_t += trans_errors[i];
    sum_sq_t += trans_errors[i] * trans_errors[i];
    sum_r += rot_errors[i];
    sum_sq_r += rot_errors[i] * rot_errors[i];
  }
  size_t k = trans_errors.size();
  m.rmse_trans = std::sqrt(sum_sq_t / k);
  m.mean_trans = sum_t / k;
  m.std_trans = std::sqrt(sum_sq_t / k - m.mean_trans * m.mean_trans);

  m.rmse_rot_deg = std::sqrt(sum_sq_r / k);
  m.mean_rot_deg = sum_r / k;

  std::vector<double> sorted_t = trans_errors;
  std::nth_element(sorted_t.begin(), sorted_t.begin() + k / 2, sorted_t.end());
  m.median_trans = sorted_t[k / 2];

  return m;
}

double compute_rpe_rmse(const std::vector<SE3>& est_poses,
                         const std::vector<SE3>& gt_poses,
                         const std::vector<double>& timestamps) {
  if (est_poses.size() < 2 || gt_poses.size() < 2) return -1.0;
  double sum_sq = 0.0;
  int count = 0;
  for (size_t i = 1; i < std::min(est_poses.size(), gt_poses.size()); ++i) {
    SE3 delta_est = est_poses[i-1].inverse() * est_poses[i];
    SE3 delta_gt = gt_poses[i-1].inverse() * gt_poses[i];
    SE3 error = delta_est.inverse() * delta_gt;
    double trans_error = error.trans.norm();
    sum_sq += trans_error * trans_error;
    count++;
  }
  return count > 0 ? std::sqrt(sum_sq / count) : -1.0;
}

Eigen::MatrixXd compute_laplacian(const Eigen::MatrixXd& W) {
  int n = W.rows();
  Eigen::VectorXd D = W.rowwise().sum();
  Eigen::MatrixXd L = -W;
  for (int i = 0; i < n; ++i) {
    L(i, i) += D(i);
  }
  return L;
}

Eigen::MatrixXd normalize_laplacian(const Eigen::MatrixXd& W) {
  int n = W.rows();
  Eigen::VectorXd D = W.rowwise().sum();
  Eigen::MatrixXd D_inv_sqrt = Eigen::VectorXd::Ones(n).asDiagonal();
  for (int i = 0; i < n; ++i) {
    if (D(i) > 1e-10) {
      D_inv_sqrt(i, i) = 1.0 / std::sqrt(D(i));
    }
  }
  return D_inv_sqrt * compute_laplacian(W) * D_inv_sqrt;
}

std::pair<Eigen::VectorXd, Eigen::MatrixXd>
partial_eigen_decomposition(const Eigen::MatrixXd& L, int num_eigenvalues) {
  int n = L.rows();
  int k = std::min(num_eigenvalues, n);
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigs(L);
  Eigen::VectorXd values = eigs.eigenvalues();
  Eigen::MatrixXd vectors = eigs.eigenvectors();
  return {values.head(k), vectors.leftCols(k)};
}

} // namespace math
} // namespace st_slam
