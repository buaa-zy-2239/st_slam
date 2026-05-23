#include "st_slam/frontend/ncio.h"
#include <cmath>
#include <algorithm>

namespace st_slam {

NCIO::NCIO(double kinetic_noise, double shock_gain,
            double covariance_max_det, double covariance_threshold)
  : kinetic_noise_(kinetic_noise), shock_gain_(shock_gain),
    covariance_max_det_(covariance_max_det),
    covariance_threshold_(covariance_threshold),
    Sigma_deg_(Mat6::Zero()),
    Q_kin_(Mat6::Identity() * kinetic_noise),
    elapsed_time_(0.0),
    has_prev_(false) {}

void NCIO::Reset() {
  Sigma_deg_.setZero();
  elapsed_time_ = 0.0;
  buffer_.clear();
  has_prev_ = false;
}

void NCIO::Update(const Vec3& accel, const Vec3& gyro, double dt) {
  IntegrateShockEnergy(accel, gyro, dt);

  Sigma_deg_ = Sigma_deg_ + (Q_kin_) * dt;

  double det = GetCovarianceDeterminant();
  if (det > covariance_max_det_) {
    double scale = std::pow(covariance_max_det_ / det, 1.0 / 6.0);
    Sigma_deg_ *= scale;
  }

  elapsed_time_ += dt;
}

void NCIO::IntegrateShockEnergy(const Vec3& accel, const Vec3& gyro, double dt) {
  if (!has_prev_) {
    prev_accel_ = accel;
    prev_gyro_ = gyro;
    has_prev_ = true;
    return;
  }

  Vec3 accel_jerk = (accel - prev_accel_) / std::max(dt, 1e-6);
  Vec3 gyro_jolt = (gyro - prev_gyro_) / std::max(dt, 1e-6);

  double shock_energy = accel_jerk.squaredNorm() + gyro_jolt.squaredNorm();

  Mat6 Q_shock = Mat6::Identity() * (shock_gain_ * shock_energy * dt);

  Sigma_deg_ = Sigma_deg_ + Q_shock;

  prev_accel_ = accel;
  prev_gyro_ = gyro;
}

bool NCIO::CheckDegenerateCTrigger() const {
  return GetCovarianceDeterminant() > covariance_threshold_;
}

double NCIO::GetCovarianceDeterminant() const {
  return Sigma_deg_.determinant();
}

void NCIO::RecordInertialData(double timestamp, const Vec3& accel, const Vec3& gyro) {
  buffer_.push_back({timestamp, accel, gyro});
  if (buffer_.size() > 1000) {
    buffer_.pop_front();
  }
}

} // namespace st_slam
