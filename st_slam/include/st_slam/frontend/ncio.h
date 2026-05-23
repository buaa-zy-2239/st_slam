#ifndef ST_SLAM_NCIO_H
#define ST_SLAM_NCIO_H

#include "st_slam/core/types.h"
#include "st_slam/core/math_utils.h"
#include <deque>

namespace st_slam {

class NCIO {
public:
  explicit NCIO(double kinetic_noise = 0.01,
                double shock_gain = 0.1,
                double covariance_max_det = 1e6,
                double covariance_threshold = 100.0);

  void Reset();

  void Update(const Vec3& accel, const Vec3& gyro, double dt);

  void IntegrateShockEnergy(const Vec3& accel, const Vec3& gyro, double dt);

  bool CheckDegenerateCTrigger() const;

  Mat6 GetDegenerateCovariance() const { return Sigma_deg_; }

  double GetCovarianceDeterminant() const;

  void RecordInertialData(double timestamp, const Vec3& accel, const Vec3& gyro);

private:
  double kinetic_noise_;
  double shock_gain_;
  double covariance_max_det_;
  double covariance_threshold_;

  Mat6 Sigma_deg_;
  Mat6 Q_kin_;
  double elapsed_time_;

  struct InertialSample {
    double timestamp;
    Vec3 accel;
    Vec3 gyro;
  };
  std::deque<InertialSample> buffer_;
  Vec3 prev_accel_;
  Vec3 prev_gyro_;
  bool has_prev_;
};

} // namespace st_slam

#endif
