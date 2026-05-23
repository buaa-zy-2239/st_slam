#ifndef ST_SLAM_TDQ_BS_H
#define ST_SLAM_TDQ_BS_H

#include "st_slam/core/types.h"
#include "st_slam/core/math_utils.h"
#include "st_slam/midend/voxel_grid.h"
#include <vector>

namespace st_slam {

struct DualQuaternion {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Quat real;
  Quat dual;

  DualQuaternion() : real(Quat::Identity()), dual(0, 0, 0, 0) {}

  DualQuaternion(const Quat& r, const Quat& d) : real(r), dual(d) {}

  explicit DualQuaternion(const SE3& pose) {
    real = pose.rot;
    Vec3 t = pose.trans;
    dual = Quat(0, t(0) * 0.5, t(1) * 0.5, t(2) * 0.5) * real;
  }

  DualQuaternion operator+(const DualQuaternion& other) const {
    return DualQuaternion(
      Quat(real.coeffs() + other.real.coeffs()),
      Quat(dual.coeffs() + other.dual.coeffs()));
  }

  DualQuaternion operator*(double w) const {
    return DualQuaternion(
      Quat(real.coeffs() * w),
      Quat(dual.coeffs() * w));
  }
};

inline DualQuaternion dq_conjugate(const DualQuaternion& dq) {
  return DualQuaternion(dq.real.conjugate(), dq.dual.conjugate());
}

inline DualQuaternion dq_normalize(const DualQuaternion& dq) {
  double n = dq.real.norm();
  if (n < 1e-10) return dq;
  return DualQuaternion(Quat(dq.real.coeffs() / n),
                         Quat(dq.dual.coeffs() / n));
}

inline SE3 dq_to_se3(const DualQuaternion& dq) {
  DualQuaternion nq = dq_normalize(dq);
  Quat r = nq.real;
  Quat t = nq.dual * Quat(2, 0, 0, 0) * nq.real.conjugate();
  return SE3(r, Vec3(t.x(), t.y(), t.z()));
}

inline DualQuaternion dq_blend(const std::vector<DualQuaternion>& dqs,
                                const std::vector<double>& weights) {
  DualQuaternion result;
  double total_weight = 0.0;

  for (size_t i = 0; i < dqs.size(); ++i) {
    DualQuaternion wdq = dqs[i] * weights[i];
    if (result.real.coeffs().dot(wdq.real.coeffs()) < 0) {
      wdq = dqs[i] * (-weights[i]);
    }
    result = result + wdq;
    total_weight += weights[i];
  }

  if (total_weight > 1e-10) {
    result = result * (1.0 / total_weight);
  }

  return dq_normalize(result);
}

class TDQBS {
public:
  explicit TDQBS(double blending_threshold = 0.02);

  SE3 ComputeBlendedTransform(
    const Vec3& point,
    const std::vector<VoxelControlNode>& nodes,
    const VoxelGrid& grid) const;

  static void GetTrilinearWeights(const Vec3& point,
                                   const Vec3& voxel_center,
                                   double voxel_size,
                                   std::vector<double>& weights,
                                   std::vector<Vec3>& node_positions);

private:
  double blending_threshold_;
};

} // namespace st_slam

#endif
