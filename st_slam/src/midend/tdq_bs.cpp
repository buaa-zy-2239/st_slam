#include "st_slam/midend/tdq_bs.h"
#include <cmath>
#include <algorithm>

namespace st_slam {

TDQBS::TDQBS(double blending_threshold)
  : blending_threshold_(blending_threshold) {}

SE3 TDQBS::ComputeBlendedTransform(
  const Vec3& point,
  const std::vector<VoxelControlNode>& nodes,
  const VoxelGrid& grid) const {
  (void)grid;
  (void)blending_threshold_;

  std::vector<DualQuaternion> dqs;
  std::vector<double> weights;

  for (const auto& node : nodes) {
    Vec3 diff = point - node.center;
    double dist = diff.norm();
    if (dist > 1e-6) {
      double w = 1.0 / (dist * dist);
      weights.push_back(w);
    } else {
      weights.push_back(1e6);
    }
    dqs.emplace_back(SE3(node.rotation, node.position));
  }

  if (dqs.empty()) return SE3::Identity();

  DualQuaternion blended = dq_blend(dqs, weights);
  return dq_to_se3(blended);
}

void TDQBS::GetTrilinearWeights(const Vec3& point,
                                 const Vec3& voxel_center,
                                 double voxel_size,
                                 std::vector<double>& weights,
                                 std::vector<Vec3>& node_positions) {
  weights.clear();
  node_positions.clear();

  double half = voxel_size / 2.0;
  Vec3 offset = point - voxel_center;
  double u = offset(0) / half;
  double v = offset(1) / half;
  double w = offset(2) / half;

  u = std::clamp(u, -1.0, 1.0);
  v = std::clamp(v, -1.0, 1.0);
  w = std::clamp(w, -1.0, 1.0);

  double corners[2] = {-1, 1};
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      for (int k = 0; k < 2; ++k) {
        double weight = (1.0 - corners[i] * u) * (1.0 - corners[j] * v) *
                         (1.0 - corners[k] * w) / 8.0;
        weights.push_back(weight);
        node_positions.push_back(Vec3(
          voxel_center(0) + corners[i] * half,
          voxel_center(1) + corners[j] * half,
          voxel_center(2) + corners[k] * half
        ));
      }
    }
  }
}

} // namespace st_slam
