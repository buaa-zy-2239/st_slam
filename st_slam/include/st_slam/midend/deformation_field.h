#ifndef ST_SLAM_DEFORMATION_FIELD_H
#define ST_SLAM_DEFORMATION_FIELD_H

#include "st_slam/core/types.h"
#include "st_slam/midend/voxel_grid.h"
#include "st_slam/midend/vb_ks.h"
#include "st_slam/midend/tdq_bs.h"
#include <memory>

namespace st_slam {

class DeformationField {
public:
  explicit DeformationField(double voxel_size = 0.2,
                             int resolution = 8,
                             double regularization_weight = 0.1,
                             double epsilon = 1e-8);

  void Initialize(const std::vector<Surfel>& surfels);

  Vec3 DeformPoint(const Vec3& point) const;

  Vec3 DeformNormal(const Vec3& point, const Vec3& normal) const;

  void AddSurfels(const std::vector<Surfel>& surfels);

  const VBKS& GetVBKS() const { return *vb_ks_; }
  const VoxelGrid& GetVoxelGrid() const { return *voxel_grid_; }

private:
  std::shared_ptr<VoxelGrid> voxel_grid_;
  std::unique_ptr<VBKS> vb_ks_;
  std::unique_ptr<TDQBS> tdq_bs_;
  double regularization_weight_;
  double epsilon_;

  static Mat3 compute_adjugate(const Mat3& J);

  static Vec3 adjugate_normal_transform(const Mat3& J, const Vec3& N);
};

} // namespace st_slam

#endif
