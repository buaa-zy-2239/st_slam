#ifndef ST_SLAM_VB_KS_H
#define ST_SLAM_VB_KS_H

#include "st_slam/core/types.h"
#include "st_slam/midend/voxel_grid.h"
#include <memory>
#include <vector>

namespace st_slam {

class VBKS {
public:
  explicit VBKS(double voxel_size = 0.2, int resolution = 8,
                double boundary_delta = 0.05);

  void SetSurfelVoxelAssignments(std::vector<Surfel>& surfels);

  void UpdateNodePoses(const std::vector<VoxelControlNode>& nodes);

  Vec3 DeformPoint(const Vec3& point) const;

  void OptimizeDeformation(
    const std::vector<Surfel>& surfels,
    const std::vector<Vec3>& target_positions,
    const std::vector<Vec3>& target_normals);

  VoxelGrid& GetVoxelGrid() { return *voxel_grid_; }

private:
  std::unique_ptr<VoxelGrid> voxel_grid_;
  double voxel_size_;
  int resolution_;
  double boundary_delta_;
};

} // namespace st_slam

#endif
