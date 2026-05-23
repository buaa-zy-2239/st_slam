#include "st_slam/midend/vb_ks.h"
#include "st_slam/core/math_utils.h"
#include "st_slam/midend/tdq_bs.h"
#include <iostream>

namespace st_slam {

VBKS::VBKS(double voxel_size, int resolution, double boundary_delta)
  : voxel_size_(voxel_size), resolution_(resolution),
    boundary_delta_(boundary_delta) {
  voxel_grid_ = std::make_unique<VoxelGrid>(voxel_size, resolution);
}

void VBKS::SetSurfelVoxelAssignments(std::vector<Surfel>& surfels) {
  for (auto& surfel : surfels) {
    surfel.voxel_id = voxel_grid_->AssignToVoxel(surfel.position);
  }
}

void VBKS::UpdateNodePoses(const std::vector<VoxelControlNode>& nodes) {
  (void)nodes;
}

Vec3 VBKS::DeformPoint(const Vec3& point) const {
  VoxelKey key = voxel_grid_->GetVoxelKey(point);
  auto nodes = voxel_grid_->GetActiveNodes();

  auto it = std::find_if(nodes.begin(), nodes.end(),
    [&key](const VoxelControlNode& n) {
      return n.grid_idx[0] == key.x && n.grid_idx[1] == key.y && n.grid_idx[2] == key.z;
    });

  if (it == nodes.end()) return point;

  Vec3 diff = point - it->center;
  Vec3 deformed = it->rotation * diff + it->position;

  return deformed;
}

void VBKS::OptimizeDeformation(
  const std::vector<Surfel>& surfels,
  const std::vector<Vec3>& target_positions,
  const std::vector<Vec3>& target_normals) {
  (void)surfels;
  (void)target_positions;
  (void)target_normals;
}

} // namespace st_slam
