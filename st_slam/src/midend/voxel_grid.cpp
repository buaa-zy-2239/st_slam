#include "st_slam/midend/voxel_grid.h"
#include <cmath>
#include <algorithm>

namespace st_slam {

VoxelGrid::VoxelGrid(double voxel_size, int resolution)
  : voxel_size_(voxel_size), resolution_(resolution) {}

VoxelKey VoxelGrid::GetVoxelKey(const Vec3& point) const {
  return {
    static_cast<int>(std::floor(point(0) / voxel_size_)),
    static_cast<int>(std::floor(point(1) / voxel_size_)),
    static_cast<int>(std::floor(point(2) / voxel_size_))
  };
}

Vec3 VoxelGrid::GetVoxelCenter(const VoxelKey& key) const {
  return Vec3(
    (key.x + 0.5) * voxel_size_,
    (key.y + 0.5) * voxel_size_,
    (key.z + 0.5) * voxel_size_
  );
}

int VoxelGrid::AssignToVoxel(const Vec3& point) {
  VoxelKey key = GetVoxelKey(point);
  auto it = voxel_to_id_.find(key);
  if (it != voxel_to_id_.end()) {
    return it->second;
  }
  int id = next_voxel_id_++;
  voxel_to_id_[key] = id;

  VoxelControlNode node;
  node.center = GetVoxelCenter(key);
  node.position = node.center;
  node.rotation = Quat::Identity();
  node.grid_idx[0] = key.x;
  node.grid_idx[1] = key.y;
  node.grid_idx[2] = key.z;
  node.active = true;
  voxel_nodes_.push_back(node);

  return id;
}

std::vector<VoxelKey> VoxelGrid::GetNeighborKeys(const VoxelKey& key, int radius) const {
  std::vector<VoxelKey> neighbors;
  for (int dx = -radius; dx <= radius; ++dx) {
    for (int dy = -radius; dy <= radius; ++dy) {
      for (int dz = -radius; dz <= radius; ++dz) {
        if (dx == 0 && dy == 0 && dz == 0) continue;
        neighbors.push_back({key.x + dx, key.y + dy, key.z + dz});
      }
    }
  }
  return neighbors;
}

bool VoxelGrid::IsOnBoundary(const Vec3& point, double threshold) const {
  VoxelKey key = GetVoxelKey(point);
  Vec3 center = GetVoxelCenter(key);
  Vec3 offset = point - center;
  double half_size = voxel_size_ / 2.0;
  return (std::abs(std::abs(offset(0)) - half_size) < threshold ||
          std::abs(std::abs(offset(1)) - half_size) < threshold ||
          std::abs(std::abs(offset(2)) - half_size) < threshold);
}

std::vector<VoxelControlNode> VoxelGrid::GetActiveNodes() const {
  std::vector<VoxelControlNode> active;
  for (const auto& node : voxel_nodes_) {
    if (node.active) {
      active.push_back(node);
    }
  }
  return active;
}

void VoxelGrid::Clear() {
  voxel_to_id_.clear();
  voxel_nodes_.clear();
  next_voxel_id_ = 0;
}

int VoxelGrid::GetVoxelId(const VoxelKey& key) const {
  auto it = voxel_to_id_.find(key);
  if (it != voxel_to_id_.end()) {
    return it->second;
  }
  return -1;
}

} // namespace st_slam
