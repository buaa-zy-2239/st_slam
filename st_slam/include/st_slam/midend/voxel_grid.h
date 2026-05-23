#ifndef ST_SLAM_VOXEL_GRID_H
#define ST_SLAM_VOXEL_GRID_H

#include "st_slam/core/types.h"
#include <unordered_map>
#include <vector>

namespace st_slam {

struct VoxelKey {
  int x, y, z;
  bool operator==(const VoxelKey& o) const {
    return x == o.x && y == o.y && z == o.z;
  }
};

struct VoxelKeyHash {
  size_t operator()(const VoxelKey& k) const {
    return ((k.x * 73856093) ^ (k.y * 19349663) ^ (k.z * 83492791));
  }
};

class VoxelGrid {
public:
  explicit VoxelGrid(double voxel_size = 0.2, int resolution = 8);

  VoxelKey GetVoxelKey(const Vec3& point) const;

  Vec3 GetVoxelCenter(const VoxelKey& key) const;

  int AssignToVoxel(const Vec3& point);

  std::vector<VoxelKey> GetNeighborKeys(const VoxelKey& key, int radius = 1) const;

  bool IsOnBoundary(const Vec3& point, double threshold) const;

  std::vector<VoxelControlNode> GetActiveNodes() const;

  void Clear();

  size_t NumActiveVoxels() const { return voxel_to_id_.size(); }
  int GetVoxelId(const VoxelKey& key) const;

private:
  double voxel_size_;
  int resolution_;
  int next_voxel_id_ = 0;
  std::unordered_map<VoxelKey, int, VoxelKeyHash> voxel_to_id_;
  std::vector<VoxelControlNode> voxel_nodes_;
};

} // namespace st_slam

#endif
