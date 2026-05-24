#include "st_slam/perception/topological_graph.h"
#include <algorithm>
#include <iostream>

namespace st_slam {

void TopologicalGraph::AddNode(int id, const SE3& pose) {
  if (nodes_.find(id) != nodes_.end()) return;
  nodes_[id] = {id, pose};
  adj_[id] = {};
}

void TopologicalGraph::AddEdge(int from_id, int to_id, const SE3& relative_pose,
                                double weight, bool is_loop) {
  if (nodes_.find(from_id) == nodes_.end() || nodes_.find(to_id) == nodes_.end()) return;

  for (const auto& e : adj_[from_id]) {
    if (e.to_id == to_id) return;
  }

  adj_[from_id].push_back({from_id, to_id, relative_pose, weight, is_loop});
}

const TopoNode* TopologicalGraph::GetNode(int id) const {
  auto it = nodes_.find(id);
  return (it != nodes_.end()) ? &it->second : nullptr;
}

std::vector<int> TopologicalGraph::GetNeighbors(int id) const {
  std::vector<int> neighbors;
  auto it = adj_.find(id);
  if (it == adj_.end()) return neighbors;
  for (const auto& edge : it->second) {
    neighbors.push_back(edge.to_id);
  }
  return neighbors;
}

int TopologicalGraph::NumEdges() const {
  int count = 0;
  for (const auto& [_, edges] : adj_) count += (int)edges.size();
  return count;
}

int TopologicalGraph::NumLoopEdges() const {
  int count = 0;
  for (const auto& [_, edges] : adj_) {
    for (const auto& e : edges) {
      if (e.is_loop) count++;
    }
  }
  return count;
}

double TopologicalGraph::GetEdgeChi2(int from_id, int to_id) const {
  auto it = adj_.find(from_id);
  if (it == adj_.end()) return -1.0;
  for (const auto& edge : it->second) {
    if (edge.to_id == to_id) {
      const TopoNode* from_node = GetNode(from_id);
      const TopoNode* to_node = GetNode(to_id);
      if (!from_node || !to_node) return -1.0;
      SE3 traj_rel = from_node->pose.inverse() * to_node->pose;
      SE3 drift = traj_rel.inverse() * edge.relative_pose;
      AngleAxis aa(drift.rot);
      Vec3 rot_err = aa.axis() * aa.angle();
      double chi2 = 50.0 * rot_err.squaredNorm() + 500.0 * drift.trans.squaredNorm();
      return chi2;
    }
  }
  return -1.0;
}

} // namespace st_slam
