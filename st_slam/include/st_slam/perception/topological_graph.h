#ifndef ST_SLAM_TOPOLOGICAL_GRAPH_H
#define ST_SLAM_TOPOLOGICAL_GRAPH_H

#include "st_slam/core/types.h"
#include <vector>
#include <unordered_map>

namespace st_slam {

struct TopoNode {
  int id;
  SE3 pose;
};

struct TopoEdge {
  int from_id;
  int to_id;
  SE3 relative_pose;
  double weight;
  bool is_loop;
};

class TopologicalGraph {
public:
  TopologicalGraph() = default;
  ~TopologicalGraph() = default;

  void AddNode(int id, const SE3& pose);
  void AddEdge(int from_id, int to_id, const SE3& relative_pose,
               double weight, bool is_loop = false);

  const TopoNode* GetNode(int id) const;
  std::vector<int> GetNeighbors(int id) const;
  const std::vector<TopoEdge>& GetEdges(int id) const { return adj_.at(id); }

  int NumNodes() const { return (int)nodes_.size(); }
  int NumEdges() const;
  int NumLoopEdges() const;

  bool HasNode(int id) const { return nodes_.find(id) != nodes_.end(); }

  double GetEdgeChi2(int from_id, int to_id) const;

  const std::unordered_map<int, TopoNode>& GetAllNodes() const { return nodes_; }

private:
  std::unordered_map<int, TopoNode> nodes_;
  std::unordered_map<int, std::vector<TopoEdge>> adj_;
};

} // namespace st_slam

#endif
