#include "st_slam/backend/pose_graph.h"
#include "st_slam/core/math_utils.h"
#include <ceres/rotation.h>
#include <iostream>
#include <algorithm>
#include <unordered_set>

namespace st_slam {

PoseGraph::PoseGraph(double rot_info_weight, double trans_info_weight)
  : rot_info_weight_(rot_info_weight),
    trans_info_weight_(trans_info_weight) {}

void PoseGraph::Clear() {
  edges_.clear();
}

void PoseGraph::AddEdge(int from_id, int to_id, const SE3& relative_pose,
                         const Mat6& information) {
  PoseGraphEdge edge;
  edge.from_id = from_id;
  edge.to_id = to_id;
  edge.relative_pose = relative_pose;
  edge.information = information;
  edges_.push_back(edge);
}

Mat6 PoseGraph::DefaultInformation(double rot_w, double trans_w) {
  Mat6 info = Mat6::Zero();
  info.block<3,3>(0,0) = Mat3::Identity() * rot_w;
  info.block<3,3>(3,3) = Mat3::Identity() * trans_w;
  return info;
}

void PoseGraph::BuildFromKeyframes(const std::unordered_map<int, KeyFrame>& keyframes) {
  // DON'T clear edges_! We want to preserve loop edges!
  // Only rebuild odometry edges between consecutive keyframes
  // while keeping any loop edges that were added by LoopCloser
  
  // First, remove ONLY odometry edges (consecutive keyframes)
  int initial_size = edges_.size();
  auto it = edges_.begin();
  while (it != edges_.end()) {
    int diff = std::abs(it->to_id - it->from_id);
    if (diff == 1) {
      it = edges_.erase(it);
    } else {
      ++it;
    }
  }

  std::vector<int> kf_ids;
  for (const auto& [id, _] : keyframes) kf_ids.push_back(id);
  std::sort(kf_ids.begin(), kf_ids.end());

  for (size_t i = 1; i < kf_ids.size(); ++i) {
    int from_id = kf_ids[i-1];
    int to_id = kf_ids[i];

    auto from_it = keyframes.find(from_id);
    auto to_it = keyframes.find(to_id);
    if (from_it == keyframes.end() || to_it == keyframes.end()) continue;

    SE3 rel = from_it->second.pose.inverse() * to_it->second.pose;
    Mat6 info = DefaultInformation(
      rot_info_weight_ * 50.0, trans_info_weight_ * 50.0);
    AddEdge(from_id, to_id, rel, info);
  }
}

bool PoseGraph::Optimize(std::unordered_map<int, KeyFrame>& keyframes) {
  if (edges_.size() < 3) {
    std::cout << "[PGO] Skipped: only " << edges_.size() << " edges (< 3)\n";
    return false;
  }

  std::vector<int> kf_ids;
  for (const auto& [id, _] : keyframes) kf_ids.push_back(id);
  std::sort(kf_ids.begin(), kf_ids.end());
  if (kf_ids.empty()) return false;

  std::unordered_map<int, std::array<double, 7>> params;
  for (int id : kf_ids) {
    auto& kf = keyframes[id];
    auto& p = params[id];
    p[0] = kf.pose.rot.w();
    p[1] = kf.pose.rot.x();
    p[2] = kf.pose.rot.y();
    p[3] = kf.pose.rot.z();
    p[4] = kf.pose.trans(0);
    p[5] = kf.pose.trans(1);
    p[6] = kf.pose.trans(2);
  }

  ceres::Problem problem;

  for (int id : kf_ids) {
    problem.AddParameterBlock(params[id].data(), 7);
  }

  int added = 0;
  for (const auto& edge : edges_) {
    if (params.find(edge.from_id) == params.end() ||
        params.find(edge.to_id) == params.end()) continue;

    Mat6 sqrt_info = edge.information.llt().matrixL();
    ceres::CostFunction* cost =
      PoseGraph3DError::Create(edge.relative_pose, sqrt_info);

    int diff = std::abs(edge.to_id - edge.from_id);
    ceres::LossFunction* loss = (diff != 1) ?
      new ceres::CauchyLoss(0.5) : nullptr;

    problem.AddResidualBlock(cost, loss,
                              params[edge.from_id].data(),
                              params[edge.to_id].data());
    added++;
  }

  if (added < 2) return false;

  // Active window: recent KFs + loop-adjacent KFs stay unfixed
  int window_size = 20;
  int max_id = kf_ids.back();
  int min_active_id = std::max(0, max_id - window_size);

  std::unordered_set<int> active_kfs;
  for (int id : kf_ids) {
    if (id >= min_active_id) active_kfs.insert(id);
  }
  for (const auto& edge : edges_) {
    if (std::abs(edge.to_id - edge.from_id) != 1) {
      for (int d = -2; d <= 2; ++d) {
        active_kfs.insert(edge.from_id + d);
        active_kfs.insert(edge.to_id + d);
      }
    }
  }

  int fixed_count = 0;
  for (int id : kf_ids) {
    if (active_kfs.find(id) == active_kfs.end() && id != kf_ids[0]) {
      problem.SetParameterBlockConstant(params[id].data());
      fixed_count++;
    }
  }

  ceres::Solver::Options options;
  options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
  options.max_num_iterations = 5;
  options.function_tolerance = 1e-6;
  options.parameter_tolerance = 1e-8;
  options.minimizer_progress_to_stdout = false;

  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  for (int id : kf_ids) {
    auto& p = params[id];
    Quat q(p[0], p[1], p[2], p[3]);
    q.normalize();
    keyframes[id].pose = SE3(q, Vec3(p[4], p[5], p[6]));
  }

  return summary.termination_type != ceres::FAILURE;
}

} // namespace st_slam
