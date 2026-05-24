#include "st_slam/backend/pose_graph.h"
#include "st_slam/core/math_utils.h"
#include <ceres/rotation.h>
#include <iostream>

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
  edges_.clear();

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
    Mat6 info = DefaultInformation(rot_info_weight_, trans_info_weight_);
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

  // DEBUG: Print address and edge count
  std::cout << "[ADDR DEBUG] Back-end PoseGraph Addr: " << this 
            << " | Total edges BEFORE: " << edges_.size() << std::endl;

  std::cout << "[PGO] Running global pose graph optimization on " << kf_ids.size() << " keyframes...\n";

  int odometry_edges = 0;
  int loop_edges = 0;
  for (size_t i = 1; i < kf_ids.size(); ++i) {
    odometry_edges++;
  }
  loop_edges = edges_.size() - odometry_edges;
  std::cout << "[PGO] Built " << edges_.size() << " edges (" 
            << odometry_edges << " odometry, " << loop_edges << " loop)\n";

  std::cout << "[PGO] Pre-optimization poses:\n";
  for (int id : kf_ids) {
    auto& kf = keyframes[id];
    std::cout << "  KF" << id << ": pos=(" 
              << kf.pose.trans(0) << ", " << kf.pose.trans(1) << ", " << kf.pose.trans(2) << ")\n";
  }

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
    problem.AddResidualBlock(cost, nullptr,
                              params[edge.from_id].data(),
                              params[edge.to_id].data());
    added++;
  }

  if (added < 2) return false;

  problem.SetParameterBlockConstant(params[kf_ids[0]].data());
  std::cout << "[PGO] Fixed " << 1 << " keyframe(s)\n";

  ceres::Solver::Options options;
  options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
  options.max_num_iterations = 30;
  options.function_tolerance = 1e-8;
  options.parameter_tolerance = 1e-10;
  options.minimizer_progress_to_stdout = false;

  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  for (int id : kf_ids) {
    auto& p = params[id];
    Quat q(p[0], p[1], p[2], p[3]);
    q.normalize();
    keyframes[id].pose = SE3(q, Vec3(p[4], p[5], p[6]));
  }

  std::cout << "[PGO] Post-optimization poses:\n";
  for (int id : kf_ids) {
    auto& kf = keyframes[id];
    std::cout << "  KF" << id << ": pos=(" 
              << kf.pose.trans(0) << ", " << kf.pose.trans(1) << ", " << kf.pose.trans(2) << ")\n";
  }

  return summary.termination_type != ceres::FAILURE;
}

} // namespace st_slam
