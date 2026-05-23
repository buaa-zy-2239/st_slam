#include "st_slam/backend/loop_detector.h"
#include "st_slam/backend/spectral_heat_kernel.h"
#include "st_slam/backend/l_scov.h"
#include "st_slam/core/math_utils.h"
#include <algorithm>
#include <cmath>
#include <iostream>

namespace st_slam {

LoopDetector::LoopDetector(int min_keyframe_gap, double similarity_threshold)
  : min_keyframe_gap_(min_keyframe_gap),
    similarity_threshold_(similarity_threshold) {
  spectral_ = std::make_unique<SpectralHeatKernel>(-5.0, 5.0, 3, 30);
  l_scov_ = std::make_unique<LSCOV>(4, 0.05, 0.2);
}

void LoopDetector::AddKeyFrame(int frame_id, const SE3& pose,
                                const std::vector<int>& connections) {
  KeyFrameGraphNode node;
  node.frame_id = frame_id;
  node.pose = pose;
  node.connections = connections;

  keyframe_graph_.push_back(node);

  if (keyframe_graph_.size() > 200) {
    keyframe_graph_.pop_front();
  }
}

Eigen::VectorXd LoopDetector::ComputeSignature(
  const std::vector<KeyFrameGraphNode>& subgraph) {
  int n = static_cast<int>(subgraph.size());
  if (n < 3) return Eigen::VectorXd::Zero(3);

  std::vector<Vec3> positions;
  for (const auto& node : subgraph) {
    positions.push_back(node.pose.trans);
  }

  SpectralHeatKernel local_spectral(-5.0, 5.0, 3, std::min(30, n));
  local_spectral.BuildGraph(positions, 2.0);
  return local_spectral.ComputeHeatKernelSignature();
}

LoopCandidate LoopDetector::DetectLoop(int current_frame_id) {
  LoopCandidate candidate;
  candidate.query_id = current_frame_id;
  candidate.match_id = -1;
  candidate.verified = false;

  if (keyframe_graph_.size() < 10) return candidate;

  int n = static_cast<int>(keyframe_graph_.size());
  std::vector<KeyFrameGraphNode> recent_nodes(
    keyframe_graph_.end() - std::min(10, n), keyframe_graph_.end());
  Eigen::VectorXd query_sig = ComputeSignature(recent_nodes);

  double best_similarity = similarity_threshold_;
  int best_match_idx = -1;

  for (int i = 0; i < n - min_keyframe_gap_; ++i) {
    int window_size = std::min(10, static_cast<int>(keyframe_graph_.size()) - i);
    std::vector<KeyFrameGraphNode> cand_nodes(
      keyframe_graph_.begin() + i,
      keyframe_graph_.begin() + i + window_size);

    Eigen::VectorXd cand_sig = ComputeSignature(cand_nodes);

    double sim = 1.0 - spectral_->CompareSignatures(query_sig, cand_sig);

    if (sim > best_similarity) {
      best_similarity = sim;
      best_match_idx = keyframe_graph_[i].frame_id;
    }
  }

  if (best_match_idx >= 0) {
    candidate.match_id = best_match_idx;
    candidate.similarity_score = best_similarity;
    candidates_.push_back(candidate);
  }

  return candidate;
}

bool LoopDetector::VerifyLoop(LoopCandidate& candidate) {
  if (candidate.match_id < 0) return false;

  int n = static_cast<int>(keyframe_graph_.size());

  auto query_it = std::find_if(keyframe_graph_.begin(), keyframe_graph_.end(),
    [&](const KeyFrameGraphNode& node) {
      return node.frame_id == candidate.query_id;
    });

  auto match_it = std::find_if(keyframe_graph_.begin(), keyframe_graph_.end(),
    [&](const KeyFrameGraphNode& node) {
      return node.frame_id == candidate.match_id;
    });

  if (query_it == keyframe_graph_.end() || match_it == keyframe_graph_.end()) {
    return false;
  }

  int query_end = static_cast<int>(query_it - keyframe_graph_.begin());
  int match_end = static_cast<int>(match_it - keyframe_graph_.begin());

  int window = 10;
  int q_start = std::max(0, query_end - window);
  int m_start = std::max(0, match_end - window);

  std::vector<Vec3> q_pos, m_pos;
  for (int i = q_start; i <= query_end; ++i) {
    q_pos.push_back(keyframe_graph_[i].pose.trans);
  }
  for (int i = m_start; i <= match_end; ++i) {
    m_pos.push_back(keyframe_graph_[i].pose.trans);
  }

  SpectralHeatKernel q_spec(-5.0, 5.0, 3, std::min(30, static_cast<int>(q_pos.size())));
  q_spec.BuildGraph(q_pos, 2.0);
  q_spec.ComputeHeatKernelSignature();

  SpectralHeatKernel m_spec(-5.0, 5.0, 3, std::min(30, static_cast<int>(m_pos.size())));
  m_spec.BuildGraph(m_pos, 2.0);
  m_spec.ComputeHeatKernelSignature();

  double spectral_dist, hamming_dist;
  bool verified = l_scov_->VerifyLoopCandidate(q_spec, m_spec,
                                                spectral_dist, hamming_dist);

  candidate.spectral_distance = spectral_dist;
  candidate.verified = verified;

  return verified;
}

} // namespace st_slam
