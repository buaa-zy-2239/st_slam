#ifndef ST_SLAM_LOOP_DETECTOR_H
#define ST_SLAM_LOOP_DETECTOR_H

#include "st_slam/core/types.h"
#include "st_slam/backend/spectral_heat_kernel.h"
#include "st_slam/backend/l_scov.h"
#include <memory>
#include <vector>
#include <deque>

namespace st_slam {

class LoopDetector {
public:
  explicit LoopDetector(int min_keyframe_gap = 50,
                         double similarity_threshold = 0.6);

  void AddKeyFrame(int frame_id, const SE3& pose,
                    const std::vector<int>& connections);

  LoopCandidate DetectLoop(int current_frame_id);

  std::vector<LoopCandidate> GetAllLoopCandidates() const { return candidates_; }

  bool VerifyLoop(LoopCandidate& candidate);

private:
  int min_keyframe_gap_;
  double similarity_threshold_;

  struct KeyFrameGraphNode {
    int frame_id;
    SE3 pose;
    std::vector<int> connections;
    Eigen::VectorXd spectral_signature;
    Eigen::VectorXd moment_vector;
  };

  std::deque<KeyFrameGraphNode> keyframe_graph_;
  std::vector<LoopCandidate> candidates_;

  std::unique_ptr<SpectralHeatKernel> spectral_;
  std::unique_ptr<LSCOV> l_scov_;

  Eigen::VectorXd ComputeSignature(const std::vector<KeyFrameGraphNode>& subgraph);
};

} // namespace st_slam

#endif
