#ifndef ST_SLAM_LOOP_CLOSER_H
#define ST_SLAM_LOOP_CLOSER_H

#include "st_slam/core/types.h"
#include "st_slam/frontend/local_map.h"
#include "st_slam/frontend/pnp_solver.h"
#include <vector>
#include <memory>

namespace st_slam {

class LoopCloser {
public:
  explicit LoopCloser(const std::string& vocab_path = "",
                       double fx = 525.0, double fy = 525.0,
                       double cx = 319.5, double cy = 239.5);

  ~LoopCloser();

  bool LoadVocabulary(const std::string& vocab_path);

  void BuildVocabulary(const std::vector<cv::Mat>& all_descriptors);

  int AddKeyFrame(int kf_id, const cv::Mat& descriptors);

  std::vector<LoopCandidate> DetectCandidates(int current_kf_id,
                                               float min_score = 0.05f,
                                               int max_candidates = 3);

  bool GeometricVerification(int query_kf_id, int match_kf_id,
                              const Frame& query_frame,
                              const Frame& match_frame,
                              SE3& relative_pose);

  bool IsInitialized() const { return vocab_ != nullptr; }
  bool HasDatabase() const { return db_ != nullptr; }

  void SaveVocabulary(const std::string& path) const;
  void LoadOrBuildVocabulary(const std::vector<cv::Mat>& all_descriptors,
                              const std::string& vocab_path = "");

  // Database stats
  int NumKeyFrames() const;

private:
  void* vocab_;   // DBoW3::Vocabulary* (opaque to avoid header dependency)
  void* db_;      // DBoW3::Database*
  std::vector<int> kf_id_map_;

  std::unique_ptr<PnPSolver> pnp_solver_;

  double fx_, fy_, cx_, cy_;
};

} // namespace st_slam

#endif
