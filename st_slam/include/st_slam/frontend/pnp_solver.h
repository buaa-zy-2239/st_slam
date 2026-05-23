#ifndef ST_SLAM_PNP_SOLVER_H
#define ST_SLAM_PNP_SOLVER_H

#include "st_slam/core/types.h"
#include <vector>
#include <opencv2/core.hpp>

namespace st_slam {

struct PnPResult {
  SE3 pose;
  std::vector<int> inlier_indices;
  std::vector<cv::DMatch> inlier_matches;
  double inlier_ratio;
  double reprojection_error;
  bool success;
};

struct Match3D2D {
  Vec3 world_pt;
  cv::Point2f img_pt;
  int query_idx;
  int train_idx;
};

class PnPSolver {
public:
  explicit PnPSolver(double fx, double fy, double cx, double cy,
                     double reproj_threshold = 5.0,
                     int min_inliers = 10,
                     int max_iterations = 200);

  PnPResult EstimatePose(
    const std::vector<cv::DMatch>& matches,
    const Frame& ref_frame,
    const Frame& cur_frame,
    const SE3& initial_guess) const;

  PnPResult EstimatePoseFromLandmarks(
    const std::vector<Match3D2D>& correspondences,
    const SE3& initial_guess) const;

  bool RefinePoseWithInliers(
    SE3& pose,
    const std::vector<Match3D2D>& inlier_correspondences) const;

  static bool IsGoodPose(const PnPResult& result,
                          double min_inlier_ratio = 0.3);

private:
  double fx_, fy_, cx_, cy_;
  double reproj_threshold_;
  int min_inliers_;
  int max_iterations_;
  cv::Mat camera_matrix_;

  bool CheckChirality(const SE3& pose, const Vec3& world_pt,
                       const cv::Point2f& img_pt) const;
};

} // namespace st_slam

#endif
