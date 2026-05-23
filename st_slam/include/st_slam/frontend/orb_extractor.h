#ifndef ST_SLAM_ORB_EXTRACTOR_H
#define ST_SLAM_ORB_EXTRACTOR_H

#include "st_slam/core/types.h"
#include <opencv2/features2d.hpp>
#include <memory>

namespace st_slam {

class ORBExtractor {
public:
  explicit ORBExtractor(int n_features = 1500,
                         float scale_factor = 1.2f,
                         int n_levels = 8,
                         int ini_threshold = 20,
                         int min_threshold = 7);

  void Extract(Frame& frame);

  void ExtractWithMask(Frame& frame, const cv::Mat& mask);

  void SetMaxFeatures(int n) {
    auto* orb = dynamic_cast<cv::ORB*>(orb_.get());
    if (orb) orb->setMaxFeatures(n);
  }

  int GetNumLevels() const { return n_levels_; }
  float GetScaleFactor() const { return scale_factor_; }

  static void ComputeStereoMatches(
    const Frame& frame1, const Frame& frame2,
    std::vector<cv::DMatch>& matches);

  static Vec3 TriangulateKeypoint(
    const cv::KeyPoint& kp1, const cv::KeyPoint& kp2,
    const SE3& pose1, const SE3& pose2,
    double fx, double fy, double cx, double cy);

private:
  cv::Ptr<cv::Feature2D> orb_;
  int n_features_;
  float scale_factor_;
  int n_levels_;
};

} // namespace st_slam

#endif
