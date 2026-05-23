#include "st_slam/frontend/orb_extractor.h"
#include <opencv2/calib3d.hpp>
#include <cstring>

namespace st_slam {

ORBExtractor::ORBExtractor(int n_features, float scale_factor,
                             int n_levels, int ini_threshold,
                             int min_threshold)
  : n_features_(n_features), scale_factor_(scale_factor), n_levels_(n_levels) {
  orb_ = cv::ORB::create(
    n_features, scale_factor, n_levels, 31,
    0, 2, cv::ORB::HARRIS_SCORE, 31, ini_threshold);
}

void ORBExtractor::Extract(Frame& frame) {
  orb_->detectAndCompute(frame.gray, cv::noArray(),
                         frame.keypoints, frame.descriptors);
}

void ORBExtractor::ExtractWithMask(Frame& frame, const cv::Mat& mask) {
  orb_->detectAndCompute(frame.gray, mask,
                         frame.keypoints, frame.descriptors);
}

void ORBExtractor::ComputeStereoMatches(
  const Frame& frame1, const Frame& frame2,
  std::vector<cv::DMatch>& matches) {
  if (frame1.descriptors.empty() || frame2.descriptors.empty()) return;

  cv::BFMatcher matcher(cv::NORM_HAMMING, false);
  std::vector<std::vector<cv::DMatch>> knn_matches;
  matcher.knnMatch(frame1.descriptors, frame2.descriptors, knn_matches, 2);

  const float lowe_ratio = 0.75f;
  matches.clear();
  matches.reserve(knn_matches.size());

  for (size_t i = 0; i < knn_matches.size(); ++i) {
    if (knn_matches[i].size() < 2) continue;
    if (knn_matches[i][0].distance < lowe_ratio * knn_matches[i][1].distance) {
      matches.push_back(knn_matches[i][0]);
    }
  }

  std::sort(matches.begin(), matches.end(),
    [](const cv::DMatch& a, const cv::DMatch& b) { return a.distance < b.distance; });

  if (matches.size() > 300) matches.resize(300);
}

Vec3 ORBExtractor::TriangulateKeypoint(
  const cv::KeyPoint& kp1, const cv::KeyPoint& kp2,
  const SE3& pose1, const SE3& pose2,
  double fx, double fy, double cx, double cy) {
  Mat3 K;
  K << fx, 0, cx,
       0, fy, cy,
       0,  0,  1;

  Mat4 T1 = pose1.toMatrix();
  Mat4 T2 = pose2.toMatrix();

  std::vector<cv::Point2d> pts1 = {cv::Point2d(kp1.pt.x, kp1.pt.y)};
  std::vector<cv::Point2d> pts2 = {cv::Point2d(kp2.pt.x, kp2.pt.y)};

  cv::Mat P1(3, 4, CV_64F);
  cv::Mat P2(3, 4, CV_64F);
  Eigen::Matrix<double, 3, 4, Eigen::RowMajor> P1_eig = K * T1.block<3,4>(0,0);
  Eigen::Matrix<double, 3, 4, Eigen::RowMajor> P2_eig = K * T2.block<3,4>(0,0);
  std::memcpy(P1.data, P1_eig.data(), 3 * 4 * sizeof(double));
  std::memcpy(P2.data, P2_eig.data(), 3 * 4 * sizeof(double));

  cv::Mat pts4d;
  cv::triangulatePoints(P1, P2, pts1, pts2, pts4d);

  Vec3 world(pts4d.at<double>(0) / pts4d.at<double>(3),
             pts4d.at<double>(1) / pts4d.at<double>(3),
             pts4d.at<double>(2) / pts4d.at<double>(3));
  return world;
}

} // namespace st_slam
