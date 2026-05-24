#include "st_slam/frontend/pnp_solver.h"
#include <opencv2/calib3d.hpp>
#include <iostream>

namespace st_slam {

PnPSolver::PnPSolver(double fx, double fy, double cx, double cy,
                     double reproj_threshold, int min_inliers,
                     int max_iterations)
  : fx_(fx), fy_(fy), cx_(cx), cy_(cy),
    reproj_threshold_(reproj_threshold),
    min_inliers_(min_inliers),
    max_iterations_(max_iterations) {
  camera_matrix_ = (cv::Mat_<double>(3,3) << fx_, 0, cx_, 0, fy_, cy_, 0, 0, 1);
}

PnPResult PnPSolver::EstimatePose(
  const std::vector<cv::DMatch>& matches,
  const Frame& ref_frame,
  const Frame& cur_frame,
  const SE3& initial_guess) const {
  PnPResult result;
  result.success = false;
  result.inlier_ratio = 0;
  result.reprojection_error = -1;

  std::vector<Match3D2D> correspondences;
  correspondences.reserve(matches.size());

  for (const auto& m : matches) {
    int ref_idx = m.queryIdx;
    int cur_idx = m.trainIdx;

    if (ref_idx >= (int)ref_frame.keypoints_3d.size() ||
        cur_idx >= (int)cur_frame.keypoints.size()) continue;

    Vec3 world_pt = ref_frame.keypoints_3d[ref_idx];
    if (world_pt.norm() < 1e-6) continue;

    correspondences.push_back({
      world_pt,
      cv::Point2f(cur_frame.keypoints[cur_idx].pt.x,
                   cur_frame.keypoints[cur_idx].pt.y),
      ref_idx, cur_idx
    });
  }

  if ((int)correspondences.size() < min_inliers_) return result;

  return EstimatePoseFromLandmarks(correspondences, initial_guess);
}

bool PnPSolver::CheckChirality(const SE3& pose,
                                 const Vec3& world_pt,
                                 const cv::Point2f& img_pt) const {
  Vec3 cam_pt = pose.inverse() * world_pt;
  if (cam_pt(2) < 0.01) return false;

  double u = fx_ * cam_pt(0) / cam_pt(2) + cx_;
  double v = fy_ * cam_pt(1) / cam_pt(2) + cy_;
  double dx = u - img_pt.x;
  double dy = v - img_pt.y;
  return (dx*dx + dy*dy) < reproj_threshold_ * reproj_threshold_ * 4;
}

PnPResult PnPSolver::EstimatePoseFromLandmarks(
  const std::vector<Match3D2D>& correspondences,
  const SE3& initial_guess) const {
  PnPResult result;
  result.success = false;

  if ((int)correspondences.size() < min_inliers_) return result;

  std::vector<cv::Point3f> world_pts;
  std::vector<cv::Point2f> img_pts;
  world_pts.reserve(correspondences.size());
  img_pts.reserve(correspondences.size());

  for (const auto& m : correspondences) {
    world_pts.emplace_back(m.world_pt(0), m.world_pt(1), m.world_pt(2));
    img_pts.push_back(m.img_pt);
  }

  cv::Mat rvec, tvec, inliers_cv;

  SE3 initial_tcw = initial_guess.inverse();
  AngleAxis aa_init(initial_tcw.rot);
  rvec = (cv::Mat_<double>(3,1) << aa_init.axis()(0) * aa_init.angle(),
                                       aa_init.axis()(1) * aa_init.angle(),
                                       aa_init.axis()(2) * aa_init.angle());
  tvec = (cv::Mat_<double>(3,1) << initial_tcw.trans(0),
                                       initial_tcw.trans(1),
                                       initial_tcw.trans(2));

  bool ok = cv::solvePnPRansac(
    world_pts, img_pts, camera_matrix_, cv::Mat(),
    rvec, tvec, true, max_iterations_,
    reproj_threshold_, 0.99, inliers_cv,
    cv::SOLVEPNP_ITERATIVE);

  if (!ok || inliers_cv.rows < min_inliers_) {
    rvec = cv::Mat::zeros(3, 1, CV_64F);
    tvec = cv::Mat::zeros(3, 1, CV_64F);
    ok = cv::solvePnPRansac(
      world_pts, img_pts, camera_matrix_, cv::Mat(),
      rvec, tvec, false, max_iterations_ * 2,
      reproj_threshold_ * 1.5, 0.99, inliers_cv,
      cv::SOLVEPNP_P3P);
  }

  if (!ok || inliers_cv.rows < min_inliers_) {
    return result;
  }

  std::vector<int> inlier_vec;
  inlier_vec.reserve(inliers_cv.rows);
  for (int i = 0; i < inliers_cv.rows; ++i) {
    int idx = inliers_cv.at<int>(i);
    inlier_vec.push_back(idx);
  }

  AngleAxis aa_solved(
    std::sqrt(rvec.at<double>(0)*rvec.at<double>(0) +
              rvec.at<double>(1)*rvec.at<double>(1) +
              rvec.at<double>(2)*rvec.at<double>(2)),
    Vec3(rvec.at<double>(0), rvec.at<double>(1), rvec.at<double>(2)).normalized());

  SE3 raw_pose(Quat(aa_solved),
               Vec3(tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2)));
  SE3 camera_pose = raw_pose.inverse();

  std::vector<int> chirality_inliers;
  std::vector<Match3D2D> chirality_corrs;
  std::vector<cv::DMatch> chirality_matches;
  chirality_inliers.reserve(inlier_vec.size());

  for (int idx : inlier_vec) {
    if (CheckChirality(camera_pose, correspondences[idx].world_pt,
                        correspondences[idx].img_pt)) {
      chirality_inliers.push_back(idx);
      chirality_corrs.push_back(correspondences[idx]);
      chirality_matches.push_back({
        correspondences[idx].query_idx,
        correspondences[idx].train_idx, 0
      });
    }
  }

  if ((int)chirality_corrs.size() < min_inliers_) {
    return result;
  }

  result.pose = camera_pose;
  result.inlier_ratio = (double)chirality_corrs.size() / correspondences.size();
  result.inlier_indices = chirality_inliers;
  result.inlier_matches = chirality_matches;

  if (chirality_corrs.size() >= 4) {
    RefinePoseWithInliers(result.pose, chirality_corrs);
  }

  result.success = true;

  double sum_err = 0;
  int valid_count = 0;
  for (const auto& m : chirality_corrs) {
    Vec3 proj = result.pose * m.world_pt;
    if (proj(2) < 0.01) continue;
    double u = fx_ * proj(0) / proj(2) + cx_;
    double v = fy_ * proj(1) / proj(2) + cy_;
    double dx = u - m.img_pt.x;
    double dy = v - m.img_pt.y;
    sum_err += std::sqrt(dx*dx + dy*dy);
    valid_count++;
  }
  result.reprojection_error = valid_count > 0 ? sum_err / valid_count : -1;

  if ((int)chirality_corrs.size() >= 15 && result.reprojection_error < reproj_threshold_ * 0.8) {
    std::vector<cv::Point3f> refine_wp;
    std::vector<cv::Point2f> refine_ip;
    refine_wp.reserve(chirality_corrs.size());
    refine_ip.reserve(chirality_corrs.size());
    for (const auto& m : chirality_corrs) {
      Vec3 proj = result.pose * m.world_pt;
      if (proj(2) > 0.01) {
        refine_wp.emplace_back(m.world_pt(0), m.world_pt(1), m.world_pt(2));
        refine_ip.push_back(m.img_pt);
      }
    }
    if ((int)refine_wp.size() >= min_inliers_) {
      cv::Mat rvec2, tvec2;
      SE3 inv2 = result.pose.inverse();
      AngleAxis aa2(inv2.rot);
      rvec2 = (cv::Mat_<double>(3,1) << aa2.axis()(0) * aa2.angle(),
                                         aa2.axis()(1) * aa2.angle(),
                                         aa2.axis()(2) * aa2.angle());
      tvec2 = (cv::Mat_<double>(3,1) << inv2.trans(0), inv2.trans(1), inv2.trans(2));

      cv::solvePnP(refine_wp, refine_ip, camera_matrix_, cv::Mat(),
                   rvec2, tvec2, true, cv::SOLVEPNP_ITERATIVE);

      AngleAxis aa_final(
        std::sqrt(rvec2.at<double>(0)*rvec2.at<double>(0) +
                  rvec2.at<double>(1)*rvec2.at<double>(1) +
                  rvec2.at<double>(2)*rvec2.at<double>(2)),
        Vec3(rvec2.at<double>(0), rvec2.at<double>(1), rvec2.at<double>(2)).normalized());
      result.pose = SE3(Quat(aa_final),
                        Vec3(tvec2.at<double>(0), tvec2.at<double>(1), tvec2.at<double>(2))).inverse();
    }
  }

  return result;
}

bool PnPSolver::RefinePoseWithInliers(
  SE3& pose,
  const std::vector<Match3D2D>& inlier_correspondences) const {
  if ((int)inlier_correspondences.size() < 4) return false;

  std::vector<cv::Point3f> world_pts;
  std::vector<cv::Point2f> img_pts;
  world_pts.reserve(inlier_correspondences.size());
  img_pts.reserve(inlier_correspondences.size());

  for (const auto& m : inlier_correspondences) {
    Vec3 proj = pose * m.world_pt;
    if (proj(2) > 0.01) {
      world_pts.emplace_back(m.world_pt(0), m.world_pt(1), m.world_pt(2));
      img_pts.push_back(m.img_pt);
    }
  }

  if ((int)world_pts.size() < 4) return false;

  SE3 inv = pose.inverse();
  AngleAxis aa(inv.rot);
  cv::Mat rvec = (cv::Mat_<double>(3,1) << aa.axis()(0) * aa.angle(),
                                            aa.axis()(1) * aa.angle(),
                                            aa.axis()(2) * aa.angle());
  cv::Mat tvec = (cv::Mat_<double>(3,1) << inv.trans(0), inv.trans(1), inv.trans(2));

  cv::solvePnP(world_pts, img_pts, camera_matrix_, cv::Mat(),
               rvec, tvec, true, cv::SOLVEPNP_ITERATIVE);

  AngleAxis aa_refined(
    std::sqrt(rvec.at<double>(0)*rvec.at<double>(0) +
              rvec.at<double>(1)*rvec.at<double>(1) +
              rvec.at<double>(2)*rvec.at<double>(2)),
    Vec3(rvec.at<double>(0), rvec.at<double>(1), rvec.at<double>(2)).normalized());

  pose = SE3(Quat(aa_refined),
             Vec3(tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2))).inverse();
  return true;
}

bool PnPSolver::IsGoodPose(const PnPResult& result, double min_inlier_ratio) {
  return result.success &&
         result.inlier_ratio >= min_inlier_ratio &&
         (int)result.inlier_matches.size() >= 10 &&
         result.reprojection_error >= 0 &&
         result.reprojection_error < 10.0;
}

} // namespace st_slam
