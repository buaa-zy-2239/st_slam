#include "st_slam/frontend/frame.h"
#include <opencv2/imgproc.hpp>
#include <limits>
#include <algorithm>

namespace st_slam {

FrameHandler::FrameHandler(double fx, double fy, double cx, double cy,
                            double depth_factor,
                            double depth_min, double depth_max)
  : fx_(fx), fy_(fy), cx_(cx), cy_(cy),
    depth_factor_(depth_factor),
    depth_min_(depth_min), depth_max_(depth_max) {}

void FrameHandler::SetCameraParams(double fx, double fy, double cx, double cy) {
  fx_ = fx; fy_ = fy; cx_ = cx; cy_ = cy;
}

double FrameHandler::SampleDepthMedian(const cv::Mat& depth, int u, int v,
                                         int patch_size, double depth_factor) {
  int half = patch_size / 2;
  int h = depth.rows, w = depth.cols;

  std::vector<double> valid_depths;
  valid_depths.reserve(patch_size * patch_size);

  for (int dy = -half; dy <= half; ++dy) {
    for (int dx = -half; dx <= half; ++dx) {
      int pu = u + dx;
      int pv = v + dy;
      if (pu < 0 || pu >= w || pv < 0 || pv >= h) continue;

      double d;
      if (depth.type() == CV_16U) {
        unsigned short val = depth.at<unsigned short>(pv, pu);
        if (val == 0) continue;
        d = val / depth_factor;
      } else if (depth.type() == CV_32F) {
        float val = depth.at<float>(pv, pu);
        if (val <= 0) continue;
        d = val;
      } else {
        continue;
      }
      valid_depths.push_back(d);
    }
  }

  if (valid_depths.empty()) return 0;

  std::nth_element(valid_depths.begin(),
                    valid_depths.begin() + valid_depths.size() / 2,
                    valid_depths.end());
  double median = valid_depths[valid_depths.size() / 2];

  double sum = 0, count = 0;
  double threshold = median * 0.15;
  for (double d : valid_depths) {
    if (std::abs(d - median) < threshold) {
      sum += d;
      count++;
    }
  }
  return count > 0 ? sum / count : median;
}

void FrameHandler::BackProjectDepth(Frame& frame) {
  if (frame.depth.empty()) return;

  frame.keypoints_3d.clear();
  frame.keypoints_3d.reserve(frame.keypoints.size());

  for (const auto& kp : frame.keypoints) {
    int u = cvRound(kp.pt.x);
    int v = cvRound(kp.pt.y);
    if (u < 1 || u >= frame.depth.cols - 1 ||
        v < 1 || v >= frame.depth.rows - 1) {
      frame.keypoints_3d.push_back(Vec3::Zero());
      continue;
    }

    double d = SampleDepthMedian(frame.depth, u, v, 3, depth_factor_);
    if (!CheckDepth(d, depth_min_, depth_max_)) {
      frame.keypoints_3d.push_back(Vec3::Zero());
      continue;
    }

    Vec3 pt3d = PixelToCamera(kp.pt, d, fx_, fy_, cx_, cy_);
    frame.keypoints_3d.push_back(pt3d);
  }
}

int FrameHandler::ComputeKeypointsFromDepth(Frame& frame) {
  if (frame.depth.empty()) return 0;

  int valid = 0;
  std::vector<cv::KeyPoint> valid_kps;
  std::vector<Vec3> valid_pts3d;

  for (size_t i = 0; i < frame.keypoints.size(); ++i) {
    int u = cvRound(frame.keypoints[i].pt.x);
    int v = cvRound(frame.keypoints[i].pt.y);
    if (u < 1 || u >= frame.depth.cols - 1 ||
        v < 1 || v >= frame.depth.rows - 1) continue;

    double d = SampleDepthMedian(frame.depth, u, v, 3, depth_factor_);
    if (!CheckDepth(d, depth_min_, depth_max_)) continue;

    Vec3 pt3d = PixelToCamera(frame.keypoints[i].pt, d, fx_, fy_, cx_, cy_);
    valid_kps.push_back(frame.keypoints[i]);
    valid_pts3d.push_back(pt3d);
    valid++;
  }

  frame.keypoints = std::move(valid_kps);
  frame.keypoints_3d = std::move(valid_pts3d);
  return valid;
}

void FrameHandler::ComputePhotometricError(Frame& frame) {
  if (frame.gray.empty() || frame.keypoints.empty()) return;

  const int patch_size = 4;
  const int half_patch = patch_size / 2;

  frame.photometric_errors.clear();
  frame.photometric_errors.reserve(frame.keypoints.size());

  for (size_t i = 0; i < frame.keypoints.size(); ++i) {
    int u = cvRound(frame.keypoints[i].pt.x);
    int v = cvRound(frame.keypoints[i].pt.y);

    if (u < half_patch || u >= frame.gray.cols - half_patch ||
        v < half_patch || v >= frame.gray.rows - half_patch) {
      frame.photometric_errors.push_back(std::numeric_limits<float>::max());
      continue;
    }

    float mean = 0.0f;
    int count = 0;
    for (int dy = -half_patch; dy <= half_patch; ++dy) {
      for (int dx = -half_patch; dx <= half_patch; ++dx) {
        mean += frame.gray.at<unsigned char>(v + dy, u + dx);
        count++;
      }
    }
    mean /= count;

    float var = 0.0f;
    for (int dy = -half_patch; dy <= half_patch; ++dy) {
      for (int dx = -half_patch; dx <= half_patch; ++dx) {
        float diff = frame.gray.at<unsigned char>(v + dy, u + dx) - mean;
        var += diff * diff;
      }
    }
    var /= count;

    frame.photometric_errors.push_back(std::sqrt(var));
  }
}

bool FrameHandler::CheckDepth(double d, double min_d, double max_d) {
  return d > min_d && d < max_d;
}

Vec3 FrameHandler::PixelToCamera(const cv::Point2d& pt, double d,
                                  double fx, double fy,
                                  double cx, double cy) {
  double x = (pt.x - cx) * d / fx;
  double y = (pt.y - cy) * d / fy;
  return Vec3(x, y, d);
}

cv::Point2d FrameHandler::CameraToPixel(const Vec3& pt,
                                          double fx, double fy,
                                          double cx, double cy) {
  double u = pt(0) * fx / pt(2) + cx;
  double v = pt(1) * fy / pt(2) + cy;
  return cv::Point2d(u, v);
}

} // namespace st_slam
