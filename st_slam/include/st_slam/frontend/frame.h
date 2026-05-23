#ifndef ST_SLAM_FRAME_H
#define ST_SLAM_FRAME_H

#include "st_slam/core/types.h"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <memory>

namespace st_slam {

class FrameHandler {
public:
  FrameHandler(double fx, double fy, double cx, double cy,
                double depth_factor = 5000.0,
                double depth_min = 0.1, double depth_max = 10.0);

  void SetCameraParams(double fx, double fy, double cx, double cy);

  void BackProjectDepth(Frame& frame);

  int ComputeKeypointsFromDepth(Frame& frame);

  void ComputePhotometricError(Frame& frame);

  static bool CheckDepth(double d, double min_d, double max_d);

  static Vec3 PixelToCamera(const cv::Point2d& pt, double d,
                             double fx, double fy, double cx, double cy);

  static cv::Point2d CameraToPixel(const Vec3& pt,
                                    double fx, double fy, double cx, double cy);

  static double SampleDepthMedian(const cv::Mat& depth, int u, int v,
                                    int patch_size = 3, double depth_factor = 5000.0);

private:
  double fx_, fy_, cx_, cy_;
  double depth_factor_;
  double depth_min_, depth_max_;
};

} // namespace st_slam

#endif
