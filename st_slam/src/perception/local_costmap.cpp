#include "st_slam/perception/local_costmap.h"
#include <opencv2/imgproc.hpp>
#include <chrono>
#include <iostream>

namespace st_slam {

LocalCostmap::LocalCostmap(const CostmapConfig& config)
  : cfg_(config) {
  grid_map_ = cv::Mat(cfg_.height, cfg_.width, CV_8UC1, cv::Scalar(255));
}

cv::Mat LocalCostmap::Update(const SE3& T_w_b, const std::vector<Vec3>& active_mps) {
  auto start = std::chrono::steady_clock::now();
  num_obs_ = 0;

  grid_map_.setTo(cv::Scalar(255));

  SE3 T_b_w = T_w_b.inverse();
  double inv_res = 1.0 / cfg_.resolution;
  int center_cols = cfg_.width / 2;
  int center_rows = cfg_.height - static_cast<int>(cfg_.robot_x_offset * inv_res);

  for (const auto& P_w : active_mps) {
    Vec3 P_b = T_b_w * P_w;

    if (P_b(2) < cfg_.min_obstacle_height || P_b(2) > cfg_.max_obstacle_height) {
      continue;
    }

    int u = center_cols - static_cast<int>(P_b(1) * inv_res);
    int v = center_rows - static_cast<int>(P_b(0) * inv_res);

    if (u >= 0 && u < cfg_.width && v >= 0 && v < cfg_.height) {
      grid_map_.at<uchar>(v, u) = 0;
      num_obs_++;
    }
  }

  cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
  cv::dilate(grid_map_, grid_map_, kernel);

  auto end = std::chrono::steady_clock::now();
  last_update_ms_ = std::chrono::duration<double, std::milli>(end - start).count();
  return grid_map_;
}

} // namespace st_slam
