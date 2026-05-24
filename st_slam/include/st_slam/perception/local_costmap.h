#ifndef ST_SLAM_LOCAL_COSTMAP_H
#define ST_SLAM_LOCAL_COSTMAP_H

#include "st_slam/core/types.h"
#include <vector>
#include <opencv2/core.hpp>

namespace st_slam {

struct CostmapConfig {
  double resolution = 0.05;
  int width = 80;
  int height = 80;
  double robot_x_offset = 2.0;
  double min_obstacle_height = 0.05;
  double max_obstacle_height = 1.20;
};

class LocalCostmap {
public:
  explicit LocalCostmap(const CostmapConfig& config = CostmapConfig());
  ~LocalCostmap() = default;

  cv::Mat Update(const SE3& T_w_b, const std::vector<Vec3>& active_mps);

  double last_update_ms() const { return last_update_ms_; }
  int num_obstacle_points() const { return num_obs_; }

  const cv::Mat& grid() const { return grid_map_; }

private:
  CostmapConfig cfg_;
  cv::Mat grid_map_;
  double last_update_ms_ = 0;
  int num_obs_ = 0;
};

} // namespace st_slam

#endif
