#ifndef ST_SLAM_LOCAL_MAP_H
#define ST_SLAM_LOCAL_MAP_H

#include "st_slam/core/types.h"
#include "st_slam/core/math_utils.h"
#include <vector>
#include <deque>
#include <memory>
#include <unordered_map>
#include <array>
#include <ceres/ceres.h>

namespace st_slam {

struct MapPoint {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Vec3 position;
  cv::Mat descriptor;
  int id;
  int first_keyframe_id;
  std::vector<int> observations;
  int num_observations;
  double depth;
  double reprojection_error_avg;
  int age;
};

struct KeyFrame {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  int id;
  double timestamp;
  SE3 pose;
  std::vector<int> map_points;
  std::vector<cv::KeyPoint> keypoints;
  cv::Mat descriptors;
  bool is_keyframe;
};

class LocalMap {
public:
  explicit LocalMap(int window_size = 7, int min_features_for_kf = 50);

  int AddKeyFrame(const Frame& frame);

  int AddMapPoint(const Vec3& position, const cv::Mat& descriptor,
                   int keyframe_id, double depth);

  void AssociateKeyFrameWithMap(int keyframe_id,
                                 const std::vector<int>& map_point_ids);

  bool IsKeyFrame(const Frame& frame) const;

  std::vector<MapPoint> GetLocalMapPoints(int current_kf_id) const;

  struct KFPair {
    int ref_kf_id;
    int cur_kf_id;
    std::vector<cv::DMatch> matches;
  };
  void AddMatch(const KFPair& match);

  int NumKeyFrames() const { return keyframes_.size(); }
  int NumMapPoints() const { return map_points_.size(); }
  int NextKFID() const { return next_kf_id_; }

  void RunLocalBA(int current_frame_kf_id = -1);

  void CullBadMapPoints(double max_reproj_error = 8.0,
                         int min_observations = 2,
                         double max_depth_ratio = 50.0);

  KeyFrame* GetKeyFrame(int id);
  MapPoint* GetMapPoint(int id);
  const KeyFrame* GetKeyFrame(int id) const;
  const MapPoint* GetMapPoint(int id) const;

  std::unordered_map<int, KeyFrame>& GetAllKeyframes() { return keyframes_; }
  const std::unordered_map<int, KeyFrame>& GetAllKeyframes() const { return keyframes_; }

  void UpdateKeyFramePose(int kf_id, const SE3& pose);

  void GetRecentKeyFrames(int n, std::vector<KeyFrame*>& out);

  void Clear();

private:
  int window_size_;
  int min_features_for_kf_;
  int next_kf_id_ = 0;
  int next_mp_id_ = 0;

  std::unordered_map<int, KeyFrame> keyframes_;
  std::unordered_map<int, MapPoint> map_points_;
  std::vector<KFPair> matches_;

  void BuildLocalBAProblem(ceres::Problem& problem,
                            std::vector<int>& opt_kf_ids,
                            std::vector<int>& opt_mp_ids,
                            int current_kf_id);

  std::unordered_map<int, std::array<double, 7>> kf_map_;
  std::unordered_map<int, std::array<double, 3>> mp_map_;
};

} // namespace st_slam

#endif
