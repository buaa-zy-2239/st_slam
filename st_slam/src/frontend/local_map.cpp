#include "st_slam/frontend/local_map.h"
#include "st_slam/core/math_utils.h"
#include <Eigen/Core>
#include <algorithm>
#include <iostream>
#include <ceres/rotation.h>

namespace st_slam {

struct ReprojectionError {
  ReprojectionError(double u, double v, double fx, double fy,
                    double cx, double cy)
    : u_(u), v_(v), fx_(fx), fy_(fy), cx_(cx), cy_(cy) {}

  template <typename T>
  bool operator()(const T* const q, const T* const t,
                  const T* const point, T* residuals) const {
    T p[3];
    ceres::QuaternionRotatePoint(q, point, p);
    p[0] += t[0];
    p[1] += t[1];
    p[2] += t[2];

    T u_pred = fx_ * p[0] / p[2] + cx_;
    T v_pred = fy_ * p[1] / p[2] + cy_;

    residuals[0] = u_pred - T(u_);
    residuals[1] = v_pred - T(v_);
    return true;
  }

  static ceres::CostFunction* Create(double u, double v,
                                       double fx, double fy,
                                       double cx, double cy) {
    return new ceres::AutoDiffCostFunction<ReprojectionError, 2, 4, 3, 3>(
      new ReprojectionError(u, v, fx, fy, cx, cy));
  }

private:
  double u_, v_, fx_, fy_, cx_, cy_;
};

LocalMap::LocalMap(int window_size, int min_features_for_kf)
  : window_size_(window_size),
    min_features_for_kf_(min_features_for_kf) {}

int LocalMap::AddKeyFrame(const Frame& frame) {
  int id = next_kf_id_++;
  KeyFrame kf;
  kf.id = id;
  kf.timestamp = frame.timestamp;
  kf.pose = frame.pose;
  kf.keypoints = frame.keypoints;
  kf.descriptors = frame.descriptors.clone();
  kf.is_keyframe = true;
  keyframes_[id] = kf;
  return id;
}

int LocalMap::AddMapPoint(const Vec3& position, const cv::Mat& descriptor,
                           int keyframe_id, double depth) {
  int id = next_mp_id_++;
  MapPoint mp;
  mp.position = position;
  mp.descriptor = descriptor.clone();
  mp.id = id;
  mp.first_keyframe_id = keyframe_id;
  mp.observations.push_back(keyframe_id);
  mp.num_observations = 1;
  mp.depth = depth;
  mp.reprojection_error_avg = 0;
  mp.age = 0;
  map_points_[id] = mp;

  auto it = keyframes_.find(keyframe_id);
  if (it != keyframes_.end()) {
    it->second.map_points.push_back(id);
  }
  return id;
}

void LocalMap::AssociateKeyFrameWithMap(int keyframe_id,
                                          const std::vector<int>& map_point_ids) {
  auto kf_it = keyframes_.find(keyframe_id);
  if (kf_it == keyframes_.end()) return;

  for (int mp_id : map_point_ids) {
    auto mp_it = map_points_.find(mp_id);
    if (mp_it == map_points_.end()) continue;

    if (std::find(kf_it->second.map_points.begin(),
                   kf_it->second.map_points.end(), mp_id) ==
        kf_it->second.map_points.end()) {
      kf_it->second.map_points.push_back(mp_id);
    }
    if (std::find(mp_it->second.observations.begin(),
                   mp_it->second.observations.end(), keyframe_id) ==
        mp_it->second.observations.end()) {
      mp_it->second.observations.push_back(keyframe_id);
      mp_it->second.num_observations++;
    }
  }
}

bool LocalMap::IsKeyFrame(const Frame& frame) const {
  if (keyframes_.empty()) return true;
  if ((int)frame.keypoints.size() < min_features_for_kf_) return false;

  auto last_kf_it = keyframes_.end();
  int max_id = -1;
  for (auto it = keyframes_.begin(); it != keyframes_.end(); ++it) {
    if (it->first > max_id) {
      max_id = it->first;
      last_kf_it = it;
    }
  }
  if (last_kf_it == keyframes_.end()) return true;

  const KeyFrame& last_kf = last_kf_it->second;
  SE3 delta = last_kf.pose.inverse() * frame.pose;
  double trans_dist = delta.trans.norm();
  AngleAxis aa(delta.rot);
  double rot_deg = aa.angle() * 180.0 / M_PI;

  return trans_dist > 0.05 || rot_deg > 3.0;
}

std::vector<MapPoint> LocalMap::GetLocalMapPoints(int current_kf_id) const {
  std::vector<int> local_mp_ids;
  std::vector<int> recent_kf_ids;

  int start_id = std::max(0, current_kf_id - window_size_);
  for (int id = start_id; id <= current_kf_id; ++id) {
    auto it = keyframes_.find(id);
    if (it != keyframes_.end()) recent_kf_ids.push_back(id);
  }

  for (int kf_id : recent_kf_ids) {
    auto kf_it = keyframes_.find(kf_id);
    if (kf_it == keyframes_.end()) continue;
    for (int mp_id : kf_it->second.map_points) {
      if (std::find(local_mp_ids.begin(), local_mp_ids.end(), mp_id) ==
          local_mp_ids.end()) {
        local_mp_ids.push_back(mp_id);
      }
    }
  }

  std::vector<MapPoint> result;
  for (int mp_id : local_mp_ids) {
    auto it = map_points_.find(mp_id);
    if (it != map_points_.end()) result.push_back(it->second);
  }
  return result;
}

void LocalMap::AddMatch(const KFPair& match) {
  matches_.push_back(match);
}

KeyFrame* LocalMap::GetKeyFrame(int id) {
  auto it = keyframes_.find(id);
  return (it != keyframes_.end()) ? &it->second : nullptr;
}

MapPoint* LocalMap::GetMapPoint(int id) {
  auto it = map_points_.find(id);
  return (it != map_points_.end()) ? &it->second : nullptr;
}

const KeyFrame* LocalMap::GetKeyFrame(int id) const {
  auto it = keyframes_.find(id);
  return (it != keyframes_.end()) ? &it->second : nullptr;
}

const MapPoint* LocalMap::GetMapPoint(int id) const {
  auto it = map_points_.find(id);
  return (it != map_points_.end()) ? &it->second : nullptr;
}

void LocalMap::GetRecentKeyFrames(int n, std::vector<KeyFrame*>& out) {
  out.clear();
  std::vector<int> ids;
  for (const auto& [id, _] : keyframes_) ids.push_back(id);
  std::sort(ids.begin(), ids.end(), std::greater<int>());
  int count = 0;
  for (int id : ids) {
    if (count >= n) break;
    auto it = keyframes_.find(id);
    if (it != keyframes_.end()) {
      out.push_back(&it->second);
      count++;
    }
  }
}

void LocalMap::CullBadMapPoints(double max_reproj_error,
                                  int min_observations,
                                  double max_depth_ratio) {
  std::vector<int> to_remove;
  for (auto& [mp_id, mp] : map_points_) {
    mp.age++;
    if (mp.reprojection_error_avg > max_reproj_error && mp.age > 3) {
      to_remove.push_back(mp_id);
      continue;
    }
    if (mp.num_observations < min_observations && mp.age > 5) {
      to_remove.push_back(mp_id);
      continue;
    }
    if (mp.age > 2) {
      double max_depth = 0;
      for (int kf_id : mp.observations) {
        auto kf = keyframes_.find(kf_id);
        if (kf != keyframes_.end()) {
          double d = (kf->second.pose.inverse() * mp.position)(2);
          if (d > max_depth) max_depth = d;
        }
      }
      if (max_depth / std::max(mp.depth, 0.01) > max_depth_ratio) {
        to_remove.push_back(mp_id);
        continue;
      }
    }
  }

  for (int mp_id : to_remove) {
    for (auto& [kf_id, kf] : keyframes_) {
      kf.map_points.erase(
        std::remove(kf.map_points.begin(), kf.map_points.end(), mp_id),
        kf.map_points.end());
    }
    map_points_.erase(mp_id);
  }
}

void LocalMap::Clear() {
  keyframes_.clear();
  map_points_.clear();
  matches_.clear();
  next_kf_id_ = 0;
  next_mp_id_ = 0;
}

void LocalMap::UpdateKeyFramePose(int kf_id, const SE3& pose) {
  auto it = keyframes_.find(kf_id);
  if (it != keyframes_.end()) {
    it->second.pose = pose;
  }
}

void LocalMap::RunLocalBA(int current_frame_kf_id) {
  if ((int)keyframes_.size() < 2 || (int)map_points_.size() < 10) return;

  ceres::Problem problem;
  std::vector<int> opt_kf_ids;
  std::vector<int> opt_mp_ids;
  BuildLocalBAProblem(problem, opt_kf_ids, opt_mp_ids, current_frame_kf_id);

  if (problem.NumResiduals() == 0) return;

  ceres::Solver::Options options;
  options.linear_solver_type = ceres::SPARSE_SCHUR;
  options.minimizer_progress_to_stdout = false;
  options.max_num_iterations = 15;
  options.function_tolerance = 1e-6;
  options.parameter_tolerance = 1e-8;
  options.num_threads = 1;

  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  for (int kf_id : opt_kf_ids) {
    auto kf = keyframes_.find(kf_id);
    if (kf != keyframes_.end()) {
      auto& param = kf_map_[kf_id];
      Quat q(param[0], param[1], param[2], param[3]);
      Vec3 t(param[4], param[5], param[6]);
      kf->second.pose = SE3(q.normalized(), t);
    }
  }

  for (int mp_id : opt_mp_ids) {
    auto mp = map_points_.find(mp_id);
    if (mp != map_points_.end()) {
      auto& param = mp_map_[mp_id];
      Vec3 new_pos(param[0], param[1], param[2]);
      double dist = (new_pos - mp->second.position).norm();
      if (dist < 1.0) {
        mp->second.position = new_pos;
      }
    }
  }
}

void LocalMap::BuildLocalBAProblem(ceres::Problem& problem,
                                     std::vector<int>& opt_kf_ids,
                                     std::vector<int>& opt_mp_ids,
                                     int current_kf_id) {
  kf_map_.clear();
  mp_map_.clear();
  opt_kf_ids.clear();
  opt_mp_ids.clear();

  double fx = 525.0, fy = 525.0, cx = 319.5, cy = 239.5;
  double cauchy_c = 1.345;

  int start_kf = std::max(0, next_kf_id_ - 1 - window_size_);
  for (int kf_id = start_kf; kf_id < next_kf_id_; ++kf_id) {
    auto kf_it = keyframes_.find(kf_id);
    if (kf_it == keyframes_.end()) continue;

    KeyFrame& kf = kf_it->second;
    std::array<double, 7>& param = kf_map_[kf_id];
    param[0] = kf.pose.rot.w();
    param[1] = kf.pose.rot.x();
    param[2] = kf.pose.rot.y();
    param[3] = kf.pose.rot.z();
    param[4] = kf.pose.trans(0);
    param[5] = kf.pose.trans(1);
    param[6] = kf.pose.trans(2);
    opt_kf_ids.push_back(kf_id);
  }

  for (int kf_id : opt_kf_ids) {
    auto kf_it = keyframes_.find(kf_id);
    if (kf_it == keyframes_.end()) continue;
    KeyFrame& kf = kf_it->second;

    for (int mp_id : kf.map_points) {
      auto mp_it = map_points_.find(mp_id);
      if (mp_it == map_points_.end()) continue;
      if (mp_map_.find(mp_id) == mp_map_.end()) {
        std::array<double, 3>& mp_param = mp_map_[mp_id];
        mp_param[0] = mp_it->second.position(0);
        mp_param[1] = mp_it->second.position(1);
        mp_param[2] = mp_it->second.position(2);
        opt_mp_ids.push_back(mp_id);
      }
    }
  }

  // Ceres 2.0: must AddParameterBlock BEFORE adding residual blocks
  // Split KF into rotation (4) and translation (3) for compatibility with residual blocks
  std::unordered_map<int, double*> kf_rot_ptr, kf_trans_ptr;
  for (int kf_id : opt_kf_ids) {
    kf_rot_ptr[kf_id] = kf_map_[kf_id].data();
    kf_trans_ptr[kf_id] = kf_map_[kf_id].data() + 4;
    problem.AddParameterBlock(kf_rot_ptr[kf_id], 4);
    problem.AddParameterBlock(kf_trans_ptr[kf_id], 3);
  }
  for (int mp_id : opt_mp_ids) {
    problem.AddParameterBlock(mp_map_[mp_id].data(), 3);
  }

  ceres::LossFunction* loss = new ceres::CauchyLoss(cauchy_c);

  int residual_count = 0;
  for (int kf_id : opt_kf_ids) {
    auto kf_it = keyframes_.find(kf_id);
    if (kf_it == keyframes_.end()) continue;
    KeyFrame& kf = kf_it->second;

    for (size_t kp_idx = 0; kp_idx < kf.map_points.size(); ++kp_idx) {
      int mp_id = kf.map_points[kp_idx];
      if (mp_map_.find(mp_id) == mp_map_.end()) continue;
      if (kp_idx >= kf.keypoints.size()) continue;

      const auto& kp = kf.keypoints[kp_idx];
      auto* cost = ReprojectionError::Create(
        kp.pt.x, kp.pt.y, fx, fy, cx, cy);

      problem.AddResidualBlock(cost, loss,
                                kf_map_[kf_id].data(),
                                kf_map_[kf_id].data() + 4,
                                mp_map_[mp_id].data());
      residual_count++;
    }
  }

  // Fix the first keyframe in the window to ground the optimization
  if (!opt_kf_ids.empty()) {
    int anchor_id = opt_kf_ids[0];
    if (kf_rot_ptr.find(anchor_id) != kf_rot_ptr.end()) {
      problem.SetParameterBlockConstant(kf_rot_ptr[anchor_id]);
      problem.SetParameterBlockConstant(kf_trans_ptr[anchor_id]);
    }
  }
}

} // namespace st_slam
