#include "st_slam/frontend/tracking.h"
#include "st_slam/core/config.h"
#include "st_slam/core/math_utils.h"
#include <Eigen/Geometry>
#include <Eigen/Dense>
#include <opencv2/calib3d.hpp>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <deque>

namespace st_slam {

Tracking::Tracking(const STSLAMConfig& config)
  : config_(config),
    state_(TrackingState::INITIALIZING),
    frame_counter_(0),
    num_loops_detected_(0) {

  orb_extractor_ = std::make_unique<ORBExtractor>(1500, 1.2f, 8, 20, 7);
  frame_handler_ = std::make_unique<FrameHandler>(
    525.0, 525.0, 319.5, 239.5, 5000.0, 0.1, 10.0);
  st_nsp_ = std::make_unique<STNSP>(
    config.nsp_beta, config.nsp_tau1, config.nsp_degeneracy_threshold);
  ncio_ = std::make_unique<NCIO>(
    config.ncio_kinetic_noise, config.ncio_shock_gain,
    config.ncio_covariance_max_det, config.ncio_covariance_threshold);
  t_pvd_ = std::make_unique<TPVD>(
    config.tpvd_eta, config.tpvd_gamma_init, config.tpvd_temporal_threshold);
  a_csg_ = std::make_unique<ACSG>(
    config.acsg_mu0, config.acsg_alpha_init,
    config.acsg_alpha_max_multiplier, config.acsg_delta, config.acsg_c_default);
  pnp_solver_ = std::make_unique<PnPSolver>(
    525.0, 525.0, 319.5, 239.5, 3.0, 12, 300);
  local_map_ = std::make_unique<LocalMap>(20, 40);
  pose_graph_ = std::make_unique<PoseGraph>(100.0, 1000.0);
  local_costmap_ = std::make_unique<LocalCostmap>();
  topo_graph_ = std::make_unique<TopologicalGraph>();
  pose_interpolator_ = std::make_unique<PoseInterpolator>();
  
  if (config.use_loop_closure) {
    loop_closer_ = std::make_unique<LoopCloser>(config.vocab_path, 525.0, 525.0, 319.5, 239.5, local_map_.get());
    if (!config.vocab_path.empty()) {
      std::cout << "[Tracking] LoopCloser initialized with vocab: " << config.vocab_path << std::endl;
    } else {
      std::cout << "[Tracking] LoopCloser initialized (no vocab, will build online)" << std::endl;
    }
  }

  current_pose_ = SE3::Identity();
  last_pose_ = SE3::Identity();
}

void Tracking::Reset() {
  state_ = TrackingState::INITIALIZING;
  frame_counter_ = 0;
  current_pose_ = SE3::Identity();
  last_pose_ = SE3::Identity();
  ncio_->Reset();
  t_pvd_->Reset();
  a_csg_->Reset();
  local_map_->Clear();
  last_keyframe_id_ = -1;
  last_near_kf_id_ = -1;
  has_imu_integration_ = false;
}

void Tracking::SetIMUData(double timestamp, const Vec3& accel, const Vec3& gyro) {
  ncio_->RecordInertialData(timestamp, accel, gyro);
  imu_accel_buffer_.push_back(accel);
  imu_gyro_buffer_.push_back(gyro);
  if (imu_accel_buffer_.size() > 100) {
    imu_accel_buffer_.erase(imu_accel_buffer_.begin());
    imu_gyro_buffer_.erase(imu_gyro_buffer_.begin());
  }
  UpdateGravityFromAccel(accel);
}

void Tracking::UpdateGravityFromAccel(const Vec3& accel) {
  accel_buffer_.push_back(accel);
  if (accel_buffer_.size() > 50) {
    accel_buffer_.pop_front();
  }
  if (accel_buffer_.size() < 5) return;

  Vec3 accel_avg = Vec3::Zero();
  for (const auto& a : accel_buffer_) accel_avg += a;
  accel_avg /= accel_buffer_.size();

  double g_mag = accel_avg.norm();
  if (g_mag > 8.0 && g_mag < 11.0) {
    gravity_cam_frame_ = accel_avg.normalized() * 9.81;
  }
}

void Tracking::ApplyGravityConstraint(SE3& pose) {
  if (accel_buffer_.size() < 10) return;

  double accel_var = 0;
  Vec3 accel_mean = Vec3::Zero();
  for (const auto& a : accel_buffer_) accel_mean += a;
  accel_mean /= accel_buffer_.size();
  for (const auto& a : accel_buffer_) {
    Vec3 d = a - accel_mean;
    accel_var += d.squaredNorm();
  }
  accel_var /= accel_buffer_.size();
  if (accel_var > 0.5) return;

  Vec3 grav_world(0, 1, 0);
  Vec3 grav_cam = gravity_cam_frame_.normalized();

  Vec3 grav_in_world = pose.rot.conjugate() * grav_cam;
  grav_in_world.normalize();

  double cos_angle = grav_in_world.dot(grav_world);
  cos_angle = std::clamp(cos_angle, -1.0, 1.0);
  double angle = std::acos(cos_angle);
  if (angle < 0.02) return;

  Vec3 axis = grav_in_world.cross(grav_world);
  if (axis.norm() < 1e-6) return;
  axis.normalize();

  double strength = 0.3;
  Quat correction(AngleAxis(angle * strength, axis));
  pose.rot = pose.rot * correction;
}

bool Tracking::Initialize(const Frame& frame) {
  if (init_pose_override_) {
    current_pose_ = *init_pose_override_;
    last_pose_ = *init_pose_override_;
    std::cout << "[CRITICAL SEED] SLAM Seeded with GT Origin!" << std::endl;
    std::cout << "  Init pose: pos=(" << current_pose_.trans(0) << ", "
              << current_pose_.trans(1) << ", " << current_pose_.trans(2) << ")" << std::endl;
  } else {
    current_pose_ = SE3::Identity();
    last_pose_ = SE3::Identity();
  }
  
  ref_frame_ = frame;
  ref_frame_.pose = current_pose_;
  state_ = TrackingState::TRACKING_GOOD;
  frame_counter_ = 0;

  // CRITICAL FIX: Set frame.pose to identity for the first frame!
  // This fixes the "all keyframes have zero pose" bug that caused PGO to do nothing
  Frame frame_with_pose = frame;
  frame_with_pose.pose = current_pose_;

  int kf_id = local_map_->AddKeyFrame(frame_with_pose);
  last_keyframe_id_ = kf_id;
  int num_mps = CreateMapPointsFromKeyFrame(kf_id);

  std::cout << "[Tracking] Initialized from frame " << frame.id
            << " (" << frame.keypoints.size() << " features, "
            << num_mps << " map points)" << std::endl;
  return true;
}

void Tracking::SetInitPose(const SE3& pose) {
  init_pose_override_ = pose;
  std::cout << "[Config] Init pose override set: pos=(" << pose.trans(0) << ", "
            << pose.trans(1) << ", " << pose.trans(2) << ")" << std::endl;
}

void Tracking::ForceCurrentPose(const SE3& pose) {
  current_pose_ = pose;
  last_pose_ = pose;
  std::cout << "[CRITICAL] Current pose forced to: pos=(" << pose.trans(0) << ", "
            << pose.trans(1) << ", " << pose.trans(2) << ")" << std::endl;
}

SE3 Tracking::PredictPoseFromMotionModel() const {
  if (frame_counter_ < 2) return current_pose_;
  SE3 motion = last_pose_.inverse() * current_pose_;
  return current_pose_ * motion;
}

SE3 Tracking::PredictPoseFromIMU(double dt, const Vec3& accel, const Vec3& gyro) {
  if (!has_imu_integration_ || dt < 1e-6) {
    has_imu_integration_ = true;
    last_velocity_ = Vec3::Zero();
    return current_pose_;
  }

  Vec3 w = gyro * dt;
  double w_norm = w.norm();
  Quat delta_q = Quat::Identity();
  if (w_norm > 1e-10) {
    delta_q = Quat(AngleAxis(w_norm, w / w_norm));
  }
  Quat new_rot = current_pose_.rot * delta_q;

  Vec3 accel_world = current_pose_.rot * accel;
  Vec3 gravity(0, 0, -9.81);
  Vec3 vel_delta = (accel_world + gravity) * dt;
  Vec3 new_vel = last_velocity_ + vel_delta;
  Vec3 new_trans = current_pose_.trans + last_velocity_ * dt + 0.5 * vel_delta * dt;

  last_velocity_ = new_vel;
  return SE3(new_rot, new_trans);
}

bool Tracking::TryCreateKeyFrame(Frame& frame) {
  if (local_map_->IsKeyFrame(frame)) {
    int kf_id = local_map_->AddKeyFrame(frame);
    int num_mps = CreateMapPointsFromKeyFrame(kf_id);
    last_keyframe_id_ = kf_id;

    // Store KF data for loop detection
    StoredKeyFrame skf;
    skf.id = kf_id;
    skf.pose = current_pose_;
    skf.keypoints = frame.keypoints;
    skf.descriptors = frame.descriptors.clone();
    skf.timestamp = frame.timestamp;
    kf_history_.push_back(skf);
    if (kf_history_.size() > 500) kf_history_.pop_front();
    
    // Add KF to LoopCloser database
    if (loop_closer_) {
      loop_closer_->AddKeyFrame(kf_id, frame.descriptors);
    }

    // Topological graph: add node and edge from previous node
    topo_graph_->AddNode(kf_id, current_pose_);
    if (last_topo_kf_id_ >= 0) {
      SE3 rel = local_map_->GetKeyFrame(last_topo_kf_id_)->pose.inverse() * current_pose_;
      topo_graph_->AddEdge(last_topo_kf_id_, kf_id, rel, 50.0, false);
    }
    last_topo_kf_id_ = kf_id;

    return true;
  }
  return false;
}

int Tracking::CreateMapPointsFromKeyFrame(int kf_id) {
  const KeyFrame* kf = local_map_->GetKeyFrame(kf_id);
  if (!kf) return 0;

  int count = 0;
  std::vector<int> mp_ids(kf->keypoints.size(), -1);
  for (size_t i = 0; i < kf->keypoints.size(); ++i) {
    if (i < ref_frame_.keypoints_3d.size() && ref_frame_.keypoints_3d[i].norm() > 1e-6) {
      Vec3 world_pt = current_pose_ * ref_frame_.keypoints_3d[i];
      double depth = ref_frame_.keypoints_3d[i](2);
      int mp_id = local_map_->AddMapPoint(world_pt, ref_frame_.descriptors.row(i), kf_id, depth);
      mp_ids[i] = mp_id;
      count++;
    }
  }
  local_map_->AssociateKeyFrameWithMap(kf_id, mp_ids);
  return count;
}

SE3 Tracking::AlignFramePnP(const Frame& ref_frame, const Frame& cur_frame,
                              const std::vector<cv::DMatch>& matches,
                              const SE3& initial_guess) {
  PnPResult result = pnp_solver_->EstimatePose(matches, ref_frame, cur_frame, initial_guess);

  if (result.success && result.inlier_ratio > 0.3) {
    return result.pose;
  }

  if (result.success) {
    return result.pose;
  }

  return initial_guess;
}

TrackingReport Tracking::TrackFrameWithPnP(Frame& frame) {
  auto start = std::chrono::steady_clock::now();

  SE3 pred_pose = PredictPoseFromMotionModel();
  frame.pose = pred_pose;

  std::vector<cv::DMatch> matches;
  ORBExtractor::ComputeStereoMatches(ref_frame_, frame, matches);

  if (matches.size() >= 10) {
    PnPResult pnp_result = pnp_solver_->EstimatePose(matches, ref_frame_, frame, pred_pose);

    if (pnp_result.success && pnp_result.inlier_ratio > 0.3) {
      current_pose_ = pnp_result.pose;
      frame.pose = current_pose_;
      report_.inlier_ratio = pnp_result.inlier_ratio;
      report_.num_matches = pnp_result.inlier_matches.size();

      Mat6 H = ComputeGeometricHessian(pnp_result.inlier_matches, ref_frame_, frame);
      VecX residual = ComputeResidualVector(pnp_result.inlier_matches, ref_frame_, frame);

      double cond;
      DegeneracyState degen = st_nsp_->AnalyzeHessian(H, cond);
      report_.degeneracy = degen;
      report_.hessian_condition_number = cond;

      a_csg_->ComputeCauchyScale(H, residual);
      report_.mu_damping = a_csg_->ComputeDampingMu(H, residual, 0.01);
      report_.cauchy_scale = a_csg_->GetCurrentC();

      if (degen == DegeneracyState::NONE) {
        state_ = TrackingState::TRACKING_GOOD;
      } else {
        state_ = TrackingState::TRACKING_DEGRADED;
      }
    } else {
      current_pose_ = pred_pose;
      report_.inlier_ratio = 0;
      report_.num_matches = 0;
      state_ = TrackingState::TRACKING_DEGRADED;
      report_.degeneracy = DegeneracyState::FULL_DEGENERATE;
    }
  } else {
    current_pose_ = pred_pose;
    report_.inlier_ratio = 0;
    report_.num_matches = 0;
    state_ = TrackingState::TRACKING_LOST;
    report_.degeneracy = DegeneracyState::FULL_DEGENERATE;
  }

  // Remove the verbose TRACKING debug block (delta_t print)
  if (frame_counter_ > 0) {
    Vec3 delta_t = current_pose_.trans - last_pose_.trans;
  }

  last_pose_ = current_pose_;
  last_frame_ = frame;
  ref_frame_ = frame;
  ref_frame_.pose = current_pose_;

  // Lightweight backend: KF management + loop detection + global PGO
  if (state_ == TrackingState::TRACKING_GOOD) {
    if (TryCreateKeyFrame(frame)) {
      local_map_->CullBadMapPoints(8.0, 2, 50.0);

      // Every keyframe after the first 5: detect loops
      if (local_map_->NumKeyFrames() > 5) {
        int edges_before = pose_graph_->NumEdges();
        pose_graph_->BuildFromKeyframes(local_map_->GetAllKeyframes());
        bool loop_found = DetectLoopCorrection();

        // Only run PGO when a new loop edge was added
        bool new_loop = loop_found || (pose_graph_->NumEdges() > edges_before);
        if (new_loop && pose_graph_->NumEdges() > 3) {
          auto* opt_kf = local_map_->GetKeyFrame(last_keyframe_id_);
          SE3 kf_before_opt;
          if (opt_kf) kf_before_opt = opt_kf->pose;

          pose_graph_->Optimize(local_map_->GetAllKeyframes());

          if (opt_kf) {
            SE3 kf_after_opt = opt_kf->pose;
            SE3 correction = kf_before_opt.inverse() * kf_after_opt;
            current_pose_ = correction * current_pose_;
            last_pose_ = correction * last_pose_;
          }
        }
      }
    }
  }

  auto end = std::chrono::steady_clock::now();
  report_.tracking_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
  report_.state = state_;
  report_.num_features = frame.keypoints.size();

  CheckDegenerateCState();
  return report_;
}

TrackingReport Tracking::TrackWithLocalMap(Frame& frame) {
  auto start = std::chrono::steady_clock::now();

  SE3 pred_pose = PredictPoseFromMotionModel();
  frame.pose = pred_pose;

  if (last_keyframe_id_ < 0) return TrackFrameWithPnP(frame);

  std::vector<cv::DMatch> matches;
  ORBExtractor::ComputeStereoMatches(ref_frame_, frame, matches);

  PnPResult pnp_result = pnp_solver_->EstimatePose(matches, ref_frame_, frame, pred_pose);

  if (pnp_result.success && pnp_result.inlier_ratio > 0.25) {
    current_pose_ = pnp_result.pose;
    frame.pose = current_pose_;
    report_.inlier_ratio = pnp_result.inlier_ratio;
    report_.num_matches = pnp_result.inlier_matches.size();

    Mat6 H = ComputeGeometricHessian(pnp_result.inlier_matches, ref_frame_, frame);
    VecX residual = ComputeResidualVector(pnp_result.inlier_matches, ref_frame_, frame);

    double cond;
    DegeneracyState degen = st_nsp_->AnalyzeHessian(H, cond);
    report_.degeneracy = degen;
    report_.hessian_condition_number = cond;

    a_csg_->ComputeCauchyScale(H, residual);
    report_.mu_damping = a_csg_->ComputeDampingMu(H, residual, 0.01);
    report_.cauchy_scale = a_csg_->GetCurrentC();

    state_ = (degen == DegeneracyState::NONE) ?
             TrackingState::TRACKING_GOOD : TrackingState::TRACKING_DEGRADED;
  } else {
    current_pose_ = pred_pose;
    report_.inlier_ratio = 0;
    report_.num_matches = 0;
    report_.degeneracy = DegeneracyState::FULL_DEGENERATE;
    state_ = TrackingState::TRACKING_DEGRADED;
  }

  last_frame_ = frame;

  if (state_ == TrackingState::TRACKING_GOOD) {
    if (TryCreateKeyFrame(frame)) {
      local_map_->CullBadMapPoints(8.0, 2, 50.0);
      if (local_map_->NumKeyFrames() > 5) {
        int edges_before = pose_graph_->NumEdges();
        pose_graph_->BuildFromKeyframes(local_map_->GetAllKeyframes());
        bool loop_found = DetectLoopCorrection();

        bool new_loop = loop_found || (pose_graph_->NumEdges() > edges_before);
         if (new_loop && pose_graph_->NumEdges() > 3) {
          auto* opt_kf = local_map_->GetKeyFrame(last_keyframe_id_);
          SE3 kf_before_opt;
          if (opt_kf) kf_before_opt = opt_kf->pose;

          pose_graph_->Optimize(local_map_->GetAllKeyframes());

          if (opt_kf) {
            SE3 kf_after_opt = opt_kf->pose;
            SE3 correction = kf_before_opt.inverse() * kf_after_opt;
            current_pose_ = correction * current_pose_;
            last_pose_ = correction * last_pose_;
          }
        }
      }
    }
  }

  last_pose_ = current_pose_;
  ref_frame_ = frame;
  ref_frame_.pose = current_pose_;

  auto end = std::chrono::steady_clock::now();
  report_.tracking_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
  report_.state = state_;
  report_.num_features = frame.keypoints.size();

  CheckDegenerateCState();
  return report_;
}

TrackingReport Tracking::Relocalize(Frame& frame) {
  auto start = std::chrono::steady_clock::now();

  SE3 identity_guess = SE3::Identity();
  std::vector<cv::DMatch> matches;
  ORBExtractor::ComputeStereoMatches(ref_frame_, frame, matches);

  PnPResult result = pnp_solver_->EstimatePose(matches, ref_frame_, frame, identity_guess);

  if (result.success && result.inlier_ratio > 0.2) {
    current_pose_ = result.pose;
    frame.pose = result.pose;
    state_ = TrackingState::TRACKING_GOOD;
    report_.inlier_ratio = result.inlier_ratio;
    report_.num_matches = result.inlier_matches.size();
  } else {
    state_ = TrackingState::TRACKING_LOST;
    report_.inlier_ratio = 0;
    report_.num_matches = 0;
  }

  auto end = std::chrono::steady_clock::now();
  report_.tracking_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
  report_.state = state_;
  return report_;
}

TrackingReport Tracking::TrackFrame(Frame& frame) {
  frame_counter_++;

  if (state_ == TrackingState::INITIALIZING) {
    orb_extractor_->Extract(frame);
    frame_handler_->BackProjectDepth(frame);
    Initialize(frame);
    report_.state = TrackingState::INITIALIZING;
    report_.tracking_time_ms = 0.0;
    report_.num_features = frame.keypoints.size();
    report_.inlier_ratio = 1.0;
    report_.num_matches = frame.keypoints.size();
    return report_;
  }

  orb_extractor_->Extract(frame);
  frame_handler_->BackProjectDepth(frame);

  TrackingReport rep;
  if (local_map_->NumKeyFrames() >= 3) {
    rep = TrackWithLocalMap(frame);
  } else {
    rep = TrackFrameWithPnP(frame);
  }

  UpdateLocalCostmap();
  pose_interpolator_->PushPose(current_pose_);
  return rep;
}

TrackingReport Tracking::StepHabitat(const cv::Mat& rgb, const cv::Mat& depth, double timestamp) {
  Frame frame;
  frame.rgb = rgb.clone();
  cv::cvtColor(rgb, frame.gray, cv::COLOR_BGR2GRAY);
  frame.depth = depth.clone();
  frame.timestamp = timestamp;
  return TrackFrame(frame);
}

void Tracking::UpdateLocalCostmap() {
  const auto& kfs = local_map_->GetAllKeyframes();
  std::vector<Vec3> active_mps;
  int last_id = -1;
  for (const auto& [id, _] : kfs) if (id > last_id) last_id = id;
  for (int id = std::max(0, last_id - 5); id <= last_id; ++id) {
    auto* kf = local_map_->GetKeyFrame(id);
    if (!kf) continue;
    for (int mp_id : kf->map_points) {
      if (mp_id < 0) continue;
      auto* mp = local_map_->GetMapPoint(mp_id);
      if (mp) active_mps.push_back(mp->position);
    }
  }
  if (!active_mps.empty()) {
    local_costmap_->Update(current_pose_, active_mps);
  }
}

void Tracking::RunNCIO(double dt, const Vec3& accel, const Vec3& gyro) {
  ncio_->Update(accel, gyro, dt);
  last_imu_timestamp_ += dt;
}

bool Tracking::CheckDegenerateCState() {
  if (ncio_->CheckDegenerateCTrigger()) {
    state_ = TrackingState::TRACKING_DEGENERATE_C;
    return true;
  }
  return false;
}

Mat6 Tracking::ComputeGeometricHessian(
  const std::vector<cv::DMatch>& matches,
  const Frame& ref_frame,
  const Frame& cur_frame) {
  Mat6 H = Mat6::Zero();
  if (matches.empty()) return H;

  double fx = 525.0, fy = 525.0;
  int valid = 0;

  for (const auto& m : matches) {
    int ref_idx = m.queryIdx;
    int cur_idx = m.trainIdx;

    if (ref_idx >= (int)ref_frame.keypoints_3d.size()) continue;
    if (ref_frame.keypoints_3d[ref_idx].norm() < 1e-6) continue;

    Vec3 world_pt = ref_frame.pose * ref_frame.keypoints_3d[ref_idx];
    Vec3 pt3d = current_pose_.inverse() * world_pt;
    if (pt3d(2) < 0.1) continue;

    double x = pt3d(0) / pt3d(2);
    double y = pt3d(1) / pt3d(2);

    MatX J(2, 6);
    J << fx / pt3d(2), 0, -fx * x / pt3d(2), -fx * x * y, fx * (1 + x * x), -fx * y,
         0, fy / pt3d(2), -fy * y / pt3d(2), -fy * (1 + y * y), fy * x * y, fy * x;

    H += J.transpose() * J;
    valid++;
  }

  if (valid < 6) {
    H += Mat6::Identity() * 1e-3;
  }
  return H;
}

VecX Tracking::ComputeResidualVector(
  const std::vector<cv::DMatch>& matches,
  const Frame& ref_frame,
  const Frame& cur_frame) {
  if (matches.empty()) return VecX::Zero(1);

  double fx = 525.0, fy = 525.0;
  double cx = 319.5, cy = 239.5;
  std::vector<double> res;

  for (const auto& m : matches) {
    int ref_idx = m.queryIdx;
    int cur_idx = m.trainIdx;

    if (ref_idx >= (int)ref_frame.keypoints_3d.size() ||
        cur_idx >= (int)cur_frame.keypoints.size()) continue;
    if (ref_frame.keypoints_3d[ref_idx].norm() < 1e-6) continue;

    Vec3 world_pt = ref_frame.pose * ref_frame.keypoints_3d[ref_idx];
    Vec3 pt3d = current_pose_.inverse() * world_pt;
    if (pt3d(2) < 0.1) continue;

    double u_proj = fx * pt3d(0) / pt3d(2) + cx;
    double v_proj = fy * pt3d(1) / pt3d(2) + cy;

    res.push_back(u_proj - cur_frame.keypoints[cur_idx].pt.x);
    res.push_back(v_proj - cur_frame.keypoints[cur_idx].pt.y);
  }

  VecX residuals(res.size());
  for (size_t i = 0; i < res.size(); ++i) residuals(i) = res[i];
  return residuals;
}

bool Tracking::DetectLoopCorrection() {
  if (kf_history_.size() < 4) {
    return false;
  }
  if (last_frame_.descriptors.empty()) {
    return false;
  }
  if (!loop_closer_ || !loop_closer_->HasDatabase()) {
    return false;
  }

  int current_kf_id = last_keyframe_id_;

  // Step 1: Use LoopCloser for BoW-based loop candidate detection
  auto candidates = loop_closer_->DetectCandidates(current_kf_id, 0.001f, 3); // Lowered from 0.05 to 0.001
  if (candidates.empty()) return false;

  // Step 2: Geometric verification on top candidates
  int best_score = 0;
  int best_kf_id = -1;
  SE3 best_loop_pose;
  
  for (const auto& cand : candidates) {
    SE3 loop_pose;
    // Get the past KF from local_map_
    const auto* past_kf = local_map_->GetKeyFrame(cand.match_id);
    if (!past_kf) continue;
    
    Frame pseudo_frame;
    pseudo_frame.keypoints = past_kf->keypoints;
    pseudo_frame.descriptors = past_kf->descriptors.clone();
    pseudo_frame.keypoints_3d.resize(pseudo_frame.keypoints.size(), Vec3::Zero());
    
    for (size_t i = 0; i < past_kf->map_points.size() && i < pseudo_frame.keypoints.size(); ++i) {
      int mp_id = past_kf->map_points[i];
      auto* mp = local_map_->GetMapPoint(mp_id);
      if (mp) {
        pseudo_frame.keypoints_3d[i] = mp->position;
      }
    }
    
    if (loop_closer_->GeometricVerification(current_kf_id, cand.match_id, last_frame_, pseudo_frame, loop_pose)) {
      std::cout << "[LOOP] KF" << cand.match_id << "→" << current_kf_id
                << " score=" << cand.similarity_score << "\n";
      best_kf_id = cand.match_id;
      best_loop_pose = loop_pose;
      break;
    }
  }

  if (best_kf_id < 0) return false;

  // Add loop constraint to pose graph
  // Umeyama T_q_m = T_q_cam^{-1} * T_m_cam transforms match→query camera frame
  // PoseGraph3DError: error = measured^{-1} * computed_rel
  //   where measured = edge.relative_pose
  //         computed_rel = T_from^{-1} * T_to
  // When from=match and to=query: computed_rel = T_m^{-1} * T_q
  //   measured should = T_m^{-1} * T_q
  // T_q_m.inverse() = T_m^{-1} * T_q → correct!
  SE3 rel_loop = best_loop_pose.inverse();
  Mat6 info = PoseGraph::DefaultInformation(50.0, 50.0);

  pose_graph_->AddEdge(best_kf_id, current_kf_id, rel_loop, info);

  // Add loop edge to topological graph (weak weight)
  topo_graph_->AddEdge(best_kf_id, current_kf_id, rel_loop, 0.2, true);

  // Notify pose interpolator to start smooth transition
  pose_interpolator_->NotifyLoopClosure();

  num_loops_detected_++;

  return true;
}

bool Tracking::MatchWithPastKeyFrame(int past_kf_id, const Frame& current_frame,
                                      SE3& loop_pose) {
  // Find the stored KF data
  StoredKeyFrame target_skf;
  bool found = false;
  for (const auto& skf : kf_history_) {
    if (skf.id == past_kf_id) {
      target_skf = skf;
      found = true;
      break;
    }
  }
  if (!found || target_skf.descriptors.empty()) return false;

  // Match current frame descriptors against past KF descriptors
  cv::BFMatcher matcher(cv::NORM_HAMMING, false);
  std::vector<std::vector<cv::DMatch>> knn;
  matcher.knnMatch(current_frame.descriptors, target_skf.descriptors, knn, 2);

  std::vector<cv::DMatch> good_matches;
  for (size_t i = 0; i < knn.size(); ++i) {
    if (knn[i].size() >= 2 &&
        knn[i][0].distance < 0.75f * knn[i][1].distance) {
      good_matches.push_back(knn[i][0]);
    }
  }

  if (good_matches.size() < 15) return false;

  // Build 3D-2D correspondences: world points from the past KF's map
  const auto* past_keyframe = local_map_->GetKeyFrame(past_kf_id);
  if (!past_keyframe) return false;

  Frame pseudo_frame;
  pseudo_frame.keypoints = past_keyframe->keypoints;
  pseudo_frame.descriptors = past_keyframe->descriptors.clone();
  pseudo_frame.keypoints_3d.resize(pseudo_frame.keypoints.size(), Vec3::Zero());

  // Use actual map points: the past KF's map_points store 3D world positions
  for (size_t i = 0; i < past_keyframe->map_points.size() &&
       i < pseudo_frame.keypoints.size(); ++i) {
    int mp_id = past_keyframe->map_points[i];
    auto* mp = local_map_->GetMapPoint(mp_id);
    if (mp) {
      pseudo_frame.keypoints_3d[i] = mp->position;
    }
  }

  // Use our PnP solver
  // Flip match indices: EstimatePose expects queryIdx=ref_frame(pseudo), trainIdx=cur_frame(current)
  std::vector<cv::DMatch> flipped_matches;
  flipped_matches.reserve(good_matches.size());
  for (const auto& m : good_matches) {
    flipped_matches.emplace_back(m.trainIdx, m.queryIdx, m.distance);
  }
  PnPResult result = pnp_solver_->EstimatePose(
    flipped_matches, pseudo_frame, current_frame, current_pose_);

  if (result.success && result.inlier_ratio > 0.25) {
    loop_pose = result.pose;

    // Refine with only inliers
    if (result.inlier_matches.size() >= 10) {
      std::vector<Match3D2D> corrs;
      for (const auto& m : result.inlier_matches) {
        int ref_idx = m.queryIdx;
        if (ref_idx < (int)pseudo_frame.keypoints_3d.size() &&
            pseudo_frame.keypoints_3d[ref_idx].norm() > 1e-6) {
          corrs.push_back({
            pseudo_frame.keypoints_3d[ref_idx],
            cv::Point2f(current_frame.keypoints[m.trainIdx].pt.x,
                         current_frame.keypoints[m.trainIdx].pt.y),
            ref_idx, m.trainIdx
          });
        }
      }
      if (corrs.size() >= 10) {
        pnp_solver_->RefinePoseWithInliers(loop_pose, corrs);
      }
    }
    return true;
  }
  return false;
}

void Tracking::DistributeLoopCorrection(int from_kf_id, const SE3& delta_pose) {
  // Apply correction to this KF and all subsequent KFs in history
  bool apply = false;
  for (auto& skf : kf_history_) {
    if (skf.id == from_kf_id) {
      apply = true;
    }
    if (apply) {
      skf.pose = skf.pose * delta_pose;
      local_map_->UpdateKeyFramePose(skf.id, skf.pose);
    }
  }

  // Calculate smoothed correction: distribute 70% to current, rest to all
  SE3 smooth_correction = SE3(
    Quat::Identity().slerp(0.7, delta_pose.rot),
    delta_pose.trans * 0.7);

  current_pose_ = current_pose_ * smooth_correction;
  last_pose_ = last_pose_ * smooth_correction;
  ref_frame_.pose = current_pose_;
}

} // namespace st_slam
