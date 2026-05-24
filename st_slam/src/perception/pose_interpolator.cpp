#include "st_slam/perception/pose_interpolator.h"
#include <iostream>

namespace st_slam {

PoseInterpolator::PoseInterpolator(int recovery_frames, double max_jump_m)
  : recovery_frames_(recovery_frames), max_jump_m_(max_jump_m) {}

void PoseInterpolator::PushPose(const SE3& global_pose) {
  last_global_pose_ = global_pose;

  if (state_ == InterpolatorState::NORMAL) {
    // no-op
  }
  else if (state_ == InterpolatorState::LOOP_TRIGGERED) {
    post_loop_pose_ = global_pose;
    recovery_counter_ = recovery_frames_;
    state_ = InterpolatorState::SMOOTH_RECOVERY;
  }
  else if (state_ == InterpolatorState::SMOOTH_RECOVERY) {
    recovery_counter_--;
    if (recovery_counter_ <= 0) {
      state_ = InterpolatorState::NORMAL;
    }
  }
}

void PoseInterpolator::NotifyLoopClosure() {
  if (state_ != InterpolatorState::NORMAL) return;
  pre_loop_pose_ = last_global_pose_;
  state_ = InterpolatorState::LOOP_TRIGGERED;
}

SE3 PoseInterpolator::GetSmoothPose() const {
  if (state_ == InterpolatorState::NORMAL) {
    return last_global_pose_;
  }
  if (state_ == InterpolatorState::LOOP_TRIGGERED) {
    return pre_loop_pose_;
  }

  int elapsed = recovery_frames_ - recovery_counter_;
  double alpha = (double)(elapsed + 1) / (double)(recovery_frames_ + 1);
  return Interpolate(pre_loop_pose_, post_loop_pose_, alpha);
}

SE3 PoseInterpolator::Interpolate(const SE3& from, const SE3& to, double alpha) const {
  alpha = std::max(0.0, std::min(1.0, alpha));
  Quat q = from.rot.slerp(alpha, to.rot);
  Vec3 t = from.trans * (1.0 - alpha) + to.trans * alpha;
  return SE3(q, t);
}

} // namespace st_slam
