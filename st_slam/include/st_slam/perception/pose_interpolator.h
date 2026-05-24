#ifndef ST_SLAM_POSE_INTERPOLATOR_H
#define ST_SLAM_POSE_INTERPOLATOR_H

#include "st_slam/core/types.h"
#include <deque>

namespace st_slam {

enum class InterpolatorState {
  NORMAL,
  LOOP_TRIGGERED,
  SMOOTH_RECOVERY
};

class PoseInterpolator {
public:
  PoseInterpolator(int recovery_frames = 5, double max_jump_m = 0.5);

  void PushPose(const SE3& global_pose);
  void NotifyLoopClosure();

  SE3 GetSmoothPose() const;
  InterpolatorState GetState() const { return state_; }
  int RecoveryFramesRemaining() const { return recovery_counter_; }

private:
  InterpolatorState state_ = InterpolatorState::NORMAL;
  int recovery_frames_;
  double max_jump_m_;
  int recovery_counter_ = 0;

  SE3 last_global_pose_;
  SE3 pre_loop_pose_;
  SE3 post_loop_pose_;

  SE3 Interpolate(const SE3& from, const SE3& to, double alpha) const;
};

} // namespace st_slam

#endif
