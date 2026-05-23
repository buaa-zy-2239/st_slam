#ifndef ST_SLAM_TRACKING_H
#define ST_SLAM_TRACKING_H

#include "st_slam/core/types.h"
#include "st_slam/core/config.h"
#include "st_slam/frontend/orb_extractor.h"
#include "st_slam/frontend/frame.h"
#include "st_slam/frontend/st_nsp.h"
#include "st_slam/frontend/ncio.h"
#include "st_slam/frontend/t_pvd.h"
#include "st_slam/frontend/a_csg.h"
#include "st_slam/backend/pose_graph.h"
#include "st_slam/frontend/pnp_solver.h"
#include "st_slam/frontend/local_map.h"
#include <memory>

namespace st_slam {

class Tracking {
public:
  explicit Tracking(const STSLAMConfig& config);

  TrackingReport TrackFrame(Frame& frame);

  void SetIMUData(double timestamp, const Vec3& accel, const Vec3& gyro);

  void Reset();

  SE3 GetCurrentPose() const { return current_pose_; }
  TrackingState GetState() const { return state_; }

  Frame& GetLastFrame() { return last_frame_; }
  const Frame& GetLastFrame() const { return last_frame_; }
  Frame& GetReferenceFrame() { return ref_frame_; }

  LocalMap& GetLocalMap() { return *local_map_; }

private:
  STSLAMConfig config_;
  TrackingState state_;
  TrackingReport report_;
  SE3 current_pose_;
  SE3 last_pose_;
  Frame last_frame_;
  Frame ref_frame_;
  int frame_counter_;

  std::unique_ptr<ORBExtractor> orb_extractor_;
  std::unique_ptr<FrameHandler> frame_handler_;
  std::unique_ptr<STNSP> st_nsp_;
  std::unique_ptr<NCIO> ncio_;
  std::unique_ptr<TPVD> t_pvd_;
  std::unique_ptr<ACSG> a_csg_;
  std::unique_ptr<PnPSolver> pnp_solver_;
  std::unique_ptr<LocalMap> local_map_;
  std::unique_ptr<PoseGraph> pose_graph_;

  std::vector<Vec3> imu_accel_buffer_;
  std::vector<Vec3> imu_gyro_buffer_;

  int last_keyframe_id_ = -1;
  int last_near_kf_id_ = -1;
  double last_imu_timestamp_ = 0;
  Vec3 last_velocity_ = Vec3::Zero();
  bool has_imu_integration_ = false;
  Vec3 gravity_cam_frame_ = Vec3(0, 9.81, 0);
  std::deque<Vec3> accel_buffer_;

  bool Initialize(const Frame& frame);

  TrackingReport TrackFrameWithPnP(Frame& frame);

  TrackingReport TrackWithLocalMap(Frame& frame);

  TrackingReport Relocalize(Frame& frame);

  SE3 PredictPoseFromIMU(double dt, const Vec3& accel, const Vec3& gyro);

  SE3 PredictPoseFromMotionModel() const;

  SE3 AlignFramePnP(const Frame& ref_frame, const Frame& cur_frame,
                     const std::vector<cv::DMatch>& matches,
                     const SE3& initial_guess);

  bool TryCreateKeyFrame(Frame& frame);

  int CreateMapPointsFromKeyFrame(int kf_id);

  bool CheckDegenerateCState();

  Mat6 ComputeGeometricHessian(const std::vector<cv::DMatch>& matches,
                                const Frame& ref_frame,
                                const Frame& cur_frame);

  VecX ComputeResidualVector(const std::vector<cv::DMatch>& matches,
                              const Frame& ref_frame,
                              const Frame& cur_frame);

  void RunNCIO(double dt, const Vec3& accel, const Vec3& gyro);

  void UpdateGravityFromAccel(const Vec3& accel);
  void ApplyGravityConstraint(SE3& pose);

  // Loop closure via spatial revisit + geometric verification
  struct StoredKeyFrame {
    int id;
    SE3 pose;
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    double timestamp;
  };
  std::deque<StoredKeyFrame> kf_history_;

  bool DetectLoopCorrection();
  bool MatchWithPastKeyFrame(int past_kf_id, const Frame& current_frame,
                              SE3& loop_pose);
  void DistributeLoopCorrection(int from_kf_id, const SE3& delta_pose);
};

} // namespace st_slam

#endif
