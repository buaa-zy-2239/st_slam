#ifndef ST_SLAM_TYPES_H
#define ST_SLAM_TYPES_H

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/StdVector>
#include <vector>
#include <memory>
#include <opencv2/core.hpp>

namespace st_slam {

using Vec2 = Eigen::Vector2d;
using Vec3 = Eigen::Vector3d;
using Vec4 = Eigen::Vector4d;
using Vec6 = Eigen::Matrix<double, 6, 1>;
using VecX = Eigen::VectorXd;

using Mat3 = Eigen::Matrix3d;
using Mat4 = Eigen::Matrix4d;
using Mat6 = Eigen::Matrix<double, 6, 6>;
using MatX = Eigen::MatrixXd;

using Quat = Eigen::Quaterniond;
using Isometry3 = Eigen::Isometry3d;
using Affine3 = Eigen::Affine3d;
using AngleAxis = Eigen::AngleAxisd;

struct SE3 {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Quat rot;
  Vec3 trans;

  SE3() : rot(Quat::Identity()), trans(Vec3::Zero()) {}
  SE3(const Quat& r, const Vec3& t) : rot(r), trans(t) {}
  explicit SE3(const Mat4& m) : rot(Quat(m.block<3,3>(0,0))), trans(m.block<3,1>(0,3)) {}

  Mat4 toMatrix() const {
    Mat4 m = Mat4::Identity();
    m.block<3,3>(0,0) = rot.toRotationMatrix();
    m.block<3,1>(0,3) = trans;
    return m;
  }

  static SE3 Identity() { return SE3(); }

  SE3 inverse() const {
    return SE3(rot.conjugate(), -(rot.conjugate() * trans));
  }

  SE3 operator*(const SE3& other) const {
    return SE3(rot * other.rot, rot * other.trans + trans);
  }

  Vec3 operator*(const Vec3& p) const {
    return rot * p + trans;
  }
};

struct Point3D {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Vec3 position;
  Vec3 normal;
  Vec3 color;
  int id;
  float weight;
};

struct Surfel {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Vec3 position;
  Vec3 normal;
  Vec3 color;
  float radius;
  float confidence;
  int voxel_id;
  int timestamp;
};

struct KeyPoint {
  cv::KeyPoint kp;
  cv::Mat descriptor;
  int octave;
  float response;
  Vec3 world_pos;
  int landmark_id;
};

struct Frame {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  int id;
  double timestamp;
  cv::Mat rgb;
  cv::Mat depth;
  cv::Mat gray;

  std::vector<cv::KeyPoint> keypoints;
  cv::Mat descriptors;
  std::vector<Vec3> keypoints_3d;

  SE3 pose;

  std::vector<float> photometric_errors;
  double exposure_time;
};

struct Landmark {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Vec3 position;
  Vec3 normal;
  cv::Mat descriptor;
  std::vector<int> observations;
  int id;
  int ref_frame_id;
};

struct VoxelControlNode {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Vec3 center;
  Vec3 position;
  Quat rotation;
  std::vector<int> surfel_ids;
  int grid_idx[3];
  bool active;
};

enum class TrackingState {
  INITIALIZING,
  TRACKING_GOOD,
  TRACKING_DEGRADED,
  TRACKING_LOST,
  TRACKING_DEGENERATE_C
};

enum class DegeneracyState {
  NONE,
  PARTIAL,
  FULL_DEGENERATE
};

struct TrackingReport {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  TrackingState state;
  DegeneracyState degeneracy;
  double hessian_condition_number;
  double inlier_ratio;
  double photometric_gradient;
  double covariance_det;
  double tracking_time_ms;
  int num_features;
  int num_matches;
  double mu_damping;
  double cauchy_scale;
};

struct MappingConfig {
  int grid_resolution = 8;
  double voxel_size = 0.1;
  double surfel_radius = 0.02;
  double depth_min = 0.1;
  double depth_max = 10.0;
  int min_keypoint_distance = 20;
  double max_reprojection_error = 3.0;
};

struct LoopCandidate {
  int query_id;
  int match_id;
  double similarity_score;
  double spectral_distance;
  Mat4 relative_transform;
  bool verified;
};

using KeyFramePtr = std::shared_ptr<Frame>;
using LandmarkPtr = std::shared_ptr<Landmark>;

} // namespace st_slam

#endif
