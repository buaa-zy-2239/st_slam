#ifndef ST_SLAM_POSE_GRAPH_H
#define ST_SLAM_POSE_GRAPH_H

#include "st_slam/core/types.h"
#include "st_slam/frontend/local_map.h"
#include <vector>
#include <unordered_map>
#include <ceres/ceres.h>
#include <ceres/rotation.h>

namespace st_slam {

struct PoseGraphEdge {
  int from_id;
  int to_id;
  SE3 relative_pose;
  Mat6 information;
};

class PoseGraph {
public:
  explicit PoseGraph(double rot_info_weight = 100.0,
                     double trans_info_weight = 1000.0);

  void Clear();

  void AddEdge(int from_id, int to_id, const SE3& relative_pose,
               const Mat6& information);

  void BuildFromKeyframes(const std::unordered_map<int, KeyFrame>& keyframes);

  bool Optimize(std::unordered_map<int, KeyFrame>& keyframes);

  int NumEdges() const { return edges_.size(); }

  static Mat6 DefaultInformation(double rot_w = 100.0, double trans_w = 1000.0);

private:
  std::vector<PoseGraphEdge> edges_;
  double rot_info_weight_;
  double trans_info_weight_;

  struct PoseGraph3DError {
    PoseGraph3DError(const SE3& measured, const Mat6& sqrt_info)
      : measured_(measured), sqrt_info_(sqrt_info) {}

    template <typename T>
    bool operator()(const T* const p_from, const T* const p_to,
                    T* residuals) const {
      T q_from[4] = {p_from[0], p_from[1], p_from[2], p_from[3]};
      T t_from[3] = {p_from[4], p_from[5], p_from[6]};
      T q_to[4] = {p_to[0], p_to[1], p_to[2], p_to[3]};
      T t_to[3] = {p_to[4], p_to[5], p_to[6]};

      T q_from_inv[4] = {q_from[0], -q_from[1], -q_from[2], -q_from[3]};
      T diff[3] = {t_to[0] - t_from[0], t_to[1] - t_from[1], t_to[2] - t_from[2]};
      T t_rel[3];
      ceres::QuaternionRotatePoint(q_from_inv, diff, t_rel);

      T q_rel[4];
      ceres::QuaternionProduct(q_from_inv, q_to, q_rel);

      T m_q[4] = {T(measured_.rot.w()), T(measured_.rot.x()),
                   T(measured_.rot.y()), T(measured_.rot.z())};
      T m_t[3] = {T(measured_.trans(0)), T(measured_.trans(1)),
                   T(measured_.trans(2))};

      T q_error[4];
      ceres::QuaternionProduct(q_rel, m_q, q_error);

      T rot_res[3] = {T(2.0)*q_error[1], T(2.0)*q_error[2], T(2.0)*q_error[3]};
      T trans_res[3] = {t_rel[0]-m_t[0], t_rel[1]-m_t[1], t_rel[2]-m_t[2]};

      for (int i = 0; i < 6; ++i) {
        residuals[i] = T(0);
        for (int j = 0; j < 6; ++j) {
          residuals[i] += T(sqrt_info_(i,j)) *
            (j < 3 ? rot_res[j] : trans_res[j-3]);
        }
      }
      return true;
    }

    static ceres::CostFunction* Create(const SE3& measured, const Mat6& sqrt_info) {
      return new ceres::AutoDiffCostFunction<PoseGraph3DError, 6, 7, 7>(
        new PoseGraph3DError(measured, sqrt_info));
    }

  private:
    SE3 measured_;
    Mat6 sqrt_info_;
  };
};

} // namespace st_slam

#endif
