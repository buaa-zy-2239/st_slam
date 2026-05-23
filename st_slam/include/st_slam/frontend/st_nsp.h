#ifndef ST_SLAM_ST_NSP_H
#define ST_SLAM_ST_NSP_H

#include "st_slam/core/types.h"
#include "st_slam/core/math_utils.h"

namespace st_slam {

class STNSP {
public:
  explicit STNSP(double beta = 10.0, double tau1 = 0.1,
                  double degeneracy_threshold = 0.05);

  DegeneracyState AnalyzeHessian(const Mat6& H_geo, double& condition_number);

  Mat6 ComputeSafeProjection(const Mat6& H_geo, DegeneracyState state);

  Vec6 ProjectIncrement(const Vec6& raw_increment,
                         const Mat6& P_safe,
                         DegeneracyState state);

  double GetDegeneracyMeasure() const { return degeneracy_measure_; }

  void SetBeta(double b) { beta_ = b; }
  void SetTau1(double t) { tau1_ = t; }

private:
  double beta_;
  double tau1_;
  double degeneracy_threshold_;
  double degeneracy_measure_;
  Vec6 eigenvalues_;
};

} // namespace st_slam

#endif
