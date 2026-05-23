#ifndef ST_SLAM_A_CSG_H
#define ST_SLAM_A_CSG_H

#include "st_slam/core/types.h"
#include "st_slam/core/math_utils.h"

namespace st_slam {

class ACSG {
public:
  explicit ACSG(double mu0 = 0.01, double alpha_init = 0.1,
                double alpha_max_multiplier = 10.0,
                double delta = 0.1, double c_default = 1.345);

  double ComputeDampingMu(const MatX& J, const VecX& residual,
                           double mu_previous);

  double ComputeCauchyScale(const MatX& H_geo, const VecX& residual);

  double GetCurrentC() const { return c_current_; }
  double GetCurrentMu() const { return mu_current_; }

  void Reset();

private:
  double mu0_;
  double mu_current_;
  double alpha_current_;
  double alpha_max_multiplier_;
  double alpha_max_;
  double delta_;
  double c_default_;
  double c_current_;

  double ComputeFrobeniusScale(const MatX& J) const;
};

} // namespace st_slam

#endif
