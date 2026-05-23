#include "st_slam/frontend/a_csg.h"
#include <Eigen/SVD>
#include <cmath>

namespace st_slam {

ACSG::ACSG(double mu0, double alpha_init,
            double alpha_max_multiplier, double delta, double c_default)
  : mu0_(mu0), mu_current_(mu0), alpha_current_(alpha_init),
    alpha_max_multiplier_(alpha_max_multiplier),
    alpha_max_(alpha_init * alpha_max_multiplier),
    delta_(delta), c_default_(c_default), c_current_(c_default) {}

void ACSG::Reset() {
  mu_current_ = mu0_;
  alpha_current_ = alpha_current_;
  c_current_ = c_default_;
}

double ACSG::ComputeFrobeniusScale(const MatX& J) const {
  return std::sqrt((J * J.transpose()).trace());
}

double ACSG::ComputeDampingMu(const MatX& J, const VecX& residual,
                               double mu_previous) {
  double frob_norm = ComputeFrobeniusScale(J);
  double resid_norm = residual.norm();

  double psi_val = c_current_ * c_current_ *
                    std::log(1.0 + resid_norm * resid_norm /
                                   (c_current_ * c_current_));

  mu_current_ = mu0_ + alpha_current_ * frob_norm * frob_norm * psi_val;

  mu_current_ = std::clamp(mu_current_, mu0_,
                            mu0_ * alpha_max_multiplier_);

  return mu_current_;
}

double ACSG::ComputeCauchyScale(const MatX& H_geo, const VecX& residual) {
  Eigen::JacobiSVD<MatX> svd(H_geo, Eigen::ComputeThinU | Eigen::ComputeThinV);
  const auto& singular_values = svd.singularValues();

  double kappa = singular_values(0) /
                 std::max(singular_values(singular_values.size() - 1), 1e-12);

  double trace_H = H_geo.trace();
  double r0_norm = residual.norm();

  double num = 2.0 * kappa * r0_norm;
  double den = delta_ * std::max(trace_H, 1e-12);
  double val = num / std::max(den, 1e-12);

  c_current_ = std::max(std::sqrt(val), c_default_ * 0.5);
  c_current_ = std::clamp(c_current_, c_default_ * 0.1, c_default_ * 10.0);

  return c_current_;
}

} // namespace st_slam
