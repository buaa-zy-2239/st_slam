#include "st_slam/frontend/t_pvd.h"
#include "st_slam/core/math_utils.h"
#include <cmath>
#include <algorithm>

namespace st_slam {

TPVD::TPVD(double eta, double gamma_init, double temporal_threshold, size_t history_window)
  : eta_(eta), gamma_init_(gamma_init), gamma_current_(gamma_init),
    temporal_threshold_(temporal_threshold),
    history_window_(history_window) {}

void TPVD::Reset() {
  gamma_current_ = gamma_init_;
  residual_history_.clear();
  timestamp_history_.clear();
}

double TPVD::ComputeModulationFactor(const Frame& current_frame,
                                      double frame_timestamp) {
  double mean_error = 0.0;
  if (!current_frame.photometric_errors.empty()) {
    double sum = 0.0;
    for (float e : current_frame.photometric_errors) {
      if (std::isfinite(e) && e < 1e6f) {
        sum += e;
      }
    }
    mean_error = sum / current_frame.photometric_errors.size();
  }

  residual_history_.push_back(mean_error);
  timestamp_history_.push_back(frame_timestamp);

  if (residual_history_.size() > history_window_) {
    residual_history_.pop_front();
    timestamp_history_.pop_front();
  }

  double et_dot = ComputeTemporalDerivative();

  gamma_current_ = gamma_init_ * std::exp(-eta_ * std::abs(et_dot));

  return gamma_current_;
}

double TPVD::ComputeTemporalDerivative() const {
  if (residual_history_.size() < 3) return 0.0;

  std::vector<double> residuals(residual_history_.begin(), residual_history_.end());
  return math::total_variation_derivative(residuals, 1);
}

} // namespace st_slam
