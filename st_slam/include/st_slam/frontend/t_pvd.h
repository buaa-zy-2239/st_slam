#ifndef ST_SLAM_T_PVD_H
#define ST_SLAM_T_PVD_H

#include "st_slam/core/types.h"
#include <deque>

namespace st_slam {

class TPVD {
public:
  explicit TPVD(double eta = 0.5, double gamma_init = 1.0,
                double temporal_threshold = 0.1,
                size_t history_window = 10);

  double ComputeModulationFactor(const Frame& current_frame,
                                  double frame_timestamp);

  double GetCurrentGamma() const { return gamma_current_; }

  void Reset();

private:
  double eta_;
  double gamma_init_;
  double gamma_current_;
  double temporal_threshold_;
  size_t history_window_;

  std::deque<double> residual_history_;
  std::deque<double> timestamp_history_;

  double ComputeTemporalDerivative() const;
};

} // namespace st_slam

#endif
