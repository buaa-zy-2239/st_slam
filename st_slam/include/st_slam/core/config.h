#ifndef ST_SLAM_CONFIG_H
#define ST_SLAM_CONFIG_H

#include <string>
#include <memory>

namespace st_slam {

struct STSLAMConfig {
  // === Frontend ===
  // ST-NSP
  double nsp_degeneracy_threshold = 0.05;
  double nsp_beta = 10.0;
  double nsp_tau1 = 0.1;

  // NCIO
  double ncio_kinetic_noise = 0.01;
  double ncio_shock_gain = 0.1;
  double ncio_covariance_max_det = 1e6;
  double ncio_covariance_threshold = 100.0;

  // T-PVD
  double tpvd_eta = 0.5;
  double tpvd_gamma_init = 1.0;
  double tpvd_temporal_threshold = 0.1;

  // A-CSG
  double acsg_mu0 = 0.01;
  double acsg_alpha_init = 0.1;
  double acsg_alpha_max_multiplier = 10.0;
  double acsg_delta = 0.1;
  double acsg_c_default = 1.345;

  // === Mid-End ===
  // VB-KS
  int vbks_voxel_resolution = 8;
  double vbks_voxel_size = 0.2;
  double vbks_boundary_delta = 0.05;

  // TDQ-BS
  double tdq_bs_blending_threshold = 0.02;

  // Deformation
  double deform_regularization_weight = 0.1;
  double deform_epsilon = 1e-8;

  // === Backend ===
  // Spectral Heat Kernel
  double shk_ln_t_min = -5.0;
  double shk_ln_t_max = 5.0;
  int shk_num_moments = 3;
  int shk_num_eigenvalues = 30;

  // L-SCOV
  int lscov_num_eigenvectors = 4;
  double lscov_spectral_noise_std = 0.05;
  double lscov_hamming_threshold = 0.2;

  // Loop Detector
  int loop_min_keyframe_gap = 50;
  double loop_similarity_threshold = 0.6;

  // === System ===
  int num_threads = 4;
  int core_frontend_start = 0;
  int core_frontend_end = 1;
  int core_midend_start = 2;
  int core_midend_end = 4;
  int core_backend = 5;

  bool enable_h_mbt = true;
  double h_mbt_bucket_size_bytes = 64.0;
  double h_mbt_max_l3_usage = 0.15;

  size_t spsc_queue_size = 32;

  // === Dataset ===
  std::string dataset_path;
  std::string association_file = "associations.txt";
  int start_frame = 0;
  int max_frames = -1;

  static STSLAMConfig& Instance() {
    static STSLAMConfig config;
    return config;
  }
};

} // namespace st_slam

#endif
