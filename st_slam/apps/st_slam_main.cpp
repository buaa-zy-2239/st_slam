#include "st_slam/core/types.h"
#include "st_slam/core/math_utils.h"
#include "st_slam/frontend/tracking.h"
#include "st_slam/frontend/orb_extractor.h"
#include "st_slam/backend/spectral_heat_kernel.h"
#include "st_slam/backend/l_scov.h"
#include "st_slam/backend/loop_detector.h"
#include "st_slam/system/spsc_queue.h"
#include "st_slam/system/thread_pool.h"
#include "st_slam/system/h_mbt.h"
#include "st_slam/system/slam_benchmark.h"
#include "st_slam/backend/pose_graph.h"
#include "dataset_reader.h"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <chrono>
#include <thread>
#include <vector>
#include <memory>
#include <numeric>

using namespace st_slam;

struct EvaluationMetrics {
  double ate_rmse;
  double ate_mean;
  double ate_median;
  double ate_std;

  double rpe_rmse_trans;
  double rpe_mean_trans;
  double rpe_median_trans;
  double rpe_std_trans;
  double rpe_rmse_rot_deg;
  double rpe_mean_rot_deg;

  double mean_tracking_time_ms;
  double max_tracking_time_ms;
  double min_tracking_time_ms;
  double std_tracking_time_ms;
  double p50_tracking_time_ms;
  double p95_tracking_time_ms;
  double p99_tracking_time_ms;

  int total_frames;
  int lost_frames;
  int partial_degraded_frames;
  int full_degenerate_frames;
  double tracking_success_rate;
  int num_loops_detected;

  double mean_condition_number;
  double mean_inlier_ratio;
  double mean_matches;
  double mean_cauchy_scale;
};

void PrintMetricsTable(const EvaluationMetrics& m,
                        const std::string& seq_display,
                        benchmark::SceneType scene) {
  benchmark::QualityLevel ql = benchmark::ClassifyATEQuality(m.ate_rmse, scene);

  std::cout << "\n=============================================================\n";
  std::cout << "     ST-SLAM 4.0 Comprehensive Evaluation\n";
  std::cout << "     " << seq_display;
  if (scene == benchmark::SceneType::DYNAMIC) std::cout << "  [DYNAMIC]";
  std::cout << "\n";
  std::cout << "=============================================================\n";

  std::cout << "\n--- Dataset & General ---\n";
  std::cout << "  Total Frames:            " << m.total_frames << "\n";
  std::cout << "  Tracking Success Rate:    " << std::fixed << std::setprecision(1)
            << m.tracking_success_rate * 100.0 << "%\n";
  std::cout << "  Lost / Degenerate-C:      " << m.lost_frames << "\n";

  std::cout << "\n--- ATE (Absolute Trajectory Error) ---\n";
  std::cout << "  RMSE:                    " << std::setprecision(4) << m.ate_rmse << " m\n";
  std::cout << "  Mean:                    " << std::setprecision(4) << m.ate_mean << " m\n";
  std::cout << "  Median:                  " << std::setprecision(4) << m.ate_median << " m\n";
  std::cout << "  Std:                     " << std::setprecision(4) << m.ate_std << " m\n";

  std::cout << "\n  Quality Assessment:       " << benchmark::QualityLabel(ql)
            << " " << benchmark::QualityColorEmoji(ql) << "\n";

  std::cout << "\n--- RPE (Relative Pose Error) ---\n";
  std::cout << "  Translation RMSE:        " << std::setprecision(4) << m.rpe_rmse_trans << " m\n";
  std::cout << "  Translation Mean:        " << std::setprecision(4) << m.rpe_mean_trans << " m\n";
  std::cout << "  Translation Median:      " << std::setprecision(4) << m.rpe_median_trans << " m\n";
  std::cout << "  Translation Std:         " << std::setprecision(4) << m.rpe_std_trans << " m\n";
  std::cout << "  Rotation RMSE:           " << std::setprecision(3) << m.rpe_rmse_rot_deg << " deg\n";
  std::cout << "  Rotation Mean:           " << std::setprecision(3) << m.rpe_mean_rot_deg << " deg\n";

  std::cout << "\n--- Timing Performance (CPU only) ---\n";
  std::cout << "  Mean:                    " << std::setprecision(2) << m.mean_tracking_time_ms << " ms\n";
  std::cout << "  P50 (Median):            " << std::setprecision(2) << m.p50_tracking_time_ms << " ms\n";
  std::cout << "  Std:                     " << std::setprecision(2) << m.std_tracking_time_ms << " ms\n";
  std::cout << "  Min:                     " << std::setprecision(2) << m.min_tracking_time_ms << " ms\n";
  std::cout << "  Max:                     " << std::setprecision(2) << m.max_tracking_time_ms << " ms\n";
  std::cout << "  P95:                     " << std::setprecision(2) << m.p95_tracking_time_ms << " ms\n";
  std::cout << "  P99:                     " << std::setprecision(2) << m.p99_tracking_time_ms << " ms\n";

  std::cout << "\n--- Degeneracy Statistics ---\n";
  std::cout << "  Partial Degraded Frames: " << m.partial_degraded_frames << "\n";
  std::cout << "  Fully Degenerate Frames: " << m.full_degenerate_frames << "\n";
  std::cout << "  Mean Condition Number:   " << std::setprecision(2) << m.mean_condition_number << "\n";

  std::cout << "\n--- Feature & Optimization ---\n";
  std::cout << "  Mean Inlier Ratio:       " << std::setprecision(2) << m.mean_inlier_ratio << "\n";
  std::cout << "  Mean Matches:            " << std::setprecision(1) << m.mean_matches << "\n";
  std::cout << "  Mean Cauchy Scale:       " << std::setprecision(3) << m.mean_cauchy_scale << "\n";

  std::cout << "\n--- Loop Closure ---\n";
  std::cout << "  Loops Detected:          " << m.num_loops_detected << "\n";
  std::cout << "=============================================================\n";
}

EvaluationMetrics Evaluate(
  const std::vector<SE3>& est_poses,
  const std::vector<double>& est_timestamps,
  const std::vector<double>& tracking_times,
  const std::vector<TrackingReport>& reports,
  const TUMDatasetReader& reader,
  int total_frames, int lost_frames,
  int loops_detected) {
  EvaluationMetrics m;
  m.total_frames = total_frames;
  m.lost_frames = lost_frames;
  m.num_loops_detected = loops_detected;
  m.tracking_success_rate = total_frames > 0 ?
    1.0 - static_cast<double>(lost_frames) / total_frames : 0.0;

  if (est_poses.size() < 2) {
    m.ate_rmse = -1; m.ate_mean = -1; m.ate_median = -1; m.ate_std = -1;
    m.rpe_rmse_trans = -1; m.rpe_mean_trans = -1; m.rpe_median_trans = -1; m.rpe_std_trans = -1;
    m.rpe_rmse_rot_deg = -1; m.rpe_mean_rot_deg = -1;
    m.mean_tracking_time_ms = 0; m.max_tracking_time_ms = 0; m.min_tracking_time_ms = 0;
    m.std_tracking_time_ms = 0; m.p50_tracking_time_ms = 0; m.p95_tracking_time_ms = 0; m.p99_tracking_time_ms = 0;
    m.partial_degraded_frames = 0; m.full_degenerate_frames = 0;
    m.mean_condition_number = 0; m.mean_inlier_ratio = 0; m.mean_matches = 0; m.mean_cauchy_scale = 0;
    return m;
  }

  std::vector<SE3> gt_poses;
  gt_poses.reserve(est_timestamps.size());
  for (double ts : est_timestamps) {
    gt_poses.push_back(reader.GetGroundTruth(ts));
  }

  std::vector<SE3> aligned_est = est_poses;
  math::align_trajectory(aligned_est, gt_poses);

  m.ate_rmse = math::compute_ate_rmse(aligned_est, gt_poses);
  m.ate_mean = math::compute_ate_mean(aligned_est, gt_poses);
  m.ate_median = math::compute_ate_median(aligned_est, gt_poses);
  m.ate_std = math::compute_ate_std(aligned_est, gt_poses);

  math::RPEMetrics rpe = math::compute_rpe_full(aligned_est, gt_poses, est_timestamps);
  m.rpe_rmse_trans = rpe.rmse_trans;
  m.rpe_mean_trans = rpe.mean_trans;
  m.rpe_median_trans = rpe.median_trans;
  m.rpe_std_trans = rpe.std_trans;
  m.rpe_rmse_rot_deg = rpe.rmse_rot_deg;
  m.rpe_mean_rot_deg = rpe.mean_rot_deg;

  if (!tracking_times.empty()) {
    size_t n = tracking_times.size();
    double sum = std::accumulate(tracking_times.begin(), tracking_times.end(), 0.0);
    double sq_sum = std::inner_product(tracking_times.begin(), tracking_times.end(),
                                        tracking_times.begin(), 0.0);
    m.mean_tracking_time_ms = sum / n;
    m.std_tracking_time_ms = std::sqrt(std::max(0.0, sq_sum / n - m.mean_tracking_time_ms * m.mean_tracking_time_ms));
    m.max_tracking_time_ms = *std::max_element(tracking_times.begin(), tracking_times.end());
    m.min_tracking_time_ms = *std::min_element(tracking_times.begin(), tracking_times.end());

    auto sorted_t = tracking_times;
    std::sort(sorted_t.begin(), sorted_t.end());
    m.p50_tracking_time_ms = sorted_t[static_cast<size_t>(n * 0.50)];
    m.p95_tracking_time_ms = sorted_t[static_cast<size_t>(n * 0.95)];
    m.p99_tracking_time_ms = sorted_t[static_cast<size_t>(n * 0.99)];
  } else {
    m.mean_tracking_time_ms = 0; m.max_tracking_time_ms = 0; m.min_tracking_time_ms = 0;
    m.std_tracking_time_ms = 0; m.p95_tracking_time_ms = 0; m.p99_tracking_time_ms = 0;
  }

  m.partial_degraded_frames = 0;
  m.full_degenerate_frames = 0;
  double sum_cond = 0, sum_inlier = 0, sum_matches = 0, sum_c = 0;
  for (const auto& r : reports) {
    if (r.degeneracy == DegeneracyState::PARTIAL) m.partial_degraded_frames++;
    if (r.degeneracy == DegeneracyState::FULL_DEGENERATE) m.full_degenerate_frames++;
    sum_cond += r.hessian_condition_number;
    sum_inlier += r.inlier_ratio;
    sum_matches += r.num_matches;
    sum_c += r.cauchy_scale;
  }
  size_t nr = reports.size();
  m.mean_condition_number = nr > 0 ? sum_cond / nr : 0;
  m.mean_inlier_ratio = nr > 0 ? sum_inlier / nr : 0;
  m.mean_matches = nr > 0 ? sum_matches / nr : 0;
  m.mean_cauchy_scale = nr > 0 ? sum_c / nr : 0;

  return m;
}

int RunSingleSequence(const std::string& dataset_path, int max_frames,
                       bool show_comparison) {
  TUMDatasetReader reader(dataset_path);
  if (reader.NumFrames() == 0) {
    std::cerr << "No frames loaded: " << dataset_path << std::endl;
    return 1;
  }

  int total = std::min(max_frames, reader.NumFrames());

  std::string seq_display = "custom";
  benchmark::SceneType scene = benchmark::ClassifySequence(dataset_path);
  for (const auto& [key, info] : benchmark::TUMSequences()) {
    if (dataset_path.find(key) != std::string::npos) {
      seq_display = info.display_name;
      break;
    }
  }

  std::cout << "\n==============================================\n";
  std::cout << "  ST-SLAM 4.0  —  " << seq_display;
  if (scene == benchmark::SceneType::DYNAMIC) std::cout << " (DYNAMIC)";
  std::cout << "\n";
  std::cout << "  Frames:   " << total << " / " << reader.NumFrames() << "\n";
  std::cout << "  GT poses: " << reader.GroundTruth().size() << "\n";
  std::cout << "==============================================\n";

  STSLAMConfig& config = STSLAMConfig::Instance();
  config.dataset_path = dataset_path;
  config.max_frames = max_frames;
  
  // Auto-detect vocabulary file
  std::vector<std::string> vocab_candidates = {
    "data/ORBvoc.txt",
    "../data/ORBvoc.txt",
    "../../data/ORBvoc.txt",
    "/content/st_slam/data/ORBvoc.txt",
    "/content/st_slam/st_slam/data/ORBvoc.txt"
  };
  for (const auto& vocab : vocab_candidates) {
    std::ifstream test(vocab);
    if (test.good()) {
      config.vocab_path = vocab;
      std::cout << "[Config] Found vocabulary: " << vocab << std::endl;
      break;
    }
  }
  if (config.vocab_path.empty()) {
    std::cout << "[Config] No vocabulary file found, will build online" << std::endl;
  }

  Tracking tracking(config);

  std::vector<SE3> estimated_poses;
  std::vector<double> frame_timestamps;
  std::vector<double> tracking_times;
  std::vector<TrackingReport> tracking_reports;
  int lost_count = 0;

  auto bench_start = std::chrono::steady_clock::now();

  for (int i = 0; i < total; ++i) {
    Frame frame = reader.GetFrame(i);
    if (frame.rgb.empty() || frame.depth.empty()) continue;

    if (reader.HasIMU()) {
      Vec3 accel = reader.GetInterpolatedAccel(frame.timestamp);
      tracking.SetIMUData(frame.timestamp, accel, Vec3::Zero());
    }

    TrackingReport report = tracking.TrackFrame(frame);

    estimated_poses.push_back(tracking.GetCurrentPose());
    frame_timestamps.push_back(frame.timestamp);
    tracking_times.push_back(report.tracking_time_ms);
    tracking_reports.push_back(report);

    if (report.state == TrackingState::TRACKING_LOST ||
        report.state == TrackingState::TRACKING_DEGENERATE_C) {
      lost_count++;
    }

    std::cout << "\r  [" << std::setw(3) << (i+1) << "/" << total << "]"
              << " t=" << std::setw(5) << std::setprecision(1) << std::fixed << report.tracking_time_ms << "ms"
              << " m=" << std::setw(4) << report.num_matches;
    if (report.degeneracy == DegeneracyState::FULL_DEGENERATE) std::cout << " [DEG]";
    else if (report.degeneracy == DegeneracyState::PARTIAL) std::cout << " [deg]";
    std::cout << std::flush;
  }
  
  int loop_count = tracking.GetNumLoopsDetected();

  auto bench_end = std::chrono::steady_clock::now();
  double total_wall_time_s = std::chrono::duration<double>(bench_end - bench_start).count();
  std::cout << "\n\n  Wall time: " << std::setprecision(1) << total_wall_time_s
            << "s (" << std::setprecision(2) << total_wall_time_s / total << "s/frame)\n";

  // Post-hoc global PGO: optimize all keyframes' poses
  std::cout << "\n  [PGO] Running global pose graph optimization...\n";
  PoseGraph final_pgo(100.0, 1000.0);
  final_pgo.BuildFromKeyframes(tracking.GetLocalMap().GetAllKeyframes());
  std::cout << "  [PGO] Built " << final_pgo.NumEdges() << " edges\n";
  bool pgo_ok = final_pgo.Optimize(tracking.GetLocalMap().GetAllKeyframes());
   if (pgo_ok) {
     std::cout << "  [PGO] SUCCESS\n";
   }

  // Recompute estimated_poses from optimized keyframes
  // For frames between KFs, apply interpolation of the correction
  for (size_t i = 0; i < estimated_poses.size(); ++i) {
    double ts = frame_timestamps[i];
    const auto& all_kfs = tracking.GetLocalMap().GetAllKeyframes();
    if (all_kfs.empty()) break;

    // Find nearest KF by timestamp (exact match or closest before)
    int best_kf = -1;
    double best_ts = -1e9;
    for (const auto& [id, kf] : all_kfs) {
      if (kf.timestamp <= ts && kf.timestamp > best_ts) {
        best_ts = kf.timestamp;
        best_kf = id;
      }
    }
    if (best_kf >= 0) {
      const auto& opt_kf = all_kfs.at(best_kf);
      estimated_poses[i] = opt_kf.pose;
    }
  }

  EvaluationMetrics metrics = Evaluate(
    estimated_poses, frame_timestamps, tracking_times,
    tracking_reports, reader,
    total, lost_count, loop_count);

  PrintMetricsTable(metrics, seq_display, scene);

  if (show_comparison && metrics.ate_rmse >= 0) {
    benchmark::PrintSOTAComparison(
      metrics.ate_rmse, scene, seq_display,
      metrics.rpe_rmse_trans, metrics.rpe_rmse_rot_deg);
    benchmark::PrintEfficiencyComparison(
      metrics.mean_tracking_time_ms, metrics.p95_tracking_time_ms,
      metrics.tracking_success_rate, metrics.full_degenerate_frames);
  }

  std::ofstream pose_file("estimated_poses.txt");
  if (pose_file.is_open()) {
    pose_file << "# ST-SLAM 4.0 trajectory: " << seq_display << "\n";
    pose_file << "# timestamp tx ty tz qx qy qz qw\n";
    for (size_t i = 0; i < estimated_poses.size(); ++i) {
      const SE3& pose = estimated_poses[i];
      pose_file << std::fixed << std::setprecision(6)
                << frame_timestamps[i] << " "
                << pose.trans(0) << " " << pose.trans(1) << " " << pose.trans(2) << " "
                << pose.rot.x() << " " << pose.rot.y() << " " << pose.rot.z() << " " << pose.rot.w()
                << "\n";
    }
    std::cout << "  -> estimated_poses.txt\n";
  }

  {
    std::ofstream f("tracking_times.txt");
    for (double t : tracking_times) f << t << "\n";
  }

  {
    std::ofstream f("metrics_report.txt");
    f << "SEQUENCE " << seq_display << "\n";
    f << "SCENE_TYPE " << (scene == benchmark::SceneType::STATIC ? "STATIC" : "DYNAMIC") << "\n";
    f << "ATE_RMSE " << metrics.ate_rmse << "\n";
    f << "ATE_MEAN " << metrics.ate_mean << "\n";
    f << "ATE_MEDIAN " << metrics.ate_median << "\n";
    f << "ATE_STD " << metrics.ate_std << "\n";
    f << "RPE_RMSE_TRANS " << metrics.rpe_rmse_trans << "\n";
    f << "RPE_RMSE_ROT_DEG " << metrics.rpe_rmse_rot_deg << "\n";
    f << "RPE_MEAN_TRANS " << metrics.rpe_mean_trans << "\n";
    f << "RPE_MEDIAN_TRANS " << metrics.rpe_median_trans << "\n";
    f << "RPE_MEAN_ROT_DEG " << metrics.rpe_mean_rot_deg << "\n";
    f << "MEAN_TRACKING_MS " << metrics.mean_tracking_time_ms << "\n";
    f << "P50_TRACKING_MS " << metrics.p50_tracking_time_ms << "\n";
    f << "P95_TRACKING_MS " << metrics.p95_tracking_time_ms << "\n";
    f << "P99_TRACKING_MS " << metrics.p99_tracking_time_ms << "\n";
    f << "MAX_TRACKING_MS " << metrics.max_tracking_time_ms << "\n";
    f << "SUCCESS_RATE " << metrics.tracking_success_rate << "\n";
    f << "DEGENERATE_FRAMES " << metrics.full_degenerate_frames << "\n";
    f << "PARTIAL_DEGRADED " << metrics.partial_degraded_frames << "\n";
    f << "MEAN_CONDITION_NUM " << metrics.mean_condition_number << "\n";
    f << "MEAN_INLIER_RATIO " << metrics.mean_inlier_ratio << "\n";
    f << "MEAN_MATCHES " << metrics.mean_matches << "\n";
    f << "MEAN_CAUCHY_SCALE " << metrics.mean_cauchy_scale << "\n";
    f << "LOOPS_DETECTED " << metrics.num_loops_detected << "\n";
    f << "WALL_CLOCK_S " << total_wall_time_s << "\n";
    if (scene == benchmark::SceneType::STATIC) {
      f << "SOTA_TARGET_ATE < 0.010\n";
      f << "EXCELLENT_TARGET_ATE < 0.015\n";
      f << "GOOD_TARGET_ATE < 0.030\n";
    } else {
      f << "SOTA_TARGET_ATE < 0.015\n";
      f << "EXCELLENT_TARGET_ATE < 0.025\n";
      f << "GOOD_TARGET_ATE < 0.050\n";
    }
    f << "QUALITY " << benchmark::QualityLabel(
      benchmark::ClassifyATEQuality(metrics.ate_rmse, scene)) << "\n";
    std::cout << "  -> metrics_report.txt\n";
  }

  std::cout << std::endl;
  return 0;
}

void PrintUsage(const char* prog) {
  std::cout << "ST-SLAM 4.0 Benchmark Tool\n\n";
  std::cout << "Usage:\n";
  std::cout << "  Single sequence:\n";
  std::cout << "    " << prog << " /path/to/dataset [max_frames]\n\n";
  std::cout << "  Batch benchmark on all TUM sequences:\n";
  std::cout << "    " << prog << " --batch <dataset_root_dir> [max_frames]\n\n";
  std::cout << "  List known sequences:\n";
  std::cout << "    " << prog << " --list\n\n";
  std::cout << "Examples:\n";
  std::cout << "  " << prog << " /data/tum/rgbd_dataset_freiburg1_xyz 200\n";
  std::cout << "  " << prog << " --batch /data/tum 200\n";
}

int main(int argc, char** argv) {
  if (argc < 2) {
    std::string default_path = "/home/zhang/ORB_SLAM_Learning/datasets/tum/rgbd_dataset_freiburg1_xyz";
    std::cout << "[Default mode] Running on fr1_xyz (50 frames)\n";
    return RunSingleSequence(default_path, 50, true);
  }

  std::string arg1 = argv[1];

  if (arg1 == "--list" || arg1 == "-l") {
    std::cout << "\nKnown TUM RGB-D Sequences:\n";
    std::cout << "  " << std::left << std::setw(38) << "Folder name"
              << std::setw(14) << "Display"
              << std::setw(10) << "Type"
              << std::setw(8) << "Frames"
              << "  Description\n";
    std::cout << "  " << std::string(95, '-') << "\n";
    for (const auto& [key, info] : benchmark::TUMSequences()) {
      std::cout << "  " << std::left << std::setw(38) << key
                << std::setw(14) << info.display_name
                << std::setw(10) << (info.scene_type == benchmark::SceneType::STATIC ? "STATIC" : "DYNAMIC")
                << std::setw(8) << info.num_frames
                << info.description << "\n";
    }
    return 0;
  }

  if (arg1 == "--batch" || arg1 == "-b") {
    if (argc < 3) {
      std::cerr << "Error: --batch requires dataset root directory\n";
      return 1;
    }
    std::string root = argv[2];
    int max_frames = (argc > 3) ? std::stoi(argv[3]) : 200;

    std::vector<benchmark::BatchResult> batch_results;
    const std::vector<std::string> batch_seqs = {
      "rgbd_dataset_freiburg1_xyz",
      "rgbd_dataset_freiburg1_desk",
      "rgbd_dataset_freiburg2_desk",
      "rgbd_dataset_freiburg3_office",
      "rgbd_dataset_freiburg3_walking_xyz",
      "rgbd_dataset_freiburg3_walking_halfsphere",
    };

    for (const auto& seq : batch_seqs) {
      std::string path = root + "/" + seq;
      auto it = benchmark::TUMSequences().find(seq);
      std::string display = (it != benchmark::TUMSequences().end()) ? it->second.display_name : seq;
      benchmark::SceneType scene = benchmark::ClassifySequence(path);

      std::cout << "\n" << std::string(60, '=') << "\n";
      std::cout << "  Processing: " << display << "\n";
      std::cout << std::string(60, '=') << "\n";

      TUMDatasetReader reader(path);
      if (reader.NumFrames() == 0) {
        std::cout << "  [SKIP] No frames loaded\n";
        continue;
      }
      int total = std::min(max_frames, reader.NumFrames());
      STSLAMConfig& config = STSLAMConfig::Instance();
      config.dataset_path = path;
      config.max_frames = max_frames;
      Tracking tracking(config);

      std::vector<SE3> poses;
      std::vector<double> timestamps, times;
      std::vector<TrackingReport> reports;
      int lost = 0;
      int loops = 0;

      for (int i = 0; i < total; ++i) {
        Frame f = reader.GetFrame(i);
        if (f.rgb.empty() || f.depth.empty()) continue;

        if (reader.HasIMU()) {
          Vec3 accel = reader.GetInterpolatedAccel(f.timestamp);
          tracking.SetIMUData(f.timestamp, accel, Vec3::Zero());
        }

        auto r = tracking.TrackFrame(f);
        poses.push_back(tracking.GetCurrentPose());
        timestamps.push_back(f.timestamp);
        times.push_back(r.tracking_time_ms);
        reports.push_back(r);
        if (r.state == TrackingState::TRACKING_LOST ||
            r.state == TrackingState::TRACKING_DEGENERATE_C) lost++;
        std::cout << "\r    [" << (i+1) << "/" << total << "]" << std::flush;
      }

      std::vector<SE3> gt;
      for (double ts : timestamps) gt.push_back(reader.GetGroundTruth(ts));
      std::vector<SE3> aligned = poses;
      math::align_trajectory(aligned, gt);
      double ate = math::compute_ate_rmse(aligned, gt);
      auto rpe = math::compute_rpe_full(aligned, gt, timestamps);
      double mean_t = !times.empty() ? std::accumulate(times.begin(), times.end(), 0.0) / times.size() : 0;
      int degen = 0;
      for (const auto& r : reports) if (r.degeneracy == DegeneracyState::FULL_DEGENERATE) degen++;
      double sr = total > 0 ? 1.0 - static_cast<double>(lost) / total : 0;

      batch_results.push_back({display, ate, rpe.rmse_trans, mean_t, sr, degen,
                                benchmark::ClassifyATEQuality(ate, scene)});
    }

    benchmark::PrintBatchSummary(batch_results);
    return 0;
  }

  int max_frames = (argc > 2) ? std::stoi(argv[2]) : 200;
  return RunSingleSequence(arg1, max_frames, true);
}
