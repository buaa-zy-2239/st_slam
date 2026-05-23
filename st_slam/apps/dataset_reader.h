#ifndef ST_SLAM_DATASET_READER_H
#define ST_SLAM_DATASET_READER_H

#include "st_slam/core/types.h"
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <sstream>
#include <opencv2/imgcodecs.hpp>

namespace st_slam {

struct TUMData {
  double timestamp;
  std::string rgb_path;
  std::string depth_path;
};

struct GroundTruthPose {
  double timestamp;
  SE3 pose;
};

struct IMUData {
  double timestamp;
  Vec3 accel;
  Vec3 gyro;
};

class TUMDatasetReader {
public:
  explicit TUMDatasetReader(const std::string& dataset_root);

  static void CreateAssociationFile(const std::string& dataset_root);

  bool HasNext() const;
  int NumFrames() const { return static_cast<int>(data_list_.size()); }

  Frame NextFrame();
  Frame GetFrame(int idx);

  const std::vector<TUMData>& DataList() const { return data_list_; }
  const std::vector<GroundTruthPose>& GroundTruth() const { return ground_truth_; }

  std::vector<double> GetTimestamps() const;

  SE3 GetGroundTruth(double timestamp) const;

  bool HasIMU() const { return !imu_data_.empty(); }
  const std::vector<IMUData>& GetIMUData() const { return imu_data_; }

  std::vector<IMUData> GetIMUBetween(double t0, double t1) const;
  Vec3 GetInterpolatedAccel(double timestamp) const;

  double GetFx() const { return fx_; }
  double GetFy() const { return fy_; }
  double GetCx() const { return cx_; }
  double GetCy() const { return cy_; }
  double GetDepthFactor() const { return depth_factor_; }

private:
  void LoadAssociationFile(const std::string& assoc_path);
  void LoadGroundTruth(const std::string& gt_path);
  void LoadIMUData(const std::string& imu_path);

  std::string dataset_root_;
  std::vector<TUMData> data_list_;
  std::vector<GroundTruthPose> ground_truth_;
  std::vector<IMUData> imu_data_;
  int current_idx_ = 0;
  double fx_ = 525.0, fy_ = 525.0;
  double cx_ = 319.5, cy_ = 239.5;
  double depth_factor_ = 5000.0;
};

} // namespace st_slam

#endif
