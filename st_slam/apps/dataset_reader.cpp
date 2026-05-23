#include "dataset_reader.h"
#include <sys/stat.h>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <opencv2/imgproc.hpp>
#include "st_slam/core/math_utils.h"

namespace st_slam {

TUMDatasetReader::TUMDatasetReader(const std::string& dataset_root)
  : dataset_root_(dataset_root) {
  std::string assoc_path = dataset_root + "/associations.txt";
  struct stat buffer;
  if (stat(assoc_path.c_str(), &buffer) != 0) {
    CreateAssociationFile(dataset_root);
  }
  LoadAssociationFile(assoc_path);

  std::string gt_path = dataset_root + "/groundtruth.txt";
  if (stat(gt_path.c_str(), &buffer) == 0) {
    LoadGroundTruth(gt_path);
  }

  std::string imu_path = dataset_root + "/accelerometer.txt";
  if (stat(imu_path.c_str(), &buffer) == 0) {
    LoadIMUData(imu_path);
  }
}

void TUMDatasetReader::CreateAssociationFile(const std::string& dataset_root) {
  std::string rgb_path = dataset_root + "/rgb.txt";
  std::string depth_path = dataset_root + "/depth.txt";
  std::string out_path = dataset_root + "/associations.txt";

  auto load_timestamps = [](const std::string& path) {
    std::vector<std::pair<double, std::string>> entries;
    std::ifstream file(path);
    if (!file.is_open()) return entries;
    std::string line;
    while (std::getline(file, line)) {
      if (line.empty() || line[0] == '#') continue;
      std::istringstream iss(line);
      double ts;
      std::string filepath;
      iss >> ts >> filepath;
      entries.emplace_back(ts, filepath);
    }
    return entries;
  };

  auto rgb_entries = load_timestamps(rgb_path);
  auto depth_entries = load_timestamps(depth_path);

  std::ofstream out(out_path);
  size_t i = 0, j = 0;
  while (i < rgb_entries.size() && j < depth_entries.size()) {
    double diff = rgb_entries[i].first - depth_entries[j].first;
    if (std::abs(diff) < 0.02) {
      out << std::fixed << std::setprecision(6)
          << rgb_entries[i].first << " "
          << rgb_entries[i].second << " "
          << depth_entries[j].first << " "
          << depth_entries[j].second << std::endl;
      i++; j++;
    } else if (diff < 0) {
      i++;
    } else {
      j++;
    }
  }
}

void TUMDatasetReader::LoadAssociationFile(const std::string& assoc_path) {
  std::ifstream file(assoc_path);
  if (!file.is_open()) return;
  std::string line;
  while (std::getline(file, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream iss(line);
    TUMData data;
    std::string rgb_rel, depth_rel;
    iss >> data.timestamp >> rgb_rel >> data.timestamp >> depth_rel;
    if (dataset_root_.back() == '/') {
      data.rgb_path = dataset_root_ + rgb_rel;
      data.depth_path = dataset_root_ + depth_rel;
    } else {
      data.rgb_path = dataset_root_ + "/" + rgb_rel;
      data.depth_path = dataset_root_ + "/" + depth_rel;
    }
    data_list_.push_back(data);
  }
}

void TUMDatasetReader::LoadGroundTruth(const std::string& gt_path) {
  std::ifstream file(gt_path);
  if (!file.is_open()) return;
  std::string line;
  while (std::getline(file, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream iss(line);
    GroundTruthPose gt;
    double tx, ty, tz, qx, qy, qz, qw;
    iss >> gt.timestamp >> tx >> ty >> tz >> qx >> qy >> qz >> qw;
    gt.pose = SE3(Quat(qw, qx, qy, qz), Vec3(tx, ty, tz));
    ground_truth_.push_back(gt);
  }
}

void TUMDatasetReader::LoadIMUData(const std::string& imu_path) {
  std::ifstream file(imu_path);
  if (!file.is_open()) return;
  std::string line;
  while (std::getline(file, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream iss(line);
    IMUData imu;
    double ax, ay, az;
    iss >> imu.timestamp >> ax >> ay >> az;
    imu.accel = Vec3(ax, ay, az);
    imu.gyro = Vec3::Zero();
    imu_data_.push_back(imu);
  }
  if (!imu_data_.empty()) {
    for (size_t i = 1; i < imu_data_.size(); ++i) {
      double dt = imu_data_[i].timestamp - imu_data_[i-1].timestamp;
      if (dt > 1e-6) {
        Vec3 da = (imu_data_[i].accel - imu_data_[i-1].accel) / dt;
        imu_data_[i].gyro = da;
      }
    }
  }
}

bool TUMDatasetReader::HasNext() const {
  return current_idx_ < static_cast<int>(data_list_.size());
}

Frame TUMDatasetReader::NextFrame() {
  if (!HasNext()) return Frame();
  return GetFrame(current_idx_++);
}

Frame TUMDatasetReader::GetFrame(int idx) {
  Frame frame;
  const auto& data = data_list_[idx];
  frame.id = idx;
  frame.timestamp = data.timestamp;

  frame.rgb = cv::imread(data.rgb_path, cv::IMREAD_COLOR);
  frame.depth = cv::imread(data.depth_path, cv::IMREAD_UNCHANGED);
  if (frame.rgb.empty() || frame.depth.empty()) return frame;

  cv::cvtColor(frame.rgb, frame.gray, cv::COLOR_BGR2GRAY);
  return frame;
}

std::vector<double> TUMDatasetReader::GetTimestamps() const {
  std::vector<double> timestamps;
  for (const auto& d : data_list_) timestamps.push_back(d.timestamp);
  return timestamps;
}

SE3 TUMDatasetReader::GetGroundTruth(double timestamp) const {
  if (ground_truth_.empty()) return SE3::Identity();
  auto it = std::lower_bound(ground_truth_.begin(), ground_truth_.end(), timestamp,
    [](const GroundTruthPose& gp, double ts) { return gp.timestamp < ts; });
  if (it == ground_truth_.begin()) return it->pose;
  if (it == ground_truth_.end()) return (it - 1)->pose;
  double t0 = (it - 1)->timestamp;
  double t1 = it->timestamp;
  double alpha = (timestamp - t0) / (t1 - t0);
  alpha = std::clamp(alpha, 0.0, 1.0);
  const SE3& p0 = (it - 1)->pose;
  const SE3& p1 = it->pose;
  Vec6 twist = math::se3_log(p0.inverse() * p1);
  return p0 * math::se3_exp(twist * alpha);
}

std::vector<IMUData> TUMDatasetReader::GetIMUBetween(double t0, double t1) const {
  std::vector<IMUData> result;
  for (const auto& imu : imu_data_) {
    if (imu.timestamp >= t0 && imu.timestamp <= t1) {
      result.push_back(imu);
    }
  }
  return result;
}

Vec3 TUMDatasetReader::GetInterpolatedAccel(double timestamp) const {
  if (imu_data_.empty()) return Vec3::Zero();
  auto it = std::lower_bound(imu_data_.begin(), imu_data_.end(), timestamp,
    [](const IMUData& imu, double ts) { return imu.timestamp < ts; });
  if (it == imu_data_.begin()) return it->accel;
  if (it == imu_data_.end()) return (it - 1)->accel;
  double t0 = (it - 1)->timestamp;
  double t1 = it->timestamp;
  double alpha = (timestamp - t0) / (t1 - t0);
  alpha = std::clamp(alpha, 0.0, 1.0);
  return (it - 1)->accel * (1.0 - alpha) + it->accel * alpha;
}

} // namespace st_slam
