#include <iostream>
#include <vector>
#include <cmath>
#include <Eigen/Dense>
#include <Eigen/Geometry>

using Vec3 = Eigen::Vector3d;
using Quat = Eigen::Quaterniond;
using Mat3 = Eigen::Matrix3d;
using SE3 = Eigen::Isometry3d;

struct KeyFrame {
  int id;
  SE3 pose;
};

void PrintSeparator(const std::string& title) {
  std::cout << "\n";
  std::cout << "========================================\n";
  std::cout << title << "\n";
  std::cout << "========================================\n";
}

SE3 CreateSE3(const Vec3& trans, const Quat& rot) {
  SE3 T;
  T.linear() = rot.matrix();
  T.translation() = trans;
  return T;
}

SE3 InverseSE3(const SE3& T) {
  SE3 T_inv;
  T_inv.linear() = T.linear().transpose();
  T_inv.translation() = -T.linear().transpose() * T.translation();
  return T_inv;
}

void TestPoseGraphOptimizeAndWriteback() {
  PrintSeparator("TEST 1: PoseGraph Optimize Writeback");

  SE3 T0 = CreateSE3(Vec3(0, 0, 0), Quat(1, 0, 0, 0));
  SE3 T1 = CreateSE3(Vec3(1, 0, 0), Quat(1, 0.1, 0, 0));
  SE3 T2 = CreateSE3(Vec3(2, 0, 0), Quat(1, 0.2, 0, 0));

  std::unordered_map<int, KeyFrame> keyframes;
  keyframes[0] = {0, T0};
  keyframes[1] = {1, T1};
  keyframes[2] = {2, T2};

  std::cout << "BEFORE: Address of keyframes[1].pose: " << &keyframes[1].pose << "\n";
  std::cout << "BEFORE Optimize:\n";
  for (auto& [id, kf] : keyframes) {
    std::cout << "  KF" << id << " pose: pos=("
              << kf.pose.translation()(0) << ", " << kf.pose.translation()(1) << ", " << kf.pose.translation()(2) << ")\n";
  }

  std::cout << "\nSimulating PoseGraph::Optimize()...\n";
  std::cout << "  (In real code, this would call ceres::Solve())\n";
  std::cout << "  (Keyframes passed by reference via Optimize(keyframes))\n\n";

  for (auto& [id, kf] : keyframes) {
    Vec3 new_trans = kf.pose.translation() + Vec3(0.02, 0.01, 0.01);
    kf.pose.translation() = new_trans;
  }

  std::cout << "AFTER Optimize (simulated correction):\n";
  for (auto& [id, kf] : keyframes) {
    std::cout << "  KF" << id << " pose: pos=("
              << kf.pose.translation()(0) << ", " << kf.pose.translation()(1) << ", " << kf.pose.translation()(2) << ")\n";
  }

  std::cout << "\nCRITICAL CHECK:\n";
  std::cout << "  If BEFORE != AFTER -> Writeback IS working!\n";
  std::cout << "  If BEFORE == AFTER -> Writeback NOT working!\n";
}

void TestLocalMapKeyFrameReference() {
  PrintSeparator("TEST 2: LocalMap KeyFrame Reference Consistency");

  std::unordered_map<int, KeyFrame> local_map;
  local_map[0] = {0, SE3::Identity()};
  local_map[1] = {1, SE3::Identity()};

  KeyFrame* kf1_ptr = &local_map[1];
  KeyFrame& kf1_ref = local_map[1];

  std::cout << "KeyFrame object addresses:\n";
  std::cout << "  &local_map[1]: " << static_cast<void*>(&local_map[1]) << "\n";
  std::cout << "  kf1_ptr:       " << static_cast<void*>(kf1_ptr) << "\n";
  std::cout << "  &kf1_ref:      " << static_cast<void*>(&kf1_ref) << "\n\n";

  std::cout << "CRITICAL: These MUST be the same address!\n";
  std::cout << "If different, then GetKeyFrame() and GetAllKeyFrames()[id]\n";
  std::cout << "return DIFFERENT objects, causing PGO to fail silently.\n\n";

  kf1_ref.pose.translation() = Vec3(1, 2, 3);
  std::cout << "After modifying kf1_ref.pose:\n";
  std::cout << "  local_map[1].pose.translation(): ("
            << local_map[1].pose.translation()(0) << ", "
            << local_map[1].pose.translation()(1) << ", "
            << local_map[1].pose.translation()(2) << ")\n";
  std::cout << "  (Should be (1, 2, 3) if reference works)\n";
}

void TestCorrectionPropagation() {
  PrintSeparator("TEST 3: PGO Correction Propagation");

  SE3 current_pose_before_opt = CreateSE3(Vec3(0.5, 0.1, 0.2), Quat(1, 0, 0, 0));
  SE3 kf_before_opt = CreateSE3(Vec3(0, 0, 0), Quat(1, 0, 0, 0));
  SE3 kf_after_opt = CreateSE3(Vec3(0.02, 0.01, 0.01), Quat(1, 0.01, 0, 0));

  std::cout << "BEFORE correction:\n";
  std::cout << "  current_pose_: pos=("
            << current_pose_before_opt.translation()(0) << ", "
            << current_pose_before_opt.translation()(1) << ", "
            << current_pose_before_opt.translation()(2) << ")\n";

  SE3 correction = InverseSE3(kf_before_opt) * kf_after_opt;
  SE3 current_pose_after_opt = correction * current_pose_before_opt;

  std::cout << "\nCorrection: delta_pos=("
            << correction.translation()(0) << ", "
            << correction.translation()(1) << ", "
            << correction.translation()(2) << ")\n";

  std::cout << "\nAFTER correction:\n";
  std::cout << "  current_pose_: pos=("
            << current_pose_after_opt.translation()(0) << ", "
            << current_pose_after_opt.translation()(1) << ", "
            << current_pose_after_opt.translation()(2) << ")\n";

  std::cout << "\nChange in current_pose_: ("
            << (current_pose_after_opt.translation() - current_pose_before_opt.translation())(0) << ", "
            << (current_pose_after_opt.translation() - current_pose_before_opt.translation())(1) << ", "
            << (current_pose_after_opt.translation() - current_pose_before_opt.translation())(2) << ")\n";

  std::cout << "\nCRITICAL: If this change is ~0, correction is NOT being applied!\n";
}

void TestGroundTruthVsEstimated() {
  PrintSeparator("TEST 4: Ground Truth vs Estimated Scale");

  std::vector<Vec3> gt_positions = {
    Vec3(1.233, -0.013, 1.693),
    Vec3(1.150, 0.000, 1.550),
    Vec3(0.950, -0.004, 1.290)
  };

  std::vector<Vec3> est_positions = {
    Vec3(0.001, 0.000, -0.001),
    Vec3(0.001, 0.001, -0.001),
    Vec3(0.001, 0.001, -0.001)
  };

  std::cout << "Ground Truth (fr1/desk):\n";
  for (size_t i = 0; i < gt_positions.size(); ++i) {
    std::cout << "  Frame " << (i * 50) << ": ("
              << gt_positions[i](0) << ", "
              << gt_positions[i](1) << ", "
              << gt_positions[i](2) << ")\n";
  }

  std::cout << "\nEstimated (current SLAM):\n";
  for (size_t i = 0; i < est_positions.size(); ++i) {
    std::cout << "  Frame " << (i * 50) << ": ("
              << est_positions[i](0) << ", "
              << est_positions[i](1) << ", "
              << est_positions[i](2) << ")\n";
  }

  std::cout << "\nCRITICAL: Ground truth moves 0.3-0.4m, but estimated < 0.01m!\n";
  std::cout << "This is NOT a PGO problem - it's a tracking/origin problem!\n";
  std::cout << "The SLAM never learns its absolute world position.\n";
}

int main() {
  std::cout << "========================================\n";
  std::cout << "PGO Pose Sync Debug Test Suite\n";
  std::cout << "========================================\n";
  std::cout << "\nThis test suite verifies the data flow of PGO optimization.\n";
  std::cout << "It simulates the key operations WITHOUT depending on st_slam libs.\n\n";

  TestPoseGraphOptimizeAndWriteback();
  TestLocalMapKeyFrameReference();
  TestCorrectionPropagation();
  TestGroundTruthVsEstimated();

  PrintSeparator("RECOMMENDATIONS");

  std::cout << "1. Add runtime address checks:\n";
  std::cout << "   - &local_map_[id] vs &all_kfs[id] must match\n";
  std::cout << "   - Use static_cast<void*>(&keyframe.pose) to compare\n\n";

  std::cout << "2. Print before/after PGO:\n";
  std::cout << "   - Print keyframe.pose before Optimize()\n";
  std::cout << "   - Print keyframe.pose after Optimize()\n";
  std::cout << "   - If same, writeback is broken\n\n";

  std::cout << "3. Verify current_pose_ update:\n";
  std::cout << "   - Print current_pose_ after PGO\n";
  std::cout << "   - Compare with keyframe pose\n";
  std::cout << "   - They MUST differ by the correction\n\n";

  std::cout << "4. The real problem: Absolute position is unknown!\n";
  std::cout << "   - Ground truth: camera at (1.2, -0.01, 1.7)\n";
  std::cout << "   - Our SLAM: camera at (0, 0, 0)\n";
  std::cout << "   - Even perfect PGO can't fix this mismatch\n";

  return 0;
}
