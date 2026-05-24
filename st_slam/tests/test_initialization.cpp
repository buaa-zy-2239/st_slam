#include <iostream>
#include <cassert>
#include <cmath>
#include "st_slam/frontend/tracking.h"
#include "st_slam/frontend/local_map.h"
#include "st_slam/core/types.h"
#include <opencv2/core.hpp>

using namespace st_slam;

void Test1_BasicInitialization() {
  std::cout << "\n[Test1] Basic Initialization Test\n";
  std::cout << "=" << std::string(50, '=') << "\n";

  STSLAMConfig config;
  config.vocab_path = "";
  config.use_loop_closure = false;
  Tracking tracker(config);

  Frame frame;
  frame.id = 0;
  frame.timestamp = 0.0;
  frame.keypoints.resize(100);
  frame.descriptors = cv::Mat(100, 32, CV_8U, cv::Scalar(0));
  frame.pose = SE3::Identity();

  bool result = tracker.InitializeForTesting(frame);

  std::cout << "Initialize result: " << (result ? "PASS" : "FAIL") << "\n";
  assert(result && "Initialization should succeed");

  std::cout << "[Test1] PASSED\n\n";
}

void Test2_KeyFramePoseConsistency() {
  std::cout << "\n[Test2] KeyFrame Pose Consistency Test\n";
  std::cout << "=" << std::string(50, '=') << "\n";

  LocalMap local_map(10, 50);

  Frame frame1;
  frame1.id = 0;
  frame1.timestamp = 0.0;
  frame1.keypoints.resize(100);
  frame1.descriptors = cv::Mat(100, 32, CV_8U, cv::Scalar(0));
  frame1.pose = SE3::Identity();

  int kf_id1 = local_map.AddKeyFrame(frame1);
  std::cout << "Added KF0 with id=" << kf_id1 << "\n";

  const KeyFrame* kf1 = local_map.GetKeyFrame(kf_id1);
  assert(kf1 != nullptr && "KF1 should exist");
  std::cout << "KF0 pose: trans=("
            << kf1->pose.trans(0) << ", "
            << kf1->pose.trans(1) << ", "
            << kf1->pose.trans(2) << ")\n";

  assert(kf1->pose.trans.norm() < 1e-6 &&
         "First KF should have identity pose");

  SE3 expected_pose(Quat::Identity(), Vec3(0.1, 0.2, 0.05));
  Frame frame2 = frame1;
  frame2.id = 1;
  frame2.pose = expected_pose;

  int kf_id2 = local_map.AddKeyFrame(frame2);
  std::cout << "Added KF1 with id=" << kf_id2 << "\n";

  const KeyFrame* kf2 = local_map.GetKeyFrame(kf_id2);
  assert(kf2 != nullptr && "KF2 should exist");

  double pose_error = (kf2->pose.trans - expected_pose.trans).norm();
  std::cout << "KF1 pose error: " << pose_error << "m\n";

  assert(pose_error < 1e-6 && "KF2 pose should match expected");

  std::cout << "[Test2] PASSED\n\n";
}

void Test3_BuildSequentialEdges() {
  std::cout << "\n[Test3] Build Sequential Edges Test\n";
  std::cout << "=" << std::string(50, '=') << "\n";

  LocalMap local_map(10, 50);
  PoseGraph pose_graph(100.0, 1000.0);

  std::vector<SE3> true_poses;
  true_poses.push_back(SE3::Identity());
  true_poses.push_back(SE3(Quat::Identity(), Vec3(0.1, 0.0, 0.0)));
  true_poses.push_back(SE3(Quat::Identity(), Vec3(0.2, 0.0, 0.0)));
  true_poses.push_back(SE3(Quat::Identity(), Vec3(0.3, 0.0, 0.0)));

  for (size_t i = 0; i < true_poses.size(); ++i) {
    Frame frame;
    frame.id = i;
    frame.timestamp = i * 0.1;
    frame.keypoints.resize(100);
    frame.descriptors = cv::Mat(100, 32, CV_8U, cv::Scalar(0));
    frame.pose = true_poses[i];

    int kf_id = local_map.AddKeyFrame(frame);
    std::cout << "Added KF" << i << " with id=" << kf_id
              << ", pose=(" << true_poses[i].trans(0) << ", "
              << true_poses[i].trans(1) << ", "
              << true_poses[i].trans(2) << ")\n";
  }

  pose_graph.BuildFromKeyframes(local_map.GetAllKeyframes());

  std::cout << "Built " << pose_graph.NumEdges() << " edges\n";
  assert(pose_graph.NumEdges() == 3 &&
         "Should have 3 edges for 4 keyframes");

  std::cout << "[Test3] PASSED\n\n";
}

int main() {
  std::cout << "\n";
  std::cout << "╔" << std::string(68, '=') << "╗\n";
  std::cout << "║" << std::string(10, ' ') << "ST-SLAM 4.0 Unit Tests"
            << std::string(30, ' ') << "║\n";
  std::cout << "╚" << std::string(68, '=') << "╝\n";

  try {
    Test1_BasicInitialization();
    Test2_KeyFramePoseConsistency();
    Test3_BuildSequentialEdges();

    std::cout << "\n";
    std::cout << "╔" << std::string(68, '=') << "╗\n";
    std::cout << "║" << std::string(20, ' ') << "ALL TESTS PASSED"
              << std::string(31, ' ') << "║\n";
    std::cout << "╚" << std::string(68, '=') << "╝\n";

    return 0;
  } catch (const std::exception& e) {
    std::cerr << "\n[ERROR] Test failed: " << e.what() << "\n";
    return 1;
  }
}
