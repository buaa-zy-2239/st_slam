#include <iostream>
#include <cassert>
#include <cmath>
#include "st_slam/backend/pose_graph.h"
#include "st_slam/frontend/local_map.h"
#include "st_slam/core/types.h"
#include <unordered_map>

using namespace st_slam;

void Test1_AddEdgeBasic() {
  std::cout << "\n[Test1] AddEdge Basic Test\n";
  std::cout << "=" << std::string(50, '=') << "\n";

  PoseGraph pose_graph(100.0, 1000.0);

  SE3 rel_pose(Quat::Identity(), Vec3(0.1, 0.0, 0.0));
  Mat6 info = PoseGraph::DefaultInformation(100.0, 1000.0);

  pose_graph.AddEdge(0, 1, rel_pose, info);
  std::cout << "Added edge: KF0 -> KF1\n";

  assert(pose_graph.NumEdges() == 1 && "Should have 1 edge");

  pose_graph.AddEdge(1, 2, rel_pose, info);
  pose_graph.AddEdge(2, 3, rel_pose, info);
  std::cout << "Added 2 more edges\n";

  assert(pose_graph.NumEdges() == 3 && "Should have 3 edges");

  std::cout << "[Test1] PASSED\n\n";
}

void Test2_BuildFromKeyframes() {
  std::cout << "\n[Test2] Build From Keyframes Test\n";
  std::cout << "=" << std::string(50, '=') << "\n";

  LocalMap local_map(10, 50);
  PoseGraph pose_graph(100.0, 1000.0);

  for (int i = 0; i < 5; ++i) {
    Frame frame;
    frame.id = i;
    frame.timestamp = i * 0.1;
    frame.keypoints.resize(100);
    frame.descriptors = cv::Mat(100, 32, CV_8U, cv::Scalar(0));
    frame.pose = SE3(Quat::Identity(), Vec3(i * 0.1, 0.0, 0.0));

    local_map.AddKeyFrame(frame);
    std::cout << "Added KF" << i << " at x=" << (i * 0.1) << "m\n";
  }

  pose_graph.BuildFromKeyframes(local_map.GetAllKeyframes());

  std::cout << "Built " << pose_graph.NumEdges() << " edges\n";
  assert(pose_graph.NumEdges() == 4 &&
         "Should have 4 edges for 5 keyframes");

  std::cout << "[Test2] PASSED\n\n";
}

void Test3_AddLoopEdge() {
  std::cout << "\n[Test3] Add Loop Edge Test\n";
  std::cout << "=" << std::string(50, '=') << "\n";

  LocalMap local_map(10, 50);
  PoseGraph pose_graph(100.0, 1000.0);

  for (int i = 0; i < 5; ++i) {
    Frame frame;
    frame.id = i;
    frame.timestamp = i * 0.1;
    frame.keypoints.resize(100);
    frame.descriptors = cv::Mat(100, 32, CV_8U, cv::Scalar(0));
    frame.pose = SE3(Quat::Identity(), Vec3(i * 0.1, 0.0, 0.0));

    local_map.AddKeyFrame(frame);
  }

  pose_graph.BuildFromKeyframes(local_map.GetAllKeyframes());
  std::cout << "Built " << pose_graph.NumEdges() << " sequential edges\n";

  SE3 loop_rel(Quat::Identity(), Vec3(0.0, 0.0, 0.02));
  Mat6 loop_info = PoseGraph::DefaultInformation(50.0, 500.0);
  pose_graph.AddEdge(0, 4, loop_rel, loop_info);
  std::cout << "Added loop edge: KF0 -> KF4\n";

  assert(pose_graph.NumEdges() == 5 &&
         "Should have 5 edges (4 sequential + 1 loop)");

  std::cout << "[Test3] PASSED\n\n";
}

void Test4_OptimizeWithLoop() {
  std::cout << "\n[Test4] Optimize With Loop Test\n";
  std::cout << "=" << std::string(50, '=') << "\n";

  LocalMap local_map(10, 50);
  PoseGraph pose_graph(100.0, 1000.0);

  Vec3 translations[] = {
    Vec3(0.0, 0.0, 0.0),
    Vec3(0.1, 0.0, 0.0),
    Vec3(0.2, 0.0, 0.0),
    Vec3(0.3, 0.0, 0.0),
    Vec3(0.4, 0.0, 0.0)
  };

  for (int i = 0; i < 5; ++i) {
    Frame frame;
    frame.id = i;
    frame.timestamp = i * 0.1;
    frame.keypoints.resize(100);
    frame.descriptors = cv::Mat(100, 32, CV_8U, cv::Scalar(0));
    frame.pose = SE3(Quat::Identity(), translations[i]);

    local_map.AddKeyFrame(frame);
  }

  pose_graph.BuildFromKeyframes(local_map.GetAllKeyframes());
  std::cout << "Built " << pose_graph.NumEdges() << " sequential edges\n";

  SE3 loop_rel(Quat::Identity(), Vec3(0.0, 0.0, 0.03));
  Mat6 loop_info = PoseGraph::DefaultInformation(50.0, 500.0);
  pose_graph.AddEdge(0, 4, loop_rel, loop_info);
  std::cout << "Added loop edge\n";

  auto& keyframes = local_map.GetAllKeyframes();
  std::cout << "Pre-optimization:\n";
  for (const auto& [id, kf] : keyframes) {
    std::cout << "  KF" << id << ": pos=("
              << kf.pose.trans(0) << ", "
              << kf.pose.trans(1) << ", "
              << kf.pose.trans(2) << ")\n";
  }

  bool optimize_result = pose_graph.Optimize(keyframes);
  std::cout << "Optimize result: " << (optimize_result ? "SUCCESS" : "FAILED") << "\n";

  assert(optimize_result && "Optimization should succeed");

  std::cout << "Post-optimization:\n";
  for (const auto& [id, kf] : keyframes) {
    std::cout << "  KF" << id << ": pos=("
              << kf.pose.trans(0) << ", "
              << kf.pose.trans(1) << ", "
              << kf.pose.trans(2) << ")\n";
  }

  double kf4_trans = keyframes[4].pose.trans.norm();
  std::cout << "KF4 distance from origin: " << kf4_trans << "m\n";

  assert(kf4_trans < 0.1 && "KF4 should be pulled back towards origin by loop");

  std::cout << "[Test4] PASSED\n\n";
}

int main() {
  std::cout << "\n";
  std::cout << "╔" << std::string(68, '=') << "╗\n";
  std::cout << "║" << std::string(15, ' ') << "PoseGraph Tests"
            << std::string(35, ' ') << "║\n";
  std::cout << "╚" << std::string(68, '=') << "╝\n";

  try {
    Test1_AddEdgeBasic();
    Test2_BuildFromKeyframes();
    Test3_AddLoopEdge();
    Test4_OptimizeWithLoop();

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
