#include <iostream>
#include <cassert>
#include "st_slam/backend/pose_graph.h"
#include "st_slam/frontend/local_map.h"
#include "st_slam/core/types.h"

using namespace st_slam;

void TestLoopEdgePreservation_Fixed() {
  std::cout << "\n[DEBUG] Test: Loop Edge Preservation (Fixed Order)\n";
  
  LocalMap local_map(10, 50);
  PoseGraph pose_graph(100.0, 1000.0);
  
  // Create keyframes
  for (int i = 0; i < 6; ++i) {
    Frame frame;
    frame.id = i;
    frame.timestamp = i * 0.1;
    frame.keypoints.resize(100);
    frame.descriptors = cv::Mat(100, 32, CV_8U, cv::Scalar(0));
    frame.pose = SE3(Quat::Identity(), Vec3(i * 0.1, 0.0, 0.0));
    local_map.AddKeyFrame(frame);
  }
  
  // Step 1: Build sequential edges FIRST (correct order)
  std::cout << "[DEBUG] Building sequential edges FIRST\n";
  pose_graph.BuildFromKeyframes(local_map.GetAllKeyframes());
  std::cout << "[DEBUG] Edges after BuildFromKeyframes: " << pose_graph.NumEdges() << "\n";
  assert(pose_graph.NumEdges() == 5 && "Should have 5 sequential edges");
  
  // Step 2: Add loop edge AFTER (correct order)
  SE3 loop_rel(Quat::Identity(), Vec3(0.0, 0.0, 0.023));
  Mat6 loop_info = PoseGraph::DefaultInformation(50.0, 500.0);
  std::cout << "[DEBUG] Adding loop edge KF0->KF5 AFTER\n";
  pose_graph.AddEdge(0, 5, loop_rel, loop_info);
  
  std::cout << "[DEBUG] Edges after loop edge: " << pose_graph.NumEdges() << "\n";
  assert(pose_graph.NumEdges() == 6 && "Should have 6 edges (5 sequential + 1 loop)");
  
  std::cout << "[DEBUG] PASS: Loop edge preserved with correct order!\n";
  std::cout << "[DEBUG] PASSED\n";
}

int main() {
  std::cout << "[DEBUG] ======================================\n";
  std::cout << "[DEBUG] Loop Edge Preservation Test\n";
  std::cout << "[DEBUG] ======================================\n";
  
  try {
    // Test with correct order: BuildFromKeyframes FIRST, then AddEdge
    TestLoopEdgePreservation_Fixed();
    
    std::cout << "\n[DEBUG] ======================================\n";
    std::cout << "[DEBUG] ALL TESTS PASSED\n";
    std::cout << "[DEBUG] ======================================\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[DEBUG] TEST FAILED: " << e.what() << "\n";
    return 1;
  }
}
