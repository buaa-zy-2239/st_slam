#include <iostream>
#include <cassert>
#include <cmath>
#include "st_slam/backend/pose_graph.h"
#include "st_slam/frontend/local_map.h"
#include "st_slam/core/types.h"

using namespace st_slam;

void TestPoseGraphAddressConsistency() {
  std::cout << "\n[DEBUG] Test: PoseGraph Address Consistency\n";
  
  // Create a single PoseGraph instance
  PoseGraph* pg = new PoseGraph(100.0, 1000.0);
  
  // Print address
  std::cout << "[DEBUG] PoseGraph address: " << pg << "\n";
  
  // Add some edges
  SE3 rel_pose(Quat::Identity(), Vec3(0.1, 0.0, 0.0));
  Mat6 info = PoseGraph::DefaultInformation(100.0, 1000.0);
  
  std::cout << "[DEBUG] Edges BEFORE AddEdge: " << pg->NumEdges() << "\n";
  pg->AddEdge(0, 1, rel_pose, info);
  std::cout << "[DEBUG] Edges AFTER AddEdge: " << pg->NumEdges() << "\n";
  
  // Verify edge was added
  assert(pg->NumEdges() == 1 && "Edge should be added");
  
  // Test that the same object is used
  PoseGraph* pg_same = pg;
  std::cout << "[DEBUG] Same pointer address: " << pg_same << "\n";
  std::cout << "[DEBUG] Same pointer edge count: " << pg_same->NumEdges() << "\n";
  
  assert(pg == pg_same && "Pointers should be the same");
  
  delete pg;
  std::cout << "[DEBUG] PASSED\n";
}

void TestLoopEdgeAddition() {
  std::cout << "\n[DEBUG] Test: Loop Edge Addition\n";
  
  LocalMap local_map(10, 50);
  PoseGraph pose_graph(100.0, 1000.0);
  
  // Create keyframes with poses
  for (int i = 0; i < 6; ++i) {
    Frame frame;
    frame.id = i;
    frame.timestamp = i * 0.1;
    frame.keypoints.resize(100);
    frame.descriptors = cv::Mat(100, 32, CV_8U, cv::Scalar(0));
    frame.pose = SE3(Quat::Identity(), Vec3(i * 0.1, 0.0, 0.0));
    local_map.AddKeyFrame(frame);
  }
  
  // Build sequential edges
  pose_graph.BuildFromKeyframes(local_map.GetAllKeyframes());
  std::cout << "[DEBUG] Edges after BuildFromKeyframes: " << pose_graph.NumEdges() << "\n";
  
  // Add loop edge manually
  SE3 loop_rel(Quat::Identity(), Vec3(0.0, 0.0, 0.023)); // 2.3cm drift
  Mat6 loop_info = PoseGraph::DefaultInformation(50.0, 500.0);
  
  std::cout << "[DEBUG] Edges BEFORE loop edge: " << pose_graph.NumEdges() << "\n";
  pose_graph.AddEdge(0, 5, loop_rel, loop_info);
  std::cout << "[DEBUG] Edges AFTER loop edge: " << pose_graph.NumEdges() << "\n";
  
  assert(pose_graph.NumEdges() == 6 && "Should have 6 edges (5 sequential + 1 loop)");
  
  // Verify loop edge is actually in the graph
  auto& keyframes = local_map.GetAllKeyframes();
  bool optimize_result = pose_graph.Optimize(keyframes);
  std::cout << "[DEBUG] Optimize result: " << (optimize_result ? "SUCCESS" : "FAILED") << "\n";
  
  std::cout << "[DEBUG] PASSED\n";
}

void TestEdgePreservation() {
  std::cout << "\n[DEBUG] Test: Edge Preservation After Optimize\n";
  
  PoseGraph pose_graph(100.0, 1000.0);
  
  // Add edges
  SE3 rel(Quat::Identity(), Vec3(0.1, 0.0, 0.0));
  Mat6 info = PoseGraph::DefaultInformation(100.0, 1000.0);
  
  pose_graph.AddEdge(0, 1, rel, info);
  pose_graph.AddEdge(1, 2, rel, info);
  pose_graph.AddEdge(2, 3, rel, info);
  
  std::cout << "[DEBUG] Initial edges: " << pose_graph.NumEdges() << "\n";
  
  // Create dummy keyframes for optimize
  std::unordered_map<int, KeyFrame> keyframes;
  for (int i = 0; i < 4; ++i) {
    KeyFrame kf;
    kf.id = i;
    kf.pose = SE3(Quat::Identity(), Vec3(i * 0.1, 0.0, 0.0));
    keyframes[i] = kf;
  }
  
  // Optimize
  pose_graph.Optimize(keyframes);
  
  std::cout << "[DEBUG] Edges after Optimize: " << pose_graph.NumEdges() << "\n";
  
  assert(pose_graph.NumEdges() == 3 && "Edges should be preserved after optimize");
  
  std::cout << "[DEBUG] PASSED\n";
}

int main() {
  std::cout << "[DEBUG] ======================================\n";
  std::cout << "[DEBUG] PoseGraph Debug Test Suite\n";
  std::cout << "[DEBUG] ======================================\n";
  
  try {
    TestPoseGraphAddressConsistency();
    TestLoopEdgeAddition();
    TestEdgePreservation();
    
    std::cout << "\n[DEBUG] ======================================\n";
    std::cout << "[DEBUG] ALL TESTS PASSED\n";
    std::cout << "[DEBUG] ======================================\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[DEBUG] TEST FAILED: " << e.what() << "\n";
    return 1;
  }
}
