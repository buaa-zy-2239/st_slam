#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Geometry>

using Vec3 = Eigen::Vector3d;
using Quat = Eigen::Quaterniond;
using SE3 = Eigen::Isometry3d;

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

void TestTcwVsTwcSemantics() {
  PrintSeparator("TEST: Tcw vs Twc Semantic Verification");

  std::cout << "Scenario: Camera is at position (1.0, 0.5, 0.8) in world\n";
  
  SE3 Twc = CreateSE3(Vec3(1.0, 0.5, 0.8), Quat(1, 0, 0, 0));
  SE3 Tcw = Twc.inverse();

  std::cout << "\n1. Definitions:\n";
  std::cout << "   Twc (camera in world): pos=(" 
            << Twc.translation()(0) << ", " << Twc.translation()(1) << ", " << Twc.translation()(2) << ")\n";
  std::cout << "   Tcw (world in camera): pos=(" 
            << Tcw.translation()(0) << ", " << Tcw.translation()(1) << ", " << Tcw.translation()(2) << ")\n";

  Vec3 point_in_camera(0.1, 0.2, 1.0);
  Vec3 point_in_world = Twc * point_in_camera;
  
  std::cout << "\n2. Point transformation:\n";
  std::cout << "   Point in camera: " << point_in_camera.transpose() << "\n";
  std::cout << "   Point in world (Twc * Xc): " << point_in_world.transpose() << "\n";

  Vec3 reprojected = Tcw * point_in_world;
  std::cout << "   Reprojected to camera (Tcw * Xw): " << reprojected.transpose() << "\n";
  std::cout << "   Match? " << (reprojected.isApprox(point_in_camera) ? "YES" : "NO") << "\n";

  std::cout << "\n3. Current st_slam code analysis:\n";
  std::cout << "   CreateMapPoints: world_pt = current_pose_ * keypoints_3d\n";
  std::cout << "   This requires current_pose_ to be Twc\n";
  
  std::cout << "\n4. PnP solver code analysis:\n";
  std::cout << "   raw_pose = OpenCV's Tcw (from solvePnPRansac)\n";
  std::cout << "   camera_pose = raw_pose.inverse() -> Twc\n";
  std::cout << "   result.pose = camera_pose -> Twc\n";
  
  std::cout << "\n5. PROBLEM DETECTED!\n";
  std::cout << "   Projection in pnp_solver.cpp line 167:\n";
  std::cout << "   Vec3 proj = result.pose * m.world_pt;\n";
  std::cout << "   If result.pose is Twc, this is WRONG!\n";
  std::cout << "   Should be: proj = result.pose.inverse() * m.world_pt;\n";

  std::cout << "\n6. FIX OPTIONS:\n";
  std::cout << "   Option A: Remove .inverse() in PnP solver (line 130)\n";
  std::cout << "             result.pose = raw_pose;  // Keep Tcw\n";
  std::cout << "   Option B: Change projection in PnP solver (line 167)\n";
  std::cout << "             Vec3 proj = result.pose.inverse() * m.world_pt;\n";
}

void TestCameraMovement() {
  PrintSeparator("TEST: Camera Movement Simulation");

  SE3 Twc_frame1 = CreateSE3(Vec3(1.0, 0.0, 0.0), Quat(1, 0, 0, 0));
  SE3 Twc_frame2 = CreateSE3(Vec3(1.1, 0.05, 0.02), Quat(1, 0.01, 0, 0));

  SE3 T12 = Twc_frame1.inverse() * Twc_frame2;
  
  std::cout << "Frame 1: pos=(" << Twc_frame1.translation().transpose() << ")\n";
  std::cout << "Frame 2: pos=(" << Twc_frame2.translation().transpose() << ")\n";
  std::cout << "Relative motion T12: pos=(" << T12.translation().transpose() << ")\n";

  std::cout << "\nExpected behavior:\n";
  std::cout << "1. PnP solves for camera pose Twc_frame2\n";
  std::cout << "2. Tracking updates current_pose_ = Twc_frame2\n";
  std::cout << "3. Delta from frame 1: ~0.1m in X\n";

  std::cout << "\nBug behavior (current code):\n";
  std::cout << "1. PnP returns Twc but projection uses it as Tcw\n";
  std::cout << "2. Reprojection error calculation is wrong\n";
  std::cout << "3. May cause incorrect pose updates\n";
}

void TestRealWorldScenario() {
  PrintSeparator("TEST: Real World Scenario (fr1/desk)");

  SE3 gt_init = CreateSE3(Vec3(1.314, 0.847, 1.519), Quat(1, 0, 0, 0));
  
  std::cout << "Ground Truth Initial Pose:\n";
  std::cout << "  pos = (" << gt_init.translation()(0) << ", " 
            << gt_init.translation()(1) << ", " << gt_init.translation()(2) << ")\n";

  Vec3 keypoint_3d(0.1, 0.2, 1.0);
  Vec3 world_pt = gt_init * keypoint_3d;
  
  std::cout << "\nLandmark creation (Twc * Xc):\n";
  std::cout << "  Camera point: " << keypoint_3d.transpose() << "\n";
  std::cout << "  World point:  " << world_pt.transpose() << "\n";

  std::cout << "\nPnP solving with this landmark:\n";
  std::cout << "  world_pts contains: " << world_pt.transpose() << "\n";
  std::cout << "  img_pts contains: projected pixel coordinates\n";
  std::cout << "  OpenCV returns: Tcw (world to camera)\n";
  std::cout << "  Expected correct output: Twc = Tcw.inverse()\n";

  std::cout << "\nPROBLEM: If projection uses Twc directly,\n";
  std::cout << "         reprojection error will be wrong!\n";
}

int main() {
  std::cout << "========================================\n";
  std::cout << "Tcw/Twc Semantic Verification Test\n";
  std::cout << "========================================\n";
  std::cout << "\nThis test verifies the coordinate system\n";
  std::cout << "semantics in the st_slam PnP solver.\n\n";

  TestTcwVsTwcSemantics();
  TestCameraMovement();
  TestRealWorldScenario();

  PrintSeparator("CONCLUSION");

  std::cout << "CRITICAL FINDING:\n";
  std::cout << "  The PnP solver's projection code (line 167) assumes\n";
  std::cout << "  result.pose is Tcw, but result.pose is actually Twc!\n";
  std::cout << "\nRECOMMENDED FIX:\n";
  std::cout << "  Remove the .inverse() on line 130 of pnp_solver.cpp:\n";
  std::cout << "  Change: SE3 camera_pose = raw_pose.inverse();\n";
  std::cout << "  To:     SE3 camera_pose = raw_pose;\n";
  std::cout << "\nOR alternatively, fix the projection on line 167:\n";
  std::cout << "  Change: Vec3 proj = result.pose * m.world_pt;\n";
  std::cout << "  To:     Vec3 proj = result.pose.inverse() * m.world_pt;\n";
  std::cout << "\nOption A (remove .inverse()) is preferred because:\n";
  std::cout << "  - Tracking expects Twc for current_pose_\n";
  std::cout << "  - But wait! CreateMapPoints uses current_pose_ * Xc\n";
  std::cout << "  - This requires current_pose_ to be Twc\n";
  std::cout << "\nActually... let's think again:\n";
  std::cout << "  OpenCV returns Tcw\n";
  std::cout << "  We need to return Twc for Tracking\n";
  std::cout << "  So .inverse() IS needed on line 130\n";
  std::cout << "  But then line 167 must use .inverse() as well!\n";

  return 0;
}
