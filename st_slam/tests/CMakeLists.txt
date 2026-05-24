cmake_minimum_required(VERSION 3.10)
project(st_slam_tests)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED core imgproc features2d calib3d)
find_package(Ceres REQUIRED)
find_package(Threads REQUIRED)

set(ST_SLAM_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/..)

include_directories(
  ${ST_SLAM_ROOT}/include
  ${EIGEN3_INCLUDE_DIR}
  ${OpenCV_INCLUDE_DIRS}
  ${CERES_INCLUDE_DIRS}
)

add_executable(test_initialization
  test_initialization.cpp
)

target_link_libraries(test_initialization
  ${ST_SLAM_ROOT}/build/libst_slam.so
  ${OpenCV_LIBS}
  ${CERES_LIBRARIES}
  Threads::Threads
)

add_test(NAME InitializationTest COMMAND test_initialization)

add_executable(test_pose_graph
  test_pose_graph.cpp
)

target_link_libraries(test_pose_graph
  ${ST_SLAM_ROOT}/build/libst_slam.so
  ${OpenCV_LIBS}
  ${CERES_LIBRARIES}
  Threads::Threads
)

add_test(NAME PoseGraphTest COMMAND test_pose_graph)

add_executable(test_pose_graph_debug
  test_pose_graph_debug.cpp
)

target_link_libraries(test_pose_graph_debug
  ${ST_SLAM_ROOT}/build/libst_slam.so
  ${OpenCV_LIBS}
  ${CERES_LIBRARIES}
  Threads::Threads
)

add_test(NAME PoseGraphDebugTest COMMAND test_pose_graph_debug)

add_executable(test_loop_edge_preserve
  test_loop_edge_preserve.cpp
)

target_link_libraries(test_loop_edge_preserve
  ${ST_SLAM_ROOT}/build/libst_slam.so
  ${OpenCV_LIBS}
  ${CERES_LIBRARIES}
  Threads::Threads
)

add_test(NAME LoopEdgePreserveTest COMMAND test_loop_edge_preserve)
