cmake_minimum_required(VERSION 3.10)
project(pgo_pose_sync_test)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_SOURCE_DIR "/home/zhang/ORB_SLAM_Learning/st_slam")

find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}
  ${EIGEN3_INCLUDE_DIR}
  ${OpenCV_INCLUDE_DIRS}
)

add_executable(pgo_pose_sync_test
  ${CMAKE_SOURCE_DIR}/tests/pgo_pose_sync_test.cpp
)

target_link_libraries(pgo_pose_sync_test
  ${OpenCV_LIBS}
  Eigen3::Eigen
)
