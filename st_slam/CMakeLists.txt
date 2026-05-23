cmake_minimum_required(VERSION 3.16)
project(ST_SLAM VERSION 4.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
endif()

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(OpenCV 4.0 REQUIRED COMPONENTS core imgproc imgcodecs features2d calib3d highgui)
find_package(Ceres REQUIRED)
find_package(Boost REQUIRED COMPONENTS system thread filesystem program_options)

set(ST_SLAM_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)

include_directories(
  ${ST_SLAM_INCLUDE_DIR}
  ${EIGEN3_INCLUDE_DIR}
  ${OpenCV_INCLUDE_DIRS}
  ${CERES_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
)

add_definitions(-DST_SLAM_VERSION="${PROJECT_VERSION}")

# Optional: DBoW3 for loop closure
find_package(DBoW3 QUIET)
if(DBoW3_FOUND)
  message(STATUS "DBoW3 found - enabling loop closure")
  add_definitions(-DHAS_DBOW3)
  include_directories(${DBoW3_INCLUDE_DIRS})
  set(DBOW3_LIBS ${DBoW3_LIBRARIES})
else()
  message(STATUS "DBoW3 not found - loop closure disabled (install: bash scripts/setup_colab.sh)")
  set(DBOW3_LIBS "")
endif()

file(GLOB_RECURSE ST_SLAM_SOURCES src/*.cpp)

add_library(st_slam SHARED ${ST_SLAM_SOURCES})
target_link_libraries(st_slam
  ${OpenCV_LIBS}
  ${CERES_LIBRARIES}
  ${Boost_LIBRARIES}
  ${DBOW3_LIBS}
  pthread
)

add_subdirectory(apps)
