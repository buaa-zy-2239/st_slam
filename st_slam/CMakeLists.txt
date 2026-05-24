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

# Optional: DBoW3 for loop closure (manually enabled)
option(USE_DBOW3 "Use DBoW3 for loop closure detection" OFF)
if(USE_DBOW3)
  # Try to find DBoW3 - first check standard locations
  find_path(DBOW3_INCLUDE_DIR NAMES DBoW3.h HINTS /usr/local/include /usr/local/include/DBoW3)
  find_library(DBOW3_LIBRARY NAMES DBoW3 HINTS /usr/local/lib)
  
  if(DBOW3_INCLUDE_DIR AND DBOW3_LIBRARY)
    add_definitions(-DHAS_DBOW3)
    # Handle both /usr/local/include and /usr/local/include/DBoW3 cases
    include_directories(${DBOW3_INCLUDE_DIR})
    if(DBOW3_INCLUDE_DIR MATCHES "/DBoW3$")
      get_filename_component(DBOW3_PARENT_INCLUDE ${DBOW3_INCLUDE_DIR} DIRECTORY)
      include_directories(${DBOW3_PARENT_INCLUDE})
    endif()
    set(DBOW3_LIBS ${DBOW3_LIBRARY})
    message(STATUS "DBoW3 ENABLED - loop closure active")
    message(STATUS "  Include: ${DBOW3_INCLUDE_DIR}")
    message(STATUS "  Library: ${DBOW3_LIBRARY}")
  else()
    message(WARNING "DBoW3 not found - disabling loop closure")
    set(USE_DBOW3 OFF)
    set(DBOW3_LIBS "")
  endif()
else()
  message(STATUS "DBoW3 disabled (use -DUSE_DBOW3=ON to enable)")
  set(DBOW3_LIBS "")
endif()

file(GLOB_RECURSE ST_SLAM_SOURCES src/*.cpp)
list(FILTER ST_SLAM_SOURCES EXCLUDE REGEX ".*bridge/.*")

add_library(st_slam SHARED ${ST_SLAM_SOURCES})
target_link_libraries(st_slam
  ${OpenCV_LIBS}
  ${CERES_LIBRARIES}
  ${Boost_LIBRARIES}
  ${DBOW3_LIBS}
  pthread
)

add_subdirectory(apps)

# Optional pybind11 bridge for Habitat-Sim
option(BUILD_PYTHON_BINDINGS "Build pybind11 Python bindings for Habitat-Sim bridge" OFF)
if(BUILD_PYTHON_BINDINGS)
  find_package(pybind11 REQUIRED)
  pybind11_add_module(st_slam_tracker src/bridge/py_st_slam.cpp)
  target_link_libraries(st_slam_tracker PRIVATE st_slam)
  target_include_directories(st_slam_tracker PRIVATE ${ST_SLAM_INCLUDE_DIR})
  set_target_properties(st_slam_tracker PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python")
endif()
