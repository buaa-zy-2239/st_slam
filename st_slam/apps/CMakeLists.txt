add_executable(st_slam_main
  st_slam_main.cpp
  dataset_reader.cpp
)

target_link_libraries(st_slam_main
  st_slam
  ${OpenCV_LIBS}
  ${CERES_LIBRARIES}
  ${Boost_LIBRARIES}
  pthread
)
