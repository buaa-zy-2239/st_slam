#include "st_slam/frontend/tracking.h"
#include "st_slam/core/config.h"
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

namespace py = pybind11;
namespace st_slam {

class HabitatTracker {
public:
  HabitatTracker() {
    STSLAMConfig config;
    config.use_loop_closure = false;
    tracking_ = std::make_unique<Tracking>(config);
    frame_count_ = 0;
  }

  void step(const py::array_t<uint8_t>& rgb,
            const py::array_t<float>& depth,
            double fx, double fy, double cx, double cy) {
    py::buffer_info rbuf = rgb.request();
    py::buffer_info dbuf = depth.request();

    cv::Mat cv_rgb((int)rbuf.shape[0], (int)rbuf.shape[1], CV_8UC3, rbuf.ptr);
    cv::Mat cv_depth((int)dbuf.shape[0], (int)dbuf.shape[1], CV_32FC1, dbuf.ptr);

    double ts = frame_count_++ * 0.033;
    tracking_->StepHabitat(cv_rgb, cv_depth, ts);
  }

  std::vector<double> get_smooth_pose() const {
    SE3 p = tracking_->GetSmoothPose();
    return {p.trans(0), p.trans(1), p.trans(2),
            p.rot.x(), p.rot.y(), p.rot.z(), p.rot.w()};
  }

  std::vector<double> get_global_pose() const {
    SE3 p = tracking_->GetCurrentPose();
    return {p.trans(0), p.trans(1), p.trans(2),
            p.rot.x(), p.rot.y(), p.rot.z(), p.rot.w()};
  }

  py::array_t<uint8_t> get_local_costmap() const {
    auto& cm = tracking_->GetLocalCostmap();
    const cv::Mat& grid = cm.grid();
    py::array_t<uint8_t> result({grid.rows, grid.cols});
    auto buf = result.mutable_unchecked<2>();
    for (int r = 0; r < grid.rows; ++r)
      for (int c = 0; c < grid.cols; ++c)
        buf(r, c) = grid.at<uchar>(r, c);
    return result;
  }

  std::string get_state() const {
    auto state = tracking_->GetState();
    switch (state) {
      case TrackingState::INITIALIZING: return "INITIALIZING";
      case TrackingState::TRACKING_GOOD: return "TRACKING_GOOD";
      case TrackingState::TRACKING_LOST: return "TRACKING_LOST";
      case TrackingState::TRACKING_DEGENERATE_C: return "DEGENERATE";
      default: return "UNKNOWN";
    }
  }

  int get_num_loops() const { return tracking_->GetNumLoopsDetected(); }
  int get_num_kfs() const { return tracking_->GetLocalMap().NumKeyFrames(); }
  int get_num_mps() const { return tracking_->GetLocalMap().NumMapPoints(); }
  double get_success_rate() const { return tracking_->GetLastFrame().id > 0 ? 1.0 : 0.0; }

private:
  std::unique_ptr<Tracking> tracking_;
  int frame_count_;
};

} // namespace st_slam

PYBIND11_MODULE(st_slam_tracker, m) {
  m.doc() = "st_slam 5.0 Habitat-Sim tracker bridge";

  py::class_<st_slam::HabitatTracker>(m, "Tracker")
    .def(py::init<>())
    .def("step", &st_slam::HabitatTracker::step,
         py::arg("rgb"), py::arg("depth"),
         py::arg("fx") = 525.0, py::arg("fy") = 525.0,
         py::arg("cx") = 319.5, py::arg("cy") = 239.5)
    .def("get_smooth_pose", &st_slam::HabitatTracker::get_smooth_pose)
    .def("get_global_pose", &st_slam::HabitatTracker::get_global_pose)
    .def("get_local_costmap", &st_slam::HabitatTracker::get_local_costmap)
    .def("get_state", &st_slam::HabitatTracker::get_state)
    .def("get_num_loops", &st_slam::HabitatTracker::get_num_loops)
    .def("get_num_kfs", &st_slam::HabitatTracker::get_num_kfs)
    .def("get_num_mps", &st_slam::HabitatTracker::get_num_mps);
}
