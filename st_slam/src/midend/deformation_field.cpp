#include "st_slam/midend/deformation_field.h"
#include "st_slam/core/math_utils.h"
#include <Eigen/LU>
#include <cmath>
#include <algorithm>

namespace st_slam {

DeformationField::DeformationField(double voxel_size, int resolution,
                                     double regularization_weight,
                                     double epsilon)
  : regularization_weight_(regularization_weight),
    epsilon_(epsilon) {
  voxel_grid_ = std::make_shared<VoxelGrid>(voxel_size, resolution);
  vb_ks_ = std::make_unique<VBKS>(voxel_size, resolution, 0.05);
  tdq_bs_ = std::make_unique<TDQBS>(0.02);
}

void DeformationField::Initialize(const std::vector<Surfel>& surfels) {
  std::vector<Surfel> copy = surfels;
  vb_ks_->SetSurfelVoxelAssignments(copy);
}

Vec3 DeformationField::DeformPoint(const Vec3& point) const {
  Vec3 deformed = vb_ks_->DeformPoint(point);
  return deformed;
}

Vec3 DeformationField::DeformNormal(const Vec3& point, const Vec3& normal) const {
  VoxelKey key = voxel_grid_->GetVoxelKey(point);
  auto nodes = voxel_grid_->GetActiveNodes();

  auto it = std::find_if(nodes.begin(), nodes.end(),
    [&key](const VoxelControlNode& n) {
      return n.grid_idx[0] == key.x && n.grid_idx[1] == key.y && n.grid_idx[2] == key.z;
    });

  if (it == nodes.end()) return normal;

  Mat3 J = it->rotation.toRotationMatrix();

  return adjugate_normal_transform(J, normal);
}

void DeformationField::AddSurfels(const std::vector<Surfel>& surfels) {
  std::vector<Surfel> copy = surfels;
  vb_ks_->SetSurfelVoxelAssignments(copy);
}

Mat3 DeformationField::compute_adjugate(const Mat3& J) {
  Mat3 adj;
  adj(0,0) = J(1,1)*J(2,2) - J(1,2)*J(2,1);
  adj(0,1) = J(0,2)*J(2,1) - J(0,1)*J(2,2);
  adj(0,2) = J(0,1)*J(1,2) - J(0,2)*J(1,1);
  adj(1,0) = J(1,2)*J(2,0) - J(1,0)*J(2,2);
  adj(1,1) = J(0,0)*J(2,2) - J(0,2)*J(2,0);
  adj(1,2) = J(0,2)*J(1,0) - J(0,0)*J(1,2);
  adj(2,0) = J(1,0)*J(2,1) - J(1,1)*J(2,0);
  adj(2,1) = J(0,1)*J(2,0) - J(0,0)*J(2,1);
  adj(2,2) = J(0,0)*J(1,1) - J(0,1)*J(1,0);
  return adj;
}

Vec3 DeformationField::adjugate_normal_transform(const Mat3& J, const Vec3& N) {
  Mat3 adj = compute_adjugate(J);
  double det = J.determinant();
  double omega = std::cbrt(det * det + 1e-8);

  Vec3 N_transformed = (1.0 / omega) * adj.transpose() * N;
  return N_transformed.normalized();
}

} // namespace st_slam
