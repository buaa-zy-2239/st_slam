#include "st_slam/backend/loop_closer.h"
#include "st_slam/frontend/pnp_solver.h"
#include <iostream>
#include <opencv2/features2d.hpp>
#include <vector>
#include <Eigen/SVD>
#include <numeric>

#ifdef HAS_DBOW3
// Try both include styles - DBoW3.h may be in DBoW3/ subdir or directly in include path
#ifdef __has_include
#if __has_include(<DBoW3/DBoW3.h>)
#include <DBoW3/DBoW3.h>
#elif __has_include(<DBoW3.h>)
#include <DBoW3.h>
#else
#error "DBoW3 headers not found!"
#endif
#else
// Fallback - try both paths
#include <DBoW3/DBoW3.h>
#endif
#else
// Stub implementation when DBoW3 is not available
namespace DBoW3 {
  struct BowVector : public std::map<unsigned int, double> {
  };
  struct QueryResult {
    unsigned int Id;
    double Score;
  };
  struct QueryResults : public std::vector<QueryResult> {
  };
  class Vocabulary {
  public:
    Vocabulary() = default;
    void load(const std::string&) {}
    void create(const std::vector<cv::Mat>&) {}
    void save(const std::string&) const {}
    int size() const { return 0; }
    void transform(const cv::Mat&, BowVector&) const {}
    double score(const BowVector&, const BowVector&) const { return 0.0; }
  };
  using EntryId = unsigned int;
  class Database {
  public:
    Database() = default;
    Database(const Vocabulary&, bool, int) {}
    void setVocabulary(const Vocabulary&) {}
    EntryId add(const cv::Mat&) { return 0; }
    void query(const cv::Mat&, QueryResults&, int, int) const {}
    void query(const BowVector&, QueryResults&, int, int) const {}
    void query(EntryId, QueryResults&, int) const {}
  };
}
#endif

namespace st_slam {

// 3D-3D Umeyama alignment: solve T_a_b s.t. pts_a ≈ T_a_b * pts_b
static bool Align3D3D_Umeyama(
    const std::vector<Vec3>& pts_a,
    const std::vector<Vec3>& pts_b,
    SE3& T_a_b) {
  int n = (int)pts_a.size();
  if (n < 3) return false;

  Vec3 ca = Vec3::Zero(), cb = Vec3::Zero();
  for (int i = 0; i < n; ++i) { ca += pts_a[i]; cb += pts_b[i]; }
  ca /= n; cb /= n;

  Eigen::Matrix3d H = Eigen::Matrix3d::Zero();
  for (int i = 0; i < n; ++i) {
    Vec3 da = pts_a[i] - ca, db = pts_b[i] - cb;
    H += da * db.transpose();
  }

  Eigen::JacobiSVD<Eigen::Matrix3d> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Matrix3d R = svd.matrixV() * svd.matrixU().transpose();
  if (R.determinant() < 0) {
    Eigen::Matrix3d V = svd.matrixV();
    V.col(2) = -V.col(2);
    R = V * svd.matrixU().transpose();
  }

  Vec3 t = ca - R * cb;
  T_a_b = SE3(Quat(R), t);
  return true;
}

// RANSAC: find best T_q_m from 3D-3D correspondences
static bool RunUmeyamaRANSAC(
    const std::vector<Vec3>& pts_q,
    const std::vector<Vec3>& pts_m,
    SE3& T_q_m,
    double threshold = 0.15,
    int max_iters = 200,
    int min_inliers = 12) {
  int n = (int)pts_q.size();
  if (n < 3 || n != (int)pts_m.size()) return false;

  std::vector<int> idx(n);
  std::iota(idx.begin(), idx.end(), 0);

  int best_cnt = 0;
  SE3 best_T;

  for (int iter = 0; iter < max_iters; ++iter) {
    std::vector<Vec3> sq, sm;
    for (int k = 0; k < 3; ++k) {
      int ri = rand() % n;
      sq.push_back(pts_q[ri]);
      sm.push_back(pts_m[ri]);
    }

    SE3 T;
    if (!Align3D3D_Umeyama(sq, sm, T)) continue;

    int cnt = 0;
    for (int i = 0; i < n; ++i) {
      if ((T * pts_m[i] - pts_q[i]).norm() < threshold) cnt++;
    }

    if (cnt > best_cnt) {
      best_cnt = cnt;
      best_T = T;
    }
  }

  if (best_cnt < min_inliers) return false;

  // Refine with all inliers
  std::vector<Vec3> iq, im;
  for (int i = 0; i < n; ++i) {
    if ((best_T * pts_m[i] - pts_q[i]).norm() < threshold) {
      iq.push_back(pts_q[i]);
      im.push_back(pts_m[i]);
    }
  }
  return Align3D3D_Umeyama(iq, im, T_q_m);
}

LoopCloser::LoopCloser(const std::string& vocab_path,
                       double fx, double fy,
                       double cx, double cy,
                       LocalMap* local_map)
  : vocab_(nullptr), db_(nullptr),
    fx_(fx), fy_(fy), cx_(cx), cy_(cy),
    local_map_(local_map) {
#ifdef HAS_DBOW3
  vocab_ = new DBoW3::Vocabulary();
  db_ = new DBoW3::Database();
  if (!vocab_path.empty()) {
    LoadVocabulary(vocab_path);
  }
#endif
  pnp_solver_ = std::make_unique<PnPSolver>(fx_, fy_, cx_, cy_, 6.0, 4, 200);
}

LoopCloser::~LoopCloser() {
#ifdef HAS_DBOW3
  delete (DBoW3::Vocabulary*)vocab_;
  delete (DBoW3::Database*)db_;
#endif
}

bool LoopCloser::LoadVocabulary(const std::string& vocab_path) {
#ifdef HAS_DBOW3
  try {
    auto* v = (DBoW3::Vocabulary*)vocab_;
    auto* d = (DBoW3::Database*)db_;
    v->load(vocab_path);
    d->setVocabulary(*v, true, 0);
    std::cout << "[LoopCloser] Loaded vocabulary from " << vocab_path
              << " (" << v->size() << " words)" << std::endl;
    return true;
  } catch (const std::exception& e) {
    std::cerr << "[LoopCloser] Failed: " << e.what() << std::endl;
    return false;
  }
#else
  (void)vocab_path;
  std::cerr << "[LoopCloser] DBoW3 not available (run: bash scripts/setup_colab.sh)" << std::endl;
  return false;
#endif
}

void LoopCloser::BuildVocabulary(const std::vector<cv::Mat>& all_descriptors) {
#ifdef HAS_DBOW3
  if (all_descriptors.empty()) return;
  auto* v = (DBoW3::Vocabulary*)vocab_;
  auto* d = (DBoW3::Database*)db_;
  std::cout << "[LoopCloser] Building vocabulary from "
            << all_descriptors.size() << " frames..." << std::endl;
  v->create(all_descriptors);
  d->setVocabulary(*v);
  std::cout << "[LoopCloser] Done: " << v->size() << " words" << std::endl;
#else
  (void)all_descriptors;
#endif
}

int LoopCloser::AddKeyFrame(int kf_id, const cv::Mat& descriptors) {
#ifdef HAS_DBOW3
  if (!db_ || descriptors.empty()) return -1;
  auto* v = (DBoW3::Vocabulary*)vocab_;
  auto* d = (DBoW3::Database*)db_;
  DBoW3::EntryId entry_id = d->add(descriptors);
  if ((int)kf_id_map_.size() <= (int)entry_id) {
    kf_id_map_.resize(entry_id + 1, -1);
    descriptors_cache_.resize(entry_id + 1);
  }
  kf_id_map_[entry_id] = kf_id;
  descriptors_cache_[entry_id] = descriptors.clone();
  return (int)entry_id;
#else
  (void)kf_id;
  (void)descriptors;
  return -1;
#endif
}

std::vector<LoopCandidate> LoopCloser::DetectCandidates(int current_kf_id,
                                                          float min_score,
                                                          int max_candidates) {
  std::vector<LoopCandidate> candidates;
#ifdef HAS_DBOW3
  if (!db_ || !vocab_) {
    return candidates;
  }

  auto* v = (DBoW3::Vocabulary*)vocab_;
  auto* d = (DBoW3::Database*)db_;

  int current_entry = -1;
  for (size_t i = 0; i < kf_id_map_.size(); ++i) {
    if (kf_id_map_[i] == current_kf_id) { current_entry = (int)i; break; }
  }
  if (current_entry < 0) {
    return candidates;
  }
  if (current_entry >= (int)descriptors_cache_.size()) {
    return candidates;
  }
  if (descriptors_cache_[current_entry].empty()) {
    return candidates;
  }

  // Calculate current BoW vector
  DBoW3::BowVector bow_vec_current;
  v->transform(descriptors_cache_[current_entry], bow_vec_current);
  
  // Calculate similarity score against ALL previous keyframes (manually)!
  std::vector<std::pair<double, DBoW3::EntryId>> scores_and_ids;
  for (DBoW3::EntryId e_id = 0; e_id < (DBoW3::EntryId)descriptors_cache_.size(); ++e_id) {
    if (e_id == current_entry) continue;
    if (descriptors_cache_[e_id].empty()) continue;
    
    // Compute BoW for this entry
    DBoW3::BowVector bow_vec_e;
    v->transform(descriptors_cache_[e_id], bow_vec_e);
    
    // Compute similarity score between current and e_id
    double score = v->score(bow_vec_current, bow_vec_e);
    
    // Store negative score for ascending sort (so highest score is first)
    scores_and_ids.emplace_back(-score, e_id);
  }
  
  // Sort by score descending
  std::sort(scores_and_ids.begin(), scores_and_ids.end());
  
  int added = 0;
  for (const auto& pair : scores_and_ids) {
    if (added >= max_candidates) break;
    double score = -pair.first;
    DBoW3::EntryId e_id = pair.second;
    
    int match_id = ((int)e_id < (int)kf_id_map_.size()) ? kf_id_map_[e_id] : -1;
    if (match_id < 0) continue;
    if (std::abs(match_id - current_kf_id) < 4) continue;
    
    LoopCandidate cand;
    cand.query_id = current_kf_id;
    cand.match_id = match_id;
    cand.similarity_score = score;
    candidates.push_back(cand);
    added++;
  }
#else
  (void)current_kf_id; (void)min_score; (void)max_candidates;
#endif
  
  return candidates;
}

bool LoopCloser::GeometricVerification(int query_kf_id, int match_kf_id,
                                         const Frame& query_frame,
                                         const Frame& match_frame,
                                         SE3& relative_pose) {
  if (!local_map_) return false;

  const KeyFrame* match_kf = local_map_->GetKeyFrame(match_kf_id);
  const KeyFrame* query_kf = local_map_->GetKeyFrame(query_kf_id);
  if (!match_kf || !query_kf) return false;

  cv::BFMatcher matcher(cv::NORM_HAMMING, false);
  std::vector<std::vector<cv::DMatch>> knn;
  matcher.knnMatch(query_frame.descriptors, match_kf->descriptors, knn, 2);

  std::vector<cv::DMatch> good_matches;
  for (size_t i = 0; i < knn.size(); ++i) {
    if (knn[i].size() >= 2 &&
        knn[i][0].distance < 0.75f * knn[i][1].distance) {
      good_matches.push_back(knn[i][0]);
    }
  }
  if ((int)good_matches.size() < 10) return false;

  // Build 3D-3D correspondences from STABLE MapPoints (both sides)
  // P_q = query_kf->pose.inverse() * query_mp->position  (stable, multi-frame)
  // P_m = match_kf->pose.inverse() * match_mp->position  (stable, multi-frame)
  SE3 T_q_w = query_kf->pose;
  SE3 T_m_w = match_kf->pose;
  std::vector<Vec3> pts_q, pts_m;
  for (const auto& m : good_matches) {
    int q_idx = m.queryIdx;
    int m_idx = m.trainIdx;

    // Query side: use stable map point from query_kf
    if (q_idx >= 0 && q_idx < (int)query_kf->map_points.size()) {
      int q_mp_id = query_kf->map_points[q_idx];
      if (q_mp_id >= 0) {
        const MapPoint* q_mp = local_map_->GetMapPoint(q_mp_id);
        if (q_mp && q_mp->position.norm() > 1e-6) {
          Vec3 p_q_cam = T_q_w.inverse() * q_mp->position;
          if (p_q_cam(2) >= 0.01) {
            // Match side: use stable map point from match_kf
            if (m_idx >= 0 && m_idx < (int)match_kf->map_points.size()) {
              int m_mp_id = match_kf->map_points[m_idx];
              if (m_mp_id >= 0) {
                const MapPoint* m_mp = local_map_->GetMapPoint(m_mp_id);
                if (m_mp && m_mp->position.norm() > 1e-6) {
                  Vec3 p_m_cam = T_m_w.inverse() * m_mp->position;
                  if (p_m_cam(2) >= 0.01) {
                    pts_q.push_back(p_q_cam);
                    pts_m.push_back(p_m_cam);
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  if ((int)pts_q.size() < 10) return false;

  SE3 T_q_m;
  if (!RunUmeyamaRANSAC(pts_q, pts_m, T_q_m, 0.2, 300, 10)) return false;

  double dist = T_q_m.trans.norm();
  AngleAxis aa(T_q_m.rot);
  double deg = aa.angle() * 180.0 / M_PI;

  if (dist > 8.0 || deg > 90) return false;

  // DRIFT CONSISTENCY + CHI-SQUARED PRUNING
  SE3 T_traj_rel = T_m_w.inverse() * query_kf->pose;
  SE3 drift = T_traj_rel.inverse() * T_q_m;
  double drift_dist = drift.trans.norm();
  AngleAxis drift_aa(drift.rot);
  double drift_deg = drift_aa.angle() * 180.0 / M_PI;

  // Compute chi2 with same info weights as PoseGraph::DefaultInformation(50,500)
  Vec3 rot_err = drift_aa.axis() * drift_aa.angle();
  Vec3 trans_err = drift.trans;
  double chi2 = 50.0 * rot_err.squaredNorm() + 500.0 * trans_err.squaredNorm();

  std::cout << "[LOOP] " << (int)pts_q.size() << "pts"
            << " drift=" << (drift_dist*100) << "cm " << drift_deg << "deg"
            << " chi2=" << chi2 << "\n";

  // χ²(6) threshold = 20 (ultra-tight, single best loop breaks through)
  if (drift_dist < 0.3 && drift_deg < 15 && chi2 < 20.0) {
    relative_pose = T_q_m;
    return true;
  }

  return false;
}

void LoopCloser::LoadOrBuildVocabulary(const std::vector<cv::Mat>& all_descriptors,
                                         const std::string& vocab_path) {
  if (!vocab_path.empty() && LoadVocabulary(vocab_path)) return;
  if (!all_descriptors.empty()) {
    BuildVocabulary(all_descriptors);
    std::string path = "st_slam_vocab.yml.gz";
    SaveVocabulary(path);
    std::cout << "[LoopCloser] Saved vocab to " << path << std::endl;
  }
}

void LoopCloser::SaveVocabulary(const std::string& path) const {
#ifdef HAS_DBOW3
  if (vocab_) ((DBoW3::Vocabulary*)vocab_)->save(path);
#else
  (void)path;
#endif
}

int LoopCloser::NumKeyFrames() const {
  return (int)kf_id_map_.size();
}

} // namespace st_slam
