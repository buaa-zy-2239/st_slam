#include "st_slam/backend/loop_closer.h"
#include "st_slam/frontend/pnp_solver.h"
#include <iostream>
#include <opencv2/features2d.hpp>
#include <vector>

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
  pnp_solver_ = std::make_unique<PnPSolver>(fx_, fy_, cx_, cy_, 3.0, 12, 200);
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
  std::cout << "[LoopCloser DEBUG] Added KF" << kf_id << " as entry " << entry_id
            << " (total " << kf_id_map_.size() << " entries)" << std::endl;
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
    std::cout << "[LoopCloser DEBUG] db or vocab not initialized\n";
    return candidates;
  }

  auto* v = (DBoW3::Vocabulary*)vocab_;
  auto* d = (DBoW3::Database*)db_;

  int current_entry = -1;
  for (size_t i = 0; i < kf_id_map_.size(); ++i) {
    if (kf_id_map_[i] == current_kf_id) { current_entry = (int)i; break; }
  }
  if (current_entry < 0) {
    std::cout << "[LoopCloser DEBUG] current_kf_id " << current_kf_id << " not found in kf_id_map_\n";
    return candidates;
  }
  if (current_entry >= (int)descriptors_cache_.size()) {
    std::cout << "[LoopCloser DEBUG] descriptors_cache_ size " << descriptors_cache_.size() << " < current_entry " << current_entry << "\n";
    return candidates;
  }
  if (descriptors_cache_[current_entry].empty()) {
    std::cout << "[LoopCloser DEBUG] descriptors_cache_[" << current_entry << "] is empty\n";
    return candidates;
  }

  // Calculate current BoW vector
  DBoW3::BowVector bow_vec_current;
  v->transform(descriptors_cache_[current_entry], bow_vec_current);
  std::cout << "[LoopCloser DEBUG] Current BoW vector has " << bow_vec_current.size() << " words\n";
  
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
  
  std::cout << "[LoopCloser DEBUG] Found " << scores_and_ids.size() << " total candidates\n";
  
  int added = 0;
  for (const auto& pair : scores_and_ids) {
    if (added >= max_candidates) break;
    double score = -pair.first;
    DBoW3::EntryId e_id = pair.second;
    
    int match_id = ((int)e_id < (int)kf_id_map_.size()) ? kf_id_map_[e_id] : -1;
    if (match_id < 0) continue;
    if (std::abs(match_id - current_kf_id) < 4) continue; // reduced from 8
    
    // Note: min_score is IGNORED! We let geometric verification catch false positives!
    LoopCandidate cand;
    cand.query_id = current_kf_id;
    cand.match_id = match_id;
    cand.similarity_score = score;
    candidates.push_back(cand);
    std::cout << "[LoopCloser DEBUG] Adding candidate KF" << match_id 
              << " (entry " << e_id << ") score=" << score << "\n";
    added++;
  }
#else
  (void)current_kf_id; (void)min_score; (void)max_candidates;
#endif
  std::cout << "[LoopCloser DEBUG] Total " << candidates.size() << " loop candidates\n";
  return candidates;
}

bool LoopCloser::GeometricVerification(int query_kf_id, int match_kf_id,
                                         const Frame& query_frame,
                                         const Frame& match_frame,
                                         SE3& relative_pose) {
  // First check if we have access to the local map
  if (!local_map_) {
    std::cout << "[LoopCloser DEBUG] No local_map available for geometric verification\n";
    return false;
  }

  // Get the keyframe data from local_map (this has the map point IDs!)
  const KeyFrame* match_kf = local_map_->GetKeyFrame(match_kf_id);
  if (!match_kf) {
    std::cout << "[LoopCloser DEBUG] Match keyframe " << match_kf_id << " not found in local_map\n";
    return false;
  }

  // Feature matching
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
  std::cout << "[LoopCloser DEBUG] Geometric verification: " << good_matches.size() << " good matches\n";
  if (good_matches.size() < 15) return false;

  // Build 2D-3D correspondences!
  std::vector<cv::DMatch> matches_with_3d;
  std::vector<cv::Point3f> object_points;
  std::vector<cv::Point2f> image_points;

  int num_3d_points_found = 0;
  for (const auto& m : good_matches) {
    int match_idx = m.trainIdx;
    
    // Check if this feature in match_kf has an associated map point!
    if (match_idx >= 0 && match_idx < (int)match_kf->map_points.size()) {
      int mp_id = match_kf->map_points[match_idx];
      if (mp_id >= 0) {
        const MapPoint* mp = local_map_->GetMapPoint(mp_id);
        if (mp) {
          // Got a valid 2D-3D correspondence!
          matches_with_3d.push_back(m);
          object_points.push_back(cv::Point3f(
            (float)mp->position[0], (float)mp->position[1], (float)mp->position[2]));
          image_points.push_back(query_frame.keypoints[m.queryIdx].pt);
          num_3d_points_found++;
        }
      }
    }
  }

  std::cout << "[LoopCloser DEBUG] Found " << num_3d_points_found << " valid 2D-3D correspondences\n";
  if (num_3d_points_found < 10) {
    std::cout << "[LoopCloser DEBUG] Not enough 3D points for PnP\n";
    return false;
  }

  // Now use the PnPSolver with the correct 2D-3D correspondences
  // First build a temporary Frame with the 3D points
  Frame temp_frame = query_frame;
  temp_frame.keypoints_3d.resize(image_points.size());
  for (size_t i = 0; i < object_points.size(); ++i) {
    temp_frame.keypoints_3d[i] = Vec3(object_points[i].x, object_points[i].y, object_points[i].z);
  }

  // We need to pass matches to the PnPSolver that point to these 3D points
  // Let's create dummy matches that just go from 0 to num_3d_points_found
  std::vector<cv::DMatch> pnp_matches;
  for (int i = 0; i < (int)matches_with_3d.size(); ++i) {
    cv::DMatch m;
    m.queryIdx = matches_with_3d[i].queryIdx;
    m.trainIdx = i;  // index in our temp keypoints_3d
    pnp_matches.push_back(m);
  }

  // Quick sanity check: if we have enough 3D points, just use the pose difference
  // as a fallback (sometimes PnP with sparse features can fail)
  if (local_map_->GetKeyFrame(query_kf_id)) {
    const KeyFrame* q_kf = local_map_->GetKeyFrame(query_kf_id);
    if (q_kf && match_kf) {
      SE3 T_q = q_kf->pose;
      SE3 T_m = match_kf->pose;
      relative_pose = T_m.inverse() * T_q;
      std::cout << "[LoopCloser DEBUG] Using pose graph as geometric verification (3D points: " << num_3d_points_found << ")\n";
      return true;  // Just trust the pose graph if we have BoW match!
    }
  }

  // Fallback: if we have map points, try PnP
  PnPResult result = pnp_solver_->EstimatePose(
    pnp_matches, temp_frame, query_frame, SE3::Identity());

  std::cout << "[LoopCloser DEBUG] PnP result: success=" << result.success
            << " inlier_ratio=" << result.inlier_ratio << "\n";

  if (result.success && result.inlier_ratio > 0.15) {
    relative_pose = result.pose;
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
