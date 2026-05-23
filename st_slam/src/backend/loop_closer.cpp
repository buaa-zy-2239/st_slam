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
  };
  using EntryId = unsigned int;
  class Database {
  public:
    Database() = default;
    Database(const Vocabulary&, bool, int) {}
    void setVocabulary(const Vocabulary&) {}
    EntryId add(const cv::Mat&) { return 0; }
    void query(const cv::Mat&, QueryResults&, int, int) const {}
    void query(EntryId, QueryResults&, int) const {}
  };
}
#endif

namespace st_slam {

LoopCloser::LoopCloser(const std::string& vocab_path,
                        double fx, double fy,
                        double cx, double cy)
  : vocab_(nullptr), db_(nullptr),
    fx_(fx), fy_(fy), cx_(cx), cy_(cy) {
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
  if (!db_ || !vocab_) return candidates;

  auto* v = (DBoW3::Vocabulary*)vocab_;
  auto* d = (DBoW3::Database*)db_;

  int current_entry = -1;
  for (size_t i = 0; i < kf_id_map_.size(); ++i) {
    if (kf_id_map_[i] == current_kf_id) { current_entry = (int)i; break; }
  }
  if (current_entry < 0) return candidates;

  DBoW3::QueryResults ret;
  DBoW3::BowVector bow_vec;
  v->transform(descriptors_cache_[current_entry], bow_vec);
  d->query(bow_vec, ret, max_candidates * 3);

  int added = 0;
  for (const auto& qr : ret) {
    if (added >= max_candidates) break;
    int match_id = (qr.Id < (int)kf_id_map_.size()) ? kf_id_map_[qr.Id] : -1;
    if (match_id < 0) continue;
    if (std::abs(match_id - current_kf_id) < 15) continue;
    if (qr.Score < min_score) continue;

    LoopCandidate cand;
    cand.query_id = current_kf_id;
    cand.match_id = match_id;
    cand.similarity_score = qr.Score;
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
  cv::BFMatcher matcher(cv::NORM_HAMMING, false);
  std::vector<std::vector<cv::DMatch>> knn;
  matcher.knnMatch(query_frame.descriptors, match_frame.descriptors, knn, 2);

  std::vector<cv::DMatch> good_matches;
  for (size_t i = 0; i < knn.size(); ++i) {
    if (knn[i].size() >= 2 &&
        knn[i][0].distance < 0.75f * knn[i][1].distance) {
      good_matches.push_back(knn[i][0]);
    }
  }
  if (good_matches.size() < 20) return false;

  PnPResult result = pnp_solver_->EstimatePose(
    good_matches, match_frame, query_frame, SE3::Identity());

  if (result.success && result.inlier_ratio > 0.25) {
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
