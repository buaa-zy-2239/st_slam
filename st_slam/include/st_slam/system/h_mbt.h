#ifndef ST_SLAM_H_MBT_H
#define ST_SLAM_H_MBT_H

#include <atomic>
#include <thread>
#include <chrono>

namespace st_slam {

class HMBT {
public:
  explicit HMBT(double bucket_size_bytes = 64.0,
                double max_l3_usage = 0.15,
                double max_bandwidth_mbps = 100.0);

  void StartThrottling();
  void StopThrottling();

  void ThrottledTransfer(const void* src, void* dst, size_t size);

  void InsertTokenBucketBarrier();

  void YieldSlice();

  bool IsThrottling() const { return throttling_.load(); }

  void SetMaxBandwidth(double mbps) { max_bandwidth_mbps_ = mbps; }

private:
  double bucket_size_bytes_;
  double max_l3_usage_;
  double max_bandwidth_mbps_;
  std::atomic<bool> throttling_;
  std::atomic<size_t> bytes_transferred_;
  std::chrono::steady_clock::time_point last_token_refill_;

  void RefillTokens();
  bool AcquireTokens(size_t bytes);
};

} // namespace st_slam

#endif
