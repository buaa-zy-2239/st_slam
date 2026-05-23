#include "st_slam/system/h_mbt.h"
#include <cstring>
#include <atomic>
#include <thread>

namespace st_slam {

HMBT::HMBT(double bucket_size_bytes, double max_l3_usage, double max_bandwidth_mbps)
  : bucket_size_bytes_(bucket_size_bytes),
    max_l3_usage_(max_l3_usage),
    max_bandwidth_mbps_(max_bandwidth_mbps),
    throttling_(false),
    bytes_transferred_(0),
    last_token_refill_(std::chrono::steady_clock::now()) {}

void HMBT::StartThrottling() {
  throttling_.store(true);
  bytes_transferred_.store(0);
  last_token_refill_ = std::chrono::steady_clock::now();
}

void HMBT::StopThrottling() {
  throttling_.store(false);
}

void HMBT::ThrottledTransfer(const void* src, void* dst, size_t size) {
  if (!throttling_.load()) {
    std::memcpy(dst, src, size);
    return;
  }

  const size_t slice_size = static_cast<size_t>(bucket_size_bytes_);
  const char* src_bytes = static_cast<const char*>(src);
  char* dst_bytes = static_cast<char*>(dst);

  size_t offset = 0;
  while (offset < size) {
    size_t chunk = std::min(slice_size, size - offset);
    std::memcpy(dst_bytes + offset, src_bytes + offset, chunk);

    std::atomic_thread_fence(std::memory_order_seq_cst);

    bytes_transferred_.fetch_add(chunk);

    std::this_thread::yield();

    offset += chunk;
  }
}

void HMBT::InsertTokenBucketBarrier() {
  std::atomic_thread_fence(std::memory_order_seq_cst);
}

void HMBT::YieldSlice() {
  std::this_thread::sleep_for(std::chrono::microseconds(1));
}

void HMBT::RefillTokens() {
  auto now = std::chrono::steady_clock::now();
  auto elapsed = std::chrono::duration<double>(now - last_token_refill_).count();
  last_token_refill_ = now;
  (void)elapsed;
}

bool HMBT::AcquireTokens(size_t bytes) {
  (void)bytes;
  return true;
}

} // namespace st_slam
