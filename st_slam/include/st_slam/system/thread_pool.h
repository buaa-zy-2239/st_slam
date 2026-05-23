#ifndef ST_SLAM_THREAD_POOL_H
#define ST_SLAM_THREAD_POOL_H

#include "st_slam/core/types.h"
#include <thread>
#include <vector>
#include <functional>
#include <future>

namespace st_slam {

class ThreadAffinityManager {
public:
  static bool SetThreadAffinity(std::thread& t, int core_id);

  static bool SetThreadAffinity(pthread_t thread, int core_id);

  static bool SetRealtimePriority(pthread_t thread, int priority);

  static bool SetRoundRobinPriority(pthread_t thread, int priority);

  static bool SetBatchScheduling(pthread_t thread);
};

} // namespace st_slam

#endif
