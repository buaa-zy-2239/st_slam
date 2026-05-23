#include "st_slam/system/thread_pool.h"
#include <iostream>
#include <pthread.h>
#include <sched.h>

namespace st_slam {

bool ThreadAffinityManager::SetThreadAffinity(std::thread& t, int core_id) {
  return SetThreadAffinity(t.native_handle(), core_id);
}

bool ThreadAffinityManager::SetThreadAffinity(pthread_t thread, int core_id) {
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(core_id, &cpuset);

  int result = pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset);
  if (result != 0) {
    std::cerr << "[ThreadAffinity] Failed to set affinity to core "
              << core_id << " (errno=" << result << ")" << std::endl;
    return false;
  }
  return true;
}

bool ThreadAffinityManager::SetRealtimePriority(pthread_t thread, int priority) {
  sched_param sch_params;
  sch_params.sched_priority = priority;
  int result = pthread_setschedparam(thread, SCHED_FIFO, &sch_params);
  if (result != 0) {
    std::cerr << "[ThreadAffinity] Failed to set SCHED_FIFO priority="
              << priority << " (errno=" << result << ")" << std::endl;
    return false;
  }
  return true;
}

bool ThreadAffinityManager::SetRoundRobinPriority(pthread_t thread, int priority) {
  sched_param sch_params;
  sch_params.sched_priority = priority;
  int result = pthread_setschedparam(thread, SCHED_RR, &sch_params);
  if (result != 0) {
    std::cerr << "[ThreadAffinity] Failed to set SCHED_RR priority="
              << priority << " (errno=" << result << ")" << std::endl;
    return false;
  }
  return true;
}

bool ThreadAffinityManager::SetBatchScheduling(pthread_t thread) {
  sched_param sch_params;
  sch_params.sched_priority = 0;
  int result = pthread_setschedparam(thread, SCHED_BATCH, &sch_params);
  if (result != 0) {
    std::cerr << "[ThreadAffinity] Failed to set SCHED_BATCH (errno="
              << result << ")" << std::endl;
    return false;
  }
  return true;
}

} // namespace st_slam
