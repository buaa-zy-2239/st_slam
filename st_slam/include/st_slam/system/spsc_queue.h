#ifndef ST_SLAM_SPSC_QUEUE_H
#define ST_SLAM_SPSC_QUEUE_H

#include <atomic>
#include <memory>
#include <type_traits>

namespace st_slam {

template<typename T>
class SPSCQueue {
  static_assert(std::is_move_constructible<T>::value,
                "T must be move constructible");

public:
  explicit SPSCQueue(size_t capacity)
    : capacity_(capacity), buffer_(std::make_unique<std::aligned_storage_t<sizeof(T), alignof(T)>[]>(capacity)),
      write_idx_(0), read_idx_(0) {}

  ~SPSCQueue() {
    T* ptr;
    while (Dequeue(ptr)) {
      ptr->~T();
    }
  }

  template<typename U>
  bool Enqueue(U&& item) {
    size_t write = write_idx_.load(std::memory_order_relaxed);
    size_t read = read_idx_.load(std::memory_order_acquire);

    if (write - read >= capacity_) {
      return false;
    }

    new (&buffer_[write % capacity_]) T(std::forward<U>(item));
    write_idx_.store(write + 1, std::memory_order_release);

    return true;
  }

  bool Dequeue(T*& item) {
    size_t read = read_idx_.load(std::memory_order_relaxed);
    size_t write = write_idx_.load(std::memory_order_acquire);

    if (read >= write) {
      return false;
    }

    item = reinterpret_cast<T*>(&buffer_[read % capacity_]);
    read_idx_.store(read + 1, std::memory_order_release);

    return true;
  }

  void Consume() {
    T* ptr;
    while (Dequeue(ptr)) {
      ptr->~T();
    }
  }

  size_t Size() const {
    size_t write = write_idx_.load(std::memory_order_acquire);
    size_t read = read_idx_.load(std::memory_order_acquire);
    return write - read;
  }

  bool Empty() const {
    return Size() == 0;
  }

  size_t Capacity() const { return capacity_; }

  void Reset() {
    T* ptr;
    while (Dequeue(ptr)) {
      ptr->~T();
    }
    write_idx_.store(0, std::memory_order_release);
    read_idx_.store(0, std::memory_order_release);
  }

private:
  size_t capacity_;
  std::unique_ptr<std::aligned_storage_t<sizeof(T), alignof(T)>[]> buffer_;
  std::atomic<size_t> write_idx_;
  std::atomic<size_t> read_idx_;
};

template<typename T>
using ShadowBufferSPSC = SPSCQueue<T>;

} // namespace st_slam

#endif
