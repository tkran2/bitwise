#pragma once
// A small persistent thread pool for data-parallel loops.
//
// Determinism note: every kernel that uses this pool assigns each output element to
// exactly one task and never splits a reduction across tasks (unless the kernel mode
// explicitly asks for it), so results do not depend on the thread count or on how
// work is partitioned.

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace bw {

class ThreadPool {
 public:
  explicit ThreadPool(int num_threads) : num_threads_(num_threads < 1 ? 1 : num_threads) {
    for (int t = 1; t < num_threads_; ++t) workers_.emplace_back([this, t] { worker(t); });
  }

  ~ThreadPool() {
    {
      std::lock_guard<std::mutex> lock(mu_);
      stop_ = true;
      ++generation_;
    }
    cv_.notify_all();
    for (auto& w : workers_) w.join();
  }

  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;

  int size() const { return num_threads_; }

  // Runs fn(begin, end) over a static partition of [0, n) into contiguous chunks.
  void parallel_for(std::size_t n, const std::function<void(std::size_t, std::size_t)>& fn) {
    if (n == 0) return;
    const std::size_t tasks = std::min<std::size_t>(n, static_cast<std::size_t>(num_threads_));
    if (tasks == 1) {
      fn(0, n);
      return;
    }
    auto body = [&](int t) {
      if (static_cast<std::size_t>(t) >= tasks) return;
      const std::size_t begin = n * t / tasks, end = n * (t + 1) / tasks;
      if (begin < end) fn(begin, end);
    };
    run(body);
  }

 private:
  void run(const std::function<void(int)>& job) {
    {
      std::lock_guard<std::mutex> lock(mu_);
      job_ = &job;
      pending_.store(num_threads_ - 1, std::memory_order_relaxed);
      ++generation_;
    }
    cv_.notify_all();
    job(0);
    while (pending_.load(std::memory_order_acquire) != 0) std::this_thread::yield();
    job_ = nullptr;
  }

  void worker(int tid) {
    std::size_t seen = 0;
    for (;;) {
      const std::function<void(int)>* job;
      {
        std::unique_lock<std::mutex> lock(mu_);
        cv_.wait(lock, [&] { return generation_ != seen; });
        seen = generation_;
        if (stop_) return;
        job = job_;
      }
      (*job)(tid);
      pending_.fetch_sub(1, std::memory_order_acq_rel);
    }
  }

  int num_threads_;
  std::vector<std::thread> workers_;
  std::mutex mu_;
  std::condition_variable cv_;
  std::size_t generation_ = 0;
  bool stop_ = false;
  const std::function<void(int)>* job_ = nullptr;
  std::atomic<int> pending_{0};
};

}  // namespace bw
