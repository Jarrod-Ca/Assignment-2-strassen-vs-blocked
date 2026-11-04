// cache_pressure.hpp - a background thread that occupies part of the shared L3.
//
// Used by experiment E6 to test, rather than assert, the claim that the 96 MB L3
// barely matters to either algorithm (prediction P7). The thread runs on a
// different physical core from the benchmark and repeatedly reads one double per
// 64-byte line of a private buffer. While the buffer fits in L3 (all sizes
// used are < 96 MB), its lines stay resident and the benchmark thread
// effectively has (96 MB - buffer) of L3 left.
//
// Confound acknowledged and controlled: the thread also consumes L3 bandwidth.
// A 1 MB buffer fits in the antagonist's own L2, so it occupies no L3 but keeps
// a busy neighbour; comparing 1 MB with 0 (thread off) isolates that effect.
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <thread>
#include <vector>

namespace bench {

class CachePressure {
public:
    // bytes == 0: no thread is started. Blocks until the buffer has been swept
    // `warm_passes` times so the L3 is already occupied when timing starts.
    CachePressure(std::size_t bytes, unsigned cpu, int warm_passes = 3);
    ~CachePressure();
    CachePressure(const CachePressure&) = delete;
    CachePressure& operator=(const CachePressure&) = delete;

    bool active() const { return thread_.joinable(); }
    bool pinned() const { return pinned_.load(); }
    std::uint64_t passes() const { return passes_.load(); }
    std::size_t bytes() const { return bytes_; }

private:
    void run(unsigned cpu);
    std::size_t bytes_;
    std::vector<double> buffer_;
    std::atomic<bool> stop_{false};
    std::atomic<bool> pinned_{false};
    std::atomic<std::uint64_t> passes_{0};
    std::thread thread_;
};

}  // namespace bench
