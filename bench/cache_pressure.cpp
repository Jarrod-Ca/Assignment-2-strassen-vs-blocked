#include "cache_pressure.hpp"

#include "environment.hpp"

namespace bench {

namespace {
volatile double g_pressure_sink = 0.0;  // keeps the sweep from being optimised away
}

CachePressure::CachePressure(std::size_t bytes, unsigned cpu, int warm_passes) : bytes_(bytes) {
    if (bytes_ == 0) return;
    buffer_.assign(bytes_ / sizeof(double), 1.0);  // touch every page now, not while timing
    thread_ = std::thread([this, cpu] { run(cpu); });
    while (passes_.load() < static_cast<std::uint64_t>(warm_passes)) std::this_thread::yield();
}

CachePressure::~CachePressure() {
    stop_.store(true);
    if (thread_.joinable()) thread_.join();
}

void CachePressure::run(unsigned cpu) {
    pinned_.store(pin_current_thread(cpu));
    const std::size_t stride = 64 / sizeof(double);  // one read per cache line
    double sum = 0.0;
    while (!stop_.load(std::memory_order_relaxed)) {
        for (std::size_t i = 0; i < buffer_.size(); i += stride) sum += buffer_[i];
        passes_.fetch_add(1, std::memory_order_relaxed);
    }
    g_pressure_sink = sum;
}

}  // namespace bench
