// environment.hpp - everything about the machine and build that a results file
// must carry, plus the OS calls that control where and how the benchmark runs.
#pragma once

#include <string>
#include <vector>

namespace bench {

struct Environment {
    std::string cpu;            // CPUID brand string
    unsigned logical_cpus = 0;  // std::thread::hardware_concurrency()
    std::string os;
    double ram_gb = 0.0;
    std::string compiler;       // id + version
    std::string build_type;     // CMake build type, plus NDEBUG state
    std::string build_flags;    // the project-wide flags from CMakeLists.txt
    std::string git_commit;     // short hash, or "unknown"
    bool git_dirty = false;     // uncommitted changes to tracked files
    std::string power_plan;     // Windows active power scheme ("n/a" elsewhere)
    double timer_tick_ns = 0;   // steady_clock period
    double timer_resolution_ns = 0;  // smallest observed non-zero difference
    double timer_overhead_ns = 0;    // median cost of one now() call
};

Environment capture_environment();

bool release_build();  // NDEBUG defined

// Pin the calling thread to one logical CPU. Returns false (and changes
// nothing) if the CPU index is out of range or the OS refuses.
bool pin_current_thread(unsigned cpu);

// Raise process/thread scheduling priority so other work is less likely to
// pre-empt a timed region. Returns a description of what was applied.
std::string raise_priority();

// Seconds since an arbitrary epoch (steady_clock).
double now_seconds();

}  // namespace bench
