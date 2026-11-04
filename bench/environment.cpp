#include "environment.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <thread>

#include "build_info.hpp"

#if defined(_WIN32)
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <intrin.h>
#define MM_POPEN _popen
#define MM_PCLOSE _pclose
#define MM_NULL_REDIRECT " 2>NUL"
#else
#include <cpuid.h>
#include <pthread.h>
#include <sched.h>
#include <sys/utsname.h>
#include <unistd.h>
#define MM_POPEN popen
#define MM_PCLOSE pclose
#define MM_NULL_REDIRECT " 2>/dev/null"
#endif

namespace bench {

namespace {

std::string trim(std::string s) {
    const char* ws = " \t\r\n";
    const auto b = s.find_first_not_of(ws);
    if (b == std::string::npos) return "";
    const auto e = s.find_last_not_of(ws);
    return s.substr(b, e - b + 1);
}

// Run a shell command and capture stdout (empty string on failure).
std::string run_command(const std::string& cmd) {
    std::string out;
    FILE* pipe = MM_POPEN((cmd + MM_NULL_REDIRECT).c_str(), "r");
    if (!pipe) return out;
    std::array<char, 256> buf{};
    while (std::fgets(buf.data(), static_cast<int>(buf.size()), pipe)) out += buf.data();
    MM_PCLOSE(pipe);
    return trim(out);
}

std::string cpu_brand() {
    unsigned regs[12] = {};
#if defined(_WIN32)
    int r[4];
    for (int i = 0; i < 3; ++i) {
        __cpuid(r, static_cast<int>(0x80000002u + i));
        std::memcpy(regs + 4 * i, r, sizeof r);
    }
#else
    for (unsigned i = 0; i < 3; ++i)
        __get_cpuid(0x80000002u + i, &regs[4 * i], &regs[4 * i + 1], &regs[4 * i + 2], &regs[4 * i + 3]);
#endif
    char buf[49] = {};
    std::memcpy(buf, regs, 48);
    return trim(buf);
}

std::string os_description() {
#if defined(_WIN32)
    const std::string v = run_command("cmd /c ver");
    return v.empty() ? "Windows" : v;
#else
    utsname u{};
    if (uname(&u) != 0) return "unknown";
    return std::string(u.sysname) + " " + u.release + " " + u.machine;
#endif
}

double ram_gb() {
#if defined(_WIN32)
    MEMORYSTATUSEX m{};
    m.dwLength = sizeof m;
    if (!GlobalMemoryStatusEx(&m)) return 0.0;
    return static_cast<double>(m.ullTotalPhys) / (1024.0 * 1024.0 * 1024.0);
#else
    const long pages = sysconf(_SC_PHYS_PAGES), page = sysconf(_SC_PAGE_SIZE);
    return pages > 0 && page > 0 ? static_cast<double>(pages) * page / (1024.0 * 1024.0 * 1024.0) : 0.0;
#endif
}

std::string power_plan() {
#if defined(_WIN32)
    // e.g. "Power Scheme GUID: e9a42b02-...  (Ultimate Performance)" -> "Ultimate Performance"
    const std::string s = run_command("powercfg /getactivescheme");
    const auto l = s.rfind('('), r = s.rfind(')');
    if (l != std::string::npos && r != std::string::npos && r > l) return s.substr(l + 1, r - l - 1);
    return s.empty() ? "unknown" : s;
#else
    return "n/a (not Windows)";
#endif
}

void measure_timer(Environment& env) {
    using clock = std::chrono::steady_clock;
    env.timer_tick_ns = 1e9 * static_cast<double>(clock::period::num) / static_cast<double>(clock::period::den);
    // Resolution: smallest non-zero step between consecutive readings.
    double min_step = 1e18;
    for (int i = 0; i < 20000; ++i) {
        const auto a = clock::now();
        auto b = clock::now();
        while (b == a) b = clock::now();
        min_step = std::min(min_step, std::chrono::duration<double, std::nano>(b - a).count());
    }
    env.timer_resolution_ns = min_step;
    // Overhead: median per-call cost over batches of 1000 back-to-back calls.
    std::vector<double> per_call;
    for (int batch = 0; batch < 101; ++batch) {
        const auto t0 = clock::now();
        for (int i = 0; i < 1000; ++i) (void)clock::now();
        const auto t1 = clock::now();
        per_call.push_back(std::chrono::duration<double, std::nano>(t1 - t0).count() / 1000.0);
    }
    std::nth_element(per_call.begin(), per_call.begin() + 50, per_call.end());
    env.timer_overhead_ns = per_call[50];
}

}  // namespace

bool release_build() {
#ifdef NDEBUG
    return true;
#else
    return false;
#endif
}

Environment capture_environment() {
    Environment env;
    env.cpu = cpu_brand();
    env.logical_cpus = std::thread::hardware_concurrency();
    env.os = os_description();
    env.ram_gb = ram_gb();
    env.compiler = MM_CXX_COMPILER_ID;
    env.build_type = std::string(MM_BUILD_TYPE) + (release_build() ? " (NDEBUG)" : " (asserts on)");
    env.build_flags = MM_BUILD_FLAGS;
    const std::string git = std::string("git -C \"") + MM_SOURCE_DIR + "\"";
    env.git_commit = run_command(git + " rev-parse --short HEAD");
    if (env.git_commit.empty()) env.git_commit = "unknown";
    env.git_dirty = !run_command(git + " status --porcelain --untracked-files=no").empty();
    env.power_plan = power_plan();
    measure_timer(env);
    return env;
}

bool pin_current_thread(unsigned cpu) {
    if (cpu >= std::thread::hardware_concurrency()) return false;
#if defined(_WIN32)
    if (cpu >= 8 * sizeof(DWORD_PTR)) return false;
    return SetThreadAffinityMask(GetCurrentThread(), static_cast<DWORD_PTR>(1) << cpu) != 0;
#else
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    return pthread_setaffinity_np(pthread_self(), sizeof set, &set) == 0;
#endif
}

std::string raise_priority() {
#if defined(_WIN32)
    const bool p = SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS) != 0;
    const bool t = SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST) != 0;
    return std::string("process HIGH_PRIORITY_CLASS ") + (p ? "ok" : "FAILED") + ", thread HIGHEST " +
           (t ? "ok" : "FAILED");
#else
    return "unchanged (non-Windows)";
#endif
}

double now_seconds() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

}  // namespace bench
