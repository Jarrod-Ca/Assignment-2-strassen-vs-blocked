// test_framework.hpp - a deliberately tiny, dependency-free test harness.
// (No GoogleTest: the build must work offline on both MSVC and g++ with nothing
// to download.) Tests self-register; main() in test_main.cpp runs them all and
// returns non-zero on any failure, which is what CTest checks.
#pragma once

#include <cmath>
#include <functional>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace mmtest {

struct TestCase {
    std::string name;
    std::function<void()> fn;
};

inline std::vector<TestCase>& registry() {
    static std::vector<TestCase> tests;
    return tests;
}

inline int& failure_count() {
    static int failures = 0;
    return failures;
}

struct Registrar {
    Registrar(const char* name, std::function<void()> fn) { registry().push_back({name, std::move(fn)}); }
};

inline void report_failure(const char* file, int line, const std::string& msg) {
    ++failure_count();
    std::cerr << "    FAILED " << file << ":" << line << ": " << msg << "\n";
}

}  // namespace mmtest

#define MM_CONCAT_INNER(a, b) a##b
#define MM_CONCAT(a, b) MM_CONCAT_INNER(a, b)

#define TEST(name)                                                              \
    static void name();                                                         \
    static ::mmtest::Registrar MM_CONCAT(registrar_, name)(#name, &name);       \
    static void name()

#define CHECK(cond)                                                             \
    do {                                                                        \
        if (!(cond)) ::mmtest::report_failure(__FILE__, __LINE__, #cond);       \
    } while (0)

#define CHECK_EQ(a, b)                                                          \
    do {                                                                        \
        const auto& va_ = (a);                                                  \
        const auto& vb_ = (b);                                                  \
        if (!(va_ == vb_))                                                      \
            ::mmtest::report_failure(__FILE__, __LINE__,                        \
                                     std::string(#a " == " #b " (got ") +       \
                                         std::to_string(va_) + " vs " +         \
                                         std::to_string(vb_) + ")");            \
    } while (0)

#define CHECK_NEAR(a, b, tol)                                                   \
    do {                                                                        \
        const double va_ = (a), vb_ = (b);                                      \
        if (!(std::fabs(va_ - vb_) <= (tol)))                                   \
            ::mmtest::report_failure(__FILE__, __LINE__,                        \
                                     std::string(#a " ~= " #b " (got ") +       \
                                         std::to_string(va_) + " vs " +         \
                                         std::to_string(vb_) + ")");            \
    } while (0)

#define CHECK_THROWS(expr)                                                      \
    do {                                                                        \
        bool threw_ = false;                                                    \
        try { (void)(expr); } catch (...) { threw_ = true; }                    \
        if (!threw_) ::mmtest::report_failure(__FILE__, __LINE__,               \
                                              "expected exception: " #expr);    \
    } while (0)
