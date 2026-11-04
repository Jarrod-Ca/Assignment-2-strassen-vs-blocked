#include <exception>
#include <iostream>

#include "test_framework.hpp"

int main() {
    int failed_tests = 0;
    for (const auto& t : mmtest::registry()) {
        const int before = mmtest::failure_count();
        try {
            t.fn();
        } catch (const std::exception& e) {
            mmtest::report_failure(__FILE__, __LINE__, std::string("uncaught exception: ") + e.what());
        }
        const bool ok = mmtest::failure_count() == before;
        if (!ok) ++failed_tests;
        std::cout << (ok ? "[PASS] " : "[FAIL] ") << t.name << "\n";
    }
    std::cout << "\n" << (mmtest::registry().size() - failed_tests) << "/" << mmtest::registry().size()
              << " tests passed\n";
    return failed_tests == 0 ? 0 : 1;
}
