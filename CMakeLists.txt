cmake_minimum_required(VERSION 3.21)
project(MatMulComparison
        DESCRIPTION "Cache-blocked iterative vs Strassen matrix multiplication: an empirical comparison"
        LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ---------------------------------------------------------------------------
# Compiler flags shared by every target. Both compared algorithms are built with
# exactly the same flags (fairness), single-threaded, AVX2 enabled, and with
# strict IEEE floating point:
#   * no -ffast-math / /fp:fast  - would let the compiler reassociate sums, which
#     changes numerical error (one of our measured quantities) and breaks the
#     error-free transforms in the double-double reference;
#   * no FMA contraction (-ffp-contract=off; MSVC /fp:precise doesn't contract
#     by default) - so a*b+c is rounded identically on MSVC and g++.
# Release optimisation level comes from CMake's defaults: /O2 (MSVC), -O3 (GCC/Clang).
# ---------------------------------------------------------------------------
add_library(mm_flags INTERFACE)
if(MSVC)
  set(MM_FLAGS /W4 /permissive- /EHsc /fp:precise /arch:AVX2)
else()
  set(MM_FLAGS -Wall -Wextra -Wpedantic -mavx2 -mfma -ffp-contract=off)
endif()
target_compile_options(mm_flags INTERFACE ${MM_FLAGS})
# Recorded verbatim in every results/run_info.txt.
string(TOUPPER "${CMAKE_BUILD_TYPE}" MM_BUILD_TYPE_UPPER)
string(JOIN " " MM_FLAGS_STR ${MM_FLAGS} ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${MM_BUILD_TYPE_UPPER}})

# Shared infrastructure: storage, generators, comparison, reference oracle.
add_library(mm_common STATIC
  src/common/matrix.cpp
  src/common/generators.cpp
  src/common/compare.cpp
  src/reference/naive.cpp)
target_include_directories(mm_common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(mm_common PUBLIC mm_flags)

# ---------------------------------------------------------------------------
# The two compared algorithms, each in its own module/library.
# ---------------------------------------------------------------------------
# Algorithm A: cache-blocked iterative multiplication.
add_library(mm_blocked STATIC src/algo_blocked/blocked.cpp)
target_link_libraries(mm_blocked PUBLIC mm_common)

# Algorithm B: Strassen (uses Algorithm A's kernel below its cutoff).
add_library(mm_strassen STATIC src/algo_strassen/strassen.cpp)
target_link_libraries(mm_strassen PUBLIC mm_blocked)

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
enable_testing()
add_executable(mm_tests
  tests/test_main.cpp
  tests/test_matrix.cpp
  tests/test_generators.cpp
  tests/test_reference.cpp
  tests/test_blocked.cpp
  tests/test_strassen.cpp)
target_link_libraries(mm_tests PRIVATE mm_common mm_blocked mm_strassen)
add_test(NAME mm_tests COMMAND mm_tests)

# ---------------------------------------------------------------------------
# Benchmark harness
# ---------------------------------------------------------------------------
find_package(Threads REQUIRED)
configure_file(bench/build_info.hpp.in ${CMAKE_CURRENT_BINARY_DIR}/generated/build_info.hpp @ONLY)
add_executable(mm_bench
  bench/bench_main.cpp
  bench/experiments.cpp
  bench/environment.cpp
  bench/cache_pressure.cpp)
target_include_directories(mm_bench PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/generated bench)
target_link_libraries(mm_bench PRIVATE mm_strassen Threads::Threads)
