// generators.hpp - seeded input families for correctness tests and benchmarks.
//
// All families are bit-reproducible across compilers/platforms (see rng.hpp).
// Each family exists to probe something specific about the algorithms:
//
//   Uniform   entries ~ U[-1, 1). The default timing workload; values are
//             "well behaved" so timings reflect the algorithm, not denormals.
//   SmallInt  integers in [-bound, bound] stored as doubles. Every partial sum
//             is an exactly representable integer while n * (growth * bound)^2
//             stays below 2^53, so classical and Strassen results can be
//             compared for *exact* equality - a sharp correctness test.
//   WideRange +/- m * 2^e with m ~ U[1, 2) and integer e ~ U[-max_exp, max_exp].
//             Large dynamic range stresses Strassen's numerical stability
//             (its additions/subtractions mix entries of very different scale).
//             Built with std::ldexp, which is exact, so still bit-reproducible.
//   Identity  for sanity checks.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include "common/matrix.hpp"

namespace mm {

enum class Family { Uniform, SmallInt, WideRange, Identity };

struct InputSpec {
    Family family = Family::Uniform;
    std::int64_t int_bound = 8;  // SmallInt only
    int max_exp = 20;            // WideRange only
};

// `stride` == 0 means tightly packed; otherwise the row pitch in elements.
Matrix make_uniform(std::size_t rows, std::size_t cols, std::uint64_t seed, std::size_t stride = 0);
Matrix make_small_int(std::size_t rows, std::size_t cols, std::uint64_t seed, std::int64_t bound,
                      std::size_t stride = 0);
Matrix make_wide_range(std::size_t rows, std::size_t cols, std::uint64_t seed, int max_exp,
                       std::size_t stride = 0);
Matrix make_identity(std::size_t n, std::size_t stride = 0);

// Dispatch on an InputSpec.
Matrix make_matrix(const InputSpec& spec, std::size_t rows, std::size_t cols, std::uint64_t seed,
                   std::size_t stride = 0);

const char* family_name(Family f);
Family parse_family(const std::string& name);  // throws std::invalid_argument

}  // namespace mm
