// blocked.hpp - Algorithm A: cache-blocked (tiled) iterative matrix multiplication.
//
// Idea: the classical Theta(m*k*n) triple loop does exactly the same arithmetic
// whatever order the loops run in, but the order determines how often data is
// re-fetched from slow memory. We split the iteration space into T x T x T
// tiles. A tile step touches one T x T block each of A, B and C
// (3 * T^2 * 8 bytes), so if that fits in a cache level, every element loaded
// into that cache is reused ~T times before eviction instead of ~once.
//
// Inside a tile the loop order is i-k-j: the innermost loop runs along a row of
// C and a row of B (both contiguous in row-major storage) with a single scalar
// from A, which the compiler vectorises (AVX2: 4 doubles per instruction).
//
// No hand-written intrinsics or register micro-kernel: both compared algorithms
// get the same level of implementation effort, and Strassen uses this exact
// kernel for its base case.
#pragma once

#include <cstddef>

#include "common/matrix.hpp"

namespace mm::blocked {

struct Params {
    // Tile edge length T (in elements). An experimental variable, swept in the
    // benchmarks rather than hard-coded. Must be >= 1. T = 1 degenerates to an
    // untiled i-k-j loop; T >= max(m, k, n) means a single tile.
    std::size_t tile = 64;
};

// Throws std::invalid_argument if the parameters are unusable.
void validate(const Params& p);

// C = A * B. A is m x k, B is k x n, C is m x n. C must not overlap A or B
// (A and B may overlap each other). Any strides are allowed.
void multiply(ConstMatrixView A, ConstMatrixView B, MatrixView C, const Params& p = {});

// C += A * B. Same contract as multiply().
void multiply_add(ConstMatrixView A, ConstMatrixView B, MatrixView C, const Params& p = {});

}  // namespace mm::blocked
