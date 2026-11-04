// strassen.hpp - Algorithm B: Strassen's recursive matrix multiplication.
//
// Idea: split each n x n matrix into four (n/2) x (n/2) quadrants. The classical
// block product needs 8 quadrant multiplications; Strassen (1969) found a way to
// do it with 7, at the price of 18 quadrant additions/subtractions:
//
//   M1 = (A11 + A22)(B11 + B22)      C11 = M1 + M4 - M5 + M7
//   M2 = (A21 + A22) B11             C12 = M3 + M5
//   M3 = A11 (B12 - B22)             C21 = M2 + M4
//   M4 = A22 (B21 - B11)             C22 = M1 - M2 + M3 + M6
//   M5 = (A11 + A12) B22
//   M6 = (A21 - A11)(B11 + B12)
//   M7 = (A12 - A22)(B21 + B22)
//
// Applied recursively: T(n) = 7 T(n/2) + Theta(n^2) = Theta(n^log2(7)) ~ n^2.807.
//
// Practical details, each one an explicit, measurable parameter:
//   * cutoff    - subproblems of size <= cutoff are multiplied by Algorithm A's
//                 blocked kernel instead of recursing further (recursing to 1x1
//                 would spend almost all its time on additions and call overhead).
//   * padding   - recursion needs even sizes at every level, so n is padded with
//                 zeros up to a size m that halves cleanly down to the cutoff:
//                   Minimal  : m = s * 2^d with d the fewest halvings that bring
//                              ceil(n / 2^d) <= cutoff  (e.g. n=1025, cutoff=64 -> 1056)
//                   NextPow2 : m = next power of two >= n  (e.g. 1025 -> 2048)
//                   Aligned  : like Minimal, but the leaf size s is rounded up to a
//                              multiple of leaf_align (taking one more halving if
//                              that would exceed the cutoff), so every leaf row and
//                              every temporary's row stride is a whole number of
//                              leaf_align doubles (e.g. n=1028, cutoff=128, align 4:
//                              leaf 65 -> 68, m = 1088). Added after experiment E4
//                              showed Minimal is slow exactly when the leaf size is
//                              not a multiple of 4 (follow-up experiment E9).
//   * workspace - each recursion level needs 3 temporaries of (n/2)^2 (two operand
//                 sums and one product); total ~ n^2 doubles, allocated once per
//                 Plan, never inside the recursion.
//   * temp_pad  - optional row-stride padding of those temporaries, to test the
//                 power-of-two cache-set aliasing hypothesis.
#pragma once

#include <cstddef>
#include <cstdint>

#include "algo_blocked/blocked.hpp"
#include "common/matrix.hpp"

namespace mm::strassen {

enum class Padding { Minimal, NextPow2, Aligned };

struct Params {
    std::size_t cutoff = 128;     // recurse only while size > cutoff; must be >= 1
    Padding padding = Padding::Minimal;
    std::size_t leaf_align = 4;   // Aligned padding only: leaf size rounded up to a multiple of this
    blocked::Params base{};       // tile size used by the base-case kernel
    // Extra elements added to the row stride of every recursion temporary
    // (stride = h + temp_pad). At power-of-two h, a stride of h doubles maps
    // successive rows onto the same L1 cache sets; a small pad breaks that
    // pattern. 0 = tightly packed. An experimental variable (E5), not a fix
    // applied by default.
    std::size_t temp_pad = 0;
};

// Exact operation and memory counts for one multiplication. Produced both by
// actually running (Plan::execute) and analytically (predict), so tests can
// confirm the cost model the report reasons with matches what the code does.
struct Stats {
    std::size_t n = 0;               // logical size
    std::size_t padded_n = 0;        // size the recursion actually runs on (m)
    std::size_t depth = 0;           // number of Strassen levels (0 = pure blocked)
    std::size_t base_size = 0;       // subproblem size at the leaves
    std::uint64_t base_calls = 0;    // leaf multiplications = 7^depth
    std::uint64_t base_flops = 0;    // 2 * base_size^3 per leaf
    std::uint64_t add_flops = 0;     // element additions/subtractions (18 per level-node * h^2)
    std::uint64_t copy_elems = 0;    // element copies inside the recursion (1 * h^2 per node)
    std::size_t workspace_bytes = 0; // recursion temporaries
    std::size_t padding_bytes = 0;   // padded copies of A, B, C (0 when m == n)
    std::uint64_t total_flops() const { return base_flops + add_flops; }
};

void validate(const Params& p);  // throws std::invalid_argument

// The size m that n is padded to under p (m == n when no padding is needed).
std::size_t padded_size(std::size_t n, const Params& p);

// Analytic Stats for an n x n product, without running anything.
Stats predict(std::size_t n, const Params& p);

// A Plan fixes n and the parameters and owns all memory the multiplication
// needs (workspace and, if required, padded buffers). Constructing it does
// all allocation; execute() allocates nothing. This separates "cost of
// Strassen's arithmetic" from "cost of obtaining its memory", which the
// benchmarks can then measure independently.
class Plan {
public:
    Plan(std::size_t n, const Params& p = {});
    // C = A * B for n x n matrices. C must not overlap A or B. Any strides allowed.
    Stats execute(ConstMatrixView A, ConstMatrixView B, MatrixView C);
    std::size_t n() const { return n_; }
    const Params& params() const { return params_; }

private:
    std::size_t n_, m_;
    Params params_;
    AlignedBuffer workspace_;
    Matrix Ap_, Bp_, Cp_;  // only allocated when m_ != n_
};

// Convenience: build a Plan and execute it once (includes allocation cost).
Stats multiply(ConstMatrixView A, ConstMatrixView B, MatrixView C, const Params& p = {});

const char* padding_name(Padding p);
Padding parse_padding(const char* name);  // "minimal" | "nextpow2" | "aligned"; throws on anything else

}  // namespace mm::strassen
