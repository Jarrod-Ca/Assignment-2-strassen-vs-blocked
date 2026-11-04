#include "algo_blocked/blocked.hpp"

#include <algorithm>
#include <stdexcept>

#include "common/platform.hpp"

namespace mm::blocked {

void validate(const Params& p) {
    if (p.tile == 0) throw std::invalid_argument("blocked: tile must be >= 1");
}

namespace {

void check_args(ConstMatrixView A, ConstMatrixView B, ConstMatrixView C, const Params& p) {
    validate(p);
    require_mult_shapes(A, B, C);
    require_no_alias(A, B, C);
}

// c[0:w] += a * b[0:w]. A separate function so `restrict` applies to parameters,
// the form compilers reliably honour; it lets the loop vectorise with no runtime
// aliasing check. Inlined by the optimiser.
inline void axpy_row(double* MM_RESTRICT c, const double* MM_RESTRICT b, double a, std::size_t w) {
    for (std::size_t j = 0; j < w; ++j) c[j] += a * b[j];
}

// The kernel. Preconditions already checked by the public entry points.
void accumulate(ConstMatrixView A, ConstMatrixView B, MatrixView C, std::size_t T) {
    const std::size_t m = A.rows(), k = A.cols(), n = B.cols();
    for (std::size_t ii = 0; ii < m; ii += T) {
        const std::size_t i_end = std::min(ii + T, m);
        for (std::size_t kk = 0; kk < k; kk += T) {
            const std::size_t k_end = std::min(kk + T, k);
            for (std::size_t jj = 0; jj < n; jj += T) {
                const std::size_t width = std::min(jj + T, n) - jj;
                // One tile step: C[ii:i_end, jj:jj+width] += A[ii:i_end, kk:k_end] * B[kk:k_end, jj:jj+width]
                for (std::size_t i = ii; i < i_end; ++i) {
                    double* c = C.row(i) + jj;
                    const double* a_row = A.row(i);
                    for (std::size_t q = kk; q < k_end; ++q) axpy_row(c, B.row(q) + jj, a_row[q], width);
                }
            }
        }
    }
}

}  // namespace

void multiply_add(ConstMatrixView A, ConstMatrixView B, MatrixView C, const Params& p) {
    check_args(A, B, C, p);
    accumulate(A, B, C, p.tile);
}

void multiply(ConstMatrixView A, ConstMatrixView B, MatrixView C, const Params& p) {
    check_args(A, B, C, p);  // validate before touching C
    for (std::size_t i = 0; i < C.rows(); ++i) std::fill(C.row(i), C.row(i) + C.cols(), 0.0);
    accumulate(A, B, C, p.tile);
}

}  // namespace mm::blocked
