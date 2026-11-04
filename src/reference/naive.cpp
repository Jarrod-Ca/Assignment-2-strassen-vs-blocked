#include "reference/naive.hpp"

#include <cmath>

namespace mm::reference {

void naive_multiply(ConstMatrixView A, ConstMatrixView B, MatrixView C) {
    require_mult_shapes(A, B, C);
    const std::size_t m = A.rows(), k = A.cols(), n = B.cols();
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (std::size_t p = 0; p < k; ++p) sum += A(i, p) * B(p, j);
            C(i, j) = sum;
        }
    }
}

namespace {
// Knuth's TwoSum: s + e == a + b exactly.
inline void two_sum(double a, double b, double& s, double& e) {
    s = a + b;
    const double bb = s - a;
    e = (a - (s - bb)) + (b - bb);
}
// Requires |a| >= |b| (or a == 0): s + e == a + b exactly.
inline void fast_two_sum(double a, double b, double& s, double& e) {
    s = a + b;
    e = b - (s - a);
}
}  // namespace

void reference_multiply_dd(ConstMatrixView A, ConstMatrixView B, MatrixView C) {
    require_mult_shapes(A, B, C);
    const std::size_t m = A.rows(), k = A.cols(), n = B.cols();
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            double hi = 0.0, lo = 0.0;  // running sum = hi + lo
            for (std::size_t p = 0; p < k; ++p) {
                const double a = A(i, p), b = B(p, j);
                const double prod = a * b;
                const double prod_err = std::fma(a, b, -prod);  // exact: a*b == prod + prod_err
                double s, e;
                two_sum(hi, prod, s, e);
                e += lo + prod_err;
                fast_two_sum(s, e, hi, lo);
            }
            C(i, j) = hi + lo;
        }
    }
}

}  // namespace mm::reference
