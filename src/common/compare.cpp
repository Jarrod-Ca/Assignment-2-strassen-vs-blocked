#include "common/compare.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace mm {

namespace {
void require_same_shape(ConstMatrixView X, ConstMatrixView Y) {
    if (X.rows() != Y.rows() || X.cols() != Y.cols()) throw std::invalid_argument("shape mismatch");
}

// Scaled sum of squares (LAPACK dnrm2 style) so huge WideRange entries don't overflow.
template <typename F>
double scaled_norm(std::size_t rows, std::size_t cols, F value_at) {
    double scale = 0.0, ssq = 1.0;
    for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t j = 0; j < cols; ++j) {
            const double a = std::fabs(value_at(i, j));
            if (a == 0.0) continue;
            if (scale < a) {
                ssq = 1.0 + ssq * (scale / a) * (scale / a);
                scale = a;
            } else {
                ssq += (a / scale) * (a / scale);
            }
        }
    }
    return scale * std::sqrt(ssq);
}
}  // namespace

bool exactly_equal(ConstMatrixView X, ConstMatrixView Y) {
    if (X.rows() != Y.rows() || X.cols() != Y.cols()) return false;
    for (std::size_t i = 0; i < X.rows(); ++i)
        for (std::size_t j = 0; j < X.cols(); ++j)
            if (!(X(i, j) == Y(i, j))) return false;
    return true;
}

double max_abs_diff(ConstMatrixView X, ConstMatrixView Y) {
    require_same_shape(X, Y);
    double m = 0.0;
    for (std::size_t i = 0; i < X.rows(); ++i)
        for (std::size_t j = 0; j < X.cols(); ++j) m = std::max(m, std::fabs(X(i, j) - Y(i, j)));
    return m;
}

double max_abs(ConstMatrixView X) {
    double m = 0.0;
    for (std::size_t i = 0; i < X.rows(); ++i)
        for (std::size_t j = 0; j < X.cols(); ++j) m = std::max(m, std::fabs(X(i, j)));
    return m;
}

double frobenius_norm(ConstMatrixView X) {
    return scaled_norm(X.rows(), X.cols(), [&](std::size_t i, std::size_t j) { return X(i, j); });
}

double frobenius_diff(ConstMatrixView X, ConstMatrixView Y) {
    require_same_shape(X, Y);
    return scaled_norm(X.rows(), X.cols(), [&](std::size_t i, std::size_t j) { return X(i, j) - Y(i, j); });
}

double relative_error_fro(ConstMatrixView C, ConstMatrixView C_ref) {
    const double denom = frobenius_norm(C_ref);
    const double num = frobenius_diff(C, C_ref);
    return denom == 0.0 ? num : num / denom;
}

}  // namespace mm
