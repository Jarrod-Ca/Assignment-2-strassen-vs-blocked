// compare.hpp - comparison and norm utilities used by tests and error measurement.
#pragma once

#include "common/matrix.hpp"

namespace mm {

bool exactly_equal(ConstMatrixView X, ConstMatrixView Y);    // shapes and every element (==)
double max_abs_diff(ConstMatrixView X, ConstMatrixView Y);   // max_ij |X_ij - Y_ij|
double max_abs(ConstMatrixView X);                           // max_ij |X_ij|
double frobenius_norm(ConstMatrixView X);                    // sqrt(sum X_ij^2), overflow-safe
double frobenius_diff(ConstMatrixView X, ConstMatrixView Y); // ||X - Y||_F

// ||C - C_ref||_F / ||C_ref||_F  (returns ||C||_F if C_ref is zero).
double relative_error_fro(ConstMatrixView C, ConstMatrixView C_ref);

// max_ij |C_ij - Cref_ij| / D_ij over entries with D_ij > 0. With D = |A||B| this
// is the quantity the classical componentwise bound controls:
// |C - C_hat| <= gamma_n |A||B|, so for a classical algorithm the result is
// <= gamma_n. Strassen satisfies no such bound. Returns +inf if some D_ij == 0
// while C_ij != Cref_ij.
double max_componentwise_ratio(ConstMatrixView C, ConstMatrixView C_ref, ConstMatrixView D);

// max_ij |C_ij - Cref_ij| / |Cref_ij| over entries with Cref_ij != 0
// (worst relative error of any single entry). Returns +inf if some Cref_ij == 0
// while C_ij != 0.
double max_entrywise_relative_error(ConstMatrixView C, ConstMatrixView C_ref);

// Element-wise absolute value: dst_ij = |src_ij| (shapes must match).
void abs_into(ConstMatrixView src, MatrixView dst);

// Sum of all entries (a cheap fingerprint of a result, used by the benchmark
// harness to consume every output and to check algorithms agree).
double checksum(ConstMatrixView X);

}  // namespace mm
