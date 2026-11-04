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

}  // namespace mm
