// naive.hpp - reference implementations. NOT one of the two compared algorithms.
//
// These exist only to check the compared algorithms (src/algo_blocked,
// src/algo_strassen) and to measure their numerical error. They are written
// for obviousness, not speed.
#pragma once

#include "common/matrix.hpp"

namespace mm::reference {

// Textbook i-j-k triple loop in plain double arithmetic. C = A * B (C is overwritten).
// C must not alias A or B.
void naive_multiply(ConstMatrixView A, ConstMatrixView B, MatrixView C);

// Same i-j-k loop, but each dot product is accumulated in double-double
// arithmetic (~106-bit significand: exact products via FMA, error-free
// TwoSum additions) and rounded to double once at the end. The result is
// (almost always) the correctly rounded exact product, so it serves as the
// "ground truth" for error measurements.
//
// Why not long double? On MSVC long double is the same 64-bit type as double,
// so it would silently give no extra precision on our benchmark platform.
// Requires strict IEEE semantics: build with /fp:precise (MSVC) or without
// -ffast-math and with -ffp-contract=off (GCC/Clang). See CMakeLists.txt.
void reference_multiply_dd(ConstMatrixView A, ConstMatrixView B, MatrixView C);

}  // namespace mm::reference
