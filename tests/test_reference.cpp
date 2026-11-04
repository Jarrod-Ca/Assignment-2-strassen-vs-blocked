#include "common/compare.hpp"
#include "common/generators.hpp"
#include "reference/naive.hpp"
#include "test_framework.hpp"

using namespace mm;
using reference::naive_multiply;
using reference::reference_multiply_dd;

TEST(naive_matches_hand_computed_rectangular_product) {
    // [1 2 3]   [ 7  8]   [ 58  64]
    // [4 5 6] x [ 9 10] = [139 154]
    //           [11 12]
    Matrix A(2, 3), B(3, 2), C(2, 2);
    double a[] = {1, 2, 3, 4, 5, 6}, b[] = {7, 8, 9, 10, 11, 12};
    for (int i = 0; i < 6; ++i) { A(i / 3, i % 3) = a[i]; B(i / 2, i % 2) = b[i]; }
    naive_multiply(A, B, C);
    CHECK(C(0, 0) == 58 && C(0, 1) == 64 && C(1, 0) == 139 && C(1, 1) == 154);
    Matrix D(2, 2);
    reference_multiply_dd(A, B, D);
    CHECK(exactly_equal(C, D));
}

TEST(identity_is_neutral) {
    Matrix A = make_uniform(13, 13, 99), I = make_identity(13), C(13, 13);
    naive_multiply(A, I, C);
    CHECK(exactly_equal(A, C));
    naive_multiply(I, A, C);
    CHECK(exactly_equal(A, C));
}

TEST(strided_views_give_same_result_as_packed) {
    Matrix A = make_uniform(17, 17, 1), B = make_uniform(17, 17, 2);
    Matrix As = make_uniform(17, 17, 1, 24), Bs = make_uniform(17, 17, 2, 32);
    Matrix C(17, 17), Cs(17, 17, 20);
    naive_multiply(A, B, C);
    naive_multiply(As, Bs, Cs);
    CHECK(exactly_equal(C, Cs));
}

TEST(sub_block_views_multiply_correctly) {
    // Multiply the top-left 5x5 blocks of two 8x8 matrices and compare with packed copies.
    Matrix A = make_small_int(8, 8, 3, 5), B = make_small_int(8, 8, 4, 5);
    Matrix A5(5, 5), B5(5, 5);
    for (std::size_t i = 0; i < 5; ++i)
        for (std::size_t j = 0; j < 5; ++j) { A5(i, j) = A(i, j); B5(i, j) = B(i, j); }
    Matrix C(5, 5), C5(5, 5);
    naive_multiply(A.view().block(0, 0, 5, 5), B.view().block(0, 0, 5, 5), C);
    naive_multiply(A5, B5, C5);
    CHECK(exactly_equal(C, C5));
}

TEST(small_int_products_are_exact_in_both_references) {
    Matrix A = make_small_int(64, 64, 10, 8), B = make_small_int(64, 64, 11, 8);
    Matrix C(64, 64), D(64, 64);
    naive_multiply(A, B, C);
    reference_multiply_dd(A, B, D);
    CHECK(exactly_equal(C, D));
}

TEST(double_double_reference_survives_catastrophic_cancellation) {
    // Row [1e16, 1, -1e16] . column [1, 1, 1]: exact answer is 1.
    // Plain double: 1e16 + 1 rounds to 1e16 (ulp there is 2), then -1e16 gives 0.
    Matrix A(1, 3), B(3, 1), C(1, 1), D(1, 1);
    A(0, 0) = 1e16; A(0, 1) = 1.0; A(0, 2) = -1e16;
    B(0, 0) = B(1, 0) = B(2, 0) = 1.0;
    naive_multiply(A, B, C);
    reference_multiply_dd(A, B, D);
    CHECK(C(0, 0) == 0.0);  // documents the failure mode the dd reference exists to avoid
    CHECK(D(0, 0) == 1.0);
}

TEST(uniform_naive_is_close_to_dd_reference) {
    Matrix A = make_uniform(48, 48, 20), B = make_uniform(48, 48, 21), C(48, 48), D(48, 48);
    naive_multiply(A, B, C);
    reference_multiply_dd(A, B, D);
    const double err = relative_error_fro(C, D);
    CHECK(err > 0.0);     // naive does round...
    CHECK(err < 1e-14);   // ...but only at the level of a few ulps
}

TEST(compare_utilities) {
    Matrix X(2, 2), Y(2, 2);
    X(0, 0) = 3.0; X(1, 1) = 4.0;
    Y = X;
    Y(1, 0) = -0.5;
    CHECK(frobenius_norm(X) == 5.0);
    CHECK(max_abs(Y) == 4.0);
    CHECK(max_abs_diff(X, Y) == 0.5);
    CHECK(!exactly_equal(X, Y));
    CHECK_NEAR(relative_error_fro(Y, X), 0.1, 1e-15);
    Matrix Huge(1, 2);
    Huge(0, 0) = 1e300; Huge(0, 1) = 1e300;  // naive sum of squares would overflow
    CHECK_NEAR(frobenius_norm(Huge) / 1e300, std::sqrt(2.0), 1e-15);
    CHECK_THROWS(max_abs_diff(X, Matrix(3, 3)));
}
