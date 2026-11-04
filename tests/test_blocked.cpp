#include "algo_blocked/blocked.hpp"
#include "common/compare.hpp"
#include "common/generators.hpp"
#include "reference/naive.hpp"
#include "test_framework.hpp"

using namespace mm;

namespace {
// Tiles chosen to hit: degenerate (1), not dividing typical sizes (7), small (16),
// default (64), and larger than every test size (1000 -> single tile).
const std::size_t kTiles[] = {1, 7, 16, 64, 1000};
}  // namespace

TEST(blocked_exact_on_small_int_all_sizes_1_to_130) {
    for (std::size_t n = 1; n <= 130; ++n) {
        Matrix A = make_small_int(n, n, 100 + n, 8), B = make_small_int(n, n, 200 + n, 8);
        Matrix ref(n, n);
        reference::naive_multiply(A, B, ref);
        for (std::size_t t : kTiles) {
            Matrix C(n, n);
            blocked::multiply(A, B, C, {t});
            CHECK(exactly_equal(C, ref));
        }
    }
}

TEST(blocked_rectangular_shapes) {
    const std::size_t shapes[][3] = {{1, 1, 1}, {1, 50, 1}, {50, 1, 50}, {3, 70, 5}, {65, 33, 17}, {17, 129, 64}};
    for (const auto& s : shapes) {
        Matrix A = make_small_int(s[0], s[1], 1, 8), B = make_small_int(s[1], s[2], 2, 8);
        Matrix ref(s[0], s[2]);
        reference::naive_multiply(A, B, ref);
        for (std::size_t t : kTiles) {
            Matrix C(s[0], s[2]);
            blocked::multiply(A, B, C, {t});
            CHECK(exactly_equal(C, ref));
        }
    }
}

TEST(blocked_uniform_close_to_dd_reference) {
    const std::size_t n = 97;
    Matrix A = make_uniform(n, n, 5), B = make_uniform(n, n, 6), ref(n, n), C(n, n);
    reference::reference_multiply_dd(A, B, ref);
    for (std::size_t t : kTiles) {
        blocked::multiply(A, B, C, {t});
        CHECK(relative_error_fro(C, ref) < 1e-14);
    }
}

TEST(blocked_strided_and_sub_block_views) {
    const std::size_t n = 45;
    Matrix A = make_small_int(n, n, 7, 8, 64), B = make_small_int(n, n, 8, 8, 50);  // padded strides
    Matrix ref(n, n), C(n, n, 47);
    reference::naive_multiply(A, B, ref);
    blocked::multiply(A, B, C, {16});
    CHECK(exactly_equal(C, ref));
    // Write into the bottom-right block of a larger matrix; the rest must be untouched.
    Matrix big(60, 60);
    big.fill(-3.0);
    blocked::multiply(A, B, big.view().block(15, 15, n, n), {16});
    CHECK(exactly_equal(big.view().block(15, 15, n, n), ref));
    CHECK(big(0, 0) == -3.0 && big(14, 59) == -3.0 && big(59, 14) == -3.0);
}

TEST(blocked_multiply_add_accumulates) {
    const std::size_t n = 30;
    Matrix A = make_small_int(n, n, 9, 8), B = make_small_int(n, n, 10, 8), ref(n, n), C(n, n);
    reference::naive_multiply(A, B, ref);
    C.fill(2.0);
    blocked::multiply_add(A, B, C, {7});
    bool ok = true;
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) ok = ok && C(i, j) == ref(i, j) + 2.0;
    CHECK(ok);
}

TEST(blocked_rejects_bad_arguments) {
    Matrix A(4, 4), B(4, 4), C(4, 4), wrong(3, 4);
    CHECK_THROWS(blocked::multiply(A, B, C, {0}));
    CHECK_THROWS(blocked::multiply(A, wrong, C));
    CHECK_THROWS(blocked::multiply(A, B, A));                 // C aliases A
    CHECK_THROWS(blocked::multiply_add(A, B, B.view()));      // C aliases B
    C.fill(5.0);
    CHECK_THROWS(blocked::multiply(A, B, C, {0}));
    CHECK(C(0, 0) == 5.0);  // invalid params are rejected before C is modified
    blocked::multiply(Matrix(0, 0), Matrix(0, 0), Matrix(0, 0).view());  // empty is a no-op
}
