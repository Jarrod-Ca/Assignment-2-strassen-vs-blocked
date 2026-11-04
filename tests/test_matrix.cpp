#include <cstdint>

#include "common/matrix.hpp"
#include "test_framework.hpp"

using namespace mm;

TEST(matrix_is_zero_initialised_and_aligned) {
    Matrix M(7, 5);
    CHECK_EQ(M.rows(), 7u);
    CHECK_EQ(M.cols(), 5u);
    CHECK_EQ(M.stride(), 5u);
    CHECK(reinterpret_cast<std::uintptr_t>(M.data()) % kAlignment == 0);
    for (std::size_t i = 0; i < 7; ++i)
        for (std::size_t j = 0; j < 5; ++j) CHECK(M(i, j) == 0.0);
}

TEST(matrix_custom_stride) {
    Matrix M(4, 3, 8);
    CHECK_EQ(M.stride(), 8u);
    CHECK_EQ(M.bytes(), 4u * 8u * sizeof(double));
    M(2, 1) = 5.0;
    CHECK(M.data()[2 * 8 + 1] == 5.0);
    CHECK_THROWS(Matrix(2, 4, 3));  // stride < cols
}

TEST(matrix_copy_is_deep_and_move_steals) {
    Matrix A(3, 3);
    A(1, 1) = 2.0;
    Matrix B = A;
    B(1, 1) = 9.0;
    CHECK(A(1, 1) == 2.0);
    Matrix C = std::move(B);
    CHECK(C(1, 1) == 9.0);
    CHECK(B.data() == nullptr);
}

TEST(view_blocks_share_storage) {
    Matrix M(4, 4);
    MatrixView q = M.view().block(2, 2, 2, 2);  // bottom-right quadrant
    q(0, 0) = 7.0;
    q(1, 1) = 8.0;
    CHECK(M(2, 2) == 7.0);
    CHECK(M(3, 3) == 8.0);
    CHECK_EQ(q.stride(), 4u);
    CHECK_THROWS(M.view().block(3, 3, 2, 2));
    ConstMatrixView cq = q;  // mutable -> const conversion
    CHECK(cq(0, 0) == 7.0);
}

TEST(require_mult_shapes_rejects_mismatch) {
    Matrix A(2, 3), B(3, 4), C(2, 4), Bad(2, 4);
    require_mult_shapes(A, B, C);  // must not throw
    CHECK_THROWS(require_mult_shapes(A, Bad, C));
    CHECK_THROWS(require_mult_shapes(A, B, Bad.view().block(0, 0, 2, 3)));
}

TEST(overlap_detection) {
    Matrix M(8, 8), N(8, 8);
    MatrixView top = M.view().block(0, 0, 4, 8), bottom = M.view().block(4, 0, 4, 8);
    CHECK(!views_overlap(top, bottom));
    CHECK(views_overlap(M, top));
    CHECK(!views_overlap(M, N));
    CHECK(views_overlap(M.view().block(0, 0, 2, 2), M.view().block(1, 1, 2, 2)));
    CHECK(!views_overlap(M.view().block(0, 0, 0, 0), M));  // empty views never overlap
    CHECK_THROWS(require_no_alias(M, N, M));
    CHECK_THROWS(require_no_alias(N, M, M.view().block(2, 2, 2, 2)));
    require_no_alias(top, top, bottom);  // inputs may alias each other; only C is checked
}

TEST(aligned_buffer_alignment_and_move) {
    AlignedBuffer buf(100);
    CHECK_EQ(buf.size(), 100u);
    CHECK_EQ(buf.bytes(), 800u);
    CHECK(reinterpret_cast<std::uintptr_t>(buf.data()) % kAlignment == 0);
    buf.data()[99] = 1.5;
    AlignedBuffer moved = std::move(buf);
    CHECK(moved.data()[99] == 1.5);
    CHECK(buf.data() == nullptr);
    AlignedBuffer empty(0);
    CHECK(empty.data() == nullptr);
}
