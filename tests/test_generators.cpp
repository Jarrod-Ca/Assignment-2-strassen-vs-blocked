#include <cmath>

#include "common/generators.hpp"
#include "common/rng.hpp"
#include "test_framework.hpp"

using namespace mm;

// Golden values recorded on Linux/g++. If these fail on Windows/MSVC, the input
// instances would differ between platforms, which would break reproducibility.
TEST(rng_matches_golden_values_across_platforms) {
    Rng rng(42);
    CHECK(rng.next_u64() == 1546998764402558742ULL);
    CHECK(rng.next_u64() == 6990951692964543102ULL);
    CHECK(rng.next_u64() == 12544586762248559009ULL);
    Matrix U = make_uniform(2, 2, 42);
    CHECK(U(0, 0) == -0x1.aa1fd347cf45p-1);
    CHECK(U(1, 1) == 0x1.b2e2b51c0ecd8p-1);
    Matrix W = make_wide_range(2, 2, 7, 20);
    CHECK(W(0, 1) == -0x1.fb2938731e807p+10);
    CHECK_EQ(derive_seed(42, 0), 14473931205035997723ULL);
}

TEST(generators_are_deterministic) {
    const InputSpec specs[] = {{Family::Uniform}, {Family::SmallInt, 8}, {Family::WideRange, 0, 20}};
    for (const auto& s : specs) {
        Matrix X = make_matrix(s, 9, 11, 123), Y = make_matrix(s, 9, 11, 123), Z = make_matrix(s, 9, 11, 124);
        bool same = true, differs = false;
        for (std::size_t i = 0; i < 9; ++i)
            for (std::size_t j = 0; j < 11; ++j) {
                same = same && X(i, j) == Y(i, j);
                differs = differs || X(i, j) != Z(i, j);
            }
        CHECK(same);
        CHECK(differs);
    }
}

TEST(generators_respect_their_ranges) {
    Matrix U = make_uniform(50, 50, 1);
    Matrix S = make_small_int(50, 50, 2, 8);
    Matrix W = make_wide_range(50, 50, 3, 20);
    bool saw_neg_bound = false, saw_pos_bound = false;
    for (std::size_t i = 0; i < 50; ++i)
        for (std::size_t j = 0; j < 50; ++j) {
            CHECK(U(i, j) >= -1.0 && U(i, j) < 1.0);
            CHECK(S(i, j) == std::floor(S(i, j)) && std::fabs(S(i, j)) <= 8.0);
            saw_neg_bound = saw_neg_bound || S(i, j) == -8.0;
            saw_pos_bound = saw_pos_bound || S(i, j) == 8.0;
            const double a = std::fabs(W(i, j));
            CHECK(a >= std::ldexp(1.0, -20) && a < std::ldexp(1.0, 21));
        }
    CHECK(saw_neg_bound && saw_pos_bound);  // inclusive bounds actually reachable
}

TEST(generators_honour_stride_and_identity) {
    Matrix U = make_uniform(3, 3, 5, 8);
    CHECK_EQ(U.stride(), 8u);
    Matrix packed = make_uniform(3, 3, 5);
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j) CHECK(U(i, j) == packed(i, j));  // stride doesn't change values
    Matrix I = make_identity(4);
    for (std::size_t i = 0; i < 4; ++i)
        for (std::size_t j = 0; j < 4; ++j) CHECK(I(i, j) == (i == j ? 1.0 : 0.0));
    CHECK(parse_family("widerange") == Family::WideRange);
    CHECK_THROWS(parse_family("nope"));
}
