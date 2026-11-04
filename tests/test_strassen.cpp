#include "algo_strassen/strassen.hpp"
#include "common/compare.hpp"
#include "common/generators.hpp"
#include "reference/naive.hpp"
#include "test_framework.hpp"

using namespace mm;
using strassen::Padding;

namespace {
strassen::Params params(std::size_t cutoff, Padding pad, std::size_t tile = 16, std::size_t temp_pad = 0) {
    strassen::Params p;
    p.cutoff = cutoff;
    p.padding = pad;
    p.base.tile = tile;
    p.temp_pad = temp_pad;
    return p;
}

bool same_stats(const strassen::Stats& a, const strassen::Stats& b) {
    return a.n == b.n && a.padded_n == b.padded_n && a.depth == b.depth && a.base_size == b.base_size &&
           a.base_calls == b.base_calls && a.base_flops == b.base_flops && a.add_flops == b.add_flops &&
           a.copy_elems == b.copy_elems && a.workspace_bytes == b.workspace_bytes &&
           a.padding_bytes == b.padding_bytes;
}
}  // namespace

TEST(strassen_padded_size_rules) {
    // Minimal padding: fewest halvings d with ceil(n/2^d) <= cutoff, then m = ceil(n/2^d) * 2^d.
    CHECK_EQ(strassen::padded_size(1025, params(64, Padding::Minimal)), 1056u);  // 33 * 2^5
    CHECK_EQ(strassen::padded_size(1024, params(64, Padding::Minimal)), 1024u);
    CHECK_EQ(strassen::padded_size(1000, params(64, Padding::Minimal)), 1008u);  // 63 * 2^4
    CHECK_EQ(strassen::padded_size(100, params(128, Padding::Minimal)), 100u);   // below cutoff: untouched
    CHECK_EQ(strassen::padded_size(129, params(128, Padding::Minimal)), 130u);   // 65 * 2
    CHECK_EQ(strassen::padded_size(7, params(1, Padding::Minimal)), 8u);         // cutoff 1 -> power of two
    // NextPow2 padding.
    CHECK_EQ(strassen::padded_size(1025, params(64, Padding::NextPow2)), 2048u);
    CHECK_EQ(strassen::padded_size(1000, params(64, Padding::NextPow2)), 1024u);
    CHECK_EQ(strassen::padded_size(64, params(64, Padding::NextPow2)), 64u);
}

TEST(strassen_aligned_padding_rules) {
    auto aligned = [](std::size_t cutoff, std::size_t align) {
        strassen::Params p = params(cutoff, Padding::Aligned);
        p.leaf_align = align;
        return p;
    };
    CHECK_EQ(strassen::padded_size(1028, aligned(128, 4)), 1088u);  // leaf 65 -> 68, 68 * 16
    CHECK_EQ(strassen::padded_size(1028, aligned(128, 8)), 1152u);  // leaf 65 -> 72, 72 * 16
    CHECK_EQ(strassen::padded_size(1024, aligned(128, 4)), 1024u);  // leaf 128 already aligned
    CHECK_EQ(strassen::padded_size(1020, aligned(128, 8)), 1024u);  // leaf 128 (1020 -> 510 -> 255 -> 128)
    CHECK_EQ(strassen::padded_size(100, aligned(128, 4)), 100u);    // below cutoff: no recursion, no padding
    // Rounding past the cutoff forces one more halving: n = 127 * 2 = 254, cutoff 126, align 4:
    // minimal leaf 127 -> 128 > 126, so halve again: leaf ceil(254/4) = 64 (aligned), m = 256.
    CHECK_EQ(strassen::padded_size(254, aligned(126, 4)), 256u);
    // The leaf of every aligned plan is a multiple of leaf_align.
    for (std::size_t n = 129; n < 1200; n += 7)
        for (std::size_t a : {4u, 8u}) {
            const auto st = strassen::predict(n, aligned(128, a));
            CHECK(st.base_size % a == 0 && st.base_size <= 128);
        }
    CHECK_THROWS(strassen::padded_size(500, aligned(2, 4)));  // leaf_align > cutoff
    CHECK(strassen::parse_padding("aligned") == Padding::Aligned);
    CHECK(std::string(strassen::padding_name(Padding::Aligned)) == "aligned");
}

TEST(strassen_aligned_padding_is_exact) {
    for (std::size_t n : {33u, 61u, 100u, 129u, 200u}) {
        Matrix A = make_small_int(n, n, 700 + n, 8), B = make_small_int(n, n, 800 + n, 8), ref(n, n);
        reference::naive_multiply(A, B, ref);
        for (std::size_t c : {8u, 16u, 33u})
            for (std::size_t a : {4u, 8u}) {
                strassen::Params p = params(c, Padding::Aligned, 7);
                p.leaf_align = a;
                Matrix C(n, n);
                const strassen::Stats st = strassen::multiply(A, B, C, p);
                CHECK(exactly_equal(C, ref));
                CHECK(same_stats(st, strassen::predict(n, p)));
            }
    }
}

TEST(strassen_exact_on_small_int_all_sizes_1_to_130) {
    // Integer inputs make every intermediate exact, so Strassen must equal naive bit-for-bit.
    // Cutoff 1 = "pure" Strassen down to 1x1 (7^8 leaves at n=256, so only run it for
    // n <= 40 to keep the suite fast in Debug builds); the others exercise mixed depths.
    const std::size_t cutoffs[] = {1, 2, 5, 16, 33};
    for (std::size_t n = 1; n <= 130; ++n) {
        Matrix A = make_small_int(n, n, 300 + n, 8), B = make_small_int(n, n, 400 + n, 8), ref(n, n);
        reference::naive_multiply(A, B, ref);
        for (std::size_t c : cutoffs)
            for (Padding pad : {Padding::Minimal, Padding::NextPow2}) {
                if (c == 1 && n > 40) continue;
                Matrix C(n, n);
                strassen::multiply(A, B, C, params(c, pad, 7));
                CHECK(exactly_equal(C, ref));
            }
    }
}

TEST(strassen_uniform_error_is_small_and_larger_than_blocked) {
    const std::size_t n = 256;
    Matrix A = make_uniform(n, n, 11), B = make_uniform(n, n, 12), ref(n, n), Cs(n, n), Cb(n, n);
    reference::reference_multiply_dd(A, B, ref);
    strassen::multiply(A, B, Cs, params(8, Padding::Minimal));
    blocked::multiply(A, B, Cb, {32});
    const double es = relative_error_fro(Cs, ref), eb = relative_error_fro(Cb, ref);
    CHECK(es < 1e-13);
    CHECK(es > eb);  // 5 Strassen levels add rounding error beyond the classical algorithm
}

TEST(strassen_strided_inputs_and_output_block) {
    const std::size_t n = 96;  // even at every level down to cutoff 12 -> no padding copy path
    Matrix A = make_small_int(n, n, 21, 8, 128), B = make_small_int(n, n, 22, 8, 100), ref(n, n);
    reference::naive_multiply(A, B, ref);
    Matrix big(120, 120);
    big.fill(-7.0);
    MatrixView out = big.view().block(10, 20, n, n);
    strassen::multiply(A, B, out, params(12, Padding::Minimal));
    CHECK(exactly_equal(out, ref));
    CHECK(big(9, 20) == -7.0 && big(10, 19) == -7.0 && big(106, 50) == -7.0 && big(50, 116) == -7.0);
    // Padded path with strided views too.
    const std::size_t odd = 77;
    Matrix A2 = make_small_int(odd, odd, 23, 8, 90), B2 = make_small_int(odd, odd, 24, 8, 80), ref2(odd, odd);
    reference::naive_multiply(A2, B2, ref2);
    MatrixView out2 = big.view().block(0, 0, odd, odd);
    strassen::multiply(A2, B2, out2, params(10, Padding::Minimal));
    CHECK(exactly_equal(out2, ref2));
}

TEST(strassen_plan_is_reusable) {
    const std::size_t n = 70;  // padded (minimal, cutoff 8 -> 72)
    strassen::Plan plan(n, params(8, Padding::Minimal));
    for (std::uint64_t seed = 0; seed < 3; ++seed) {
        Matrix A = make_small_int(n, n, 2 * seed, 8), B = make_small_int(n, n, 2 * seed + 1, 8);
        Matrix ref(n, n), C(n, n);
        reference::naive_multiply(A, B, ref);
        plan.execute(A, B, C);
        CHECK(exactly_equal(C, ref));  // stale data from a previous execution must not leak in
    }
    Matrix wrong(71, 71);
    CHECK_THROWS(plan.execute(wrong, wrong, wrong.view()));
}

TEST(strassen_measured_operation_counts_match_cost_model) {
    // The report's cost model (predict) must describe what the code actually does.
    const std::size_t sizes[] = {1, 2, 3, 64, 65, 100, 128, 129, 255, 256};
    const std::size_t cutoffs[] = {1, 4, 16, 64};
    for (std::size_t n : sizes)
        for (std::size_t c : cutoffs)
            for (Padding pad : {Padding::Minimal, Padding::NextPow2}) {
                const auto p = params(c, pad);
                Matrix A = make_uniform(n, n, 1), B = make_uniform(n, n, 2), C(n, n);
                const strassen::Stats measured = strassen::multiply(A, B, C, p);
                CHECK(same_stats(measured, strassen::predict(n, p)));
            }
}

TEST(strassen_temp_pad_is_exact_and_accounted) {
    // Padding the temporaries' stride must not change the answer, only the workspace size.
    for (std::size_t n : {16u, 61u, 128u}) {
        Matrix A = make_small_int(n, n, 500 + n, 8), B = make_small_int(n, n, 600 + n, 8), ref(n, n);
        reference::naive_multiply(A, B, ref);
        for (std::size_t pad : {1u, 3u, 8u}) {
            const auto p = params(8, Padding::Minimal, 16, pad);
            Matrix C(n, n);
            const strassen::Stats st = strassen::multiply(A, B, C, p);
            CHECK(exactly_equal(C, ref));
            CHECK(same_stats(st, strassen::predict(n, p)));
        }
    }
    // n = 256, cutoff 32, pad 8: temporaries are h x (h + 8), rounded to 8 doubles.
    const auto st = strassen::predict(256, params(32, Padding::Minimal, 16, 8));
    CHECK_EQ(st.workspace_bytes, 3u * (128 * 136 + 64 * 72 + 32 * 40) * sizeof(double));
}

TEST(strassen_cost_model_closed_forms) {
    // n = 256, cutoff 32: depth 3, 7^3 = 343 leaves of size 32.
    const auto st = strassen::predict(256, params(32, Padding::Minimal));
    CHECK_EQ(st.depth, 3u);
    CHECK_EQ(st.base_size, 32u);
    CHECK_EQ(st.base_calls, 343u);
    CHECK_EQ(st.base_flops, 343ull * 2 * 32 * 32 * 32);
    // additions: 18 * (128^2 + 7 * 64^2 + 49 * 32^2)
    CHECK_EQ(st.add_flops, 18ull * (128 * 128 + 7 * 64 * 64 + 49 * 32 * 32));
    CHECK_EQ(st.workspace_bytes, 3u * (128 * 128 + 64 * 64 + 32 * 32) * sizeof(double));
    CHECK_EQ(st.padding_bytes, 0u);
    // Fewer total flops than classical 2n^3 at this size: 7/8 per level on the multiply part.
    CHECK(st.base_flops < 2ull * 256 * 256 * 256);
    // n below the cutoff: no recursion at all, plain blocked.
    const auto flat = strassen::predict(100, params(128, Padding::Minimal));
    CHECK_EQ(flat.depth, 0u);
    CHECK_EQ(flat.base_calls, 1u);
    CHECK_EQ(flat.add_flops, 0u);
    CHECK_EQ(flat.workspace_bytes, 0u);
}

TEST(strassen_rejects_bad_arguments) {
    Matrix A(8, 8), B(8, 8), C(8, 8), rect(8, 6), rectC(8, 6);
    CHECK_THROWS(strassen::multiply(A, B, C, params(0, Padding::Minimal)));
    CHECK_THROWS(strassen::multiply(A, B, C, params(4, Padding::Minimal, 0)));
    CHECK_THROWS(strassen::multiply(A, rect, rectC.view()));  // non-square
    CHECK_THROWS(strassen::multiply(A, B, A));                // aliasing
    CHECK_THROWS(strassen::parse_padding("pow2"));
    CHECK(strassen::parse_padding("nextpow2") == Padding::NextPow2);
    CHECK(std::string(strassen::padding_name(Padding::Minimal)) == "minimal");
    strassen::multiply(Matrix(0, 0), Matrix(0, 0), Matrix(0, 0).view());  // empty is a no-op
}
