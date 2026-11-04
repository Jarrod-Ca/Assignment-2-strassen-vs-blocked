#include "algo_strassen/strassen.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>

#include "common/platform.hpp"

namespace mm::strassen {

namespace {

using CV = ConstMatrixView;
using MV = MatrixView;

// Each temporary is h rows of stride (h + pad), rounded up to a whole number of
// cache lines (8 doubles) so every temporary starts 64-byte aligned.
std::size_t temp_elems(std::size_t h, std::size_t pad) { return (h * (h + pad) + 7) / 8 * 8; }

std::size_t workspace_elems(std::size_t m, const Params& p) {
    std::size_t total = 0;
    for (std::size_t s = m; s > p.cutoff; s /= 2) total += 3 * temp_elems(s / 2, p.temp_pad);
    return total;
}

// ---- element-wise helpers on h x h views (Z must not alias X or Y) ----------
// Row kernels take restrict-qualified parameters (the form compilers reliably
// honour) so each vectorises without a runtime aliasing check.
inline void add_row(double* MM_RESTRICT z, const double* MM_RESTRICT x, const double* MM_RESTRICT y, std::size_t w) {
    for (std::size_t j = 0; j < w; ++j) z[j] = x[j] + y[j];
}
inline void sub_row(double* MM_RESTRICT z, const double* MM_RESTRICT x, const double* MM_RESTRICT y, std::size_t w) {
    for (std::size_t j = 0; j < w; ++j) z[j] = x[j] - y[j];
}
inline void add_into_row(double* MM_RESTRICT z, const double* MM_RESTRICT x, std::size_t w) {
    for (std::size_t j = 0; j < w; ++j) z[j] += x[j];
}
inline void sub_into_row(double* MM_RESTRICT z, const double* MM_RESTRICT x, std::size_t w) {
    for (std::size_t j = 0; j < w; ++j) z[j] -= x[j];
}

void add(CV X, CV Y, MV Z) {  // Z = X + Y
    for (std::size_t i = 0; i < Z.rows(); ++i) add_row(Z.row(i), X.row(i), Y.row(i), Z.cols());
}
void sub(CV X, CV Y, MV Z) {  // Z = X - Y
    for (std::size_t i = 0; i < Z.rows(); ++i) sub_row(Z.row(i), X.row(i), Y.row(i), Z.cols());
}
void add_into(CV X, MV Z) {  // Z += X
    for (std::size_t i = 0; i < Z.rows(); ++i) add_into_row(Z.row(i), X.row(i), Z.cols());
}
void sub_into(CV X, MV Z) {  // Z -= X
    for (std::size_t i = 0; i < Z.rows(); ++i) sub_into_row(Z.row(i), X.row(i), Z.cols());
}
void copy(CV X, MV Z) {
    for (std::size_t i = 0; i < Z.rows(); ++i) std::memcpy(Z.row(i), X.row(i), Z.cols() * sizeof(double));
}

struct Context {
    std::size_t cutoff;
    std::size_t temp_pad;
    blocked::Params base;
    Stats* stats;
};

// C = A * B, all n x n. n is either <= cutoff or even (guaranteed by padding).
// `ws` points at this level's workspace; children use the space after it.
void recurse(CV A, CV B, MV C, double* ws, Context& ctx) {
    const std::size_t n = A.rows();
    if (n <= ctx.cutoff) {
        blocked::multiply(A, B, C, ctx.base);
        ctx.stats->base_calls += 1;
        ctx.stats->base_flops += 2ull * n * n * n;
        return;
    }
    const std::size_t h = n / 2;
    const CV A11 = A.block(0, 0, h, h), A12 = A.block(0, h, h, h), A21 = A.block(h, 0, h, h), A22 = A.block(h, h, h, h);
    const CV B11 = B.block(0, 0, h, h), B12 = B.block(0, h, h, h), B21 = B.block(h, 0, h, h), B22 = B.block(h, h, h, h);
    const MV C11 = C.block(0, 0, h, h), C12 = C.block(0, h, h, h), C21 = C.block(h, 0, h, h), C22 = C.block(h, h, h, h);

    const std::size_t te = temp_elems(h, ctx.temp_pad), ts = h + ctx.temp_pad;
    const MV T1(ws, h, h, ts), T2(ws + te, h, h, ts), M(ws + 2 * te, h, h, ts);
    double* child = ws + 3 * te;

    // Products go straight into C quadrants where possible (M1, M2, M3), so only
    // one quadrant copy is needed per node. Comments track what each C quadrant holds.
    add(A11, A22, T1); add(B11, B22, T2); recurse(T1, T2, C11, child, ctx);  // C11 = M1
    copy(C11, C22);                                                           // C22 = M1
    add(A21, A22, T1); recurse(T1, B11, C21, child, ctx);                     // C21 = M2
    sub_into(C21, C22);                                                       // C22 = M1 - M2
    sub(B12, B22, T2); recurse(A11, T2, C12, child, ctx);                     // C12 = M3
    add_into(C12, C22);                                                       // C22 = M1 - M2 + M3
    sub(B21, B11, T2); recurse(A22, T2, M, child, ctx);                       // M = M4
    add_into(M, C11); add_into(M, C21);                                       // C11 = M1 + M4, C21 = M2 + M4 (done)
    add(A11, A12, T1); recurse(T1, B22, M, child, ctx);                       // M = M5
    sub_into(M, C11); add_into(M, C12);                                       // C11 = M1 + M4 - M5, C12 = M3 + M5 (done)
    sub(A21, A11, T1); add(B11, B12, T2); recurse(T1, T2, M, child, ctx);     // M = M6
    add_into(M, C22);                                                         // C22 = M1 - M2 + M3 + M6 (done)
    sub(A12, A22, T1); add(B21, B22, T2); recurse(T1, T2, M, child, ctx);     // M = M7
    add_into(M, C11);                                                         // C11 = M1 + M4 - M5 + M7 (done)

    // 10 operand additions + 8 accumulations = 18 additions of h^2 elements each.
    ctx.stats->add_flops += 18ull * h * h;
    ctx.stats->copy_elems += 1ull * h * h;
}

void copy_into_top_left(CV src, MV dst) {
    for (std::size_t i = 0; i < src.rows(); ++i) std::memcpy(dst.row(i), src.row(i), src.cols() * sizeof(double));
}

}  // namespace

void validate(const Params& p) {
    if (p.cutoff == 0) throw std::invalid_argument("strassen: cutoff must be >= 1");
    if (p.padding != Padding::Minimal && p.padding != Padding::NextPow2)
        throw std::invalid_argument("strassen: unknown padding mode");
    blocked::validate(p.base);
}

std::size_t padded_size(std::size_t n, const Params& p) {
    validate(p);
    if (n <= p.cutoff) return n;
    if (p.padding == Padding::NextPow2) {
        std::size_t m = 1;
        while (m < n) m *= 2;
        return m;
    }
    std::size_t s = n, d = 0;
    while (s > p.cutoff) { s = (s + 1) / 2; ++d; }  // ceil-halving: s = ceil(n / 2^d)
    return s << d;
}

Stats predict(std::size_t n, const Params& p) {
    Stats st;
    st.n = n;
    st.padded_n = padded_size(n, p);
    std::size_t s = st.padded_n;
    std::uint64_t nodes = 1;  // 7^level
    while (s > p.cutoff) {
        const std::uint64_t h = s / 2;
        st.add_flops += nodes * 18ull * h * h;
        st.copy_elems += nodes * h * h;
        nodes *= 7;
        s /= 2;
        ++st.depth;
    }
    st.base_size = s;
    st.base_calls = n == 0 ? 0 : nodes;
    st.base_flops = st.base_calls * 2ull * s * s * s;
    st.workspace_bytes = workspace_elems(st.padded_n, p) * sizeof(double);
    st.padding_bytes = st.padded_n == n ? 0 : 3 * st.padded_n * st.padded_n * sizeof(double);
    return st;
}

Plan::Plan(std::size_t n, const Params& p) : n_(n), m_(padded_size(n, p)), params_(p) {
    workspace_ = AlignedBuffer(workspace_elems(m_, p));
    if (m_ != n_) {
        Ap_ = Matrix(m_, m_);  // zero-initialised: the padding stays zero across executions
        Bp_ = Matrix(m_, m_);
        Cp_ = Matrix(m_, m_);
    }
}

Stats Plan::execute(ConstMatrixView A, ConstMatrixView B, MatrixView C) {
    require_mult_shapes(A, B, C);
    if (A.rows() != n_ || A.cols() != n_ || B.cols() != n_)
        throw std::invalid_argument("strassen::Plan::execute: plan is for " + std::to_string(n_) + "x" +
                                    std::to_string(n_) + " matrices");
    require_no_alias(A, B, C);

    Stats st;
    st.n = n_;
    st.padded_n = m_;
    st.workspace_bytes = workspace_.bytes();
    st.padding_bytes = Ap_.bytes() + Bp_.bytes() + Cp_.bytes();
    if (n_ == 0) return st;

    Context ctx{params_.cutoff, params_.temp_pad, params_.base, &st};
    if (m_ == n_) {
        recurse(A, B, C, workspace_.data(), ctx);
    } else {
        copy_into_top_left(A, Ap_.view());  // rows/cols beyond n stay zero from construction
        copy_into_top_left(B, Bp_.view());
        recurse(Ap_, Bp_, Cp_.view(), workspace_.data(), ctx);
        copy_into_top_left(Cp_.view().block(0, 0, n_, n_), C);
    }
    std::size_t s = m_;
    while (s > params_.cutoff) { s /= 2; ++st.depth; }
    st.base_size = s;
    return st;
}

Stats multiply(ConstMatrixView A, ConstMatrixView B, MatrixView C, const Params& p) {
    require_mult_shapes(A, B, C);
    if (A.rows() != A.cols() || B.rows() != B.cols())
        throw std::invalid_argument("strassen: only square matrices are supported");
    Plan plan(A.rows(), p);
    return plan.execute(A, B, C);
}

const char* padding_name(Padding p) { return p == Padding::NextPow2 ? "nextpow2" : "minimal"; }

Padding parse_padding(const char* name) {
    const std::string s(name);
    if (s == "minimal") return Padding::Minimal;
    if (s == "nextpow2") return Padding::NextPow2;
    throw std::invalid_argument("unknown padding mode: " + s);
}

}  // namespace mm::strassen
