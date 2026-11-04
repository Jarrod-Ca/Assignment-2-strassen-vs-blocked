// matrix.hpp - dense row-major matrix storage and non-owning views.
//
// Every algorithm in this project reads and writes matrices through these types,
// so storage layout (alignment, leading dimension) is identical for all of them.
//
// Layout: element (i, j) lives at data[i * stride + j]. `stride` (the leading
// dimension) may be larger than `cols`; this lets us (a) take sub-block views
// without copying, which Strassen needs for its quadrants, and (b) pad rows
// deliberately as an experimental variable (power-of-two strides can cause
// cache-set aliasing).
#pragma once

#include <cstddef>
#include <new>
#include <stdexcept>
#include <type_traits>

namespace mm {

// 64 bytes = one cache line on the target CPU; also satisfies AVX2's 32-byte alignment.
inline constexpr std::size_t kAlignment = 64;

// Non-owning view. T is `double` (mutable) or `const double` (read-only).
template <typename T>
class BasicView {
public:
    BasicView() = default;
    BasicView(T* data, std::size_t rows, std::size_t cols, std::size_t stride)
        : data_(data), rows_(rows), cols_(cols), stride_(stride) {}

    // Allow MatrixView -> ConstMatrixView, never the reverse.
    template <typename U, typename = std::enable_if_t<std::is_same_v<T, const U>>>
    BasicView(const BasicView<U>& other)  // NOLINT(google-explicit-constructor)
        : data_(other.data()), rows_(other.rows()), cols_(other.cols()), stride_(other.stride()) {}

    T* data() const { return data_; }
    std::size_t rows() const { return rows_; }
    std::size_t cols() const { return cols_; }
    std::size_t stride() const { return stride_; }

    T& operator()(std::size_t i, std::size_t j) const { return data_[i * stride_ + j]; }
    T* row(std::size_t i) const { return data_ + i * stride_; }

    // Sub-block [r0, r0+nr) x [c0, c0+nc). Shares storage with the parent.
    BasicView block(std::size_t r0, std::size_t c0, std::size_t nr, std::size_t nc) const {
        if (r0 + nr > rows_ || c0 + nc > cols_) throw std::out_of_range("BasicView::block out of range");
        return BasicView(data_ + r0 * stride_ + c0, nr, nc, stride_);
    }

private:
    T* data_ = nullptr;
    std::size_t rows_ = 0, cols_ = 0, stride_ = 0;
};

using MatrixView = BasicView<double>;
using ConstMatrixView = BasicView<const double>;

// Owning, 64-byte-aligned, zero-initialised matrix.
class Matrix {
public:
    Matrix() = default;
    // stride == 0 means "tightly packed" (stride = cols).
    Matrix(std::size_t rows, std::size_t cols, std::size_t stride = 0);
    ~Matrix();

    Matrix(const Matrix& other);
    Matrix& operator=(const Matrix& other);
    Matrix(Matrix&& other) noexcept;
    Matrix& operator=(Matrix&& other) noexcept;

    std::size_t rows() const { return rows_; }
    std::size_t cols() const { return cols_; }
    std::size_t stride() const { return stride_; }
    double* data() { return data_; }
    const double* data() const { return data_; }

    double& operator()(std::size_t i, std::size_t j) { return data_[i * stride_ + j]; }
    double operator()(std::size_t i, std::size_t j) const { return data_[i * stride_ + j]; }

    MatrixView view() { return MatrixView(data_, rows_, cols_, stride_); }
    ConstMatrixView view() const { return ConstMatrixView(data_, rows_, cols_, stride_); }
    operator MatrixView() { return view(); }             // NOLINT(google-explicit-constructor)
    operator ConstMatrixView() const { return view(); }  // NOLINT(google-explicit-constructor)

    void fill(double value);
    // Bytes actually allocated (rows * stride * sizeof(double)); used for memory accounting.
    std::size_t bytes() const { return rows_ * stride_ * sizeof(double); }

private:
    void release() noexcept;
    double* data_ = nullptr;
    std::size_t rows_ = 0, cols_ = 0, stride_ = 0;
};

// Throws std::invalid_argument unless C(m x n) = A(m x k) * B(k x n) is well-formed.
void require_mult_shapes(ConstMatrixView A, ConstMatrixView B, ConstMatrixView C);

// True if the memory spans of the two views overlap (conservative: compares the
// address ranges [first element, last element], ignoring gaps between rows).
bool views_overlap(ConstMatrixView X, ConstMatrixView Y);

// Throws std::invalid_argument if the output view overlaps either input. Every
// multiply in this project writes C while still reading A and B, so aliasing
// would silently corrupt the result.
void require_no_alias(ConstMatrixView A, ConstMatrixView B, ConstMatrixView C);

// Uninitialised, 64-byte-aligned scratch memory (e.g. Strassen's workspace).
// Not zeroed: callers always write before reading.
class AlignedBuffer {
public:
    AlignedBuffer() = default;
    explicit AlignedBuffer(std::size_t count);
    ~AlignedBuffer();
    AlignedBuffer(const AlignedBuffer&) = delete;
    AlignedBuffer& operator=(const AlignedBuffer&) = delete;
    AlignedBuffer(AlignedBuffer&& other) noexcept;
    AlignedBuffer& operator=(AlignedBuffer&& other) noexcept;

    double* data() { return data_; }
    std::size_t size() const { return count_; }
    std::size_t bytes() const { return count_ * sizeof(double); }

private:
    double* data_ = nullptr;
    std::size_t count_ = 0;
};

}  // namespace mm
