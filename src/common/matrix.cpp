#include "common/matrix.hpp"

#include <algorithm>
#include <cstring>
#include <string>
#include <utility>

namespace mm {

namespace {
double* allocate(std::size_t count) {
    if (count == 0) return nullptr;
    void* p = ::operator new(count * sizeof(double), std::align_val_t{kAlignment});
    std::memset(p, 0, count * sizeof(double));
    return static_cast<double*>(p);
}
}  // namespace

Matrix::Matrix(std::size_t rows, std::size_t cols, std::size_t stride)
    : rows_(rows), cols_(cols), stride_(stride == 0 ? cols : stride) {
    if (stride_ < cols_) throw std::invalid_argument("Matrix: stride < cols");
    data_ = allocate(rows_ * stride_);
}

Matrix::~Matrix() { release(); }

void Matrix::release() noexcept {
    if (data_) ::operator delete(data_, std::align_val_t{kAlignment});
    data_ = nullptr;
}

Matrix::Matrix(const Matrix& other) : rows_(other.rows_), cols_(other.cols_), stride_(other.stride_) {
    data_ = allocate(rows_ * stride_);
    if (data_) std::memcpy(data_, other.data_, bytes());
}

Matrix& Matrix::operator=(const Matrix& other) {
    if (this != &other) {
        Matrix tmp(other);
        *this = std::move(tmp);
    }
    return *this;
}

Matrix::Matrix(Matrix&& other) noexcept
    : data_(std::exchange(other.data_, nullptr)),
      rows_(std::exchange(other.rows_, 0)),
      cols_(std::exchange(other.cols_, 0)),
      stride_(std::exchange(other.stride_, 0)) {}

Matrix& Matrix::operator=(Matrix&& other) noexcept {
    if (this != &other) {
        release();
        data_ = std::exchange(other.data_, nullptr);
        rows_ = std::exchange(other.rows_, 0);
        cols_ = std::exchange(other.cols_, 0);
        stride_ = std::exchange(other.stride_, 0);
    }
    return *this;
}

void Matrix::fill(double value) {
    for (std::size_t i = 0; i < rows_; ++i) std::fill(data_ + i * stride_, data_ + i * stride_ + cols_, value);
}

void require_mult_shapes(ConstMatrixView A, ConstMatrixView B, ConstMatrixView C) {
    if (A.cols() != B.rows() || C.rows() != A.rows() || C.cols() != B.cols()) {
        throw std::invalid_argument("shape mismatch: A is " + std::to_string(A.rows()) + "x" +
                                    std::to_string(A.cols()) + ", B is " + std::to_string(B.rows()) + "x" +
                                    std::to_string(B.cols()) + ", C is " + std::to_string(C.rows()) + "x" +
                                    std::to_string(C.cols()));
    }
}

}  // namespace mm
