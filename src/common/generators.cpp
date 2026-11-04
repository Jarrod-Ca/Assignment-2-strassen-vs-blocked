#include "common/generators.hpp"

#include <cmath>
#include <stdexcept>

#include "common/rng.hpp"

namespace mm {

Matrix make_uniform(std::size_t rows, std::size_t cols, std::uint64_t seed, std::size_t stride) {
    Matrix M(rows, cols, stride);
    Rng rng(seed);
    for (std::size_t i = 0; i < rows; ++i)
        for (std::size_t j = 0; j < cols; ++j) M(i, j) = 2.0 * rng.next_unit() - 1.0;  // exact
    return M;
}

Matrix make_small_int(std::size_t rows, std::size_t cols, std::uint64_t seed, std::int64_t bound,
                      std::size_t stride) {
    if (bound < 0) throw std::invalid_argument("make_small_int: bound < 0");
    Matrix M(rows, cols, stride);
    Rng rng(seed);
    for (std::size_t i = 0; i < rows; ++i)
        for (std::size_t j = 0; j < cols; ++j) M(i, j) = static_cast<double>(rng.next_int(-bound, bound));
    return M;
}

Matrix make_wide_range(std::size_t rows, std::size_t cols, std::uint64_t seed, int max_exp,
                       std::size_t stride) {
    if (max_exp < 0) throw std::invalid_argument("make_wide_range: max_exp < 0");
    Matrix M(rows, cols, stride);
    Rng rng(seed);
    for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t j = 0; j < cols; ++j) {
            const double mantissa = 1.0 + rng.next_unit();                       // [1, 2)
            const int e = static_cast<int>(rng.next_int(-max_exp, max_exp));
            const double sign = (rng.next_u64() >> 63) ? -1.0 : 1.0;
            M(i, j) = sign * std::ldexp(mantissa, e);                            // exact
        }
    }
    return M;
}

Matrix make_identity(std::size_t n, std::size_t stride) {
    Matrix M(n, n, stride);
    for (std::size_t i = 0; i < n; ++i) M(i, i) = 1.0;
    return M;
}

Matrix make_matrix(const InputSpec& spec, std::size_t rows, std::size_t cols, std::uint64_t seed,
                   std::size_t stride) {
    switch (spec.family) {
        case Family::Uniform: return make_uniform(rows, cols, seed, stride);
        case Family::SmallInt: return make_small_int(rows, cols, seed, spec.int_bound, stride);
        case Family::WideRange: return make_wide_range(rows, cols, seed, spec.max_exp, stride);
        case Family::Identity:
            if (rows != cols) throw std::invalid_argument("identity must be square");
            return make_identity(rows, stride);
    }
    throw std::invalid_argument("unknown family");
}

const char* family_name(Family f) {
    switch (f) {
        case Family::Uniform: return "uniform";
        case Family::SmallInt: return "smallint";
        case Family::WideRange: return "widerange";
        case Family::Identity: return "identity";
    }
    return "unknown";
}

Family parse_family(const std::string& name) {
    if (name == "uniform") return Family::Uniform;
    if (name == "smallint") return Family::SmallInt;
    if (name == "widerange") return Family::WideRange;
    if (name == "identity") return Family::Identity;
    throw std::invalid_argument("unknown input family: " + name);
}

}  // namespace mm
