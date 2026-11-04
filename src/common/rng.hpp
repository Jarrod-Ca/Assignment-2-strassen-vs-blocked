// rng.hpp - portable, bit-reproducible random number generation.
//
// Why not <random>? The *engines* (e.g. std::mt19937_64) are specified exactly,
// but the *distributions* (std::uniform_real_distribution etc.) are not: MSVC's
// STL and GCC's libstdc++ produce different numbers from the same seed. Our
// benchmarks run on Windows/MSVC and our tests also run on Linux/g++, and the
// methodology promises "same input instances for every algorithm" - so every
// value here is produced with integer arithmetic plus exact conversions only.
//
// Engine: xoshiro256** (Blackman & Vigna, 2018), seeded via SplitMix64.
#pragma once

#include <cstdint>
#include <limits>
#include <stdexcept>

namespace mm {

// SplitMix64 step: advances `state` and returns a well-mixed 64-bit value.
inline std::uint64_t splitmix64(std::uint64_t& state) {
    std::uint64_t z = (state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

// Derive an independent seed for sub-stream `stream` of a base seed
// (e.g. matrix A uses stream 0, matrix B uses stream 1).
inline std::uint64_t derive_seed(std::uint64_t base, std::uint64_t stream) {
    std::uint64_t s = base ^ (0xD1B54A32D192ED03ULL * (stream + 1));
    return splitmix64(s);
}

class Rng {
public:
    explicit Rng(std::uint64_t seed) {
        std::uint64_t x = seed;
        for (auto& s : s_) s = splitmix64(x);
    }

    std::uint64_t next_u64() {
        const std::uint64_t result = rotl(s_[1] * 5, 7) * 9;
        const std::uint64_t t = s_[1] << 17;
        s_[2] ^= s_[0];
        s_[3] ^= s_[1];
        s_[1] ^= s_[2];
        s_[0] ^= s_[3];
        s_[2] ^= t;
        s_[3] = rotl(s_[3], 45);
        return result;
    }

    // Uniform in [0, 1) with 53 random bits; the conversion is exact.
    double next_unit() { return static_cast<double>(next_u64() >> 11) * 0x1.0p-53; }

    // Uniform integer in [lo, hi] inclusive, unbiased (rejection sampling).
    std::int64_t next_int(std::int64_t lo, std::int64_t hi) {
        if (lo > hi) throw std::invalid_argument("Rng::next_int: lo > hi");
        const std::uint64_t range = static_cast<std::uint64_t>(hi) - static_cast<std::uint64_t>(lo) + 1;
        if (range == 0) return static_cast<std::int64_t>(next_u64());  // full 64-bit range
        const std::uint64_t limit = std::numeric_limits<std::uint64_t>::max() -
                                    std::numeric_limits<std::uint64_t>::max() % range;
        std::uint64_t x;
        do { x = next_u64(); } while (x >= limit);
        return static_cast<std::int64_t>(static_cast<std::uint64_t>(lo) + x % range);
    }

private:
    static std::uint64_t rotl(std::uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
    std::uint64_t s_[4];
};

}  // namespace mm
