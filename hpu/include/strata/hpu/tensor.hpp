#pragma once

#include <bit>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace strata::hpu {
// F8 is the Gaudi2 MME FP8 format syn_type_fp8_143 (1 sign, 4 exponent, 3 mantissa bits, exponent bias 7,
// largest finite value 240). It only feeds GEMMs and casts; it is never a host input or output.
// I32 holds graph-internal indices (gather); it is never a host input or output either.
// U4 is packed syn_type_uint4 storage: two codes per byte, with optional per-output
// quantization metadata. The installed Gaudi2 path explicitly casts to a supported
// MME operand type; this storage enum does not imply native mixed U4/BF16 GEMM.
enum class DType { F32, BF16, U8, F8, I32, U4 };
inline uint64_t element_bytes(DType dtype) {
    switch(dtype) {
        case DType::F32:return 4;case DType::BF16:return 2;case DType::U8:return 1;case DType::F8:return 1;case DType::I32:return 4;
        case DType::U4:throw std::invalid_argument("U4 elements are four bits; use TensorDesc::bytes()");
    }
    throw std::invalid_argument("invalid tensor dtype");
}

inline uint64_t checked_product(uint64_t a, uint64_t b) {
    if (b && a > std::numeric_limits<uint64_t>::max() / b)
        throw std::overflow_error("tensor size overflow");
    return a * b;
}
inline void check_range(uint64_t capacity, uint64_t offset, uint64_t bytes) {
    if (offset > capacity || bytes > capacity - offset)
        throw std::out_of_range("buffer range exceeds allocation");
}

// Logical row-major dimensions. SDK dimensions are reversed at the graph boundary.
struct TensorDesc {
    std::string name;
    DType dtype = DType::F32;
    std::vector<uint64_t> shape;
    bool persistent = true;

    uint64_t bytes() const {
        if (name.empty() || shape.empty() || shape.size() > 5)
            throw std::invalid_argument("tensor needs a name and rank in [1,5]");
        uint64_t count = 1;
        for (auto dim : shape) {
            if (!dim) throw std::invalid_argument("zero tensor dimension");
            count = checked_product(count, dim);
        }
        if (dtype == DType::U4) {
            if (count & 1) throw std::invalid_argument("U4 tensor needs an even element count");
            return count / 2;
        }
        return checked_product(element_bytes(dtype), count);
    }
};

// IEEE round-to-nearest-even, preserving NaNs instead of rounding them to Inf.
inline uint16_t to_bf16(float value) {
    uint32_t bits = std::bit_cast<uint32_t>(value);
    if ((bits & 0x7fffffffU) > 0x7f800000U)
        return static_cast<uint16_t>((bits >> 16) | 0x0040U);
    bits += 0x7fffU + ((bits >> 16) & 1U);
    return static_cast<uint16_t>(bits >> 16);
}
inline float from_bf16(uint16_t value) {
    return std::bit_cast<float>(static_cast<uint32_t>(value) << 16);
}
// Branch-free scans the compiler vectorizes: the same answer as std::isfinite / std::isnan on every element.
inline bool all_finite(std::span<const float> values) {
    uint32_t bad = 0;
    for (const float value : values) bad |= uint32_t((std::bit_cast<uint32_t>(value) & 0x7f800000U) == 0x7f800000U);
    return !bad;
}
inline bool any_nan(std::span<const float> values) {
    uint32_t bad = 0;
    for (const float value : values) bad |= uint32_t((std::bit_cast<uint32_t>(value) & 0x7fffffffU) > 0x7f800000U);
    return bad != 0;
}
} // namespace strata::hpu
