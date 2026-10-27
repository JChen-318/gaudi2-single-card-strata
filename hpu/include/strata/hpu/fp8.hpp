#pragma once

// Host side of the opt-in FP8 weight path: the Gaudi2 MME FP8 format syn_type_fp8_143 (DType::F8).
// One sign bit, four exponent bits with bias 7, three mantissa bits; the largest finite value is 240
// (exponent field 14, mantissa 7) and exponent field 15 is never produced. Subnormals step by 2^-9.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <span>
#include <stdexcept>
#include <vector>

namespace strata::hpu {
inline constexpr float fp8_143_max = 240.0f;

inline float f32_from_fp8_143(uint8_t code) {
    const int exponent = (code >> 3) & 15, mantissa = code & 7;
    const float magnitude = exponent ? std::ldexp(1.0f + float(mantissa) / 8.0f, exponent - 7)
                                     : std::ldexp(float(mantissa), -9);
    return (code & 0x80) ? -magnitude : magnitude;
}

// Round to nearest, ties to even; magnitudes past 240 saturate to 240. Reference form; the bit form below
// is the one the loader uses and tests/fp8_encode.cpp pins the two to the same byte.
inline uint8_t fp8_143_from_f32_reference(float value) {
    if (!std::isfinite(value)) throw std::invalid_argument("FP8 encoding requires finite values");
    const uint8_t sign = std::signbit(value) ? 0x80 : 0;
    const float magnitude = std::fabs(value);
    if (magnitude >= fp8_143_max) return sign | 0x77;
    int binary = 0;
    (void)std::frexp(magnitude, &binary);
    int exponent = binary - 1; // magnitude = 1.m * 2^exponent
    if (magnitude == 0 || exponent < -6) {
        // Subnormal: an integer count of 2^-9; a count of 8 is exactly the smallest normal (code 0x08).
        const auto count = std::nearbyint(std::ldexp(magnitude, 9));
        return sign | uint8_t(count);
    }
    auto mantissa = std::nearbyint((std::ldexp(magnitude, -exponent) - 1.0f) * 8.0f);
    if (mantissa == 8) { mantissa = 0; ++exponent; }
    if (exponent > 7) return sign | 0x77;
    return sign | uint8_t(((exponent + 7) << 3) | int(mantissa));
}
// The same law on the F32 bits: for a normal FP8 result the 23-bit mantissa is rounded to 3 bits
// (ties to even) and the exponent rebased from 127 to 7; below 2^-6 the value is a rounded count of 2^-9.
inline uint8_t fp8_143_from_f32(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof bits);
    const uint32_t magnitude = bits & 0x7fffffffu;
    if (magnitude >= 0x7f800000u) throw std::invalid_argument("FP8 encoding requires finite values");
    const uint8_t sign = uint8_t((bits >> 24) & 0x80u);
    if (magnitude >= 0x43700000u) return sign | 0x77; // >= 240
    if (magnitude >= 0x3c800000u) {                   // >= 2^-6: normal
        const uint32_t rounded = magnitude + 0x7ffffu + ((magnitude >> 20) & 1u);
        const uint32_t code = (rounded >> 20) - (120u << 3);
        return sign | uint8_t(code > 0x77u ? 0x77u : code);
    }
    return sign | uint8_t(std::nearbyint(std::fabs(value) * 512.0f));
}

// Per-row symmetric scaling: row r is stored as fp8(w / scale[r]) with scale[r] = max|w_r| / 240, so
// w ~= f32_from_fp8_143(code) * scale[r]. An all-zero row keeps scale 1 and zero codes.
struct Fp8Rows {
    std::vector<uint8_t> codes;
    std::vector<float> scales;
};
// One row into `codes` (row.size() bytes); returns its scale.
inline float quantize_fp8_row(std::span<const float> row, uint8_t* codes) {
    float peak = 0;
    for (const float v : row) {
        if (!std::isfinite(v)) throw std::invalid_argument("FP8 rows require finite weights");
        peak = std::max(peak, std::fabs(v));
    }
    const float scale = peak > 0 ? peak / fp8_143_max : 1.0f;
    for (size_t c = 0; c < row.size(); ++c) codes[c] = fp8_143_from_f32(row[c] / scale);
    return scale;
}
inline Fp8Rows quantize_fp8_rows(std::span<const float> values, uint64_t rows, uint64_t columns) {
    if (!rows || !columns || values.size() != rows * columns) throw std::invalid_argument("FP8 rows shape mismatch");
    Fp8Rows out;
    out.codes.resize(values.size());
    out.scales.resize(rows);
    for (uint64_t r = 0; r < rows; ++r) out.scales[r] = quantize_fp8_row(values.subspan(r * columns, columns), out.codes.data() + r * columns);
    return out;
}
}
