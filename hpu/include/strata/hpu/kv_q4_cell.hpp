#pragma once
// hpu/include/strata/hpu/kv_q4_cell.hpp - upstream's Q4_0 KV cell (Hadamard rotation + ggml q4_0) as a host law.
//
// --kv q4_0 changes only KV storage (include/strata/kernels/kv_q4.hpp:1-12): every 256-value head row is rotated by
// the orthonormal Fast Walsh-Hadamard matrix (scale 1/16, self-inverse, src/kernels/cuda/kv_q4.cu:25-69), then
// quantized in ggml q4_0 blocks of 32 values. One block is an FP16 scale `d` and 16 code bytes (18 B); element j
// lives in the low nibble of qs[j] and element j+16 in the high one (kv_q4.hpp:23-33). The CUDA law is:
//     amax |x| with the largest signed value `mval` breaking ties (all lanes agree on d), d = mval / -8,
//     q = clamp(trunc(x * (1/d) + 8.5), 0, 15), x' = fp16(((q - 8) * fp16(d)))        (:75-91, :165-173)
// where `1/d` uses the UNROUNDED scale, not the stored FP16 one. head_dim must be 256 (need_256, :177-182).
//
// The rotation happens before append and the query would be rotated the same way. This backend keeps Q unrotated:
// QsaState/DenseKvState store the rotated codes and rotate a gathered row back with H before attention reads it,
// so <q, H'k> == <q, k> up to the 4-bit rounding. That is the named FP-rounding divergence in hpu/PROGRESS.md:
// upstream feeds FP16 rotated rows to a kernel that back-rotates the output, this port feeds FP32 unrotated rows.
//
// No state: QsaState/DenseKvState call this, and hpu/tests/kv_cell.cpp pins the arithmetic independently.
#include "strata/hpu/kv_mode.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/kv_q4.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>

namespace strata::hpu {
inline constexpr uint32_t kv_q4_block = strata::kernels::QK4_0;
inline constexpr uint32_t kv_q4_rotation = 256;
static_assert(kv_q4_block == 32, "upstream's q4_0 block is 32 values");
static_assert(sizeof(strata::kernels::block_q4_0) == 18, "upstream's q4_0 block is 18 bytes");

inline uint32_t kv_q4_blocks_per_head(const uint32_t width) {
    if (width != kv_q4_rotation)
        throw std::invalid_argument("Q4_0 KV requires head_dim 256 (the Hadamard rotation's size)");
    return width / kv_q4_block;
}

inline uint64_t kv_q4_bytes_per_head(const uint32_t width) {
    return uint64_t(kv_q4_blocks_per_head(width)) * sizeof(strata::kernels::block_q4_0);
}

// One cell = one token x one layer: rotated K and V codes, no separate scale rows.
inline uint64_t kv_q4_cell_bytes(const uint32_t width, const uint32_t heads) {
    const uint64_t per_head = kv_q4_bytes_per_head(width) * 2;
    if (heads && per_head > UINT64_MAX / heads) throw std::overflow_error("Q4_0 KV cell byte count overflow");
    return per_head * heads;
}

// fwht256_kernel (:25-69): every value is scaled by 1/16 BEFORE the butterflies, so the transform is orthonormal
// and its own inverse. The CUDA kernel spreads the 8 index bits over lanes and registers; the standard in-place
// transform visits the same index pairs in the same h order, and additions are exact under the same FP32 rounding.
inline void kv_fwht256(std::span<const float> src, std::span<float> dst) {
    if (src.empty() || src.size() % kv_q4_rotation || dst.size() != src.size())
        throw std::invalid_argument("Q4_0 rotation shape mismatch");
    for (size_t i = 0; i < src.size(); ++i) {
        if (!std::isfinite(src[i])) throw std::invalid_argument("Q4_0 KV requires finite values");
        dst[i] = src[i] * (1.0f / 16.0f);
    }
    for (size_t base = 0; base < dst.size(); base += kv_q4_rotation)
        for (uint32_t h = 1; h < kv_q4_rotation; h *= 2)
            for (uint32_t i = 0; i < kv_q4_rotation; i += 2 * h)
                for (uint32_t j = 0; j < h; ++j) {
                    const float x = dst[base + i + j], y = dst[base + i + j + h];
                    dst[base + i + j] = x + y;
                    dst[base + i + j + h] = x - y;
                }
}

inline void kv_q4_rotate_rows(std::span<const float> rows, std::span<float> rotated) {
    kv_fwht256(rows, rotated);   // the rotation is its own inverse
}

inline void kv_q4_unrotate_rows(std::span<const float> rotated, std::span<float> rows) {
    kv_fwht256(rotated, rows);
}

// One 32-value group -> one 18-byte block. Returns the stored scale bits.
inline uint16_t kv_q4_quantize_block(std::span<const float> values, std::span<uint8_t> bytes) {
    if (values.size() != kv_q4_block || bytes.size() != sizeof(strata::kernels::block_q4_0))
        throw std::invalid_argument("Q4_0 KV block shape mismatch");
    float amax = 0, mval = 0;
    bool first = true;
    for (const float x : values) {
        if (!std::isfinite(x)) throw std::invalid_argument("Q4_0 KV requires finite values");
        const float a = std::fabs(x);
        // q4_group's reduction (:79-82): |x| wins, but an exact tie picks the larger signed value so every lane
        // ends with the same d.
        if (first || a > amax || (a == amax && x > mval)) {
            amax = a;
            mval = x;
            first = false;
        }
    }
    const float d = mval / -8.0f;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;   // the UNROUNDED scale, as __fdividef's input
    const uint16_t bits = strata::kernels::f16_from_f32(d);
    if (!std::isfinite(strata::kernels::f32_from_f16(bits))) throw std::invalid_argument("Q4_0 KV scale overflows FP16");
    bytes[0] = uint8_t(bits & 0xFF);
    bytes[1] = uint8_t(bits >> 8);
    uint8_t q[kv_q4_block];
    for (uint32_t i = 0; i < kv_q4_block; ++i) {
        // __float2int_rz (:85): truncation toward zero, then clamp. Out-of-int-range results (a denormal d makes
        // 1/d overflow) are INT_MIN on CUDA and therefore clamp to 0; the guard reproduces that instead of
        // relying on undefined float->int conversion.
        const float scaled = values[i] * id + 8.5f;
        int code = 0;
        if (std::isfinite(scaled) && scaled > float(std::numeric_limits<int>::min()) &&
            scaled < float(std::numeric_limits<int>::max()))
            code = int(std::trunc(scaled));
        q[i] = uint8_t(code < 0 ? 0 : (code > 15 ? 15 : code));
    }
    for (uint32_t j = 0; j < kv_q4_block / 2; ++j) bytes[2 + j] = uint8_t(q[j] | (q[j + kv_q4_block / 2] << 4));
    return bits;
}

// One stored element as the FP16 bits the attention path sees: fp16((q - 8) * fp16(d)), kv_gather_q4_kernel:165-173.
inline uint16_t kv_q4_bits(std::span<const uint8_t> block, const uint32_t element) {
    if (block.size() != sizeof(strata::kernels::block_q4_0) || element >= kv_q4_block)
        throw std::invalid_argument("Q4_0 KV block/element shape mismatch");
    const uint16_t bits = uint16_t(block[0] | (uint16_t(block[1]) << 8));
    const float d = strata::kernels::f32_from_f16(bits);
    const uint8_t byte = block[2 + (element & 15)];
    const int q = element < 16 ? (byte & 0x0F) - 8 : (byte >> 4) - 8;
    return strata::kernels::f16_from_f32(float(q) * d);
}

inline float kv_q4_value(std::span<const uint8_t> block, const uint32_t element) {
    return strata::kernels::f32_from_f16(kv_q4_bits(block, element));
}

// One rotated 256-value row -> 144 stored bytes.
inline void kv_q4_quantize_row(std::span<const float> row, std::span<uint8_t> bytes) {
    if (row.size() != kv_q4_rotation || bytes.size() != kv_q4_bytes_per_head(kv_q4_rotation))
        throw std::invalid_argument("Q4_0 KV row shape mismatch");
    for (uint32_t b = 0; b < kv_q4_blocks_per_head(kv_q4_rotation); ++b)
        (void)kv_q4_quantize_block(row.subspan(uint64_t(b) * kv_q4_block, kv_q4_block),
                                   bytes.subspan(uint64_t(b) * sizeof(strata::kernels::block_q4_0),
                                                 sizeof(strata::kernels::block_q4_0)));
}

inline void kv_q4_dequantize_row(std::span<const uint8_t> bytes, std::span<uint16_t> bits) {
    if (bits.size() != kv_q4_rotation || bytes.size() != kv_q4_bytes_per_head(kv_q4_rotation))
        throw std::invalid_argument("Q4_0 KV row shape mismatch");
    for (uint32_t b = 0; b < kv_q4_blocks_per_head(kv_q4_rotation); ++b) {
        const auto block = bytes.subspan(uint64_t(b) * sizeof(strata::kernels::block_q4_0),
                                         sizeof(strata::kernels::block_q4_0));
        for (uint32_t t = 0; t < kv_q4_block; ++t) bits[uint64_t(b) * kv_q4_block + t] = kv_q4_bits(block, t);
    }
}

// One stored row -> the unrotated FP32 row attention consumes. `out` is also the transform's scratch, so no
// allocation happens per row.
inline void kv_q4_dequantize_unrotated(std::span<const uint8_t> bytes, std::span<float> out) {
    if (out.size() != kv_q4_rotation || bytes.size() != kv_q4_bytes_per_head(kv_q4_rotation))
        throw std::invalid_argument("Q4_0 KV row shape mismatch");
    for (uint32_t b = 0; b < kv_q4_blocks_per_head(kv_q4_rotation); ++b) {
        const auto block = bytes.subspan(uint64_t(b) * sizeof(strata::kernels::block_q4_0),
                                         sizeof(strata::kernels::block_q4_0));
        const float d = strata::kernels::f32_from_f16(uint16_t(block[0] | (uint16_t(block[1]) << 8)));
        for (uint32_t t = 0; t < kv_q4_block; ++t) {
            const uint8_t byte = block[2 + (t & 15)];
            const int q = t < 16 ? (byte & 0x0F) - 8 : (byte >> 4) - 8;
            out[uint64_t(b) * kv_q4_block + t] =
                strata::kernels::f32_from_f16(strata::kernels::f16_from_f32(float(q) * d));
        }
    }
    kv_q4_unrotate_rows(out, out);
}
} // namespace strata::hpu
