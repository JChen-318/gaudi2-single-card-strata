#pragma once
// hpu/include/strata/hpu/kv_cell.hpp - upstream's INT8 and Q4_0 KV cells as host-checkable laws.
//
// --kv int8 changes only KV storage (include/strata/kernels/kv_q8.hpp:10-12): K and V are stored as int8 codes
// with one FP16 scale per 64 values, and every reader dequantizes into the same FP16 scratch the attention path
// already consumes. The law is the CUDA append/gather pair:
//     scale = fp16(max|x| / 127),  code = clamp(rint(x / scale), -127, 127),  x' = fp16(code * scale)
// with the quantization performed against the STORED (already rounded) scale (src/kernels/cuda/kv_q8.cu:47-68,
// :95-104). Layout mirrors the FP16 pools: codes [head][group][64], scales [head][group].
// --kv q4_0 stores Hadamard-rotated 18-byte q4_0 blocks instead (kv_q4_cell.hpp): kv_token_bytes prices both
// formats from one law so the trunk arena, the MTP drafter and the parked share cannot undercount either.
//
// This header carries no state: QsaState and DenseKvState call it, and hpu/tests/kv_cell.cpp pins the arithmetic
// independently. `rint` here is CUDA's __float2int_rn (nearest, ties to even), not std::round.
#include "strata/hpu/kv_mode.hpp"
#include "strata/hpu/kv_q4_cell.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/kv_q8.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <span>
#include <stdexcept>

namespace strata::hpu {
inline constexpr uint32_t kv_int8_group = strata::kernels::KV_Q8_GROUP;
static_assert(kv_int8_group == 64, "upstream's INT8 KV group is 64 values");

inline uint32_t kv_int8_groups(const uint32_t width) {
    if (!width || width % kv_int8_group)
        throw std::invalid_argument("INT8 KV requires a head width that is a multiple of 64");
    return width / kv_int8_group;
}

// One cell = one token x one layer: K and V codes plus both scale rows.
inline uint64_t kv_int8_cell_bytes(const uint32_t width, const uint32_t heads) {
    const uint64_t per_head = uint64_t(width) * 2 + uint64_t(kv_int8_groups(width)) * 4;
    if (heads && per_head > UINT64_MAX / heads) throw std::overflow_error("INT8 KV cell byte count overflow");
    return per_head * heads;
}

// K8V4 (--kv k8v4): INT8 K codes with their one FP16 scale per 64 values, plus rotated Q4_0 V blocks and no V
// scale rows (the block carries its own d). Upstream prices the same three runs (src/core/layer.cpp:649-656):
// 816 B per cell at width 256 / 2 heads. The Q4 half requires head_dim 256, and INT8 requires a multiple of 64.
inline uint64_t kv_hybrid_cell_bytes(const uint32_t width, const uint32_t heads) {
    const uint64_t groups = kv_int8_groups(width);
    const uint64_t per_head = uint64_t(width) + groups * 2 + kv_q4_bytes_per_head(width);
    if (heads && per_head > UINT64_MAX / heads) throw std::overflow_error("K8V4 KV cell byte count overflow");
    return per_head * heads;
}

// One token's K+V in the selected format: what --kv prices the trunk arena, the MTP drafter's state and the
// parked share with. FP16 is four bytes per value (K and V); INT8 is upstream's cell above; Q4_0 is upstream's
// rotated 18-byte blocks (kv_q4_cell.hpp); K8V4 is the INT8 K half plus the rotated Q4_0 V half.
inline uint64_t kv_token_bytes(const uint32_t width, const uint32_t heads, const KvMode mode) {
    if (mode == KvMode::Int8) return kv_int8_cell_bytes(width, heads);
    if (mode == KvMode::Q4) return kv_q4_cell_bytes(width, heads);
    if (mode == KvMode::K8V4) return kv_hybrid_cell_bytes(width, heads);
    const uint64_t values = uint64_t(width) * heads;
    if (heads && values / heads != width) throw std::overflow_error("FP16 KV cell byte count overflow");
    return values * 4;
}

// up to one group of values -> codes + one scale
inline int kv_int8_round_nearest_even(const float value) {
    const float lower = std::floor(value);
    const float fraction = value - lower;
    int rounded = int(lower);
    if (fraction > 0.5f || (fraction == 0.5f && (rounded & 1))) ++rounded;
    return rounded;
}

inline void kv_int8_quantize(std::span<const float> values, std::span<int8_t> codes, std::span<uint16_t> scales) {
    if (values.empty() || values.size() % kv_int8_group || codes.size() != values.size() ||
        scales.size() != values.size() / kv_int8_group)
        throw std::invalid_argument("INT8 KV quantize shape mismatch");
    for (size_t group = 0; group < scales.size(); ++group) {
        float amax = 0;
        for (size_t t = 0; t < kv_int8_group; ++t) {
            const float value = values[group * kv_int8_group + t];
            if (!std::isfinite(value)) throw std::invalid_argument("INT8 KV requires finite values");
            amax = std::max(amax, std::fabs(value));
        }
        scales[group] = strata::kernels::f16_from_f32(amax / 127.0f);
        const float scale = strata::kernels::f32_from_f16(scales[group]);   // quantize against the STORED scale
        if (!std::isfinite(scale)) throw std::invalid_argument("INT8 KV scale overflows FP16");
        for (size_t t = 0; t < kv_int8_group; ++t) {
            int q = 0;
            if (scale > 0.0f) {
                q = kv_int8_round_nearest_even(values[group * kv_int8_group + t] / scale);
                q = std::clamp(q, -127, 127);
            }
            codes[group * kv_int8_group + t] = int8_t(q);
        }
    }
}

// What the attention path sees: fp16(code * stored scale), as kv_gather_q8_kernel writes it (:100-103).
inline uint16_t kv_int8_bits(const int8_t code, const uint16_t scale) {
    return strata::kernels::f16_from_f32(float(code) * strata::kernels::f32_from_f16(scale));
}

inline float kv_int8_value(const int8_t code, const uint16_t scale) {
    return strata::kernels::f32_from_f16(kv_int8_bits(code, scale));
}

inline void kv_int8_dequantize(std::span<const int8_t> codes, std::span<const uint16_t> scales,
                               std::span<uint16_t> bits) {
    if (codes.empty() || codes.size() % kv_int8_group || bits.size() != codes.size() ||
        scales.size() != codes.size() / kv_int8_group)
        throw std::invalid_argument("INT8 KV dequantize shape mismatch");
    for (size_t i = 0; i < codes.size(); ++i) bits[i] = kv_int8_bits(codes[i], scales[i / kv_int8_group]);
}
} // namespace strata::hpu
