// hpu/tests/kv_cell.cpp - independent evidence for upstream's INT8, Q4_0 and hybrid K8V4 KV cells (--kv int8 /
// q4_0 / k8v4).
//
// INT8 (include/strata/kernels/kv_q8.hpp:5-12, src/kernels/cuda/kv_q8.cu:47-68, :95-104):
//     scale = fp16(max|x| / 127),  code = clamp(rint(x / scale), -127, 127),  x' = fp16(code * scale)
// quantized against the STORED, already rounded scale, with rint = nearest, ties to even (__float2int_rn).
// Q4_0 (include/strata/kernels/kv_q4.hpp:1-12, src/kernels/cuda/kv_q4.cu:25-69, :75-91, :165-173):
//     rotate every 256-value row by the 1/16-scaled Fast Walsh-Hadamard matrix (self-inverse),
//     per 32 values: amax |x| (larger signed value breaks ties), d = mval/-8,
//     q = clamp(trunc(x * (1/d) + 8.5), 0, 15) with the UNROUNDED 1/d, x' = fp16((q-8) * fp16(d)),
//     element j in the low nibble of qs[j], element j+16 in the high one; 18 bytes per block.
// Nothing here calls the quantizers under test: both oracles are written from those sentences, so a mutation of
// the headers (rounding away, unrounded/stored scale swap, missing clamp, wrong stride, nibble order, FWHT
// scale, tie-break) has to disagree with this file.
#include "strata/hpu/kv_cell.hpp"
#include "strata/kernels/f16_bits.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <span>
#include <string>
#include <vector>

namespace {
int failures = 0;
void check(const bool ok, const std::string& what) {
    if (!ok) { ++failures; std::cout << "FAIL " << what << "\n"; }
}
template <class F> bool throws(F&& f) {
    try { f(); } catch (const std::exception&) { return true; }
    return false;
}

// Independent oracle: the sentence above, no shared helper except the FP16 codec the harness already uses.
struct Oracle {
    std::vector<int8_t> codes;
    std::vector<uint16_t> scales;
};
Oracle oracle(std::span<const float> values) {
    Oracle out;
    out.codes.resize(values.size());
    out.scales.resize(values.size() / 64);
    for (size_t group = 0; group < out.scales.size(); ++group) {
        float amax = 0;
        for (size_t t = 0; t < 64; ++t) amax = std::max(amax, std::fabs(values[group * 64 + t]));
        out.scales[group] = strata::kernels::f16_from_f32(amax / 127.0f);
        const double scale = double(strata::kernels::f32_from_f16(out.scales[group]));
        for (size_t t = 0; t < 64; ++t) {
            int code = 0;
            if (scale > 0) {
                const double ratio = double(values[group * 64 + t]) / scale;
                code = int(std::rint(ratio));   // nearest, ties to even (the default rounding mode)
                code = std::clamp(code, -127, 127);
            }
            out.codes[group * 64 + t] = int8_t(code);
        }
    }
    return out;
}

void shapes_and_bytes() {
    using strata::hpu::kv_int8_cell_bytes;
    using strata::hpu::kv_int8_groups;
    check(kv_int8_groups(64) == 1 && kv_int8_groups(256) == 4, "group count is width/64");
    for (const uint32_t bad : {0u, 1u, 63u, 65u, 100u}) check(throws([&] { (void)kv_int8_groups(bad); }), "non-multiple-of-64 width refused");
    // 2 heads x (64 K + 64 V bytes + one 2-byte K scale + one 2-byte V scale) = 264.
    check(kv_int8_cell_bytes(64, 2) == 264, "cell bytes width 64/2 heads");
    check(kv_int8_cell_bytes(256, 2) == 1056, "cell bytes width 256/2 heads");
    check(kv_int8_cell_bytes(128, 1) == 264, "cell bytes width 128/1 head");
    check(throws([] { (void)kv_int8_cell_bytes(0, 1); }), "zero-width cell refused");
    check(throws([] { (void)kv_int8_cell_bytes(65, 1); }), "65-wide cell refused");
    // The price --kv charges the plan and the parked share, from the same law: FP16 is 4 bytes/value, INT8 is
    // the cell above. A plan that undercounts --kv int8 would break these, not a copy of the arithmetic.
    using strata::hpu::kv_token_bytes;
    check(kv_token_bytes(64, 2, strata::hpu::KvMode::Fp16) == 512, "one token's K+V bytes follow the selected format (fp16)");
    check(kv_token_bytes(64, 2, strata::hpu::KvMode::Int8) == 264, "one token's K+V bytes follow the selected format (int8)");
    check(kv_token_bytes(128, 1, strata::hpu::KvMode::Fp16) == 512, "fp16 width 128/1 head");
    check(kv_token_bytes(128, 1, strata::hpu::KvMode::Int8) == 264, "int8 width 128/1 head");
    check(throws([] { (void)kv_token_bytes(65, 1, strata::hpu::KvMode::Int8); }), "int8 token price refuses a partial group");
}

void rounding() {
    using strata::hpu::kv_int8_round_nearest_even;
    // Ties go to the even neighbour, like CUDA's __float2int_rn. std::round would give 1, 3, 3 ...
    const int ties[][2] = {{0, 0}, {1, 2}, {2, 2}, {3, 4}, {-1, 0}, {-2, -2}, {-3, -2}, {-4, -4}, {126, 126}, {127, 128}};
    for (const auto& item : ties) {
        const float input = item[0] + 0.5f;
        check(kv_int8_round_nearest_even(input) == item[1], "tie-to-even at " + std::to_string(input));
    }
    check(kv_int8_round_nearest_even(0.49f) == 0 && kv_int8_round_nearest_even(0.51f) == 1, "non-ties round nearest");
    check(kv_int8_round_nearest_even(-0.51f) == -1 && kv_int8_round_nearest_even(-0.49f) == 0, "negative non-ties round nearest");
}

void quantization() {
    // Group 0: amax = 1.0, whose stored scale is a little below 1/127. The values 1.5s, 2.5s, -1.5s, -2.5s
    // (s = the STORED scale) sit exactly on tie points of the stored scale: the stored-scale law gives 2, 2, -2,
    // -2, while dividing by the unrounded amax/127 would fall just short and give 1, 2, -1, -2. That is how this
    // test sees a lost "quantize against the stored scale".
    const float stored = strata::kernels::f32_from_f16(strata::kernels::f16_from_f32(1.0f / 127.0f));
    std::vector<float> values(128);
    values[0] = 1.0f; values[1] = -1.0f;
    values[2] = 1.5f * stored; values[3] = -1.5f * stored;
    values[4] = 2.5f * stored; values[5] = -2.5f * stored;
    values[6] = 0.5f * stored; values[7] = -0.5f * stored;
    for (size_t i = 8; i < 64; ++i) values[i] = (float(int(i) - 36) / 41.0f);
    // Group 1: a different magnitude, so a wrong group stride/scale index cannot pass.
    for (size_t i = 64; i < 128; ++i) values[i] = (float(int(i) - 96) / 7.0f) * 30.0f;

    const auto expected = oracle(values);
    std::vector<int8_t> codes(values.size());
    std::vector<uint16_t> scales(2);
    strata::hpu::kv_int8_quantize(values, codes, scales);
    check(scales == expected.scales, "stored scale is fp16(amax/127)");
    check(codes == expected.codes, "codes come from the stored scale, nearest-even");
    check(codes[2] == 2 && codes[3] == -2 && codes[4] == 2 && codes[5] == -2, "stored-scale tie points (unrounded scale gives 1/-1)");
    check(codes[0] == 127 && codes[1] == -127, "amax maps to the 127 boundary");
    // int8_t cannot exceed 127, so the clamp's upper half is checked through the codes above (127/-127).
    for (const int8_t code : codes) check(int(code) >= -127, "code not below -127");

    std::vector<uint16_t> bits(values.size());
    strata::hpu::kv_int8_dequantize(codes, scales, bits);
    for (size_t i = 0; i < values.size(); ++i) {
        check(bits[i] == strata::kernels::f16_from_f32(float(codes[i]) * strata::kernels::f32_from_f16(scales[i / 64])),
              "dequantized bit is fp16(code*scale)");
        check(bits[i] == strata::kernels::f16_from_f32(strata::hpu::kv_int8_value(codes[i], scales[i / 64])),
              "value reader and bit reader agree");
    }
    // The error of one stored value is at most half a code step plus the final FP16 rounding.
    for (size_t group = 0; group < scales.size(); ++group) {
        const double step = double(strata::kernels::f32_from_f16(scales[group]));
        for (size_t t = 0; t < 64; ++t) {
            const double decoded = double(strata::kernels::f32_from_f16(bits[group * 64 + t]));
            check(std::abs(decoded - double(values[group * 64 + t])) <= step * 0.5 + step * 0.001,
                  "round-trip error bounded by half a step");
        }
    }
    // The two groups really carry different scales: value 70 decodes with group 1's scale.
    check(scales[0] != scales[1], "per-group scales differ");
    check(bits[70] == strata::hpu::kv_int8_bits(codes[70], scales[1]) &&
          bits[70] != strata::hpu::kv_int8_bits(codes[70], scales[0]),
          "second group uses its own scale");
}

void edge_groups() {
    // An all-zero group stores scale 0 and code 0, and decodes to +0.
    std::vector<float> zeros(64, 0.0f);
    std::vector<int8_t> codes(64);
    std::vector<uint16_t> scales(1);
    strata::hpu::kv_int8_quantize(zeros, codes, scales);
    check(scales[0] == 0, "zero group stores a zero scale");
    for (const int8_t code : codes) check(code == 0, "zero group stores zero codes");
    std::vector<uint16_t> bits(64);
    strata::hpu::kv_int8_dequantize(codes, scales, bits);
    for (const uint16_t bit : bits) check(bit == 0, "zero group decodes to +0");

    // A group whose amax/127 underflows FP16 keeps upstream's behaviour: scale 0, codes 0, zeros out.
    std::vector<float> tiny(64, 1e-8f);
    strata::hpu::kv_int8_quantize(tiny, codes, scales);
    check(scales[0] == 0 && std::all_of(codes.begin(), codes.end(), [](int8_t c) { return c == 0; }),
          "FP16-underflowing scale zeroes the group");

    // 65536 is finite but not FP16-representable. The FP16 pool refuses it; the INT8 cell stores it through a
    // finite FP16 scale and code 127, and the reader's fp16(127*scale) saturates to +inf - the same bit the
    // upstream __float2half_rn(code*scale) writes. Only the cell (codes + a finite scale) is guaranteed finite.
    std::vector<float> large(64, 1.0f);
    large[5] = 65536.0f;
    strata::hpu::kv_int8_quantize(large, codes, scales);
    check(codes[5] == 127, "65536 quantizes to code 127");
    check(std::isfinite(strata::kernels::f32_from_f16(scales[0])), "65536's stored scale is finite");
    check(strata::hpu::kv_int8_bits(codes[5], scales[0]) == 0x7C00u, "127*scale beyond FP16 saturates to +inf, as upstream");

    // Non-finite input is refused, and a scale that would overflow FP16 (amax/127 > 65504) is refused rather
    // than stored as inf and decoded as NaN.
    for (const float bad : {NAN, INFINITY, -INFINITY}) {
        std::vector<float> row(64, 1.0f);
        row[17] = bad;
        check(throws([&] { strata::hpu::kv_int8_quantize(row, codes, scales); }), "non-finite value refused");
    }
    std::vector<float> huge(64, 1.0f);
    huge[0] = 1e30f;
    check(throws([&] { strata::hpu::kv_int8_quantize(huge, codes, scales); }), "FP16-infinite scale refused");
}

void refusals() {
    std::vector<float> values(64, 1.0f);
    std::vector<int8_t> codes(64);
    std::vector<uint16_t> scales(1);
    auto refused = [&](const size_t value_count, const size_t code_count, const size_t scale_count) {
        return throws([&] {
            strata::hpu::kv_int8_quantize(std::span<const float>(values).first(value_count),
                                          std::span<int8_t>(codes).first(code_count),
                                          std::span<uint16_t>(scales).first(scale_count));
        });
    };
    check(refused(0, 0, 0), "empty quantize refused");
    check(refused(63, 63, 1), "63 values refused");
    check(refused(64, 63, 1), "code count mismatch refused");
    check(refused(64, 64, 0), "scale count mismatch refused");
    check(refused(128, 128, 1), "missing second group's scale refused");
    std::vector<uint16_t> bits(64);
    check(throws([&] { strata::hpu::kv_int8_dequantize(codes, scales, std::span<uint16_t>(bits).first(63)); }), "dequantize bit count mismatch refused");
    check(throws([&] { strata::hpu::kv_int8_dequantize(std::span<const int8_t>(codes).first(63), scales, bits); }), "dequantize 63 codes refused");
    check(throws([&] { strata::hpu::kv_int8_dequantize(codes, std::span<const uint16_t>(scales).first(0), bits); }), "dequantize scale count mismatch refused");
}

// --- Q4_0 KV cell -----------------------------------------------------------------------------

constexpr uint32_t q4_row = 256;

// Independent FWHT: scale every value by 1/16 first, then the standard radix-2 butterflies (h = 1..128).
void hadamard_oracle(std::span<const float> src, std::span<float> dst) {
    for (uint32_t i = 0; i < q4_row; ++i) dst[i] = src[i] * (1.0f / 16.0f);
    for (uint32_t h = 1; h < q4_row; h *= 2)
        for (uint32_t i = 0; i < q4_row; i += 2 * h)
            for (uint32_t j = 0; j < h; ++j) {
                const float x = dst[i + j], y = dst[i + j + h];
                dst[i + j] = x + y;
                dst[i + j + h] = x - y;
            }
}

struct Q4BlockOracle {
    uint16_t scale = 0;
    std::array<uint8_t, 16> qs{};
    std::array<int, 32> q{};
};

int q4_code_oracle(const float value, const float id) {
    const float scaled = value * id + 8.5f;
    int code = 0;
    if (std::isfinite(scaled) && scaled > float(std::numeric_limits<int>::min()) &&
        scaled < float(std::numeric_limits<int>::max()))
        code = int(std::trunc(scaled));
    return code < 0 ? 0 : (code > 15 ? 15 : code);
}

Q4BlockOracle q4_block_oracle(std::span<const float> values) {
    Q4BlockOracle out;
    float amax = 0, mval = 0;
    bool first = true;
    for (const float x : values) {
        const float a = std::fabs(x);
        if (first || a > amax || (a == amax && x > mval)) { amax = a; mval = x; first = false; }
    }
    const float d = mval / -8.0f;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
    out.scale = strata::kernels::f16_from_f32(d);
    for (uint32_t i = 0; i < 32; ++i) out.q[i] = q4_code_oracle(values[i], id);
    for (uint32_t j = 0; j < 16; ++j) out.qs[j] = uint8_t(out.q[j] | (out.q[j + 16] << 4));
    return out;
}

void q4_shapes_and_bytes() {
    using strata::hpu::KvMode;
    using strata::hpu::kv_q4_blocks_per_head;
    using strata::hpu::kv_q4_bytes_per_head;
    using strata::hpu::kv_q4_cell_bytes;
    using strata::hpu::kv_token_bytes;
    check(kv_q4_blocks_per_head(256) == 8, "Q4 head is eight 32-value blocks");
    check(kv_q4_bytes_per_head(256) == 144, "Q4 head is 144 bytes");
    check(kv_q4_cell_bytes(256, 2) == 576, "Q4 cell bytes width 256/2 heads");
    check(kv_q4_cell_bytes(256, 0) == 0, "zero-head Q4 cell is empty");
    for (const uint32_t bad : {0u, 64u, 128u, 255u, 512u})
        check(throws([&] { (void)kv_q4_bytes_per_head(bad); }), "non-256 Q4 head width refused");
    check(kv_token_bytes(256, 2, KvMode::Q4) == 576, "one token's K+V bytes follow the selected format (q4_0)");
    check(kv_token_bytes(256, 2, KvMode::Fp16) == 2048, "fp16 token price width 256/2 heads");
    check(kv_token_bytes(256, 2, KvMode::Int8) == 1056, "int8 token price width 256/2 heads");
}

void hybrid_shapes_and_drafter() {
    using strata::hpu::KvMode;
    using strata::hpu::kv_drafter_mode;
    using strata::hpu::kv_hybrid_cell_bytes;
    using strata::hpu::kv_token_bytes;
    // Upstream's hybrid price (src/core/layer.cpp:649-656): 256 K code bytes + 4*2 K scale bytes + 144 V q4_0
    // bytes per head = 408, 816 per 2-head cell. It must not coincide with any single-format price.
    check(kv_hybrid_cell_bytes(256, 2) == 816, "K8V4 cell bytes width 256/2 heads");
    check(kv_hybrid_cell_bytes(256, 1) == 408, "K8V4 cell bytes width 256/1 head");
    check(kv_hybrid_cell_bytes(256, 0) == 0, "zero-head K8V4 cell is empty");
    check(kv_hybrid_cell_bytes(256, 2) == 2 * (256 + 4 * 2 + 144),
          "K8V4 is exactly the INT8 K half (codes + 4 scales) plus the Q4_0 V half per head");
    check(kv_token_bytes(256, 2, KvMode::K8V4) == 816, "one token's K+V bytes follow the selected format (k8v4)");
    for (const uint32_t bad : {0u, 64u, 128u, 255u, 512u})
        check(throws([&] { (void)kv_hybrid_cell_bytes(bad, 2); }), "non-256 K8V4 head width refused");
    // The drafter follows the trunk except under k8v4, where upstream's ring state is plain INT8.
    check(kv_token_bytes(256, 2, kv_drafter_mode(KvMode::K8V4)) == 1056, "k8v4 drafter is priced as plain INT8");
}

void q4_rotation() {
    std::vector<float> row(q4_row), other(q4_row);
    for (uint32_t i = 0; i < q4_row; ++i) {
        row[i] = (i % 17 == 0) ? 12000.0f : (i % 13 == 0 ? -7000.0f : float(int(i % 31) - 15) * 0.37f) + float(i) * 1e-3f;
        other[i] = float(int(i % 23) - 11) * 0.71f - float(i) * 5e-4f;
    }
    std::vector<float> rotated(q4_row), expected(q4_row);
    strata::hpu::kv_q4_rotate_rows(row, rotated);
    hadamard_oracle(row, expected);
    check(rotated == expected, "rotation matches the 1/16-first FWHT in upstream's order");

    std::vector<float> rotated_other(q4_row);
    strata::hpu::kv_q4_rotate_rows(other, rotated_other);
    double norm = 0, rotated_norm = 0, inner = 0, rotated_inner = 0;
    for (uint32_t i = 0; i < q4_row; ++i) {
        norm += double(row[i]) * row[i];
        rotated_norm += double(rotated[i]) * rotated[i];
        inner += double(row[i]) * other[i];
        rotated_inner += double(rotated[i]) * rotated_other[i];
    }
    check(std::abs(rotated_norm - norm) <= norm * 1e-4, "FWHT preserves the row norm");
    check(std::abs(rotated_inner - inner) <= (std::abs(inner) + norm) * 1e-4, "FWHT preserves inner products");

    std::vector<float> back(q4_row);
    strata::hpu::kv_q4_unrotate_rows(rotated, back);
    double max_value = 0, max_error = 0;
    for (uint32_t i = 0; i < q4_row; ++i) {
        max_value = std::max(max_value, double(std::fabs(row[i])));
        max_error = std::max(max_error, double(std::fabs(back[i] - row[i])));
    }
    check(max_error <= max_value * 1e-4, "FWHT is its own inverse");
}

void q4_block_law() {
    // values[0]/values[1] have the same |x|: the larger signed value must win as mval, giving d < 0.
    // 1 + 2^-11 is not FP16-representable; its FP16 round is 1.0, so d = -0.125 exactly.
    // values[2] sits where the UNROUNDED 1/d gives code 5 and the stored 1/d would give 4.
    std::vector<float> values(32);
    values[0] = -1.00048828125f;
    values[1] = 1.00048828125f;
    values[2] = 0.4376f;
    for (uint32_t i = 3; i < 32; ++i) values[i] = float(int(i) - 18) * 0.05f;   // stays below 1.000488
    values[7] = -0.9f;   // saturates to 15
    values[8] = 0.9f;    // ... and 0 on the other side

    std::array<uint8_t, sizeof(strata::kernels::block_q4_0)> bytes{};
    const uint16_t bits = strata::hpu::kv_q4_quantize_block(values, bytes);
    const Q4BlockOracle expected = q4_block_oracle(values);
    check(bits == expected.scale, "stored block scale is fp16(mval/-8)");
    check((bytes[2] & 0x0F) == 15 && (bytes[3] & 0x0F) == 0 && expected.q[0] == 15 && expected.q[1] == 0,
          "amax tie picks the larger signed value");
    const float stored_d = strata::kernels::f32_from_f16(strata::kernels::f16_from_f32(1.00048828125f / -8.0f));
    check(expected.q[2] == 5 && (bytes[4] & 0x0F) == 5 && q4_code_oracle(0.4376f, 1.0f / stored_d) == 4,
          "codes quantize against the unrounded 1/d");
    for (uint32_t j = 0; j < 16; ++j) {
        check(bytes[2 + j] == expected.qs[j], "stored byte is q[j] | q[j+16] << 4");
        check((bytes[2 + j] & 0x0F) == uint8_t(expected.q[j]) &&
              (bytes[2 + j] >> 4) == uint8_t(expected.q[j + 16]), "nibble layout (low j, high j+16)");
    }
    check(bytes[2] == expected.qs[0] && expected.qs[0] != uint8_t(expected.q[0] | (expected.q[0] << 4)),
          "fixture separates the two nibble halves");
    for (uint32_t t = 0; t < 32; ++t) {
        const uint16_t want =
            strata::kernels::f16_from_f32(float(expected.q[t] - 8) * strata::kernels::f32_from_f16(expected.scale));
        check(strata::hpu::kv_q4_bits(bytes, t) == want, "stored element decodes to fp16((q-8)*stored d)");
        check(strata::hpu::kv_q4_value(bytes, t) == strata::kernels::f32_from_f16(want),
              "value reader and bit reader agree (q4_0)");
    }
    check(strata::hpu::kv_q4_value(bytes, 1) == 1.0f,
          "positive amax decodes through q-8 = -8 and the stored d = -0.125");

    // An all-zero block stores the oracle's scale (negative zero is still zero) and zero-valued codes.
    std::vector<float> zeros(32, 0.0f);
    std::array<uint8_t, sizeof(strata::kernels::block_q4_0)> zero_bytes{};
    const uint16_t zero_bits = strata::hpu::kv_q4_quantize_block(zeros, zero_bytes);
    const Q4BlockOracle zero_expected = q4_block_oracle(zeros);
    check(zero_bits == zero_expected.scale, "zero block stores fp16(mval/-8)");
    for (uint32_t t = 0; t < 32; ++t)
        check(strata::hpu::kv_q4_value(zero_bytes, t) == 0.0f, "zero block decodes to zero");
}

void q4_row_law() {
    std::vector<float> row(q4_row);
    for (uint32_t i = 0; i < q4_row; ++i) {
        const float scale = 0.25f + 0.5f * float(i / 32);
        row[i] = scale * float(int(i % 37) - 18) / 19.0f + (i % 3 ? 0.0f : 1e-4f * float(i));
    }
    std::vector<float> rotated(q4_row);
    strata::hpu::kv_q4_rotate_rows(row, rotated);
    std::vector<float> expected_rotated(q4_row);
    hadamard_oracle(row, expected_rotated);

    std::array<Q4BlockOracle, 8> blocks{};
    std::array<uint8_t, 144> expected{};
    for (uint32_t b = 0; b < 8; ++b) {
        blocks[b] = q4_block_oracle(std::span<const float>(expected_rotated).subspan(b * 32, 32));
        expected[b * 18] = uint8_t(blocks[b].scale & 0xFF);
        expected[b * 18 + 1] = uint8_t(blocks[b].scale >> 8);
        std::copy(blocks[b].qs.begin(), blocks[b].qs.end(), expected.begin() + (b * 18 + 2));
    }
    std::array<uint8_t, 144> bytes{};
    strata::hpu::kv_q4_quantize_row(rotated, bytes);
    check(bytes == expected, "row bytes are eight independent q4_0 blocks");

    std::array<uint16_t, q4_row> bits{};
    strata::hpu::kv_q4_dequantize_row(expected, bits);
    std::vector<float> dequantized(q4_row);
    for (uint32_t b = 0; b < 8; ++b)
        for (uint32_t t = 0; t < 32; ++t) {
            const uint16_t want = strata::kernels::f16_from_f32(
                float(blocks[b].q[t] - 8) * strata::kernels::f32_from_f16(blocks[b].scale));
            check(bits[b * 32 + t] == want, "row dequantize reads each block's own scale");
            dequantized[b * 32 + t] = strata::kernels::f32_from_f16(want);
        }

    std::vector<float> unrotated(q4_row), expected_unrotated(q4_row);
    strata::hpu::kv_q4_dequantize_unrotated(expected, unrotated);
    hadamard_oracle(dequantized, expected_unrotated);
    check(unrotated == expected_unrotated, "unrotated decode is the FWHT of the stored blocks");
}

void q4_refusals() {
    std::vector<float> values(32, 1.0f);
    std::array<uint8_t, sizeof(strata::kernels::block_q4_0)> bytes{};
    auto block = [&](const size_t vc, const size_t bc) {
        return throws([&] {
            strata::hpu::kv_q4_quantize_block(std::span<const float>(values).first(vc),
                                              std::span<uint8_t>(bytes).first(bc));
        });
    };
    check(block(0, 18), "empty Q4 block refused");
    check(block(31, 18), "31-value Q4 block refused");
    check(block(32, 17), "short Q4 block bytes refused");
    values[3] = NAN;
    check(block(32, 18), "non-finite Q4 block value refused");
    values[3] = 1.0f;
    values[0] = 1e30f;
    check(block(32, 18), "FP16-infinite Q4 scale refused");

    std::vector<float> row(128, 1.0f), out(128);
    std::array<uint8_t, 144> row_bytes{};
    check(throws([&] { strata::hpu::kv_q4_quantize_row(row, row_bytes); }), "128-value Q4 row refused");
    std::vector<float> full_row(q4_row, 1.0f);
    check(throws([&] { strata::hpu::kv_q4_quantize_row(full_row, std::span<uint8_t>(row_bytes).first(143)); }),
          "short Q4 row bytes refused");
    std::vector<uint16_t> bits(q4_row);
    check(throws([&] { strata::hpu::kv_q4_dequantize_row(row_bytes, std::span<uint16_t>(bits).first(255)); }),
          "Q4 dequantize row shape refused");
    check(throws([&] { strata::hpu::kv_q4_dequantize_unrotated(row_bytes, out); }),
          "Q4 unrotated decode needs a 256-value output");
    check(throws([&] { strata::hpu::kv_fwht256(row, out); }), "partial FWHT refused");
    std::vector<float> bad(q4_row, 1.0f), scratch(q4_row);
    bad[9] = INFINITY;
    check(throws([&] { strata::hpu::kv_fwht256(bad, scratch); }), "non-finite FWHT input refused");
}
} // namespace

int main() {
    shapes_and_bytes();
    rounding();
    quantization();
    edge_groups();
    refusals();
    q4_shapes_and_bytes();
    hybrid_shapes_and_drafter();
    q4_rotation();
    q4_block_law();
    q4_row_law();
    q4_refusals();
    if (failures) {
        std::cout << "FAILED kv-cell checks: " << failures << "\n";
        return 1;
    }
    std::cout << "PASS INT8 KV cell: fp16(amax/127) stored scale, nearest-even codes against that stored scale, "
                 "clamp to [-127,127], group-major decode, zero/underflow groups, 65536 stored through FP16, "
                 "non-finite and FP16-infinite refusal, exact cell byte counts; no device\n";
    std::cout << "PASS Q4_0 KV cell: 1/16-first self-inverse FWHT, fp16(mval/-8) block scale with the larger signed "
                 "value breaking amax ties, trunc(x*(1/d)+8.5) against the UNROUNDED 1/d, clamp 0..15, low/high "
                 "nibble order, per-block decode, rotated rows back to unrotated rows, exact 144/576-byte cell, "
                 "non-finite and FP16-infinite refusal; no device\n";
    std::cout << "PASS K8V4 KV cell: INT8 K half plus rotated Q4_0 V half, 408 bytes per 256-wide head and 816 per "
                 "2-head token, exactly the two halves' sum, non-256 widths refused, drafter priced as plain INT8; "
                 "no device\n";
    return 0;
}
