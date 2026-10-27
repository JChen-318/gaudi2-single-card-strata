#pragma once
// GGUF IQ4_NL (20) / IQ4_XS (23) row decoding, shared by the host-side resident packers. Host only: this is the
// exact inverse of the GGUF dequantizer, so kv[code] * scale reproduces llama.cpp's value bit for bit.
#include "strata/kernels/f16_bits.hpp"
#include <cstdint>

namespace strata::hpu::detail {
inline bool iq4_type(uint32_t type) { return type == 20 || type == 23; }
inline float iq4_half(const uint8_t* p) { return strata::kernels::f32_from_f16(uint16_t(p[0]) | uint16_t(p[1] << 8)); }
// One row as codebook indices [0,16) and one F32 scale per 32 columns. codes and scales must hold columns entries.
inline void decode_iq4_row(const uint8_t* row, uint32_t type, uint64_t columns, uint8_t* codes, float* scales) {
    if (type == 20) {
        for (uint64_t b = 0; b < columns / 32; ++b) {
            const uint8_t* p = row + 18 * b;
            scales[b] = iq4_half(p);
            for (int j = 0; j < 16; ++j) {
                codes[32 * b + j] = p[2 + j] & 15;
                codes[32 * b + 16 + j] = p[2 + j] >> 4;
            }
        }
        return;
    }
    for (uint64_t s = 0; s < columns / 256; ++s) {
        const uint8_t* p = row + 136 * s;
        const float d = iq4_half(p);
        const unsigned high = p[2] | (p[3] << 8);
        for (int ib = 0; ib < 8; ++ib) {
            const int ls = ((p[4 + ib / 2] >> 4 * (ib % 2)) & 15) | (((high >> 2 * ib) & 3) << 4);
            scales[8 * s + ib] = d * float(ls - 32);
            for (int j = 0; j < 16; ++j) {
                codes[256 * s + 32 * ib + j] = p[8 + 16 * ib + j] & 15;
                codes[256 * s + 32 * ib + 16 + j] = p[8 + 16 * ib + j] >> 4;
            }
        }
    }
}
}
