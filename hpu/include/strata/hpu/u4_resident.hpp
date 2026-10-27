#pragma once
// The resident U4 expert store: one layer's routed experts plus the shared expert in HBM as syn_type_uint4 codes,
// plus one F32 (scale, offset) pair per output channel and per K group:
//     w(e,n,k) = scale[e,g,n] * code(e,n,k) + offset[e,g,n],  g = k / group
// The graph explicitly dequantizes raw codes to BF16 before the MME (Gaudi2 rejects mixed BF16/U4 GEMM).
// An all-ones scale gives raw code values and a fused epilogue applies
// the affine above, because the MME's quantization metadata is fixed when the graph is compiled and cannot carry a
// different table per expert. Each expert's codes for one projection form a contiguous [groups][rows][group/2]
// block, which is the [batch=e*groups+g][N][K/2] row-major weight one batch GEMM's transpose_b layout reads; two
// codes share a byte along k, low nibble first (the device probe settles the order).
//
// Six regions, each one contiguous tensor over all blocks (routed experts then the shared expert):
//   gu codes   [blocks][gu_groups][gu_rows][gu_group/2]
//   gu scale   [blocks][gu_groups][gu_rows] F32      gu offset  same shape
//   down codes [blocks][down_groups][down_rows][down_group/2]
//   down scale [blocks][down_groups][down_rows] F32  down offset same shape
// The routed part of a codes region is exactly the batch weight of one grouped batch GEMM, and the shared expert
// is its [groups][rows][group/2] tail. The two are bound as separate, non-overlapping views of the same region
// (Synapse rejects overlapping persistent tensors), so one upload serves both. Every region start and every
// per-block slice is a multiple of 128 bytes, so each expert is also a legal aligned tensor view.
// The same store serves the per-slot U4 GEMV (decode and prefill-window overflow), so one resident format covers
// the MME path and the row path. Host arithmetic only: no device and no Synapse types, so it is testable offline.
#include <cstdint>
#include <numeric>
#include <span>
#include <stdexcept>
#include <vector>

namespace strata::hpu {
// One GGUF matrix to pack: an IQ4_NL (20) or IQ4_XS (23) row-major payload, exactly what WeightSlice exposes.
struct U4Source {
    std::span<const uint8_t> bytes;
    uint64_t row_bytes = 0, rows = 0, columns = 0;
    uint32_t type = 0;
};
// The largest even divisor of `columns` no larger than `preferred`: the K span one affine pair covers.
inline uint64_t u4_resident_group(uint64_t columns, uint64_t preferred) {
    if (columns < 2) throw std::invalid_argument("U4 resident group needs a non-empty width");
    uint64_t span = columns < preferred ? columns : preferred;
    if (span & 1) --span;
    for (; span >= 2; span -= 2)
        if (columns % span == 0) return span;
    throw std::invalid_argument("U4 resident group has no even divisor");
}
struct U4LayerStore {
    uint64_t experts = 0, hidden = 0, ff = 0;
    uint64_t gu_rows = 0, down_rows = 0;                     // 2*ff, hidden
    uint64_t gu_group = 0, down_group = 0;                   // K span of one affine pair
    uint64_t gu_groups = 0, down_groups = 0;
    uint64_t gu_codes_block = 0, gu_affine_block = 0;        // per-block bytes of one region
    uint64_t down_codes_block = 0, down_affine_block = 0;
    uint64_t gu_codes_start = 0, gu_scale_start = 0, gu_offset_start = 0;
    uint64_t down_codes_start = 0, down_scale_start = 0, down_offset_start = 0;
    uint64_t total = 0;
    // The production geometry (h=2560, ff=640) picks gu group 512 (5 groups, K=512, 256-byte code rows) and down
    // group 320 (2 groups, K=320, 160-byte code rows): the fewest GEMM slices whose groups still fit each row's
    // block-to-block magnitude spread.
    static U4LayerStore make(uint64_t experts, uint64_t hidden, uint64_t ff,
                             uint64_t gu_preferred = 512, uint64_t down_preferred = 320) {
        U4LayerStore s;
        s.experts = experts;
        s.hidden = hidden;
        s.ff = ff;
        s.gu_rows = 2 * ff;
        s.down_rows = hidden;
        s.gu_group = u4_resident_group(hidden, gu_preferred);
        s.down_group = u4_resident_group(ff, down_preferred);
        s.gu_groups = hidden / s.gu_group;
        s.down_groups = ff / s.down_group;
        s.gu_codes_block = s.gu_groups * s.gu_rows * (s.gu_group / 2);
        s.gu_affine_block = s.gu_groups * s.gu_rows * 4;
        s.down_codes_block = s.down_groups * s.down_rows * (s.down_group / 2);
        s.down_affine_block = s.down_groups * s.down_rows * 4;
        const auto aligned = [](uint64_t bytes) { return bytes % 128 == 0; };
        if (!experts || !hidden || !ff || !aligned(s.gu_codes_block) || !aligned(s.gu_affine_block) ||
            !aligned(s.down_codes_block) || !aligned(s.down_affine_block))
            throw std::invalid_argument("resident U4 layout needs blocks of 128-byte-aligned bytes");
        const uint64_t blocks = experts + 1;
        s.gu_codes_start = 0;
        s.gu_scale_start = s.gu_codes_start + blocks * s.gu_codes_block;
        s.gu_offset_start = s.gu_scale_start + blocks * s.gu_affine_block;
        s.down_codes_start = s.gu_offset_start + blocks * s.gu_affine_block;
        s.down_scale_start = s.down_codes_start + blocks * s.down_codes_block;
        s.down_offset_start = s.down_scale_start + blocks * s.down_affine_block;
        s.total = s.down_offset_start + blocks * s.down_affine_block;
        return s;
    }
    uint64_t blocks() const { return experts + 1; }
    uint64_t bytes() const { return total; }
    uint64_t block_bytes() const {
        return gu_codes_block + 2 * gu_affine_block + down_codes_block + 2 * down_affine_block;
    }
    // The gate/up batch weight's flat [batch=e*groups+g][n][k/2] byte, and the affine pair at [batch][n].
    uint64_t gu_code_at(uint64_t e, uint64_t g, uint64_t n) const {
        return gu_codes_start + (e * gu_groups + g) * gu_rows * (gu_group / 2) + n * (gu_group / 2);
    }
    uint64_t gu_scale_at(uint64_t e, uint64_t g, uint64_t n) const {
        return gu_scale_start + ((e * gu_groups + g) * gu_rows + n) * 4;
    }
    uint64_t gu_offset_at(uint64_t e, uint64_t g, uint64_t n) const {
        return gu_offset_start + ((e * gu_groups + g) * gu_rows + n) * 4;
    }
    uint64_t down_code_at(uint64_t e, uint64_t g, uint64_t n) const {
        return down_codes_start + (e * down_groups + g) * down_rows * (down_group / 2) + n * (down_group / 2);
    }
    uint64_t down_scale_at(uint64_t e, uint64_t g, uint64_t n) const {
        return down_scale_start + ((e * down_groups + g) * down_rows + n) * 4;
    }
    uint64_t down_offset_at(uint64_t e, uint64_t g, uint64_t n) const {
        return down_offset_start + ((e * down_groups + g) * down_rows + n) * 4;
    }
    // One packed block in staging order; pack_u4_block fills it, place_u4_block copies the six slices out.
    uint64_t staging_gu_codes() const { return 0; }
    uint64_t staging_gu_scale() const { return gu_codes_block; }
    uint64_t staging_gu_offset() const { return gu_codes_block + gu_affine_block; }
    uint64_t staging_down_codes() const { return gu_codes_block + 2 * gu_affine_block; }
    uint64_t staging_down_scale() const { return gu_codes_block + 2 * gu_affine_block + down_codes_block; }
    uint64_t staging_down_offset() const {
        return gu_codes_block + 2 * gu_affine_block + down_codes_block + down_affine_block;
    }
};
// Experts packed and uploaded per round: every worker packs into its own block of the pinned staging buffer, then
// the round's slices go up as six contiguous spans.
inline constexpr uint64_t kU4PackChunk = 8;
inline uint64_t u4_staging_bytes(const U4LayerStore& store) {
    return kU4PackChunk * store.block_bytes();
}
// Packs one expert into a staging block of store.block_bytes(): gate and up are [ff, hidden] GGUF IQ4 matrices
// stacked into the [2*ff, hidden] gate/up weight, down is [hidden, ff]. Throws if the sources' types or widths do
// not match the store, so the router's layer and the packed layer cannot disagree.
void pack_u4_block(const U4Source& gate, const U4Source& up, const U4Source& down, std::span<uint8_t> block,
                   const U4LayerStore& store, bool low_first = true);
// Copies a staging block to block `expert` of `layer` (size store.bytes()); expert == store.experts is the shared
// expert, which is the tail every region already reserves. Only that block's bytes are written.
void place_u4_block(std::span<const uint8_t> block, std::span<uint8_t> layer, const U4LayerStore& store,
                    uint64_t expert);
// Reference inverse of the store: what the MME-times-affine epilogue reconstructs, [rows, columns] row-major.
// gu selects the gate/up weight ([2*ff, hidden]) or down ([hidden, ff]); g of column k is k / group.
std::vector<float> dequantize_u4_matrix(std::span<const uint8_t> layer, const U4LayerStore& store, uint64_t expert,
                                        bool gu, bool low_first = true);
}
