// Host side of the resident U4 store (strata/hpu/u4_resident.hpp): the affine choice is u4_quant.hpp's, so one
// code path decides how every expert is quantized; this file only lays the result out as the MME's [batch,N,K]
// tensor and gives the host test an inverse to compare against. Every row is decoded from GGUF IQ4 and re-fit on
// its own, so a block's sixteen values never influence another block's affine pair.
#include "strata/hpu/u4_resident.hpp"
#include "strata/hpu/u4_quant.hpp"
#include "iq4_decode.hpp"
#include <cstring>
#include <stdexcept>

namespace strata::hpu {
namespace {
constexpr int8_t kvalues[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
void check_store(const U4LayerStore& store) {
    if (!store.experts || !store.hidden || !store.ff || store.gu_group < 2 || store.down_group < 2 ||
        (store.gu_group & 1) || (store.down_group & 1) || store.hidden % store.gu_group ||
        store.ff % store.down_group || store.gu_rows != 2 * store.ff || store.down_rows != store.hidden ||
        store.gu_groups != store.hidden / store.gu_group || store.down_groups != store.ff / store.down_group ||
        store.bytes() != store.blocks() * store.block_bytes())
        throw std::invalid_argument("invalid resident U4 store");
}
void check_source(const U4Source& source, uint64_t rows, uint64_t columns) {
    if (!detail::iq4_type(source.type) || source.rows != rows || source.columns != columns ||
        source.row_bytes * source.rows > source.bytes.size() || (source.type == 23 && columns % 256) ||
        (source.type == 20 && columns % 32))
        throw std::invalid_argument("resident U4 pack needs IQ4_NL/IQ4_XS rows of the store's width");
}
// One projection's staging slices: rows is the projection's total row count (gu stacks gate then up), and row n's
// group g lands at (g*rows + n), the flat [g][n] order the batch GEMM's [batch=e*groups+g][N] weight reads.
struct RowSink {
    std::span<uint8_t> codes, scales, offsets;
    uint64_t rows = 0, group = 0;
};
void pack_rows(const U4Source& source, uint64_t first_row, uint64_t count, const RowSink& sink, bool low_first,
               std::vector<uint8_t>& codes, std::vector<float>& scales, std::vector<float>& values) {
    const uint64_t width = source.columns, half = sink.group / 2, groups = width / sink.group;
    for (uint64_t i = 0; i < count; ++i) {
        const uint64_t n = first_row + i;
        detail::decode_iq4_row(source.bytes.data() + i * source.row_bytes, source.type, width, codes.data(),
                               scales.data());
        for (uint64_t k = 0; k < width; ++k) values[k] = float(kvalues[codes[k]]) * scales[k / 32];
        const auto matrix = quantize_u4_matrix(std::span<const float>(values), 1, width, sink.group, low_first);
        for (uint64_t g = 0; g < groups; ++g) {
            const uint64_t slot = g * sink.rows + n;
            std::memcpy(sink.codes.data() + slot * half, matrix.codes.data() + g * half, half);
            std::memcpy(sink.scales.data() + slot * 4, &matrix.scales[g], 4);
            std::memcpy(sink.offsets.data() + slot * 4, &matrix.offsets[g], 4);
        }
    }
}
}
void pack_u4_block(const U4Source& gate, const U4Source& up, const U4Source& down, std::span<uint8_t> block,
                   const U4LayerStore& store, bool low_first) {
    check_store(store);
    check_source(gate, store.ff, store.hidden);
    check_source(up, store.ff, store.hidden);
    check_source(down, store.hidden, store.ff);
    if (block.size() < store.block_bytes()) throw std::invalid_argument("resident U4 staging block too small");
    const auto sink = [&](uint64_t codes_at, uint64_t scales_at, uint64_t codes_bytes, uint64_t affine_bytes,
                          uint64_t rows, uint64_t group) {
        return RowSink{block.subspan(codes_at, codes_bytes), block.subspan(scales_at, affine_bytes),
                       block.subspan(scales_at + affine_bytes, affine_bytes), rows, group};
    };
    const auto gu = sink(store.staging_gu_codes(), store.staging_gu_scale(), store.gu_codes_block,
                         store.gu_affine_block, store.gu_rows, store.gu_group);
    std::vector<uint8_t> codes(store.hidden);
    std::vector<float> scales(store.hidden / 32), values(store.hidden);
    pack_rows(gate, 0, store.ff, gu, low_first, codes, scales, values);
    pack_rows(up, store.ff, store.ff, gu, low_first, codes, scales, values);
    const auto d = sink(store.staging_down_codes(), store.staging_down_scale(), store.down_codes_block,
                        store.down_affine_block, store.down_rows, store.down_group);
    std::vector<uint8_t> down_codes(store.ff);
    std::vector<float> down_scales(store.ff / 32), down_values(store.ff);
    pack_rows(down, 0, store.hidden, d, low_first, down_codes, down_scales, down_values);
}
void place_u4_block(std::span<const uint8_t> block, std::span<uint8_t> layer, const U4LayerStore& store,
                    uint64_t expert) {
    check_store(store);
    if (expert >= store.blocks() || block.size() < store.block_bytes() || layer.size() < store.bytes())
        throw std::invalid_argument("resident U4 place needs a valid expert/block/layer");
    const auto copy = [&](uint64_t staging, uint64_t region, uint64_t slice) {
        std::memcpy(layer.data() + region + expert * slice, block.data() + staging, slice);
    };
    copy(store.staging_gu_codes(), store.gu_codes_start, store.gu_codes_block);
    copy(store.staging_gu_scale(), store.gu_scale_start, store.gu_affine_block);
    copy(store.staging_gu_offset(), store.gu_offset_start, store.gu_affine_block);
    copy(store.staging_down_codes(), store.down_codes_start, store.down_codes_block);
    copy(store.staging_down_scale(), store.down_scale_start, store.down_affine_block);
    copy(store.staging_down_offset(), store.down_offset_start, store.down_affine_block);
}
std::vector<float> dequantize_u4_matrix(std::span<const uint8_t> layer, const U4LayerStore& store, uint64_t expert,
                                        bool gu, bool low_first) {
    check_store(store);
    if (expert >= store.blocks() || layer.size() < store.bytes())
        throw std::invalid_argument("resident U4 dequantize needs a valid expert/layer");
    const uint64_t rows = gu ? store.gu_rows : store.down_rows;
    const uint64_t columns = gu ? store.hidden : store.ff;
    const uint64_t group = gu ? store.gu_group : store.down_group;
    const uint64_t groups = gu ? store.gu_groups : store.down_groups;
    std::vector<float> out(rows * columns);
    for (uint64_t g = 0; g < groups; ++g)
        for (uint64_t n = 0; n < rows; ++n) {
            const uint8_t* codes = layer.data() + (gu ? store.gu_code_at(expert, g, n) : store.down_code_at(expert, g, n));
            float scale, offset;
            std::memcpy(&scale, layer.data() + (gu ? store.gu_scale_at(expert, g, n) : store.down_scale_at(expert, g, n)), 4);
            std::memcpy(&offset, layer.data() + (gu ? store.gu_offset_at(expert, g, n) : store.down_offset_at(expert, g, n)), 4);
            float* row = out.data() + n * columns + g * group;
            for (uint64_t p = 0; p < group / 2; ++p) {
                const uint8_t byte = codes[p];
                row[2 * p] = scale * float(low_first ? byte & 15 : byte >> 4) + offset;
                row[2 * p + 1] = scale * float(low_first ? byte >> 4 : byte & 15) + offset;
            }
        }
    return out;
}
}
