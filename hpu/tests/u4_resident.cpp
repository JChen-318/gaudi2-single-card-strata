// Host-only check of the resident U4 store (strata/hpu/u4_resident.hpp). No device, no Synapse: the store layout
// and the packer are pure host arithmetic, and the epilogue that will consume it applies exactly the affine the
// reference inverse below applies. The fixture is a synthetic GGUF matrix whose dequantized values are known by
// construction (the codes and F16 block scales are chosen here, not read back through the decoder), so this checks
//   * the layer size the planner prices, every region start and every per-block slice at 128-byte alignment;
//   * that the flat [batch=e*groups+g][n][k/2] address is the address the expert-block layout uses, so the
//     routed region is the transpose_b batch weight of one group GEMM and the shared expert is its tail;
//   * the IQ4_NL and IQ4_XS row decoders against that construction, through the quantizer, bit for bit;
//   * that pack_u4_block fills every code and every scale/offset slot, and place_u4_block writes only its own
//     expert's bytes (the other blocks stay poisoned, an earlier block stays untouched);
//   * the weight error the MME path starts from at group 512 (gate/up) and 320 (down) versus a whole-row group.
#include "strata/hpu/u4_quant.hpp"
#include "strata/hpu/u4_resident.hpp"
#include "strata/kernels/f16_bits.hpp"
#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace strata::hpu;
namespace {
constexpr int8_t kvalues[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
constexpr uint32_t kPoison = 0x7FC00000u;  // NaN in every F32 affine slot
void check(bool ok, const std::string& why) { if (!ok) throw std::runtime_error(why); }
template <typename F> void check_throws(F&& f, const std::string& why) {
    try { f(); } catch (const std::invalid_argument&) { return; }
    throw std::runtime_error(why);
}

struct Fixture {
    std::shared_ptr<std::vector<uint8_t>> owner;
    U4Source source;
    std::vector<float> values;  // what the GGUF dequantizer must produce
};
// A GGUF IQ4_NL [rows,columns] matrix: 18 bytes per 32 values (F16 scale then 16 bytes of code pairs).
Fixture make_iq4_nl(uint64_t rows, uint64_t columns, uint32_t seed) {
    const uint64_t blocks = columns / 32;
    auto bytes = std::make_shared<std::vector<uint8_t>>(rows * blocks * 18);
    Fixture out;
    out.values.assign(rows * columns, 0.0f);
    std::mt19937 generator(seed);
    std::uniform_int_distribution<int> code(0, 15);
    for (uint64_t r = 0; r < rows; ++r)
        for (uint64_t b = 0; b < blocks; ++b) {
            uint8_t* p = bytes->data() + (r * blocks + b) * 18;
            // Block magnitudes vary 8x across a row; that spread is what a whole-row affine pair has to absorb and
            // a 512- or 320-column group does not.
            const float magnitude = 0.25f + 1.75f * float((r + b) % 5) / 4.0f;
            const uint16_t bits = strata::kernels::f16_from_f32(magnitude);
            p[0] = uint8_t(bits & 255);
            p[1] = uint8_t(bits >> 8);
            const float scale = strata::kernels::f32_from_f16(bits);
            for (int j = 0; j < 16; ++j) {
                int low = code(generator), high = code(generator);
                // An occasional extreme value: the per-group fit must keep the bulk of the block accurate.
                if ((r * columns + b * 32 + j) % 997 == 0) low = 15;
                if ((r * columns + b * 32 + 16 + j) % 991 == 0) high = 0;
                p[2 + j] = uint8_t(low | (high << 4));
                out.values[r * columns + b * 32 + j] = float(kvalues[low]) * scale;
                out.values[r * columns + b * 32 + 16 + j] = float(kvalues[high]) * scale;
            }
        }
    out.owner = bytes;
    out.source = U4Source{std::span<const uint8_t>(bytes->data(), bytes->size()), columns / 32 * 18, rows, columns, 20};
    return out;
}
// A GGUF IQ4_XS [rows,columns] matrix: 136 bytes per 256 values (F16 d, 2-bit highs, 8 ls nibbles, codes).
Fixture make_iq4_xs(uint64_t rows, uint64_t columns, uint32_t seed) {
    const uint64_t blocks = columns / 256;
    auto bytes = std::make_shared<std::vector<uint8_t>>(rows * blocks * 136);
    Fixture out;
    out.values.assign(rows * columns, 0.0f);
    std::mt19937 generator(seed);
    std::uniform_int_distribution<int> code(0, 15);
    for (uint64_t r = 0; r < rows; ++r)
        for (uint64_t s = 0; s < blocks; ++s) {
            uint8_t* p = bytes->data() + (r * blocks + s) * 136;
            const float d = 0.125f + 0.125f * float((r + s) % 4);
            const uint16_t bits = strata::kernels::f16_from_f32(d);
            p[0] = uint8_t(bits & 255);
            p[1] = uint8_t(bits >> 8);
            const float drift = strata::kernels::f32_from_f16(bits);
            unsigned high = 0;
            for (int ib = 0; ib < 8; ++ib) {
                const int ls = (ib * 7 + 5) % 64;
                high |= unsigned((ls >> 4) & 3) << (2 * ib);
                const uint8_t keep = uint8_t((ib % 2) ? 0x0F : 0xF0);   // the other sub-block shares this byte
                p[4 + ib / 2] = uint8_t((p[4 + ib / 2] & keep) | ((ls & 15) << (4 * (ib % 2))));
                const float scale = drift * float(ls - 32);
                for (int j = 0; j < 16; ++j) {
                    const int low = code(generator), top = code(generator);
                    p[8 + 16 * ib + j] = uint8_t(low | (top << 4));
                    out.values[r * columns + s * 256 + 32 * ib + j] = float(kvalues[low]) * scale;
                    out.values[r * columns + s * 256 + 32 * ib + 16 + j] = float(kvalues[top]) * scale;
                }
            }
            p[2] = uint8_t(high & 255);
            p[3] = uint8_t(high >> 8);
        }
    out.owner = bytes;
    out.source = U4Source{std::span<const uint8_t>(bytes->data(), bytes->size()), columns / 256 * 136, rows, columns, 23};
    return out;
}
// One matrix's rows quantized the way the packer must: [rows,columns] U4 at `group`, dequantized back.
std::vector<float> expected_rows(const Fixture& part, uint64_t columns, uint64_t group) {
    std::vector<float> out(part.values.size());
    for (uint64_t n = 0; n < part.source.rows; ++n) {
        const auto matrix = quantize_u4_matrix(std::span<const float>(part.values.data() + n * columns, columns), 1,
                                              columns, group);
        const auto row = dequantize_u4_matrix(matrix);
        std::memcpy(out.data() + n * columns, row.data(), columns * sizeof(float));
    }
    return out;
}
// Every affine slot of `expert` must hold a real value: the sentinel is NaN, so a slot the packer skipped shows.
void check_affines(const std::vector<uint8_t>& layer, const U4LayerStore& store, uint64_t expert) {
    const auto scan = [&](uint64_t start, uint64_t slice) {
        const uint8_t* p = layer.data() + start + expert * slice;
        for (uint64_t i = 0; i < slice; i += 4) {
            float value;
            std::memcpy(&value, p + i, 4);
            check(std::isfinite(value), "an affine slot was left at the NaN sentinel");
        }
    };
    scan(store.gu_scale_start, store.gu_affine_block);
    scan(store.gu_offset_start, store.gu_affine_block);
    scan(store.down_scale_start, store.down_affine_block);
    scan(store.down_offset_start, store.down_affine_block);
}
// The block still holds the poison pattern: no other expert's write reached it.
void check_untouched(const std::vector<uint8_t>& layer, const U4LayerStore& store, uint64_t expert) {
    const auto scan = [&](uint64_t start, uint64_t slice) {
        const uint8_t* p = layer.data() + start + expert * slice;
        for (uint64_t i = 0; i < slice; i += 4) {
            uint32_t bits;
            std::memcpy(&bits, p + i, 4);
            check(bits == kPoison, "an expert write touched another block's affine slot");
        }
    };
    scan(store.gu_scale_start, store.gu_affine_block);
    scan(store.gu_offset_start, store.gu_affine_block);
    scan(store.down_scale_start, store.down_affine_block);
    scan(store.down_offset_start, store.down_affine_block);
    const auto codes = [&](uint64_t start, uint64_t slice) {
        const uint8_t* p = layer.data() + start + expert * slice;
        for (uint64_t i = 0; i < slice; ++i) check(p[i] == 0xCC, "an expert write touched another block's codes");
    };
    codes(store.gu_codes_start, store.gu_codes_block);
    codes(store.down_codes_start, store.down_codes_block);
}
void poison(std::vector<uint8_t>& layer, const U4LayerStore& store) {
    std::fill(layer.begin(), layer.end(), 0xCC);
    const auto fill = [&](uint64_t start, uint64_t slice) {
        for (uint64_t e = 0; e < store.blocks(); ++e)
            for (uint64_t i = 0; i < slice; i += 4)
                std::memcpy(layer.data() + start + e * slice + i, &kPoison, 4);
    };
    fill(store.gu_scale_start, store.gu_affine_block);
    fill(store.gu_offset_start, store.gu_affine_block);
    fill(store.down_scale_start, store.down_affine_block);
    fill(store.down_offset_start, store.down_affine_block);
}
// The production layer the planner prices: 512 routed experts plus the shared one at h=2560/ff=640.
void production_geometry() {
    check(u4_resident_group(2560, 512) == 512, "the gate/up group is not 512 for hidden 2560");
    check(u4_resident_group(640, 320) == 320, "the down group is not 320 for FF 640");
    const auto store = U4LayerStore::make(512, 2560, 640);
    check(store.gu_rows == 1280 && store.down_rows == 2560, "expert matrix row counts");
    check(store.gu_group == 512 && store.gu_groups == 5, "gate/up K split");
    check(store.down_group == 320 && store.down_groups == 2, "down K split");
    check(store.gu_codes_block == 1'638'400 && store.gu_affine_block == 25'600, "gate/up block bytes");
    check(store.down_codes_block == 819'200 && store.down_affine_block == 20'480, "down block bytes");
    check(store.block_bytes() == 2'549'760, "one expert block is not 2,549,760 bytes");
    check(store.blocks() == 513, "block count is 512 routed plus the shared expert");
    check(store.bytes() == 513 * 2'549'760, "layer store bytes are not blocks*block_bytes");
    check(store.bytes() == 1'308'026'880, "the layer store is not 1,308,026,880 bytes");
    check(store.bytes() == 62'785'290'240 / 48, "one layer is not a forty-eighth of the model payload");
    // The six regions are consecutive [blocks] tensors; each start and each per-block slice stays 128-byte aligned.
    check(store.gu_scale_start == store.blocks() * store.gu_codes_block, "gu scale region start");
    check(store.gu_offset_start == store.gu_scale_start + store.blocks() * store.gu_affine_block, "gu offset start");
    check(store.down_codes_start == store.gu_offset_start + store.blocks() * store.gu_affine_block, "down codes start");
    check(store.down_scale_start == store.down_codes_start + store.blocks() * store.down_codes_block, "down scale start");
    check(store.down_offset_start == store.down_scale_start + store.blocks() * store.down_affine_block,
          "down offset start");
    check(store.bytes() == store.down_offset_start + store.blocks() * store.down_affine_block, "last region end");
    for (const uint64_t value : {store.gu_codes_start, store.gu_scale_start, store.gu_offset_start,
                                 store.down_codes_start, store.down_scale_start, store.down_offset_start})
        check(value % 128 == 0, "a region start is not 128-byte aligned");
    for (const uint64_t value : {store.gu_codes_block, store.gu_affine_block, store.down_codes_block,
                                 store.down_affine_block, store.block_bytes(), store.bytes()})
        check(value % 128 == 0, "a per-block slice or the store size is not 128-byte aligned");
    // The flat batch-tensor address and the expert-block address must be the same byte for every (e, g, n): the
    // routed codes are exactly one [E*G][N][K/2] batch weight, and the shared expert starts where it ends.
    for (const uint64_t e : {uint64_t(0), uint64_t(1), uint64_t(511), uint64_t(512)})
        for (uint64_t g = 0; g < store.gu_groups; ++g)
            for (uint64_t n = 0; n < store.gu_rows; n += 97) {
                const uint64_t batch = (e * store.gu_groups + g) * store.gu_rows * (store.gu_group / 2) +
                                       n * (store.gu_group / 2);
                check(store.gu_code_at(e, g, n) == store.gu_codes_start + batch,
                      "the batch-tensor and expert-block code offsets disagree");
            }
    for (const uint64_t e : {uint64_t(0), uint64_t(777), uint64_t(512)})
        for (uint64_t g = 0; g < store.down_groups; ++g)
            for (uint64_t n = 0; n < store.down_rows; n += 131) {
                const uint64_t batch = (e * store.down_groups + g) * store.down_rows * (store.down_group / 2) +
                                       n * (store.down_group / 2);
                check(store.down_code_at(e, g, n) == store.down_codes_start + batch,
                      "the down batch-tensor and expert-block code offsets disagree");
            }
    check(store.gu_code_at(512, 0, 0) == store.gu_codes_start + 512 * 5 * 1280 * 256,
          "the shared expert does not start where the routed gate/up batch weight ends");
    check(store.gu_scale_at(512, 0, 0) == store.gu_scale_start + 512 * 5 * 1280 * 4,
          "the shared expert does not start where the routed gate/up affines end");
    check(store.down_code_at(512, 0, 0) == store.down_codes_start + 512 * 2 * 2560 * 160,
          "the shared expert does not start where the routed down batch weight ends");
    // A geometry whose block is not 128-byte aligned must be rejected, not silently mis-bound.
    check_throws([] { U4LayerStore::make(4, 100, 100); }, "an unaligned U4 block layout was accepted");
    check_throws([] { U4LayerStore::make(0, 2560, 640); }, "an empty expert count was accepted");
}
void packed_store() {
    const uint64_t hidden = 2560, ff = 640;
    const auto store = U4LayerStore::make(2, hidden, ff);  // two routed experts plus the shared block
    check(store.bytes() == 3 * 2'549'760, "the packing store is not three production blocks");
    const auto gate = make_iq4_nl(ff, hidden, 20261024);
    const auto up = make_iq4_nl(ff, hidden, 20261025);
    const auto down = make_iq4_nl(hidden, ff, 20261026);
    const auto ogate = make_iq4_nl(ff, hidden, 20261027);
    const auto oup = make_iq4_nl(ff, hidden, 20261028);
    const auto odown = make_iq4_nl(hidden, ff, 20261029);

    std::vector<uint8_t> layer(store.bytes());
    poison(layer, store);
    std::vector<uint8_t> block(store.block_bytes(), 0xCC);
    pack_u4_block(gate.source, up.source, down.source, block, store);
    place_u4_block(block, layer, store, 0);
    check_affines(layer, store, 0);
    check_untouched(layer, store, 1);
    check_untouched(layer, store, store.experts);

    std::vector<float> gu_expected(store.gu_rows * hidden), down_expected(hidden * ff);
    std::memcpy(gu_expected.data(), expected_rows(gate, hidden, store.gu_group).data(), gate.values.size() * sizeof(float));
    std::memcpy(gu_expected.data() + ff * hidden, expected_rows(up, hidden, store.gu_group).data(),
                up.values.size() * sizeof(float));
    std::memcpy(down_expected.data(), expected_rows(down, ff, store.down_group).data(), down.values.size() * sizeof(float));
    const auto restored_gu = dequantize_u4_matrix(layer, store, 0, true);
    const auto restored_down = dequantize_u4_matrix(layer, store, 0, false);
    check(std::memcmp(restored_gu.data(), gu_expected.data(), gu_expected.size() * sizeof(float)) == 0,
          "gate/up does not reconstruct the quantizer's own values bit for bit");
    check(std::memcmp(restored_down.data(), down_expected.data(), down_expected.size() * sizeof(float)) == 0,
          "down does not reconstruct the quantizer's own values bit for bit");

    // Expert 1's write must leave expert 0 byte-identical and the shared block poisoned.
    const std::vector<uint8_t> snapshot(layer);
    std::vector<uint8_t> block1(store.block_bytes(), 0xCC);
    pack_u4_block(ogate.source, oup.source, odown.source, block1, store);
    place_u4_block(block1, layer, store, 1);
    check_affines(layer, store, 1);
    check_untouched(layer, store, store.experts);
    const auto same_slice = [&](uint64_t start, uint64_t slice) {
        return std::memcmp(layer.data() + start, snapshot.data() + start, slice) == 0;
    };
    check(same_slice(store.gu_codes_start, store.gu_codes_block), "expert 1 overwrote expert 0's gate/up codes");
    check(same_slice(store.gu_scale_start, store.gu_affine_block), "expert 1 overwrote expert 0's gate/up scales");
    check(same_slice(store.gu_offset_start, store.gu_affine_block), "expert 1 overwrote expert 0's gate/up offsets");
    check(same_slice(store.down_codes_start, store.down_codes_block), "expert 1 overwrote expert 0's down codes");
    check(same_slice(store.down_scale_start, store.down_affine_block), "expert 1 overwrote expert 0's down scales");
    check(same_slice(store.down_offset_start, store.down_affine_block), "expert 1 overwrote expert 0's down offsets");
    const auto other_gu = dequantize_u4_matrix(layer, store, 1, true);
    check(std::memcmp(other_gu.data(), gu_expected.data(), gu_expected.size() * sizeof(float)) != 0,
          "expert 1 reads back expert 0's values: the comparison above is not sensitive");

    // The shared expert is the tail block: placing the same packing at index `experts` reads back expert 0's values.
    place_u4_block(block, layer, store, store.experts);
    check_affines(layer, store, store.experts);
    const auto shared_gu = dequantize_u4_matrix(layer, store, store.experts, true);
    check(std::memcmp(shared_gu.data(), gu_expected.data(), gu_expected.size() * sizeof(float)) == 0,
          "the shared expert tail does not read back the packed block");
    const auto shared_down = dequantize_u4_matrix(layer, store, store.experts, false);
    check(std::memcmp(shared_down.data(), down_expected.data(), down_expected.size() * sizeof(float)) == 0,
          "the shared expert tail does not read back the packed down block");
    check(std::memcmp(layer.data() + store.gu_codes_start, snapshot.data() + store.gu_codes_start,
                      store.gu_codes_block) == 0,
          "the shared expert write touched expert 0's block");

    // The nibble order is part of the format: reading high-first must disagree, or the check above proves nothing.
    const auto flipped = dequantize_u4_matrix(layer, store, 0, true, false);
    check(std::memcmp(flipped.data(), gu_expected.data(), gu_expected.size() * sizeof(float)) != 0,
          "the high-nibble-first read matches too: the nibble-order check is not sensitive");

    // Quantization error the MME path starts from, and the whole-row group it must beat.
    std::vector<float> originals(gate.values);
    originals.insert(originals.end(), up.values.begin(), up.values.end());
    const double relative = u4_relative_rms(originals, restored_gu);
    double gate_rms = 0, up_rms = 0;
    {
        const std::vector<float> a(originals.begin(), originals.begin() + int64_t(ff * hidden));
        const std::vector<float> b(restored_gu.begin(), restored_gu.begin() + int64_t(ff * hidden));
        gate_rms = u4_relative_rms(a, b);
        const std::vector<float> c(originals.begin() + int64_t(ff * hidden), originals.end());
        const std::vector<float> d(restored_gu.begin() + int64_t(ff * hidden), restored_gu.end());
        up_rms = u4_relative_rms(c, d);
    }
    const double down_rms = u4_relative_rms(down.values, restored_down);
    const auto row_matrix = quantize_u4_matrix(std::span<const float>(gate.values.data(), hidden), 1, hidden, hidden);
    const double whole_row = u4_relative_rms(
        std::vector<float>(gate.values.begin(), gate.values.begin() + int64_t(hidden)), dequantize_u4_matrix(row_matrix));
    std::cout << "U4 RESIDENT store_bytes=" << store.bytes() << " block=" << store.block_bytes()
              << " gu_group=" << store.gu_group << " down_group=" << store.down_group << " rel_rms=" << relative
              << " gate=" << gate_rms << " up=" << up_rms << " down=" << down_rms
              << " whole_row=" << whole_row << '\n';
    check(relative < 0.15, "the gate/up group-512 error is too large for the MME path");
    check(down_rms < 0.15, "the down group-320 error is too large for the MME path");
    check(relative < whole_row, "the group-512 split is not better than one affine pair per row");

    check_throws([&] { dequantize_u4_matrix(layer, store, store.blocks(), true); },
                 "an out-of-range expert was read");
    check_throws([&] { place_u4_block(block, layer, store, store.blocks()); }, "an out-of-range expert was written");
    U4Source narrow = gate.source;
    narrow.columns = 512;
    check_throws([&] { pack_u4_block(narrow, up.source, down.source, block, store); },
                 "a source of the wrong width was packed");
}
// IQ4_XS through the same store: d * (ls - 32) per 32-column block, four bits per code just the same.
void iq4_xs_path() {
    const uint64_t hidden = 256, ff = 256;
    const auto store = U4LayerStore::make(1, hidden, ff);
    check(store.gu_group == 256 && store.gu_groups == 1, "IQ4_XS gate/up group");
    check(store.down_group == 256 && store.down_groups == 1, "IQ4_XS down group");
    check(store.bytes() == 2 * 104'448, "IQ4_XS store bytes");
    const auto gate = make_iq4_xs(ff, hidden, 20261030);
    const auto up = make_iq4_xs(ff, hidden, 20261031);
    const auto down = make_iq4_xs(hidden, ff, 20261032);
    std::vector<uint8_t> layer(store.bytes(), 0xCC);
    std::vector<uint8_t> block(store.block_bytes(), 0xCC);
    pack_u4_block(gate.source, up.source, down.source, block, store);
    place_u4_block(block, layer, store, 0);
    check_affines(layer, store, 0);
    std::vector<float> gu_expected(store.gu_rows * hidden);
    std::memcpy(gu_expected.data(), expected_rows(gate, hidden, store.gu_group).data(), gate.values.size() * sizeof(float));
    std::memcpy(gu_expected.data() + ff * hidden, expected_rows(up, hidden, store.gu_group).data(),
                up.values.size() * sizeof(float));
    const auto restored_gu = dequantize_u4_matrix(layer, store, 0, true);
    const auto restored_down = dequantize_u4_matrix(layer, store, 0, false);
    check(std::memcmp(restored_gu.data(), gu_expected.data(), gu_expected.size() * sizeof(float)) == 0,
          "IQ4_XS gate/up does not reconstruct the quantizer's values");
    check(std::memcmp(restored_down.data(), expected_rows(down, ff, store.down_group).data(),
                      down.values.size() * sizeof(float)) == 0,
          "IQ4_XS down does not reconstruct the quantizer's values");
    std::vector<float> originals(gate.values);
    originals.insert(originals.end(), up.values.begin(), up.values.end());
    std::cout << "U4 RESIDENT iq4_xs rel_rms=" << u4_relative_rms(originals, restored_gu)
              << " down=" << u4_relative_rms(down.values, restored_down) << '\n';
}
}
int main() {
    try {
        production_geometry();
        packed_store();
        iq4_xs_path();
        std::cout << "PASS U4 resident store\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
