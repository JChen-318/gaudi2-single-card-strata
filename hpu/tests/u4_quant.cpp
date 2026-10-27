// Host-side checks for the IQ4->U4 requantizer. No device, no Synapse: the affine choice and the packed byte
// order are pure host arithmetic, so the quality of 4-bit affine quantization is measured before any card time.
// The error printed here is the *weight* error only; the model-level gate remains the KL/top-1 comparison.
#include "strata/hpu/u4_quant.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>
using namespace strata::hpu;
namespace {
void check(bool ok, const std::string& why) {
    if (!ok) throw std::runtime_error(why);
}
constexpr uint64_t kRows = 8, kColumns = 2560;
// The exact-affine case: a matrix that is itself scale*code+offset must come back with zero error.
void exact_affine() {
    constexpr uint64_t rows = 3, columns = 64;
    std::vector<float> source(rows * columns);
    for (uint64_t r = 0; r < rows; ++r) {
        const float scale = 0.25f * float(r + 1), offset = -2.0f + 0.5f * float(r);
        for (uint64_t k = 0; k < columns; ++k)
            source[r * columns + k] = scale * float((r * 13 + k * 7) % 16) + offset;
    }
    const auto matrix = quantize_u4_matrix(source, rows, columns);
    const auto restored = dequantize_u4_matrix(matrix);
    const double relative = u4_relative_rms(source, restored);
    double worst = 0;
    for (size_t i = 0; i < source.size(); ++i) worst = std::max(worst, std::abs(double(source[i] - restored[i])));
    std::cout << "U4 QUANT exact-affine relative_rms=" << relative << " max_abs_error=" << worst << '\n';
    check(worst < 1e-5, "an exactly representable matrix did not round-trip");
}
// IQ4-shaped stress data: 32-value blocks whose scale varies by a factor 4 across a row, values on the IQ4
// codebook grid, plus an 8x spike every 997 columns. Per-row quantization cannot absorb the block spread; this
// fixture says how much of the error is granularity and how much is the 4-bit step itself.
std::vector<float> iq4_style(bool outliers) {
    constexpr int8_t codebook[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
    std::mt19937 generator(20261024);
    std::uniform_int_distribution<int> code(0, 15);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<float> source(kRows * kColumns);
    for (uint64_t r = 0; r < kRows; ++r)
        for (uint64_t k = 0; k < kColumns; ++k) {
            const float magnitude = 0.5f + 1.5f * float((r + k / 32) % 5) / 4.0f;
            float value = float(codebook[code(generator)]) / 127.0f * magnitude;
            if (outliers && k % 997 == 0) value *= 8.0f;
            value += 0.02f * normal(generator); // GGUF weights are not exactly on the codebook grid
            source[r * kColumns + k] = value;
        }
    return source;
}
// Transformer-shaped data: roughly Gaussian values, the block RMS wandering by +-20% along a row, and a 0.1%
// tail six times the block RMS. This is the distribution a per-32-block format is tuned for, without the IQ4
// codebook's own level set, so it isolates the affine 4-bit step.
std::vector<float> gaussian_rows() {
    std::mt19937 generator(20261025);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::uniform_real_distribution<float> spread(0.8f, 1.2f);
    std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
    std::vector<float> source(kRows * kColumns);
    for (uint64_t r = 0; r < kRows; ++r)
        for (uint64_t k = 0; k < kColumns; ++k) {
            float value = spread(generator) * normal(generator);
            if (uniform(generator) < 0.001f) value *= 6.0f;
            source[r * kColumns + k] = value;
        }
    return source;
}
struct Curve { uint64_t group = 0; double relative = 0, worst_row = 0; };
Curve measure(const std::vector<float>& source, uint64_t group) {
    const auto matrix = quantize_u4_matrix(source, kRows, kColumns, group);
    const auto restored = dequantize_u4_matrix(matrix);
    Curve out{group, u4_relative_rms(source, restored), 0};
    for (uint64_t r = 0; r < kRows; ++r) {
        const std::vector<float> a(source.begin() + int64_t(r * kColumns), source.begin() + int64_t((r + 1) * kColumns));
        const std::vector<float> b(restored.begin() + int64_t(r * kColumns), restored.begin() + int64_t((r + 1) * kColumns));
        out.worst_row = std::max(out.worst_row, u4_relative_rms(a, b));
    }
    return out;
}
void curve(const char* label, const std::vector<float>& source) {
    std::cout << "U4 QUANT " << label << " columns=" << kColumns << " rows=" << kRows << '\n';
    for (const uint64_t group : {uint64_t(0), uint64_t(640), uint64_t(320), uint64_t(160), uint64_t(32)}) {
        const auto point = measure(source, group);
        std::cout << "  group=" << (point.group ? std::to_string(point.group) : "row") << " relative_rms="
                  << std::fixed << std::setprecision(6) << point.relative << " worst_row=" << point.worst_row
                  << std::defaultfloat << '\n';
    }
}
void quality() {
    const auto iq4 = iq4_style(true);
    curve("stress/iq4-codebook+outliers", iq4);
    // The block-sized group must absorb the 4x spread, which is what proves the group plumbing (not just the fit)
    // works. Per-row was measured at 0.25 here before clipping, so this gate fails unless both parts are right.
    const auto block = measure(iq4, 32);
    check(block.relative < 0.15, "per-32 U4 error is too large even for the 4-bit step");
    std::cout << "U4 QUANT stress gate group=32 relative_rms=" << block.relative << " < 0.15\n";
    const auto gaussian = gaussian_rows();
    curve("realistic/gaussian+tail", gaussian);
    const auto row = measure(gaussian, 0), sliced = measure(gaussian, 320);
    check(sliced.relative <= row.relative + 1e-9, "a per-320 group must never quantize worse than one pair per row");
    check(sliced.relative < 0.15, "the production per-320 group exceeds the 4-bit error budget");
    std::cout << "U4 QUANT realistic gate group=320 relative_rms=" << sliced.relative
              << " <= row=" << row.relative << " and < 0.15\n";
}
void packing_and_mutation() {
    U4Matrix matrix;
    matrix.rows = 1;
    matrix.columns = 4;
    matrix.group_columns = 4;
    matrix.low_first = true;
    matrix.codes = {0xBA, 0x32}; // codes 10,11,2,3
    matrix.scales = {1.0f};
    matrix.offsets = {0.0f};
    const auto restored = dequantize_u4_matrix(matrix);
    check(restored[0] == 10 && restored[1] == 11 && restored[2] == 2 && restored[3] == 3,
          "low-nibble-first packing did not round-trip");
    matrix.low_first = false;
    const auto swapped = dequantize_u4_matrix(matrix);
    check(swapped[0] == 11 && swapped[1] == 10 && swapped[2] == 3 && swapped[3] == 2,
          "high-nibble-first packing is not the inverse of low-first");
    matrix.low_first = true;
    matrix.group_columns = 2;
    matrix.scales = {1.0f, -1.0f};
    matrix.offsets = {0.0f, 2.0f};
    const auto grouped = dequantize_u4_matrix(matrix);
    check(grouped[0] == 10 && grouped[1] == 11 && grouped[2] == 0 && grouped[3] == -1,
          "per-group affine pairs are not applied to their own columns");
    std::cout << "U4 QUANT packing low-first=10,11,2,3 high-first=11,10,3,2 groups=10,11,0,-1\n";
}
}
int main() {
    try {
        exact_affine();
        quality();
        packing_and_mutation();
        std::cout << "PASS U4 quantizer\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
