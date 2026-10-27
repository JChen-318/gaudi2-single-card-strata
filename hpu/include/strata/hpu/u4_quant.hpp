#pragma once
// Host-side U4 weight quantization for the MME expert path. The MME decompresses syn_type_uint4 weights with a
// scale/zero point bound to the tensor, but the resident expert graph runs one batch GEMM over all experts and
// therefore binds a single uniform (scale=1, zp=0) pair, recovering each expert's own precision in an epilogue:
//     w(e,n,k) = scale[e,n,group(k)]*code(e,n,k) + offset[e,n,group(k)]
// `group_columns` is the K span sharing one affine pair. One pair per row is the fewest GEMM slices; a shorter
// span absorbs a row's block-to-block magnitude variation at the cost of one GEMM per group. Codes are always
// assigned against the F32 pair the epilogue will use, so the stored pair and the stored codes agree exactly.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace strata::hpu {
struct U4Matrix {
    uint64_t rows = 0, columns = 0, group_columns = 0;
    bool low_first = true;
    std::vector<uint8_t> codes;  // rows*columns codes packed two per byte along a row (row stride columns/2)
    std::vector<float> scales;   // [rows*groups]
    std::vector<float> offsets;  // [rows*groups]
    uint64_t groups() const { return group_columns ? (columns + group_columns - 1) / group_columns : 0; }
    uint64_t group(uint64_t column) const { return column / group_columns; }
};
namespace u4_detail {
struct Fit { double scale = 0, offset = 0, mse = std::numeric_limits<double>::infinity(); };
// One (scale, offset) pair for values[0..n): the range candidates are a clipped +-t*sigma around the mean and the
// raw min/max, each followed by two least-squares refits with the codes held fixed. A row whose block magnitudes
// vary by 4x, or whose largest value is eight times the rest, is fit on its bulk instead of on its extremes.
inline Fit fit_group(const float* values, uint64_t n, std::vector<uint8_t>& codes) {
    double sum = 0, sum_sq = 0;
    float lo = values[0], hi = values[0];
    for (uint64_t k = 0; k < n; ++k) {
        const double v = values[k];
        sum += v;
        sum_sq += v * v;
        lo = std::min(lo, values[k]);
        hi = std::max(hi, values[k]);
    }
    const double mean = sum / double(n);
    const double variance = std::max(0.0, sum_sq / double(n) - mean * mean);
    const double sigma = std::sqrt(variance);
    Fit best;
    const auto consider = [&](double low, double high) {
        if (!(high > low) || !std::isfinite(low) || !std::isfinite(high)) return;
        double scale = (high - low) / 15.0, offset = low;
        for (int pass = 0; pass < 3; ++pass) {
            for (uint64_t k = 0; k < n; ++k)
                codes[k] = uint8_t(std::clamp<long>(std::lround((double(values[k]) - offset) / scale), 0, 15));
            if (pass == 2) break;
            double sum_c = 0, sum_w = 0, sum_cc = 0, sum_cw = 0;
            for (uint64_t k = 0; k < n; ++k) {
                const double code = codes[k];
                sum_c += code;
                sum_w += values[k];
                sum_cc += code * code;
                sum_cw += code * double(values[k]);
            }
            const double count = double(n), denominator = count * sum_cc - sum_c * sum_c;
            if (!(denominator > 0)) break;
            const double fit_scale = (count * sum_cw - sum_c * sum_w) / denominator;
            const double fit_offset = (sum_w - fit_scale * sum_c) / count;
            if (!(fit_scale > 0) || !std::isfinite(fit_scale) || !std::isfinite(fit_offset)) break;
            scale = fit_scale;
            offset = fit_offset;
        }
        double error = 0;
        for (uint64_t k = 0; k < n; ++k) {
            const double restored = double(scale * double(codes[k]) + offset);
            error += (double(values[k]) - restored) * (double(values[k]) - restored);
        }
        const double mse = error / double(n);
        if (mse < best.mse) best = Fit{scale, offset, mse};
    };
    if (sigma > 0) for (const double t : {2.2, 2.8}) consider(std::max(mean - t * sigma, double(lo)), std::min(mean + t * sigma, double(hi)));
    consider(lo, hi);
    return best;
}
}
inline U4Matrix quantize_u4_matrix(std::span<const float> weights, uint64_t rows, uint64_t columns,
                                   uint64_t group_columns = 0, bool low_first = true) {
    if (!rows || !columns || (columns & 1) || weights.size() != rows * columns)
        throw std::invalid_argument("U4 quantization needs a non-empty [rows,even columns] matrix");
    if (!group_columns) group_columns = columns;
    if ((group_columns & 1) || columns % group_columns)
        throw std::invalid_argument("U4 group must be an even divisor of the row width");
    U4Matrix out;
    out.rows = rows;
    out.columns = columns;
    out.group_columns = group_columns;
    out.low_first = low_first;
    out.codes.resize(rows * columns / 2);
    const uint64_t groups = out.groups();
    out.scales.resize(rows * groups);
    out.offsets.resize(rows * groups);
    std::vector<uint8_t> codes(group_columns);
    for (uint64_t r = 0; r < rows; ++r) {
        const float* row = weights.data() + r * columns;
        uint8_t* packed = out.codes.data() + r * (columns / 2);
        for (uint64_t g = 0; g < groups; ++g) {
            const float* values = row + g * group_columns;
            for (uint64_t k = 0; k < group_columns; ++k)
                if (!std::isfinite(values[k])) throw std::invalid_argument("non-finite U4 weight row");
            const auto fit = u4_detail::fit_group(values, group_columns, codes);
            const float fs = float(fit.scale), fo = float(fit.offset);
            for (uint64_t k = 0; k < group_columns; ++k)
                codes[k] = uint8_t(std::clamp<long>(std::lround((double(values[k]) - double(fo)) / double(fs)), 0, 15));
            out.scales[r * groups + g] = fs;
            out.offsets[r * groups + g] = fo;
            uint8_t* group_bytes = packed + g * (group_columns / 2);
            for (uint64_t k = 0; k < group_columns; k += 2)
                group_bytes[k / 2] = low_first ? uint8_t(codes[k] | (codes[k + 1] << 4))
                                               : uint8_t((codes[k] << 4) | codes[k + 1]);
        }
    }
    return out;
}
inline std::vector<float> dequantize_u4_matrix(const U4Matrix& matrix) {
    if (!matrix.rows || !matrix.columns || (matrix.columns & 1) || !matrix.group_columns ||
        matrix.columns % matrix.group_columns || (matrix.group_columns & 1) ||
        matrix.codes.size() != matrix.rows * matrix.columns / 2 ||
        matrix.scales.size() != matrix.rows * matrix.groups() ||
        matrix.offsets.size() != matrix.rows * matrix.groups())
        throw std::invalid_argument("invalid U4 matrix");
    const uint64_t groups = matrix.groups();
    std::vector<float> out(matrix.rows * matrix.columns);
    for (uint64_t r = 0; r < matrix.rows; ++r) {
        const uint8_t* packed = matrix.codes.data() + r * (matrix.columns / 2);
        for (uint64_t k = 0; k < matrix.columns; k += 2) {
            const uint8_t byte = packed[k / 2];
            const unsigned low = byte & 15, high = byte >> 4;
            const uint64_t g = k / matrix.group_columns;
            const float scale = matrix.scales[r * groups + g], offset = matrix.offsets[r * groups + g];
            out[r * matrix.columns + k] = scale * float(matrix.low_first ? low : high) + offset;
            out[r * matrix.columns + k + 1] = scale * float(matrix.low_first ? high : low) + offset;
        }
    }
    return out;
}
// Root-mean-square relative error against the source's own RMS, the number the IQ4->U4 requantizer reports.
inline double u4_relative_rms(const std::vector<float>& source, const std::vector<float>& restored) {
    if (source.size() != restored.size() || source.empty()) throw std::invalid_argument("U4 error shape mismatch");
    double error = 0, signal = 0;
    for (size_t i = 0; i < source.size(); ++i) {
        error += double(source[i] - restored[i]) * double(source[i] - restored[i]);
        signal += double(source[i]) * double(source[i]);
    }
    return signal > 0 ? std::sqrt(error / signal) : 0;
}
}
