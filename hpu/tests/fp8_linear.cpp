// Host semantic test of STRATA_HPU_FP8_DENSE's one-pass pair: tpc/fp8_encode_stacked.c (per-row scale, the hi and
// lo codes of row r stored as rows r and rows+r) and tpc/fp8_linear_finish.c
// (out[r][n] = (hi + lo/16) * x_scale[r] * w_scale[n]). The TPC intrinsics run on the CPU, so this pins the two
// kernels' indexing, scale placement, chunk tails and formula; the converter's bytes are pinned by
// tests/fp8_encode.cpp and the device's own division by tests/fp8_encode_device.cpp.
#include "strata/hpu/fp8.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace fp8_host {
using int5 = std::array<int, 5>;
inline int5 offset{}, extent{};
inline int5 get_index_space_offset() { return offset; }
inline int5 get_index_space_size() { return extent; }
inline int5 operator+(int5 a, const int5& b) { for (unsigned i = 0; i < 5; ++i) a[i] += b[i]; return a; }

struct float64 {
    std::array<float, 64> lanes{};
    float64(float x = 0) { lanes.fill(x); }
    float64& operator+=(const float64& b) { for (unsigned i = 0; i < 64; ++i) lanes[i] += b.lanes[i]; return *this; }
};
#define FP8_HOST_OP(op) inline float64 operator op(const float64& a, const float64& b) { \
    float64 out; for (unsigned i = 0; i < 64; ++i) out.lanes[i] = a.lanes[i] op b.lanes[i]; return out; }
FP8_HOST_OP(+)
FP8_HOST_OP(-)
FP8_HOST_OP(*)
FP8_HOST_OP(/)
#undef FP8_HOST_OP
inline float64 operator-(const float64& a) { return float64(0) - a; }
inline float64 v_f32_sel_less_f32_b(const float64& a, const float64& b, const float64& yes, const float64& no) {
    float64 out;
    for (unsigned i = 0; i < 64; ++i) out.lanes[i] = a.lanes[i] < b.lanes[i] ? yes.lanes[i] : no.lanes[i];
    return out;
}
inline float64 v_f32_max_b(const float64& a, const float64& b) {
    float64 out;
    for (unsigned i = 0; i < 64; ++i) out.lanes[i] = std::max(a.lanes[i], b.lanes[i]);
    return out;
}
// On the TPC a reduce leaves the scalar in every lane; fp8_encode_stacked.c stores the result with a full
// (non-partial) store into the {1, rows} scale tensor, which keeps lane 0 as the row's scale.
inline float64 v_f32_reduce_max(const float64& a) {
    float peak = a.lanes[0];
    for (unsigned i = 1; i < 64; ++i) peak = std::max(peak, a.lanes[i]);
    return float64(peak);
}

struct float256 { float64 v1, v2, v3, v4; };
struct minifloat256 { std::array<uint8_t, 256> codes{}; };
constexpr int SW_LINEAR = 0;

// Kernel-space tensor: dim 0 is the fastest-varying axis, as in the TPC coordinate convention (the SDK descriptor
// reverses the logical shape, hence glue.cpp's {columns, 2*rows} / {1, rows} checks). `written` tracks stores.
struct tensor {
    void* data = nullptr;
    uint64_t element = 4;
    std::vector<uint64_t> dims;
    std::vector<uint8_t>* written = nullptr;
    bool contains(const int5& position) const {
        for (size_t i = 0; i < 5; ++i)
            if (position[i] < 0 || uint64_t(position[i]) >= (i < dims.size() ? dims[i] : 1)) return false;
        return true;
    }
    uint64_t index(const int5& position) const {
        uint64_t result = 0, stride = 1;
        for (size_t i = 0; i < dims.size(); ++i) { result += uint64_t(position[i]) * stride; stride *= dims[i]; }
        return result;
    }
};
inline tensor make_f32(std::vector<float>& storage, std::vector<uint64_t> dims, std::vector<uint8_t>& written) {
    uint64_t count = 1;
    for (const auto d : dims) count *= d;
    if (storage.size() != count) throw std::logic_error("F32 fixture storage does not match its shape");
    written.assign(count, 0);
    return {storage.data(), 4, std::move(dims), &written};
}
inline tensor make_f8(std::vector<uint8_t>& storage, std::vector<uint64_t> dims, std::vector<uint8_t>& written) {
    uint64_t count = 1;
    for (const auto d : dims) count *= d;
    if (storage.size() != count) throw std::logic_error("FP8 fixture storage does not match its shape");
    written.assign(count, 0);
    return {storage.data(), 1, std::move(dims), &written};
}
inline float* gen_addr(const int5& position, tensor t) {
    if (t.element != 4) throw std::runtime_error("scalar address of a non-F32 tensor");
    if (!t.contains(position)) throw std::out_of_range("scalar tensor access outside the tensor");
    return static_cast<float*>(t.data) + t.index(position);
}
inline float s_f32_ld_g(float* p) { return *p; }
inline float64 v_f32_ld_tnsr_b(int5 position, tensor t) {
    if (t.element != 4) throw std::runtime_error("F32 vector load from a non-F32 tensor");
    float64 out;
    for (unsigned lane = 0; lane < 64; ++lane) {
        auto p = position; p[0] += int(lane);
        if (t.contains(p)) out.lanes[lane] = static_cast<float*>(t.data)[t.index(p)];
    }
    return out;
}
inline void v_f32_st_tnsr(int5 position, tensor t, const float64& value) {
    if (t.element != 4) throw std::runtime_error("F32 vector store to a non-F32 tensor");
    for (unsigned lane = 0; lane < 64; ++lane) {
        auto p = position; p[0] += int(lane);
        if (!t.contains(p)) continue; // broadcast store: only the tensor's own elements land
        const auto i = t.index(p);
        static_cast<float*>(t.data)[i] = value.lanes[lane];
        if (t.written) (*t.written)[i] = 1;
    }
}
inline void v_f32_st_tnsr_partial(int5 position, tensor t, const float64& value, int last, int first) {
    if (t.element != 4) throw std::runtime_error("F32 partial store to a non-F32 tensor");
    if (first < 0 || last < first || last > 63) throw std::runtime_error("F32 partial store lane range");
    for (int lane = first; lane <= last; ++lane) {
        auto p = position; p[0] += lane;
        if (!t.contains(p)) throw std::out_of_range("F32 partial store outside the tensor");
        const auto i = t.index(p);
        static_cast<float*>(t.data)[i] = value.lanes[lane];
        if (t.written) (*t.written)[i] = 1;
    }
}
inline void v_f8_st_tnsr_partial(int5 position, tensor t, const minifloat256& value, int last, int first) {
    if (t.element != 1) throw std::runtime_error("FP8 partial store to a non-F8 tensor");
    if (first < 0 || last < first || last > 255) throw std::runtime_error("FP8 partial store lane range");
    for (int lane = first; lane <= last; ++lane) {
        auto p = position; p[0] += lane;
        if (!t.contains(p)) throw std::out_of_range("FP8 partial store outside the tensor");
        const auto i = t.index(p);
        static_cast<uint8_t*>(t.data)[i] = value.codes[lane];
        if (t.written) (*t.written)[i] = 1;
    }
}
inline minifloat256 v_convert_f32_to_f8_all_b(const float256& x, int) {
    const float64* parts[4] = {&x.v1, &x.v2, &x.v3, &x.v4};
    minifloat256 out;
    for (unsigned part = 0; part < 4; ++part)
        for (unsigned lane = 0; lane < 64; ++lane)
            out.codes[part * 64 + lane] = strata::hpu::fp8_143_from_f32_reference(parts[part]->lanes[lane]);
    return out;
}
inline float256 v_convert_f8_to_f32_all_b(const minifloat256& x, int) {
    float256 out;
    float64* parts[4] = {&out.v1, &out.v2, &out.v3, &out.v4};
    for (unsigned part = 0; part < 4; ++part)
        for (unsigned lane = 0; lane < 64; ++lane)
            parts[part]->lanes[lane] = strata::hpu::f32_from_fp8_143(x.codes[part * 64 + lane]);
    return out;
}
#define __global__
#define main encode_stacked_kernel
#include "../tpc/fp8_encode_stacked.c"
#undef main
#define main linear_finish_kernel
#include "../tpc/fp8_linear_finish.c"
#undef main
#undef __global__
}

namespace {
using namespace fp8_host;
using strata::hpu::f32_from_fp8_143;
using strata::hpu::fp8_143_from_f32_reference;

const std::array<float, 256>& fp8_value_table() {
    static const std::array<float, 256> table = [] {
        std::array<float, 256> values{};
        for (unsigned code = 0; code < 256; ++code) values[code] = f32_from_fp8_143(uint8_t(code));
        return values;
    }();
    return table;
}
uint64_t next_random(uint64_t& state) {
    state = state * 6364136223846793005ull + 1442695040888963407ull;
    return state >> 11;
}
std::string real(double value) {
    std::ostringstream out;
    out << std::scientific << value;
    return out.str();
}
float random_unit(uint64_t& state) {
    return float(double(next_random(state) & ((uint64_t(1) << 53) - 1)) * (2.0 / double(uint64_t(1) << 53)) - 1.0);
}
// Row patterns cycle through the cases the encode has to survive: all-zero, uniform, alternating signs, huge and
// tiny magnitudes, random, one saturated element above a dense field, and all-negative.
std::vector<float> row_values(uint32_t rows, uint32_t columns, uint64_t seed) {
    std::vector<float> values(uint64_t(rows) * columns);
    for (uint32_t row = 0; row < rows; ++row) {
        for (uint32_t col = 0; col < columns; ++col) {
            float& v = values[uint64_t(row) * columns + col];
            const float sign = (col & 1) ? -1.0f : 1.0f;
            const float magnitude = 0.5f + 0.5f * std::fabs(random_unit(seed));
            switch (row % 8) {
            case 0: v = 0; break;
            case 1: v = 1.0f; break;
            case 2: v = 3.5f * sign; break;
            case 3: v = 1e20f * sign * magnitude; break;
            case 4: v = 1e-20f * sign * magnitude; break;
            case 5: v = random_unit(seed); break;
            case 6: v = col == 7 ? 240.0f : 0.001f * magnitude; break;
            default: v = -std::fabs(random_unit(seed)) - 0.25f; break;
            }
        }
    }
    return values;
}

struct encoded {
    std::vector<uint8_t> stacked;
    std::vector<float> scale;
};

encoded encode(uint32_t rows, uint32_t columns, std::vector<float>& values) {
    if (values.size() != uint64_t(rows) * columns) throw std::logic_error("encode fixture shape");
    encoded out;
    out.stacked.assign(uint64_t(2) * rows * columns, 0);
    out.scale.assign(rows, 0);
    std::vector<uint8_t> input_written, stacked_written, scale_written;
    tensor input = make_f32(values, {columns, rows}, input_written);
    tensor stacked = make_f8(out.stacked, {columns, uint64_t(2) * rows}, stacked_written);
    tensor scale = make_f32(out.scale, {1, rows}, scale_written);
    for (uint32_t row = 0; row < rows; ++row) {
        offset = {int(row), 0, 0, 0, 0};
        extent = {1, 1, 1, 1, 1};
        encode_stacked_kernel(input, stacked, scale, int(columns), int(rows));
    }
    for (size_t i = 0; i < stacked_written.size(); ++i)
        if (!stacked_written[i]) throw std::runtime_error("encode left stacked code " + std::to_string(i) + " unwritten");
    for (uint32_t row = 0; row < rows; ++row)
        if (!scale_written[row]) throw std::runtime_error("encode left scale row " + std::to_string(row) + " unwritten");
    return out;
}

void check_encode(uint32_t rows, uint32_t columns, const std::vector<float>& values, const encoded& e) {
    double worst = 0;
    for (uint32_t row = 0; row < rows; ++row) {
        double peak_value = 0;
        for (uint32_t col = 0; col < columns; ++col)
            peak_value = std::max(peak_value, double(std::fabs(values[uint64_t(row) * columns + col])));
        const float peak = float(peak_value);
        const volatile float product = peak * (1.0f / 240.0f);
        const float expected = product + 1e-30f;
        if (std::memcmp(&e.scale[row], &expected, sizeof expected) != 0)
            throw std::runtime_error("encode scale row " + std::to_string(row) + ": got " + real(e.scale[row]) +
                                     " expected " + real(expected) + " (peak " + real(peak) + ")");
        for (uint32_t col = 0; col < columns; ++col) {
            const size_t i = uint64_t(row) * columns + col;
            const float normalized = values[i] / e.scale[row];
            const uint8_t hi = fp8_143_from_f32_reference(normalized);
            const float restored = f32_from_fp8_143(hi);
            const uint8_t lo = fp8_143_from_f32_reference((normalized - restored) * 16.0f);
            const uint8_t actual_hi = e.stacked[uint64_t(col) + uint64_t(row) * columns];
            const uint8_t actual_lo = e.stacked[uint64_t(col) + uint64_t(rows + row) * columns];
            if (actual_hi != hi || actual_lo != lo)
                throw std::runtime_error("encode code row " + std::to_string(row) + " col " + std::to_string(col) +
                                         ": hi " + std::to_string(actual_hi) + "/" + std::to_string(hi) + " lo " +
                                         std::to_string(actual_lo) + "/" + std::to_string(lo));
            const float approximate = f32_from_fp8_143(actual_hi) + f32_from_fp8_143(actual_lo) * (1.0f / 16.0f);
            const double error = std::fabs(double(approximate) * e.scale[row] - values[i]);
            const double bound = std::ldexp(double(std::fabs(values[i])), -7) + std::ldexp(double(e.scale[row]), -7);
            if (!(error <= bound))
                throw std::runtime_error("encode reconstruction row " + std::to_string(row) + " col " + std::to_string(col) +
                                         ": error " + real(error) + " above " + real(bound));
            worst = std::max(worst, error / double(e.scale[row]));
        }
    }
    std::cout << "PASS encode rows=" << rows << " columns=" << columns
              << " worst_reconstruction=" << worst << " row_scales_exact\n";
}

void run_finish(uint32_t rows, uint32_t n, std::vector<float>& both, std::vector<float>& xs, std::vector<float>& ws,
                std::vector<float>& out, std::vector<uint8_t>& out_written) {
    std::vector<uint8_t> both_written, xs_written, ws_written;
    tensor both_t = make_f32(both, {n, uint64_t(2) * rows}, both_written);
    tensor xs_t = make_f32(xs, {1, rows}, xs_written);
    tensor ws_t = make_f32(ws, {n, 1}, ws_written);
    tensor out_t = make_f32(out, {n, rows}, out_written);
    const uint32_t chunks = (n + 63) / 64;
    for (uint32_t member = 0; member < rows * chunks; ++member) {
        offset = {int(member), 0, 0, 0, 0};
        extent = {1, 1, 1, 1, 1};
        linear_finish_kernel(both_t, xs_t, ws_t, out_t, int(n), int(rows));
    }
}

void check_finish(uint32_t rows, uint32_t n, uint64_t seed) {
    std::vector<float> both(uint64_t(2) * rows * n), xs(rows), ws(n), out(uint64_t(rows) * n, -12345.0f);
    for (auto& v : both) {
        const float r = random_unit(seed);
        v = r < -0.97f ? 1e-20f : r > 0.97f ? -1e-20f : 4.0f * r;
    }
    for (uint32_t row = 0; row < rows; ++row) xs[row] = row == 1 ? 0.0f : row % 3 == 0 ? 1e-20f : 0.5f + float(row % 5);
    for (uint32_t col = 0; col < n; ++col)
        ws[col] = ((col & 1) ? -1.0f : 1.0f) * (col % 7 == 3 ? 1e20f : 0.5f + float(col % 5) * 0.25f);
    std::vector<uint8_t> out_written;
    run_finish(rows, n, both, xs, ws, out, out_written);
    for (size_t i = 0; i < out_written.size(); ++i)
        if (!out_written[i]) throw std::runtime_error("finish left output element " + std::to_string(i) + " unwritten");
    for (uint32_t row = 0; row < rows; ++row) {
        for (uint32_t col = 0; col < n; ++col) {
            float expected;
            {
                const volatile float low = both[uint64_t(col) + uint64_t(rows + row) * n] * (1.0f / 16.0f);
                const volatile float product = both[uint64_t(col) + uint64_t(row) * n] + low;
                const volatile float scaled = product * xs[row];
                const volatile float result = scaled * ws[col];
                expected = result;
            }
            const float actual = out[uint64_t(col) + uint64_t(row) * n];
            if (std::memcmp(&actual, &expected, sizeof expected) != 0)
                throw std::runtime_error("finish row " + std::to_string(row) + " column " + std::to_string(col) + ": got " +
                                         real(actual) + " expected " + real(expected));
        }
    }
    std::cout << "PASS finish rows=" << rows << " columns=" << n << " chunks=" << (n + 63) / 64 << " exact_formula\n";
}

// The graph's own chain: encode x, the FP8 GEMM of the stacked hi/lo rows against the weight codes, then finish.
// The GEMM is summed in double here (the MME's accumulation order is not modelled). The identity checked is the
// one the kernels own: out equals the product of the activations reconstructed from the stacked codes, so a
// swapped hi/lo row, a wrong scale axis or a missing 1/16 all break it. The distance to x @ dequantized_w is
// reported, not bounded tightly: a row whose peak dwarfs its field represents that field only to the FP8
// subnormal floor, which tests/fp8_encode.cpp measures at the element level.
void check_end_to_end(uint32_t rows, uint32_t columns, uint32_t n, uint64_t seed) {
    std::vector<float> x = row_values(rows, columns, seed);
    const encoded e = encode(rows, columns, x);
    std::vector<uint8_t> codes(uint64_t(n) * columns);
    for (auto& code : codes) code = uint8_t(next_random(seed) & 0xff);
    std::vector<float> ws(n);
    for (uint32_t row = 0; row < n; ++row)
        ws[row] = ((row & 1) ? -1.0f : 1.0f) * (0.25f + float(next_random(seed) % 101) / 128.0f);
    const auto& table = fp8_value_table();
    std::vector<float> both(uint64_t(2) * rows * n);
    for (uint32_t row = 0; row < rows; ++row) {
        for (uint32_t out_col = 0; out_col < n; ++out_col) {
            double hi = 0, lo = 0;
            for (uint32_t k = 0; k < columns; ++k) {
                const double w = table[codes[uint64_t(out_col) * columns + k]];
                hi += double(table[e.stacked[uint64_t(k) + uint64_t(row) * columns]]) * w;
                lo += double(table[e.stacked[uint64_t(k) + uint64_t(rows + row) * columns]]) * w;
            }
            both[uint64_t(out_col) + uint64_t(row) * n] = float(hi);
            both[uint64_t(out_col) + uint64_t(rows + row) * n] = float(lo);
        }
    }
    std::vector<float> xs = e.scale, out(uint64_t(rows) * n);
    std::vector<uint8_t> out_written;
    run_finish(rows, n, both, xs, ws, out, out_written);
    double worst_identity = 0, worst_fidelity = 0;
    for (uint32_t row = 0; row < rows; ++row) {
        for (uint32_t out_col = 0; out_col < n; ++out_col) {
            double expected = 0, magnitude = 0, ideal = 0;
            for (uint32_t k = 0; k < columns; ++k) {
                const double w = double(table[codes[uint64_t(out_col) * columns + k]]) * ws[out_col];
                const double restored = (double(table[e.stacked[uint64_t(k) + uint64_t(row) * columns]]) +
                                         double(table[e.stacked[uint64_t(k) + uint64_t(rows + row) * columns]]) / 16.0) *
                                        double(e.scale[row]);
                expected += restored * w;
                magnitude += std::fabs(restored * w);
                ideal += double(x[uint64_t(row) * columns + k]) * w;
            }
            const double actual = out[uint64_t(out_col) + uint64_t(row) * n];
            const double error = std::fabs(actual - expected);
            if (!std::isfinite(actual) || !(error <= 1e-5 * magnitude + 1e-30))
                throw std::runtime_error("end-to-end identity row " + std::to_string(row) + " column " +
                                         std::to_string(out_col) + ": got " + real(actual) + " expected " +
                                         real(expected) + " error " + real(error) + " above " + real(1e-5 * magnitude));
            worst_identity = std::max(worst_identity, magnitude > 0 ? error / magnitude : 0.0);
            const double fidelity = magnitude > 0 ? std::fabs(actual - ideal) / magnitude : 0.0;
            if (fidelity > 0.05)
                throw std::runtime_error("end-to-end fidelity row " + std::to_string(row) + " column " +
                                         std::to_string(out_col) + ": " + real(fidelity) +
                                         " of the row magnitude away from x @ dequantized_w");
            worst_fidelity = std::max(worst_fidelity, fidelity);
        }
    }
    std::cout << "PASS end_to_end rows=" << rows << " columns=" << columns << " out_columns=" << n
              << " identity_error=" << worst_identity << " x_fidelity_error=" << worst_fidelity << '\n';
}
}

int main() {
    try {
        const std::pair<uint32_t, uint32_t> encode_shapes[] = {{1, 64}, {7, 128}, {8, 256}, {9, 320}, {64, 640}, {16, 2560}, {1, 65536}};
        uint64_t seed = 0x9e3779b97f4a7c15ull;
        for (const auto& shape : encode_shapes) {
            std::vector<float> values = row_values(shape.first, shape.second, seed);
            const encoded e = encode(shape.first, shape.second, values);
            check_encode(shape.first, shape.second, values, e);
        }
        const std::pair<uint32_t, uint32_t> finish_shapes[] = {{1, 64}, {4, 100}, {64, 64}, {8, 1000}, {64, 640}};
        for (const auto& shape : finish_shapes) check_finish(shape.first, shape.second, seed);
        const std::array<uint32_t, 3> gemm_shapes[] = {{8, 64, 64}, {7, 128, 100}, {8, 320, 192}, {16, 640, 640}, {16, 2560, 640}};
        for (const auto& shape : gemm_shapes) check_end_to_end(shape[0], shape[1], shape[2], seed);
        std::cout << "PASS FP8 dense one-pass pair: " << std::size(encode_shapes) << " encode, " << std::size(finish_shapes)
                  << " finish and " << std::size(gemm_shapes) << " end-to-end shapes\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
