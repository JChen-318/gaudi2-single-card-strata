#include "strata/hpu/expert.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

using namespace strata::hpu;
float rounded(float value) { return from_bf16(to_bf16(value)); }
void run(const Device& device, uint64_t tokens) {
    strata::core::ModelGeometry geometry;
    const uint64_t h = geometry.n_embd, ff = geometry.n_ff;
    auto recipe = build_bf16_expert(device, geometry, tokens);
    HostBuffer hx(device, tokens * h * 2), hg(device, ff * h * 2), hu(device, ff * h * 2);
    HostBuffer hd(device, h * ff * 2), hy(device, tokens * h * 2);
    DeviceBuffer dx(device, hx.size()), dg(device, hg.size()), du(device, hu.size());
    DeviceBuffer dd(device, hd.size()), dy(device, hy.size());
    auto* x = static_cast<uint16_t*>(hx.data());
    auto* wg = static_cast<uint16_t*>(hg.data());
    auto* wu = static_cast<uint16_t*>(hu.data());
    auto* wd = static_cast<uint16_t*>(hd.data());
    for (uint64_t i = 0; i < tokens * h; ++i)
        x[i] = to_bf16(static_cast<float>(static_cast<int>((i * 7 + 3) % 19) - 9) / 32);
    for (uint64_t i = 0; i < ff * h; ++i) {
        wg[i] = to_bf16(static_cast<float>(static_cast<int>((i * 11 + 5) % 23) - 11) / 256);
        wu[i] = to_bf16(static_cast<float>(static_cast<int>((i * 13 + 7) % 29) - 14) / 256);
    }
    Stream copy(device), compute(device), output(device);
    Event ready(device), done(device);
    for (unsigned phase = 0; phase < 2; ++phase) {
        const float sign = phase ? -1.0f : 1.0f;
        for (uint64_t i = 0; i < h * ff; ++i)
            wd[i] = to_bf16(sign * static_cast<float>(static_cast<int>((i * 17 + 9) % 31) - 15) / 128);
        upload(copy, dx, hx, hx.size());
        upload(copy, dg, hg, hg.size());
        upload(copy, du, hu, hu.size());
        upload(copy, dd, hd, hd.size());
        ready.record(copy);
        compute.wait(ready);
        const std::vector<Binding> bindings{{"x", &dx}, {"w_gate", &dg}, {"w_up", &du}, {"w_down", &dd}, {"out", &dy}};
        recipe->enqueue(compute, bindings);
        output.wait(recipe->completion());
        download(output, hy, dy, hy.size());
        done.record(output);
        done.synchronize();
        std::vector<float> intermediate(tokens * ff);
        for (uint64_t token = 0; token < tokens; ++token) {
            for (uint64_t row = 0; row < ff; ++row) {
                double gate = 0, up = 0;
                for (uint64_t k = 0; k < h; ++k) {
                    const auto value = from_bf16(x[token * h + k]);
                    gate += static_cast<double>(value) * from_bf16(wg[row * h + k]);
                    up += static_cast<double>(value) * from_bf16(wu[row * h + k]);
                }
                const float g = rounded(static_cast<float>(gate));
                const float u = rounded(static_cast<float>(up));
                const float sig = rounded(1.0f / (1.0f + std::exp(-g)));
                intermediate[token * ff + row] = rounded(rounded(g * sig) * u);
            }
        }
        double max_error = 0, error2 = 0, reference2 = 0;
        for (uint64_t token = 0; token < tokens; ++token) {
            for (uint64_t row = 0; row < h; ++row) {
                double expected = 0;
                for (uint64_t k = 0; k < ff; ++k)
                    expected += static_cast<double>(intermediate[token * ff + k]) * from_bf16(wd[row * ff + k]);
                expected = rounded(static_cast<float>(expected));
                const auto actual = from_bf16(static_cast<const uint16_t*>(hy.data())[token * h + row]);
                if (!std::isfinite(actual)) throw std::runtime_error("non-finite expert output");
                const auto error = std::abs(actual - expected);
                max_error = std::max(max_error, error);
                error2 += error * error;
                reference2 += expected * expected;
            }
        }
        const double relative_l2 = std::sqrt(error2 / std::max(reference2, 1e-30));
        if (relative_l2 > 0.02 || max_error > 0.002)
            throw std::runtime_error("expert parity failed: relative_l2=" + std::to_string(relative_l2) +
                                     " max_abs=" + std::to_string(max_error));
        std::cout << "PASS expert T=" << tokens << " H=" << h << " FF=" << ff << " phase=" << phase
                  << " relative_l2=" << relative_l2 << " max_abs_error=" << max_error << '\n';
    }
}
int main() {
    try {
        Device device;
        run(device, 1);
        run(device, 8);
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
