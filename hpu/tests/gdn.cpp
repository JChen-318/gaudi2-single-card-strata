#include "strata/hpu/gdn.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

using namespace strata::hpu;
int main() {
    try {
        Device device;
        strata::core::ModelGeometry geometry;
        const uint64_t s = geometry.ssm_state_size, hk = geometry.ssm_k_heads, hv = geometry.ssm_v_heads;
        auto recipe = build_gdn_step(device, geometry);
        HostBuffer initial(device, s * hv * s * 4), q(device, hk * s * 4), k(device, hk * s * 4);
        HostBuffer v(device, hv * s * 4), gate(device, hv * 4), beta(device, hv * 4);
        HostBuffer next_state(device, initial.size()), output(device, v.size());
        DeviceBuffer state_a(device, initial.size()), state_b(device, initial.size());
        DeviceBuffer dq(device, q.size()), dk(device, k.size()), dv(device, v.size());
        DeviceBuffer dg(device, gate.size()), db(device, beta.size()), dout(device, v.size());
        auto* sq = static_cast<float*>(q.data());
        auto* sk = static_cast<float*>(k.data());
        auto* sv = static_cast<float*>(v.data());
        auto* sg = static_cast<float*>(gate.data());
        auto* sb = static_cast<float*>(beta.data());
        auto* si = static_cast<float*>(initial.data());
        for (uint64_t i = 0; i < s * hv * s; ++i)
            si[i] = static_cast<float>(static_cast<int>((i * 13 + 3) % 41) - 20) / 512;
        for (uint64_t head = 0; head < hk; ++head) {
            double qnorm = 0, knorm = 0;
            for (uint64_t i = 0; i < s; ++i) {
                const auto index = head * s + i;
                sq[index] = static_cast<float>(static_cast<int>((head * 17 + i * 7) % 31) - 15);
                sk[index] = static_cast<float>(static_cast<int>((head * 11 + i * 13) % 37) - 18);
                qnorm += sq[index] * sq[index];
                knorm += sk[index] * sk[index];
            }
            for (uint64_t i = 0; i < s; ++i) {
                sq[head * s + i] /= static_cast<float>(std::sqrt(qnorm) * std::sqrt(s));
                sk[head * s + i] /= static_cast<float>(std::sqrt(knorm));
            }
        }
        for (uint64_t i = 0; i < hv * s; ++i)
            sv[i] = static_cast<float>(static_cast<int>((i * 19 + 1) % 43) - 21) / 128;
        std::vector<float> reference(si, si + s * hv * s), expected(hv * s);
        Stream input(device), compute(device), readback(device);
        Event ready(device), done(device);
        upload(input, state_a, initial, initial.size());
        upload(input, dq, q, q.size());
        upload(input, dk, k, k.size());
        upload(input, dv, v, v.size());
        DeviceBuffer* current = &state_a;
        DeviceBuffer* next = &state_b;
        for (unsigned phase = 0; phase < 4; ++phase) {
            for (uint64_t head = 0; head < hv; ++head) {
                sg[head] = phase == 0 ? 0.0f : -0.1f * static_cast<float>(1 + (head + phase) % 11);
                sb[head] = phase == 0 || phase == 3 ? 0.0f : 0.1f * static_cast<float>(1 + head % 8);
            }
            upload(input, dg, gate, gate.size());
            upload(input, db, beta, beta.size());
            ready.record(input);
            compute.wait(ready);
            const std::vector<Binding> bindings{{"state", current}, {"q", &dq}, {"k", &dk},
                {"v", &dv}, {"gate", &dg}, {"beta", &db}, {"state_out", next}, {"out", &dout}};
            recipe->enqueue(compute, bindings);
            readback.wait(recipe->completion());
            download(readback, next_state, *next, next_state.size());
            download(readback, output, dout, output.size());
            done.record(readback);
            // Independent scalar reference with Strata's modulo head pairing and update order.
            for (uint64_t head = 0; head < hv; ++head) {
                const auto key_head = head % hk;
                const float decay = std::exp(sg[head]);
                for (uint64_t j = 0; j < s; ++j) {
                    float contraction = 0;
                    for (uint64_t i = 0; i < s; ++i) {
                        auto& cell = reference[(i * hv + head) * s + j];
                        cell *= decay;
                        contraction += cell * sk[key_head * s + i];
                    }
                    const float delta = (sv[head * s + j] - contraction) * sb[head];
                    float answer = 0;
                    for (uint64_t i = 0; i < s; ++i) {
                        auto& cell = reference[(i * hv + head) * s + j];
                        cell += sk[key_head * s + i] * delta;
                        answer += cell * sq[key_head * s + i];
                    }
                    expected[head * s + j] = answer;
                }
            }
            done.synchronize();
            double state_error = 0, output_error = 0;
            for (uint64_t i = 0; i < reference.size(); ++i) {
                const auto actual = static_cast<const float*>(next_state.data())[i];
                if (!std::isfinite(actual)) throw std::runtime_error("non-finite GDN state");
                state_error = std::max(state_error, static_cast<double>(std::abs(actual - reference[i])));
            }
            for (uint64_t i = 0; i < expected.size(); ++i) {
                const auto actual = static_cast<const float*>(output.data())[i];
                if (!std::isfinite(actual)) throw std::runtime_error("non-finite GDN output");
                output_error = std::max(output_error, static_cast<double>(std::abs(actual - expected[i])));
            }
            if (state_error > 1e-5 || output_error > 1e-5)
                throw std::runtime_error("GDN parity failed: state=" + std::to_string(state_error) +
                                         " output=" + std::to_string(output_error));
            std::cout << "PASS GDN S=" << s << " key_heads=" << hk << " value_heads=" << hv
                      << " phase=" << phase << " max_state_error=" << state_error
                      << " max_output_error=" << output_error << " modulo_pairing=ok\n";
            std::swap(current, next);
        }
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
