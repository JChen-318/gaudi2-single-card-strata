// Host semantic test for STRATA_HPU_GDN_WINDOW (tpc/gdn_step_window.c): the whole-window kernel must be
// bit-identical to the per-row chain of tpc/gdn_step.c it replaces in define_gdn_prefill_final. Both run
// from their original TPC sources under the scalar/vector shims, so the claim tested is exactly the one
// gdn.cpp:352 relies on: a member owns one (64 value lanes, head) state slice, copies state_in once, then
// walks the rows in order and reads back the running state it wrote. The fixture also has to be sensitive:
// a row computed from the initial state must differ from the same row in the chain, otherwise carrying
// state between rows would be untested. Host only: no ISA execution, no device, no performance claim.
#include "tpc_emulator.hpp"
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace emulated {
#define __global__
#define main step_kernel
#include "../tpc/gdn_step.c"
#undef main
#define main window_kernel
#include "../tpc/gdn_step_window.c"
#undef main
#undef __global__
}
using namespace emulated;

namespace {
void require(bool ok, const std::string& why) { if (!ok) throw std::runtime_error(why); }
uint32_t next(uint32_t& state) { state = state * 1664525u + 1013904223u; return state; }
float uniform(uint32_t& state, float low, float high) {
    return low + (high - low) * (float(next(state) >> 8) / float(1u << 24));
}
bool same_bits(float a, float b) { return std::bit_cast<uint32_t>(a) == std::bit_cast<uint32_t>(b); }
void require_bits(const std::vector<float>& got, const std::vector<float>& want, const std::string& what) {
    require(got.size() == want.size(), what + ": size differs");
    for (size_t i = 0; i < got.size(); ++i)
        require(same_bits(got[i], want[i]), what + ": value " + std::to_string(i) + " differs (" +
            std::to_string(got[i]) + " vs " + std::to_string(want[i]) + ")");
}

void run(int width, int hk, int hv, int rows, uint32_t seed) {
    require(width % 64 == 0 && hv % hk == 0, "fixture geometry is not a whole number of 64-lane state blocks");
    const int blocks = width / 64;
    const std::string tag = "S=" + std::to_string(width) + " hk=" + std::to_string(hk) + " hv=" + std::to_string(hv) +
                            " rows=" + std::to_string(rows);
    std::vector<float> initial(size_t(width) * hv * width), q(size_t(width) * hk * rows),
        k(size_t(width) * hk * rows), v(size_t(width) * hv * rows), gate(size_t(hv) * rows),
        beta(size_t(hv) * rows);
    for (auto& value : initial) value = uniform(seed, -0.5f, 0.5f);
    for (auto& value : q) value = uniform(seed, -1.0f, 1.0f);
    for (auto& value : k) value = uniform(seed, -1.0f, 1.0f);
    for (auto& value : v) value = uniform(seed, -1.0f, 1.0f);
    for (auto& value : gate) value = uniform(seed, -1.5f, -0.5f);   // exp(gate) decays every row
    for (auto& value : beta) value = uniform(seed, 0.2f, 0.9f);
    const auto state_in = initial;
    const int5 members{blocks, hv, 1, 1, 1};

    // The per-row chain the host graph issues today: one step node per row, ping-pong state buffers.
    std::vector<float> chain_out(size_t(width) * hv * rows), chain_state;
    std::vector<float> state = initial, ping(initial.size());
    std::vector<float> q_row(size_t(width) * hk), k_row(size_t(width) * hk), v_row(size_t(width) * hv),
        gate_row(static_cast<size_t>(hv)), beta_row(static_cast<size_t>(hv)), out_row(size_t(width) * hv);
    for (int row = 0; row < rows; ++row) {
        std::copy_n(q.begin() + size_t(row) * width * hk, q_row.size(), q_row.begin());
        std::copy_n(k.begin() + size_t(row) * width * hk, k_row.size(), k_row.begin());
        std::copy_n(v.begin() + size_t(row) * width * hv, v_row.size(), v_row.begin());
        std::copy_n(gate.begin() + size_t(row) * hv, gate_row.size(), gate_row.begin());
        std::copy_n(beta.begin() + size_t(row) * hv, beta_row.size(), beta_row.begin());
        offset = {}; extent = members;
        step_kernel({&state, {uint64_t(width), uint64_t(hv), uint64_t(width)}},
                    {&q_row, {uint64_t(width), uint64_t(hk)}},
                    {&k_row, {uint64_t(width), uint64_t(hk)}},
                    {&v_row, {uint64_t(width), uint64_t(hv)}},
                    {&gate_row, {uint64_t(hv)}},
                    {&beta_row, {uint64_t(hv)}},
                    {&ping, {uint64_t(width), uint64_t(hv), uint64_t(width)}},
                    {&out_row, {uint64_t(width), uint64_t(hv)}}, width, hk);
        std::copy(out_row.begin(), out_row.end(), chain_out.begin() + size_t(row) * width * hv);
        state.swap(ping);
    }
    chain_state = state;

    // The whole-window kernel: one launch, every (block, head) member walks the rows itself.
    std::vector<float> window_state(initial.size(), -7.0f), window_out(size_t(width) * hv * rows, -7.0f);
    offset = {}; extent = members;
    window_kernel({&initial, {uint64_t(width), uint64_t(hv), uint64_t(width)}},
                  {&q, {uint64_t(width), uint64_t(hk), uint64_t(rows)}},
                  {&k, {uint64_t(width), uint64_t(hk), uint64_t(rows)}},
                  {&v, {uint64_t(width), uint64_t(hv), uint64_t(rows)}},
                  {&gate, {uint64_t(hv), uint64_t(rows)}},
                  {&beta, {uint64_t(hv), uint64_t(rows)}},
                  {&window_state, {uint64_t(width), uint64_t(hv), uint64_t(width)}},
                  {&window_out, {uint64_t(width), uint64_t(hv), uint64_t(rows)}}, width, hk, rows);

    require_bits(window_state, chain_state, tag + ": final state");
    require_bits(window_out, chain_out, tag + ": per-row outputs");
    require_bits(initial, state_in, tag + ": the window kernel modified state_in");
    for (float value : window_out) require(std::isfinite(value), tag + ": a row output is not finite");

    // The fixture must distinguish "row r runs on the state the previous row left" from "row r runs on the
    // initial state": otherwise the equality above could hold for the wrong reason.
    if (rows > 1) {
        std::vector<float> fresh(initial.size()), sensitivity_in = state_in;
        std::vector<float> one_shot(size_t(width) * hv);
        std::copy_n(q.begin() + size_t(width) * hk, q_row.size(), q_row.begin());
        std::copy_n(k.begin() + size_t(width) * hk, k_row.size(), k_row.begin());
        std::copy_n(v.begin() + size_t(width) * hv, v_row.size(), v_row.begin());
        std::copy_n(gate.begin() + size_t(hv), gate_row.size(), gate_row.begin());
        std::copy_n(beta.begin() + size_t(hv), beta_row.size(), beta_row.begin());
        offset = {}; extent = members;
        step_kernel({&sensitivity_in, {uint64_t(width), uint64_t(hv), uint64_t(width)}},
                    {&q_row, {uint64_t(width), uint64_t(hk)}},
                    {&k_row, {uint64_t(width), uint64_t(hk)}},
                    {&v_row, {uint64_t(width), uint64_t(hv)}},
                    {&gate_row, {uint64_t(hv)}},
                    {&beta_row, {uint64_t(hv)}},
                    {&fresh, {uint64_t(width), uint64_t(hv), uint64_t(width)}},
                    {&one_shot, {uint64_t(width), uint64_t(hv)}}, width, hk);
        bool differs = false;
        for (int i = 0; i < width * hv; ++i)
            differs = differs || !same_bits(one_shot[size_t(i)], chain_out[size_t(width) * hv + size_t(i)]);
        require(differs, tag + ": row 1 is insensitive to the row-0 state, so the chain check proves nothing");
    }
    std::cout << "PASS GDN whole-window kernel bit-identical to the per-row chain: " << tag << '\n';
}
}

int main() {
    try {
        run(64, 1, 3, 1, 0x2f6e2b1u);
        run(64, 2, 6, 5, 0x71ac9d3u);
        run(128, 16, 48, 4, 0x4d3c2b1u);
        run(128, 4, 8, 9, 0x9e3779b9u);
        // The production prefill geometry: 128 value lanes x 48 heads, 6 key heads, a 64-row window.
        run(128, 6, 48, 64, 0x1f123bb5u);
        std::cout << "PASS GDN window semantics: whole-window kernel equals the per-row chain bit for bit; no device\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
