// hpu/tests/window_sampling.cpp - host evidence for the per-sequence sampling fan-out of a multi-sequence window.
//
// The laws: exactly one token per sequence, sampled from ITS OWN last row; the draw counter is that row's ABSOLUTE
// position (so a windowed step and a scalar step with the same seed agree - checked below against the scalar sampler
// itself); a missing/short/non-finite row is refused with the row and sequence named; and a bias that does not match the
// vocabulary is refused rather than silently ignored.
#include "strata/hpu/window_sampling.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using strata::hpu::build_window_row_plan;
using strata::hpu::sample_window_rows;
using strata::hpu::sample;
using strata::hpu::WindowSampleRequest;
using strata::hpu::WindowSequence;
using strata::kernels::SamplerParams;

int failures = 0;
void check(const bool ok, const std::string& what) {
    if (!ok) {
        ++failures;
        std::cout << "FAIL " << what << "\n";
    }
}
std::string refusal(auto&& call) {
    try {
        call();
    } catch (const std::invalid_argument& error) {
        return error.what();
    }
    return {};
}
bool has(const std::string& message, const std::string& needle) {
    return message.find(needle) != std::string::npos;
}
SamplerParams greedy() {
    SamplerParams params;
    params.greedy = true;
    params.top_p = 1;
    params.top_k = 64;
    return params;
}
} // namespace

int main() {
    const uint32_t vocabulary = 8;
    const auto plan = build_window_row_plan(
        std::vector<WindowSequence>{{.sequence = 0, .kv_base = 0, .first_position = 5, .rows = 2},
                                    {.sequence = 1, .kv_base = 100, .first_position = 9, .rows = 1}},
        4096, 4096, 8);
    // Rows: 0,1 belong to sequence 0; row 2 to sequence 1. Sequence 0 samples row 1, sequence 1 samples row 2.
    check((plan.sample_rows == std::vector<uint32_t>{1, 2}), "the plan says which row each sequence samples");

    // 1. greedy: each sequence's token comes from its own row, and the two rows disagree by construction
    std::vector<float> logits(size_t(plan.rows()) * vocabulary, -10.0f);
    logits[0 * vocabulary + 3] = 99.0f;   // row 0 (not sampled by anyone)
    logits[1 * vocabulary + 5] = 99.0f;   // row 1 -> sequence 0
    logits[2 * vocabulary + 6] = 99.0f;   // row 2 -> sequence 1
    {
        std::vector<int32_t> s0{}, s1{};
        std::vector<WindowSampleRequest> requests{{.history = s0, .params = greedy(), .bias = {}},
                                                  {.history = s1, .params = greedy(), .bias = {}}};
        const auto result = sample_window_rows(plan, logits, vocabulary, requests);
        check(result.tokens.size() == 2 && result.rows.size() == 2, "one token per sequence");
        check((result.rows == std::vector<uint32_t>{1, 2}), "the sampled rows are the plan's sample rows");
        check(result.tokens[0] == 5, "sequence 0 takes its own row's argmax, not row 0's");
        check(result.tokens[1] == 6, "sequence 1 takes its own row's argmax");
    }

    // 2. the draw counter is the absolute position: a stochastic draw must equal the scalar sampler called with
    //    counter == position, which is what makes a windowed step reproduce a scalar step at the same seed.
    {
        std::vector<float> row(vocabulary, 0.0f);
        row[2] = 1.0f;
        row[5] = 1.0f;
        std::vector<float> slab(size_t(plan.rows()) * vocabulary, 0.0f);
        std::copy(row.begin(), row.end(), slab.begin() + 1 * vocabulary);
        std::copy(row.begin(), row.end(), slab.begin() + 2 * vocabulary);
        SamplerParams params;
        params.top_p = 1;
        params.top_k = 64;
        params.temperature = 1;
        params.seed = 12345;
        std::vector<int32_t> history{};
        std::vector<WindowSampleRequest> requests{{.history = history, .params = params, .bias = {}},
                                                  {.history = history, .params = params, .bias = {}}};
        const auto result = sample_window_rows(plan, slab, vocabulary, requests);
        auto scalar_a = params;
        scalar_a.counter = plan.row_position[plan.sample_rows[0]];   // its sampled row's position
        auto scalar_b = params;
        scalar_b.counter = plan.row_position[plan.sample_rows[1]];   // its sampled row's position
        check(result.tokens[0] == sample(std::span<const float>(row), history, scalar_a),
              "sequence 0's draw equals the scalar sampler at its sampled row position");
        check(result.tokens[1] == sample(std::span<const float>(row), history, scalar_b),
              "sequence 1's draw equals the scalar sampler at its sampled row position");
    }

    // 3. refusals: a non-finite row, a slab that does not match the plan, a missing request, a wrong bias width
    {
        auto bad = logits;
        bad[2 * vocabulary + 0] = std::numeric_limits<float>::quiet_NaN();
        std::vector<int32_t> history{};
        std::vector<WindowSampleRequest> requests{{.history = history, .params = greedy(), .bias = {}},
                                                  {.history = history, .params = greedy(), .bias = {}}};
        const auto message = refusal([&] { (void)sample_window_rows(plan, bad, vocabulary, requests); });
        check(has(message, "sequence 1 row 2") && has(message, "non-finite"), "a non-finite row names sequence and row");
        check(has(refusal([&] { (void)sample_window_rows(plan, bad, vocabulary, requests); }), "non-finite"),
              "the refusal is the non-finite one, not a shape one");
        check(has(refusal([&] { (void)sample_window_rows(plan, logits, vocabulary,
                                                        std::span<const WindowSampleRequest>(requests).first(1)); }),
                  "one request per sequence"),
              "a missing request is refused");
        check(has(refusal([&] { (void)sample_window_rows(plan, std::span<const float>(logits).first(3), vocabulary,
                                                        requests); }),
                  "does not match the plan"),
              "a short logits slab is refused");
        std::vector<float> narrow{0.1f, 0.2f};
        std::vector<WindowSampleRequest> wrong_bias{{.history = history, .params = greedy(), .bias = narrow},
                                                    {.history = history, .params = greedy(), .bias = {}}};
        check(has(refusal([&] { (void)sample_window_rows(plan, logits, vocabulary, wrong_bias); }),
                  "bias does not match the vocabulary"),
              "a bias of the wrong width is refused");
    }

    if (failures != 0) {
        std::cout << "FAILED window-sampling checks: " << failures << "\n";
        return 1;
    }
    std::cout << "PASS window sampling: one token per sequence from its own sample row, the draw counter equals that "
                 "row's absolute position (checked against the scalar sampler), a non-finite row names its sequence and "
                 "row, and a missing request, a short slab and a mis-sized bias are refused; no device\n";
    return 0;
}
