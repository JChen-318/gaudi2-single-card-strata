#pragma once
// hpu/include/strata/hpu/window_sampling.hpp - sample ONE token per sequence out of a multi-sequence window.
//
// A window that carries several sequences produces one logits row per row of the plan, and each sequence must be
// sampled from ITS OWN last row (`WindowRowPlan::sample_rows`) - never from another sequence's, and never from a row in
// the padded tail. This header is that fan-out, kept pure so a host test can pin it:
//
//   * exactly one token per sequence, taken from that sequence's `sample_rows` entry;
//   * the draw counter is the sequence's absolute POSITION, matching the port's Philox convention
//     (`sp.counter = position`, so a seed gives the same token whether it comes from a scalar or a windowed step);
//   * a row that is missing, too short, or carries a non-finite value is refused; the refusal names the row and the
//     sequence, because a silent zero would enter that sequence's history;
//   * the caller's `history` span is per sequence (penalties only look at that sequence's tokens).
//
// Header-only over the existing sampler, no SDK.
#include "strata/hpu/sampler.hpp"
#include "strata/hpu/window_rows.hpp"

#include <cmath>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace strata::hpu {
struct WindowSampleRequest {
    std::span<const int32_t> history;              // that sequence's own recent tokens (penalties)
    strata::kernels::SamplerParams params;         // seed/temperature/top-k/... ; `counter` is overwritten per row
    std::span<const float> bias;                   // optional per-vocabulary bias
};

struct WindowSampleResult {
    std::vector<int32_t> tokens;   // [sequences] in plan order
    std::vector<uint32_t> rows;    // [sequences] the plan row each token came from
};

// `logits` is the window's logits slab, `row_floats` wide per row (the vocabulary). Each sequence is sampled from
// `plan.sample_rows[s]`, with the draw counter set to that row's absolute position.
inline WindowSampleResult sample_window_rows(const WindowRowPlan& plan, const std::span<const float> logits,
                                             const uint32_t row_floats,
                                             const std::span<const WindowSampleRequest> requests) {
    if (requests.size() != plan.sequences())
        throw std::invalid_argument("window sampling: one request per sequence is required");
    if (row_floats == 0) throw std::invalid_argument("window sampling: the vocabulary width is zero");
    if (logits.size() != size_t(plan.rows()) * row_floats)
        throw std::invalid_argument("window sampling: the logits slab does not match the plan's rows");
    WindowSampleResult result;
    result.tokens.reserve(plan.sequences());
    result.rows.reserve(plan.sequences());
    for (uint32_t sequence = 0; sequence < plan.sequences(); ++sequence) {
        const uint32_t row = plan.sample_rows[sequence];
        if (row >= plan.rows())
            throw std::invalid_argument("window sampling: sequence " + std::to_string(sequence) +
                                        " samples a row outside the window");
        const auto at = size_t(row) * row_floats;
        const std::span<const float> source(logits.data() + at, row_floats);
        for (const float value : source)
            if (!std::isfinite(value))
                throw std::invalid_argument("window sampling: sequence " + std::to_string(sequence) + " row " +
                                            std::to_string(row) + " has a non-finite logit");
        const WindowSampleRequest& request = requests[sequence];
        if (!request.bias.empty() && request.bias.size() != row_floats)
            throw std::invalid_argument("window sampling: sequence " + std::to_string(sequence) +
                                        " bias does not match the vocabulary");
        auto params = request.params;
        // Philox(seed, position): the same convention the scalar token loop uses, so a windowed step and a scalar step
        // with the same seed produce the same token.
        params.counter = plan.row_position[row];
        result.tokens.push_back(sample(source, request.history, params, request.bias));
        result.rows.push_back(row);
    }
    return result;
}
} // namespace strata::hpu
