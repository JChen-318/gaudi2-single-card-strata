// hpu/tests/spec_gate.cpp - host evidence for upstream's draft-confidence gate: the law, the per-request key, and
// the verify-window truncation the port did not have before (strata/hpu/spec_gate.hpp).
//
// Upstream truncates a window on the drafter's OWN probabilities: `T = 1; while (T < S_mtp && dprob[T-1] >=
// req_spec_min_p) ++T;` (src/program/generate.cpp:12566-12568, the same walk at :12041-12043). `T` counts ROWS and
// row 0 is the target's own token, so `T-1` drafts join - a first draft below the gate leaves a one-row window. The
// field default is 0.0 = no gate at all (:609) and the request key is clamped into [0,1] (:10697). `aux_cpus=0|1`
// (:10698) picks where this request's non-pool threads run, with -1 = the start-up value (:10666), and `pcie_frac`
// (:10696) asks for the host->device transfer split this single-card backend has no pipeline for, so it is refused
// BY NAME rather than silently ignored.
#include "strata/hpu/spec_gate.hpp"
#include "strata/hpu/generation.hpp"
#include "strata/hpu/protocol.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace strata::hpu;

int failures = 0;
void check(const bool ok, const std::string& what) {
    if (!ok) {
        ++failures;
        std::cout << "FAIL " << what << "\n";
    }
}
template <class F>
bool refused(F&& call, const std::string& needle) {
    try {
        call();
    } catch (const std::exception& error) {
        return std::string(error.what()).find(needle) != std::string::npos;
    }
    return false;
}

constexpr uint32_t VOCAB = 5, CONTEXT = 16;
constexpr float HIGH = 0.95f, MID = 0.30f, LOW = 0.05f;

// The gate law itself, against the upstream walk transcribed by hand.
void gate_law() {
    check(clamp_spec_min_p(0.7) == 0.7f, "a value inside the domain survives, as upstream's clamp leaves it");
    check(clamp_spec_min_p(1.5) == 1.0f && clamp_spec_min_p(-0.5) == 0.0f,
          "an out-of-range request is clamped into [0,1], not refused (generate.cpp:10697)");
    check(clamp_spec_min_p(std::numeric_limits<double>::quiet_NaN()) == 0.0f,
          "a NaN is not a probability: it becomes no gate instead of comparing against every draft");
    check(!spec_gate_armed(0.0f) && spec_gate_armed(0.01f),
          "0 means no gate at all, which is upstream's default (:609) and the whole reason the cut is tested first");

    const std::vector<float> confidences{HIGH, MID, LOW};
    const std::span<const float> probabilities(confidences);
    check(gated_draft_count(probabilities, 3, 0.0f) == 3, "an unarmed gate keeps every proposed draft");
    check(gated_draft_count(probabilities, 3, 0.5f) == 1,
          "only the drafts up to the first confidence below the gate join (upstream's T-1, :12567)");
    check(gated_draft_count(probabilities, 3, HIGH) == 1, "a confidence EQUAL to the gate still joins (>= upstream)");
    check(gated_draft_count(probabilities, 3, 0.97f) == 0,
          "a first draft below the gate leaves a one-row window: upstream's T stays 1, it does not keep a free draft");
    check(gated_draft_count({}, 3, 0.5f) == 0,
          "a drafter that reported no confidence cannot claim a row, which is upstream's `avail` guard read conservatively");
    check(gated_draft_count(probabilities, 0, 0.5f) == 0, "no proposals means no window rows, gate or not");
    check(gated_draft_count(probabilities, 2, 0.5f) == 1, "the count never exceeds what was proposed");
    check(gated_draft_count(std::vector<float>{HIGH, HIGH}, 2, HIGH) == 2, "every draft at or above the gate joins");
}

// A scripted target whose chain is deterministic, plus a drafter that continues it and reports confidences. The
// round trace is what makes the assertions discriminative: the WINDOW SIZES differ while the emitted chain must not.
struct Trace {
    std::vector<uint32_t> asked;                            // the `maximum` each round proposed against
    std::vector<uint32_t> confidence_spans;                 // how many confidences the engine asked for
    std::vector<std::vector<int32_t>> rounds;                   // tokens emitted per window
};
struct Backend {
    std::shared_ptr<Trace> trace = std::make_shared<Trace>();
    GenerationOperations operations() {
        GenerationOperations op;
        op.reset = [] {};
        op.decode = [](const int32_t token, const std::function<bool()>& cancelled) {
            if (cancelled && cancelled()) throw Cancelled();
            std::vector<float> logits(VOCAB, -10.0f);
            logits[size_t((token + 1) % int32_t(VOCAB))] = 10.0f;
            return logits;
        };
        op.checkpoint = [] { return std::function<void()> { [] {} }; };
        op.draft = [trace = trace](const DraftContext& context, const uint32_t maximum,
                                    const std::function<bool()>&) {
            const float reported[3] = {HIGH, MID, LOW};
            for (uint32_t row = 0; row < maximum && row < context.confidence.size(); ++row)
                context.confidence[row] = reported[std::min<size_t>(row, 2)];
            trace->asked.push_back(maximum);
            trace->confidence_spans.push_back(uint32_t(context.confidence.size()));
            trace->rounds.emplace_back();                     // every draft call opens the window it proposes for
            std::vector<DraftRow> rows;
            const int32_t last = context.history.empty() ? 0 : context.history.back();
            for (uint32_t row = 0; row < maximum; ++row)
                rows.push_back({int32_t((last + 1 + int32_t(row)) % int32_t(VOCAB)), {}, {}});
            return rows;
        };
        return op;
    }
};

struct Run { GenerationResult result; std::shared_ptr<Trace> trace; };
Run execute(const float startup_gate, const float request_gate) {
    GenerationOptions options;
    options.max_draft = 3;
    options.cache_slots = 0;
    options.cache_bytes = 1u << 20;
    options.spec_min_p = startup_gate;
    Backend backend;
    auto trace = backend.trace;
    Generation generation(VOCAB, CONTEXT, {}, backend.operations(), options);
    auto request = parse_generation_request("GEN 6 0,1", VOCAB, CONTEXT);
    request.spec_min_p = request_gate;
    GenerationEvents events;
    events.token = [trace](const int32_t token) {
        if (trace->rounds.empty()) trace->rounds.emplace_back();
        trace->rounds.back().push_back(token);
    };
    return {generation.run(request, events), trace};
}
std::vector<int32_t> flattened(const Run& run) {
    std::vector<int32_t> all;
    for (const auto& round : run.trace->rounds) all.insert(all.end(), round.begin(), round.end());
    return all;
}
std::string shape(const Run& run) {
    std::string text;
    for (const auto& round : run.trace->rounds) text += std::to_string(round.size()) + " ";
    return text;
}

void window_is_truncated() {
    const auto ungated = execute(0.0f, -1.0f);
    const auto startup = execute(0.5f, -1.0f);
    const auto requested = execute(0.0f, 0.5f);
    const auto stopped = execute(0.0f, 0.97f);
    check(ungated.result.generated == 6 && startup.result.generated == 6 && requested.result.generated == 6 &&
              stopped.result.generated == 6,
          "every gate setting still emits max_new tokens - the gate is a policy, not a different decoder");
    check(flattened(ungated) == flattened(startup) && flattened(ungated) == flattened(requested) &&
              flattened(ungated) == flattened(stopped),
          "and it emits THE SAME chain: gating changes window sizes, never the tokens");
    check(shape(ungated) == "4 2 ", "ungated: 3 accepted drafts plus the target row, then the remainder (" + shape(ungated) + ")");
    check(shape(startup) == "2 2 2 ", "the gate cuts each window to the drafts at or above it (" + shape(startup) + ")");
    check(shape(requested) == shape(startup), "the request key and the start-up value reach the same law (:10665)");
    check(shape(stopped) == "1 1 1 1 2 ",
          "a gate no draft clears gives one-row windows; the last token comes from the ordinary round that proposes "
          "nothing (maximum 0), which is why it joins the previous batch (" + shape(stopped) + ")");
    check(ungated.trace->confidence_spans == std::vector<uint32_t>({0, 0}),
          "an unarmed gate asks the drafter for no confidences at all");
    check(startup.trace->asked.size() == startup.trace->confidence_spans.size() &&
              std::equal(startup.trace->asked.begin(), startup.trace->asked.end(), startup.trace->confidence_spans.begin()),
          "an armed gate asks for exactly the drafter confidences it will read (:12567 walks dprob)");
    check(ungated.result.verify_windows < startup.result.verify_windows,
          "the truncation shows up in the window count: the same chain costs more windows");
    check(refused([&] { GenerationOptions options; options.spec_min_p = 1.5f;
                        (void)Generation(VOCAB, CONTEXT, {}, Backend().operations(), options); },
                   "invalid spec draft gate"),
            "a start-up gate outside [0,1] is refused instead of silently compared");
}

void request_keys() {
    const auto plain = parse_generation_request("GEN 4 0,1", VOCAB, CONTEXT);
    check(plain.spec_min_p == -1.0f && plain.aux_cpus == -1,
          "an absent key keeps upstream's sentinel meaning: the start-up value, not a guess");
    const auto gated = parse_generation_request("GEN 4 spec_min_p=0.7 aux_cpus=1 0,1", VOCAB, CONTEXT);
    check(gated.spec_min_p == 0.7f && gated.aux_cpus == 1, "both keys land on the request");
    const auto clamped = parse_generation_request("GEN 4 spec_min_p=9 0,1", VOCAB, CONTEXT);
    check(clamped.spec_min_p == 1.0f, "an out-of-range request gate is clamped, as upstream clamps it");
    const auto off = parse_generation_request("GEN 4 aux_cpus=0 0,1", VOCAB, CONTEXT);
    check(off.aux_cpus == 0, "aux_cpus=0 is a placement off for this request, distinct from absent");
    check(refused([&] { (void)parse_generation_request("GEN 4 spec_min_p=0.7 spec_min_p=0.2 0,1", VOCAB, CONTEXT); },
                   "duplicate sampling key"),
            "a key named twice is refused rather than last-one-wins");
    check(refused([&] { (void)parse_generation_request("GEN 4 pcie_frac=0.5 0,1", VOCAB, CONTEXT); },
                   "pcie_frac is not implemented on a single-card backend"),
            "the transfer-split key is refused BY NAME with its reason, not swallowed as an unknown key");
    check(refused([&] { (void)parse_generation_request("GEN 4 spec_min_p=x 0,1", VOCAB, CONTEXT); },
                   "invalid request number"),
            "a non-numeric gate is refused by the port's number law");
}

}

int main() {
    try {
        gate_law();
        window_is_truncated();
        request_keys();
    } catch (const std::exception& error) {
        std::cout << "FAIL spec-gate check threw before finishing: " << error.what() << "\n";
        return 1;
    }
    if (failures) {
        std::cout << "spec gate: " << failures << " failed\n";
        return 1;
    }
    std::cout << "PASS draft gate law, verify-window truncation by start-up and per-request value, request keys with upstream's clamps\n";
    return 0;
}
