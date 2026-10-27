// hpu/tests/stage_stats.cpp - host evidence for the `--stats` report this backend can honestly produce.
//
// Upstream's --stats prints per-token wall clock, six token host phases, the PLE io report, the CPU pool and its three
// phases, blob reads and dispatch statistics (src/program/generate.cpp:14080-14125). This backend has no per-phase
// timers inside the token loop and no CPU pool phases, so those are NOT printed; what is printed is the generation
// result, the expert cache counters and the PLE reader's own report. The test pins the labels, the arithmetic, and the
// omission rules (a term without data must be absent, not a zero).
#include "strata/hpu/stage_stats.hpp"

#include <iostream>
#include <string>

namespace {
using strata::hpu::format_stats;
using strata::hpu::StageStats;

int failures = 0;
void check(const bool ok, const std::string& what) {
    if (!ok) {
        ++failures;
        std::cout << "FAIL " << what << "\n";
    }
}
bool has(const std::string& text, const std::string& needle) {
    return text.find(needle) != std::string::npos;
}
int lines(const std::string& text) {
    int count = 0;
    for (const char c : text) if (c == '\n') ++count;
    return count;
}
} // namespace

int main() {
    // 1. a full report: the arithmetic and the unit conversions, with upstream's label column
    {
        StageStats stats;
        stats.prompt_ms = 120.0;
        stats.decode_ms = 250.0;
        stats.generated = 50;
        stats.prompt_tokens = 32;
        stats.reused = 8;
        stats.verify_windows = 4;
        stats.drafts_proposed = 12;
        stats.drafts_accepted = 7;
        stats.expert_cache_configured = true;
        stats.expert_hits = 100;
        stats.expert_loads = 20;
        stats.expert_evictions = 3;
        stats.expert_failed = 1;
        stats.expert_entries = 16;
        stats.expert_resident_bytes = 3ull << 30;                       // 3.00 GiB
        stats.expert_peak_bytes = (3ull << 30) + (1ull << 29);          // + 0.50 GiB
        stats.blobs_read = 20;
        stats.ple_io = "PLE direct, cache 1048576 rows";
        const auto text = format_stats(stats);
        // The values are asserted whole; the gutter is asserted once, separately, so this test does not just mirror
        // the formatter's own padding arithmetic.
        check(has(text, "5.000 ms/token (WALL CLOCK: the whole token loop)"), "per-token ms is decode_ms/generated");
        check(has(text, "50 generated, 32 prompt, 8 reused from a parked prefix"), "the token counts are reported");
        check(has(text, "120.0 ms prompt"), "the prompt time is reported");
        check(has(text, "4 windows, 12 drafts proposed, 7 accepted"), "the verify window line");
        check(has(text, "100 hits, 20 loads, 3 evictions, 1 failed, 16 entries, 3.00 GiB resident (3.50 GiB peak)"),
              "the expert cache line carries both memory figures");
        check(has(text, "20 blobs read"), "the blob counter");
        check(has(text, "PLE direct, cache 1048576 rows"), "the PLE io report");
        check(lines(text) == 7, "seven lines for the seven terms that have data");
        // every label sits in the same 24-column gutter, so the report greps like upstream's
        for (const auto& label : {std::string("  per token"), std::string("  tokens"), std::string("  expert cache"),
                                  std::string("  PLE reader")})
            check(has(text, label + std::string(24 - label.size(), ' ')), "the label gutter: " + label);
    }

    // 2. no token generated: no division, and the line says so instead of printing inf
    {
        StageStats stats;
        const auto text = format_stats(stats);
        check(has(text, "(no token generated)"), "an empty run says so instead of printing inf");
        check(!has(text, "inf") && !has(text, "nan"), "no inf/nan can appear");
        check(lines(text) == 3, "only the always-present terms are printed (per token, tokens, prompt)");
        check(!has(text, "expert cache") && !has(text, "PLE reader") && !has(text, "verify windows"),
              "terms without data are omitted, not printed as zero");
    }

    // 3. the terms that only exist when their feature is on
    {
        StageStats stats;
        stats.generated = 1;
        stats.decode_ms = 1.5;
        stats.prompt_tokens = 1;
        const auto text = format_stats(stats);
        check(has(text, "1.500 ms/token"), "a single token still divides correctly");
        check(!has(text, "expert cache"), "an unconfigured expert cache contributes no line");
        check(!has(text, "expert blobs"), "no blob line without a blob read");
    }

    if (failures != 0) {
        std::cout << "FAILED stage-stats checks: " << failures << "\n";
        return 1;
    }
    std::cout << "PASS stage stats: the per-token figure is decode_ms/generated and refuses to divide by zero, the "
                 "token/prompt/verify/expert-cache/bulk/PLE lines carry the numbers this backend actually has, every "
                 "label sits in upstream's 24-column gutter, and a term without data is omitted instead of printed as "
                 "zero; no device\n";
    return 0;
}
