#pragma once
// hpu/include/strata/hpu/stage_stats.hpp - upstream's `--stats` report, limited to the terms this backend actually has.
//
// Upstream prints per-token wall clock, the six token host phases, the PLE io report, the CPU expert pool, its three
// phases, the blob read count and the dispatch statistics (src/program/generate.cpp:14080-14125). This backend has no
// per-phase timers inside the token loop (the Model owns that path) and no CPU worker pool phases, so printing those
// numbers would be invention. What it DOES have is: the generation result's timing and counts, the expert cache's
// counters, and the PLE reader's own io report. Those are printed with upstream's label shapes, and a term that has no
// data is omitted rather than printed as zero.
//
// Header-only and free of the SDK so a host test can pin the labels, the arithmetic and the omission rules.
#include <cstdint>
#include <string>
#include <vector>

namespace strata::hpu {
struct StageStats {
    double prompt_ms = 0, decode_ms = 0;
    uint64_t prompt_tokens = 0, generated = 0, reused = 0;
    uint64_t verify_windows = 0, drafts_proposed = 0, drafts_accepted = 0;
    uint64_t expert_hits = 0, expert_loads = 0, expert_evictions = 0, expert_failed = 0, expert_entries = 0;
    uint64_t expert_resident_bytes = 0, expert_peak_bytes = 0;
    uint64_t blobs_read = 0;
    std::string ple_io;   // PleInput::io_report(); empty when the model has no PLE table
    bool expert_cache_configured = false;
};

namespace stage_stats_detail {
inline std::string gib(const uint64_t bytes) {
    // Two decimals, GiB, the unit the rest of this port's reports use.
    char text[32];
    const double value = double(bytes) / double(1ull << 30);
    std::snprintf(text, sizeof text, "%.2f GiB", value);
    return text;
}
} // namespace stage_stats_detail

inline std::string format_stats(const StageStats& stats) {
    using stage_stats_detail::gib;
    std::vector<std::string> lines;
    auto add = [&lines](const char* label, const std::string& value) {
        // Upstream's `%-24s` column, so the report greps the same way.
        std::string line(label);
        if (line.size() < 24) line.append(24 - line.size(), ' ');
        lines.push_back(line + value);
    };
    char buffer[256];
    // The per-token term only exists once something was generated; dividing by zero would print inf and look like a
    // finding rather than an empty run.
    if (stats.generated > 0) {
        std::snprintf(buffer, sizeof buffer, "%.3f ms/token (WALL CLOCK: the whole token loop)", stats.decode_ms / double(stats.generated));
        add("  per token", buffer);
    } else {
        add("  per token", "(no token generated)");
    }
    std::snprintf(buffer, sizeof buffer, "%llu generated, %llu prompt, %llu reused from a parked prefix",
                  (unsigned long long)stats.generated, (unsigned long long)stats.prompt_tokens,
                  (unsigned long long)stats.reused);
    add("  tokens", buffer);
    std::snprintf(buffer, sizeof buffer, "%.1f ms prompt", stats.prompt_ms);
    add("  prompt", buffer);
    if (stats.verify_windows > 0) {
        std::snprintf(buffer, sizeof buffer, "%llu windows, %llu drafts proposed, %llu accepted",
                      (unsigned long long)stats.verify_windows, (unsigned long long)stats.drafts_proposed,
                      (unsigned long long)stats.drafts_accepted);
        add("  verify windows", buffer);
    }
    if (stats.expert_cache_configured) {
        std::snprintf(buffer, sizeof buffer, "%llu hits, %llu loads, %llu evictions, %llu failed, %llu entries, %s resident (%s peak)",
                      (unsigned long long)stats.expert_hits, (unsigned long long)stats.expert_loads,
                      (unsigned long long)stats.expert_evictions, (unsigned long long)stats.expert_failed,
                      (unsigned long long)stats.expert_entries, gib(stats.expert_resident_bytes).c_str(),
                      gib(stats.expert_peak_bytes).c_str());
        add("  expert cache", buffer);
    }
    if (stats.blobs_read > 0) {
        std::snprintf(buffer, sizeof buffer, "%llu blobs read", (unsigned long long)stats.blobs_read);
        add("  expert blobs", buffer);
    }
    if (!stats.ple_io.empty()) add("  PLE reader", stats.ple_io);
    std::string out;
    for (const auto& line : lines) {
        out += line;
        out += '\n';
    }
    return out;
}
} // namespace strata::hpu
