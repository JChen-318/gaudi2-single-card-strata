#pragma once
// hpu/include/strata/hpu/ple_mode.hpp - upstream's --no-ple diagnostic ablation (src/program/generate.cpp:1944,
// :2635-2645, :3434).
//
// The PLE table supplies layer 1's per-token embedding. --no-ple runs the model WITHOUT that input: upstream
// allocates no PLE buffers, prints that the tokens are not this model's, and calls it an A/B diagnostic. It is
// never a normal inference mode, and it cannot be combined with --ple-gguf (the table would have nothing to feed).
// This header is the single admission law for the CLI and Model, so the two cannot disagree.
#include <stdexcept>
#include <string>
#include <string_view>

namespace strata::hpu {
struct PleMode {
    bool enabled = true;
    std::string external;   // the external table path; empty in ablation mode
};

inline PleMode resolve_ple_mode(const bool no_ple,const std::string_view ple_gguf) {
    if (no_ple && !ple_gguf.empty())
        throw std::invalid_argument("--no-ple and --ple-gguf are mutually exclusive");
    return no_ple ? PleMode{false,{}} : PleMode{true,std::string(ple_gguf)};
}

// Upstream's two-line warning at :3434-3436, unchanged: a run without PLE is only an A/B arm.
inline const char* ple_off_notice() {
    return "strata generate: PLE OFF by explicit --no-ple diagnostic request.\n"
           "  The tokens below are NOT this model's; this is only useful for A/B measurement.\n";
}
} // namespace strata::hpu
