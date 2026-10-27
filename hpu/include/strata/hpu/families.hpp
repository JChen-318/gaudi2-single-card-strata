#pragma once
// hpu/include/strata/hpu/families.hpp - which GGUF architectures THIS build runs, said once.
//
// Upstream's `--families` (src/program/generate.cpp:1811-1817, in the pre-scan loop before any model path exists)
// prints the architectures the build accepts and exits, because its setup step asks before installing one. Its list
// is compile-time: `qwen4exp`, plus `qwen35moe` only when STRATA_HAVE_QWEN35 was defined.
//
// This backend reads a GGUF's `general.architecture` and refuses anything else (hpu/src/weights.cpp used to carry
// that literal and its own sentence), so the printed list and the reader's check were two places that could disagree -
// a table that claims a family the loader would reject is worse than no table. They are one place now: the check and
// the report both come from supported_families(), and an unsupported file is told WHICH families this build runs.

#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace strata::hpu {

// The families this build can load. A second one is added HERE and nowhere else, so `--families` cannot go stale.
inline std::vector<std::string> supported_families() {
    return {"qwen4exp"};
}

// Upstream's `strata --families` line (:1813/1815): the names space-separated, on one line. The separator rule is
// visible only with two names, so it lives in a helper the host test can hand a two-name list; this build itself has one.
inline std::string join_families(const std::vector<std::string>& families) {
    std::ostringstream text;
    bool first = true;
    for (const auto& family : families) {
        if (!first) text << ' ';
        text << family;
        first = false;
    }
    return text.str();
}

inline std::string families_line() { return join_families(supported_families()); }

inline void require_supported_family(const std::string& architecture) {
    for (const auto& family : supported_families())
        if (family == architecture) return;
    std::ostringstream message;
    message << "HPU model catalog requires one of the architectures this build runs: " << families_line()
            << " (this file declares '" << architecture << "')";
    throw std::invalid_argument(message.str());
}

}
