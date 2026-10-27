#pragma once
// hpu/include/strata/hpu/spec_gate.hpp - upstream's draft-confidence gate, in one place so the CLI, the per-request
// key and the verify window cannot disagree.
//
// `--spec-min-p` (field at src/program/generate.cpp:609, default 0.0) says: a draft enters the verify window only
// while every draft before it - and itself - reaches this probability under the drafter. Upstream applies it in two
// distinct places, and both matter:
//   * the verify-window truncation: `T = 1; while (T < S_mtp && dprob[T-1] >= req_spec_min_p) ++T;`
//     (:12566-12568, and the same walk over the chain's own probabilities at :12041-12043, :12269, :12302);
//   * the MTP drafter's own extension loop (`if (use_mtp && o.spec_min_p > 0.0)`, :13755), which this port already
//     runs from ModelOptions::mtp_min_p (hpu/src/model.cpp:695).
// A value of 0 means NO gate at all - upstream tests it before cutting, so the default run is bounded only by the
// window and the stop tokens. The per-request key `spec_min_p=` is clamped into [0, 1] at :10697, and a request that
// does not name it keeps the start-up value (:10665).

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <span>

namespace strata::hpu {

// Upstream's clamp (:10697): out-of-range requests are pulled into [0, 1], not refused. A NaN is not a probability,
// so it becomes 0 = no gate rather than an silently-true comparison against every draft.
inline float clamp_spec_min_p(const double value) {
    if(!std::isfinite(value)) return 0.0f;
    return float(std::clamp(value, 0.0, 1.0));
}

// Upstream's gate test: the cut happens only when the operator raised it above zero (:12566, :13755).
inline bool spec_gate_armed(const float min_p) { return min_p > 0.0f; }

// Upstream's window walk (:12566-12568): `T = 1; while (T < S_mtp && dprob[T-1] >= gate) ++T;`. `T` COUNTS ROWS and
// row 0 is the token the target itself predicts, so the number of DRAFTS that join is `T - 1`: the first draft has to
// clear the gate on its own, and a draft below it - or one the drafter gave no confidence for - ends the window.
// With nothing accepted, upstream runs a one-row window, which is this port's ordinary token path. The returned count
// is therefore in [0, proposed], never the "+1 always" reading.
inline uint32_t gated_draft_count(const std::span<const float> probabilities,const uint32_t proposed,const float min_p) {
    if(proposed == 0) return 0;
    if(!spec_gate_armed(min_p)) return proposed;
    uint32_t kept = 0;
    while(kept < proposed && kept < probabilities.size() && probabilities[kept] >= min_p) ++kept;
    return kept;
}

}
