#pragma once
#include "strata/core/geometry.hpp"
#include <cstdint>
namespace strata::hpu {
struct PrefillMemoryPlan {uint32_t rows=0;uint64_t host_bytes=0,hbm_bytes=0;};
struct CandidateMemoryPlan {uint32_t rows=0;uint64_t logits_bytes=0,host_bytes=0,hbm_bytes=0,head_pinned_bytes=0;};
// The same arithmetic as PrefillMemoryPlan, kept split by the graph that owns it so a refusal can name the term
// that broke the budget instead of only the total. Summing these fields is what plan_prefill_window returns.
struct PrefillWindowTerms {
    // Session current/next rows, layer residual/readback/FFN staging, PLE and MTP passthrough plus grouped
    // routing/expert CPU scratch. Charged once per layer because the layer-major path staged per layer.
    uint64_t staging_hbm=0,staging_allowance_hbm=0,staging_host=0;
    uint64_t hc_hbm=0,gdn_hbm=0,gdn_host=0,qsa_hbm=0,qsa_host=0,ple_hbm=0,ple_host=0;
};
// Additional host slab/sampling/history and native head slot/workspace reserve.
// Head pinned slots must fit the separately charged global pinned cap.
// Native layer graphs share
// plan_prefill_window(max(prompt_rows,candidate_rows)), charged only once.
CandidateMemoryPlan plan_candidate_window(const core::ModelGeometry&,uint32_t requested,uint32_t context);
// Conservative payload and compiler allowance for the layer-major grouped-MoE
// path. Actual allocations still obey global caps. No prompt*vocabulary slab.
PrefillMemoryPlan plan_prefill_window(const core::ModelGeometry&,uint32_t requested,uint32_t context);
// The same charge with its terms named, so a budget table can say which graph's staging a wider window makes
// expensive. rows is the window as admitted (already clamped to context and to the resident window cap).
PrefillWindowTerms plan_prefill_window_terms(const core::ModelGeometry&,uint32_t rows,uint32_t context);
// Largest window <=128/context whose incremental payload fits both remaining
// budgets, after all optional model features have been charged. Zero means
// token fallback; this is policy admission, not measured runtime workspace.
PrefillMemoryPlan select_prefill_window(const core::ModelGeometry&,uint32_t context,
                                      uint64_t host_available,uint64_t hbm_available);
}
