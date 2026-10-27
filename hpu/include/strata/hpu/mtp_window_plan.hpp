#pragma once
#include "strata/core/geometry.hpp"
#include "strata/hpu/prefill_plan.hpp"
#include <cstdint>
#include <functional>
namespace strata::hpu {
struct MtpWindowMemoryPlan {uint32_t rows=0;uint64_t host_bytes=0,hbm_bytes=0,pinned_bytes=0;};
// Additional dynamic FC/HC/attention slots, owned host staging and compiler
// allowance. Private weights/KV/expert cache/singleton recipes are separate.
MtpWindowMemoryPlan plan_mtp_window(const core::ModelGeometry&,uint32_t requested,uint32_t context,uint32_t attention_window=0);
struct ModelWindowMemoryPlan {PrefillMemoryPlan target;MtpWindowMemoryPlan mtp;uint64_t additional_host_bytes=0;};
using WindowHostCost=std::function<uint64_t(uint32_t)>;
// Largest target window whose target+MTP+optional diagnostic costs fit after
// persistent features are charged. The callback prices each candidate (including
// scalar fallback); the caller charges/allocates that cost once after selection.
// minimum_rows preserves an explicitly requested verify window. No feasible
// candidate throws before device creation instead of disabling requested features.
ModelWindowMemoryPlan select_model_prefill_window(const core::ModelGeometry&,uint32_t context,uint64_t host_available,uint64_t hbm_available,uint64_t pinned_limit,bool mtp,uint32_t attention_window=0,const WindowHostCost& additional_host={},uint32_t minimum_rows=0);
}
