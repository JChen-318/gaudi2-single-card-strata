#pragma once
#include "strata/hpu/weights.hpp"
#include "strata/kernels/rope_scaling.hpp"
namespace strata::hpu {
struct RopeOptions {
    std::optional<strata::kernels::RopeScalingType> type;
    std::optional<double> factor,freq_base,freq_scale,orig_ctx,ext_factor,attn_factor,beta_fast,beta_slow;
};
// Explicit options > GGUF > original Strata defaults. Resolved once before
// any state/cache or recipe exists; None clears inactive scaling knobs.
strata::kernels::RopeScaling resolve_rope(const WeightTable&,const RopeOptions& = {});
}
