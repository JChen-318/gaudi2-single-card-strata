#pragma once
#include "strata/hpu/verify.hpp"
namespace strata::hpu {
struct MtpSamplingOptions {bool coupled=false,gate_pick=true;float temperature_scale=1,min_temperature=0;};
struct MtpPick {DraftRow row;float confidence=0;};
MtpPick pick_mtp_draft(std::span<const float> logits,std::span<const int32_t> history,
    strata::kernels::SamplerParams,std::span<const float> bias,VerifyMode,MtpSamplingOptions={});
}
