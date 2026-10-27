#pragma once
#include "strata/hpu/weights.hpp"
#include <array>
namespace strata::hpu {
// Original NVFP4 .scale tensors: one positive finite F32 per routed expert,
// or one value for a shared expert. Other projections use exactly 1.
std::array<float,3> decode_expert_scales(std::array<uint32_t,3> types,
    const std::array<std::span<const uint8_t>,3>& scale_payloads,uint32_t expert,uint32_t count);
std::array<float,3> expert_global_scales(const WeightTable&,uint32_t layer,int32_t expert);
void validate_expert_global_scales(const WeightTable&,uint32_t layer);
// Assemble original native expert bytes gate|up|down[|sg,su,sd,0]. One bounded
// cache miss owns this vector; global scales never fold into weight bytes/up.
std::vector<uint8_t> native_expert_blob(const std::array<WeightSlice,3>&,uint64_t up_offset,
    uint64_t down_offset,uint64_t total_bytes,uint64_t tail_offset,std::array<float,3> scales);
}
