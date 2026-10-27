#pragma once
#include <cstdint>
#include <array>
#include <functional>
#include <span>
#include <vector>
namespace strata::hpu {
using ExpertProjection=std::function<std::vector<float>(uint32_t role,std::span<const float>)>;
// Shared production flow: F32 gate/up -> unrounded SiLU*up -> down's own
// activation quantization. Projection callbacks return unweighted values.
// Optional NVFP4 global scales apply to each projection's F32 result. In
// particular down's scale applies AFTER its own activation quantization/dot.
std::vector<float> execute_packed_expert(std::span<const float> rows,uint32_t hidden,uint32_t intermediate,const ExpertProjection&,std::array<float,3> output_scales={1,1,1});
}
