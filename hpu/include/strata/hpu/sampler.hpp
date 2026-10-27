#pragma once
#include "strata/kernels/sampler.hpp"
#include <span>
#include <vector>
namespace strata::hpu {
struct SampleDistribution {std::vector<int32_t> ids;std::vector<double> probabilities;};
// Same chain as Strata: bias -> penalties -> top-k (<=64) -> top-p -> min-p
// -> temperature -> Philox/Gumbel pick. All pointers here are host values.
// SamplerParams::logit_bias must be null; pass bias as a checked span instead.
SampleDistribution sampling_distribution(std::span<const float> logits,std::span<const int32_t> history,
    const strata::kernels::SamplerParams&,std::span<const float> bias={});
int32_t sample(std::span<const float> logits,std::span<const int32_t> history,
    const strata::kernels::SamplerParams&,std::span<const float> bias={});
float philox_uniform(uint64_t seed,uint64_t counter);
}
