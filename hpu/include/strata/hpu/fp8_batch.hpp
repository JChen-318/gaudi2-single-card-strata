#pragma once
#include "strata/hpu/recipe.hpp"
namespace strata::hpu {
// BF16 activations, lossless raw U4->FP8 weights, scaled hi (+lo/16)
// activations stacked inside each batch member. Does not remove IQ4->U4
// requantization error. terms is 1 (diagnostic) or 2 (hi/lo).
void define_u4_fp8_batch(GraphBuilder&,TensorId a,TensorId codes,TensorId out,
                         uint64_t batch,uint64_t rows,uint64_t k,uint64_t n,
                         uint32_t terms,const std::string& prefix);
}
