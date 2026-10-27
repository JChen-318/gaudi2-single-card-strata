#pragma once
#include "strata/core/geometry.hpp"
#include "strata/hpu/recipe.hpp"

namespace strata::hpu {
// BF16 reference execution for a routed/shared expert. The source GGUF decoder
// and CPU activation-quantization parity are separate integration work.
// Bind x[T,H], w_gate[FF,H], w_up[FF,H], w_down[H,FF], out[T,H].
// Router weights are NOT applied here: the combine stage owns them.
std::unique_ptr<Recipe> build_bf16_expert(const Device&, const core::ModelGeometry&, uint64_t tokens);
void define_bf16_expert(GraphBuilder&,const core::ModelGeometry&,uint64_t tokens);
// merged: "w_router" is [experts+1, h] with the shared gate as its last row (MoeBlock under MERGED_GEMM).
void define_moe_router(GraphBuilder&,const core::ModelGeometry&,uint64_t tokens,bool merged=false);
std::unique_ptr<Recipe> build_moe_router(const Device&,const core::ModelGeometry&,uint64_t tokens);
} // namespace strata::hpu
