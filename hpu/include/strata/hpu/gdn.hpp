#pragma once
#include "strata/core/geometry.hpp"
#include "strata/hpu/recipe.hpp"
#include "strata/hpu/gdn_window_flow.hpp"

#include <string>
#include <utility>
#include <vector>

namespace strata::hpu {
// Same step contract/layout as include/strata/kernels/gdn.hpp, except separate
// state/output buffers for explicit ownership. q is normalized and scaled;
// k normalized; beta already sigmoid; gate is the pre-exp value.
std::unique_ptr<Recipe> build_gdn_step(const Device&, const core::ModelGeometry&);
void define_gdn_step(GraphBuilder&, const core::ModelGeometry&);
// Complete non-projection body for one token. Conv state is HPU time-major
// [taps-1,C], conv weights [taps,C] (transposed from the GGUF [C,taps]).
// Inputs: state,conv_state,qkv,alpha,raw_beta,z,conv_weights,norm,dt,ssm_a.
// Outputs: state_out,conv_state_out,out. All FP32. Debug mode additionally
// exposes activated,q,k,v,gate,beta,recurrent as persistent outputs.
void define_gdn_body(GraphBuilder&, const core::ModelGeometry&, bool debug_intermediates = false);
std::unique_ptr<Recipe> build_gdn_body(const Device&, const core::ModelGeometry&, bool debug_intermediates = false);
// BF16 MME projection reference block: x[1,H] has already been normalized by
// the outer layer. Five BF16 projection weights plus the FP32 body above.
// This is not CPU ActQ parity or a complete HC/residual Strata layer.
// merged (STRATA_HPU_MERGED_GEMM): the four input projections are one GEMM over w_qkvzab (see gdn.cpp).
void define_gdn_block(GraphBuilder&, const core::ModelGeometry&, bool merged = false);
std::unique_ptr<Recipe> build_gdn_block(const Device&, const core::ModelGeometry&);
// Bounded causal graph: four batched BF16 input projections and one batched
// output projection; the unchanged FP32 body is linked row-by-row. Only final
// state/conv_state are persistent. No input/output alias or tensor cache fence.
// row_states (STRATA_HPU_DFLASH verify): every row's states are persistent instead, as
// row_<i>_state_out / row_<i>_conv_state_out, so any accepted prefix can be committed.
void define_gdn_window(GraphBuilder&,const core::ModelGeometry&,uint32_t rows,bool row_states=false,bool merged=false);
void define_gdn_prefill_final(GraphBuilder&,const core::ModelGeometry&,uint32_t rows,bool merged=false);
// STRATA_HPU_GDN_CHUNKED: the recurrence of `rows` tokens (a multiple of kGdnChunk) as the chunked gated delta rule
// (src/gdn_chunked.cpp): per chunk and value head the intra-chunk system is solved on the MME and the state is read
// and written once per chunk instead of once per row. state {s, hv, s} ([key][head][value], tpc/gdn_step.c's
// layout), q/k {rows, hk, s}, v {rows, hv, s}, gate (log decay) / beta {rows, hv} -> next {s, hv, s}, recurrent
// {rows, hv, s}. The per-row chain's mathematics in a different summation order (not bit-identical). The graph
// reads the persistent constants gdn_chunked_constants() names (under the current scope); the caller sets them.
inline constexpr uint32_t kGdnChunk=64;
void define_gdn_chunked_recurrence(GraphBuilder&,uint64_t s,uint64_t hk,uint64_t hv,uint32_t rows,TensorId state,TensorId q,
                                   TensorId k,TensorId v,TensorId gate,TensorId beta,TensorId next,TensorId recurrent);
std::vector<std::pair<std::string,std::vector<float>>> gdn_chunked_constants();
// Whether define_gdn_prefill_final takes the chunked recurrence for `rows`.
bool gdn_prefill_chunked(uint32_t rows);
std::unique_ptr<Recipe> build_gdn_window(const Device&,const core::ModelGeometry&,uint32_t rows);
} // namespace strata::hpu
