#pragma once
#include "strata/hpu/recipe.hpp"
namespace strata::hpu {
struct HcConfig {
    uint32_t hidden=2560,streams=4,rank=320;
    float epsilon=1e-6f;
    void validate() const;
};
// Inputs residual[HC,H] F32, norm[HC,H] multiplicative F32, BF16 down/up
// and (except final_mix) inject weights. Outputs mixed[H] F32, inject[HC] F32.
// Runtime inverse_streams[1] must equal 1/HC. Activations round to BF16 at
// both GEMMs; accumulations and nonlinear results remain FP32.
// merged (STRATA_HPU_MERGED_GEMM): down and inject are one GEMM over w_down_inject [rank + HC, HC*H].
// codes: under STRATA_HPU_HC_DECODE_TPC + HC_MIX_Q8 the mix also writes "x_codes"/"x_scales" (the FFN read).
void define_hc_read(GraphBuilder&,const HcConfig&,bool final_mix=false,bool merged=false,bool codes=false);
// The BF16 copy of a decode read's "mixed" (`row`, its F32 [1, H] view) in scope `scope`: the read's own
// "mixed_bf16" under STRATA_HPU_HC_DECODE_TPC, otherwise a cast of `row` into a new tensor `name`.
TensorId hc_read_bf16(GraphBuilder&,const std::string& scope,TensorId row,const std::string& name,uint64_t hidden);
std::unique_ptr<Recipe> build_hc_read(const Device&,const HcConfig&,bool final_mix=false);
void define_hc_write(GraphBuilder&,const HcConfig&);
std::unique_ptr<Recipe> build_hc_write(const Device&,const HcConfig&);
// Multi-token HC uses [T,HC,H] residuals, one shared [HC,H] gamma and
// multi-row MME down/up/inject projections. T is bounded to1..128.
// The BF16 copy of the window read's "mixed" in scope `scope` (e.g. "attn."): its fused output under
// STRATA_HPU_HC_FUSED, otherwise a cast into a new tensor `name`.
TensorId hc_read_window_bf16(GraphBuilder&,const std::string& scope,const std::string& name,uint64_t rows,uint64_t hidden);
void define_hc_read_window(GraphBuilder&,const HcConfig&,uint32_t rows,bool final_mix=false,bool merged=false);
void define_hc_write_window(GraphBuilder&,const HcConfig&,uint32_t rows,bool internal=false);
// PLE takes already gathered embedding[1,H] BF16 and hidden[HC,H] F32.
// HPU history[9,HC*H] and conv_weights[4,HC*H] are time-major (transposed
// from native channel-major/GGUF). All three norm weights are multiplicative.
// Outputs out[HC,H] and history_out[9,HC*H]. Optional debug exports intermediates.
void define_ple_block(GraphBuilder&,const HcConfig&,bool debug=false);
std::unique_ptr<Recipe> build_ple_block(const Device&,const HcConfig&,bool debug=false);
// Two multi-row BF16-input/F32-output projections followed by the unchanged
// per-row native PLE body. Distinct causal history tensors, final state only.
// row_history (STRATA_HPU_DFLASH verify): every row's history is persistent as row_<i>_history_out instead.
// resident: a resident verify-chain window (up to kResidentWindowRows rows) that keeps only the final history.
void define_ple_window(GraphBuilder&,const HcConfig&,uint32_t rows,bool row_history=false,bool resident=false);
std::unique_ptr<Recipe> build_ple_window(const Device&,const HcConfig&,uint32_t rows);
}
