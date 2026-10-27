#pragma once
#include <cstdint>
namespace strata::hpu::abi {
// The widest resident prefill window, and the MoE slot budget that bounds it (rows * per_row). Both live here
// because glue.cpp's dispatch checks and the host-side guards must agree: the window kernels loop over rows and
// their index spaces do not depend on the row count, so raising the window is one edit in each place rather than
// a per-kernel rewrite. 8192 rows need 90112 slots, hence the budget below (2048 rows needed 22528): a wider window
// spreads one dequantization of every expert's IQ4 weights (prefill MoE, MOE_MME) over more tokens.
inline constexpr uint32_t kMaxWindowRows=8192;
inline constexpr uint32_t kMaxMoeSlots=90112;
struct RowNorm {int32_t width,rows,shared_gamma;float epsilon;};
struct Fp8Encode {int32_t columns,rows;};
struct Fp8Batch {int32_t columns,rows,terms;};
static_assert(sizeof(Fp8Batch)==12);
static_assert(sizeof(Fp8Encode)==8);
// strata_fp8_gemv_f32: `rows` x `columns` FP8 codes + per-row F32 scales per expert; x shared (one row) or per expert.
struct Fp8Gemv {int32_t columns,rows,shared_x;};
static_assert(sizeof(Fp8Gemv)==12);
// STRATA_HPU_RESIDENT_IQ4 kernels (strata/hpu/resident_experts.hpp).
struct Iq4Gemv {int32_t columns,rows,x_group,parts,slots,period;};
struct Q8Quant {int32_t columns,rows;};
struct MoeActQuant {int32_t ff,slots,parts;}; // strata_moe_act_packed_f32 (STRATA_HPU_MOE_ACT_FUSED)
struct HcWriteMoe {int32_t width,streams,rows,per_row;}; // strata_hc_write_moe_f32 (STRATA_HPU_MOE_TAIL_FUSED)
static_assert(sizeof(HcWriteMoe)==16);
static_assert(sizeof(MoeActQuant)==12);
struct RouteTopk {int32_t experts,top_k,slots,rows;};
struct RouteShare {int32_t slots,period;}; // strata_route_share_i32 (STRATA_HPU_DFLASH verify windows)
// STRATA_HPU_MOE_GROUP: slot order grouped by routed expert (strata_route_group_starts/order_f32).
struct RouteGroup {int32_t experts,slots;};
// STRATA_HPU_RESIDENT_U4 (strata/hpu/u4_moe.hpp): the grouped-MoE MME path's device nodes.
// strata_u4_moe_bucket_f32: the window's routed slots bucketed per expert, shared slots skipped. `place` maps a
// slot back to its bucket position (stale entries are detected by bucket[place[slot]] != slot).
struct U4MoeBucket {int32_t experts,rows,slots,per_row,top_k;};
// strata_u4_moe_gather_f32: BF16 A operand {batch, rows, group} and the per-(batch,row) input sums.
struct U4MoeGather {int32_t groups,group,rows,per_row;};
// strata_u4_moe_gu_epi_f32: affine epilogue + SiLU(gate)*up -> BF16 down input and its column-group sums.
struct U4MoeGuEpi {int32_t experts,groups,ff,rows,width,down_groups;};
// strata_u4_moe_down_epi_f32: affine epilogue x route weight -> expert_rows[slot].
struct U4MoeDownEpi {int32_t experts,groups,n,rows,width,apply_route;};
// strata_u4_moe_join_f32: routed rows (sparse) + the shared expert's rows -> expert_rows[slot], zero outside.
struct U4MoeJoin {int32_t per_row,top_k,bucket_len;};
static_assert(sizeof(U4MoeBucket)==20 && sizeof(U4MoeGather)==16 && sizeof(U4MoeJoin)==12);
static_assert(sizeof(U4MoeGuEpi)==24 && sizeof(U4MoeDownEpi)==24);
// STRATA_HPU_MOE_MME (strata/hpu/moe_mme.hpp): strata_iq4_expand_bf16 turns resident IQ4 rows (resident_experts.hpp)
// of `rows` x `columns` into BF16 weights {rows, columns, experts}; strata_moe_mme_join_f32 reuses U4MoeJoin.
struct Iq4Expand {int32_t rows,columns,variant,first;}; // variant: tpc/iq4_expand_bf16.c (0 = production)
static_assert(sizeof(Iq4Expand)==16);
// strata_moe_mme_act_bf16: SiLU(gate)*up from a variant-4 gate/up product (tpc/moe_mme_act.c).
struct MoeMmeAct {int32_t ff,use_totals,first;};
static_assert(sizeof(MoeMmeAct)==12);
// strata_moe_mme_count/prefix/scatter_f32: the counting-sort bucketing (rows = bucket rows per expert).
struct MoeMmeRoute {int32_t experts,slots,per_row,top_k,rows;};
// strata_moe_mme_place_f32 (STRATA_HPU_MOE_PLACE): MoeMmeRoute plus the bucket index a shared slot scatters to.
struct MoeMmePlace {int32_t experts,slots,per_row,top_k,rows,trash;};
// STRATA_HPU_QSA_DECODE_SELECT: tpc/qsa_decode_blocks.c and tpc/qsa_gather_blocks_fp8.c.
struct QsaDecodeBlocks {int32_t blocks,heads,block;};
struct QsaGatherBlocks {int32_t heads,cells,top,block,width_sel;};
// Their verify-window forms (tpc/qsa_window_blocks.c, tpc/qsa_gather_window_fp8.c): a row axis.
struct QsaWindowBlocks {int32_t blocks,heads,block,rows;};
struct QsaGatherWindow {int32_t heads,cells,top,block,width_sel,rows;};
struct QsaVerifySoftmax {int32_t rows,heads,group,cells;}; // strata_qsa_verify_softmax_f32 (STRATA_HPU_QSA_VERIFY_BF16)
// strata_moe_mme_gather_f32: the routed GEMM A operand in BF16 from the window rows.
struct MoeMmeGather {int32_t hidden,rows,per_row,first;};
static_assert(sizeof(MoeMmeRoute)==20 && sizeof(MoeMmeGather)==16);
// strata_moe_mme_tilemap_f32: the tile -> expert map of a window with `tiles` tiles of `rows` rows.
struct MoeMmeTiles {int32_t experts,tiles,rows;};
static_assert(sizeof(MoeMmeTiles)==12);
// strata_moe_mme_join_fp8_f32 (STRATA_HPU_MOE_FP8): U4MoeJoin plus the tile rows, to find a routed row's tile.
struct MoeMmeJoinFp8 {int32_t per_row,top_k,bucket_len,rows;};
static_assert(sizeof(MoeMmeJoinFp8)==16);
static_assert(sizeof(RouteGroup)==8);
static_assert(sizeof(RouteShare)==8);
static_assert(sizeof(Iq4Gemv)==24 && sizeof(Q8Quant)==8 && sizeof(RouteTopk)==16);
// STRATA_HPU_QSA_DEVICE (strata_qsa_prep_f32).
struct QsaPrep {int32_t width,query_heads,kv_heads,rotary;float epsilon;int32_t tokens;};
// STRATA_HPU_QSA_DEVICE_SELECT: tpc/qsa_index_prep.c, tpc/qsa_index_cells.c, tpc/qsa_select_mask.c.
struct QsaIndexPrep {int32_t width,heads,rotary;float epsilon;int32_t rows,gamma_heads;};
// keys: attention key positions (cache cells [0, keys - window), then the window); the window start is a tensor.
struct QsaIndexCells {int32_t width,rotary;float epsilon;int32_t keys,rows,block;};
struct QsaSelect {int32_t keys,rows,window,width,block,heads;}; // the chunk's first window row is start[1] (runtime)
static_assert(sizeof(QsaIndexPrep)==24 && sizeof(QsaIndexCells)==24 && sizeof(QsaSelect)==24);
static_assert(sizeof(QsaPrep)==24);
// strata_bw_read_probe_f32 (hpu/tests/bw_read.cpp only): a bandwidth diagnostic. One member reads `vectors`
// 256-byte vectors through `chains` register-resident independent accumulators; `ops` 0 reads, 7 adds the
// resident GEMV's instruction-per-load ratio. `offset` is in member units, so several nodes can split one
// buffer inside a single graph (the per-graph versus per-node overhead test).
struct BwProbe {int32_t members,vectors,chains,ops,offset;};
static_assert(sizeof(BwProbe)==20);
struct QsaGather {int32_t width,heads,cells,bucket,rows,mask_heads;};
static_assert(sizeof(QsaGather)==24);
struct QsaVirtual {int32_t width,heads,cache_cells,tokens,first,count,bucket;};
static_assert(sizeof(QsaVirtual)==28);
struct Hc {int32_t width,streams;};
// STRATA_HPU_HC_DECODE_TPC: tpc/hc_down_bf16.c, the decode HC read's down/inject projection (columns = HC * H).
struct HcDown {int32_t columns,rank,streams;};
struct HcDownRows {int32_t columns,rank,streams,rows;}; // strata_hc_down_rows_bf16 (STRATA_HPU_HC_WINDOW_TPC)
// tpc/hc_mix_dual.c: codes = 1 writes the packed Q8 codes and scales too (STRATA_HPU_HC_MIX_Q8).
struct HcMixDual {int32_t width,streams,codes;};
// STRATA_HPU_GEMV_TPC: tpc/gemv_bf16.c, a one-row BF16 GEMM (rounded = 1: BF16 output).
struct GemvBf16 {int32_t columns,rows,rounded;};
struct HcWindow {int32_t width,streams,rows;};
struct PleGate {int32_t width,streams;float inverse_sqrt_width;};
struct PleConv {int32_t channels,taps,dilation;};
// A resident window's PLE gate and conv in one node each (tpc/ple_gate_rows.c, tpc/ple_conv_rows.c).
struct PleGateRows {int32_t width,rows,streams;float inverse_sqrt_width;};
struct PleConvRows {int32_t channels,taps,dilation,rows;};
// tpc/qsa_select_expand_fold.c: QsaSelect plus the query group folded into the SDPA's rows.
struct QsaSelectFold {int32_t keys,rows,window,width_sel,block,heads,group;};
// tpc/qsa_gather_rows_fp8.c (sparse prefill prototype): every row's selected blocks from the FP8 cache.
struct QsaGatherRows {int32_t heads,cells,top,block,rows;};
static_assert(sizeof(RowNorm)==16 && sizeof(Hc)==8 && sizeof(PleGate)==12 && sizeof(PleConv)==12);
static_assert(sizeof(HcWindow)==12);
}
