#pragma once
#include <cstdlib>
#include <stdexcept>
namespace strata::hpu {
inline bool performance_env(const char* key) {
    const char* v=std::getenv(key);return v && v[0]=='1' && v[1]=='\0';
}
inline bool fused_fp8_enabled() {
    static const bool on=performance_env("STRATA_HPU_FUSED_FP8");return on;
}
inline bool fast_bindings_enabled() {
    static const bool on=performance_env("STRATA_HPU_FAST_BINDINGS");return on;
}
inline bool direct_experts_enabled() {
    static const bool on=performance_env("STRATA_HPU_DIRECT_EXPERTS");return on;
}
// Decode expert groups as two TPC FP8 GEMV nodes (F32 activations) instead of MME FP8 GEMMs; needs the
// joined gate/up storage of STRATA_HPU_DIRECT_EXPERTS. Changes the arithmetic (no activation quantization).
inline bool tpc_experts_enabled() {
    static const bool on=performance_env("STRATA_HPU_TPC_EXPERTS");return on;
}
// FP8 expert codes tiled across 128 output rows and adjacent input columns. A separate arithmetic
// mode: four striped BF16 MAC accumulators replace the original horizontal dot-product reduction.
inline bool tpc_colpair_enabled() {
    static const bool on=performance_env("STRATA_HPU_TPC_COLPAIR");return on;
}
// TPC GEMV with the MME path's scaled FP8 hi/lo activations, instead of BF16-only activations.
inline bool tpc_hilo_enabled() {
    static const bool on=performance_env("STRATA_HPU_TPC_HILO");return on;
}
// Decode a whole token through the fused layers with the residual kept on the device (strata/hpu/decode_chain.hpp):
// one wait per graph that feeds the host, none after the expert groups. Host-side ordering only, same graphs.
inline bool chain_decode_enabled() {
    static const bool on=performance_env("STRATA_HPU_CHAIN");return on;
}
// GDN decode kernels without per-element scalar global loads: prepare/norm reduce max and sum of squares as
// vector trees (a different rounding order, so the disk-state configuration records it) and the step reads q/k
// from broadcast tensors (same per-lane arithmetic).
// Same recurrence and per-lane operation order, caching q/k vectors in the TPC instead of scalar
// global reads or graph-level broadcast tensors. Independent of prepare/norm reduction policy.
inline bool cached_gdn_step_enabled() {
    static const bool on=performance_env("STRATA_HPU_CACHED_GDN_STEP");return on;
}
// STRATA_HPU_DFLASH verify windows (2..4 rows, with SIMD_GDN): GDN conv, prepare and norm run once for all rows
// (tpc/gdn_*_rows.c), identical per row; STRATA_HPU_DFLASH_GDN_SMALL=0 keeps one launch per row.
inline bool gdn_small_rows_enabled() {
    static const bool on=[] {const char* e=std::getenv("STRATA_HPU_DFLASH_GDN_SMALL");return !(e && *e=='0');}();return on;
}
// With MERGED_GEMM: a GDN verify window's merged projection is one F32 GEMM whose four parts are read in place by
// the conv, prepare and norm row kernels (each reads its column block). The BF16 rounding of every part and the
// four scalar bf16->f32 casts between them disappear; the parts are the GEMM's F32 accumulation, its own arithmetic.
// Default-on after the round-22 same-prefix logits comparison; STRATA_HPU_GDN_FUSED_PARTS=0 restores the parts.
inline bool gdn_fused_parts_enabled() {
    static const bool on=[] {const char* e=std::getenv("STRATA_HPU_GDN_FUSED_PARTS");return !(e && *e=='0');}();return on;
}
// RESIDENT_IQ4 expert GEMV with each column pair's loads issued ahead (tpc/iq4q8_gemv_pipe.c); identical results.
inline bool gemv_pipe_enabled() {
    static const bool on=performance_env("STRATA_HPU_GEMV_PIPE");return on;
}
// With CACHED_GDN_STEP: the step issues its state loads sixteen at a time (tpc/gdn_step_pipe.c); identical results.
inline bool gdn_step_pipe_enabled() {
    static const bool on=performance_env("STRATA_HPU_GDN_STEP_PIPE");return on;
}
// RESIDENT_IQ4 expert GEMV column parts (more members for the 24 TPCs): STRATA_HPU_GEMV_PARTS_GU (gate/up, default 2)
// and STRATA_HPU_GEMV_PARTS_DOWN (default 1). The parts' F32 sums are added afterwards, so other values change the
// summation order of the projection (its own arithmetic).
inline int gemv_parts_env(const char* name,int fallback) {
    const char* value=std::getenv(name);if(!value || !*value) return fallback;
    const int parts=std::atoi(value);return parts>0?parts:fallback;
}
inline bool simd_gdn_enabled() {
    static const bool on=performance_env("STRATA_HPU_SIMD_GDN");return on;
}
// HostRecipe keeps its inputs and its outputs in one pinned/HBM block each: one upload and one readback
// per run instead of one copy per tensor. Host-side only; the graph and its arithmetic are unchanged.
inline bool packed_io_enabled() {
    static const bool on=performance_env("STRATA_HPU_PACKED_IO");return on;
}
// Every routed and shared expert resident in HBM as IQ4 codebook codes + F32 scales (strata/hpu/resident_experts.hpp),
// routed on the device (strata_route_topk_f32) and run by TPC int8 GEMVs on Q8 activations, so a chained decode layer
// is one launch with no host round trip. A separate arithmetic mode with its own disk-state configuration marker.
inline bool resident_iq4_enabled() {
    static const bool on=performance_env("STRATA_HPU_RESIDENT_IQ4");return on;
}
// With RESIDENT_IQ4: the expert store is U4 instead of IQ4 codes - syn_type_uint4 codes plus one F32 (scale, offset)
// pair per output channel and K group, laid out so the MME reads every expert once per prefill window as one
// grouped batch GEMM (strata/hpu/u4_resident.hpp). Same source weights, its own arithmetic.
inline bool resident_u4_enabled() {
    static const bool on=resident_iq4_enabled() && performance_env("STRATA_HPU_RESIDENT_U4");return on;
}
// With RESIDENT_IQ4: a QSA decode layer whose selection is the identity (short context) runs as one launch with its
// KV cache in HBM: norm, RoPE and FP16 rounding on the device (tpc/qsa_prep.c), attention over the cached cells, the
// host state committed from the device rows after the token. Longer contexts keep the host path. Own arithmetic.
inline bool qsa_device_enabled() {
    static const bool on=performance_env("STRATA_HPU_QSA_DEVICE");return on;
}
// Exact host-selected sparse attention: retain all FP16-representable K/V in HBM and
// gather on the device. Only cell ids and newly appended rows cross PCIe. Opt-in;
// the model planner admits the full cache before allocation. No arithmetic change.
inline bool qsa_device_gather_enabled() {
    static const bool on=performance_env("STRATA_HPU_QSA_DEVICE_GATHER");return on;
}
// With QSA_DEVICE_GATHER: a prefill window past the identity selection selects on the device (the indexer scores,
// the top cells per row as an attention mask, tpc/qsa_select_mask.c) instead of QsaBlock::stage_rows on the host,
// and attends densely over the masked cache. F32 indexer arithmetic (the host ranks in double).
inline bool qsa_device_select_enabled() {
    static const bool on=performance_env("STRATA_HPU_QSA_DEVICE_SELECT");return on;
}
// STRATA_HPU_QSA_SDPA (with DEVICE_SELECT): each select row chunk attends with Synapse's fused BF16 SDPA (no
// materialized scores or probabilities; the mask is an additive BF16 input) instead of F32 GEMM + softmax + GEMM.
// STRATA_HPU_QSA_KV_FP8 (with QSA_DEVICE_GATHER): the QSA device KV cache as FP8 (e4m3) codes with one F32 scale per
// (cell, KV head) -- a quarter of the F32 cache's HBM (12 GiB -> 3 GiB at 256k context) -- dequantized where read.
inline bool qsa_kv_fp8_enabled() {
    static const bool on=performance_env("STRATA_HPU_QSA_KV_FP8");return on;
}
// STRATA_HPU_QSA_DECODE_SELECT (with QSA_DEVICE_GATHER and QSA_KV_FP8): a decoded token past the identity selection
// selects and gathers its cells on the device (block scores, top-k, block gather) in its one fused layer launch, instead
// of a projection launch, a wait and the host indexer per QSA layer.
inline bool qsa_decode_select_enabled() {
    static const bool on=performance_env("STRATA_HPU_QSA_DECODE_SELECT");return on;
}
// STRATA_HPU_CHAIN_HEAD: a decoded token's final HC read and LM head queue on the decode chain's stream right behind
// its layers (the prefill's last-row head recipe), so the token costs one wait instead of three round trips.
inline bool chain_head_enabled() {
    static const bool on=performance_env("STRATA_HPU_CHAIN_HEAD");return on;
}
// STRATA_HPU_DECODE_RECIPE_PER_LAYER: every layer decodes through its own instance of the shared decode recipes, so a
// launch sees the same weight addresses every token (Synapse patches what changed) instead of the previous layer's.
inline bool decode_recipe_per_layer() {
    static const bool on=performance_env("STRATA_HPU_DECODE_RECIPE_PER_LAYER");return on;
}
// STRATA_HPU_ROW_NORM_FAST: the HC reads' dual row norm in two passes (tpc/row_norm_dual.c with STRATA_FAST_MATH: no
// max pre-scale, plain sum) instead of three; about an ulp off the exact passes.
inline bool row_norm_fast_enabled() {
    static const bool on=performance_env("STRATA_HPU_ROW_NORM_FAST");return on;
}
// STRATA_HPU_MOE_MERGED (with MOE_MME): a GDN layer's window FFN inside its own layer graph (one launch per layer, one
// recipe for every GDN layer) instead of the shared window FFN launch after it; saves a launch boundary per layer.
inline bool moe_merged_enabled() {
    static const bool on=performance_env("STRATA_HPU_MOE_MERGED");return on;
}
// STRATA_HPU_MOE_PLACE (with MOE_MME): the bucket from the SDK scatter of tpc/moe_mme_place.c's vector-written
// places instead of tpc/moe_mme_scatter.c's per-slot scalar global stores.
inline bool moe_place_enabled() {
    static const bool on=performance_env("STRATA_HPU_MOE_PLACE");return on;
}
inline bool qsa_sdpa_enabled() {
    static const bool on=performance_env("STRATA_HPU_QSA_SDPA");return on;
}
// STRATA_HPU_QSA_SDPA_GQA (with QSA_SDPA): the SDPA takes the KV heads as they are (grouped-query attention inside the
// kernel) instead of every KV head broadcast to its query group first (8x the BF16 keys/values).
// The select windows use the SDPA up to this many keys (STRATA_HPU_QSA_SDPA_MAX_KEYS, default 131072): past it the
// per-query-head BF16 keys/values (16 x keys x 256 x 2 bytes each) cost more HBM than the F32 attention's.
// STRATA_HPU_QSA_SPARSE (with QSA_DEVICE_SELECT, FP8 KV): a prefill window past STRATA_HPU_QSA_SPARSE_MIN_KEYS keys
// (default 32768) attends only to each row's selected blocks instead of every key under a mask: the device selects
// (tpc/qsa_select_bits.c), the host lists each row's blocks from the bits, the device gathers them
// (tpc/qsa_gather_rows_fp8.c) and attends per row. Cost per row no longer grows with the context.
inline bool qsa_sparse_enabled() {
    static const bool on=performance_env("STRATA_HPU_QSA_SPARSE");return on;
}
inline uint64_t qsa_sparse_min_keys() {
    static const uint64_t keys=[] {const char* v=std::getenv("STRATA_HPU_QSA_SPARSE_MIN_KEYS");return v && *v?std::strtoull(v,nullptr,10):32768ull;}();
    return keys;
}
// STRATA_HPU_QSA_SELECT_KEEP_MIB=N: the select recipes of the widest key buckets are kept while their own HBM (each
// recipe's persistent slots) fits N MiB, instead of the four widest: a long prompt crosses a bucket per window, and
// every request after the first recompiles the buckets that were dropped. 0 (default): the four widest.
inline uint64_t qsa_select_keep_bytes() {
    static const uint64_t bytes=[] {const char* v=std::getenv("STRATA_HPU_QSA_SELECT_KEEP_MIB");return v && *v?std::strtoull(v,nullptr,10)<<20:0ull;}();
    return bytes;
}
// STRATA_HPU_QSA_SDPA_FOLD (with QSA_SDPA): the SDPA runs per KV head with each head's query group folded into the
// query rows (q {1, hk, rows*group, d}, the mask's rows repeated per group), so K/V are not repeated per query head
// (0.5 GB instead of 6.4 GB at 256K keys): every key count takes the SDPA (QSA_SDPA_MAX_KEYS is ignored).
inline bool qsa_sdpa_fold_enabled() {
    static const bool on=performance_env("STRATA_HPU_QSA_SDPA_FOLD");return on;
}
// STRATA_HPU_QSA_SDPA_BATCH (with SDPA_FOLD): the group's query heads are the SDPA batch instead of query rows
// (q {group, hk, t, d}, K/V {1, hk, K, d} and mask {1, 1, t, K} broadcast over it): no per-group mask copy.
inline bool qsa_sdpa_batch_enabled() {
    static const bool on=performance_env("STRATA_HPU_QSA_SDPA_BATCH");return on;
}
inline bool qsa_select_sdpa(uint64_t keys) {
    static const uint64_t limit=[] {const char* v=std::getenv("STRATA_HPU_QSA_SDPA_MAX_KEYS");return v && *v?std::strtoull(v,nullptr,10):131072ull;}();
    return qsa_sdpa_enabled() && (qsa_sdpa_fold_enabled() || keys<=limit);
}
inline bool qsa_sdpa_gqa_enabled() {
    static const bool on=performance_env("STRATA_HPU_QSA_SDPA_GQA");return on;
}
// Charge only real pinned inputs for the staged device-gather attention; keep
// its full device KV slabs and the existing compiler reserve in the HBM plan.
inline bool qsa_host_budget_enabled() {
    static const bool on=performance_env("STRATA_HPU_QSA_HOST_BUDGET");return on;
}
// Bound the long-context prefill KV slab independently of the full-window FFN.
// Zero preserves the original graph; only measured chunk geometries are admitted.
inline unsigned qsa_chunk_rows() {
    static const unsigned rows=[] {
        const char* value=std::getenv("STRATA_HPU_QSA_CHUNK_ROWS");
        if(!value || !*value || (value[0]=='0' && value[1]=='\0')) return 0u;
        if(value[0]=='1' && value[1]=='6' && value[2]=='\0') return 16u;
        if(value[0]=='3' && value[1]=='2' && value[2]=='\0') return 32u;
        throw std::invalid_argument("STRATA_HPU_QSA_CHUNK_ROWS must be 0,16 or32");
    }();return rows;
}
// Host-side FP16 KV packing with AVX/F16C where available. Same round-to-nearest-even bits and overflow
// refusal as the scalar converter; no disk-state format change. Kept opt-in for throughput measurements.
inline bool qsa_host_pack_enabled() {
    static const bool on=performance_env("STRATA_HPU_QSA_HOST_PACK");return on;
}
// QSA prefill windows: attention batches over the KV head and reads the shared cache in place
// (qsa.cpp define_qsa_attention_dense_cache) instead of gathering one dense K/V copy per row. That copy is
// rows x bucket x width floats and is what caps a window at roughly 128 rows; the shared read costs
// kv_heads x cells x width (4 MiB at 2048 cells). Only valid when the selection is the identity.
inline bool qsa_dense_window_enabled() {
    static const bool on=performance_env("STRATA_HPU_QSA_DENSE_WINDOW");return on;
}
// Host QSA indexer block scores: AVX2 double lanes instead of the scalar double dot product. The 128 products
// are exact and identical; only their summation order changes, so a select can differ where two blocks tie in
// the last bits. Long context only (the device path below top_k+block-1 never scores on the host).
// SIMD across independent heads, preserving every scalar accumulation order.
inline bool qsa_index_exact_enabled() {
    static const bool on=performance_env("STRATA_HPU_QSA_EXACT_INDEX");return on;
}
inline bool qsa_index_ilp_enabled() {
    static const bool on=performance_env("STRATA_HPU_QSA_INDEX_ILP");return on;
}
inline bool qsa_index_simd_enabled() {
    static const bool on=performance_env("STRATA_HPU_QSA_SIMD_INDEX");return on;
}
// Threads that split the host QSA indexer's block scores (1 = off, capped at 64). Work splitting only: every
// block is scored by exactly one thread with the same arithmetic, so the selected cells are identical.
inline int qsa_index_threads() {
    static const int count=[] {
        const char* value=std::getenv("STRATA_HPU_QSA_INDEX_THREADS");
        if(!value || !*value) return 1;
        char* end=nullptr;const long n=std::strtol(value,&end,10);
        return end!=value && !*end && n>1 && n<=64?int(n):1;
    }();
    return count;
}
// Plain greedy sampling without bias/effective penalties: vector argmax with the same lowest-id ties,
// invalid-logit rejection and banned-token behavior. Other sampling policies retain the original path.
inline bool fast_greedy_enabled() {
    static const bool on=performance_env("STRATA_HPU_FAST_GREEDY");return on;
}
// Decode the immutable canonical embedding table once into budgeted host FP32 RAM. Text lookup is a copy;
// vision and explicit embedding input routes keep their existing precedence. No arithmetic/state format change.
inline bool embedding_ram_enabled() {
    static const bool on=performance_env("STRATA_HPU_EMBED_RAM");return on;
}
// With RESIDENT_IQ4: the projections a layer applies to one input are one GEMM over their concatenated weights
// (GDN qkv/gate/alpha/beta, QSA q_gate/k/v/index, each HC read's down/inject), split afterwards. The weights are
// moved into one buffer at load (no extra HBM). Each output column is the same dot product; the MME may tile the
// longer GEMM differently, so this is its own arithmetic mode.
inline bool merged_gemm_enabled() {
    static const bool on=performance_env("STRATA_HPU_MERGED_GEMM");return on;
}
// With MERGED_GEMM: those merged projections, the HC up projections and the attention/GDN output projections keep
// FP8 weights (fp8_143, one scale per output row, converted from the BF16 load) and run as one FP8 GEMM each with
// the activation as a hi+lo FP8 pair (GraphBuilder::fp8_linear one-pass). Half the weight bytes; a re-quantization
// of the GGUF weights, so its own arithmetic (quality to be measured).
// RESIDENT_IQ4 expert activations as packed Q8 codes, broadcast inside the GEMV (tpc/q8_quant_packed.c,
// tpc/iq4q8_gemv_packed.c); identical results to Q8_SHUFFLE + GEMV_PIPE.
inline bool q8_packed_enabled() {
    static const bool on=performance_env("STRATA_HPU_Q8_PACKED");return on;
}
// With Q8_PACKED: the resident expert GEMV runs one 256-row output tile per member (tpc/iq4q8_gemv_tile.c) so a
// member holds one accumulator and keeps eight weight loads in flight instead of five. Same MAC order, so the
// outputs are bit-identical to the five-tile kernel.
inline bool gemv_tile_enabled() {
    static const bool on=performance_env("STRATA_HPU_GEMV_TILE");return on;
}
// Prefill: the resident MoE reads its slots in a device-computed order grouped by routed expert
// (tpc/route_group_starts.c, route_group_order.c and iq4q8_gemv_group.c), so each expert's weight rows are loaded
// once per window instead of once per slot that routes to it. Every slot does the packed kernel's arithmetic.
// STRATA_HPU_MOE_GROUP_MIN_ROWS=N (default 8): windows of more than N rows take MOE_GROUP / MOE_PAIR's grouped reads
// (1 = DFlash verify windows too).
inline uint32_t moe_group_min_rows() {
    static const uint32_t rows=[] {const char* e=std::getenv("STRATA_HPU_MOE_GROUP_MIN_ROWS");return e && *e?uint32_t(std::strtoul(e,nullptr,10)):8u;}();
    return rows;
}
inline bool moe_group_enabled() {
    static const bool on=performance_env("STRATA_HPU_MOE_GROUP");return on;
}
// Stable grouped routing skips nonmatching 64-slot blocks instead of scanning
// every slot with scalar loads for every expert.
inline bool moe_group_blocks_enabled() {
    static const bool on=performance_env("STRATA_HPU_MOE_GROUP_BLOCKS");return on;
}
// Prefill pairs adjacent grouped slots inside one packed IQ4 kernel, sharing
// weight loads/LUT/scales when the routed expert matches. Original Q8 math.
inline bool moe_pair_enabled() {
    static const bool on=performance_env("STRATA_HPU_MOE_PAIR");return on;
}
// With Q8_PACKED: the packed GEMV issues the following column pair's five weight loads before consuming this
// pair (tpc/iq4q8_gemv_packed_pf.c), so a member keeps more loads in flight at the decode shape where it owns a
// whole TPC. Same MAC order, so the outputs are bit-identical to the packed kernel.
inline bool gemv_pf_enabled() {
    static const bool on=performance_env("STRATA_HPU_GEMV_PF");return on;
}
// With Q8_PACKED: the resident MoE's gate/up part sum, SiLU*up and activation quantization as one node
// (tpc/moe_act_packed.c); its SiLU is not the SDK's, so the activation can differ in the last bits.
inline bool moe_act_fused_enabled() {
    static const bool on=performance_env("STRATA_HPU_MOE_ACT_FUSED");return on;
}
// With MERGED_GEMM (+ RESIDENT_IQ4): router and shared gate as one GEMM; STRATA_HPU_ROUTER_MERGE=0 keeps two.
inline bool router_merge_enabled() {
    static const bool on=merged_gemm_enabled() && resident_iq4_enabled() && [] {const char* e=std::getenv("STRATA_HPU_ROUTER_MERGE");return !e || *e!='0';}();
    return on;
}
// STRATA_HPU_HC_DECODE_TPC (with MERGED_GEMM): a decoded token's HC read in four nodes instead of eight -- the norm
// with its BF16 copy (row_norm_dual), the down/inject projection on the TPC with the read's activation folded in
// (tpc/hc_down_bf16.c), the up GEMM, and the mix with its BF16 copy (tpc/hc_mix_dual.c). The TPC sums the down
// products in another order than the MME: not the same bits.
// STRATA_HPU_HC_WINDOW_TPC (with MERGED_GEMM, BF16 dense): a verify window's HC read down/inject projection and
// activation as one TPC node (tpc/hc_down_rows_bf16.c) instead of an MME GEMM, a split and the activation ops.
// STRATA_HPU_QSA_VERIFY_BF16: the verify window gather path attends to its gathered cells as BF16 codes (exact) with
// the scales applied to the scores and probabilities, and to its own rows apart in F32: no F32 dequantized copy of
// the cells and no concatenated key/value tensors (tpc/qsa_gather_window_bf16.c).
inline bool qsa_verify_bf16_enabled() {
    static const bool on=performance_env("STRATA_HPU_QSA_VERIFY_BF16");return on;
}
// STRATA_HPU_WINDOW_INTERNAL (diagnostic): a verify window's HC mix and post-attention residual as graph-internal
// tensors instead of persistent ones.
inline bool window_internal_enabled() {
    static const bool on=performance_env("STRATA_HPU_WINDOW_INTERNAL");return on;
}
inline bool hc_window_tpc_enabled() {
    static const bool on=performance_env("STRATA_HPU_HC_WINDOW_TPC");return on;
}
inline bool hc_decode_tpc_enabled() {
    static const bool on=performance_env("STRATA_HPU_HC_DECODE_TPC");return on;
}
// STRATA_HPU_GEMV_TPC: a one-row BF16 GEMM against a [N, K] weight (K % 128 == 0, K <= 8192, N <= 65536: the decode
// projections, not the LM head) runs on the TPC (tpc/gemv_bf16.c), which streams the weights faster than the MME at
// one row. The TPC sums the products in another order: not the same bits.
inline bool gemv_tpc_enabled() {
    static const bool on=performance_env("STRATA_HPU_GEMV_TPC");return on;
}
// STRATA_HPU_MOE_TAIL_DECODE: a decoded token's route weighting and expert sum inside its FFN HC write
// (tpc/hc_write_moe.c at one row, as MOE_TAIL_FUSED does for windows): one node instead of two. Its sum runs in
// another order than the SDK reduction: not the same bits.
inline bool moe_tail_decode_enabled() {
    static const bool on=performance_env("STRATA_HPU_MOE_TAIL_DECODE");return on;
}
// STRATA_HPU_HC_MIX_Q8 (with HC_DECODE_TPC and Q8_PACKED): the decode FFN read's mix also writes the resident MoE's
// packed Q8 activation codes (the same codes), so no quantization node follows it.
inline bool hc_mix_q8_enabled() {
    static const bool on=performance_env("STRATA_HPU_HC_MIX_Q8");return on;
}
// HC reads whose norm and window mix also write their BF16 copies (tpc/row_norm_dual.c, tpc/hc_mix_window_dual.c),
// replacing the cast nodes after them; identical results.
inline bool hc_fused_enabled() {
    static const bool on=performance_env("STRATA_HPU_HC_FUSED");return on;
}
// Verify windows: the resident MoE's route weighting and per-row expert sum inside the FFN HC write
// (tpc/hc_write_moe.c); its sum order may differ from reduce_sum in the last bits.
inline bool moe_tail_fused_enabled() {
    static const bool on=performance_env("STRATA_HPU_MOE_TAIL_FUSED");return on;
}
// RESIDENT_IQ4 prefill windows of at least STRATA_HPU_MOE_MME_MIN_ROWS rows (default 256) run their MoE as grouped
// MME GEMMs over experts expanded once per window (strata/hpu/moe_mme.hpp), as a separate FFN launch per layer.
// Not bit-identical to the GEMV path (BF16 activations instead of Q8). A window whose expert buckets overflow
// produces non-finite rows by design and is replayed on the GEMV path.
// Final-state prefill windows whose rows are a multiple of 64 run the GDN recurrence chunked (gdn.hpp).
// The PLE gate's key.query dot product on vector lanes (tpc/ple_gate_simd.c): not bit-identical (summation order).
inline bool ple_gate_simd_enabled() {
    static const bool on=performance_env("STRATA_HPU_PLE_GATE_SIMD");return on;
}
// The GDN gated norm with the compiler library sigmoid and no max pre-scale (tpc/gdn_norm_fast.c): not bit-identical.
inline bool gdn_norm_fast_enabled() {
    static const bool on=performance_env("STRATA_HPU_GDN_NORM_FAST");return on;
}
inline bool gdn_chunked_enabled() {
    static const bool on=performance_env("STRATA_HPU_GDN_CHUNKED");return on;
}
// With MOE_MME: the routed experts expand to FP8 E4M3 with a scale per output row and the routed GEMMs run FP8 x FP8
// (activations FP8 with a scale per row). Not bit-identical: about 2-4% relative error per product.
// The HC mix, the GDN conv and the MoE activation built with the compiler library sigmoid (tpc/gdn_math.h,
// STRATA_FAST_MATH twins). Not bit-identical.
inline bool fast_sigmoid_enabled() {
    static const bool on=performance_env("STRATA_HPU_FAST_SIGMOID");return on;
}
inline bool moe_fp8_enabled() {
    static const bool on=performance_env("STRATA_HPU_MOE_FP8");return on;
}
inline bool moe_mme_enabled() {
    static const bool on=performance_env("STRATA_HPU_MOE_MME");return on;
}
inline unsigned moe_mme_min_rows() {
    static const unsigned rows=[] {const char* v=std::getenv("STRATA_HPU_MOE_MME_MIN_ROWS");return v && *v?unsigned(std::strtoul(v,nullptr,10)):256u;}();
    return rows;
}
// Verify windows of 2..4 rows: all recurrent steps in one kernel reading the state twice (tpc/gdn_step_rows4.c);
// the same recurrence with its sums associated differently (not bit-identical to the per-row steps).
inline bool gdn_step_rows_enabled() {
    static const bool on=performance_env("STRATA_HPU_GDN_STEP_ROWS");return on;
}
// Prefill commits full windows: retain only final GDN state and keep the recurrent state in TPC local memory.
// Research-only local-state recurrence; slower in the first end-to-end run.
inline bool prefill_local_enabled() {
    static const bool on=performance_env("STRATA_HPU_PREFILL_LOCAL");return on;
}
// Large prefill windows preserve the canonical 32-row QSA key layout and reductions.
inline bool prefill_qsa32_enabled() {
    static const bool on=performance_env("STRATA_HPU_PREFILL_QSA32");return on;
}
inline bool prefill_final_enabled() {
    static const bool on=performance_env("STRATA_HPU_PREFILL_FINAL");return on;
}
// Prefill consumes only the committed last row's logits. Keep DFlash's all-row
// head unchanged; this opt-in head binds the last residual row and has no top-k.
inline bool prefill_last_head_enabled() {
    static const bool on=performance_env("STRATA_HPU_PREFILL_LAST_HEAD");return on;
}
// Affine access patterns for row norm and HC kernels in the custom TPC glue.
inline bool tpc_access_enabled() {
    static const bool on=performance_env("STRATA_HPU_TPC_ACCESS");return on;
}
// Final-state prefill: the window's recurrent steps in one launch (tpc/gdn_step_window.c) instead of one node per
// row. Same per-row arithmetic in the same order, so the outputs and the final state are bit-identical.
inline bool gdn_step_window_enabled() {
    static const bool on=performance_env("STRATA_HPU_GDN_WINDOW");return on;
}
inline bool fp8_dense_enabled() {
    static const bool on=performance_env("STRATA_HPU_FP8_DENSE");return on;
}
// Experimental dense FP8: fixed power-of-16 scales carried by native MME exponent biases.
// One activation term, no runtime amax or product epilogue. Requires FP8_DENSE; quality must be measured.
inline bool gdn_wide_rows_enabled() {
    static const bool on=performance_env("STRATA_HPU_GDN_WIDE_ROWS");return on;
}
inline bool fp8_static_enabled() {
    static const bool on=performance_env("STRATA_HPU_FP8_STATIC");return on;
}
// Identical Q8 scales/codes; replace per-column masked reductions with TPC lane broadcasts.
inline bool q8_shuffle_enabled() {
    static const bool on=performance_env("STRATA_HPU_Q8_SHUFFLE");return on;
}
// Poll briefly for hot decode events, then retain the blocking SDK wait for longer work.
// Host scheduling only; event completion and disk-state arithmetic are unchanged.
inline bool event_poll_enabled() {
    static const bool on=performance_env("STRATA_HPU_EVENT_POLL");return on;
}
}
