#include <tpc_kernel_lib_interface.h>
#include "strata/hpu/gdn_abi.hpp"
#include "strata/hpu/model_abi.hpp"
#include "strata/hpu/packed_abi.hpp"
#include "packed_codebook_layout.inc"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>
#include <iterator>
#include <limits>

using namespace tpc_lib_api;
using namespace strata::hpu::abi;
#define ELF(name) extern "C" const unsigned char strata_##name##_elf_start[], strata_##name##_elf_end[];
ELF(gdn_step)
ELF(gdn_conv)
ELF(gdn_prepare)
ELF(gdn_norm)
ELF(row_norm)
ELF(hc_mix)
ELF(hc_write)
ELF(ple_gate)
ELF(ple_conv)
ELF(packed_matmul)
ELF(activation_quant)
ELF(hc_mix_window)
ELF(hc_write_window)
ELF(row_norm_simd)
ELF(fp8_encode)
ELF(fp8_gemv)
ELF(gdn_prepare_simd)
ELF(gdn_norm_simd)
ELF(gdn_step_bcast)
ELF(fp8_gemv_hilo)
ELF(fp8_gemv_colpair)
ELF(gdn_step_cached)
ELF(iq4q8_gemv)
ELF(q8_quant)
ELF(route_topk)
ELF(q8_quant_shuffle)
ELF(qsa_prep)
ELF(route_share)
ELF(iq4q8_gemv_rows)
ELF(gdn_step_pipe)
ELF(gdn_conv_rows)
ELF(gdn_prepare_rows)
ELF(gdn_norm_rows)
ELF(iq4q8_gemv_pipe)
ELF(q8_quant_packed)
ELF(iq4q8_gemv_packed)
ELF(iq4q8_gemv_tile)
ELF(fp8_encode_stacked)
ELF(fp8_linear_finish)
ELF(moe_act_packed)
ELF(row_norm_dual)
ELF(hc_mix_window_dual)
ELF(gdn_norm_rows_bf16)
ELF(hc_write_moe)
ELF(gdn_step_rows4)
ELF(fp8_encode_fixed)
ELF(gdn_conv_rows8)
ELF(qsa_gather)
ELF(gdn_conv_final)
ELF(gdn_step_final)
ELF(gdn_step_window)
ELF(qsa_virtual)
ELF(route_group_starts)
ELF(route_group_order)
ELF(iq4q8_gemv_group)
ELF(iq4q8_gemv_packed_pf)
ELF(u4_moe_bucket)
ELF(u4_moe_gather)
ELF(u4_moe_gu_epi)
ELF(u4_moe_down_epi)
ELF(u4_moe_join)
ELF(u4_route_blocks)
ELF(u4_bucket_blocks)
ELF(fp8_batch_encode)
ELF(fp8_batch_finish)
ELF(bw_read_probe)
ELF(route_group_blocks)
ELF(iq4q8_gemv_pair)
ELF(iq4_expand_bf16)
ELF(moe_mme_join)
ELF(moe_mme_act)
ELF(moe_mme_join_perm)
ELF(moe_mme_count)
ELF(moe_mme_prefix)
ELF(moe_mme_scatter)
ELF(moe_mme_gather)
ELF(moe_mme_tilemap)
ELF(iq4_expand_tiles)
ELF(ple_gate_simd)
ELF(gdn_norm_rows8)
ELF(gdn_norm_fast)
ELF(iq4_expand_tiles_fp8)
ELF(moe_mme_gather_fp8)
ELF(moe_mme_act_fp8)
ELF(moe_mme_join_fp8)
ELF(hc_mix_window_dual_fast)
ELF(gdn_conv_final_fast)
ELF(moe_mme_act_fast)
ELF(hc_write_moe_routed)
ELF(qsa_index_prep)
ELF(qsa_index_cells)
ELF(qsa_select_mask)
ELF(qsa_select_expand)
ELF(moe_mme_place)
ELF(row_norm_dual_fast)
ELF(qsa_gather_fp8)
ELF(qsa_decode_blocks)
ELF(qsa_gather_blocks_fp8)
ELF(qsa_window_blocks)
ELF(qsa_gather_window_fp8)
ELF(hc_down_bf16)
ELF(hc_mix_dual)
ELF(gemv_bf16)
ELF(ple_gate_rows)
ELF(ple_conv_rows)
ELF(qsa_select_expand_fold)
ELF(qsa_gather_rows_fp8)
ELF(qsa_select_bits)
ELF(hc_down_rows_bf16)
ELF(qsa_gather_window_bf16)
ELF(qsa_verify_softmax)
ELF(qsa_verify_partials)
namespace {
struct Kernel { const char* guid; const unsigned char* begin; const unsigned char* end; unsigned inputs, outputs; };
const Kernel kernels[] = {
// The quantizer is appended LAST on purpose: find() returns the index in this table, and the
// configure/infer branches match on those numbers, so inserting it in the middle would silently
// renumber the HC window kernels (which is exactly what broke hpu_hc_ple_compile once).

    {"strata_gdn_step_f32", strata_gdn_step_elf_start, strata_gdn_step_elf_end, 6, 2},
    {"strata_gdn_conv_f32", strata_gdn_conv_elf_start, strata_gdn_conv_elf_end, 3, 2},
    {"strata_gdn_prepare_f32", strata_gdn_prepare_elf_start, strata_gdn_prepare_elf_end, 5, 5},
    {"strata_gdn_norm_f32", strata_gdn_norm_elf_start, strata_gdn_norm_elf_end, 3, 1},
    {"strata_row_norm_f32", strata_row_norm_elf_start, strata_row_norm_elf_end, 2, 1},
    {"strata_hc_mix_f32", strata_hc_mix_elf_start, strata_hc_mix_elf_end, 2, 1},
    {"strata_hc_write_f32", strata_hc_write_elf_start, strata_hc_write_elf_end, 3, 1},
    {"strata_ple_gate_f32", strata_ple_gate_elf_start, strata_ple_gate_elf_end, 3, 1},
    {"strata_ple_conv_f32", strata_ple_conv_elf_start, strata_ple_conv_elf_end, 5, 2},
    {"strata_packed_matmul", strata_packed_matmul_elf_start, strata_packed_matmul_elf_end, 4, 1},
    {"strata_hc_mix_window_f32", strata_hc_mix_window_elf_start, strata_hc_mix_window_elf_end, 2, 1},
    {"strata_hc_write_window_f32", strata_hc_write_window_elf_start, strata_hc_write_window_elf_end, 3, 1},
    {"strata_activation_quant_f32", strata_activation_quant_elf_start, strata_activation_quant_elf_end, 1, 2},
    {"strata_row_norm_simd_f32", strata_row_norm_simd_elf_start, strata_row_norm_simd_elf_end, 2, 1},
    {"strata_fp8_encode_f32", strata_fp8_encode_elf_start, strata_fp8_encode_elf_end, 1, 3},
    {"strata_fp8_gemv_f32", strata_fp8_gemv_elf_start, strata_fp8_gemv_elf_end, 12, 1},
    // STRATA_HPU_SIMD_GDN variants of ids 2, 3 and 0 (same parameters; the step reads broadcast q/k).
    {"strata_gdn_prepare_simd_f32", strata_gdn_prepare_simd_elf_start, strata_gdn_prepare_simd_elf_end, 5, 5},
    {"strata_gdn_norm_simd_f32", strata_gdn_norm_simd_elf_start, strata_gdn_norm_simd_elf_end, 3, 1},
    {"strata_gdn_step_bcast_f32", strata_gdn_step_bcast_elf_start, strata_gdn_step_bcast_elf_end, 6, 2},
    {"strata_fp8_gemv_hilo_f32", strata_fp8_gemv_hilo_elf_start, strata_fp8_gemv_hilo_elf_end, 14, 1},
    {"strata_fp8_gemv_colpair_f32", strata_fp8_gemv_colpair_elf_start, strata_fp8_gemv_colpair_elf_end, 12, 1},
    {"strata_gdn_step_cached_f32", strata_gdn_step_cached_elf_start, strata_gdn_step_cached_elf_end, 6, 2},
    // STRATA_HPU_RESIDENT_IQ4: expert GEMV by id, activation quantizer, device routing.
    {"strata_iq4q8_gemv_f32", strata_iq4q8_gemv_elf_start, strata_iq4q8_gemv_elf_end, 6, 1},
    {"strata_q8_quant_f32", strata_q8_quant_elf_start, strata_q8_quant_elf_end, 1, 2},
    {"strata_route_topk_f32", strata_route_topk_elf_start, strata_route_topk_elf_end, 2, 2},
    {"strata_q8_quant_shuffle_f32", strata_q8_quant_shuffle_elf_start, strata_q8_quant_shuffle_elf_end, 1, 2},
    // STRATA_HPU_QSA_DEVICE: QSA decode norm/RoPE/FP16 rounding on the device.
    {"strata_qsa_prep_f32", strata_qsa_prep_elf_start, strata_qsa_prep_elf_end, 6, 5},
    // STRATA_HPU_DFLASH verify windows: slots sharing an expert, and the GEMV reading its weights once for them.
    {"strata_route_share_i32", strata_route_share_elf_start, strata_route_share_elf_end, 1, 1},
    {"strata_iq4q8_gemv_rows_f32", strata_iq4q8_gemv_rows_elf_start, strata_iq4q8_gemv_rows_elf_end, 7, 1},
    // STRATA_HPU_GDN_STEP_PIPE: gdn_step_cached with its state loads issued ahead (same checks as id 0).
    {"strata_gdn_step_pipe_f32", strata_gdn_step_pipe_elf_start, strata_gdn_step_pipe_elf_end, 6, 2},
    // STRATA_HPU_DFLASH verify windows: GDN conv / prepare (SIMD) / norm (SIMD) for several rows per launch.
    {"strata_gdn_conv_rows_f32", strata_gdn_conv_rows_elf_start, strata_gdn_conv_rows_elf_end, 3, 5},
    {"strata_gdn_prepare_rows_f32", strata_gdn_prepare_rows_elf_start, strata_gdn_prepare_rows_elf_end, 5, 5},
    {"strata_gdn_norm_rows_f32", strata_gdn_norm_rows_elf_start, strata_gdn_norm_rows_elf_end, 3, 1},
    // STRATA_HPU_GEMV_PIPE: iq4q8_gemv with its loads issued ahead (same checks as id 22).
    {"strata_iq4q8_gemv_pipe_f32", strata_iq4q8_gemv_pipe_elf_start, strata_iq4q8_gemv_pipe_elf_end, 6, 1},
    // STRATA_HPU_Q8_PACKED: ids 23 and 22 with packed activation codes {256, columns/16, rows}.
    {"strata_q8_quant_packed_f32", strata_q8_quant_packed_elf_start, strata_q8_quant_packed_elf_end, 1, 2},
    {"strata_iq4q8_gemv_packed_f32", strata_iq4q8_gemv_packed_elf_start, strata_iq4q8_gemv_packed_elf_end, 6, 1},
    // STRATA_HPU_FP8_DENSE one-pass FP8 linear: stacked hi/lo activation codes, and the scaled combination.
    {"strata_fp8_encode_stacked_f32", strata_fp8_encode_stacked_elf_start, strata_fp8_encode_stacked_elf_end, 1, 2},
    {"strata_fp8_linear_finish_f32", strata_fp8_linear_finish_elf_start, strata_fp8_linear_finish_elf_end, 3, 1},
    // STRATA_HPU_MOE_ACT_FUSED: gate/up part sum + SiLU(gate)*up + packed Q8 codes in one node.
    {"strata_moe_act_packed_f32", strata_moe_act_packed_elf_start, strata_moe_act_packed_elf_end, 1, 2},
    // STRATA_HPU_HC_FUSED: ids 13 and 10 with a second output, the result rounded to BF16.
    {"strata_row_norm_dual_f32", strata_row_norm_dual_elf_start, strata_row_norm_dual_elf_end, 2, 2},
    {"strata_hc_mix_window_dual_f32", strata_hc_mix_window_dual_elf_start, strata_hc_mix_window_dual_elf_end, 2, 2},
    // STRATA_HPU_HC_FUSED: id 32 with a BF16 output.
    {"strata_gdn_norm_rows_bf16_f32", strata_gdn_norm_rows_bf16_elf_start, strata_gdn_norm_rows_bf16_elf_end, 3, 1},
    // STRATA_HPU_MOE_TAIL_FUSED verify windows: route weighting + per-row expert sum + HC write.
    {"strata_hc_write_moe_f32", strata_hc_write_moe_elf_start, strata_hc_write_moe_elf_end, 4, 1},
    // STRATA_HPU_GDN_STEP_ROWS: a verify window's 2..4 recurrent steps in one kernel.
    {"strata_gdn_step_rows4_f32", strata_gdn_step_rows4_elf_start, strata_gdn_step_rows4_elf_end, 6, 5},
    {"strata_fp8_encode_fixed_f32",strata_fp8_encode_fixed_elf_start,strata_fp8_encode_fixed_elf_end,1,1},
    {"strata_gdn_conv_rows8_f32",strata_gdn_conv_rows8_elf_start,strata_gdn_conv_rows8_elf_end,3,9},
    {"strata_qsa_gather_f32",strata_qsa_gather_elf_start,strata_qsa_gather_elf_end,3,3},
    // Append variants: configure() and base_kernel() use stable existing indices.
    {"strata_iq4q8_gemv_tile_f32", strata_iq4q8_gemv_tile_elf_start, strata_iq4q8_gemv_tile_elf_end, 6, 1},
    {"strata_gdn_conv_final_f32", strata_gdn_conv_final_elf_start, strata_gdn_conv_final_elf_end, 3, 2},
    {"strata_gdn_step_final_f32", strata_gdn_step_final_elf_start, strata_gdn_step_final_elf_end, 6, 2},
    {"strata_qsa_virtual_f32", strata_qsa_virtual_elf_start, strata_qsa_virtual_elf_end, 5, 2},
    // STRATA_HPU_MOE_GROUP: the slot order grouped by routed expert, and the packed GEMV that reads it
    // (ids 52, 53 and 54; appending keeps every earlier index).
    {"strata_route_group_starts_f32", strata_route_group_starts_elf_start, strata_route_group_starts_elf_end, 1, 1},
    {"strata_route_group_order_f32", strata_route_group_order_elf_start, strata_route_group_order_elf_end, 2, 1},
    {"strata_iq4q8_gemv_group_f32", strata_iq4q8_gemv_group_elf_start, strata_iq4q8_gemv_group_elf_end, 7, 1},
    // STRATA_HPU_GEMV_PF: the packed GEMV with its next-pair loads issued ahead (same checks as id 22).
    {"strata_iq4q8_gemv_packed_pf_f32", strata_iq4q8_gemv_packed_pf_elf_start, strata_iq4q8_gemv_packed_pf_elf_end, 6, 1},
    // STRATA_HPU_GDN_WINDOW: the whole window's steps in one launch, state slice kept in HBM (gdn_step's arithmetic).
    {"strata_gdn_step_window_f32", strata_gdn_step_window_elf_start, strata_gdn_step_window_elf_end, 6, 2},
    // STRATA_HPU_RESIDENT_U4: the grouped-MoE MME path (ids 56-60; appending keeps every earlier index).
    {"strata_u4_moe_bucket_f32", strata_u4_moe_bucket_elf_start, strata_u4_moe_bucket_elf_end, 2, 3},
    {"strata_u4_moe_gather_f32", strata_u4_moe_gather_elf_start, strata_u4_moe_gather_elf_end, 2, 2},
    {"strata_u4_moe_gu_epi_f32", strata_u4_moe_gu_epi_elf_start, strata_u4_moe_gu_epi_elf_end, 5, 2},
    {"strata_u4_moe_down_epi_f32", strata_u4_moe_down_epi_elf_start, strata_u4_moe_down_epi_elf_end, 6, 1},
    {"strata_u4_moe_join_f32", strata_u4_moe_join_elf_start, strata_u4_moe_join_elf_end, 4, 1},
    {"strata_u4_route_blocks_f32",strata_u4_route_blocks_elf_start,strata_u4_route_blocks_elf_end,1,1},
    {"strata_u4_bucket_blocks_f32",strata_u4_bucket_blocks_elf_start,strata_u4_bucket_blocks_elf_end,2,3},
    {"strata_fp8_batch_encode_f32",strata_fp8_batch_encode_elf_start,strata_fp8_batch_encode_elf_end,1,2},
    {"strata_fp8_batch_finish_f32",strata_fp8_batch_finish_elf_start,strata_fp8_batch_finish_elf_end,2,1},
    // Bandwidth diagnostic (hpu/tests/bw_read.cpp). Append-only: this stays id 65.
    {"strata_bw_read_probe_f32",strata_bw_read_probe_elf_start,strata_bw_read_probe_elf_end,1,1},
    {"strata_route_group_blocks_f32",strata_route_group_blocks_elf_start,strata_route_group_blocks_elf_end,3,1},
    {"strata_iq4q8_gemv_pair_f32",strata_iq4q8_gemv_pair_elf_start,strata_iq4q8_gemv_pair_elf_end,7,1},
    {"strata_iq4_expand_bf16",strata_iq4_expand_bf16_elf_start,strata_iq4_expand_bf16_elf_end,2,1},
    {"strata_moe_mme_join_f32",strata_moe_mme_join_elf_start,strata_moe_mme_join_elf_end,3,1},
    {"strata_moe_mme_act_bf16",strata_moe_mme_act_elf_start,strata_moe_mme_act_elf_end,2,1},
    {"strata_moe_mme_join_perm_f32",strata_moe_mme_join_perm_elf_start,strata_moe_mme_join_perm_elf_end,3,1},
    {"strata_moe_mme_count_f32",strata_moe_mme_count_elf_start,strata_moe_mme_count_elf_end,1,1},
    {"strata_moe_mme_prefix_f32",strata_moe_mme_prefix_elf_start,strata_moe_mme_prefix_elf_end,1,3},
    {"strata_moe_mme_scatter_f32",strata_moe_mme_scatter_elf_start,strata_moe_mme_scatter_elf_end,3,2},
    {"strata_moe_mme_gather_f32",strata_moe_mme_gather_elf_start,strata_moe_mme_gather_elf_end,3,1},
    {"strata_moe_mme_tilemap_f32",strata_moe_mme_tilemap_elf_start,strata_moe_mme_tilemap_elf_end,3,2},
    {"strata_iq4_expand_tiles_bf16",strata_iq4_expand_tiles_elf_start,strata_iq4_expand_tiles_elf_end,3,1},
    {"strata_ple_gate_simd_f32",strata_ple_gate_simd_elf_start,strata_ple_gate_simd_elf_end,3,1},
    {"strata_gdn_norm_rows8_bf16_f32",strata_gdn_norm_rows8_elf_start,strata_gdn_norm_rows8_elf_end,3,1},
    {"strata_gdn_norm_fast_bf16_f32",strata_gdn_norm_fast_elf_start,strata_gdn_norm_fast_elf_end,3,1},
    {"strata_iq4_expand_tiles_fp8",strata_iq4_expand_tiles_fp8_elf_start,strata_iq4_expand_tiles_fp8_elf_end,3,2},
    {"strata_moe_mme_gather_fp8_f32",strata_moe_mme_gather_fp8_elf_start,strata_moe_mme_gather_fp8_elf_end,3,2},
    {"strata_moe_mme_act_fp8_bf16",strata_moe_mme_act_fp8_elf_start,strata_moe_mme_act_fp8_elf_end,4,2},
    {"strata_moe_mme_join_fp8_f32",strata_moe_mme_join_fp8_elf_start,strata_moe_mme_join_fp8_elf_end,5,1},
    // Fast-math twins of ids 40, 48 and 70 (fast_twin(): same checks and index space, their own ELF).
    {"strata_hc_mix_window_dual_fast_f32",strata_hc_mix_window_dual_fast_elf_start,strata_hc_mix_window_dual_fast_elf_end,2,2},
    {"strata_gdn_conv_final_fast_f32",strata_gdn_conv_final_fast_elf_start,strata_gdn_conv_final_fast_elf_end,3,2},
    {"strata_moe_mme_act_fast_bf16",strata_moe_mme_act_fast_elf_start,strata_moe_mme_act_fast_elf_end,2,1},
    {"strata_hc_write_moe_routed_f32",strata_hc_write_moe_routed_elf_start,strata_hc_write_moe_routed_elf_end,6,1},
    {"strata_qsa_index_prep_f32",strata_qsa_index_prep_elf_start,strata_qsa_index_prep_elf_end,3,1},
    {"strata_qsa_index_cells_f32",strata_qsa_index_cells_elf_start,strata_qsa_index_cells_elf_end,6,1},
    {"strata_qsa_select_mask_f32",strata_qsa_select_mask_elf_start,strata_qsa_select_mask_elf_end,2,2},
    {"strata_qsa_select_expand_f32",strata_qsa_select_expand_elf_start,strata_qsa_select_expand_elf_end,2,1},
    {"strata_moe_mme_place_f32",strata_moe_mme_place_elf_start,strata_moe_mme_place_elf_end,3,4},
    // STRATA_HPU_ROW_NORM_FAST: the fast-math twin of id 39 (two passes).
    {"strata_row_norm_dual_fast_f32",strata_row_norm_dual_fast_elf_start,strata_row_norm_dual_fast_elf_end,2,2},
    {"strata_qsa_gather_fp8",strata_qsa_gather_fp8_elf_start,strata_qsa_gather_fp8_elf_end,5,5},
    {"strata_qsa_decode_blocks_f32",strata_qsa_decode_blocks_elf_start,strata_qsa_decode_blocks_elf_end,2,1},
    {"strata_qsa_gather_blocks_fp8",strata_qsa_gather_blocks_fp8_elf_start,strata_qsa_gather_blocks_fp8_elf_end,7,5},
    {"strata_qsa_window_blocks_f32",strata_qsa_window_blocks_elf_start,strata_qsa_window_blocks_elf_end,2,1},
    {"strata_qsa_gather_window_fp8",strata_qsa_gather_window_fp8_elf_start,strata_qsa_gather_window_fp8_elf_end,7,5},
    // STRATA_HPU_HC_DECODE_TPC: the decode HC read's down/inject projection with its activation, and the dual mix.
    {"strata_hc_down_bf16",strata_hc_down_bf16_elf_start,strata_hc_down_bf16_elf_end,3,2},
    {"strata_hc_mix_dual_f32",strata_hc_mix_dual_elf_start,strata_hc_mix_dual_elf_end,2,4},
    // STRATA_HPU_GEMV_TPC: one-row BF16 GEMM on the TPC.
    {"strata_gemv_bf16",strata_gemv_bf16_elf_start,strata_gemv_bf16_elf_end,2,1},
    // A resident window's PLE gate and causal conv, one node each instead of one per window row.
    {"strata_ple_gate_rows_f32",strata_ple_gate_rows_elf_start,strata_ple_gate_rows_elf_end,3,1},
    {"strata_ple_conv_rows_f32",strata_ple_conv_rows_elf_start,strata_ple_conv_rows_elf_end,5,2},
    // STRATA_HPU_QSA_SDPA_FOLD: id 92's mask as the folded SDPA's BF16 mask.
    {"strata_qsa_select_expand_fold_bf16",strata_qsa_select_expand_fold_elf_start,strata_qsa_select_expand_fold_elf_end,2,1},
    // Sparse QSA prefill (prototype): per-row block gather.
    {"strata_qsa_gather_rows_fp8",strata_qsa_gather_rows_fp8_elf_start,strata_qsa_gather_rows_fp8_elf_end,7,5},
    // The sparse prefill's selection as bits + the cut block (tpc/qsa_select_bits.c; id 91's checks).
    {"strata_qsa_select_bits_f32",strata_qsa_select_bits_elf_start,strata_qsa_select_bits_elf_end,2,3},
    // STRATA_HPU_HC_WINDOW_TPC: id 100 for a verify window's rows.
    {"strata_hc_down_rows_bf16",strata_hc_down_rows_bf16_elf_start,strata_hc_down_rows_bf16_elf_end,3,2},
    // STRATA_HPU_QSA_VERIFY_BF16: id 99 with 16-rank members and BF16 (even/odd) codes.
    {"strata_qsa_gather_window_bf16",strata_qsa_gather_window_bf16_elf_start,strata_qsa_gather_window_bf16_elf_end,7,5},
    // STRATA_HPU_QSA_VERIFY_BF16: the window's scale, mask, softmax and value scale in one node.
    {"strata_qsa_verify_softmax_f32",strata_qsa_verify_softmax_elf_start,strata_qsa_verify_softmax_elf_end,6,3},
    {"strata_qsa_verify_partials_f32",strata_qsa_verify_partials_elf_start,strata_qsa_verify_partials_elf_end,4,1},
};
static_assert(std::size(kernels)==112,"kernel ids are append-only: configure() and base_kernel() index this table");
// The checks and index space of a variant are those of the kernel it replaces.
// A fast-math twin is its original kernel for every check, index space and access pattern; only its ELF differs.
int fast_twin(int id) {return id==85?40:id==86?48:id==87?70:id==94?39:id;}
int base_kernel(int id) {return id==107?91:id==80?79:id==78?7:id==71?69:id==16?2:id==17?3:id==18?0:id==21?0:id==25?23:id==29?0:id==33?22:id==34?23:id==35?22:id==39?13:id==40?10:id==41?32:id==47?22:id==54?22:id==67?53:id;}
int find(const char* name) {
    for (unsigned i = 0; i < std::size(kernels); ++i) if (!std::strcmp(name, kernels[i].guid)) return int(i);
    return -1;
}
template<class T> bool params(const UserParams& data, T& value) {
    if (!data.nodeParams || data.nodeParamsSize != sizeof(T)) return false;
    std::memcpy(&value, data.nodeParams, sizeof(T));
    return true;
}
bool width(int s) { return s > 0 && s <= 256 && s % 64 == 0; }
bool epsilon(float e) { return std::isfinite(e) && e > 0; }
bool shape(const Tensor& tensor, std::initializer_list<uint64_t> dimensions,TensorDataType type=DATA_F32) {
    if (tensor.geometry.dataType != type || tensor.geometry.dims != dimensions.size()) return false;
    unsigned axis = 0;
    for (auto size : dimensions) {
        if (tensor.geometry.maxSizes[axis] != size || tensor.geometry.minSizes[axis] != size) return false;
        ++axis;
    }
    return true;
}
// A column block [offset, offset+width) of a rank-2 F32 tensor whose dim 0 is wider (the merged projection),
// or the part's own [width, rows] tensor when offset is 0 and dim 0 holds exactly `width`.
bool block(const Tensor& tensor,uint64_t offset,uint64_t width,uint64_t rows,TensorDataType type=DATA_F32) {
    if (tensor.geometry.dataType != type || tensor.geometry.dims != 2) return false;
    if (tensor.geometry.minSizes[1] != rows || tensor.geometry.maxSizes[1] != rows) return false;
    if (tensor.geometry.minSizes[0] < offset + width || tensor.geometry.maxSizes[0] < offset + width) return false;
    return tensor.geometry.minSizes[0] == tensor.geometry.maxSizes[0];
}
void shape(TensorShapeInfo& tensor, std::initializer_list<uint64_t> sizes) {
    tensor.geometry.dims = sizes.size();
    unsigned axis = 0;
    for (auto size : sizes) {
        tensor.geometry.maxSizes[axis] = tensor.geometry.minSizes[axis] = size;
        ++axis;
    }
}
GlueCodeReturn configure(int kernel_id, const HabanaKernelParams& p, HabanaKernelInstantiation& out) {
    const auto* in = p.inputTensors;
    const auto* outputs = p.outputTensors;
    const int id = base_kernel(kernel_id);
    out.indexSpaceRank = (id == 0 || id == 55) ? 2 : 1;
    if(id==79) {
        // gdn_norm_rows_bf16 for 128-wide heads, four rows per member (tpc/gdn_norm_rows8.c, gdn_norm_fast.c).
        GdnNormRows a{};
        if(!params(p.nodeParams,a) || a.width!=128 || a.value_heads<=0 || a.value_heads>1024 || !epsilon(a.epsilon) ||
           a.rows<1 || a.rows>int32_t(kMaxWindowRows)) return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
        const uint64_t s=a.width,hv=a.value_heads,r=a.rows;
        if(!shape(in[0],{s,hv,r}) || a.z_col<0 || !block(in[1],uint64_t(a.z_col),s*hv,r) || !shape(in[2],{s}))
            return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if(!shape(outputs[0],{s,hv,r},DATA_BF16)) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        out.indexSpaceRank=2;out.indexSpaceGeometry[0]=hv;out.indexSpaceGeometry[1]=kernel_id==80?(r<=8?r:(r+7)/8):(r+3)/4; // rows per member: 8 (fast; 1 for <= 8 rows) / 4
    } else if(id==68) {
        // STRATA_HPU_MOE_MME: table {256} U8 + resident IQ4 rows {bytes, experts} U8 -> BF16 {rows, columns, experts}.
        Iq4Expand a{};
        if(!params(p.nodeParams,a) || a.rows<=0 || a.rows>65536 || a.rows%1280 || a.columns<=0 || a.columns>65536 ||
           a.columns%32 || a.variant<0 || a.variant>4 || a.first<0) return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
        const uint64_t r=uint64_t(a.rows),c=uint64_t(a.columns),bytes=r/2*c+r*(c/32)*4;
        if(!shape(in[0],{256},DATA_U8) || in[1].geometry.dims!=2 || in[1].geometry.dataType!=DATA_U8 ||
           in[1].geometry.minSizes[0]!=bytes || in[1].geometry.maxSizes[0]!=bytes || !in[1].geometry.minSizes[1] ||
           in[1].geometry.minSizes[1]!=in[1].geometry.maxSizes[1]) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        // Experts [first, first+count) of the store; count is the output's outer dimension.
        const uint64_t e=in[1].geometry.minSizes[1],count=outputs[0].geometry.maxSizes[2];
        if(!count || uint64_t(a.first)+count>e || !shape(outputs[0],{r,c,count},DATA_BF16)) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        out.indexSpaceGeometry[0]=(r/256)*(c/32)*count;
    } else if(id==69) {
        // STRATA_HPU_MOE_MME: routed {width, bucket_len} + shared {width, window rows} (F32, or BF16 in variant-4
        // column order for id 71) + place {slots} I32 -> expert_rows {width, slots} F32. One member per slot.
        U4MoeJoin a{};
        if(!params(p.nodeParams,a) || a.per_row<=1 || a.top_k<=0 || a.top_k>=a.per_row || a.bucket_len<=0)
            return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
        const uint64_t width=outputs[0].geometry.maxSizes[0],slots=outputs[0].geometry.maxSizes[1];
        const uint64_t len=uint64_t(a.bucket_len),rows=slots/uint64_t(a.per_row);
        const bool perm=kernel_id==71;
        const auto source=perm?DATA_BF16:DATA_F32;
        if(!width || width%(perm?256:64) || width>65536 || !slots || slots>kMaxMoeSlots || slots%uint64_t(a.per_row))
            return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
        if(!shape(in[0],{width,len},source) || !shape(in[1],{width,rows},source) || !shape(in[2],{slots},DATA_I32))
            return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if(!shape(outputs[0],{width,slots})) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        out.indexSpaceGeometry[0]=slots;
    } else if(id==70) {
        // STRATA_HPU_MOE_MME: gu {2*ff, rows, batch} BF16 (variant-4 column order) + totals {batch} I32
        // -> act {ff, rows, batch} BF16. Members are (row, batch).
        MoeMmeAct a{};
        if(!params(p.nodeParams,a) || a.ff<=0 || a.ff>32768 || a.ff%128 || (2*a.ff)%256 || (a.use_totals!=0 && a.use_totals!=1) ||
           a.first<0) return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
        if(in[0].geometry.dims!=3) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        const uint64_t f=uint64_t(a.ff),r=in[0].geometry.maxSizes[1],b=in[0].geometry.maxSizes[2];
        if(!r || !b || !shape(in[0],{2*f,r,b},DATA_BF16) || in[1].geometry.dataType!=DATA_I32 || in[1].geometry.dims!=1 ||
           in[1].geometry.minSizes[0]!=in[1].geometry.maxSizes[0] ||
           (a.use_totals && in[1].geometry.maxSizes[0]<uint64_t(a.first)+b)) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if(!shape(outputs[0],{f,r,b},DATA_BF16)) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        out.indexSpaceRank=2;out.indexSpaceGeometry[0]=(r+7)/8;out.indexSpaceGeometry[1]=b;
    } else if(id==72 || id==73 || id==74) {
        // STRATA_HPU_MOE_MME bucketing: count (ids {slots} I32 -> counts {experts, blocks} F32), prefix (counts ->
        // offsets {experts, blocks} F32, totals {experts} I32, overflow {experts} F32) and scatter (ids + offsets ->
        // bucket {experts*rows} I32, place {slots} I32), blocks = ceil(slots/64).
        MoeMmeRoute a{};
        if(!params(p.nodeParams,a) || a.experts<=0 || a.experts>4096 || a.experts%64 || a.slots<=1 ||
           a.slots>int32_t(kMaxMoeSlots) || a.per_row<=1 || a.top_k<=0 || a.top_k>=a.per_row || a.slots%a.per_row ||
           a.rows<=0 || a.rows>4096) return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
        const uint64_t e=uint64_t(a.experts),sl=uint64_t(a.slots),blocks=(sl+63)/64,r=uint64_t(a.rows);
        if(id==72) {
            if(!shape(in[0],{sl},DATA_I32)) return GLUE_INCOMPATIBLE_INPUT_SIZE;
            if(!shape(outputs[0],{e,blocks})) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
            // Members are (64-slot block, 64 experts) (tpc/moe_mme_count.c).
            out.indexSpaceRank=2;out.indexSpaceGeometry[0]=blocks;out.indexSpaceGeometry[1]=e/64;
        } else if(id==73) {
            if(!shape(in[0],{e,blocks})) return GLUE_INCOMPATIBLE_INPUT_SIZE;
            if(!shape(outputs[0],{e,blocks}) || !shape(outputs[1],{e},DATA_I32) || !shape(outputs[2],{e}))
                return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
            out.indexSpaceGeometry[0]=e/64;
        } else {
            // scatter: + start {experts} F32 (each expert's first tile); bucket {tiles*rows} for any tile count.
            const uint64_t len=outputs[0].geometry.maxSizes[0];
            if(!shape(in[0],{sl},DATA_I32) || !shape(in[1],{e,blocks}) || !shape(in[2],{e})) return GLUE_INCOMPATIBLE_INPUT_SIZE;
            if(!len || len%r || !shape(outputs[0],{len},DATA_I32) || !shape(outputs[1],{sl},DATA_I32)) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
            out.indexSpaceGeometry[0]=blocks;
        }
    } else if(id==93) {
        // STRATA_HPU_MOE_PLACE: ids {slots} I32 + offsets {experts, blocks} + start {experts} -> place, target and
        // slot ids {slots} I32 (tpc/moe_mme_place.c), and the bucket scatter's base {>= trash + 64} I32 (never written).
        // Members are 64-slot blocks.
        MoeMmePlace a{};
        if(!params(p.nodeParams,a) || a.experts<=0 || a.experts>4096 || a.experts%64 || a.slots<=1 ||
           a.slots>int32_t(kMaxMoeSlots) || a.per_row<=1 || a.top_k<=0 || a.top_k>=a.per_row || a.slots%a.per_row ||
           a.rows<=0 || a.rows>4096 || a.trash<0) return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
        const uint64_t e=uint64_t(a.experts),sl=uint64_t(a.slots),blocks=(sl+63)/64;
        if(!shape(in[0],{sl},DATA_I32) || !shape(in[1],{e,blocks}) || !shape(in[2],{e})) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        for(unsigned i=0;i<3;++i) if(!shape(outputs[i],{sl},DATA_I32)) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        if(outputs[3].geometry.dims!=1 || outputs[3].geometry.dataType!=DATA_I32 || outputs[3].geometry.maxSizes[0]<uint64_t(a.trash)+64)
            return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        out.indexSpaceGeometry[0]=blocks;
    } else if(id==76) {
        // STRATA_HPU_MOE_MME tile map: end/start {experts} F32 + totals {experts} I32 -> tile_expert/tile_rows {tiles} I32.
        MoeMmeTiles a{};
        if(!params(p.nodeParams,a) || a.experts<=0 || a.experts>4096 || a.tiles<=0 || a.tiles>65536 || a.rows<=0 || a.rows>4096)
            return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
        const uint64_t e=uint64_t(a.experts),x=uint64_t(a.tiles);
        if(!shape(in[0],{e}) || !shape(in[1],{e}) || !shape(in[2],{e},DATA_I32)) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if(!shape(outputs[0],{x},DATA_I32) || !shape(outputs[1],{x},DATA_I32)) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        out.indexSpaceGeometry[0]=x;
    } else if(id==77) {
        // STRATA_HPU_MOE_MME tile expansion: table {256} U8 + resident IQ4 rows {bytes, experts} U8 + tile_expert
        // {tiles} I32 -> BF16 {rows, columns, count} for tiles [first, first+count). Members (256-row tile, tile).
        Iq4Expand a{};
        if(!params(p.nodeParams,a) || a.rows<=0 || a.rows>65536 || a.rows%1280 || a.columns<=0 || a.columns>65536 ||
           a.columns%32 || a.variant!=4 || a.first<0) return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
        const uint64_t r=uint64_t(a.rows),c=uint64_t(a.columns),bytes=r/2*c+r*(c/32)*4;
        if(!shape(in[0],{256},DATA_U8) || in[1].geometry.dims!=2 || in[1].geometry.dataType!=DATA_U8 ||
           in[1].geometry.minSizes[0]!=bytes || in[1].geometry.maxSizes[0]!=bytes ||
           in[2].geometry.dims!=1 || in[2].geometry.dataType!=DATA_I32) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        const uint64_t x=in[2].geometry.maxSizes[0],count=outputs[0].geometry.maxSizes[2];
        if(!count || uint64_t(a.first)+count>x || !shape(outputs[0],{r,c,count},DATA_BF16)) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        out.indexSpaceRank=2;out.indexSpaceGeometry[0]=r/256;out.indexSpaceGeometry[1]=count;
    } else if(id==81) {
        // STRATA_HPU_MOE_FP8 tile expansion: id 77's inputs -> F8 {rows, columns, count} + F32 scale {rows, count}.
        Iq4Expand a{};
        if(!params(p.nodeParams,a) || a.rows<=0 || a.rows>65536 || a.rows%1280 || a.columns<=0 || a.columns>65536 ||
           a.columns%32 || a.variant!=4 || a.first<0) return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
        const uint64_t r=uint64_t(a.rows),c=uint64_t(a.columns),bytes=r/2*c+r*(c/32)*4;
        if(!shape(in[0],{256},DATA_U8) || in[1].geometry.dims!=2 || in[1].geometry.dataType!=DATA_U8 ||
           in[1].geometry.minSizes[0]!=bytes || in[1].geometry.maxSizes[0]!=bytes ||
           in[2].geometry.dims!=1 || in[2].geometry.dataType!=DATA_I32) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        const uint64_t x=in[2].geometry.maxSizes[0],count=outputs[0].geometry.maxSizes[2];
        if(!count || uint64_t(a.first)+count>x || !shape(outputs[0],{r,c,count},DATA_F8_143) || !shape(outputs[1],{r,count}))
            return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        out.indexSpaceRank=2;out.indexSpaceGeometry[0]=r/256;out.indexSpaceGeometry[1]=count;
    } else if(id==82) {
        // STRATA_HPU_MOE_FP8 gather: id 75's inputs -> a {hidden, rows, count} F8 + a_scale {64, rows, count} F32.
        MoeMmeGather a{};
        if(!params(p.nodeParams,a) || a.hidden<=0 || a.hidden>4096 || a.hidden%256 || a.rows<=0 || a.rows>4096 || a.per_row<=1 ||
           a.first<0) return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
        const uint64_t hd=uint64_t(a.hidden),r=uint64_t(a.rows),e=in[2].geometry.maxSizes[0],count=outputs[0].geometry.maxSizes[2];
        if(in[0].geometry.dims!=2 || in[0].geometry.dataType!=DATA_F32 || in[0].geometry.maxSizes[0]!=hd ||
           in[0].geometry.minSizes[0]!=hd || in[0].geometry.minSizes[1]!=in[0].geometry.maxSizes[1] ||
           !in[0].geometry.maxSizes[1] || in[0].geometry.maxSizes[1]>kMaxWindowRows || !e || !shape(in[2],{e},DATA_I32) ||
           in[1].geometry.dims!=1 || in[1].geometry.dataType!=DATA_I32 || in[1].geometry.maxSizes[0]<e*r) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if(!count || uint64_t(a.first)+count>e || !shape(outputs[0],{hd,r,count},DATA_F8_143) || !shape(outputs[1],{64,r,count}))
            return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        out.indexSpaceRank=2;out.indexSpaceGeometry[0]=(r+7)/8;out.indexSpaceGeometry[1]=count;
    } else if(id==83) {
        // STRATA_HPU_MOE_FP8 activation: gu {2*ff, rows, batch} BF16 (variant-4 order) + s_gu {2*ff, batch} F32 + a_scale {64, rows, batch}
        // F32 + fill I32 -> act {ff, rows, batch} F8 + act_scale {64, rows, batch} F32. Members are (8 rows, batch).
        MoeMmeAct a{};
        if(!params(p.nodeParams,a) || a.ff<=0 || a.ff>640 || a.ff%128 || (a.use_totals!=0 && a.use_totals!=1) || a.first<0)
            return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
        if(in[0].geometry.dims!=3) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        const uint64_t f=uint64_t(a.ff),r=in[0].geometry.maxSizes[1],b=in[0].geometry.maxSizes[2];
        if(!r || !b || !shape(in[0],{2*f,r,b},DATA_BF16) || !shape(in[1],{2*f,b}) || !shape(in[2],{64,r,b}) ||
           in[3].geometry.dataType!=DATA_I32 || in[3].geometry.dims!=1 || in[3].geometry.minSizes[0]!=in[3].geometry.maxSizes[0] ||
           (a.use_totals && in[3].geometry.maxSizes[0]<uint64_t(a.first)+b)) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if(!shape(outputs[0],{f,r,b},DATA_F8_143) || !shape(outputs[1],{64,r,b})) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        out.indexSpaceRank=2;out.indexSpaceGeometry[0]=(r+7)/8;out.indexSpaceGeometry[1]=b;
    } else if(id==84) {
        // STRATA_HPU_MOE_FP8 join: routed {width, bucket_len} BF16 (variant-4 order) + shared {width, window rows} BF16
        // (variant-4 order) + place {slots} I32 + s_down {width, bucket_len/rows} F32 + act_scale {64, bucket_len} F32
        // -> expert_rows {width, slots} F32. One member per slot.
        MoeMmeJoinFp8 a{};
        if(!params(p.nodeParams,a) || a.per_row<=1 || a.top_k<=0 || a.top_k>=a.per_row || a.bucket_len<=0 || a.rows<=0 ||
           a.bucket_len%a.rows) return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
        const uint64_t width=outputs[0].geometry.maxSizes[0],slots=outputs[0].geometry.maxSizes[1];
        const uint64_t len=uint64_t(a.bucket_len),rows=slots/uint64_t(a.per_row),tiles=len/uint64_t(a.rows);
        if(!width || width%256 || width>65536 || !slots || slots>kMaxMoeSlots || slots%uint64_t(a.per_row))
            return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
        if(!shape(in[0],{width,len},DATA_BF16) || !shape(in[1],{width,rows},DATA_BF16) || !shape(in[2],{slots},DATA_I32) ||
           !shape(in[3],{width,tiles}) || !shape(in[4],{64,len})) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if(!shape(outputs[0],{width,slots})) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        out.indexSpaceGeometry[0]=slots;
    } else if(id==75) {
        // STRATA_HPU_MOE_MME gather: x {hidden, window rows} F32 + bucket {experts*rows} I32 + totals {experts} I32
        // -> a {hidden, rows, experts} BF16. Members are (bucket row, expert).
        MoeMmeGather a{};
        if(!params(p.nodeParams,a) || a.hidden<=0 || a.hidden>65536 || a.hidden%128 || a.rows<=0 || a.rows>4096 || a.per_row<=1 ||
           a.first<0) return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
        const uint64_t hd=uint64_t(a.hidden),r=uint64_t(a.rows),e=in[2].geometry.maxSizes[0],count=outputs[0].geometry.maxSizes[2];
        if(in[0].geometry.dims!=2 || in[0].geometry.dataType!=DATA_F32 || in[0].geometry.maxSizes[0]!=hd ||
           in[0].geometry.minSizes[0]!=hd || in[0].geometry.minSizes[1]!=in[0].geometry.maxSizes[1] ||
           !in[0].geometry.maxSizes[1] || in[0].geometry.maxSizes[1]>kMaxWindowRows || !e || !shape(in[2],{e},DATA_I32) ||
           in[1].geometry.dims!=1 || in[1].geometry.dataType!=DATA_I32 || in[1].geometry.maxSizes[0]<e*r) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if(!count || uint64_t(a.first)+count>e || !shape(outputs[0],{hd,r,count},DATA_BF16)) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        out.indexSpaceRank=2;out.indexSpaceGeometry[0]=(r+7)/8;out.indexSpaceGeometry[1]=count;
    } else if(id==66) {
        RouteGroup a{};
        if(!params(p.nodeParams,a) || a.experts<=0 || a.experts>4096 || a.slots<=1 || a.slots>int32_t(kMaxMoeSlots))
            return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
        const uint64_t e=a.experts,s=a.slots,b=(s+63)/64;
        if(!shape(in[0],{s},DATA_I32) || !shape(in[1],{e+1}) || !shape(in[2],{b,e})) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if(!shape(outputs[0],{s})) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        out.indexSpaceGeometry[0]=e;
    } else if(id==50) {
        QsaVirtual a{};
        if(!params(p.nodeParams,a) || a.width<=0 || a.width>65536 || a.width%64 || a.heads<=0 || a.heads>1024 ||
           a.cache_cells<=0 || a.cache_cells>262144 || a.tokens<=0 || a.tokens>128 || a.first<0 || a.count<=0 ||
           a.first+a.count>a.tokens || a.bucket<=0 || a.bucket>262144) return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
        const uint64_t d=a.width,h=a.heads,c=a.cache_cells,t=a.tokens,k=a.bucket+a.count;
        if(!shape(in[0],{d,h,c}) || !shape(in[1],{d,h,c}) || !shape(in[2],{d,h,t}) || !shape(in[3],{d,h,t}) ||
           !shape(in[4],{1})) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if(!shape(outputs[0],{d,k,h}) || !shape(outputs[1],{d,k,h})) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        out.indexSpaceGeometry[0]=h*k;
    } else if(id==48) {
        GdnConvRows a{};
        if(!params(p.nodeParams,a) || a.channels<=0 || a.channels>1048576 || a.channels%64 || a.taps<2 || a.taps>4 || a.rows<1 || a.rows>int32_t(kMaxWindowRows))
            return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
        const uint64_t c=a.channels,d=a.taps,r=a.rows;
        if(!shape(in[0],{c,d-1}) || !shape(in[1],{c,r}) || !shape(in[2],{c,d})) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if(!shape(outputs[0],{c,d-1}) || !shape(outputs[1],{c,r})) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        out.indexSpaceRank=2;out.indexSpaceGeometry[0]=c/64;out.indexSpaceGeometry[1]=(r+63)/64;
    } else if(id==49) {
        GdnStepRows a{};
        if(!params(p.nodeParams,a) || !width(a.width) || a.width>128 || a.key_heads<=0 || a.rows<1 || a.rows>int32_t(kMaxWindowRows)) return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
        const uint64_t w=a.width,hk=a.key_heads,hv=in[3].geometry.maxSizes[1],r=a.rows;
        if(!hv || hv>1024 || hv%hk || !shape(in[0],{w,hv,w}) || !shape(in[1],{w,hk,r}) || !shape(in[2],{w,hk,r}) ||
           !shape(in[3],{w,hv,r}) || !shape(in[4],{hv,r}) || !shape(in[5],{hv,r})) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if(!shape(outputs[0],{w,hv,w}) || !shape(outputs[1],{w,hv,r})) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        out.indexSpaceRank=2;out.indexSpaceGeometry[0]=w/32;out.indexSpaceGeometry[1]=hv;
    } else if(id==98) {
        // STRATA_HPU_QSA_DECODE_SELECT window block scores: scores {blocks, heads, rows} + cell {1} -> {blocks, rows}.
        QsaWindowBlocks a{};
        if(!params(p.nodeParams,a) || a.blocks<=0 || a.blocks%64 || a.blocks>1048576 || a.heads<=0 || a.heads>64 || a.block<1 || a.block>64 ||
           a.rows<=0 || a.rows>64) return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
        const uint64_t b=uint64_t(a.blocks),r=uint64_t(a.rows);
        if(!shape(in[0],{b,uint64_t(a.heads),r}) || !shape(in[1],{1})) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if(!shape(outputs[0],{b,r})) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        out.indexSpaceRank=2;out.indexSpaceGeometry[0]=b/64;out.indexSpaceGeometry[1]=r;
    } else if(id==100) {
        // STRATA_HPU_HC_DECODE_TPC down/inject: x {n, 1} BF16, w {n, rank + streams} BF16, inverse_streams {1}
        // -> lo {rank, 1} BF16, inject {streams, 1} F32. Members own eight output rows (tpc/hc_down_bf16.c).
        HcDown a{};
        if(!params(p.nodeParams,a) || a.columns<=0 || a.columns%128 || a.rank<=0 || a.rank%8 || a.streams<=0 || a.streams>8)
            return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
        const uint64_t n=uint64_t(a.columns),t=uint64_t(a.rank)+uint64_t(a.streams);
        if(!shape(in[0],{n,1},DATA_BF16) || !shape(in[1],{n,t},DATA_BF16) || !shape(in[2],{1})) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if(!shape(outputs[0],{uint64_t(a.rank),1},DATA_BF16) || !shape(outputs[1],{uint64_t(a.streams),1})) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        out.indexSpaceGeometry[0]=(t+7)/8;
    } else if(id==108) {
        // STRATA_HPU_HC_WINDOW_TPC: id 100 over rows: x {n, rows} BF16, w {n, rank + streams} BF16, inverse_streams
        // {1} -> lo {rank, rows} BF16, inject {streams, rows} F32. Members own two output rows (tpc/hc_down_rows_bf16.c).
        HcDownRows a{};
        if(!params(p.nodeParams,a) || a.columns<=0 || a.columns%256 || a.rank<=0 || a.rank%2 || a.streams<=0 || a.streams>8 ||
           a.rows<1 || a.rows>64) return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
        const uint64_t n=uint64_t(a.columns),t=uint64_t(a.rank)+uint64_t(a.streams),r=uint64_t(a.rows);
        if(!shape(in[0],{n,r},DATA_BF16) || !shape(in[1],{n,t},DATA_BF16) || !shape(in[2],{1})) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if(!shape(outputs[0],{uint64_t(a.rank),r},DATA_BF16) || !shape(outputs[1],{uint64_t(a.streams),r})) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        out.indexSpaceGeometry[0]=(t+1)/2;
    } else if(id==101) {
        // STRATA_HPU_HC_DECODE_TPC mix: id 5 with the BF16 copy {width, 1} and the packed Q8 codes {256, width/16, 1}
        // U8 and scales {64, width/32, 1} of id 48. Members own 64 columns.
        HcMixDual a{};
        if(!params(p.nodeParams,a) || a.width<=0 || a.width>65536 || a.width%64 || a.streams<=0 || a.streams>32 || (a.codes!=0 && a.codes!=1))
            return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
        const uint64_t w=uint64_t(a.width),r=uint64_t(a.streams);
        if(!shape(in[0],{w,r}) || !shape(in[1],{w,r})) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        // Without codes the code/scale outputs are one-block placeholders the kernel leaves alone.
        if(!shape(outputs[0],{w}) || !shape(outputs[1],{w,1},DATA_BF16) || !shape(outputs[2],{256,a.codes?w/16:1,1},DATA_U8) ||
           !shape(outputs[3],{64,a.codes?w/32:1,1})) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        out.indexSpaceGeometry[0]=w/64;
    } else if(id==102) {
        // STRATA_HPU_GEMV_TPC: x {columns, 1} BF16, w {columns, rows} BF16 -> out {rows, 1} F32 or BF16 (rounded).
        // Members own eight output rows (tpc/gemv_bf16.c); x lives in 64 local vectors.
        GemvBf16 a{};
        if(!params(p.nodeParams,a) || a.columns<=0 || a.columns%128 || a.columns>8192 || a.rows<=0 || (a.rounded!=0 && a.rounded!=1))
            return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
        const uint64_t c=uint64_t(a.columns),r=uint64_t(a.rows);
        if(!shape(in[0],{c,1},DATA_BF16) || !shape(in[1],{c,r},DATA_BF16)) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if(!shape(outputs[0],{r,1},a.rounded?DATA_BF16:DATA_F32)) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        out.indexSpaceGeometry[0]=(r+7)/8;
    } else if(id==103) {
        // PLE gate over a window: key/query {width, rows} (rows = window rows * streams), value {width, rows/streams}.
        PleGateRows a{};
        if(!params(p.nodeParams,a) || a.width<=0 || a.width>65536 || a.width%64 || a.streams<=0 || a.streams>32 || a.rows<=0 ||
           a.rows%a.streams || a.rows>int32_t(kMaxWindowRows)*a.streams || !epsilon(a.inverse_sqrt_width)) return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
        const uint64_t w=a.width,r=a.rows;
        if(!shape(in[0],{w,r}) || !shape(in[1],{w,r}) || !shape(in[2],{w,r/uint64_t(a.streams)})) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if(!shape(outputs[0],{w,r})) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        out.indexSpaceGeometry[0]=r;
    } else if(id==104) {
        // PLE causal conv over a window: history {c, (taps-1)*dilation}, normalized/hidden/gated {c, rows},
        // weights {c, taps} -> next {c, depth}, output {c, rows}. Members: (64 channels, 64 rows).
        PleConvRows a{};
        if(!params(p.nodeParams,a) || a.channels<=0 || a.channels%64 || a.taps<2 || a.taps>16 || a.dilation<1 || a.dilation>16 ||
           a.rows<=0 || a.rows>int32_t(kMaxWindowRows)) return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
        const uint64_t c=a.channels,d=uint64_t(a.taps-1)*a.dilation,r=a.rows;
        if(!shape(in[0],{c,d}) || !shape(in[1],{c,r}) || !shape(in[2],{c,uint64_t(a.taps)}) || !shape(in[3],{c,r}) || !shape(in[4],{c,r}))
            return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if(!shape(outputs[0],{c,d}) || !shape(outputs[1],{c,r})) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        out.indexSpaceRank=2;out.indexSpaceGeometry[0]=c/64;out.indexSpaceGeometry[1]=(r+63)/64;
    } else if(id==99) {
        // STRATA_HPU_QSA_DECODE_SELECT window gather: as id 97 with a row axis (top ids/scores {top, rows}).
        QsaGatherWindow a{};
        if(!params(p.nodeParams,a) || a.heads<=0 || a.heads>32 || a.cells<=0 || a.cells>262144 || a.top<=0 || a.top>4096 ||
           a.block<1 || a.block>64 || a.width_sel<=1 || a.rows<=0 || a.rows>64) return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
        const uint64_t h=uint64_t(a.heads),c=uint64_t(a.cells),k=uint64_t(a.top),n=(k+1)*uint64_t(a.block),r=uint64_t(a.rows);
        if(!shape(in[0],{256,h,c},DATA_F8_143) || !shape(in[1],{256,h,c},DATA_F8_143) || !shape(in[2],{32,c}) || !shape(in[3],{32,c}) ||
           !shape(in[4],{k,r},DATA_I32) || !shape(in[5],{k,r}) || !shape(in[6],{1})) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if(!shape(outputs[0],{256,n,h,r},DATA_F8_143) || !shape(outputs[1],{256,n,h,r},DATA_F8_143) || !shape(outputs[2],{n,h,r}) ||
           !shape(outputs[3],{n,h,r}) || !shape(outputs[4],{n,r})) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        out.indexSpaceRank=3;out.indexSpaceGeometry[0]=k+1;out.indexSpaceGeometry[1]=h;out.indexSpaceGeometry[2]=r;
    } else if(id==109) {
        // STRATA_HPU_QSA_VERIFY_BF16: id 99 writing BF16 codes; members own 16 ranks. block == 4.
        QsaGatherWindow a{};
        if(!params(p.nodeParams,a) || a.heads<=0 || a.heads>32 || a.cells<=0 || a.cells>262144 || a.top<=0 || a.top>4096 ||
           a.block!=4 || a.width_sel<=1 || a.rows<=0 || a.rows>64) return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
        const uint64_t h=uint64_t(a.heads),c=uint64_t(a.cells),k=uint64_t(a.top),n=(k+1)*4,r=uint64_t(a.rows);
        if(!shape(in[0],{256,h,c},DATA_F8_143) || !shape(in[1],{256,h,c},DATA_F8_143) || !shape(in[2],{32,c}) || !shape(in[3],{32,c}) ||
           !shape(in[4],{k,r},DATA_I32) || !shape(in[5],{k,r}) || !shape(in[6],{1})) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if(!shape(outputs[0],{256,n,h,r},DATA_BF16) || !shape(outputs[1],{256,n,h,r},DATA_BF16) || !shape(outputs[2],{n,h,r}) ||
           !shape(outputs[3],{n,h,r}) || !shape(outputs[4],{n,r})) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        out.indexSpaceRank=3;out.indexSpaceGeometry[0]=(k+1+15)/16;out.indexSpaceGeometry[1]=h;out.indexSpaceGeometry[2]=r;
    } else if(id==110 || id==111) {
        // STRATA_HPU_QSA_VERIFY_BF16 probabilities (tpc/qsa_verify_softmax.c): scores_w {rows, group, B}, scores_c
        // {cells, group, B}, key scale {cells, B}, (110: value scale {cells, B}), mask {rows + cells, rows}, (110: parts)
        // -> 111: parts {chunks, 2, group, B}; 110: p_w {rows, group, B} F32, p16 {cells, group, B} BF16. B = rows * heads.
        QsaVerifySoftmax a{};
        if(!params(p.nodeParams,a) || a.rows<=0 || a.rows>64 || a.heads<=0 || a.heads>32 || a.group<=0 || a.group>64 ||
           a.cells<=0 || a.cells>16384) return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
        const uint64_t t=uint64_t(a.rows),b=t*uint64_t(a.heads),gq=uint64_t(a.group),n=uint64_t(a.cells),k=(n+255)/256;
        const bool last=id==110;const unsigned m=last?4:3;
        if(!shape(in[0],{t,gq,b}) || !shape(in[1],{n,gq,b}) || !shape(in[2],{n,b}) || (last && !shape(in[3],{n,b})) ||
           !shape(in[m],{t+n,t}) || (last && !shape(in[5],{k,2,gq,b}))) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if(last ? (!shape(outputs[0],{t,gq,b}) || !shape(outputs[1],{n,gq,b},DATA_BF16) || !shape(outputs[2],{n,gq,b})) : !shape(outputs[0],{k,2,gq,b}))
            return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        out.indexSpaceRank=3;out.indexSpaceGeometry[0]=k;out.indexSpaceGeometry[1]=gq;out.indexSpaceGeometry[2]=b;
    } else if(id==96) {
        // STRATA_HPU_QSA_DECODE_SELECT block scores: scores {blocks, heads} + cell {1} -> {blocks}. Members: 64 blocks.
        QsaDecodeBlocks a{};
        if(!params(p.nodeParams,a) || a.blocks<=0 || a.blocks%64 || a.blocks>1048576 || a.heads<=0 || a.heads>64 || a.block<1 || a.block>64)
            return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
        const uint64_t b=uint64_t(a.blocks);
        if(!shape(in[0],{b,uint64_t(a.heads)}) || !shape(in[1],{1})) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if(!shape(outputs[0],{b})) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        out.indexSpaceGeometry[0]=b/64;
    } else if(id==97) {
        // STRATA_HPU_QSA_DECODE_SELECT gather: FP8 cache {256, heads, cells} x2 + scales {32, cells} x2 + top ids {top} I32
        // + top scores {top} + cell {1} -> FP8 key/value {256, n, heads}, scales {n, heads} x2, mask {n}; n = (top+1)*block.
        QsaGatherBlocks a{};
        if(!params(p.nodeParams,a) || a.heads<=0 || a.heads>32 || a.cells<=0 || a.cells>262144 || a.top<=0 || a.top>4096 ||
           a.block<1 || a.block>64 || a.width_sel<=1) return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
        const uint64_t h=uint64_t(a.heads),c=uint64_t(a.cells),k=uint64_t(a.top),n=(k+1)*uint64_t(a.block);
        if(!shape(in[0],{256,h,c},DATA_F8_143) || !shape(in[1],{256,h,c},DATA_F8_143) || !shape(in[2],{32,c}) || !shape(in[3],{32,c}) ||
           !shape(in[4],{k},DATA_I32) || !shape(in[5],{k}) || !shape(in[6],{1})) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if(!shape(outputs[0],{256,n,h},DATA_F8_143) || !shape(outputs[1],{256,n,h},DATA_F8_143) || !shape(outputs[2],{n,h}) ||
           !shape(outputs[3],{n,h}) || !shape(outputs[4],{n})) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        out.indexSpaceRank=2;out.indexSpaceGeometry[0]=k+1;out.indexSpaceGeometry[1]=h;
    } else if(id==95) {
        // STRATA_HPU_QSA_KV_FP8 gather: FP8 keys/values {256, heads, cells} + scales {32, cells} F32 + ids {bucket, rows}
        // -> FP8 key/value {256, bucket, rows*heads}, their scales {bucket, rows*heads} F32 and the mask (as id 46).
        QsaGather a{};
        if(!params(p.nodeParams,a) || a.width!=256 || a.heads<=0 || a.heads>1024 || a.cells<=0 || a.cells>262144 || a.bucket<=0 ||
           a.bucket>a.cells || a.rows<=0 || a.rows>int32_t(kMaxWindowRows) || (a.mask_heads!=1 && a.mask_heads!=a.heads))
            return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
        const uint64_t d=a.width,h=a.heads,c=a.cells,b=a.bucket,r=a.rows;
        // The cache scales sit in a 32-slot (128-byte) row per cell (qsa.hpp kQsaScaleSlots).
        if(!shape(in[0],{d,h,c},DATA_F8_143) || !shape(in[1],{d,h,c},DATA_F8_143) || h>32 || !shape(in[2],{32,c}) || !shape(in[3],{32,c}) ||
           !shape(in[4],{b,r})) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if(!shape(outputs[0],{d,b,r*h},DATA_F8_143) || !shape(outputs[1],{d,b,r*h},DATA_F8_143) || !shape(outputs[2],{b,r*h}) ||
           !shape(outputs[3],{b,r*h}) || !((r==1 && a.mask_heads==1 && shape(outputs[4],{b})) ||
             shape(outputs[4],{b,1,r*uint64_t(a.mask_heads)}))) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        out.indexSpaceGeometry[0]=r*h*b;
    } else if(id==46) {
        QsaGather a{};
        if(!params(p.nodeParams,a) || a.width<=0 || a.width>65536 || a.width%64 || a.heads<=0 || a.heads>1024 ||
           a.cells<=0 || a.cells>262144 || a.bucket<=0 || a.bucket>a.cells || a.rows<=0 || a.rows>int32_t(kMaxWindowRows) ||
           (a.mask_heads!=1 && a.mask_heads!=a.heads)) return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
        const uint64_t d=a.width,h=a.heads,c=a.cells,b=a.bucket,r=a.rows;
        if(!shape(in[0],{d,h,c}) || !shape(in[1],{d,h,c}) || !shape(in[2],{b,r})) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if(!shape(outputs[0],{d,b,r*h}) || !shape(outputs[1],{d,b,r*h}) ||
           !((r==1 && a.mask_heads==1 && shape(outputs[2],{b})) ||
             shape(outputs[2],{b,1,r*uint64_t(a.mask_heads)}))) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        out.indexSpaceGeometry[0]=r*h*b;
    } else if(id==65) {
        // Bandwidth diagnostic (hpu/tests/bw_read.cpp): `members` index-space members each read `vectors`
        // 256-byte vectors of one flat F32 tensor into their own sink row. No model graph declares this node.
        BwProbe a{};
        if(!params(p.nodeParams,a) || a.members<=0 || a.members>65536 || a.vectors<=0 || a.vectors>1048576 ||
           (a.chains!=1 && a.chains!=2 && a.chains!=4 && a.chains!=8) || a.vectors%a.chains ||
           (a.ops!=0 && a.ops!=7) || a.offset<0) return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
        // Range checks, not equalities: `offset` lets several nodes share one pair of tensors, which is how the
        // test asks whether the fixed cost is charged per graph or per node.
        const uint64_t reach=(uint64_t(a.offset)+uint64_t(a.members))*uint64_t(a.vectors)*64;
        if(in[0].geometry.dataType!=DATA_F32 || in[0].geometry.dims!=1 ||
           in[0].geometry.minSizes[0]!=in[0].geometry.maxSizes[0] || in[0].geometry.minSizes[0]<reach)
            return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if(!shape(outputs[0],{uint64_t(a.members)*64})) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        out.indexSpaceGeometry[0]=uint64_t(a.members);
    } else if (id == 0) {
        GdnStep a{};
        if (!params(p.nodeParams, a) || !width(a.width) || a.key_heads <= 0) return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
        const uint64_t s = a.width, hk = a.key_heads, hv = in[3].geometry.maxSizes[1];
        // The broadcast variant reads q/k as {64 lanes, s, hk}: every lane of row (i, head) holds q[i]/k[i].
        const bool broadcast = kernel_id == 18;
        if (!hv || hv % hk || !shape(in[0], {s,hv,s}) || !(broadcast ? shape(in[1], {64,s,hk}) : shape(in[1], {s,hk})) ||
            !(broadcast ? shape(in[2], {64,s,hk}) : shape(in[2], {s,hk})) || !shape(in[3], {s,hv}) || !shape(in[4], {hv}) || !shape(in[5], {hv}))
            return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if (!shape(outputs[0], {s,hv,s}) || !shape(outputs[1], {s,hv})) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        out.indexSpaceGeometry[0] = s / 64; out.indexSpaceGeometry[1] = hv;
    } else if (id == 1) {
        GdnConv a{};
        if (!params(p.nodeParams, a) || a.channels <= 0 || a.channels > 1048576 || a.taps < 2 || a.taps > 16)
            return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
        const uint64_t c = a.channels, d = a.taps;
        if (!shape(in[0], {c,d-1}) || !shape(in[1], {c}) || !shape(in[2], {c,d})) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if (!shape(outputs[0], {c,d-1}) || !shape(outputs[1], {c})) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        out.indexSpaceGeometry[0] = (c+63)/64;
    } else if (id == 2) {
        GdnPrepare a{};
        if (!params(p.nodeParams, a) || !width(a.width) || a.key_heads <= 0 || a.value_heads <= 0 ||
            a.value_heads > 1024 || a.value_heads % a.key_heads || !epsilon(a.epsilon)) return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
        const uint64_t s = a.width, hk = a.key_heads, hv = a.value_heads, c = s*(2*hk+hv);
        if (!shape(in[0], {c})) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        for (unsigned i = 1; i < 5; ++i) if (!shape(in[i], {hv})) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if (!shape(outputs[0], {s,hk}) || !shape(outputs[1], {s,hk}) || !shape(outputs[2], {s,hv}) ||
            !shape(outputs[3], {hv}) || !shape(outputs[4], {hv})) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        out.indexSpaceGeometry[0] = hv;
    } else if (id == 3) {
        GdnNorm a{};
        if (!params(p.nodeParams, a) || !width(a.width) || a.value_heads <= 0 || a.value_heads > 1024 || !epsilon(a.epsilon))
            return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
        const uint64_t s = a.width, hv = a.value_heads;
        if (!shape(in[0], {s,hv}) || !shape(in[1], {s,hv}) || !shape(in[2], {s})) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if (!shape(outputs[0], {s,hv})) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        out.indexSpaceGeometry[0] = hv;
    } else if(id==14) {
        Fp8Encode a{};
        if(!params(p.nodeParams,a) || a.columns<=0 || a.columns>65536 || a.columns%64 || a.rows<=0 || a.rows>int32_t(kMaxMoeSlots))
            return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
        const uint64_t c=a.columns,r=a.rows;
        if(!shape(in[0],{c,r})) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if(!shape(outputs[0],{c,r},DATA_F8_143) || !shape(outputs[1],{c,r},DATA_F8_143) ||
           !shape(outputs[2],{1,r})) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        out.indexSpaceGeometry[0]=r;
    } else if(id==15 || id==19 || id==20) {
        // x {padded columns, 1 or 11} BF16 (permuted per 256-column chunk), 11 flat U8 expert buffers of
        // rows*columns codes + rows F32 scales, out {rows, 11} F32. One member per (expert, 64-row block).
        Fp8Gemv a{};
        if(!params(p.nodeParams,a) || a.columns<=0 || a.columns>2560 || a.columns%64 || a.rows<=0 || a.rows%64 || a.rows>65536 ||
           (a.shared_x!=0 && a.shared_x!=1)) return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
        const uint64_t c=a.columns,r=a.rows,padded=(c+255)/256*256,experts=11;
        if(id==20 && (a.rows%128 || a.columns%128)) return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
        if(!shape(in[0],{id==20?c:padded,a.shared_x?1:experts},DATA_BF16)) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if(id==19 && (!shape(in[1],{padded,a.shared_x?1:experts},DATA_BF16) ||
                      !shape(in[2],{1,a.shared_x?1:experts}))) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        for(unsigned i=id==19?3:1;i<(id==19?14u:12u);++i)
            if(!shape(in[i],{r*c+r*4},DATA_U8)) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if(!shape(outputs[0],{r,experts})) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        out.indexSpaceGeometry[0]=r/(id==20?128:64)*experts;
    } else if(id==51) {
        // STRATA_HPU_MOE_GROUP: ids {slots} I32 -> starts {experts+1} F32, one member per expert (plus the end).
        RouteGroup a{};
        if(!params(p.nodeParams,a) || a.experts<=0 || a.experts>4096 || a.slots<=1 || a.slots>int32_t(kMaxMoeSlots))
            return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
        if(!shape(in[0],{uint64_t(a.slots)},DATA_I32)) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if(!shape(outputs[0],{uint64_t(a.experts)+1})) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        out.indexSpaceGeometry[0]=uint64_t(a.experts)+1;
    } else if(id==52) {
        // STRATA_HPU_MOE_GROUP: ids {slots} I32 + starts {experts+1} F32 -> order {slots} F32, one member per expert.
        RouteGroup a{};
        if(!params(p.nodeParams,a) || a.experts<=0 || a.experts>4096 || a.slots<=1 || a.slots>int32_t(kMaxMoeSlots))
            return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
        if(!shape(in[0],{uint64_t(a.slots)},DATA_I32) || !shape(in[1],{uint64_t(a.experts)+1}))
            return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if(!shape(outputs[0],{uint64_t(a.slots)})) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        out.indexSpaceGeometry[0]=uint64_t(a.experts);
    } else if(id==53) {
        // iq4q8_gemv_packed's tensors (x packed as {256, columns/16, slots/x_group}) plus the slot order
        // {slots} F32 after `shared`; members are (part, row group, position) and the store index is the slot.
        Iq4Gemv a{};
        if(!params(p.nodeParams,a) || a.columns<=0 || a.columns>65536 || a.columns%32 || a.rows<=0 || a.rows>65536 || a.rows%1280 ||
           a.x_group<=0 || a.slots%a.x_group || a.period<=1 || a.slots%a.period || a.parts<=0 || (a.columns/32)%a.parts ||
           a.slots<=1 || a.slots>int32_t(kMaxMoeSlots)) return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
        const uint64_t c=a.columns,r=a.rows,s=a.slots,xrows=s/a.x_group,bytes=r/2*c+r*(c/32)*4;
        if(!shape(in[0],{256,c/16,xrows},DATA_U8) || !shape(in[1],{64,c/32,xrows}) || !shape(in[2],{256},DATA_U8) ||
           !shape(in[3],{s},DATA_I32) || in[4].geometry.dims!=2 || in[4].geometry.maxSizes[0]!=bytes ||
           in[4].geometry.dataType!=DATA_U8 || !shape(in[5],{bytes,1},DATA_U8) || !shape(in[6],{s}))
            return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if(!shape(outputs[0],{r,s,uint64_t(a.parts)})) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        out.indexSpaceGeometry[0]=kernel_id==67?((s+1)/2)*a.parts*(r/1280):s*a.parts*(r/1280);
    } else if(id==55) {
        // STRATA_HPU_GDN_WINDOW: a whole window's recurrent steps in one launch. A member owns one (64 value
        // lanes, head) state slice and walks every row; q/k/v/gate/beta are the per-row chain's row-major
        // tensors, so the index space is gdn_step's {width/64, heads}.
        GdnStepRows a{};
        if(!params(p.nodeParams,a) || !width(a.width) || a.width>128 || a.key_heads<=0 || a.rows<1 || a.rows>int32_t(kMaxWindowRows))
            return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
        const uint64_t w=a.width,hk=a.key_heads,hv=in[3].geometry.maxSizes[1],r=a.rows;
        if(!hv || hv>1024 || hv%hk || !shape(in[0],{w,hv,w}) || !shape(in[1],{w,hk,r}) || !shape(in[2],{w,hk,r}) ||
           !shape(in[3],{w,hv,r}) || !shape(in[4],{hv,r}) || !shape(in[5],{hv,r})) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if(!shape(outputs[0],{w,hv,w}) || !shape(outputs[1],{w,hv,r})) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        out.indexSpaceRank=2;out.indexSpaceGeometry[0]=w/64;out.indexSpaceGeometry[1]=hv;
    } else if(id==63 || id==64) {
        Fp8Batch a{};
        if(!params(p.nodeParams,a) || a.columns<=0 || a.columns>65536 || a.columns%64 ||
           a.rows<=0 || a.rows>4096 || (a.terms!=1 && a.terms!=2)) return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
        const uint64_t c=a.columns,r=a.rows,t=a.terms;
        if(in[0].geometry.dims!=3 || !in[0].geometry.minSizes[2] ||
           in[0].geometry.minSizes[2]!=in[0].geometry.maxSizes[2]) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        const uint64_t b=in[0].geometry.minSizes[2];
        if(id==63) {
            if(!shape(in[0],{c,r,b}) || !shape(outputs[0],{c,t*r,b},DATA_F8_143) || !shape(outputs[1],{1,r,b}))
                return GLUE_INCOMPATIBLE_INPUT_SIZE;
            out.indexSpaceGeometry[0]=b;
        } else {
            if(!shape(in[0],{c,t*r,b}) || !shape(in[1],{1,r,b}) || !shape(outputs[0],{c,r,b}))
                return GLUE_INCOMPATIBLE_INPUT_SIZE;
            out.indexSpaceGeometry[0]=b;
        }
    } else if(id==61) {
        RouteGroup a{};
        if(!params(p.nodeParams,a) || a.experts<=0 || a.experts>4096 || a.slots<=0 || a.slots>int32_t(kMaxMoeSlots))
            return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
        const uint64_t blocks=(uint64_t(a.slots)+63)/64;
        if(!shape(in[0],{uint64_t(a.slots)},DATA_I32) || !shape(outputs[0],{blocks,uint64_t(a.experts)}))
            return GLUE_INCOMPATIBLE_INPUT_SIZE;
        out.indexSpaceGeometry[0]=uint64_t(a.experts)*blocks;
    } else if(id==56 || id==62) {
        // STRATA_HPU_RESIDENT_U4: starts {experts+1} F32 + order {slots} F32 -> bucket {experts*rows} I32,
        // overflow {experts} I32 and place {slots} I32 (the slot -> bucket-position inverse). One member per
        // expert, which walks its run in the grouped slot order.
        U4MoeBucket a{};
        if(!params(p.nodeParams,a) || a.experts<=0 || a.experts>4096 || a.rows<=0 || a.rows>4096 ||
           a.slots<=1 || a.slots>int32_t(kMaxMoeSlots) || a.per_row<=1 || a.top_k<=0 || a.top_k>=a.per_row)
            return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
        if(id==62) {
            if(!shape(in[0],{uint64_t(a.slots)},DATA_I32) ||
               !shape(in[1],{(uint64_t(a.slots)+63)/64,uint64_t(a.experts)})) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        } else if(!shape(in[0],{uint64_t(a.experts)+1}) || !shape(in[1],{uint64_t(a.slots)}))
            return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if(!shape(outputs[0],{uint64_t(a.experts)*uint64_t(a.rows)},DATA_I32) ||
           !shape(outputs[1],{uint64_t(a.experts)},DATA_I32) ||
           !shape(outputs[2],{uint64_t(a.slots)},DATA_I32)) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        out.indexSpaceGeometry[0]=uint64_t(a.experts);
    } else if(id==57) {
        // x {x_rows, groups*group} F32 + map {labelled*rows} I32 -> a {batch, rows, group} BF16 in
        // [batch][rows][group] order and sumx {batch*rows} F32, batch = labelled*groups. One member per batch
        // entry; hidden = groups*group.
        U4MoeGather a{};
        if(!params(p.nodeParams,a) || a.groups<=0 || a.groups>1024 || a.group<=0 || a.group>65536 ||
           a.group%64 || a.rows<=0 || a.rows>4096 || a.per_row<=0) return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
        const uint64_t k=uint64_t(a.groups)*uint64_t(a.group),r=uint64_t(a.rows);
        if(in[0].geometry.dataType!=DATA_F32 || in[0].geometry.dims!=2 || in[0].geometry.minSizes[0]!=k ||
           in[0].geometry.maxSizes[0]!=k || in[0].geometry.minSizes[1]!=in[0].geometry.maxSizes[1] ||
           !in[0].geometry.maxSizes[1] || in[0].geometry.maxSizes[1]>kMaxWindowRows) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if(in[1].geometry.dataType!=DATA_I32 || in[1].geometry.dims!=1 ||
           in[1].geometry.minSizes[0]!=in[1].geometry.maxSizes[0] || !in[1].geometry.minSizes[0] ||
           in[1].geometry.minSizes[0]%r) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        const uint64_t labelled=in[1].geometry.minSizes[0]/r,batch=labelled*uint64_t(a.groups);
        if(!shape(outputs[0],{uint64_t(a.group),r,batch},DATA_BF16) || !shape(outputs[1],{batch*r}))
            return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        out.indexSpaceGeometry[0]=batch;
    } else if(id==60) {
        // routed {slots, width} F32 + shared {slots, width} F32 + bucket {experts*rows} I32 and
        // place {slots} I32 -> expert_rows {slots, width} F32. One member per (64-lane column block, slot).
        U4MoeJoin a{};
        if(!params(p.nodeParams,a) || a.per_row<=1 || a.top_k<=0 || a.top_k>=a.per_row || a.bucket_len<=0)
            return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
        if(in[0].geometry.dataType!=DATA_F32 || in[0].geometry.dims!=2 || in[1].geometry.dataType!=DATA_F32 ||
           in[1].geometry.dims!=2 || in[2].geometry.dataType!=DATA_I32 || in[2].geometry.dims!=1 ||
           in[2].geometry.minSizes[0]!=uint64_t(a.bucket_len) || in[2].geometry.maxSizes[0]!=uint64_t(a.bucket_len) ||
           in[3].geometry.dataType!=DATA_I32 || in[3].geometry.dims!=1) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        const uint64_t width=in[0].geometry.maxSizes[0],slots=in[0].geometry.maxSizes[1];
        if(!width || width%64 || width>65536 || !slots || slots>kMaxMoeSlots || slots%uint64_t(a.per_row))
            return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
        if(in[0].geometry.minSizes[0]!=width || in[0].geometry.minSizes[1]!=slots ||
           in[1].geometry.minSizes[0]!=width || in[1].geometry.maxSizes[0]!=width ||
           in[1].geometry.minSizes[1]!=slots || in[1].geometry.maxSizes[1]!=slots ||
           in[3].geometry.minSizes[0]!=slots || in[3].geometry.maxSizes[0]!=slots)
            return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if(!shape(outputs[0],{width,slots},DATA_F32)) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        out.indexSpaceRank=2;out.indexSpaceGeometry[0]=width/64;out.indexSpaceGeometry[1]=slots;
    } else if(id==58) {
        // out_gu {2*ff, rows, batch} F32, scale/offset {2*ff, batch} F32, map {labelled*rows} I32,
        // sumx {batch*rows} F32 -> a_down {width, rows, labelled*down_groups} BF16 and the column-group sums
        // partial {labelled*down_groups*rows*width/64} F32. Members are (labelled row, 64-column block, 8 rows).
        U4MoeGuEpi a{};
        if(!params(p.nodeParams,a) || a.experts<=0 || a.experts>4096 || a.groups<=0 || a.groups>1024 ||
           a.ff<=0 || a.ff>65536 || a.ff%64 || a.rows<=0 || a.rows>4096 || a.width<=0 || a.width>65536 ||
           a.width%64 || a.down_groups<=0 || a.down_groups>1024 || a.ff%a.width)
            return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
        const uint64_t n=2*uint64_t(a.ff),r=uint64_t(a.rows),e=uint64_t(a.experts);
        const uint64_t batch=e*uint64_t(a.groups),bdown=e*uint64_t(a.down_groups);
        if(!shape(in[0],{n,r,batch}) || !shape(in[1],{n,batch}) || !shape(in[2],{n,batch}) ||
           !shape(in[3],{e*r},DATA_I32) || !shape(in[4],{batch*r})) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if(!shape(outputs[0],{uint64_t(a.width),r,bdown},DATA_BF16) || !shape(outputs[1],{bdown*r}))
            return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        out.indexSpaceGeometry[0]=e*uint64_t(a.down_groups)*((r+7)/8);
    } else if(id==59) {
        // out_down {hidden, rows, batch} F32, scale/offset {hidden, batch} F32, map {labelled*rows} I32,
        // partial {batch*rows*width/64} F32, route {slots,1} F32 -> expert_rows {hidden, slots} F32.
        U4MoeDownEpi a{};
        if(!params(p.nodeParams,a) || a.experts<=0 || a.experts>4096 || a.groups<=0 || a.groups>1024 ||
           a.n<=0 || a.n>65536 || a.n%64 || a.rows<=0 || a.rows>4096 || a.width<=0 || a.width>65536 ||
           a.width%64 || (a.apply_route!=0 && a.apply_route!=1)) return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
        const uint64_t n=uint64_t(a.n),r=uint64_t(a.rows),e=uint64_t(a.experts);
        const uint64_t batch=e*uint64_t(a.groups);
        if(!shape(in[0],{n,r,batch}) || !shape(in[1],{n,batch}) || !shape(in[2],{n,batch}) ||
           !shape(in[3],{e*r},DATA_I32) || !shape(in[4],{batch*r})) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if(in[5].geometry.dataType!=DATA_F32 || in[5].geometry.dims!=2 || in[5].geometry.minSizes[0]!=1 ||
           in[5].geometry.maxSizes[0]!=1 || in[5].geometry.minSizes[1]!=in[5].geometry.maxSizes[1] ||
           !in[5].geometry.minSizes[1]) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        const uint64_t slots=in[5].geometry.minSizes[1];
        if(!shape(outputs[0],{n,slots})) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        out.indexSpaceGeometry[0]=e*(n/64);
    } else if(id==22) {
        // (id 35, STRATA_HPU_Q8_PACKED: x {256, columns/16, slots/x_group} U8 packed codes.)
        // x {256, columns, slots/x_group} U8 broadcast codes, dx {64, blocks, slots/x_group} F32, table {256} U8,
        // ids {slots} I32, w {bytes, experts} U8 (one resident expert per row), shared {bytes, 1} U8,
        // out {rows, slots, parts} F32. One member per (slot, part, group of five 256-row tiles).
        Iq4Gemv a{};
        if(!params(p.nodeParams,a) || a.columns<=0 || a.columns>65536 || a.columns%32 || a.rows<=0 || a.rows>65536 || a.rows%1280 ||
           a.x_group<=0 || a.slots%a.x_group || a.period<=1 || a.slots%a.period || a.parts<=0 || (a.columns/32)%a.parts ||
           a.slots<=1 || a.slots>int32_t(kMaxMoeSlots)) return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
        const uint64_t c=a.columns,r=a.rows,s=a.slots,xrows=s/a.x_group,bytes=r/2*c+r*(c/32)*4;
        if(!shape(in[0],{256,(kernel_id==35 || kernel_id==47 || kernel_id==54)?c/16:c,xrows},DATA_U8) || !shape(in[1],{64,c/32,xrows}) || !shape(in[2],{256},DATA_U8) ||
           !shape(in[3],{s},DATA_I32) || in[4].geometry.dims!=2 || in[4].geometry.maxSizes[0]!=bytes ||
           in[4].geometry.dataType!=DATA_U8 || !shape(in[5],{bytes,1},DATA_U8)) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if(!shape(outputs[0],{r,s,uint64_t(a.parts)})) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        out.indexSpaceGeometry[0]=s*a.parts*(r/(kernel_id==47?256:1280));
    } else if(id==23) {
        Q8Quant a{};
        if(!params(p.nodeParams,a) || a.columns<=0 || a.columns>65536 || a.columns%32 || a.rows<=0 || a.rows>int32_t(kMaxMoeSlots))
            return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
        const uint64_t c=a.columns,r=a.rows;
        if(!shape(in[0],{c,r})) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if(!shape(outputs[0],{256,kernel_id==34?c/16:c,r},DATA_U8) || !shape(outputs[1],{64,c/32,r})) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        out.indexSpaceGeometry[0]=r*(c/32);
    } else if(id==44) {
        Fp8Encode a{};
        if(!params(p.nodeParams,a) || a.columns<=0 || a.columns>65536 || a.columns%64 || a.rows<=0 || a.rows>64)
            return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
        const uint64_t c=a.columns,r=a.rows;
        if(!shape(in[0],{c,r})) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if(!shape(outputs[0],{c,r},DATA_F8_143)) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        out.indexSpaceGeometry[0]=r*((c+255)/256);
    } else if(id==36) {
        // x {columns, rows} F32 -> stacked {columns, 2*rows} F8 (hi rows, then lo rows), scale {1, rows} F32.
        Fp8Encode a{};
        if(!params(p.nodeParams,a) || a.columns<=0 || a.columns>65536 || a.columns%64 || a.rows<=0 || a.rows>64)
            return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
        const uint64_t c=a.columns,r=a.rows;
        if(!shape(in[0],{c,r})) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if(!shape(outputs[0],{c,2*r},DATA_F8_143) || !shape(outputs[1],{1,r})) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        out.indexSpaceGeometry[0]=r;
    } else if(id==37) {
        // both {columns, 2*rows} F32, x scale {1, rows} F32, w scale {columns, 1} F32 -> out {columns, rows} F32.
        Fp8Encode a{};
        if(!params(p.nodeParams,a) || a.columns<=0 || a.columns>1048576 || a.rows<=0 || a.rows>64)
            return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
        const uint64_t c=a.columns,r=a.rows;
        if(!shape(in[0],{c,2*r}) || !shape(in[1],{1,r}) || !shape(in[2],{c,1})) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if(!shape(outputs[0],{c,r})) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        out.indexSpaceGeometry[0]=r*((c+63)/64);
    } else if(id==38) {
        // parts {2*ff, slots, count} F32 -> codes {256, ff/16, slots} U8, scales {64, ff/32, slots} F32.
        MoeActQuant a{};
        if(!params(p.nodeParams,a) || a.ff<=0 || a.ff>65536 || a.ff%32 || a.slots<=0 || a.slots>int32_t(kMaxMoeSlots) || a.parts<=0 || a.parts>8)
            return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
        const uint64_t f=a.ff,s=a.slots,c=a.parts;
        if(!shape(in[0],{2*f,s,c})) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if(!shape(outputs[0],{256,f/16,s},DATA_U8) || !shape(outputs[1],{64,f/32,s})) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        out.indexSpaceGeometry[0]=s*(f/32);
    } else if(id==42) {
        // residual {w, r, t}, experts {w, t*per_row}, route {t*per_row}, inject {r, t} -> out {w, r, t}.
        HcWriteMoe a{};
        if(!params(p.nodeParams,a) || a.width<=0 || a.width>65536 || a.width%64 || a.streams<=0 || a.streams>32 ||
           a.rows<=0 || a.rows>int32_t(kMaxWindowRows) || a.per_row<=0 || int64_t(a.rows)*a.per_row>int64_t(kMaxMoeSlots)) return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
        const uint64_t w=a.width,r=a.streams,t=a.rows,s=t*uint64_t(a.per_row);
        if(!shape(in[0],{w,r,t}) || !shape(in[1],{w,s}) || !shape(in[2],{s}) || !shape(in[3],{r,t})) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if(!shape(outputs[0],{w,r,t})) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        // Members own 8 column tiles of one token (tpc/hc_write_moe.c).
        out.indexSpaceRank=2;out.indexSpaceGeometry[0]=(w/64+7)/8;out.indexSpaceGeometry[1]=t;
    } else if(id==88) {
        // id 42 with the expert rows read in place: residual {w, r, t}, routed {w, len} BF16, shared {w, t} BF16,
        // place {t*per_row} I32, route {t*per_row} F32, inject {r, t} -> out {w, r, t}.
        HcWriteMoe a{};
        if(!params(p.nodeParams,a) || a.width<=0 || a.width>65536 || a.width%256 || a.streams<=0 || a.streams>32 ||
           a.rows<=0 || a.rows>int32_t(kMaxWindowRows) || a.per_row<=1 || a.per_row>64 ||
           int64_t(a.rows)*a.per_row>int64_t(kMaxMoeSlots)) return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
        const uint64_t w=a.width,r=a.streams,t=a.rows,s=t*uint64_t(a.per_row),len=in[1].geometry.maxSizes[1];
        if(!shape(in[0],{w,r,t}) || !len || !shape(in[1],{w,len},DATA_BF16) || !shape(in[2],{w,t},DATA_BF16) ||
           !shape(in[3],{s},DATA_I32) || !shape(in[4],{s}) || !shape(in[5],{r,t})) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if(!shape(outputs[0],{w,r,t})) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        // Members own two 256-column groups of one token (tpc/hc_write_moe_routed.c).
        out.indexSpaceRank=2;out.indexSpaceGeometry[0]=(w/256+1)/2;out.indexSpaceGeometry[1]=t;
    } else if(id==89) {
        // STRATA_HPU_QSA_DEVICE_SELECT index query prep: raw {w, heads, rows}, gamma {w * gamma_heads}, rope {rotary, rows}.
        QsaIndexPrep a{};
        if(!params(p.nodeParams,a) || a.width<128 || a.width>65536 || a.width%64 || a.heads<=0 || a.heads>64 || a.rotary<=0 ||
           a.rotary%2 || a.rotary>64 || !epsilon(a.epsilon) || a.rows<=0 || a.rows>int32_t(kMaxWindowRows) ||
           (a.gamma_heads!=1 && a.gamma_heads!=a.heads)) return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
        const uint64_t w=a.width,hd=a.heads,r=a.rows;
        if(!shape(in[0],{w,hd,r}) || !shape(in[1],{w*uint64_t(a.gamma_heads)}) || !shape(in[2],{uint64_t(a.rotary),r}))
            return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if(!shape(outputs[0],{w,hd,r})) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        out.indexSpaceGeometry[0]=r;
    } else if(id==90) {
        // STRATA_HPU_QSA_DEVICE_SELECT block keys: start {1}, pooled {w, >= (keys - rows) / block}, tail {w, block},
        // window {w, rows}, gamma {w}, rope {rotary, >= keys} -> out {w, keys / block}. Members are blocks.
        QsaIndexCells a{};
        if(!params(p.nodeParams,a) || a.width<128 || a.width>65536 || a.width%64 || a.rotary<=0 || a.rotary%2 || a.rotary>64 ||
           !epsilon(a.epsilon) || a.rows<=0 || a.keys<=a.rows || a.block<2 || a.block>64 || a.keys%a.block) return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
        const uint64_t w=a.width,k=a.keys,rows=a.rows,blocks=(k-rows)/uint64_t(a.block);
        if(!shape(in[0],{1}) || in[1].geometry.dims!=2 || in[1].geometry.maxSizes[0]!=w || in[1].geometry.maxSizes[1]<std::max<uint64_t>(blocks,1) ||
           !shape(in[2],{w,uint64_t(a.block)}) || !shape(in[3],{w,rows}) || !shape(in[4],{w}) ||
           in[5].geometry.dims!=2 || in[5].geometry.maxSizes[0]!=uint64_t(a.rotary) || in[5].geometry.maxSizes[1]<k)
            return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if(!shape(outputs[0],{w,k/uint64_t(a.block)})) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        out.indexSpaceGeometry[0]=k/uint64_t(a.block);
    } else if(id==91) {
        // STRATA_HPU_QSA_DEVICE_SELECT block selection: start {2}, scores {keys / block, heads, rows} -> scratch and
        // kept {keys / block, rows}.
        QsaSelect a{};
        if(!params(p.nodeParams,a) || a.keys<=0 || a.keys>262144 || a.rows<=0 || a.rows>int32_t(kMaxWindowRows) ||
           a.window<=0 || a.window>a.keys || a.rows>a.window || a.width<=0 || a.block<2 || a.block>64 || (a.block&(a.block-1)) ||
           a.keys%(256*a.block) || a.heads<=0 || a.heads>64) return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
        const uint64_t kb=uint64_t(a.keys)/uint64_t(a.block),r=a.rows;
        if(!shape(in[0],{2}) || !shape(in[1],{kb,uint64_t(a.heads),r})) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        // id 107 (the sparse prefill's bits): scratch, bits {kb/16, r} and the cut block info {2, r}.
        if(kernel_id==107?(!shape(outputs[0],{kb,r}) || !shape(outputs[1],{kb/16,r}) || !shape(outputs[2],{2,r})):
                          (!shape(outputs[0],{kb,r}) || !shape(outputs[1],{kb,r}))) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        out.indexSpaceGeometry[0]=r;
    } else if(id==105) {
        // STRATA_HPU_QSA_SDPA_FOLD mask: start {2}, kept {keys, rows} -> mask {keys, group, rows} BF16.
        QsaSelectFold a{};
        if(!params(p.nodeParams,a) || a.keys<=0 || a.keys%128 || a.keys>262144 || a.rows<=0 || a.rows>int32_t(kMaxWindowRows) ||
           a.block<2 || a.block>64 || (a.block&(a.block-1)) || a.group<=0 || a.group>64) return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
        const uint64_t k=a.keys,r=a.rows;
        if(!shape(in[0],{2}) || !shape(in[1],{k,r})) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if(!shape(outputs[0],{k,uint64_t(a.group),r},DATA_BF16)) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        out.indexSpaceGeometry[0]=r;
    } else if(id==106) {
        // Sparse prefill gather: FP8 cache {256, heads, cells} x2 + scales {32, cells} x2 + ids {top, rows} I32 + start {1}
        // -> key, value {256, top*block, heads, rows} BF16 (even/odd channel order), scales {top*block, heads, rows} x2,
        // mask {top*block, rows}.
        QsaGatherRows a{};
        if(!params(p.nodeParams,a) || a.heads<=0 || a.heads>32 || a.cells<=0 || a.cells>262144 || a.top<=0 || a.top>4096 ||
           a.block!=4 || a.rows<=0 || a.rows>int32_t(kMaxWindowRows)) return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
        const uint64_t h=uint64_t(a.heads),c=uint64_t(a.cells),k=uint64_t(a.top),n=k*4,r=uint64_t(a.rows);
        if(!shape(in[0],{256,h,c},DATA_F8_143) || !shape(in[1],{256,h,c},DATA_F8_143) || !shape(in[2],{32,c}) || !shape(in[3],{32,c}) ||
           !shape(in[4],{k,r},DATA_I32) || !shape(in[5],{2,r}) || !shape(in[6],{2})) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if(!shape(outputs[0],{256,n,h,r},DATA_BF16) || !shape(outputs[1],{256,n,h,r},DATA_BF16) || !shape(outputs[2],{n,h,r}) ||
           !shape(outputs[3],{n,h,r}) || !shape(outputs[4],{n,r})) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        out.indexSpaceRank=3;out.indexSpaceGeometry[0]=(k+15)/16;out.indexSpaceGeometry[1]=h;out.indexSpaceGeometry[2]=r;
    } else if(id==92) {
        // STRATA_HPU_QSA_DEVICE_SELECT mask: start {2}, kept {keys, rows} (per position) -> mask {keys, 1, rows}.
        QsaSelect a{};
        if(!params(p.nodeParams,a) || a.keys<=0 || a.keys%64 || a.keys>262144 || a.rows<=0 || a.rows>int32_t(kMaxWindowRows) ||
           a.block<2 || a.block>64 || (a.block&(a.block-1))) return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
        const uint64_t k=a.keys,r=a.rows;
        if(!shape(in[0],{2}) || !shape(in[1],{k,r})) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if(!shape(outputs[0],{k,1,r})) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        out.indexSpaceGeometry[0]=r;
    } else if(id==43) {
        // state {s, hv, s}; q, k {s, hk, rows}; v {s, hv, rows}; gate, beta {hv, rows} ->
        // four states {s, hv, s} (the last 4-rows unused) and output {s, hv, rows}. Members as id 0.
        GdnStepRows a{};
        if(!params(p.nodeParams,a) || !width(a.width) || a.key_heads<=0 || a.rows<2 || a.rows>4) return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
        const uint64_t s=a.width,hk=a.key_heads,hv=in[3].geometry.maxSizes[1],r=a.rows;
        if(!hv || hv%hk || !shape(in[0],{s,hv,s}) || !shape(in[1],{s,hk,r}) || !shape(in[2],{s,hk,r}) || !shape(in[3],{s,hv,r}) ||
           !shape(in[4],{hv,r}) || !shape(in[5],{hv,r})) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        for(unsigned i=0;i<4;++i) if(!shape(outputs[i],{s,hv,s})) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        if(!shape(outputs[4],{s,hv,r})) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        out.indexSpaceRank=2;out.indexSpaceGeometry[0]=s/64;out.indexSpaceGeometry[1]=hv;
    } else if(id==30) {
        GdnConvRows a{};
        if(!params(p.nodeParams,a) || a.channels<=0 || a.channels>1048576 || a.taps<2 || a.taps>4 || a.rows<1 || a.rows>4)
            return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
        const uint64_t c=a.channels,d=a.taps,r=a.rows;
        if(!shape(in[0],{c,d-1}) || !block(in[1],0,c,r) || !shape(in[2],{c,d})) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        for(unsigned i=0;i<4;++i) if(!shape(outputs[i],{c,d-1})) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        if(!shape(outputs[4],{c,r})) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        out.indexSpaceGeometry[0]=(c+63)/64;
    } else if(id==45) {
        GdnConvRows a{};
        if(!params(p.nodeParams,a) || a.channels<=0 || a.channels>1048576 || a.taps<2 || a.taps>4 || a.rows<1 || a.rows>8)
            return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
        const uint64_t c=a.channels,d=a.taps,r=a.rows;
        if(!shape(in[0],{c,d-1}) || !block(in[1],0,c,r) || !shape(in[2],{c,d})) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        for(unsigned i=0;i<8;++i) if(!shape(outputs[i],{c,d-1})) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        if(!shape(outputs[8],{c,r})) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        out.indexSpaceGeometry[0]=(c+63)/64;
    } else if(id==31) {
        GdnPrepareRows a{};
        if(!params(p.nodeParams,a) || !width(a.width) || a.key_heads<=0 || a.value_heads<=0 || a.value_heads>1024 ||
           a.value_heads%a.key_heads || !epsilon(a.epsilon) || a.rows<1 || a.rows>int32_t(kMaxWindowRows)) return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
        const uint64_t s=a.width,hk=a.key_heads,hv=a.value_heads,c=s*(2*hk+hv),r=a.rows;
        if(!shape(in[0],{c,r}) || a.alpha_col<0 || a.beta_col<0 || !block(in[1],uint64_t(a.alpha_col),hv,r) ||
           !shape(in[2],{hv}) || !shape(in[3],{hv}) || !block(in[4],uint64_t(a.beta_col),hv,r))
            return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if(!shape(outputs[0],{s,hk,r}) || !shape(outputs[1],{s,hk,r}) || !shape(outputs[2],{s,hv,r}) ||
           !shape(outputs[3],{hv,r}) || !shape(outputs[4],{hv,r})) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        // 128-wide heads: members own four rows (tpc/gdn_prepare_rows.c), except in a small verify window, where
        // one row per member gives four times the members.
        out.indexSpaceRank=2;out.indexSpaceGeometry[0]=hv;out.indexSpaceGeometry[1]=s==128 && r>8?(r+3)/4:r;
    } else if(id==32) {
        GdnNormRows a{};
        if(!params(p.nodeParams,a) || !width(a.width) || a.value_heads<=0 || a.value_heads>1024 || !epsilon(a.epsilon) ||
           a.rows<1 || a.rows>int32_t(kMaxWindowRows)) return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
        const uint64_t s=a.width,hv=a.value_heads,r=a.rows;
        if(!shape(in[0],{s,hv,r}) || a.z_col<0 || !block(in[1],uint64_t(a.z_col),s*hv,r) || !shape(in[2],{s}))
            return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if(!shape(outputs[0],{s,hv,r},kernel_id==41?DATA_BF16:DATA_F32)) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        out.indexSpaceRank=2;out.indexSpaceGeometry[0]=hv;out.indexSpaceGeometry[1]=r;
    } else if(id==27) {
        RouteShare a{};
        if(!params(p.nodeParams,a) || a.slots<=1 || a.slots>128 || a.period<=1 || a.slots%a.period) return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
        if(!shape(in[0],{uint64_t(a.slots)},DATA_I32)) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if(!shape(outputs[0],{16,uint64_t(a.slots)},DATA_I32)) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        out.indexSpaceGeometry[0]=uint64_t(a.slots);
    } else if(id==28) {
        // iq4q8_gemv's tensors and members, plus share {16, slots} I32 after `shared`.
        Iq4Gemv a{};
        if(!params(p.nodeParams,a) || a.columns<=0 || a.columns>65536 || a.columns%32 || a.rows<=0 || a.rows>65536 || a.rows%1280 ||
           a.x_group<=0 || a.slots%a.x_group || a.period<=1 || a.slots%a.period || a.parts<=0 || (a.columns/32)%a.parts ||
           a.slots<=1 || a.slots>128) return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
        const uint64_t c=a.columns,r=a.rows,s=a.slots,xrows=s/a.x_group,bytes=r/2*c+r*(c/32)*4;
        if(!shape(in[0],{256,c,xrows},DATA_U8) || !shape(in[1],{64,c/32,xrows}) || !shape(in[2],{256},DATA_U8) ||
           !shape(in[3],{s},DATA_I32) || in[4].geometry.dims!=2 || in[4].geometry.maxSizes[0]!=bytes ||
           in[4].geometry.dataType!=DATA_U8 || !shape(in[5],{bytes,1},DATA_U8) || !shape(in[6],{16,s},DATA_I32)) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if(!shape(outputs[0],{r,s,uint64_t(a.parts)})) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        out.indexSpaceGeometry[0]=s*a.parts*(r/1280);
    } else if(id==24) {
        RouteTopk a{};
        if(!params(p.nodeParams,a) || a.experts<=0 || a.experts>4096 || a.experts%64 || a.top_k<=0 || a.top_k>a.experts ||
           a.slots!=a.top_k+1 || a.slots>64 || a.rows<=0 || a.rows>int32_t(kMaxWindowRows)) return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
        const uint64_t rows=a.rows,all=uint64_t(a.slots)*rows;
        if(!shape(in[0],{uint64_t(a.experts),rows}) || !shape(in[1],{1,rows})) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if(!shape(outputs[0],{all},DATA_I32) || !shape(outputs[1],{all})) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        // Up to 1024 experts and more than 24 rows, members own four rows (tpc/route_topk.c).
        out.indexSpaceGeometry[0]=a.experts<=1024 && rows>24?(rows+3)/4:rows;
    } else if(id==26) {
        QsaPrep a{};
        if(!params(p.nodeParams,a) || a.width<128 || a.width>65536 || a.width%64 || a.query_heads<=0 || a.kv_heads<=0 ||
           a.query_heads>1024 || a.kv_heads>a.query_heads || a.rotary<=0 || a.rotary%2 || a.rotary>64 || !epsilon(a.epsilon) ||
        // The kernel ignores `tokens` (tpc/qsa_prep.c: `(void)tokens;`, each member derives its row), so this is a
        // dispatch bound, not a kernel limit. 64 was too tight for the 128/256-row prefill windows, which is what
        // hpu_qsa_prep_rows_compile reproduced; it now follows the widest prefill window the engine offers.
           a.tokens<=0 || a.tokens>int32_t(kMaxWindowRows)) return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
        const uint64_t w=a.width,hq=a.query_heads,hk=a.kv_heads,t=a.tokens;
        // Every tensor has an outer token axis; one token may also keep the 2-D shapes (decode).
        const bool one=t==1;
        const auto rows=[&](const Tensor& x,uint64_t heads) {return (one && shape(x,{w,heads})) || shape(x,{w,heads,t});};
        const uint64_t rot=uint64_t(a.rotary);
        if(!rows(in[0],hq) || !rows(in[1],hk) || !rows(in[2],hk) || !shape(in[3],{w}) || !shape(in[4],{w}) ||
           !((one && shape(in[5],{rot})) || shape(in[5],{rot,t}))) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if(!rows(outputs[0],hq)) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        for(unsigned i=1;i<5;++i) if(!rows(outputs[i],hk)) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        // Members own up to four rows of one kind of one token (tpc/qsa_prep.c).
        out.indexSpaceGeometry[0]=((hq+3)/4+(hk+3)/4)*t;
    } else if (id == 4 || id == 13) {
        RowNorm a{};
        // rows: window rows x HC streams (8192 x 4 at the widest window).
        if(!params(p.nodeParams,a) || a.width<=0 || a.width>65536 || a.width%64 || a.rows<=0 || a.rows>65536 ||
           a.shared_gamma>1 || a.shared_gamma< -4096 || (a.shared_gamma<0 && a.rows%(-a.shared_gamma)) || !epsilon(a.epsilon)) return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
        const uint64_t w=a.width,r=a.rows;
        const auto gamma_rows=a.shared_gamma<0?uint64_t(-a.shared_gamma):r;
        if(!shape(in[0],{w,r}) || !(a.shared_gamma==1?shape(in[1],{w}):shape(in[1],{w,gamma_rows}))) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if(!shape(outputs[0],{w,r}) || (kernel_id==39 && !shape(outputs[1],{w,r},DATA_BF16))) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        // row_norm_dual (id 39) members own four rows, one row each up to 32 rows (tpc/row_norm_dual.c).
        out.indexSpaceGeometry[0]=kernel_id==39 && r>32?(r+3)/4:r;
    } else if (id == 5 || id == 6) {
        Hc a{};
        if(!params(p.nodeParams,a) || a.width<=0 || a.width>65536 || a.width%64 || a.streams<=0 || a.streams>32)
            return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
        const uint64_t w=a.width,r=a.streams;
        if(!shape(in[0],{w,r})) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if(id==5) {
            if(!shape(in[1],{w,r})) return GLUE_INCOMPATIBLE_INPUT_SIZE;
            if(!shape(outputs[0],{w})) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        } else {
            if(!shape(in[1],{w}) || !shape(in[2],{r})) return GLUE_INCOMPATIBLE_INPUT_SIZE;
            if(!shape(outputs[0],{w,r})) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
            out.indexSpaceRank=2;out.indexSpaceGeometry[1]=r;
        }
        out.indexSpaceGeometry[0]=w/64;
    } else if(id==7) {
        PleGate a{};
        if(!params(p.nodeParams,a) || a.width<=0 || a.width>65536 || a.width%64 || a.streams<=0 || a.streams>32 ||
           !epsilon(a.inverse_sqrt_width)) return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
        const uint64_t w=a.width,r=a.streams;
        if(!shape(in[0],{w,r}) || !shape(in[1],{w,r}) || !shape(in[2],{w})) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if(!shape(outputs[0],{w,r})) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        out.indexSpaceGeometry[0]=r;
    } else if(id==8) {
        PleConv a{};
        if(!params(p.nodeParams,a) || a.channels<=0 || a.channels>1048576 || a.channels%64 ||
           a.taps<2 || a.taps>16 || a.dilation<=0 || a.dilation>16) return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
        const uint64_t c=a.channels,t=a.taps,h=(t-1)*a.dilation;
        if(!shape(in[0],{c,h}) || !shape(in[1],{c}) || !shape(in[2],{c,t}) || !shape(in[3],{c}) || !shape(in[4],{c}))
            return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if(!shape(outputs[0],{c,h}) || !shape(outputs[1],{c})) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        out.indexSpaceGeometry[0]=c/64;
    } else if(id==9) {
        PackedMatmul a{};
        if(!params(p.nodeParams,a) || a.columns<=0 || a.columns>65536 || a.rows<=0 || a.rows>1048576 || a.batch<=0 || a.batch>128 ||
           (a.type!=2 && a.type!=3 && a.type!=6 && a.type!=7 && a.type!=8 && a.type!=11 && a.type!=12 && a.type!=13 && a.type!=14 && a.type!=20 && a.type!=23 && a.type!=16 && a.type!=17 && a.type!=18 && a.type!=21 && a.type!=22 && a.type!=29 && a.type!=40 && a.type!=42) ||
           (a.activation!=0 && a.activation!=1 && a.activation!=2) || (a.activation==1 && a.type!=42) ||
           (a.activation==2 && a.type==42)) return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
        // The 256-value K superblocks use their own superblock dot under either
        // q8_1 contract: upstream native_mmvq q4_q8_dot_impl and iq_kernels
        // vec_dot_q4_K_q8_1_impl_vmmq multiply the group terms by the sum of the
        // QUANTIZED activation codes, and q3_q8_dot_impl does the same with one
        // signed d and a -32 scale bias. Q2_0 stays routed S2 only.
        const bool superblock=a.type==11||a.type==12||a.type==13||a.type==14||a.type==16||a.type==17||a.type==18||a.type==21||a.type==22||a.type==29||a.type==23;
        const uint32_t values=(a.type==42||a.type==40)?64:superblock?256:32;
        const uint32_t size=a.type==3?20:a.type==6?22:a.type==7?24:a.type==8?34:a.type==11?110:a.type==12?144:a.type==13?176:a.type==14?210:a.type==20?18:a.type==23?136:a.type==16?66:a.type==17?74:a.type==18?98:a.type==21?110:a.type==22?82:a.type==29?56:a.type==40?36:18;
        if(a.columns%values) return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
        const uint64_t bytes=uint64_t(a.columns)/values*size;
        if(!shape(in[0],{bytes,uint64_t(a.rows)},DATA_U8) || !shape(in[1],{uint64_t(a.columns),uint64_t(a.batch)},DATA_U8) ||
           !shape(in[2],{2,uint64_t(a.columns/32),uint64_t(a.batch)}) ||
           !shape(in[3],{STRATA_PACKED_CODEBOOK_BYTES,1},DATA_U8)) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if(!shape(outputs[0],{uint64_t(a.rows),uint64_t(a.batch)})) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        out.indexSpaceRank=2;out.indexSpaceGeometry[0]=a.rows;out.indexSpaceGeometry[1]=a.batch;
    } else if(id==12) {
        ActivationQuant a{};
        if(!params(p.nodeParams,a) || a.columns<=0 || a.columns>65536 || a.columns%32 || a.batch<=0 || a.batch>128)
            return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
        // Exactly the tensors the packed dot reads afterwards: codes {columns,batch} U8 and the
        // {2,columns/32,batch} F32 metadata, so the two kernels chain inside one graph.
        if(!shape(in[0],{uint64_t(a.columns),uint64_t(a.batch)})) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if(!shape(outputs[0],{uint64_t(a.columns),uint64_t(a.batch)},DATA_U8) ||
           !shape(outputs[1],{2,uint64_t(a.columns/32),uint64_t(a.batch)})) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
        out.indexSpaceRank=2;out.indexSpaceGeometry[0]=a.columns/32;out.indexSpaceGeometry[1]=a.batch;
    } else {
        HcWindow a{};
        if(!params(p.nodeParams,a) || a.width<=0 || a.width>65536 || a.width%64 || a.streams<=0 || a.streams>32 ||
           a.rows<=0 || a.rows>int32_t(kMaxWindowRows)) return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
        const uint64_t w=a.width,r=a.streams,t=a.rows;
        if(!shape(in[0],{w,r,t})) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        if(id==10) {
            if(!shape(in[1],{w,r,t})) return GLUE_INCOMPATIBLE_INPUT_SIZE;
            if(!shape(outputs[0],{w,t}) || (kernel_id==40 && !shape(outputs[1],{w,t},DATA_BF16))) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
            out.indexSpaceRank=2;out.indexSpaceGeometry[1]=t;
        } else {
            if(!shape(in[1],{w,t}) || !shape(in[2],{r,t})) return GLUE_INCOMPATIBLE_INPUT_SIZE;
            if(!shape(outputs[0],{w,r,t})) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
            out.indexSpaceRank=3;out.indexSpaceGeometry[1]=r;out.indexSpaceGeometry[2]=t;
        }
        // hc_mix_window_dual (id 40) members own 256 columns (tpc/hc_mix_window_dual.c), hc_write_window (id 11)
        // eight 64-column tiles (tpc/hc_write_window.c).
        out.indexSpaceGeometry[0]=kernel_id==40?(w+255)/256:kernel_id==11?(w/64+7)/8:w/64;
    }
    return GLUE_SUCCESS;
}
}
#define EXPORT extern "C" __attribute__((visibility("default")))
EXPORT uint64_t GetLibVersion() { return 50; }
EXPORT GlueCodeReturn GetKernelGuids(DeviceId device, uint32_t* count, GuidInfo* guids) {
    if (!count) return GLUE_FAILED;
    const auto capacity = *count;
    *count = device == DEVICE_ID_GAUDI2 ? std::size(kernels) : 0;
    if (guids) for (unsigned i = 0; i < std::min(capacity, *count); ++i) {
        std::memset(&guids[i], 0, sizeof(guids[i]));
        std::strcpy(guids[i].name, kernels[i].guid);
    }
    return GLUE_SUCCESS;
}
EXPORT GlueCodeReturn GetShapeInference(DeviceId device, const ShapeInferenceParams* p, ShapeInferenceOutput* out) {
    if (device != DEVICE_ID_GAUDI2) return GLUE_NODE_NOT_FOUND;
    if (!p || !out || !p->inputTensors || !out->outputTensors) return GLUE_FAILED;
    const int found = find((p->pGuid ? p->pGuid : &p->guid)->name);
    if (found < 0) return GLUE_NODE_NOT_FOUND;
    if (p->inputTensorsNr != kernels[found].inputs) return GLUE_INCOMPATIBLE_INPUT_COUNT;
    if (p->outputTensorsNr != kernels[found].outputs) return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    const int id = base_kernel(fast_twin(found));
    for (unsigned i = 0; i < p->inputTensorsNr; ++i) if (!p->inputTensors[i]) return GLUE_FAILED;
    for (unsigned i = 0; i < p->outputTensorsNr; ++i) if (!out->outputTensors[i]) return GLUE_FAILED;
    if(id==9) {
        PackedMatmul a{};
        if(!params(p->nodeParams,a) || a.rows<=0 || a.batch<=0) return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
        out->outputTensors[0]->geometry=p->inputTensors[2]->geometry;
        shape(*out->outputTensors[0],{uint64_t(a.rows),uint64_t(a.batch)});
    } else if(id==12) {
        ActivationQuant a{};
        if(!params(p->nodeParams,a) || a.columns<=0 || a.columns%32 || a.batch<=0) return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
        // Both outputs are declared completely by the graph, so nothing is inferred here. Copying the
        // input's geometry instead (an F32 tensor for a U8 codes output) made synGraphCompile fail with
        // status 26; the graph's own declaration is what the MME must see.
        (void)a;
    } else if(id==15 || id==19 || id==20) {
        Fp8Gemv a{};if(!params(p->nodeParams,a) || a.rows<=0) return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
        out->outputTensors[0]->geometry=p->inputTensors[0]->geometry;
        shape(*out->outputTensors[0],{uint64_t(a.rows),11});
    } else if(id==22 || id==23 || id==24 || id==26 || id==27 || id==28 || id==30 || id==31 || id==32 || id==36 || id==37 || id==38 || id==42 || id==43 || id==44 || id==45 || id==46 || id==48 || id==49 || id==50 || id==51 || id==52 || id==53 || id==54 || id==55 || id==56 || id==57 || id==58 || id==59 || id==60 || id==61 || id==62 || id==63 || id==64 || id==66 || id==79 || id==81 || id==82 || id==83 || id==84 || id==88 || id==89 || id==90 || id==91 || id==92 || id==93 || id==95 || id==96 || id==97 || id==98 || id==99 || id==100 || id==101 || id==102 || id==103 || id==104 || id==105 || id==106 || id==107 || id==108 || id==109 || id==110 || id==111) {
        // Every output is declared completely by the graph (U8/I32 outputs must not inherit an F32 input).
    } else if(id==14) {
        // All output dtypes/shapes are explicit: do not copy F32 input geometry into HF8 outputs.
        Fp8Encode a{};if(!params(p->nodeParams,a) || a.columns<=0 || a.rows<=0) return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
    } else if (id == 2) {
        GdnPrepare a{};
        if (!params(p->nodeParams, a) || !width(a.width) || a.key_heads <= 0 || a.value_heads <= 0)
            return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
        for (unsigned i = 0; i < 5; ++i) out->outputTensors[i]->geometry = p->inputTensors[1]->geometry;
        shape(*out->outputTensors[0], {uint64_t(a.width),uint64_t(a.key_heads)});
        shape(*out->outputTensors[1], {uint64_t(a.width),uint64_t(a.key_heads)});
        shape(*out->outputTensors[2], {uint64_t(a.width),uint64_t(a.value_heads)});
    } else {
        out->outputTensors[0]->geometry = p->inputTensors[0]->geometry;
        if (id == 0) out->outputTensors[1]->geometry = p->inputTensors[3]->geometry;
        if (id == 1) out->outputTensors[1]->geometry = p->inputTensors[1]->geometry;
        if (id == 5) {
            Hc a{};if(!params(p->nodeParams,a) || a.width<=0) return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
            shape(*out->outputTensors[0],{uint64_t(a.width)});
        }
        if(id==10) {
            HcWindow a{};if(!params(p->nodeParams,a) || a.width<=0 || a.rows<=0) return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
            shape(*out->outputTensors[0],{uint64_t(a.width),uint64_t(a.rows)});
        }
        if (id == 8) out->outputTensors[1]->geometry = p->inputTensors[3]->geometry;
    }
    return GLUE_SUCCESS;
}
EXPORT GlueCodeReturn InstantiateTpcKernel(const HabanaKernelParams* p, HabanaKernelInstantiation* out) {
    if (!p || !out) return GLUE_FAILED;
    const int found = find(p->guid.name);
    if (p->deviceId != DEVICE_ID_GAUDI2 || found < 0) return GLUE_NODE_NOT_FOUND;
    const auto& kernel = kernels[found];
    const int id = fast_twin(found);
    if (p->inputTensorNr != kernel.inputs) return GLUE_INCOMPATIBLE_INPUT_COUNT;
    if (p->outputTensorNr != kernel.outputs) return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    const auto status = configure(id, *p, *out);
    if (status != GLUE_SUCCESS) {
        // Which node refused, and at which geometry: without this the SDK only reports "Generic failure (status 26)".
        std::fprintf(stderr, "glue reject guid=%s status=%d rows=%u\n", kernel.guid, int(status),
                     kernel.inputs > 3 ? unsigned(p->inputTensors[3].geometry.maxSizes[0]) : 0u);
        return status;
    }
    out->auxiliaryTensorNr = 0;
    for (unsigned i = 0; i < kernel.inputs; ++i) {
        std::memset(&out->inputTensorAccessPattern[i], 0, sizeof(TensorAccessPattern));
        out->inputTensorAccessPattern[i].allRequired = 1;
    }
    for (unsigned i = 0; i < kernel.outputs; ++i) {
        std::memset(&out->outputTensorAccessPattern[i], 0, sizeof(TensorAccessPattern));
        out->outputTensorAccessPattern[i].allRequired = 1;
        out->outputTensorAccessPattern[i].noRmwAccess = 1;
    }
    const char* access_env=std::getenv("STRATA_HPU_TPC_ACCESS");
    if(access_env && std::strcmp(access_env,"1")==0) {
        // Keep reductions' complete dimensions, but expose independent rows
        // and width tiles to the slicer. Routed weight addressing stays opaque.
        const auto full=[](TensorAccessPattern& a,const Tensor& t) {
            a.allRequired=0;
            for(unsigned d=0;d<t.geometry.dims;++d) {
                auto& m=a.mapping[d];m.indexSpaceDim=0;m.a=0;m.start_b=0;m.end_b=float(t.geometry.maxSizes[d]-1);
            }
        };
        const auto axis=[](TensorAccessPattern& a,unsigned tensor_dim,unsigned index_dim,float stride,float count) {
            auto& m=a.mapping[tensor_dim];m.indexSpaceDim=index_dim;m.a=stride;m.start_b=0;m.end_b=count-1;
        };
        if(id==4 || id==13 || id==39) {
            RowNorm a{};if(!params(p->nodeParams,a)) return GLUE_UNSUPPORTED_LAYER_CONFIGURATION;
            const float rows=id==39 && a.rows>32?4.f:1.f;  // rows per member, as configure()
            full(out->inputTensorAccessPattern[0],p->inputTensors[0]);
            axis(out->inputTensorAccessPattern[0],1,0,rows,rows);
            full(out->inputTensorAccessPattern[1],p->inputTensors[1]);
            // Shared or cyclic stream gamma is read in full; only an ordinary
            // per-row gamma has an affine row coordinate.
            if(a.shared_gamma==0) axis(out->inputTensorAccessPattern[1],1,0,rows,rows);
            for(unsigned i=0;i<kernel.outputs;++i) {
                full(out->outputTensorAccessPattern[i],p->outputTensors[i]);
                axis(out->outputTensorAccessPattern[i],1,0,rows,rows);
            }
        } else if(id==5 || id==10 || id==40) {
            const bool window=id!=5;
            const float tile=id==40?256.f:64.f;   // hc_mix_window_dual members own 256 columns
            for(unsigned i=0;i<kernel.inputs;++i) {
                full(out->inputTensorAccessPattern[i],p->inputTensors[i]);
                axis(out->inputTensorAccessPattern[i],0,0,tile,tile);
                if(window) axis(out->inputTensorAccessPattern[i],2,1,1,1);
            }
            for(unsigned i=0;i<kernel.outputs;++i) {
                full(out->outputTensorAccessPattern[i],p->outputTensors[i]);
                axis(out->outputTensorAccessPattern[i],0,0,tile,tile);
                if(window) axis(out->outputTensorAccessPattern[i],1,1,1,1);
            }
        } else if(id==6 || id==11) {
            const bool window=id==11;
            for(unsigned i=0;i<kernel.inputs;++i) full(out->inputTensorAccessPattern[i],p->inputTensors[i]);
            full(out->outputTensorAccessPattern[0],p->outputTensors[0]);
            const float tile=window?512.f:64.f;   // hc_write_window members own eight 64-column tiles
            for(auto* a:{&out->inputTensorAccessPattern[0],&out->outputTensorAccessPattern[0]}) {
                axis(*a,0,0,tile,tile);axis(*a,1,1,1,1);if(window) axis(*a,2,2,1,1);
            }
            axis(out->inputTensorAccessPattern[1],0,0,tile,tile);
            axis(out->inputTensorAccessPattern[2],0,1,1,1);
            if(window) {
                axis(out->inputTensorAccessPattern[1],1,2,1,1);
                axis(out->inputTensorAccessPattern[2],1,2,1,1);
            }
        }
    }
    if(id==63 || id==64) {
        // Each member owns one complete batch. Map that axis exactly so the
        // slicer may pipeline encode/MME/finish without full HBM temporaries.
        const auto map_batch=[](TensorAccessPattern& access,const Tensor& tensor) {
            access.allRequired=0;
            for(unsigned dim=0;dim<3;++dim) {
                auto& m=access.mapping[dim];m.indexSpaceDim=0;
                m.a=dim==2?1.f:0.f;m.start_b=0;
                m.end_b=dim==2?0.f:float(tensor.geometry.maxSizes[dim]-1);
            }
        };
        for(unsigned i=0;i<kernel.inputs;++i) map_batch(out->inputTensorAccessPattern[i],p->inputTensors[i]);
        for(unsigned i=0;i<kernel.outputs;++i) map_batch(out->outputTensorAccessPattern[i],p->outputTensors[i]);
    }
    // STRATA_HPU_MOE_SLICE: the MoE tile kernels' outputs mapped on their tile (batch) axis, so the slicer may run the
    // expansion, the gather and the activation slice by slice with the batch GEMMs through SRAM instead of full HBM
    // temporaries. Every input stays fully required (the tile -> expert / row maps are data-dependent).
    static const bool moe_slice=[] {const char* v=std::getenv("STRATA_HPU_MOE_SLICE");return v && v[0]=='1' && !v[1];}();
    if(moe_slice && (id==70 || id==75 || id==77 || id==81 || id==82 || id==83)) {
        const auto whole=[](TensorAccessPattern& a,const Tensor& t) {
            a.allRequired=0;
            for(unsigned d=0;d<t.geometry.dims;++d) {
                auto& m=a.mapping[d];m.indexSpaceDim=0;m.a=0;m.start_b=0;m.end_b=float(t.geometry.maxSizes[d]-1);
            }
        };
        const auto map=[](TensorAccessPattern& a,unsigned tensor_dim,unsigned index_dim,float stride,float count) {
            auto& m=a.mapping[tensor_dim];m.indexSpaceDim=index_dim;m.a=stride;m.start_b=0;m.end_b=count-1;
        };
        for(unsigned i=0;i<kernel.inputs;++i) whole(out->inputTensorAccessPattern[i],p->inputTensors[i]);
        for(unsigned i=0;i<kernel.outputs;++i) whole(out->outputTensorAccessPattern[i],p->outputTensors[i]);
        if(id==77 || id==81) {
            // Members (256-row tile, local tile): out {rows, columns, count}, id 81's scale {rows, count}.
            map(out->outputTensorAccessPattern[0],0,0,256,256);map(out->outputTensorAccessPattern[0],2,1,1,1);
            if(id==81) {map(out->outputTensorAccessPattern[1],0,0,256,256);map(out->outputTensorAccessPattern[1],1,1,1,1);}
        } else if(id==75 || id==82) {
            // Members (8 tile rows, local tile): a {hidden, rows, count}; id 82's a_scale {64, rows, count}.
            for(unsigned i=0;i<kernel.outputs;++i) {map(out->outputTensorAccessPattern[i],1,0,8,8);map(out->outputTensorAccessPattern[i],2,1,1,1);}
        } else if(id==83) {
            // Members (8 rows, batch): gu {2*ff, rows, batch} and a_scale {64, rows, batch} in (s_gu {2*ff, batch} by
            // batch), act {ff, rows, batch} and act_scale {64, rows, batch} out.
            for(unsigned i:{0u,2u}) {map(out->inputTensorAccessPattern[i],1,0,8,8);map(out->inputTensorAccessPattern[i],2,1,1,1);}
            map(out->inputTensorAccessPattern[1],1,1,1,1);
            for(unsigned i=0;i<kernel.outputs;++i) {map(out->outputTensorAccessPattern[i],1,0,8,8);map(out->outputTensorAccessPattern[i],2,1,1,1);}
        } else {
            // Members (8 rows, batch): gu {2*ff, rows, batch} in, act {ff, rows, batch} out.
            map(out->inputTensorAccessPattern[0],1,0,8,8);map(out->inputTensorAccessPattern[0],2,1,1,1);
            map(out->outputTensorAccessPattern[0],1,0,8,8);map(out->outputTensorAccessPattern[0],2,1,1,1);
        }
    }
    std::memcpy(out->kernel.scalarParams, p->nodeParams.nodeParams, p->nodeParams.nodeParamsSize);
    out->kernel.paramsNr = p->nodeParams.nodeParamsSize / sizeof(uint32_t);
    const auto begin = reinterpret_cast<uintptr_t>(kernel.begin), end = reinterpret_cast<uintptr_t>(kernel.end);
    if (end <= begin || end - begin > std::numeric_limits<uint32_t>::max()) return GLUE_FAILED;
    const auto bytes = static_cast<uint32_t>(end - begin), capacity = out->kernel.elfSize;
    out->kernel.elfSize = bytes;
    if (capacity < bytes || !out->kernel.kernelElf) return GLUE_INSUFFICIENT_ELF_BUFFER;
    std::memcpy(out->kernel.kernelElf, kernel.begin, bytes);
    return GLUE_SUCCESS;
}
