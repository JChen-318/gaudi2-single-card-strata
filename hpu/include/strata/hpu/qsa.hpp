#pragma once
#include "strata/hpu/qsa_state.hpp"
#include "strata/hpu/recipe.hpp"
#include <span>
#include <string>

namespace strata::hpu {
// Native FP32 MME/softmax gated attention for one selected KV bucket.
// Persistent inputs: query[HQ,D], key/value[HKV,bucket,D], mask[bucket],
// scale[1] = 1/sqrt(D), q_gate[HQ,D]. Output: out[HQ,D]. Mask is zero for
// selected cells followed by -inf padding. Caller guarantees >=1 valid cell.
// KV data has already undergone the upstream FP16 cache rounding.
void define_qsa_attention(GraphBuilder&,const QsaConfig&,uint32_t bucket,bool capture_half=false);
// Attach a gather producer to existing key/value/mask tensors. ids[rows,bucket]
// contains exact F32 cell ids, -1 for padding; cache K/V is token-major F32 on
// the FP16 grid. mask_heads is 1 for decode, kv_heads for a window.
void define_qsa_device_gather(GraphBuilder&,const QsaConfig&,uint32_t bucket,uint32_t rows,
                              TensorId key,TensorId value,TensorId mask,uint32_t mask_heads);
// The QSA device cache in graphs. A cache tensor `name` {cells, kv_heads, width} is F32, or with
// STRATA_HPU_QSA_KV_FP8 FP8 codes plus "<name>_scale" {cells, kQsaScaleSlots} F32: one scale per cell and head in
// a 128-byte slot per cell, so a binding at any cell stays 128-byte aligned (HostRecipe requires it).
// qsa_cache_read: the cache's first `cells` rows as `dtype` (F32 or BF16), dequantized when FP8.
// qsa_cache_write: `rows` F32 {cells, kv_heads, width} into the cache tensor `name` (copied, or quantized with a
// per-(cell, head) scale = max|x| / 240).
TensorId qsa_cache_read(GraphBuilder&,const std::string& name,uint64_t cells,uint64_t kv_heads,uint64_t width,DType dtype=DType::F32);
void qsa_cache_write(GraphBuilder&,const std::string& name,TensorId rows,uint64_t cells,uint64_t kv_heads,uint64_t width);
// Bytes of one cache row (all KV heads of a cell) and of its scales; the scales' buffer is unused without FP8.
inline constexpr uint64_t kQsaScaleSlots=32;
uint64_t qsa_cache_row_bytes(const QsaConfig&);
uint64_t qsa_cache_scale_bytes(const QsaConfig&);
// Host encoding of F32 cells {count, kv_heads, width} onto the device cache's FP8 grid: codes, and each cell's
// scales in its kQsaScaleSlots slot (the unused slots zero), for host uploads.
void qsa_cache_encode(std::span<const float> rows,uint64_t kv_heads,uint64_t width,uint8_t* codes,float* scales);
std::unique_ptr<Recipe> build_qsa_attention(const Device&,const QsaConfig&,uint32_t bucket,bool capture_half=false);
// BF16 projection stage, FP32 accumulation: normalized x[1,H] -> q_gate
// [HQ,2D], k/v[HKV,D], index_query[HI,DI], index_key[DI]. Normalization,
// RoPE and indexer/cache updates are performed by the CPU control plane.
// merged (STRATA_HPU_MERGED_GEMM): the five projections are one GEMM over w_qkvi (see qsa.cpp).
void define_qsa_projections(GraphBuilder&,const QsaConfig&,uint32_t hidden,bool merged=false);
std::unique_ptr<Recipe> build_qsa_projections(const Device&,const QsaConfig&,uint32_t hidden);
// Output projection: FP32 gated attention -> BF16 -> BF16 MME -> FP32 out.
// merged + STRATA_HPU_FP8_DENSE: w_out is FP8 with w_out_scale.
void define_qsa_output(GraphBuilder&,const QsaConfig&,uint32_t hidden,bool merged=false);
std::unique_ptr<Recipe> build_qsa_output(const Device&,const QsaConfig&,uint32_t hidden);
// Multi-row versions retain BF16 inputs / F32 projection outputs and perform
// attention over independently selected, FP16-rounded KV rows. Batch and KV
// head axes are merged for rank3 batch GEMM. Masks are [T*HKV,1,bucket].
// internal (merged only): q_gate/k/v/index_query are read only inside the graph (no host or binding reads them).
void define_qsa_projections_window(GraphBuilder&,const QsaConfig&,uint32_t hidden,uint32_t rows,bool merged=false,bool internal=false);
void define_qsa_attention_window(GraphBuilder&,const QsaConfig&,uint32_t bucket,uint32_t rows);
// Gather-free variant: batches over the KV head so all rows share one copy of the cache (see qsa.cpp).
void define_qsa_attention_dense_window(GraphBuilder&,const QsaConfig&,uint32_t cells,uint32_t rows);
std::unique_ptr<Recipe> build_qsa_attention_dense_window(const Device&,const QsaConfig&,uint32_t cells,uint32_t rows);
// Same attention reading the cache in its stored {cells, kv_heads, width} layout, transposed inside the graph.
void define_qsa_attention_dense_cache(GraphBuilder&,const QsaConfig&,uint32_t cells,uint32_t rows);
std::unique_ptr<Recipe> build_qsa_attention_dense_cache(const Device&,const QsaConfig&,uint32_t cells,uint32_t rows);
void define_qsa_output_window(GraphBuilder&,const QsaConfig&,uint32_t hidden,uint32_t rows,bool merged=false);
std::unique_ptr<Recipe> build_qsa_projections_window(const Device&,const QsaConfig&,uint32_t hidden,uint32_t rows);
std::unique_ptr<Recipe> build_qsa_attention_window(const Device&,const QsaConfig&,uint32_t bucket,uint32_t rows);
std::unique_ptr<Recipe> build_qsa_output_window(const Device&,const QsaConfig&,uint32_t hidden,uint32_t rows);
}
