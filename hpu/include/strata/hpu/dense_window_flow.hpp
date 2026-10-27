#pragma once
#include "strata/hpu/dense_kv.hpp"
#include "strata/hpu/kv_arena.hpp"
#include "strata/hpu/qsa_window_flow.hpp"
#include "strata/hpu/window_rows.hpp"
namespace strata::hpu {
struct DenseProjected {std::vector<float> q_gate,key,value;};
struct DenseNorms {std::span<const float> query,key;};
uint32_t dense_attention_bucket(uint32_t capacity,uint32_t cells,uint32_t window=0);
// Dense MTP attention: no pooled keys, indexer or sparse selection. Each query
// sees only its own prefix, using FP16 cache values and its own sliding window.
QsaWindowBatch stage_dense_window(DenseKvState&,const QsaConfig&,const DenseProjected&,uint32_t rows,uint32_t window,DenseNorms,const std::function<bool()>& cancelled={});
// Multi-SEQUENCE rows in one call: each row is appended into ITS OWN sequence's pool through the arena (at the plan's
// kv_base + position) and gathers that sequence's causal prefix, so one window can carry several sequences while each
// row still sees only its own history. All rows share ONE padded attention bucket (the widest sequence decides it), so
// the linear layers and the attention graph stay batched. Every touched pool rolls back on failure/cancellation.
// Refusals: rows outside 1..kQsaWindowRows, a projected shape that does not match the row count, a plan whose rows do
// not belong to the arena's regions, a row whose position is not its sequence's next cell, and any pool overflow.
QsaWindowBatch stage_arena_rows(KvArena&,const WindowRowPlan&,const QsaConfig&,const DenseProjected&,uint32_t window,
                                DenseNorms,const std::function<bool()>& cancelled={});
struct DenseWindowOperations {
    std::function<DenseProjected(std::span<const float>)> project;
    std::function<std::vector<float>(const QsaWindowBatch&)> attention;
    std::function<std::vector<float>(std::span<const float>)> output;
};
// Atomic1..128 rows, native chunks<=8, restoring all private KV on failure.
std::vector<float> execute_dense_window(DenseKvState&,const QsaConfig&,std::span<const float>,uint32_t hidden,uint32_t window,DenseNorms,const DenseWindowOperations&,const std::function<bool()>& cancelled={});
}
