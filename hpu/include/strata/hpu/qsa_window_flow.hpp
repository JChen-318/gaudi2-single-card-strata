#pragma once
#include "strata/hpu/window_limits.hpp"
#include "strata/hpu/qsa_state.hpp"
#include <functional>
namespace strata::hpu {
inline constexpr uint32_t kQsaWindowRows=8;
struct QsaProjected {std::vector<float> q_gate,key,value,index_query,index_key;};
struct QsaNorms {std::span<const float> query,key,index_query,index_key;};
struct QsaWindowBatch {
    uint32_t rows=0,bucket=0;
    // query/gate[T,HQ,D], key/value[T*HKV,bucket,D], mask[T*HKV,1,bucket]
    std::vector<float> query,gate,key,value,mask;
    // Optional [T,bucket] exact cell ids, -1 padding; no KV slabs in this mode.
    std::vector<float> ids;
};
uint32_t qsa_attention_bucket(const QsaConfig&,uint32_t selected);
// Device recipes put `extra` current rows behind `bucket` cached cells, so their
// attention width is bucket+extra.  The top selection bucket cannot be used there.
uint32_t qsa_device_bucket(const QsaConfig&,uint32_t selected,uint32_t extra);
uint64_t qsa_window_attention_pinned_bytes(const QsaConfig&,uint32_t rows,uint32_t bucket);
// Attention with device-produced KV/mask and no host outputs: query, gate,
// cell ids and scale only. Does not include a caller's injector or workspace.
uint64_t qsa_window_gather_pinned_bytes(const QsaConfig&,uint32_t rows,uint32_t bucket);
bool qsa_window_geometry_equal(const QsaConfig&,const QsaConfig&);
// Each row appends/selects against its own causal prefix before a later row is
// appended. Gathers may then use those fixed IDs because existing KV is immutable.
// Failure restores this chunk; ancestors remain valid under QsaState ancestry.
// `reuse` (optional) supplies the previous batch's buffers: every element is rewritten, so passing the same
// batch back avoids re-zeroing and reallocating [rows*HKV*bucket,width] twice per window.
QsaWindowBatch stage_qsa_window(QsaState&,const QsaProjected&,uint32_t rows,QsaNorms,const std::function<bool()>& cancelled={},
                               QsaWindowBatch* reuse=nullptr,bool selected_ids_only=false);
struct QsaWindowOperations {
    std::function<QsaProjected(std::span<const float>)> project;
    std::function<std::vector<float>(const QsaWindowBatch&)> attention;
    std::function<std::vector<float>(std::span<const float>)> output;
    bool selected_ids_only=false;
};
// Atomic1..128-row transaction, native chunks<=8. Only returns after every
// output and cancellation boundary passes. Failure rewinds the entire input.
std::vector<float> execute_qsa_window(QsaState&,std::span<const float> input,uint32_t hidden,QsaNorms,
                                      const QsaWindowOperations&,const std::function<bool()>& cancelled={});
}
