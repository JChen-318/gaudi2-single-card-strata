#pragma once
#include "strata/hpu/half_capture.hpp"
#include "strata/hpu/host_recipe.hpp"
#include "strata/hpu/qsa.hpp"
#include <functional>

namespace strata::hpu {
class QsaWindow;
struct QsaWeights {
    // Borrowed HBM BF16: w_q_gate,w_k,w_v,w_index_query,w_index_key.
    std::vector<Binding> projections;
    Binding output; // w_out
    // Multiplicative FP32 gamma, not raw (1+w) weights. Query/key gamma may
    // be shared width or per head; index_key_gamma is exactly index_width.
    std::vector<float> query_gamma,key_gamma,index_query_gamma,index_key_gamma;
};
// Stateful correctness implementation: native MME projections/attention/output,
// CPU FP64 pooling/selection/RoPE, CPU FP16 paged KV. Synchronous and serialized;
// this control-plane placement and BF16 activations are not a performance claim.
// Weights must be uploaded/ready before construction and outlive the block.
class QsaBlock {
public:
    QsaBlock(const Device&,QsaConfig,uint32_t hidden,QsaWeights,QsaWindow* window=nullptr,QsaHalfSink half_sink={});
    ~QsaBlock();
    QsaBlock(const QsaBlock&)=delete;
    QsaBlock& operator=(const QsaBlock&)=delete;
    std::vector<float> decode(std::span<const float> normalized_hidden,std::span<const float> block_bias={});
    // Fused decode (Layer, --fp8-experts): the layer's graphs run the projections and the attention; the
    // host part of decode() in between (q/k/index norms, RoPE, cache append, selection) is fused_attend(),
    // and gather_selected() writes the selected rows. It appends to the cache, so a later failure is undone
    // with the layer's checkpoint.
    struct FusedAttention {std::vector<float> query,gate;std::vector<int32_t> ids;uint32_t bucket=0;float scale=1;};
    const std::vector<Binding>& projection_bindings() const;
    const Binding& output_binding() const;
    FusedAttention fused_attend(std::span<const float> q_gate,std::span<const float> k,std::span<const float> v,
                                std::span<const float> index_query,std::span<const float> index_key);
    // Writes the cells fused_attend() selected into caller buffers ([kv_heads,bucket,width] each and the shared
    // [bucket] mask) with no temporaries: the same values gather() would return, every element written.
    void gather_selected(std::span<const int32_t> ids,uint32_t bucket,std::span<float> key,std::span<float> value,
                         std::span<float> mask);
    // Optional shared executor must outlive calls. Atomic1..128 rows, native
    // projection/attention/output chunks<=8; no additional block bias.
    std::vector<float> prefill(std::span<const float>,const std::function<bool()>& cancelled={});
    uint32_t size() const;
    QsaState::Checkpoint checkpoint() const;
    QsaState::Checkpoint snapshot() const;
    // The parked-page reuse seam (see QsaState): the retained image's part sizes and the grow-in-place park.
    void snapshot_sizes(const QsaState::Checkpoint&,uint64_t (&sizes)[5]) const;
    QsaState::Checkpoint snapshot_reused(QsaState::Checkpoint&& prior,uint32_t whole_cells,uint32_t pooled_rows_kept) const;
    const QsaConfig& config() const;
    void write_snapshot(StateWriter&,const QsaState::Checkpoint&) const;
    QsaState::Checkpoint read_snapshot(StateReader&,uint32_t cells) const;
    void restore(const QsaState::Checkpoint&);
    void reset();
    // STRATA_HPU_QSA_DEVICE: the next decode token's attention on the device (tpc/qsa_prep.c) with this block's
    // KV cache in HBM, token-major [cells, kv_heads, width] F32 holding the FP16 pool's values. Possible while the
    // selection is the identity (size+1 within the selection width), with the FP16 pool, text rotation and shared
    // query/key gammas.
    bool device_ready() const;
    // Brings the device cache up to the committed prefix (uploading on `stream` and waiting, only after a restore,
    // reset or host-path tokens) and returns this token's bucket (cached cells read by the attention) and bindings:
    // dev.k_cache/dev.v_cache (the first `bucket` cells),
    // dev.rope (the cell's cosines then sines), dev.mask ([bucket+1]: the current cell first, then cells < size),
    // dev.q_gamma/dev.k_gamma.
    struct DeviceStep {uint32_t cell=0,bucket=0;std::vector<Binding> bindings;};
    DeviceStep device_step(const Stream&);
    // After the layer's launch: copies that token's K/V rows (in `stage` at `offset`, then +kv row bytes) into
    // the cache at `cell`, queued on `stream`.
    void device_store(const Stream&,uint32_t cell,const DeviceBuffer& stage,uint64_t offset);
    // Appends that token to the host state from the device's rows (FP16-representable K/V, raw index key), so
    // snapshots, rewinds and the host path see exactly the values the device attended.
    void device_commit(std::span<const float> k,std::span<const float> v,std::span<const float> raw_index_key);
    // STRATA_HPU_DFLASH verify of `rows` tokens as one window: device_verify_ready() is device_ready() with room
    // for every row in the identity selection; device_verify_step() is device_step() for the window (dev.rope
    // covers its rows; the attention mask is verify_mask()); device_store_rows() copies the window's K rows (at
    // `offset`) and V rows (at offset + rows K/V row bytes) into the cache from `cell`; device_commit_rows()
    // appends the first `count` window rows (row-major K, V and raw index keys) to the host state.
    bool device_verify_ready(uint32_t rows) const;
    DeviceStep device_verify_step(const Stream&,uint32_t rows);
    void device_store_rows(const Stream&,uint32_t cell,uint32_t rows,const DeviceBuffer& stage,uint64_t offset);
    void device_commit_rows(std::span<const float> k,std::span<const float> v,std::span<const float> raw_index_key,uint32_t count);
    // STRATA_HPU_QSA_DEVICE_SELECT: a window past the identity selection (layer.cpp build_select_qsa_pre/_chunk/
    // _post). keys is the attention key count (device_select_keys: >= cells + rows, at most ~12% padding); the step's
    // bindings carry the whole cache, the RoPE table, the indexer gammas, the host's complete pooled blocks and the
    // open block's raw keys; the window start (step.cell) is the caller's to pass.
    bool device_select_ready(uint32_t rows) const;
    static uint32_t device_select_keys(uint32_t cells,uint32_t rows,uint64_t capacity);
    DeviceStep device_select_step(const Stream&,uint32_t rows);
    // STRATA_HPU_QSA_DECODE_SELECT: a decoded token (rows 1, layer.cpp build_fused_qsa_decode_select) or a small verify
    // window (rows 2..8, build_verify_qsa_gather) past the identity selection; step.bucket is the pooled block rows the
    // graph scores.
    bool device_decode_select_ready(uint32_t rows=1) const;
    DeviceStep device_decode_select_step(const Stream&,uint32_t rows=1);
    uint32_t index_query_gamma_heads() const;
    // Long-context verify window: the same law as stage_qsa_window(), between the layer's device projection and
    // finish graphs. Kept because the device cache path only holds the identity selection (size+rows within the
    // selection width); here norms, RoPE, append and selection run on the host for every row. stage_rows()
    // checkpoints the state and appends this window's rows; gather_row() writes one row's selection into the
    // finish graph's buffers; rewind_staged() puts the committed prefix back once the finish graph is enqueued,
    // and commit_staged(count) re-appends only the accepted rows. drop_staged() abandons a window (idempotent).
    struct StagedAttention {std::vector<float> query,gate;uint32_t bucket=0;float scale=1;};
    bool staged_verify_ready(uint32_t rows) const;
    StagedAttention stage_rows(std::span<const float> q_gate,std::span<const float> k,std::span<const float> v,
                               std::span<const float> index_query,std::span<const float> index_key,uint32_t rows);
    void gather_row(uint32_t row,std::span<float> key,std::span<float> value,std::span<float> mask);
    // Device gather retains the exact host selection and FP16 cache rounding.
    // Call after append/stage_rows; upload just rows not already mirrored in HBM.
    std::vector<Binding> device_gather_bindings(const Stream&);
    // A contiguous row slice retains the per-row causal selection of stage_rows.
    // count=0 means all rows from first to the end.
    void staged_gather_ids(std::span<float> destination,uint32_t first=0,uint32_t count=0);
    void rewind_staged();
    void commit_staged(uint32_t count);
    void drop_staged();
    bool staged_pending() const;
    // [rows, 1, bucket+rows] F32 additive mask: row j attends the `cell` cached cells and window rows 0..j.
    static std::vector<float> verify_mask(uint32_t cell,uint32_t bucket,uint32_t rows);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
