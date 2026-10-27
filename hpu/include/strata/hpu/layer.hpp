#pragma once
#include "strata/hpu/dense_weights.hpp"
#include "strata/hpu/expert_cache.hpp"
#include "strata/hpu/routing.hpp"
#include "strata/hpu/qsa_state.hpp"
#include "strata/hpu/gdn_block.hpp"
#include "strata/hpu/half_capture.hpp"
#include "strata/hpu/decode_chain.hpp"
#include <functional>

namespace strata::hpu {
class CpuExperts;
class HcWindow;
class GdnWindow;
class QsaWindow;
struct LayerOptions {
    uint32_t context=32768,page_size=4;
    uint32_t prefill_rows=0; // opt-in layer-major MoE window, at most128
    KvMode kv=KvMode::Fp16; // --kv storage format for this layer's QSA pool
    HcWindow* hc_window=nullptr; // optional shared Model-owned batched HC executor
    GdnWindow* gdn_window=nullptr; // shared bounded GDN projection/causal executor
    QsaWindow* qsa_window=nullptr; // shared bounded QSA projection/attention/output
    uint64_t dense_payload_budget=256*1024*1024,staging_budget=1024*1024;
    uint64_t position_base=0;
    strata::kernels::RopeScaling rope;
    // Request-scoped image positions (--vision); the Model owns the table, null keeps the text-only path.
    const RopePositions* positions=nullptr;
    CpuExperts* cpu=nullptr;
    uint32_t hpu_load_budget=UINT32_MAX;
    // --no-ple (strata/hpu/ple_mode.hpp): the diagnostic ablation builds no layer-1 PLE projection and
    // accepts no PLE span, matching the model plan's omission of the PLE weights.
    bool ple_enabled=true;
    // --dump-routing producer, forwarded to this layer's MoeBlock. Empty keeps the default path free.
    std::function<void(uint32_t,const Routes&)> routing_sink;
    LayerHalfSink half_sink;
    LayerQsaHalfSink qsa_half_sink; // scalar path only; window capture is separate
    // --fp8-experts: a GDN layer's single-token decode runs as two launches (HC read, GDN, HC write, HC
    // read and router; then experts, their sum and the HC write) when its MoE supports it. Off keeps the
    // per-stage path; capture sinks always use it.
    bool fused_decode=false;
};
// Complete single-token Strata layer, including layer-1 PLE and both HC
// halves. Immutable weights are owned; expert cache is shared across layers
// and must outlive this object. Inputs/outputs are FP32 residual[HC,H].
// STRATA_HPU_DFLASH verify: one layer's slice of VerifyChain::stage (K rows, V rows, raw index keys of a window).
uint64_t verify_stage_stride(const core::ModelGeometry&,uint32_t max_rows);
// The verify stage holds a slice per QSA layer only (GDN layers stage nothing): the stage's size, and where `layer`'s
// slice starts.
uint64_t verify_stage_bytes(const core::ModelGeometry&,uint32_t max_rows);
uint64_t verify_stage_offset(const core::ModelGeometry&,uint32_t layer,uint32_t max_rows);
// STRATA_HPU_QSA_DEVICE_SELECT: an upper bound on the select recipes' HBM (build_select_qsa up to the widest key bucket
// a `context` admits): their shared workspace (a few live row chunks and the per-key tables) and each bucket's own tensors.
uint64_t qsa_select_window_bytes(const QsaConfig&,uint32_t rows,uint32_t context);
class Layer {
public:
    Layer(const Device&,WeightTable,uint32_t layer,ExpertCache&,LayerOptions={});
    ~Layer();
    Layer(const Layer&)=delete;
    Layer& operator=(const Layer&)=delete;
    // ple_embedding[H] is required at layer 1 and rejected at other layers.
    std::vector<float> decode(std::span<const float> residual,std::span<const float> ple_embedding={});
    // STRATA_HPU_CHAIN: the fused decode on the chain's device residual (chain.in() -> chain.out(), then
    // advance). The host waits only for the router logits (and a QSA layer's projections); the expert group
    // is left running. Requires chainable(); a failure restores this layer's mixer like decode().
    bool chainable() const;
    void decode_chain(DecodeChain& chain,std::span<const float> ple_embedding={});
    // STRATA_HPU_DFLASH verify (the resident chain with STRATA_HPU_QSA_DEVICE): verify_ready() says whether this
    // layer can run a window of `rows` tokens; verify_chain() runs it as one launch, chain.in() -> chain.out()
    // (then advances), ple_rows [rows, hidden] at layer 1, accepting nothing; verify_commit() accepts the first
    // `accepted` rows once chain.stage_host and chain.ple_history_host hold the window's downloaded results; it is
    // host work only (nothing is queued on chain.stream).
    bool verify_ready(uint32_t rows) const;
    void verify_chain(VerifyChain& chain,std::span<const float> ple_rows={});
    void verify_commit(VerifyChain& chain,uint32_t accepted);
    // Atomic [T,HC,H] window. PLE/GDN/QSA use causal native chunks<=8;
    // Model-shared HC and grouped experts execute all T rows in batch graphs.
    std::vector<float> prefill(std::span<const float>,std::span<const float> ple_embeddings={},const std::function<bool()>& cancelled={});
    void reset();
    uint64_t size() const;
    bool healthy() const;
    uint64_t dense_payload_bytes() const;
    // Ephemeral rewind point for a model-wide token transaction. This is not
    // a persistent snapshot: a discarded branch cannot be restored/reused.
    struct Checkpoint {
    private:
        uint64_t owner=0,position=0;
        bool persistent=false;
        GdnBlock::Checkpoint gdn;
        QsaState::Checkpoint qsa;
        std::vector<float> ple_history;
        friend class Layer;
    };
    Checkpoint checkpoint() const;
    Checkpoint snapshot() const;
    static uint64_t snapshot_capacity_bytes(const Checkpoint& p) {
        return snapshot_add_bytes(snapshot_vector_bytes(p.ple_history),snapshot_add_bytes(GdnBlock::snapshot_capacity_bytes(p.gdn),QsaState::snapshot_capacity_bytes(p.qsa)));
    }
    // Upstream's reuse branch of the park (src/core/conversation_snapshot.cpp:147-181). `point` must be the
    // caller's EXCLUSIVELY owned retained image of THIS conversation: it is grown in place, with the boundaries
    // taken from the parked-page law (strata/hpu/parked_pages.hpp), and the bytes carried over are returned.
    // A GDN layer has no parked K/V image - upstream's reuse covers the QSA pool parts only - so its state is
    // re-captured and 0 is reported. Throws where upstream refuses the reuse.
    uint64_t snapshot_reused(Checkpoint& point,uint64_t unchanged_tokens) const;
    // What a park of THIS layer costs at conversation extent `cells`, part by part, the way upstream prices a
    // park with conversation_kv_capture_bytes (src/core/conversation_snapshot.cpp:124-145) instead of charging
    // the whole-arena limit: the K/V and indexer bytes the image will actually hold, plus the in-progress
    // indexer block and the PLE history that travel with it. A GDN layer contributes its recurrent state.
    uint64_t parked_bytes(uint64_t cells) const;
    void write_snapshot(StateWriter&,const Checkpoint&) const;
    Checkpoint read_snapshot(StateReader&,uint64_t position) const;
    void restore(const Checkpoint&);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
