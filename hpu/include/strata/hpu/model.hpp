#pragma once
#include "strata/kernels/ngram.hpp"
#include "strata/hpu/model_plan.hpp"
#include "strata/hpu/layer.hpp"
#include "strata/hpu/session_flow.hpp"
#include "strata/hpu/rope_config.hpp"
#include "strata/hpu/mtp.hpp"
#include "strata/hpu/mtp_sampling.hpp"
#include "strata/hpu/generation.hpp"
#include "strata/hpu/control_vector.hpp"
#include "strata/hpu/session_memory.hpp"
#include "strata/hpu/cpu_affinity.hpp"
#include "strata/hpu/vision_prompt.hpp"
#include <optional>

namespace strata::hpu {
struct ModelOptions {
    ModelMemoryOptions memory;
    bool prefill_auto=false; // resolve only after trunk/MTP/control/pack budgets
    std::optional<uint32_t> module_id;
    uint64_t position_base=0;
    RopeOptions rope;
    // --vision (src/program/generate.cpp:2104): --serve takes GENI requests with SVE1 embeddings and M-RoPE.
    // Off by default; an engine without it refuses GENI with upstream's sentence.
    bool vision=false;
    // Upstream's --eos-ids (src/program/generate.cpp:612 default pair, :2187-2191 parse that also arms). Empty
    // means the flag was absent: this backend then reads the stop list from the GGUF's tokenizer metadata, and
    // falls back to upstream's own pair for a file without it. strata/hpu/stop_tokens.hpp is that law.
    std::vector<int32_t> eos_ids;
    // --pool-workers / --pool-tasks (upstream src/program/generate.cpp:1980-1989, :5050-5051):
    // 0 workers = every physical core, 0 tasks = three per participating thread. --cpu-workers is
    // kept as a port alias for the same field, so the upstream name and the old name cannot diverge.
    uint32_t hpu_load_budget=UINT32_MAX,cpu_workers=0,cpu_tasks=0;
    // --pool-affinity (upstream src/program/generate.cpp:2006-2013) and worker pinning. Upstream's
    // default is all cores, pinned.
    CpuAffinity cpu_affinity;
    std::string mtp_runtime;
    std::string mtp_draft_vocab;
    std::string ple_gguf; // optional external PLE table, main model still owns hash metadata
    // Upstream's --no-ple (src/program/generate.cpp:1944, :2635-2645, :3434-3436): an explicit diagnostic
    // ablation that runs the model without the PLE input at all, never a normal inference mode. It is
    // mutually exclusive with --ple-gguf; strata/hpu/ple_mode.hpp is the shared admission law.
    bool no_ple=false;
    // --fp8-experts (port-only, opt-in): routed and shared experts are stored on the card as per-row-scaled
    // FP8 and run as MME FP8 GEMMs instead of the packed TPC dot (ExpertCache::set_fp8). Not upstream's
    // numeric path: the FP8 weight rounding adds ~2.5% relative error per projection.
    bool fp8_experts=false;
    // The PLE reader's own I/O options (--ple-io/--ple-row-cache/--ple-inflight,
    // strata/hpu/ple_io_mode.hpp). Row cache/queue/worker storage is charged before device creation.
    strata::kernels::PleIoOptions ple_io;
    double ple_delay_us=0;
    // Upstream's --embd-gguf (src/program/generate.cpp:397/:1972/:3032-3037): one GGUF holding
    // token_embd.weight that replaces the canonical embedding lookup for the rows it covers. Empty means
    // the flag is absent and the canonical table is used exactly as before. The row law itself lives in
    // strata/hpu/embedding_route.hpp, so prompt, window, verify replay and MTP draft cannot diverge.
    std::string embedding_gguf;
    std::string expert_pack; // original native_experts.txt / optional experts.bin
    MtpOptions mtp;
    MtpSamplingOptions mtp_sampling;
    // Upstream's expert-cache profiling/adaptation knobs: --expert-profile, --expert-profile-save,
    // --expert-profile-save-every, --adapt-every, --adapt-swaps, --adapt-decay, --adapt-min-gain,
    // --adapt-async (strata/hpu/expert_profile.hpp). All defaults are off/upstream's numbers, so the
    // default path allocates no counters and behaves exactly as before.
    ExpertProfileOptions expert_profile;
    // Upstream's expert-cache residency controls: --resident-experts, --resident-cpu-experts,
    // --resident-budget-gib, --expert-cache-per-layer, --mmap-experts (strata/hpu/expert_residency.hpp).
    // The tool parses them into these fields; the model resolves the two conditions upstream reads from other
    // options (the static profile is --expert-profile, and this single card has no remote expert cache) and
    // refuses an invalid combination before a device exists. All defaults off, so nothing changes unless asked.
    ExpertResidencyOptions residency;
    // Upstream's --spec-min-p default is 0.0 and it applies the cut ONLY when the operator raises it
    // (`if (use_mtp && o.spec_min_p > 0.0)`, src/program/generate.cpp:13755), so the default draft run is
    // bounded just by the window and the stop tokens. This backend's old 0.5 default cut every run below 50 %
    // confidence, which is a different decoder; 0 keeps --spec-min-p usable and matches upstream's default.
    float mtp_min_p=0.0f;
    ControlVectorOptions control_vector;
    // Diagnostic capture for the per-position dumps (both default off, and free when off):
    // `capture_layer_ladder` keeps the (n_layers+1)*hc*n_embd residual ladder of the last processed position
    // for --dump-layers; `capture_routing` keeps that position's routed ids/weights for --dump-routing.
    bool capture_layer_ladder=false, capture_routing=false,capture_halves=false;
    uint64_t session_min_free=4ull<<30;
    SessionMemoryProbe session_memory_probe; // optional embedding/test telemetry; default original provider
};
// Full native trunk session: GGUF embedding -> HC/PLE/mixers/MoE -> HC head.
// A Model owns one mutable sequence and serializes decode/reset. Construction
// acquires the card, so --plan/model-info must be used for no-card inspection.
class Model {
public:
    static ModelPlan preflight(const std::string& any_gguf_shard,ModelOptions={}); // never acquires a device
    Model(const std::string& any_gguf_shard,ModelOptions={});
    ~Model();
    Model(const Model&)=delete;
    Model& operator=(const Model&)=delete;
    // Routing callbacks borrow the completed physical view under the Model
    // lock, never re-enter Model. Scalar IO failure leaves the committed
    // numeric token; the caller must discard its failed request/slot owner.
    // The failed call's ephemeral routing capture is cleared on scope exit.
    std::vector<float> decode(int32_t token,const std::function<bool()>& cancelled={},const RoutingWindowSink& routing={});
    std::vector<float> prefill(std::span<const int32_t>,const std::function<bool()>& cancelled={});
    // With memory.prefill_rows, atomic layer-major micro-windows use grouped
    // MoE/HC and bounded PLE/GDN/QSA graphs. Zero keeps token fallback. Under
    // STRATA_HPU_RESIDENT_IQ4 the rows run through the shared verify chain
    // instead (one launch per layer, full windows only, then token fallback).
    // Commit notifications are synchronous under the Model lock; callbacks
    // must not re-enter Model. A thrown callback leaves its notified prefix;
    // an internal micro-window failure restores its entire unnotified window.
    // scalar=true preserves the token fallback and Model::prefill adaptation
    // cadence even when grouped prefill was configured (slot short-read tail).
    void prefill_window(std::span<const int32_t>,const PromptCommit&,const std::function<bool()>& cancelled={},
                        const RoutingWindowSink& routing={},bool scalar=false);
    // The tokens the next prefill_window() call will start with (a hint: a resident window prefetches its host rows).
    void prefill_next(std::span<const int32_t>);
    void reset();
    // Between requests. A changed arithmetic mode resets all target/MTP state.
    void set_control_vector(bool enabled);
    bool control_vector_enabled() const;
    std::string control_vector_summary() const;
    // ---- --vision's request law (GENI): install the request's per-cell (t,h,w) table and the image rows that
    // replace this prompt's embedding lookups. An empty selection restores the identity table and drops the rows,
    // which is what a text request does. Only a --vision engine accepts images; the caller plans the request with
    // vision_prompt.hpp first, so every refusal sentence is produced before any model state changes.
    struct VisionSelection {
        std::shared_ptr<const VisionEmbeddings> embeddings;
        std::vector<int64_t> rows;      // one per prompt cell, -1 for text
        std::vector<int32_t> positions; // cells*3, empty = identity
    };
    void set_vision(VisionSelection);
    std::vector<int32_t> tokens() const;
    uint64_t size() const;
    std::vector<float> logits() const;
    // Final HC residual of the last committed token, before output HC mixing.
    // MTP pairs this residual with the following token; failed/cancelled decode
    // must never publish its staged residual. Empty for a reset/root session.
    std::vector<float> residual() const;
    // The head input (upstream's `bb.mixed`) of the last committed token: --dump-mixed's one-shot value
    // (src/program/generate.cpp:14034-14052). Empty for a reset/root session.
    std::vector<float> mixed() const;
    // The residual ladder of the position just processed: (n_layers+1)*hc*n_embd floats, slot 0 the input
    // residual and 1..n_layers the value after each layer (--dump-layers' producer, upstream's `layer_stage`,
    // src/core/session.cpp:660-672). Empty unless ModelOptions::capture_layer_ladder was set.
    std::vector<float> layer_ladder() const;
    // The most recent successful physical micro-window, layer-major then row.
    // Scalar calls have one row; a long prefill retains its LAST micro-window.
    // Candidate capture preserves the initial pass, excluding commit replay.
    // Reset/restore/failed calls clear this ephemeral diagnostic, not snapshotted.
    // One exported copy is priced in dump_host_bytes; callers own retained copies.
    // Empty unless ModelOptions::capture_routing was set.
    std::vector<RoutingRecord> routing_records() const;
    // Last completed scalar target frame; grouped target windows invalidate it.
    // Packed FP16/int32 fields may be NaN as F32. One export is budgeted;
    // reset/restore/failed physical calls clear this non-snapshotted diagnostic.
    std::vector<float> half_records() const;
    std::vector<DraftRow> draft(const DraftContext&,uint32_t maximum,const std::function<bool()>& cancelled={});
    // Requires memory.verify_rows. Computes target candidates layer-major,
    // selects with the original sampler, retains only non-EOS accepted drafts.
    // Caller feeds a returned mismatch/bonus separately when generation goes on.
    VerifiedCandidates verify_candidates(const DraftContext&,std::span<const DraftRow>,const std::function<bool()>& cancelled={});
    // STRATA_HPU_DFLASH=<dir> with STRATA_HPU_DFLASH_VERIFY=1 (the resident chain with STRATA_HPU_QSA_DEVICE): one
    // speculative round. `pending` (picked by the caller from the last logits, not yet fed) and up to block-1
    // drafted tokens go through every layer as one window of at most `maximum` rows; `select` keeps a prefix of
    // it, which is committed. Returns an empty round, changing nothing, when no window can run now.
    bool dflash_ready() const;
    DflashRound dflash_window(int32_t pending,uint32_t maximum,const DflashSelection& select,const std::function<bool()>& cancelled={});
    // Batch-MTP: temporary MTP step from the last target residual and known
    // pending embedding proposes its successor. Rewind it, then verify
    // [pending,proposal] in ONE native target window. Commit pending and the
    // accepted non-EOS proposal only. nullopt leaves target/MTP unchanged.
    std::optional<SlotCandidates> slot_window(int32_t pending,std::span<const int32_t> history,
                                             const strata::kernels::SamplerParams&,const RoutingWindowSink& routing={});
    bool healthy() const;
    const ModelPlan& plan() const;
    uint32_t vocabulary() const;
    std::vector<int32_t> stop_tokens() const;
    // Immutable in-process state snapshot, with bounded shared ownership.
    // Restorable repeatedly after reset/branch replay on the same Model.
    // External cached logits must be restored alongside this state by callers.
    class Snapshot {
    public:
        Snapshot()=default;
        bool valid() const {return bool(data_);}
        explicit operator bool() const {return bool(data_);}
    private:
        struct Data;
        std::shared_ptr<const Data> data_;
        friend class Model;
    };
    Snapshot snapshot() const;
    uint64_t snapshot_size_limit() const;
    // Fresh capture at the CURRENT extent bounds additional RAM even when a
    // retained image is reused: its largest old component is already resident.
    // The arena-wide size limit remains the serialized-file bound/fallback.
    uint64_t snapshot_memory_cost() const;
    // The parked image's own cost at conversation extent `cells` (upstream's conversation_kv_capture_bytes),
    // which is what a park should be charged against the snapshot budget: snapshot_size_limit() is the
    // whole-arena reservation and prices a one-token conversation like a full-context one.
    uint64_t parked_bytes(uint64_t cells) const;
    // The parked-page reuse seam. Upstream keeps the restored/parked K/V image of the conversation it is
    // serving (include/strata/core/conversation_cache.hpp:217-233) so the next park of the same conversation
    // re-copies only what lies past the last whole page (src/core/conversation_snapshot.cpp:147-181).
    // parked_reused_bytes() reports the last park, parked_reused_total() the run's sum - upstream's
    // reused_kv_bytes counter - and parked_retained_bytes() the image held for the next park.
    uint64_t parked_reused_bytes() const;
    uint64_t parked_reused_total() const;
    uint64_t parked_retained_bytes() const;
    // Discard the retained image: the conversation it describes is gone (upstream's take_reuse() discard when
    // a session is replaced, and clear()).
    void discard_parked();
    void restore(const Snapshot&);
    // SAVE/RESTORE carry the live conversation's picture identity with the token prefix: upstream stores
    // live.imgs in the same session image (src/core/conversation_file.cpp:683-684, :758-759) and RESTORE
    // installs it on the live conversation (src/program/generate.cpp:10615). `images` are the pictures below
    // the saved extent; load() returns them for the caller's Generation::adopt().
    void save(const std::string& path,std::span<const RequestImage> images={}) const;
    std::vector<RequestImage> load(const std::string& path);
    AllocationStats hbm_allocations() const;
    AllocationStats pinned_allocations() const;
    AllocationStats snapshot_allocations() const;
    CacheStats expert_cache_stats() const;
    bool ple_ram_locked() const; // actual OS lock outcome, separate from plan/admission
    // --adapt-every: one adaptation round over the cache's routing evidence, performing the residency
    // changes it selects (the upload is the cache's own bounded staging). Returns how many landed; 0 with
    // the tier off. Call after a decode round/window; it never changes live numerics.
    uint64_t adapt_experts();
    // --expert-profile-save: rank what the run learned and write upstream's STRP file.
    bool expert_profile_save_due() const;
    bool save_expert_profile(std::string& error);
    const ExpertProfileOptions& expert_profile_options() const;
    // --resident-*/--expert-cache-per-layer: what was installed in the live cache (the per-layer quota and how
    // many experts the profile filled). Reporting only; the plan reports the same numbers before a device exists.
    const ExpertResidencyOptions& expert_residency_options() const;
    ExpertCache::ResidencyReport residency_report() const;
    // Between requests. Shrinks resident experts immediately; growth only
    // grants future capacity, bounded by the configured maximum and HBM cap.
    std::string reserve_vram(std::optional<uint64_t> reserve_bytes={});
private:
    // The resident multi-row prefill window: the verify chain's per-layer launches over [rows, streams, hidden],
    // committed only after the head. Returns the last row's logits and stages pending_residual/mixed_output.
    std::vector<float> prefill_resident_window(std::span<const int32_t>,const std::function<bool()>&);
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
