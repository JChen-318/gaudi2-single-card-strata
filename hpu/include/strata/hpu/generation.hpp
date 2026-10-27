#pragma once
#include "strata/hpu/verify.hpp"
#include "strata/hpu/session_flow.hpp"
#include "strata/hpu/prefix_cache.hpp"
#include "strata/hpu/prompt_boundaries.hpp"
#include "strata/hpu/kv_residency.hpp" // --kv / --kv-grow / --no-kv-grow / --kv-resident policy (additive)
#include "strata/hpu/spec_draft_source.hpp" // --spec-oracle/--spec-follow/--spec-corrupt (additive)
#include "strata/hpu/dump_sinks.hpp" // upstream's --dump-* per-position diagnostic records (additive)
#include "strata/hpu/window_hashes.hpp" // upstream's --window-hashes determinism record (additive)
#include "strata/hpu/vision_prompt.hpp" // --vision GENI: SVE1 image rows and the M-RoPE table
#include "strata/hpu/image_key.hpp" // upstream's ConversationImageKey and the imgs_below law
#include "strata/spec/draft_policy.hpp"
#include <functional>
#include <array>
#include <memory>
#include <memory>
#include <string>

namespace strata::hpu {
struct GenerationRequest {
    std::vector<int32_t> prompt;
    uint32_t max_new=1;
    strata::kernels::SamplerParams sampling;
    std::vector<float> bias;
    bool checkpoint=true;
    uint32_t pin=0;
    bool control_vector=true;
    // Upstream's per-request speculative keys (src/program/generate.cpp:10696-10698). Both are initialized from the
    // start-up values, so an absent key must not change anything: `spec_min_p` is the draft-confidence gate clamped
    // into [0,1] (-1 here = "not named"), and `aux_cpus` is 0|1 telling where this request's non-pool, non-host
    // threads run (-1 = the start-up --aux-cpus, upstream's own sentinel at :10666). strata/hpu/spec_gate.hpp is the
    // gate law; `pcie_frac` is refused by name in the parser because this backend has no host->device pipeline to
    // split.
    float spec_min_p=-1.0f;
    int aux_cpus=-1;
    // --vision GENI: the validated SVE1 file, the prompt-cell rows it supplies and the request's (t,h,w)
    // table. All empty for a text request, which restores the identity table in the model.
    std::shared_ptr<const VisionEmbeddings> embeddings;
    std::vector<int64_t> image_rows;
    std::vector<int32_t> positions;
    std::vector<RequestImage> images;
};
struct GenerationResult {
    uint32_t generated=0,prompt_tokens=0,reused=0;
    uint64_t verify_windows=0,drafts_proposed=0,drafts_checked=0,drafts_accepted=0;
    uint64_t chain_proposed=0,chain_accepted=0;
    // Upstream's follow report (src/program/generate.cpp:13882,13917-13919): emitted tokens whose
    // window argmax differed from the --spec-follow continuation, and the emitted count.
    uint64_t follow_differ=0,follow_emitted=0;
    // Records the per-position dumps took for this run, so the tool can print upstream's summary
    // lines (src/program/generate.cpp:14004-14018,14171) from the counts, not by re-reading files.
    uint64_t dumped_logits_rows=0,dumped_final_r_records=0,dumped_routing_records=0,dumped_half_records=0;
    double prompt_ms=0,decode_ms=0;
    std::string finish="length";
    int yielded_slot=-1;
    uint32_t yielded_tokens=0,prompt_read=0;
};
struct DraftContext {
    std::span<const int32_t> history;
    std::span<const float> logits,bias;
    strata::kernels::SamplerParams sampling; // counter for the first proposal
    VerifyMode mode=VerifyMode::Exact;
    // Synchronous borrowed <=8 continuation cells for the native selector.
    // Apply before acceptance/commit, never rewrite already committed picks.
    std::span<const int32_t> follow={};
    RoutingWindowSink routing={}; // native initial pass, before accepted-prefix commit
    // Optional borrowed output, <=7 cells: actual drafter confidence (raw
    // full-head softmax for the default MTP argmax, not a greedy point mass).
    std::span<float> confidence={};
};
// Borrowed immutable history, synchronous selection only. The owned restore
// keeps its model payload alive if the originating cache entry is later freed.
// logits() is read AFTER restore; no vocabulary buffer per source is required.
struct GenerationCheckpoint {
    std::span<const int32_t> tokens;
    std::function<void()> restore;
    uint64_t age=0;
    bool pinned=false,root=false,tail=false;
    // The pictures starting below tokens.size() when the state was captured (upstream's ConvCheckpoint::imgs,
    // generate.cpp:8449). Borrowed only for the synchronous visit.
    std::span<const RequestImage> images={};
};
using GenerationCheckpointVisitor=std::function<void(const GenerationCheckpoint&)>;
using GenerationCheckpoints=std::function<void(const GenerationCheckpointVisitor&)>;
struct GenerationRead {
    std::span<const int32_t> tokens;
    bool control_vector=true,from0=false,can_yield=false;
    GenerationCheckpoints checkpoints={}; // synchronous borrowed views, no escaping
    std::function<void()> restore; // endpoint guard for an incoming chain
    std::function<std::vector<float>()> logits; // queried after restoring each point
    // Owned copy of the pictures below tokens.size(): the chain contract and the imported entries carry them.
    std::vector<RequestImage> images;
};
struct GenerationPrefix {
    std::span<const int32_t> tokens;
    std::function<void()> restore;
    std::function<std::vector<float>()> logits;
    GenerationCheckpoints checkpoints={};
    bool from0=false;
    std::vector<RequestImage> images={};
};
// Synchronous window-local cumulative committed count and its last logits.
// May report a prefix before failing/cancelling; no per-row logits slab is
// required. The callback must finish before prefill returns and may not escape.
using PromptCommit=std::function<void(uint32_t,std::span<const float>)>;
// STRATA_HPU_DFLASH_VERIFY round (Model::dflash_window): `tokens` were committed (the pending token, then the
// accepted drafts) and `logits` is the distribution after the last of them; `proposed` drafts went through the
// target. An empty round means the model could not run one now (the caller decodes as usual). A sampled request
// under VerifyMode::Probabilistic reaches this hook with point-mass drafts: a rejected row leaves its residual
// draw with the caller, which feeds it as the next round's pending instead of drawing from p again.
struct DflashRound {std::vector<int32_t> tokens;std::vector<float> logits;uint32_t proposed=0;};
// A window for DflashSelection: its ids (pending first), the target's argmax after each row (computed on the device;
// among exactly equal logits it may not be the lowest id) and row(j), the logits after row j (downloaded on first
// use). The selection returns how many leading rows to keep, 1..rows, and must not re-enter the model.
struct DflashRows {std::span<const int32_t> ids,argmax;std::function<std::span<const float>(uint32_t)> row;};
using DflashSelection=std::function<uint32_t(const DflashRows&)>;
struct GenerationOperations {
    std::function<void()> reset;
    std::function<std::vector<float>(int32_t,const std::function<bool()>&)> decode;
    std::function<void(std::span<const int32_t>,const PromptCommit&,const std::function<bool()>&)> prefill;
    std::function<void(std::span<const int32_t>,const PromptCommit&,const std::function<bool()>&,const RoutingWindowSink&)> prefill_routing;
    // Optional: the prompt chunk the next prefill call will take (a hint only; it lets the backend build that
    // chunk's host input rows while the current one runs). Called right before prefill.
    std::function<void(std::span<const int32_t>)> prefill_next;
    // Proposal-only: must not mutate the target model. Receives the committed
    // prefix and a hard row limit. MTP may own separate state behind this hook.
    std::function<std::vector<DraftRow>(const DraftContext&,uint32_t,const std::function<bool()>&)> draft;
    // Optional monotonic-ms timing seam for the adaptive policy. Empty uses
    // steady_clock. Synthetic host clocks verify learning, never HPU speed.
    std::function<double()> policy_clock_ms;
    // Optional diagnostic publishers for upstream's residual dumps. `residual` is the post-last-layer HC
    // residual of the position just processed (hc x n_embd floats - upstream's `ss.R`, the value
    // --dump-final-r appends per position at :13363-13368 and --dump-residual writes once at :14055-14078).
    // `mixed` is the head input (n_embd floats, upstream's `bb.mixed`, --dump-mixed at :14034-14052).
    // Absent hooks mean those dumps are REFUSED at run entry, exactly as upstream refuses a dump whose
    // producer never runs (:5993-6007); the default-off path is untouched.
    std::function<std::vector<float>()> residual;
    std::function<std::vector<float>()> mixed;
    // Optional diagnostic publisher for `--dump-layers`: the residual LADDER of the position just processed,
    // (n_layers+1)*hc*n_embd floats, slot 0 the input residual and slots 1..n_layers the value after each
    // layer (upstream's `layer_stage`, src/core/session.cpp:660-672, written at generate.cpp:13382). Absent
    // means the flag is REFUSED at run entry instead of writing a ladder the model never captured.
    std::function<std::vector<float>()> layer_ladder;
    std::function<std::vector<float>()> halves; // complete scalar frame, raw packed fields preserved
    // Optional diagnostic publisher for `--dump-routing`: this position's routed experts and weights, in
    // upstream's per-layer record order (:1007-1023). Absent means the flag is REFUSED at run entry.
    std::function<std::vector<RoutingRecord>()> routing;
    // Optional atomic native target window: retains accepted non-EOS draft
    // tokens and returns their last logits. Selects before publishing tokens.
    std::function<VerifiedCandidates(const DraftContext&,std::span<const DraftRow>,const std::function<bool()>&)> verify;
    // Optional STRATA_HPU_DFLASH_VERIFY round for greedy requests: (pending, most tokens to commit, selection).
    std::function<DflashRound(int32_t,uint32_t,const DflashSelection&,const std::function<bool()>&)> dflash;
    bool verify_routing=false; // verify consumes DraftContext::routing synchronously
    uint64_t checkpoint_bytes=0;
    // Upstream prices a park by the image's own bytes (conversation_kv_capture_bytes), not by the arena's
    // reservation, so a short conversation costs what it actually parks. Optional: without it the cache keeps
    // charging checkpoint_bytes, which is what it did before the parked-page law was wired.
    std::function<uint64_t(uint64_t cells)> checkpoint_park_bytes;
    // Upstream parks one image PER STAGE when the session is layer-split (src/program/generate.cpp:8244-8285:
    // `n_st = stages.size()`, checkpoints split into per-stage parts, and `reuse.stages` carries one retained
    // image per stage). This backend has one card, so its own admission law can only ever produce no split, or a
    // same-GPU split - and generate.cpp:2382-2386 refuses to park a same-GPU split at all. Reporting more than
    // one stage here therefore means a caller presented a session this backend cannot park as a single image, and
    // the park is refused by name instead of silently storing one stage's image for a multi-stage session.
    std::function<uint32_t()> parking_stages;
    // Owned persistent restore, also used by scalar candidate transactions.
    // If no native verify hook exists, real drafts need this bounded image;
    // refusal happens before evaluating/publishing any candidate window row.
    std::function<std::function<void()>()> checkpoint;
    // The parked-page reuse seam (Model::parked_reused_bytes / Model::discard_parked). `parked_reused_bytes`
    // reports the bytes the last park carried over from the retained image instead of re-reading them from the
    // pools - upstream's reused_kv_bytes (src/program/generate.cpp:8371) - and `discard_parked` drops that
    // image when the conversation it described is gone (upstream's take_reuse() discard, generate.cpp:10599).
    std::function<uint64_t()> parked_reused_bytes;
    std::function<void()> discard_parked;
    // The size of the image the state owner is holding for the next park, so the cache can charge it the way
    // upstream's bytes() charges reuse_ (include/strata/core/conversation_cache.hpp:207).
    std::function<uint64_t()> parked_retained_bytes;
    PrefixCache::MemoryProbe available_host_bytes; // default: original process/cgroup-aware provider
    std::function<void(bool)> control_vector; // optional between-request arithmetic switch
    // --vision: install this request's M-RoPE table and image rows before any prompt work. A request
    // without images clears both, so the next text request sees the identity table upstream rebuilds.
    std::function<void(const GenerationRequest&)> set_vision;
    std::function<std::optional<GenerationPrefix>(std::span<const int32_t>,uint64_t,bool,std::span<const RequestImage>)> select_prefix;
    // Upstream's #2228 serve diagnostic (src/program/generate.cpp:10951-10983): a live state or checkpoint
    // deeper than what this run reuses has the prompt's tokens but not its pictures (an image encoded again
    // with other bits), so the deeper state cannot be reused. Reports the state's token count, the 1-based
    // picture number and that picture's start cell; the fallback to the shallower match is unchanged.
    // Absent means no report. Only the live state and checkpoints in the request's control mode are considered.
    std::function<void(uint64_t,uint64_t,int64_t)> picture_mismatch;
};
// Upstream --prefill-until (src/program/generate.cpp:13165): batch only positions [0, P) and run
// the rest of the prompt one token at a time, so logits after P can be scored against an oracle.
// The flag can only LOWER the batched span: 0, a negative value or anything >= n_prompt-1 all mean
// "the whole prompt but the last position", which is the default.
inline size_t prefill_batched_end(size_t prompt_tokens,uint64_t prefill_until,bool token_loop_suffix=false) {
    // Declared divergence, on purpose: upstream's default batches only [0, n_prompt-1) and walks the
    // final position through the token path, which produced the same logits by a longer route. This
    // backend keeps batching the whole prompt when the flag is unset, so adopting --prefill-until
    // cannot silently move the last position out of the window for every existing request.
    const size_t last=prompt_tokens?prompt_tokens-1:0;
    // Opt-in raw diagnostics use upstream's token-loop suffix, so its final
    // prompt position must pass through that loop even with an unset limit.
    // Ordinary requests retain the already declared HPU batching default.
    if(!prefill_until) return token_loop_suffix?last:prompt_tokens;
    return prefill_until<last?size_t(prefill_until):last;
}
// Upstream keeps --prompt-cache 6 conversation checkpoints between requests by default
// (src/program/generate.cpp:632), so a backend that defaults to none differs silently even with every option
// ported. This backend can only promise those checkpoints once the operator granted the RAM they cost: the
// prefix cache is charged against --snapshot-budget-mib, and slots without a byte limit is a contradiction the
// cache itself throws on (hpu/src/generation.cpp:19). So the default IS upstream's wherever it is affordable and
// zero where no budget was given - and zero is reported, never hidden. An explicit --prompt-cache always wins,
// including an explicit 0.
inline constexpr uint32_t upstream_prompt_cache_slots = 6;
inline uint32_t default_prompt_cache_slots(bool given,uint64_t snapshot_budget,uint32_t requested) {
    return given?requested:(snapshot_budget?upstream_prompt_cache_slots:0u);
}
// What the run says about the choice, so an operator sees why a default differs from upstream's number.
inline std::string prompt_cache_note(uint32_t slots) {
    return slots?"prompt_cache=default":"prompt_cache=no-snapshot-budget";
}
struct GenerationOptions {
    uint32_t max_draft=7; // original kVerifyMaxT=8, including one bonus row
    bool adaptive_drafts=false; // explicit opt-in until real Gaudi performance is measured
    // DECLARED DIVERGENCE: upstream defaults --suffix-draft to 3, so prompt-lookup drafting is on unless the
    // operator turns it off (`sfx_on = o.suffix_draft > 0 || o.lookup_chain > 0`, src/program/generate.cpp:11859,
    // with the --spec window widening separate at :2567). This backend keeps 0 = off as its default because its
    // --batch path rejects a real drafter per slot (the slots engine has no per-slot draft window), so an
    // on-by-default suffix would make `--serve --batch` unusable without an explicit --suffix-draft 0. Flipping
    // the default belongs with per-slot drafting; work/audit_cli_defaults.py records this row as declared.
    uint32_t suffix_min_match=0;

    uint32_t lookup_chain=0,lookup_chain_min=3;
    // --check-logits: upstream's full-vocabulary finite scan of every read-back row (src/program/generate.cpp:13414-
    // 13420), which turns a non-finite value into a HARD failure with its position instead of the protocol's
    // `nonfinite` finish. The scan itself is free here (the row is always copied whole for host sampling); the flag
    // only decides which report wins.
    bool check_logits=false;
    VerifyMode verify_mode=VerifyMode::Exact;
    uint32_t cache_slots=0;
    uint32_t cache_every=16384; // fresh prompt tokens since this request resumed; 0 disables intervals
    // The trailing prompt tokens read one row at a time instead of inside an N-row window. Both paths are correct,
    // but the window costs 2.87 ms per row and the single-row path 13.5 ms, so the old 64-row tail cost 874 ms of a
    // 2210 ms 512-token prefill (measured, round 32). 1 keeps only the final row sequential: prefill 232 -> 336 tok/s.
    uint32_t prefill_chunk=0,short_read=1; // zero: token path; backend controls bounded window execution
    uint32_t prefill_until=0; // upstream --prefill-until: batch [0,P) then use the token path
    bool cache_tail=false;
    uint64_t cache_bytes=1ull<<30;
    uint64_t cache_min_free=0; // CLI defaults to original 2560 MiB; excludes snapshot budget
    // Upstream's --conversation-cache-min-tokens (generate.cpp:639 default 0, parse :2107-2124, law
    // conversation_cache.hpp:206, park site :8227-8236). It lives here rather than at each park call so a caller
    // cannot park past it; PrefixCache::store is the only door.
    int64_t cache_min_tokens=0;
    // Whether an emitted stop token ENDS this run. Upstream gates the ordinary token loop on --stop-eos
    // (:13493-13494, :13511, :13881) while batch and decode-verify windows test eos_ids ungated (:10077, :12424).
    // Generation is the ordinary single-request path, so its callers pass the armed answer computed by
    // strata::hpu::stop_tokens_armed; the default keeps this header's existing callers armed, because an
    // in-process caller is not the CLI and has no flag to read. Batch slots never consult this - SlotSession's
    // own is_stop stays ungated, exactly as upstream's window is.
    bool stop_eos=true;
    // Upstream's `--window-hashes PATH` (:586, :2088, opened :13723-13727, one line per verify window
    // :13825-13834): index, position, row count, a 64-bit FNV-1a over the window's final residual rows, then the
    // window's argmaxes. Two builds doing the same arithmetic must write the same bytes, so this is a determinism
    // record, not a log. Empty = off, and nothing is collected.
    std::string window_hashes;
    // Upstream's start-up --spec-min-p (field :609, default 0.0 = no gate): it truncates the verify window by the
    // drafter's own confidences (strata/hpu/spec_gate.hpp; :12566-12568). A request may raise or lower it per run
    // with the `spec_min_p=` key, which is why the two live in different places and share one law.
    float spec_min_p=0.0f;
    // Upstream's KV residency controls (--kv / --kv-grow / --no-kv-grow / --kv-resident, the policy unit in
    // strata/hpu/kv_residency.hpp). The byte cap is enforced where the arena is allocated - the model plan -
    // because this backend commits the whole arena up front; what is decided per request is whether the
    // arena may grow, and a request that needs growth without it is refused before any state is touched.
    KvResidencyOptions kv;
    uint64_t kv_mapped_cells=0; // cells the arena commits: 0 = the full context, the only mode here
    PromptBoundaryOptions boundaries;
    // Upstream's per-position diagnostic dumps (--dump-logits/--dump-layers/--dump-halves/--dump-mixed/
    // --dump-residual/--dump-final-r/--dump-routing, src/program/generate.cpp). All paths empty = off,
    // and off costs nothing: no file is opened and the per-position writers are never reached.
    // `dump_shape` carries the model geometry the records are sized against (the caller fills n_layers,
    // hc, n_embd, n_head, head_dim, n_head_kv and n_expert; n_vocab and the position budget are taken
    // from this generation and the request). `dump_producers` declares which in-model hooks are wired;
    // a dump whose producer is absent is refused before any state is touched.
    DumpOptions dumps;
    DumpGeometry dump_shape;
    DumpProducers dump_producers;
    // Optional process-owned routing-only writer, opened once before READY.
    // Generation borrows it per request, never truncates/closes it; slot calls
    // use the same serial writer. Empty retains one-shot file ownership.
    std::shared_ptr<DumpSinks> resident_routing;
    uint32_t canonical_prefill_rows=0; // opt-in backend arithmetic unit; zero preserves the existing schedule
};
void validate_generation_options(const GenerationOptions&);
struct GenerationEvents {
    std::function<void(uint32_t)> reused;
    std::function<void(uint32_t,uint32_t,double)> progress;
    std::function<void(int32_t)> token;
    // Upstream's per-park serve line (src/program/generate.cpp:8371): the parked extent, the bytes carried over
    // from the retained image instead of re-read from the pools, and whether the park replaced a superseded copy
    // of this same conversation instead of taking another slot.
    std::function<void(uint64_t,uint64_t,bool)> parked;
    // The refusal counterpart of `parked`: how many stages the session presented, so an operator sees that
    // nothing was parked rather than reading silence as "parked and reused nothing".
    std::function<void(uint32_t)> park_refused;
    // Upstream's --conversation-cache-min-tokens skip (src/program/generate.cpp:8227-8236): the conversation's
    // extent, the configured minimum, and how many conversations are parked now. A SKIP, not a refusal and not an
    // error - upstream's park callback returns true and the request runs on - but the operator has to be told,
    // because the cache looking smaller than the traffic is otherwise indistinguishable from a budget refusal.
    std::function<void(uint64_t,uint64_t,uint64_t)> park_skipped_short;
    // Called after Model callbacks return, with fully committed metadata.
    std::function<void()> cooperate;
    std::function<bool()> has_foreign_work;
    std::function<int(const GenerationRead&)> yield; // >=0: a slot owns the partial read
};
// Serialized resident generation. Exact target verification commits only an
// accepted prefix and the first mismatch/bonus token. An optional native hook
// executes candidates layer-major; otherwise verification stays sequential.
class Generation {
public:
    Generation(uint32_t vocabulary,uint32_t context,std::vector<int32_t> stop_ids,GenerationOperations,GenerationOptions={});
    GenerationResult run(const GenerationRequest&,const GenerationEvents&,const std::function<bool()>& cancelled={});
    void reset();
    void adopt(std::span<const int32_t> tokens,std::span<const float> logits,bool control_vector=true,
               std::span<const RequestImage> images={});
    // The pictures below the live extent: upstream's live_imgs, which SAVE writes beside the token prefix
    // (src/program/generate.cpp:10488, :10522) and RESTORE installs on the live conversation (:10615).
    std::span<const RequestImage> live_images() const {return images_below(sequence_images_,fed_.size());}
    // Before another serialized session changes the ONE Model. Prefer an
    // existing parked end alias; a budget refusal drops live reuse, not cache.
    void suspend_live();
    // Explicitly mount THIS main owner's complete live state for disk SAVE.
    // Never substitute the last slot, a shorter checkpoint, or lost metadata.
    void attach_live();
    void mark_nonfinite() {nonfinite_pending_=true;}
    bool nonfinite_pending() const {return nonfinite_pending_;}
    bool flush_nonfinite(); // next accepted request, no foreign Model reset
    // `images` is the request's picture list; a live or parked prefix matches only when the pictures below its
    // length are exactly the stored ones (generate.cpp:10856-10861).
    std::optional<GenerationPrefix> select_prefix(std::span<const int32_t>,uint64_t maximum,bool control_vector,
                                                  std::span<const RequestImage> images={});
    void import_prefixes(const GenerationRead&); // reattaches endpoint even on failure
    void invalidate();
    PrefixCacheStats cache_stats() const;
    struct MetadataStats {uint64_t envelope=0,vector_bytes=0;uint32_t interrupted_entries=0;};
    MetadataStats metadata_stats() const;
    uint64_t parked_reused_bytes() const {return parked_reused_;}
    // The stage count of the most recent park this backend refused because the session was presented as a
    // multi-stage (layer-split) one, or 0 when no park has been refused.
    uint32_t park_refused_stages() const {return park_refused_stages_;}
    void release_checkpoints(); // frees parked state without changing live state
    // Upstream's speculative-decoding verification/diagnostic knobs (additive; default-off behaviour
    // is today's exactly): --spec-oracle replaces the draft source with the expected continuation,
    // --spec-corrupt N flips every Nth assembled draft, --spec-follow makes the run emit a given
    // continuation. The files are parsed and refused here, before any device work.
    // Native verify receives the borrowed continuation before selecting and committing
    // its accepted prefix. Oracle/corrupt feed ordinary point-mass/flipped drafts.
    void set_spec_knobs(const SpecKnobRequest& request);
    const SpecKnobs& spec_knobs() const { return spec_knobs_; }
    const spec::DraftPolicy& adaptive_policy() const {return draft_policy_;}
    // Upstream's SAVE admission (core::session_save_reclaim, the --session-save-reclaim knob): release
    // only as much reconstructible host cache as a snapshot of `need` bytes actually requires, keeping
    // every pin and the checkpoint `selected` names. Reports what it released; the caller decides what
    // to do when the measured room is still short.
    PrefixCache::Reclaim reclaim_for_save(uint64_t need,std::span<const int32_t> selected);
private:
    uint32_t vocabulary_,context_;
    std::vector<int32_t> stop_ids_,fed_;
    std::vector<float> logits_;
    GenerationOperations operations_;
    GenerationOptions options_;
    spec::DraftPolicy draft_policy_; // fixed arrays, process-owned; snapshots exclude learning
    PrefixCache cache_;
    SpecKnobs spec_knobs_; // default: every knob off, behaviour identical to today
    DumpSinks* dumps_=nullptr; // non-owning; armed for the duration of run(), null otherwise
    int64_t dump_positions_=0; // upstream's dump_positions (:6077): prompt tokens - 1 + max_new
    uint64_t routing_written_=0; // this GEN only; shared-file slot work is counted separately
    bool valid_=true;
    bool control_enabled_=true;
    bool model_attached_=true,foreign_scope_=false,nonfinite_pending_=false;
    std::function<void()> suspended_live_;
    struct ScalarCandidates {
        std::array<VerifyPick,8> picks{};
        uint32_t count=0,retained=0;
        uint64_t follow_differ=0,follow_emitted=0;
        bool stopped=false;
    };
    ScalarCandidates scalar_candidates(std::span<const DraftRow>,const GenerationRequest&,uint32_t emitted,const std::function<bool()>&);
    void cooperate(const GenerationEvents&);
    // Upstream's #2228 diagnosis over the live state and the cache's checkpoints (generate.cpp:10951-10983).
    void report_picture_mismatch(std::span<const int32_t>,std::span<const RequestImage>,uint64_t,bool);
    void feed(int32_t,const std::function<bool()>&,bool diagnostics=true);
    void emit_routing_window(const RoutingWindowView&,bool window);
    void reset_sequence();
    bool cache_current(bool pinned=false,bool root=false,bool tail=false);
    // The park that just happened, for the serve line: bytes reused from the retained image and whether it
    // replaced a superseded copy of the same conversation (PrefixCache::store's #342 branch).
    const GenerationEvents* park_events_=nullptr;
    uint64_t parked_reused_=0;
    // Upstream's `--window-hashes` record set (see GenerationOptions::window_hashes). The rows of ONE window are
    // gathered while it is open and hashed when it closes, so a window that never completes cannot be written as if
    // it had. `window_round` is the RAII that makes every exit path - continue, break, throw - a close.
    struct WindowRound {
        Generation* own;
        explicit WindowRound(Generation* generation);
        WindowRound(const WindowRound&) = delete;
        WindowRound& operator=(const WindowRound&) = delete;
        ~WindowRound();
    };
    void begin_window();
    void close_window();
    bool window_open_ = false;
    uint64_t window_index_ = 0;
    int64_t window_position_ = 0;
    // A row is pushed by every feed() while a window is open, including the one a round feeds ahead of time
    // for the NEXT window; the window therefore consumes its rows from the FRONT of this queue.
    std::vector<float> residual_queue_;
    std::vector<int32_t> window_tokens_;
    WindowHashLog window_hashes_;
    uint32_t park_refused_stages_=0;
    // One entry per conversation whose prompt read gave way mid-way, holding what had been fed, its pictures
    // below that point, and whether that read had started at token 0 - upstream keeps the same marker on the
    // SLOT (BSlot::partial, BSlot::partial_from0, src/program/generate.cpp:9832, :11618-11619), because a
    // `--batch` server interleaves conversations and the marker of one must not be read as the marker of another.
    // Bounded by the prompt-cache slot count, the same bound as the conversations this backend can hold.
    struct InterruptedRead {std::vector<int32_t> tokens;std::vector<RequestImage> images;bool from0=false;};
    std::vector<InterruptedRead> interrupted_reads_;
    // The pictures of the sequence the model currently represents, in prompt order. The live state at fed_.size()
    // owns images_below(sequence_images_,fed_.size()); a request with other pictures at those cells cannot
    // continue it (upstream's live_imgs, generate.cpp:12795).
    std::vector<RequestImage> sequence_images_;
};
}
