#pragma once
#include "strata/hpu/protocol.hpp"
#include <cstdint>
#include <functional>

namespace strata::hpu {
// Token histories are fixed-capacity. Images themselves are charged by the
// Model snapshot budget; this is the additional per-slot host metadata only.
inline uint64_t slot_session_host_bytes(uint32_t slots,uint32_t context) {
    if(!slots) return 0;
    if(slots>64 || !context) throw std::invalid_argument("invalid slot memory geometry");
    // 4 KiB inline metadata plus up to 64 small owned-restore callables. Model
    // image payloads remain in the separately admitted snapshot budget.
    return uint64_t(slots)*(uint64_t(context)*sizeof(int32_t)+8192);
}
struct SlotSessionOperations {
    std::function<void()> before_foreign;
    std::function<std::optional<GenerationPrefix>(std::span<const int32_t>,uint64_t,bool,std::span<const RequestImage>)> select_prefix;
    std::function<std::vector<float>()> logits;
    std::function<void(const GenerationRead&)> import_prefixes;
    std::function<void()> reset;
    // The request's M-RoPE table and prompt image rows, installed before any Model work. A text request
    // restores the identity table, exactly upstream's non-GENI branch (generate.cpp:10746-10747); a GENI
    // request without the hook is refused instead of reading the token embedding table for image cells.
    std::function<void(const GenerationRequest&)> set_vision;
    std::function<std::vector<float>(std::span<const int32_t>)> prefill;
    std::function<std::vector<float>(int32_t)> decode;
    // Same production owned-restore seam as Generation. Model snapshots bind
    // the control mode, residual, mixed, logits and optional MTP state too.
    std::function<std::function<void()>()> checkpoint;
    std::function<void(bool)> control_vector;
    std::function<void()> discard_retained;
    // Optional layer-major window. Without it, semantic checkpoint cuts still
    // use prefill(), whose owner provides the scalar/reference implementation.
    std::function<std::vector<float>(std::span<const int32_t>)> prefill_window;
    // Synchronous committed-prefix reporting for mid-window STOP. Metadata
    // callbacks do not reenter Model; a throwing producer rolls back its own
    // unfinished chunk and leaves the last reported prefix authoritative.
    std::function<void(std::span<const int32_t>,bool,const PromptCommit&,const std::function<bool()>&)> prefill_cancellable;
    // Proposal plus one native two-row target pass; nullopt leaves the Model
    // unchanged and requests scalar fallback. Only called with two outputs
    // admitted. The returned retained count excludes known pending.
    std::function<std::optional<SlotCandidates>(int32_t,std::span<const int32_t>,const strata::kernels::SamplerParams&)> speculate;
};
struct SlotCheckpointOptions {
    PromptBoundaryOptions boundaries;
    uint32_t interval=16384,prefill_chunk=0,prefill_until=0,short_read=1; // see PrefillChunkOptions: a long tail is 4.7x slower per row
    uint32_t checkpoints=6;
    bool tail=false;
    uint32_t canonical_prefill_rows=0; // opt-in backend arithmetic unit; zero preserves the existing schedule
};
// Single-card time slicing, not row batching. Idle end-state reuse follows
// upstream BSlot.ids/cvec/cached and slot_source (generate.cpp:9823-9850,
// 10075-10089, 10856-10862, 10932-10948). A normal BSlot also keeps the deepest
// eligible prompt checkpoint (:12973-12982), including its image when an
// end-state is no longer a prefix of re-rendered history. BYIELD parks a partial
// prefix and its bounded checkpoint chain. Speculative slots use one known input
// plus one proposal; this is still time slicing, not multi-slot row batching.
class SlotSession final:public BatchSlotSession {
public:
    SlotSession(uint32_t slots,uint32_t vocabulary,uint32_t context,std::vector<int32_t> stops,
                SlotSessionOperations,bool cache_idle,SlotCheckpointOptions={});
    Admission admit(int,const GenerationRequest&,uint32_t,const std::function<void(int32_t)>&) override;
    Admission admit_cooperative(int,const GenerationRequest&,uint32_t,const std::function<void(int32_t)>&,const ReadControl&) override;
    // Upstream only lets an idle slot serve a request without pictures, and a slot checkpoint must then also
    // carry none of its own (generate.cpp:10935-10948, `req_imgs.empty()` then `starts_with(c.ids, c.imgs)`).
    std::optional<GenerationPrefix> select_prefix(std::span<const int32_t>,uint64_t maximum,bool control_vector,
                                                  std::span<const RequestImage> images={});
    bool park_partial(int,const GenerationRead&) override;
    Step step(int) override;
    Window step_window(int,uint32_t limit) override;
    void stop(int) override;
    void release(int) override; // error/non-finite/failed admission: never cached
    void finish(int) override;  // ordinary stop/length/cancel: reusable if enabled
    void invalidate_caches() override;
    uint64_t reused_tokens() const {return reused_tokens_;}
    uint64_t idle_evictions() const {return idle_evictions_;}
    uint64_t checkpoint_captures() const {return checkpoint_captures_;}
    uint64_t checkpoint_evictions() const {return checkpoint_evictions_;}
    uint64_t yielded_reads() const {return yielded_reads_;}
    uint64_t interleaved_windows() const {return interleaved_windows_;}
private:
    struct PromptPoint {std::function<void()> image;uint64_t age=0;uint32_t cells=0;bool pinned=false,tail=false,root=false;};
    struct Slot {
        std::vector<int32_t> fed; // only tokens actually consumed by Model
        // Pictures of this history below fed.size(). Every checkpoint's pictures are exactly the slot's
        // pictures below its cells (upstream ConvCheckpoint::imgs, generate.cpp:8449); storing one list per
        // slot instead of a vector per point keeps Slot within its 4 KiB inline-metadata bound.
        std::vector<RequestImage> images;
        strata::kernels::SamplerParams sampling;
        std::function<void()> image;
        std::array<PromptPoint,64> checks{};
        uint32_t check_count=0; // prefixes of fed; no duplicate token histories
        int32_t pending=0; // emitted, not consumed until the next window
        bool active=false,cached=false,control=true,partial=false,from0=false,tainted=false;
    };
    std::vector<Slot> slots_;
    uint32_t vocabulary_,context_;
    std::vector<int32_t> stops_;
    SlotSessionOperations op_;
    bool cache_idle_;
    SlotCheckpointOptions options_;
    uint64_t reused_tokens_=0,idle_evictions_=0;
    uint64_t checkpoint_captures_=0,checkpoint_evictions_=0;
    uint64_t clock_=0,yielded_reads_=0,interleaved_windows_=0;
    Slot& at(int);
    std::function<void()> park(bool required=true);
    void erase_point(Slot&,uint32_t);
    void capture_prompt(Slot&,uint32_t,bool pinned,bool tail,bool root=false,std::span<const RequestImage> images={});
    GenerationRead read_view(Slot&) const;
    Slot collect_points(const GenerationRead&);
    struct ReadResult {std::vector<float> logits;uint32_t yielded=0;};
    ReadResult read_prompt(int,Slot&,const GenerationRequest&,uint32_t,const ReadControl&);
    bool is_stop(int32_t) const;
};
}
