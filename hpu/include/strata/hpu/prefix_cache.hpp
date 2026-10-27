#pragma once
#include <cstdint>
#include <functional>
#include <optional>
#include <span>
#include <vector>
#include "strata/hpu/cache_metadata.hpp"
#include "strata/hpu/image_key.hpp"

namespace strata::hpu {
struct PrefixCacheStats {uint64_t bytes=0,entries=0,saved=0,evicted=0,skipped=0,hits=0,memory_skipped=0,memory_evicted=0,superseded=0,taken=0,reuse_dropped=0,retained=0,min_tokens_skipped=0;};
// Serialized host ownership of complete, immutable model checkpoints. Both
// slots and bytes are shared across arithmetic modes; at most one prefix is
// pinned. Identical token prefixes in different modes are distinct, and so are
// identical tokens whose pictures differ: a checkpoint carries the pictures
// below its length (upstream ConvCheckpoint::imgs, generate.cpp:8449) and only a
// request whose pictures below that length are the same list may resume it
// (:10856-10861). Producer costs must include its retained state.
// Tokens/logits/images/entry overhead are added here.
class PrefixCache {
public:
    using MemoryProbe=std::function<std::optional<uint64_t>()>;
    struct Entry {
        std::vector<int32_t> tokens;
        std::vector<float> logits;
        std::vector<RequestImage> images; // pictures with start < tokens.size()
        std::function<void()> restore;
        uint64_t bytes=0,age=0;
        bool pinned=false,root=false,tail=false;
        bool control_vector=true;
    };
    // Adaptation of upstream core::session_save_reclaim (include/strata/core/session_save_reclaim.hpp:17-53),
    // the SAVE admission upstream exposes as --session-save-reclaim. Before a snapshot is written, free
    // reconstructible HOST caches only - never live device state - and re-probe after every release,
    // because capacity freed in the allocator is not necessarily RAM returned to the OS. Exactly like
    // upstream, every explicit pin and the checkpoint the save itself would select survive, and with no
    // memory reading available nothing is released at all: unknown telemetry cannot justify an eviction.
    struct Reclaim { uint64_t bytes=0; uint32_t released=0; bool enough=false,telemetry=false; };
    PrefixCache(uint32_t slots,uint64_t byte_limit,MemoryProbe={},uint64_t memory_floor=0,uint32_t context=0,
                uint32_t vocabulary=0,int64_t min_tokens=0);
    // Upstream's --conversation-cache-min-tokens test (include/strata/core/conversation_cache.hpp:206): every parked
    // conversation costs a whole slot whatever its length, so a client that interleaves short side requests with one
    // long conversation would spend the cache on the side requests and evict the long one (oldest first) - the reuse
    // it wanted. 0 parks everything, which is upstream's default (generate.cpp:639) and this port's.
    bool wants(int64_t tokens) const { return min_tokens_ <= 0 || tokens >= min_tokens_; }
    std::optional<size_t> find(std::span<const int32_t> prompt,uint64_t maximum=UINT64_MAX,bool control_vector=true,
                               std::span<const RequestImage> images={});
    std::optional<size_t> peek(std::span<const int32_t>,uint64_t maximum,bool control_vector,
                               std::span<const RequestImage> images={}) const;
    void visit_prefixes(std::span<const int32_t>,bool,const std::function<void(const Entry&)>&,
                        std::span<const RequestImage> images={}) const;
    // Every checkpoint in this control mode whose tokens are a proper prefix of `prompt`, regardless of
    // pictures: upstream's #2228 diagnosis walks each check's tokens first and only then compares pictures
    // (src/program/generate.cpp:10956-10974), so a picture-only mismatch must still be visible here.
    void visit_token_prefixes(std::span<const int32_t>,bool,const std::function<void(const Entry&)>&) const;
    void release_prefixes(std::span<const int32_t>,bool,std::span<const RequestImage> images={});
    const Entry& entry(size_t) const;
    // Upstream's take(): the parked conversation this request is about to run, moved out of the cache so the
    // model owns its image alone and can grow it when it re-parks at the end of the request.
    Entry take(size_t);
    bool pin(std::span<const int32_t> prefix,bool control_vector=true,std::span<const RequestImage> images={});
    // `images` are the pictures below tokens.size() at capture; two states with the same tokens but different
    // pictures are different conversations and are stored separately.
    bool store(std::span<const int32_t> tokens,std::span<const float> logits,uint64_t payload_cost,
        const std::function<std::function<void()>()>& producer,bool pinned=false,bool root=false,bool tail=false,
        bool control_vector=true,std::span<const RequestImage> images={});
    Reclaim reclaim_for(uint64_t need,std::span<const int32_t> selected);
    // Upstream counts the retained K/V in bytes() (include/strata/core/conversation_cache.hpp:207) and releases
    // it BEFORE evicting a parked conversation when an incoming park does not fit (:266-268) - it is the most
    // reconstructible thing in the budget, and it is also the first cache core::session_save_reclaim frees.
    // The buffers belong to the state owner, so the cache only asks for the number and tells the owner to let go.
    void set_retained(std::function<uint64_t()> bytes,std::function<void()> drop);
    void clear();
    PrefixCacheStats stats() const;
    uint64_t metadata_vector_bytes() const;
private:
    uint32_t slots_;
    uint32_t context_=0,vocabulary_=0;
    uint64_t limit_,clock_=0;
    MemoryProbe memory_probe_;
    std::function<uint64_t()> retained_bytes_;
    std::function<void()> drop_retained_;
    uint64_t retained() const {return retained_bytes_?retained_bytes_():0;}
    uint64_t memory_floor_=0;
    int64_t min_tokens_=0;
    PrefixCacheStats stats_;
    std::vector<Entry> entries_;
    void evict(size_t);
};
}
