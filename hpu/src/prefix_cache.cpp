#include "strata/hpu/prefix_cache.hpp"
#include "strata/hpu/allocation_budget.hpp"
#include "strata/core/conversation_memory.hpp"
// The retention law is called here, not transcribed: this is the same header the pinned upstream serve path
// calls (src/program/generate.cpp:8498), so the rule cannot drift from the code it is meant to match.
#include "strata/program/conv_cache.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>

namespace strata::hpu {
namespace {
bool equal(std::span<const int32_t> a,std::span<const int32_t> b) {return a.size()==b.size() && std::equal(a.begin(),a.end(),b.begin());}
}
PrefixCache::PrefixCache(uint32_t slots,uint64_t limit,MemoryProbe probe,uint64_t floor,uint32_t context,uint32_t vocabulary,int64_t min_tokens):slots_(slots),context_(context),vocabulary_(vocabulary),limit_(limit),memory_probe_(std::move(probe)),memory_floor_(floor),min_tokens_(min_tokens) {
    if(slots>kMaxGenerationCheckpoints || (slots && !limit) || bool(context)!=bool(vocabulary)) throw std::invalid_argument("invalid prefix cache limits");
    // Upstream bounds --conversation-cache-min-tokens to INT32_MAX at the parse (:2113-2118) and treats any
    // nonpositive value as "park everything" (conversation_cache.hpp:206); nothing above the bound reaches here.
    if(min_tokens_>int64_t(std::numeric_limits<int32_t>::max())) throw std::invalid_argument("invalid prefix cache limits");
    if(context) (void)generation_cache_metadata_bytes(context,vocabulary,slots);
    entries_.reserve(slots);
    if(entries_.capacity()!=slots) throw std::length_error("cache entry capacity exceeds admitted slots");
}
std::optional<size_t> PrefixCache::find(std::span<const int32_t> prompt,uint64_t maximum,bool control_vector,std::span<const RequestImage> images) {
    const auto best=peek(prompt,maximum,control_vector,images);
    if(best) {entries_[*best].age=++clock_;++stats_.hits;}
    return best;
}
std::optional<size_t> PrefixCache::peek(std::span<const int32_t> prompt,uint64_t maximum,bool control_vector,std::span<const RequestImage> images) const {
    std::optional<size_t> best;
    for(size_t i=0;i<entries_.size();++i) {
        const auto& ids=entries_[i].tokens;
        if(entries_[i].control_vector==control_vector && ids.size()<=prompt.size() && ids.size()<=maximum && (!best || ids.size()>entries_[*best].tokens.size()) &&
           std::equal(ids.begin(),ids.end(),prompt.begin()) && images_match(entries_[i].images,images,ids.size())) best=i;
    }
    return best;
}
const PrefixCache::Entry& PrefixCache::entry(size_t index) const {return entries_.at(index);}
void PrefixCache::visit_prefixes(std::span<const int32_t> prompt,bool control,const std::function<void(const Entry&)>& visit,std::span<const RequestImage> images) const {
    if(!visit) return;
    for(const auto& entry:entries_) if(entry.control_vector==control && entry.tokens.size()<=prompt.size() &&
        std::equal(entry.tokens.begin(),entry.tokens.end(),prompt.begin()) && images_match(entry.images,images,entry.tokens.size())) visit(entry);
}
void PrefixCache::visit_token_prefixes(std::span<const int32_t> prompt,bool control,const std::function<void(const Entry&)>& visit) const {
    if(!visit) return;
    for(const auto& entry:entries_) if(entry.control_vector==control && entry.tokens.size()<prompt.size() &&
        std::equal(entry.tokens.begin(),entry.tokens.end(),prompt.begin())) visit(entry);
}
void PrefixCache::release_prefixes(std::span<const int32_t> prompt,bool control,std::span<const RequestImage> images) {
    for(size_t i=0;i<entries_.size();) {
        const auto& entry=entries_[i];
        if(entry.control_vector==control && entry.tokens.size()<=prompt.size() && std::equal(entry.tokens.begin(),entry.tokens.end(),prompt.begin()) &&
           images_match(entry.images,images,entry.tokens.size())) (void)take(i);
        else ++i;
    }
}
// Upstream's take() (include/strata/core/conversation_cache.hpp:257-262): the conversation the serve path is
// about to RUN leaves the cache, and the image it parked is the one the model retains (retain(), :217-226) so
// the park at the end of this request can grow that very image instead of re-copying the whole K/V. Removing it
// from the accounting is not an eviction: upstream's `held` parameter exists precisely because the image is
// still alive, only no longer parked.
PrefixCache::Entry PrefixCache::take(size_t index) {
    auto entry=std::move(entries_.at(index));
    stats_.bytes-=entry.bytes;
    entries_.erase(entries_.begin()+std::ptrdiff_t(index));
    ++stats_.taken;
    return entry;
}
// One pinned prefix at a time, and a new pin replaces the old: that is upstream's own serve rule
// (src/program/generate.cpp:11740-11741, "One pinned prefix at a time: a new pin replaces the old; it needs
// room for the root, the pin and a rotating checkpoint"), so clearing the previous pin here is the ported
// behavior rather than a limitation of this API.
bool PrefixCache::pin(std::span<const int32_t> prefix,bool control_vector,std::span<const RequestImage> images) {
    for(size_t i=0;i<entries_.size();++i) if(entries_[i].control_vector==control_vector && equal(entries_[i].tokens,prefix) &&
        images_match(entries_[i].images,images,prefix.size())) {
        for(auto& entry:entries_) entry.pinned=false;
        entries_[i].pinned=true;entries_[i].tail=false;entries_[i].age=++clock_;return true;
    }
    return false;
}
bool PrefixCache::store(std::span<const int32_t> tokens,std::span<const float> logits,uint64_t payload,
    const std::function<std::function<void()>()>& producer,bool pinned,bool root,bool tail,bool control_vector,
    std::span<const RequestImage> images) {
    if(!slots_ || tokens.empty()) return false;
    // Upstream asks this before the park proceeds and reports it as a skip, not an error (:8227-8236): the same
    // non-error path the byte-budget and RAM-floor refusals below take. Checked here, in the one park entry, so no
    // caller can reach the cache without passing it.
    if(!wants(int64_t(tokens.size()))) {++stats_.skipped;++stats_.min_tokens_skipped;return false;}
    if(!payload || !producer || logits.empty()) throw std::invalid_argument("invalid prefix checkpoint producer/shape");
    if(context_ && (tokens.size()>context_ || logits.size()!=vocabulary_)) throw std::invalid_argument("cache checkpoint exceeds admitted metadata geometry");
    for(float value:logits) if(!std::isfinite(value)) throw std::invalid_argument("non-finite cached logits");
    for(auto token:tokens) if(token<0 || uint64_t(token)>=logits.size()) throw std::invalid_argument("invalid cached token");
    // The caller passes exactly the pictures below tokens.size() (upstream's imgs_below(req_imgs, L)).
    uint32_t previous=0;
    for(size_t i=0;i<images.size();++i) {
        if(uint64_t(images[i].start)>=tokens.size() || (i && images[i].start<=previous))
            throw std::invalid_argument("invalid cached image identity");
        previous=images[i].start;
    }
    for(auto& entry:entries_) if(entry.control_vector==control_vector && equal(entry.tokens,tokens) &&
        entry.images.size()==images.size() && std::equal(entry.images.begin(),entry.images.end(),images.begin())) {
        entry.age=++clock_;if(pinned) (void)pin(tokens,control_vector,images);
        if(root) {for(auto& other:entries_) other.root=false;entry.root=true;}
        if(!tail || root || pinned) entry.tail=false;
        return true;
    }
    const uint64_t overhead=uint64_t(tokens.size())*4+uint64_t(logits.size())*4+uint64_t(images.size())*sizeof(RequestImage)+512;
    if(payload>UINT64_MAX-overhead || payload+overhead>limit_) {++stats_.skipped;return false;}
    const auto cost=payload+overhead;
    // An unknown reading cannot establish that releasing anything will help.
    // Fail before eviction; known pressure below is handled by real releases
    // and fresh measurements, never by crediting hypothetical freed bytes.
    auto available=memory_probe_?memory_probe_():std::optional<uint64_t>{};
    if(memory_probe_ && !available) {
        ++stats_.skipped;++stats_.memory_skipped;return false;
    }
    std::vector<size_t> victims;uint64_t freed=0;
    // Only one disposable tail is retained. Complete replacement selection
    // before releasing old payloads; protected pin/root promotions are normal
    // entries and must not be erased by a later tail capture. Upstream keeps the same single live tail by
    // erasing the previous request's tail at creation (src/program/generate.cpp:8459-8466).
    auto taken=[&](size_t index){return std::find(victims.begin(),victims.end(),index)!=victims.end();};
    if(tail && !pinned && !root) for(size_t i=0;i<entries_.size();++i) if(entries_[i].tail) {
        victims.push_back(i);freed+=entries_[i].bytes;
    }
    // The slot budget runs the pinned upstream chain law itself, in the shape its caller runs it: push the
    // incoming checkpoint, then drop one victim at a time while the count exceeds --prompt-cache
    // (src/program/generate.cpp:8479-8501). The array order is the chain - this session's root first, which
    // is where upstream's positional index-0 protection lives, then the rest in insertion order, then the
    // incoming item last with a stamp newer than every other one, exactly like checks.push_back() after
    // c.used = ++check_clock. Upstream passes a null pin array unless some checkpoint is pinned
    // (generate.cpp:8490-8496), and the tail array is only ever set for --prompt-cache-tail captures.
    // A victim that names the incoming checkpoint is upstream dropping the checkpoint it just took; here
    // that is reported as a refused store, because nothing new was retained.
    const size_t incoming=entries_.size();
    std::vector<size_t> live;live.reserve(incoming+1);
    for(size_t i=0;i<incoming;++i) if(!taken(i) && entries_[i].root) live.push_back(i);
    for(size_t i=0;i<incoming;++i) if(!taken(i) && !entries_[i].root) live.push_back(i);
    live.push_back(incoming);
    bool over_cap=false;
    while(live.size()>slots_) {
        const std::unique_ptr<uint64_t[]> stamps(new uint64_t[live.size()]);
        const std::unique_ptr<bool[]> tails(new bool[live.size()]);
        const std::unique_ptr<bool[]> pins(new bool[live.size()]);
        bool any_pin=false;
        for(size_t k=0;k<live.size();++k) {
            if(live[k]==incoming) {
                stamps[k]=clock_+1;tails[k]=tail && !pinned && !root;pins[k]=pinned;
            } else {
                const auto& entry=entries_[live[k]];
                stamps[k]=entry.age;tails[k]=entry.tail;pins[k]=entry.pinned;
            }
            any_pin=any_pin||pins[k];
        }
        const size_t victim=program::conv_cache::eviction_victim(stamps.get(),live.size(),int64_t(slots_),
            tails.get(),any_pin?pins.get():nullptr);
        if(live[victim]==incoming) {over_cap=true;break;}
        victims.push_back(live[victim]);freed+=entries_[live[victim]].bytes;
        live.erase(live.begin()+std::ptrdiff_t(victim));
    }
    if(over_cap) {++stats_.skipped;return false;}
    // The byte budget is this backend's own constraint (a chain checkpoint is device-side upstream), so it
    // follows upstream's conversation-level rule instead, which releases the least recently active item and
    // refuses once only pinned ones remain (include/strata/core/conversation_cache.hpp:266-290).
    std::vector<size_t> order(entries_.size());std::iota(order.begin(),order.end(),0);
    std::sort(order.begin(),order.end(),[&](size_t a,size_t b){
        if(slots_>=2 && entries_[a].tail!=entries_[b].tail) return entries_[a].tail;
        return entries_[a].age<entries_[b].age;
    });
    // Upstream's make_room order (:266-268): when the budget is short, the retained K/V goes first because it
    // is pure reconstruction, and only then do parked conversations leave. Dropping it is reported so the caller
    // can see the reuse it cost.
    const uint64_t retained_before=retained();
    bool retained_changed=false;
    if(retained_before && stats_.bytes+retained_before-freed>limit_-cost && drop_retained_) {
        drop_retained_();++stats_.reuse_dropped;retained_changed=true;
    }
    for(auto index:order) {
        if(stats_.bytes+retained()-freed<=limit_-cost && entries_.size()-victims.size()<slots_) break;
        if(taken(index)) continue;
        if(entries_[index].pinned && !pinned) continue;
        if(entries_[index].root && slots_>=2 && !root) continue;
        victims.push_back(index);freed+=entries_[index].bytes;
    }
    if(stats_.bytes+retained()-freed>limit_-cost || entries_.size()-victims.size()>=slots_) {++stats_.skipped;return false;}
    // Complete victim selection precedes eviction. Old snapshots are released
    // before capture so the model's separate state budget cannot be exceeded.
    std::sort(victims.rbegin(),victims.rend());
    for(auto index:victims) evict(index);
    // Eviction can remove the last cache alias of the restored image. It
    // becomes an EXCLUSIVE retained charge only now, after victim selection.
    // Re-admit that ownership transition before capturing another payload.
    auto reconcile_retained=[&] {
        const auto available_bytes=limit_-cost;
        auto fits=[&]{return stats_.bytes<=available_bytes && retained()<=available_bytes-stats_.bytes;};
        if(!fits() && retained() && drop_retained_) {
            drop_retained_();++stats_.reuse_dropped;retained_changed=true;
        }
        return fits();
    };
    if(!reconcile_retained()) {++stats_.skipped;return false;}
    if(memory_probe_) {
        if(!victims.empty() || retained_changed) available=memory_probe_();
        // Cache payload ownership can outlive an entry, and small allocator
        // blocks need not be returned to the OS. Only a new process/cgroup
        // sample may admit the allocation. A lost reading stops reclamation.
        for(uint32_t removed=0;available && !core::conversation_memory_admit(available,cost,memory_floor_) && removed<slots_;++removed) {
            std::optional<size_t> victim;
            for(size_t i=0;i<entries_.size();++i) {
                const auto& candidate=entries_[i];
                if((candidate.pinned && !pinned) || (candidate.root && slots_>=2 && !root)) continue;
                if(!victim || (candidate.tail!=entries_[*victim].tail?candidate.tail:candidate.age<entries_[*victim].age)) victim=i;
            }
            if(!victim) break;
            evict(*victim);++stats_.memory_evicted;
            if(!reconcile_retained()) {++stats_.skipped;return false;}
            available=memory_probe_();
        }
        if(!core::conversation_memory_admit(available,cost,memory_floor_)) {
            ++stats_.skipped;++stats_.memory_skipped;return false;
        }
    }
    try {
        Entry entry;entry.tokens=std::vector<int32_t>(tokens.size());entry.logits=std::vector<float>(logits.size());
        entry.images=std::vector<RequestImage>(images.size());
        if(entry.tokens.capacity()!=tokens.size() || entry.logits.capacity()!=logits.size() || entry.images.capacity()!=images.size())
            throw std::length_error("cache STL capacity exceeds admitted extent");
        std::copy(tokens.begin(),tokens.end(),entry.tokens.begin());std::copy(logits.begin(),logits.end(),entry.logits.begin());
        std::copy(images.begin(),images.end(),entry.images.begin());
        entry.restore=producer();if(!entry.restore) throw std::logic_error("empty checkpoint restore callback");
        // Telemetry is not a reservation. Drop the newly owned payload if
        // pressure crossed the floor while it was being captured.
        if(memory_probe_ && !core::conversation_memory_admit(memory_probe_(),0,memory_floor_)) {
            ++stats_.skipped;++stats_.memory_skipped;return false;
        }
        entry.bytes=cost;entry.age=++clock_;entry.pinned=pinned;entry.root=root;entry.tail=tail && !root && !pinned;entry.control_vector=control_vector;
        entries_.push_back(std::move(entry));
        if(pinned) for(size_t i=0;i+1<entries_.size();++i) entries_[i].pinned=false;
        if(root) for(size_t i=0;i+1<entries_.size();++i) entries_[i].root=false;
        stats_.bytes+=cost;++stats_.saved;return true;
    } catch(const AllocationBudgetExceeded&) {++stats_.skipped;return false;}
    catch(const std::bad_alloc&) {++stats_.skipped;return false;}
}
void PrefixCache::set_retained(std::function<uint64_t()> bytes,std::function<void()> drop) {
    retained_bytes_=std::move(bytes);drop_retained_=std::move(drop);
}
void PrefixCache::clear() {
    entries_.clear();stats_.bytes=0;
    // Upstream's clear() drops the reuse slot with the parked conversations (conversation_cache.hpp:294-300).
    if(drop_retained_) drop_retained_();
}
PrefixCache::Reclaim PrefixCache::reclaim_for(uint64_t need,std::span<const int32_t> selected) {
    Reclaim out;
    if(!memory_probe_) return out;                  // upstream: unknown telemetry cannot justify an eviction
    auto room=memory_probe_();
    if(!room) return out;
    out.telemetry=true;
    auto admits=[&]{return !need || core::conversation_memory_admit(room,need,memory_floor_);};
    if(admits()) {out.enough=true;return out;}
    // Upstream's reclaim starts by dropping its page-dedup reuse pool, the cheapest reconstructible bytes there
    // are, and only then releases parked caches. The retained parked image is this backend's equivalent.
    const uint64_t held=retained();
    if(held && drop_retained_) {
        drop_retained_();++stats_.reuse_dropped;++out.released;out.bytes+=held;
        room=memory_probe_();
        if(!room) return out;                       // the release is real but unmeasured: stop, do not assume
        if(admits()) {out.enough=true;return out;}
    }
    while(true) {
        std::optional<size_t> victim;
        for(size_t i=0;i<entries_.size();++i) {
            const auto& candidate=entries_[i];
            if(candidate.pinned) continue;
            if(candidate.tokens.size()==selected.size()
                && std::equal(candidate.tokens.begin(),candidate.tokens.end(),selected.begin())) continue;
            if(!victim || candidate.age<entries_[*victim].age) victim=i;
        }
        if(!victim) return out;   // only protected caches remain: the save must fail rather than evict a pin
        out.bytes+=entries_[*victim].bytes;
        evict(*victim);++out.released;
        room=memory_probe_();     // freed capacity is measured again, never assumed
        if(!room) return out;
        if(admits()) {out.enough=true;return out;}
    }
}
// The cheapest reconstructible cache is the retained image, so a SAVE admission releases it before touching a
// parked conversation - upstream's session_save_reclaim starts by dropping its reuse pool.
void PrefixCache::evict(size_t index) {stats_.bytes-=entries_[index].bytes;entries_.erase(entries_.begin()+index);++stats_.evicted;}
PrefixCacheStats PrefixCache::stats() const {
    auto out=stats_;out.entries=entries_.size();
    // Upstream's bytes() is the parked conversations PLUS the retained K/V (conversation_cache.hpp:207), so the
    // reported total has to carry the image the state owner is still holding for the next park.
    out.retained=retained();out.bytes+=out.retained;return out;
}
uint64_t PrefixCache::metadata_vector_bytes() const {
    uint64_t total=entries_.capacity()*sizeof(Entry);
    for(const auto& entry:entries_) {
        total=cache_metadata_add(total,entry.tokens.capacity()*sizeof(int32_t));
        total=cache_metadata_add(total,entry.logits.capacity()*sizeof(float));
        total=cache_metadata_add(total,entry.images.capacity()*sizeof(RequestImage));
    }
    return total;
}
}
