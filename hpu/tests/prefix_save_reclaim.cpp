// Host test for PrefixCache::reclaim_for, this backend's adaptation of upstream
// core::session_save_reclaim (include/strata/core/session_save_reclaim.hpp:17-53), the SAVE admission
// upstream exposes as --session-save-reclaim. The rules it must keep are the ones the pinned template
// states: free only reconstructible host caches, re-measure after every release because freed capacity
// is not necessarily RAM the allocator returned, keep every explicit pin and exactly the checkpoint the
// save would select, and release NOTHING when there is no memory reading - unknown telemetry can never
// justify an eviction. Upstream's first step drops its page-dedup reuse pool, which this backend has no
// equivalent of (it stores full snapshots with restore closures), so the entries themselves are the only
// reclaimable caches here.
// Host only: no device, no measurement of real allocator behavior.
#include "strata/hpu/prefix_cache.hpp"
#include "strata/core/conversation_memory.hpp"
#include <algorithm>
#include <cstdint>
#include <functional>
#include <iostream>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>
using namespace strata::hpu;
namespace {
uint32_t cases=0;
void require(bool ok,const std::string& why){if(!ok) throw std::runtime_error(why);}
constexpr uint64_t payload=1000;
std::vector<float> logits(){std::vector<float> values(16,0.25f);return values;}   // token ids must stay inside the row
std::vector<int32_t> key(int32_t id){return std::vector<int32_t>{id,id+1};}
std::function<std::function<void()>()> producer_probe(){return []{return []{};};}
bool holds(const PrefixCache& cache,uint64_t slots,std::span<const int32_t> tokens) {
    for(uint64_t i=0;i<slots;++i) {
        const auto& entry=cache.entry(static_cast<size_t>(i));
        if(entry.tokens.size()==tokens.size()
            && std::equal(entry.tokens.begin(),entry.tokens.end(),tokens.begin())) return true;
    }
    return false;
}
struct Fixture {
    uint64_t room=0;
    uint32_t released=0;
    PrefixCache::MemoryProbe probe(){return [this]{return std::optional<uint64_t>(room);};}
};
void no_telemetry_refuses_to_evict() {
    PrefixCache cache(8,uint64_t(1)<<30);            // no probe at all
    for(int32_t id:{1,2,3}) require(cache.store(key(id),logits(),payload,producer_probe()),"store failed");
    const auto before=cache.stats();
    const auto out=cache.reclaim_for(uint64_t(1)<<40,key(1));
    require(!out.telemetry && !out.released && !out.bytes && !out.enough,
        "a reclaim with no memory reading must release nothing");
    require(before.bytes==cache.stats().bytes && before.entries==cache.stats().entries,
        "the cache changed without telemetry");
    ++cases;
}
void ample_room_releases_nothing() {
    const uint64_t need=uint64_t(1)<<20;
    Fixture room;room.room=UINT64_MAX/4;
    require(strata::core::conversation_memory_admit(std::optional<uint64_t>(room.room),need,0),
        "the fixture's premise is wrong: that reading does not admit the save");
    PrefixCache cache(8,uint64_t(1)<<30,room.probe());
    for(int32_t id:{1,2,3}) require(cache.store(key(id),logits(),payload,producer_probe()),"store failed");
    const auto out=cache.reclaim_for(need,key(1));
    require(out.telemetry && out.enough && !out.released && !out.bytes,
        "a save that already fits must not release a single cache");
    ++cases;
}
void protects_pins_and_the_selected_checkpoint() {
    const uint64_t need=uint64_t(1)<<40;
    Fixture room;room.room=UINT64_MAX/4;            // the caches must first be admissible at all
    PrefixCache cache(8,uint64_t(1)<<30,room.probe());
    require(cache.store(key(1),logits(),payload,producer_probe()),"oldest checkpoint must store");
    require(cache.store(key(2),logits(),payload,producer_probe(),true),"pinned checkpoint must store");
    require(cache.store(key(3),logits(),payload,producer_probe()),"selected checkpoint must store");
    require(cache.store(key(4),logits(),payload,producer_probe()),"newest releasable checkpoint must store");
    room.room=1;                                    // now the same reading is short of the save
    require(!strata::core::conversation_memory_admit(std::optional<uint64_t>(room.room),need,0),
        "the fixture's premise is wrong: that reading admits");
    const auto saved=cache.stats();
    const auto out=cache.reclaim_for(need,key(3));
    require(!out.enough,"a save that cannot fit must report that it is still short");
    require(out.released==2,"only the two releasable checkpoints may go, released "+std::to_string(out.released));
    require(out.bytes>0 && out.bytes<saved.bytes,"the reported bytes must be the released caches' own cost");
    require(holds(cache,cache.stats().entries,key(2)),"a pinned checkpoint was reclaimed");
    require(holds(cache,cache.stats().entries,key(3)),"the checkpoint the save would select was reclaimed");
    require(!holds(cache,cache.stats().entries,key(1)),"the oldest releasable checkpoint survived");
    require(cache.stats().entries==2 && cache.stats().bytes==saved.bytes-out.bytes,
        "the accounting did not follow the releases");
    ++cases;
}
void stops_as_soon_as_the_room_admits() {
    const uint64_t need=uint64_t(1)<<20;
    // A reading that rises with every measured release: upstream re-probes for exactly this reason,
    // because a released cache's bytes are only available once the allocator actually has them back.
    struct Rising {
        std::vector<uint64_t> readings;
        PrefixCache::MemoryProbe probe(){return [this]{
            const uint64_t value=1+readings.size()*uint64_t(512)*1024;
            readings.push_back(value);
            return std::optional<uint64_t>(value);
        };}
    } rising;
    rising.readings.push_back(UINT64_MAX/4);        // stores are admissible while the caches are built
    PrefixCache cache(8,uint64_t(1)<<30,rising.probe());
    for(int32_t id:{1,2,3,4,5}) require(cache.store(key(id),logits(),payload,producer_probe()),"store failed");
    rising.readings.clear();                         // the SAVE starts from a genuinely short reading
    const auto out=cache.reclaim_for(need,key(5));
    require(out.telemetry,"the probe reading was not recognised");
    require(out.released>=1,"a short save must release at least one cache");
    require(out.enough,"the reclaim must stop once the measured room admits the save");
    require(rising.readings.size()==size_t(out.released)+1,"the room was not re-measured exactly once per release");
    require(strata::core::conversation_memory_admit(std::optional<uint64_t>(rising.readings.back()),need,0),
        "the reclaim reported enough on a reading that does not admit the save");
    require(!strata::core::conversation_memory_admit(std::optional<uint64_t>(rising.readings[rising.readings.size()-2]),
        need,0),"the reclaim kept releasing after a measurement already admitted the save");
    require(holds(cache,cache.stats().entries,key(5)),"the selected checkpoint must survive a growing-room reclaim");
    require(out.released<5,"every releasable cache was dropped even though the room became sufficient");
    ++cases;
}
void selected_outside_the_cache_changes_nothing_extra() {
    const uint64_t need=uint64_t(1)<<40;
    Fixture room;room.room=UINT64_MAX/4;
    PrefixCache cache(8,uint64_t(1)<<30,room.probe());
    require(cache.store(key(7),logits(),payload,producer_probe()),"store failed");
    require(cache.store(key(8),logits(),payload,producer_probe()),"store failed");
    room.room=1;
    const auto out=cache.reclaim_for(need,key(9));   // nothing matches the selected prefix
    require(out.released==2 && !out.enough,"with no selected checkpoint present, every unpinned cache is releasable");
    require(cache.stats().entries==0 && cache.stats().bytes==0,"an emptied cache must report no bytes");
    ++cases;
}
}
int main(){try {
    no_telemetry_refuses_to_evict();
    ample_room_releases_nothing();
    protects_pins_and_the_selected_checkpoint();
    stops_as_soon_as_the_room_admits();
    selected_outside_the_cache_changes_nothing_extra();
    std::cout<<"PASS session-save reclaim: "<<cases<<" cases hold the pinned upstream contract - no telemetry releases nothing, "\
        "a save that already fits releases nothing, pins and the checkpoint the save would select always survive, releases go "\
        "oldest-first and stop as soon as the re-measured room admits, and the reported bytes match the released caches; no device\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
