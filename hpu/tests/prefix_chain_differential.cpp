// Differential audit of this backend's prefix retention against the pinned upstream policy.
//
// Upstream retains conversation checkpoints as a CHAIN - sorted by length, each a prefix of the next, a
// radix tree collapsed onto the one branch a single KV arena can hold - and its retention policy is the
// three overloads of strata::program/conv_cache.hpp (include/strata/program/conv_cache.hpp:35, :46, :61).
// This backend no longer has its own copy of that rule: hpu/src/prefix_cache.cpp includes the same header
// and calls it, so Part 1 below keeps an independent transcription and requires the header and the
// transcription to answer every hand-walked scenario the same way. Part 2 runs the law the way upstream's
// OWN caller builds it (src/program/generate.cpp:8479-8501: push the incoming checkpoint, then drop one
// victim at a time while the count exceeds --prompt-cache, with the stamps/tail/pin arrays rebuilt from the
// chain each round) and compares the live PrefixCache's choice with it case by case. Part 3 records what
// this backend's representation still cannot be: upstream's chain holds one KV arena's branch, this cache
// holds independent snapshots, so sibling branches coexist instead of overwriting each other.
// Host only: no device, no timing or hit-rate claim, and no assertion that this policy is optimal.
#include "strata/hpu/prefill_schedule.hpp"
#include "strata/hpu/prefix_cache.hpp"
#include "strata/program/conv_cache.hpp"
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <memory>
#include <sstream>
#include <span>
#include <string>
#include <vector>
using namespace strata::hpu;
namespace {
uint32_t policy_cases=0,live_cases=0;
std::vector<std::string> divergences;
// a std::vector<uint8_t> of 0/1 flags becomes the bool array the upstream header takes
// An empty flag vector must become a null pointer, which is what upstream passes when it has no tail or no
// pin at all (src/program/generate.cpp:8490-8496 resets pin_flags only when something is pinned). `new
// bool[0]` returns a non-null pointer to nothing, and the header then reads past it - a run of this audit
// picked up uninitialized bytes and answered a hand-walked scenario wrong.
std::unique_ptr<bool[]> bools(const std::vector<uint8_t>& values) {
    if(values.empty()) return nullptr;
    std::unique_ptr<bool[]> out(new bool[values.size()]);
    for(size_t i=0;i<values.size();++i) out[i]=values[i]!=0;
    return out;
}
void require(bool ok,const std::string& why){if(!ok) throw std::runtime_error(why);}
// ---- the transcription of the pinned upstream policy, kept as an independent oracle ----
size_t victim_plain(const uint64_t* stamps,size_t n,int64_t cap) {
    if(cap<2 || n<2) return 0;                       // no room to keep root and leaf apart: pin off
    size_t v=1;                                      // index 0 is the root and is pinned
    for(size_t i=2;i<n;++i) if(stamps[i]<stamps[v]) v=i;
    return v;
}
size_t victim_tail(const uint64_t* stamps,size_t n,int64_t cap,const bool* tail) {
    if(tail!=nullptr && n>=2 && cap>=2) {
        size_t newest=0;
        for(size_t i=1;i<n;++i) if(stamps[i]>stamps[newest]) newest=i;
        for(size_t i=1;i<n;++i) if(tail[i] && i!=newest) return i;   // never the newest, never the root
    }
    return victim_plain(stamps,n,cap);
}
size_t victim_pinned(const uint64_t* stamps,size_t n,int64_t cap,const bool* tail,const bool* pinned) {
    if(pinned==nullptr) return victim_tail(stamps,n,cap,tail);
    if(cap<2 || n<2) return 0;
    size_t newest=0;
    for(size_t i=1;i<n;++i) if(stamps[i]>stamps[newest]) newest=i;
    if(tail!=nullptr) for(size_t i=1;i<n;++i) if(tail[i] && !pinned[i] && i!=newest) return i;
    size_t v=n;                                      // least recently used among the unpinned non-root
    for(size_t i=1;i<n;++i) if(!pinned[i] && (v==n || stamps[i]<stamps[v])) v=i;
    if(v!=n) return v;
    v=1;                                             // all pinned: oldest pin, never the root
    for(size_t i=2;i<n;++i) if(stamps[i]<stamps[v]) v=i;
    return v;
}
std::string show(const std::vector<uint64_t>& stamps,const std::vector<uint8_t>& tail,const std::vector<uint8_t>& pinned,
                 int64_t cap,size_t victim) {
    std::ostringstream out;
    out<<"cap "<<cap<<" stamps [";
    for(auto stamp:stamps) out<<stamp<<' ';
    out<<"] tail [";
    for(auto value:tail) out<<(value?'T':'.')<<' ';
    out<<"] pins [";
    for(auto value:pinned) out<<(value?'P':'.')<<' ';
    out<<"] victim "<<victim;
    return out.str();
}
// ---- part 1: the header and the transcription must never disagree ----
void policy_audit() {
    struct Case { std::vector<uint64_t> stamps; std::vector<uint8_t> tail,pins; int64_t cap; size_t expected; const char* rule; };
    const std::vector<Case> cases{
        {{1,2,3,4},{},{},4,1,"least recent use among the non-root items"},
        {{4,3,2,1},{false,true,false,false},{},4,1,"the disposable tail leaves first"},
        {{5,4,3,9},{false,false,false,true},{},4,2,"the newest item is never the tail victim"},
        {{1,2,3,4},{},{},1,0,"a one-slot budget switches the pin off and the oldest leaves"},
        {{1,2,3,4},{},{},0,0,"a zero budget behaves the same way"},
        {{1,1,1,1},{},{},4,1,"equal stamps take the lowest non-root index"},
        {{9,7,5,3},{},{false,true,true,false},4,3,"several pins: the least recently used UNPINNED non-root item leaves"},
        {{9,8,7,6},{},{false,true,true,true},4,3,"all pinned: the backstop drops the oldest pin, never the root"},
        {{9,8,7,6},{false,true,false,false},{false,true,false,false},4,3,"a pinned tail is skipped by the tail rule, then LRU among the unpinned"},
        {{1}, {}, {},4,0,"a single item has no root/leaf distinction to protect"},
    };
    for(const auto& item:cases) {
        const size_t header=victim_pinned(item.stamps.data(),item.stamps.size(),item.cap,
            bools(item.tail).get(),bools(item.pins).get());
        require(header==item.expected,std::string("upstream rule \"")+item.rule+"\" answered "+
            std::to_string(header)+" but the hand-walked expectation is "+std::to_string(item.expected)+
            " | "+show(item.stamps,item.tail,item.pins,item.cap,header));
        // the transcription, and the header the port actually calls, are two independent readings
        const bool any_pin=std::any_of(item.pins.begin(),item.pins.end(),[](bool v){return v;});
        require(strata::program::conv_cache::eviction_victim(item.stamps.data(),item.stamps.size(),item.cap,
            bools(item.tail).get(),any_pin?bools(item.pins).get():nullptr)==header,
            std::string("the pinned header disagrees with the transcription for \"")+item.rule+"\"");
        require(victim_plain(item.stamps.data(),item.stamps.size(),item.cap)==
            victim_tail(item.stamps.data(),item.stamps.size(),item.cap,nullptr),
            "the no-tail overload disagrees with passing a null tail array");
        ++policy_cases;
    }
}
// ---- part 2: the live cache, compared the way upstream's caller builds its arrays ----
struct Live { size_t released,plain; std::vector<uint64_t> stamps; std::vector<uint8_t> tail,pins; int64_t cap; bool refused; };
// Fill a cache of exactly `n` slots, optionally mount through some of them so their stamps stop following
// insertion order (upstream advances c.used when a request mounts through a checkpoint,
// src/program/generate.cpp:11163-11165, and this cache ages an entry on a hit for the same reason), then add
// one more checkpoint. Upstream's eviction loop runs on the chain AFTER the push, so the comparison arrays
// are the retained entries plus the incoming one, with the incoming stamped newer than every stamp the cache
// holds (upstream sets c.used = ++check_clock before checks.push_back()). `sensitive` marks a case that must
// tell the tail-priority rule apart from plain least recent use: an earlier version of this audit claimed to
// test that rule with fixtures where both answers were the same item, and a mutation that deleted the tail
// array from the port passed it. A case that cannot change the answer proves nothing.
Live live_run(const std::string& name,const std::vector<uint8_t>& tails,const std::vector<uint8_t>& pins,
              const std::vector<uint8_t>& mounted,bool sensitive,bool incoming_tail=false,bool incoming_pin=false) {
    const uint32_t slots=uint32_t(tails.size());
    PrefixCache cache(slots,uint64_t(1)<<30);
    const std::vector<float> logits(16,0.5f);
    for(size_t i=0;i<tails.size();++i) {
        const std::vector<int32_t> tokens{int32_t(i)};
        require(cache.store(std::span<const int32_t>(tokens),std::span<const float>(logits),1024,
            []{return []{};},pins[i],i==0,tails[i]),"fixture checkpoint was not stored");
    }
    require(cache.stats().entries==slots,"the fixture did not fill its slot budget");
    for(auto index:mounted) {
        const std::vector<int32_t> tokens{int32_t(index)};
        require(bool(cache.find(std::span<const int32_t>(tokens))),"mounting the fixture checkpoint failed");
    }
    Live out;out.cap=int64_t(slots);
    uint64_t newest=0;
    std::vector<uint8_t> roots;
    for(uint32_t i=0;i<cache.stats().entries;++i) {
        const auto& entry=cache.entry(i);
        out.stamps.push_back(entry.age);out.tail.push_back(entry.tail);out.pins.push_back(entry.pinned);
        roots.push_back(uint8_t(entry.root));
        newest=std::max(newest,entry.age);
    }
    // the port feeds the law this session's root at index 0 and the rest in insertion order
    std::vector<size_t> order;
    for(size_t i=0;i<out.stamps.size();++i) if(roots[i]) order.push_back(i);
    for(size_t i=0;i<out.stamps.size();++i) if(!roots[i]) order.push_back(i);
    Live chain;chain.cap=out.cap;
    for(auto i:order) {chain.stamps.push_back(out.stamps[i]);chain.tail.push_back(out.tail[i]);chain.pins.push_back(out.pins[i]);}
    chain.stamps.push_back(newest+1);chain.tail.push_back(incoming_tail);chain.pins.push_back(incoming_pin);
    const bool any_pin=std::any_of(chain.pins.begin(),chain.pins.end(),[](uint8_t v){return v!=0;});
    const size_t victim=victim_pinned(chain.stamps.data(),chain.stamps.size(),chain.cap,
        bools(chain.tail).get(),any_pin?bools(chain.pins).get():nullptr);
    const size_t plain=victim_plain(chain.stamps.data(),chain.stamps.size(),chain.cap);
    if(sensitive) require(plain!=victim,name+": it releases item "+std::to_string(plain)+" with or without the "
        "tail-priority rule, so it cannot be testing that rule | "+show(chain.stamps,chain.tail,chain.pins,chain.cap,victim));
    out.plain=plain;
    out.refused=victim==chain.stamps.size()-1;
    out.stamps=chain.stamps;out.tail=chain.tail;out.pins=chain.pins;
    out.released=out.refused?slots:victim;   // `slots` means "the incoming item", no retained entry left
    const std::vector<int32_t> extra{int32_t(15)};
    const bool stored=cache.store(std::span<const int32_t>(extra),std::span<const float>(logits),1024,
        []{return []{};},incoming_pin,false,incoming_tail);
    require(stored!=out.refused,name+": the cache admitted or refused a checkpoint the upstream law decided "
        "the other way round | "+show(chain.stamps,chain.tail,chain.pins,chain.cap,victim)+" stored="+std::to_string(int(stored)));
    if(out.refused) {
        require(cache.stats().entries==slots,name+": a refused store changed the retained set");
        return out;
    }
    require(cache.stats().entries==slots,name+": the cache did not release exactly one checkpoint");
    for(size_t i=0;i<slots;++i) {
        bool present=false;
        for(uint32_t probe=0;probe<cache.stats().entries && !present;++probe) {
            const auto& entry=cache.entry(probe);
            present=entry.tokens.size()==1 && entry.tokens[0]==int32_t(i);
        }
        if(!present) out.released=i;
    }
    // The identity of the released item is the whole point: an earlier version of this rewrite only checked
    // that ONE checkpoint left, so a port that evicted the wrong one still passed.
    require(out.released==victim,name+": the cache released fixture item "+std::to_string(out.released)+
        " where the upstream law names "+std::to_string(victim)+" | "+show(chain.stamps,chain.tail,chain.pins,chain.cap,victim));
    bool incoming_held=false;
    for(uint32_t probe=0;probe<cache.stats().entries && !incoming_held;++probe)
        incoming_held=cache.entry(probe).tokens==std::vector<int32_t>(extra);
    // Upstream's law never names the checkpoint it just pushed (it is the newest, and the all-pinned
    // backstop scans from the oldest), so a store that succeeded must have kept it.
    require(incoming_held,name+": the store succeeded yet dropped the checkpoint it just took");
    require(out.released<slots,name+": no fixture checkpoint was released");
    return out;
}
void live_case(const std::string& name,std::vector<uint8_t> tails,std::vector<uint8_t> pins,const std::string& expectation,
               std::vector<uint8_t> mounted={},bool sensitive=false,bool incoming_tail=false,bool incoming_pin=false) {
    const auto live=live_run(name,tails,pins,mounted,sensitive,incoming_tail,incoming_pin);
    if(!live.refused) {
        require(expectation=="agree",name+": the cache released fixture item "+std::to_string(live.released)+
            " where the upstream law, fed the chain exactly as its own caller feeds it, names the incoming "
            "checkpoint instead | "+show(live.stamps,live.tail,live.pins,live.cap,live.released));
        ++live_cases;
        return;
    }
    require(expectation=="refuse",name+": the upstream law drops the incoming checkpoint here, so a store that "
        "was refused means the port diverged | "+show(live.stamps,live.tail,live.pins,live.cap,live.stamps.size()-1));
    ++live_cases;
}
void live_audit() {
    live_case("plain LRU, four slots",{false,false,false,false},{false,false,false,false},"agree");
    // The disposable tail leaves before a checkpoint that plain least recent use would name: the fixture
    // mounts through the tail so the two rules disagree, and the cache has to follow the tail rule.
    live_case("tail beats plain LRU",{false,false,true,false},{false,false,false,false},"agree",{2},true);
    // The tail that is also the newest checkpoint is the one just saved, so upstream spares it
    // (conv_cache.hpp:47-52) and the cache must still hold it afterwards.
    live_case("a new tail never evicts itself",{false,false,false,false},{false,false,false,false},"agree",{},false,true);
    // Root plus one pin fill a two-slot chain: the only item the law can name is the incoming checkpoint,
    // so upstream drops that checkpoint and this cache refuses the store rather than evicting a protected
    // prefix. This is the case the previous version of this audit could not express.
    live_case("root and pin leave no room",{false,false},{false,true},"refuse");
    // A one-slot chain has no room to keep root and leaf apart: the pin switches off and the oldest leaves
    // (conv_cache.hpp:23-24, :36). The cache evicts its pinned first item instead of refusing.
    live_case("one slot switches the pin off",{false},{true},"agree");
    // pin=N replaces the previous pin rather than adding a second one, which is upstream's own serve rule
    // (src/program/generate.cpp:11740-11741), not a limit this API invented.
    PrefixCache cache(4,uint64_t(1)<<30);
    std::vector<std::vector<int32_t>> sets{{1,2,3},{1,2,3,4,5}};
    const std::vector<float> logits(16,0.1f);
    for(size_t i=0;i<sets.size();++i)
        require(cache.store(sets[i],logits,1024,[]{return []{};}),"pin replacement checkpoint must store");
    require(cache.pin(std::span<const int32_t>(sets[0])),"the first pin was refused");
    require(cache.pin(std::span<const int32_t>(sets[1])),"the second pin was refused");
    uint32_t held=0;bool first_pin=false;
    for(uint32_t i=0;i<cache.stats().entries;++i) {
        const auto& entry=cache.entry(i);
        held+=uint32_t(entry.pinned);
        first_pin=first_pin||(entry.pinned && entry.tokens==std::vector<int32_t>(sets[0]));
    }
    require(held==1 && !first_pin,"pin=N must replace the previous pinned prefix, upstream's rule, held="+std::to_string(held));
    ++live_cases;
}
// ---- part 3: the tail bookkeeping the retention law reads ----
// Upstream keeps exactly one live tail by length (tail_ckpt_len): creating a tail erases the previous
// request's tail (src/program/generate.cpp:8459-8466), and a NON-tail checkpoint landing on the length that
// was the tail clears the flag (:8443-8446 and :8467-8468), because a periodic checkpoint sitting there is a
// normal chain item now. The schedule side of the same rule is that a chunk which is both periodic and
// near-tail is stored as a normal checkpoint (`checkpoint_at(done,nullptr,tail_checkpoint &&
// !periodic_checkpoint)`, :8564), so it cannot be the item that evicts itself. This part drives the cache and
// the pinned schedule against those three rules.
void tail_bookkeeping() {
    const std::vector<float> logits(16,0.5f);
    PrefixCache cache(4,uint64_t(1)<<30);
    auto store=[&](uint32_t id,bool tail){
        const std::vector<int32_t> tokens{int32_t(id)};
        require(cache.store(std::span<const int32_t>(tokens),std::span<const float>(logits),1024,
            []{return []{};},false,false,tail),"tail fixture checkpoint must store");
    };
    auto locate=[&](uint32_t id){const std::vector<int32_t> tokens{int32_t(id)};
        return cache.find(std::span<const int32_t>(tokens));};
    auto is_tail=[&](uint32_t id){const auto at=locate(id);return bool(at)&&cache.entry(*at).tail;};
    store(1,true);
    require(is_tail(1),"the first tail was not flagged");
    store(2,true);   // a second tail: upstream erases the previous request's tail with it
    require(!locate(1) && is_tail(2),"creating a tail left the previous tail alive (generate.cpp:8459-8466)");
    // a NON-tail checkpoint landing on the length that was the tail: the item stays, the flag goes
    store(2,false);
    require(locate(2) && !is_tail(2),"a normal checkpoint at the tail's length did not retire the tail flag "
        "(generate.cpp:8443-8446, :8467-8468)");
    ++live_cases;
    // the schedule side: a chunk that is both periodic and near-tail is reported by both rules, and the port
    // stores it as a normal checkpoint (checkpoint_at(..., tail && !periodic), generate.cpp:8564), so the
    // item that is meant to survive cannot be the disposable one
    PromptCheckpointSchedule schedule(0,40,10,true,2);
    const auto first=schedule.complete(10,10);
    require(first.periodic && !first.tail,"a full chunk away from the tail was not periodic-only");
    const auto second=schedule.complete(20,10);
    require(second.periodic && !second.tail,"the second periodic boundary also claimed the tail");
    const auto overlap=schedule.complete(30,10);   // ten tokens left: both rules fire on this chunk
    require(overlap.periodic && overlap.tail,"the near-tail overlap stopped reporting both rules");
    // the checkpoint's token id is unrelated to the chunk position; the cache bounds ids by the logits row
    store(3,overlap.tail && !overlap.periodic);
    const auto over=locate(3);
    require(over && !cache.entry(*over).tail,"the port stored a periodic/tail overlap as a disposable tail");
    require(is_tail(2)==false,"storing a normal checkpoint resurrected an older tail flag");
    ++live_cases;
}
// ---- part 3: what the representation still cannot be ----
void structural_gaps() {
    PrefixCache cache(4,uint64_t(1)<<30);
    std::vector<std::vector<int32_t>> sets{{1,2,3},{1,2,3,4,5},{1,2,6,6,6}};
    const std::vector<float> logits(16,0.1f);
    for(size_t i=0;i<sets.size();++i)
        require(cache.store(sets[i],logits,1024,[]{return []{};}),"checkpoint must store");
    const auto longest=cache.find(sets[1]);
    require(bool(longest) && cache.entry(*longest).tokens.size()==5,"find did not answer with the longest matching prefix");
    const auto branch=cache.find(sets[2]);
    require(bool(branch) && cache.entry(*branch).tokens.size()==5,"a sibling branch lost its own longest prefix");
    divergences.push_back("retained entries are not prefix-ordered: upstream's chain makes index 0 the "
        "deepest prefix every request shares and drops any checkpoint that is not on the current prompt's "
        "path (src/program/generate.cpp:11136-11138), because one KV arena holds one branch; this backend "
        "stores independent full snapshots with restore closures, so sibling branches coexist. The eviction "
        "LAW is the upstream header, but which items it may choose from is a wider set here.");
    divergences.push_back("the byte budget has no upstream chain equivalent: upstream caps the chain by "
        "item count only and prices bytes at the conversation level (include/strata/core/conversation_cache"
        ".hpp:266-290), while this cache shares one byte limit across arithmetic modes. Slot pressure runs "
        "the upstream law; byte pressure releases the least recently active unprotected item and refuses "
        "once only pinned or rooted ones remain, which is that conversation-level rule.");
    ++live_cases;
}
}
int main(){try {
    policy_audit();
    live_audit();
    tail_bookkeeping();
    structural_gaps();
    std::cout<<"PASS prefix retention differential: "+std::to_string(policy_cases)+
        " scenarios where the pinned header and an independent transcription agree, "+std::to_string(live_cases)+
        " live cases compared against the law as upstream's own caller feeds it, "+std::to_string(divergences.size())+
        " divergences declared by name:\n";
    for(const auto& line:divergences) std::cout<<"  - "+line<<"\n";
    std::cout<<"no device\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
