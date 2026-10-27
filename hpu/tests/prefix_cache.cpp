#include "strata/hpu/generation.hpp"
#include "strata/hpu/protocol.hpp"
#include "strata/hpu/allocation_budget.hpp"
#include <algorithm>
#include <array>
#include <iostream>
#include <memory>

using namespace strata::hpu;
namespace {
void require(bool ok,const char* message) {if(!ok) throw std::runtime_error(message);}
std::vector<float> next(std::span<const int32_t> tokens) {
    uint32_t hash=0;for(auto id:tokens) hash=(hash*3+uint32_t(id))%16;
    std::vector<float> out(16,-100);out[hash]=100;return out;
}
void policy() {
    PrefixCache cache(3,5000);uint32_t active=0,peak=0;int32_t restored=-1;
    struct Payload {uint32_t& active;explicit Payload(uint32_t& n):active(n){++active;}~Payload(){--active;}};
    auto put=[&](int32_t id,bool pin=false) {
        const std::array<int32_t,1> token{id};const std::array<float,8> values{};
        return cache.store(token,values,1000,[&,id] {
            auto owned=std::make_shared<Payload>(active);peak=std::max(peak,active);
            return std::function<void()>([&,owned=std::move(owned),id]{restored=id;});
        },pin);
    };
    require(put(0,true) && put(1) && put(2),"initial cache capture");
    const std::array<int32_t,2> prompt{1,7};const auto hit=cache.find(prompt);require(hit.has_value(),"prefix match missing");cache.entry(*hit).restore();
    require(restored==1 && put(3) && !cache.find(std::array<int32_t,1>{2}) && cache.find(std::array<int32_t,1>{0}),"LRU evicted recent or pinned prefix");
    require(peak==3 && active==3 && cache.stats().bytes<=5000,"capture overlapped victim state or exceeded budget");
    const auto prior=cache.stats();
    require(!cache.store(std::array<int32_t,1>{4},std::array<float,8>{},5000,[]{return std::function<void()>([]{});}),"oversized capture accepted");
    require(cache.stats().entries==prior.entries && cache.stats().evicted==prior.evicted,"oversized capture evicted existing state");
    require(!cache.store(std::array<int32_t,1>{4},std::array<float,8>{},1000,[]()->std::function<void()>{throw AllocationBudgetExceeded();}),"model budget rejection ignored");
    require(active==2 && put(5,true),"failed producer leaked checkpoint ownership");
    require(cache.find(std::array<int32_t,1>{5}) && cache.stats().skipped==2,"new pinned prefix not published");
    cache.clear();require(active==0 && !cache.stats().entries && !cache.stats().bytes,"cache clear leaked ownership");
    PrefixCache single(1,2000);
    auto producer=[] {return std::function<void()>([]{});};
    require(single.store(std::array<int32_t,1>{0},std::array<float,8>{},1000,producer,true),"single pinned capture");
    // A one-slot chain has no room to keep root and leaf apart, so upstream switches the pin off and the
    // OLDEST checkpoint leaves (include/strata/program/conv_cache.hpp:23-24 and :36; the serve loop erases
    // exactly the victim that law names, src/program/generate.cpp:8479-8501). This cache follows it instead
    // of refusing: with one slot, a pinned prefix rotates out rather than blocking every new checkpoint.
    require(single.store(std::array<int32_t,1>{1},std::array<float,8>{},1000,producer)
        && single.stats().evicted==1 && single.stats().entries==1,"one-slot capture did not rotate the oldest checkpoint out");
    require(!single.find(std::array<int32_t,1>{0}) && single.find(std::array<int32_t,1>{1}),"one-slot rotation kept the wrong checkpoint");
}
void generation() {
    std::vector<int32_t> state,emitted;uint32_t decoded=0,restores=0;bool fail_restore=false,deny_capture=false;
    GenerationOperations op;op.reset=[&]{state.clear();};
    op.available_host_bytes=[] {return std::optional<uint64_t>(UINT64_MAX);};
    op.decode=[&](int32_t token,const auto& cancelled) {if(cancelled && cancelled()) throw Cancelled();++decoded;state.push_back(token);return next(state);};
    op.checkpoint_bytes=1000;
    op.checkpoint=[&] {
        if(deny_capture) throw AllocationBudgetExceeded();
        return std::function<void()>([&,saved=state] {
            if(fail_restore) {state=saved;state.push_back(15);throw std::runtime_error("injected partial apply");}
            state=saved;++restores;
        });
    };
    GenerationOptions options;options.cache_slots=8;options.cache_bytes=20000;Generation engine(16,64,{},op,options);
    GenerationEvents events;events.token=[&](int32_t token){emitted.push_back(token);};
    auto a=parse_generation_request("GEN 3 pin=2 0,1,2,3,4",16,64);const auto first=engine.run(a,events);const auto reference=emitted;
    require(!first.reused && engine.cache_stats().entries==3,"pin/root/turn checkpoint path missing");
    auto b=parse_generation_request("GEN 2 pin=2 0,1,8,9",16,64);emitted.clear();const auto switched=engine.run(b,events);
    require(switched.reused==2 && restores==1,"shared prefix did not restore at pin boundary");
    const auto before=decoded;emitted.clear();const auto again=engine.run(a,events);
    require(again.reused==5 && decoded-before==2 && emitted==reference && restores==2,"conversation root restore changed logits/decoded cached prompt again");
    auto once=parse_generation_request("GEN 1 ckpt=0 15,14,13",16,64);(void)engine.run(once,events);
    const auto prior=engine.cache_stats().saved;const auto not_live=engine.run(once,events);
    require(!not_live.reused && engine.cache_stats().saved==prior,"ckpt=0 parked or reused one-shot state");
    deny_capture=true;auto other=parse_generation_request("GEN 1 12,11,10",16,64);const auto limited=engine.run(other,events);
    require(limited.generated==1 && engine.cache_stats().skipped>0,"optional snapshot OOM failed ordinary generation");deny_capture=false;
    fail_restore=true;const auto calls=decoded;bool fatal=false;try {(void)engine.run(a,events);} catch(const SessionFatal&) {fatal=true;}
    require(fatal && decoded==calls && !engine.cache_stats().entries,"failed cache apply continued or retained stale checkpoints");
    fail_restore=false;engine.reset();(void)engine.run(a,events);engine.release_checkpoints();
    require(engine.cache_stats().entries==0,"parked checkpoints retained before disk staging");
    const auto same=engine.run(a,events);require(same.reused==0,"longer live sequence treated as exact shorter prefix");
    const auto tokens=state;const auto values=next(state);engine.adopt(tokens,values);require(!engine.cache_stats().entries,"disk adopt retained old bank");
    for(const auto* text:{"GEN 1 ckpt=2 0,1","GEN 1 pin=-1 0,1","GEN 1 pin=2 0,1,2","GEN 1 pin=4 0,1,2"}) {
        bool caught=false;try {(void)parse_generation_request(text,16,64);} catch(const std::invalid_argument&) {caught=true;}require(caught,"invalid cache request accepted");
    }
}
void pictures() {
    PrefixCache cache(6,20000);
    auto producer=[]()->std::function<void()>{return []{};};
    const auto picture=[](uint32_t start,uint64_t key){return std::vector<RequestImage>{RequestImage{start,key}};};
    const std::array<int32_t,4> tokens{0,1,2,3};
    const std::vector<RequestImage> both{RequestImage{1,7},RequestImage{2,99}};
    require(cache.store(tokens,std::array<float,4>{},100,producer,false,false,false,true,both),"picture capture refused");
    // Upstream matches only when the pictures below the stored length are the same list (generate.cpp:10856-10861);
    // an image-free request never matches a state that contains pictures.
    require(!cache.find(tokens,UINT64_MAX,true),"image-free request resumed a checkpoint with pictures");
    require(!cache.find(tokens,UINT64_MAX,true,picture(1,7)),"request missing a picture below the checkpoint matched");
    require(!cache.find(tokens,UINT64_MAX,true,picture(1,11)),"same start with another picture key matched");
    require(!cache.find(tokens,UINT64_MAX,true,picture(3,7)),"same key at another start matched");
    require(cache.find(tokens,UINT64_MAX,true,both).has_value(),"the exact picture list missed");
    // A picture at or after the stored prefix is not part of that checkpoint: the same state serves a longer
    // prompt when its pictures below the prefix still agree.
    const std::array<int32_t,2> short_tokens{0,1};
    const std::vector<RequestImage> short_pictures{RequestImage{1,7}};
    const std::vector<RequestImage> longer_pictures{RequestImage{1,7},RequestImage{3,42}};
    require(cache.store(short_tokens,std::array<float,4>{},100,producer,false,false,false,true,short_pictures),"later-picture capture refused");
    const std::array<int32_t,4> longer{0,1,2,3};
    require(cache.find(longer,2,true,longer_pictures).has_value(),"picture after the reused prefix changed the match");
    // A picture starting exactly at the prefix length is not below it and belongs to neither side.
    require(cache.find(longer,2,true,both).has_value(),"a picture starting at the checkpoint length changed the match");
    // Identical tokens with different pictures are different conversations, even inside one cache.
    PrefixCache split(4,20000);
    const std::vector<RequestImage> first{picture(1,7)},second{picture(1,11)};
    require(split.store(short_tokens,std::array<float,4>{},100,producer,false,false,false,true,first) &&
            split.store(short_tokens,std::array<float,4>{},100,producer,false,false,false,true,second) &&
            split.stats().entries==2,"identical tokens with different pictures were aliased");
    const auto at_first=split.find(short_tokens,UINT64_MAX,true,first),at_second=split.find(short_tokens,UINT64_MAX,true,second);
    require(at_first && at_second && *at_first!=*at_second && split.entry(*at_first).images==first &&
            split.entry(*at_second).images==second,"picture-specific selection returned the wrong checkpoint");
    require(!split.peek(short_tokens,UINT64_MAX,true,picture(1,13)) && split.peek(short_tokens,UINT64_MAX,true,first),"peek ignored picture identity");
    require(!split.pin(short_tokens,true,picture(1,13)) && split.pin(short_tokens,true,first),"pin ignored picture identity");
    uint32_t visited=0;split.visit_prefixes(short_tokens,true,[&](const PrefixCache::Entry&){++visited;},first);
    require(visited==1,"visit_prefixes ignored picture identity");
    split.release_prefixes(short_tokens,true,picture(1,13));
    require(split.find(short_tokens,UINT64_MAX,true,first) && split.find(short_tokens,UINT64_MAX,true,second),"release_prefixes with another picture released the entry");
    split.release_prefixes(short_tokens,true,first);
    require(!split.find(short_tokens,UINT64_MAX,true,first) && split.find(short_tokens,UINT64_MAX,true,second),"release_prefixes released the wrong picture");
    bool caught=false;
    try {(void)cache.store(tokens,std::array<float,4>{},100,producer,false,false,false,true,picture(4,1));} catch(const std::invalid_argument&) {caught=true;}
    require(caught,"picture start outside the checkpoint accepted");caught=false;
    try {(void)cache.store(tokens,std::array<float,4>{},100,producer,false,false,false,true,
        std::vector<RequestImage>{RequestImage{2,1},RequestImage{2,2}});} catch(const std::invalid_argument&) {caught=true;}
    require(caught,"unsorted picture starts accepted");
}
void generation_pictures() {
    std::vector<int32_t> state;uint32_t decoded=0;
    GenerationOperations op;op.reset=[&]{state.clear();};
    op.available_host_bytes=[] {return std::optional<uint64_t>(UINT64_MAX);};
    op.decode=[&](int32_t token,const auto& cancelled) {if(cancelled && cancelled()) throw Cancelled();++decoded;state.push_back(token);return next(state);};
    op.checkpoint_bytes=1000;
    op.checkpoint=[&]{return std::function<void()>([&,saved=state]{state=saved;});};
    GenerationOptions options;options.cache_slots=8;options.cache_bytes=20000;Generation engine(16,64,{},op,options);
    GenerationEvents events;events.token=[&](int32_t){};
    auto request=parse_generation_request("GEN 2 pin=2 0,1,2,3,4",16,64);request.images={RequestImage{1,7}};
    (void)engine.run(request,events);
    require(engine.cache_stats().entries>=2,"picture generation captured no checkpoint");
    require(engine.run(request,events).reused==5,"the same pictures did not resume the prompt checkpoint");
    // The pin checkpoint at length 2 is a same-token shared prefix with a different continuation.
    auto shared=parse_generation_request("GEN 1 0,1,8,9",16,64);shared.images={RequestImage{1,7}};
    require(engine.run(shared,events).reused==2,"same-token shared prefix with the same picture did not resume");
    shared.images={RequestImage{1,11}};
    require(engine.run(shared,events).reused==0,"same-token shared prefix with another picture resumed the checkpoint");
    auto other=request;other.images={RequestImage{1,11}};
    require(engine.run(other,events).reused==0,"same tokens with another picture resumed the pin checkpoint");
    auto text=request;text.images.clear();
    require(engine.run(text,events).reused==0,"image-free request resumed a picture checkpoint");
}
void arithmetic_modes() {
    PrefixCache cache(2,4000);uint32_t captures=0,active=0,peak=0;bool restored=false;
    struct Payload {uint32_t& active;explicit Payload(uint32_t& n):active(n){++active;}~Payload(){--active;}};
    auto put=[&](int32_t id,bool mode,bool pin=false) {
        std::array<float,16> values{};values[mode?1:2]=100;
        return cache.store(std::array<int32_t,1>{id},values,1000,[&,mode] {
            ++captures;auto payload=std::make_shared<Payload>(active);peak=std::max(peak,active);
            return std::function<void()>([&,payload=std::move(payload),mode]{restored=mode;});
        },pin,false,false,mode);
    };
    const std::array<int32_t,2> prompt{1,3};
    require(put(1,true) && put(1,false) && captures==2 && active==2,"identical tokens in two arithmetic modes aliased");
    const auto on=cache.find(prompt,UINT64_MAX,true),off=cache.find(prompt,UINT64_MAX,false);
    require(on && off && *on!=*off && cache.entry(*on).logits[1]==100 && cache.entry(*off).logits[2]==100,"mode-specific selection returned wrong logits");
    cache.entry(*on).restore();require(restored,"steered snapshot closure changed");cache.entry(*off).restore();require(!restored,"unsteered snapshot closure changed");
    require(put(1,true) && captures==2 && cache.stats().entries==2,"duplicate capture replaced another arithmetic mode");
    require(cache.pin(std::array<int32_t,1>{1},false) && !cache.entry(*on).pinned && cache.entry(*off).pinned,"pin matched the wrong arithmetic mode");
    require(!cache.pin(std::array<int32_t,1>{2},true),"mode pin miss");
    // The retention law cannot place a third checkpoint here: the oldest item holds the positional root
    // index upstream protects and the other is pinned, so the one item the law can name is the incoming
    // checkpoint - which is upstream dropping the checkpoint it just took (src/program/generate.cpp:8479-8501
    // with include/strata/program/conv_cache.hpp:70-77). This backend reports that as a refused store and
    // leaves the retained pair untouched.
    require(!put(2,true),"a capture the retention law cannot place was published anyway");
    require(cache.find(prompt,UINT64_MAX,true) && cache.find(prompt,UINT64_MAX,false),"a refused capture disturbed the retained per-mode snapshots");
    require(cache.stats().skipped==1,"the refused capture was not counted as skipped");
    // A pinned new leaf still lands, by the law's own backstop among the pinned items, and it displaces the
    // unpinned snapshot rather than the pinned one.
    require(put(3,true,true) && active==2 && peak==2 && cache.stats().bytes<=4000,"arithmetic banks did not share slots/bytes/release-before-capture");
    require(bool(cache.find(prompt,UINT64_MAX,true)) && !cache.find(prompt,UINT64_MAX,false),"the backstop took the wrong per-mode snapshot");
    cache.clear();require(!active,"clearing both arithmetic modes leaked snapshot payloads");
}
void intervals() {
    std::vector<int32_t> state,emitted;std::vector<size_t> captured;size_t reads=0;bool fail_capture=false;
    GenerationOperations op;op.reset=[&]{state.clear();};
    op.available_host_bytes=[] {return std::optional<uint64_t>(UINT64_MAX);};
    op.decode=[&](int32_t token,const auto& cancelled) {if(cancelled && cancelled()) throw Cancelled();state.push_back(token);++reads;return next(state);};
    op.checkpoint_bytes=1000;op.checkpoint=[&] {
        if(fail_capture) throw AllocationBudgetExceeded();
        captured.push_back(state.size());return std::function<void()>([&,saved=state]{state=saved;});
    };
    GenerationOptions options;options.cache_slots=8;options.cache_bytes=20000;options.cache_every=3;
    Generation engine(16,64,{},op,options);GenerationEvents events;events.token=[&](int32_t token){emitted.push_back(token);};
    auto first=parse_generation_request("GEN 1 0,1,2,3,4,5,6,7,8,9,10,11",16,64);
    const auto result=engine.run(first,events);
    require(captured==std::vector<size_t>{3,6,9,12} && !result.reused,"periodic prompt checkpoints not placed at fresh token intervals");
    auto branch=parse_generation_request("GEN 1 0,1,2,3,4,5,6,7,15,14,13,12",16,64);
    captured.clear();const auto before=reads;const auto changed=engine.run(branch,events);
    require(changed.reused==6 && reads-before==6 && captured==std::vector<size_t>{9,12},"periodic checkpoint did not resume at deepest exact branch prefix");
    // An adopted/live prefix shifts the interval origin; it is not rounded to
    // the absolute token position. This matches upstream resume + interval.
    engine.reset();state={0,1,2,3};engine.adopt(state,next(state));captured.clear();
    const auto continued=engine.run(first,events);
    require(continued.reused==4 && captured==std::vector<size_t>{7,10,12},"periodic interval lost resume-relative origin");
    engine.reset();captured.clear();auto no_checkpoint=first;no_checkpoint.checkpoint=false;
    (void)engine.run(no_checkpoint,events);require(captured.empty(),"ckpt=0 created intermediate prompt state");
    // Optional capture failure leaves live generation usable. Cancellation
    // retains only the committed prefix and can later reuse its saved state.
    engine.reset();captured.clear();fail_capture=true;require(engine.run(first,events).generated==1,"periodic snapshot OOM aborted generation");fail_capture=false;
    engine.reset();captured.clear();events.progress=[&](uint32_t done,uint32_t,double){if(done==7) throw Cancelled();};
    const auto cancelled=engine.run(first,events);
    require(cancelled.finish=="cancelled" && state.size()==7 && captured==std::vector<size_t>{3,6,7},"cancelled prompt lost committed periodic state");
    events.progress={};options.cache_every=0;Generation disabled(16,64,{},op,options);captured.clear();disabled.reset();
    (void)disabled.run(first,events);require(captured==std::vector<size_t>{12},"disabled periodic cache still captured intermediate states");
    options.cache_min_free=100;op.available_host_bytes=[] {return std::optional<uint64_t>{};};
    Generation pressured(16,64,{},op,options);captured.clear();pressured.reset();
    require(pressured.run(first,events).generated==1 && captured.empty() && pressured.cache_stats().memory_skipped>0,"unknown process memory did not skip optional generation checkpoint");
}
void memory_floor() {
    std::optional<uint64_t> available;bool drop_after_capture=false;uint32_t probes=0,captures=0,active=0;
    struct Payload {uint32_t& active;explicit Payload(uint32_t& n):active(n){++active;}~Payload(){--active;}};
    PrefixCache cache(2,5000,[&] {++probes;return available;},100);
    auto put=[&](int32_t id) {
        return cache.store(std::array<int32_t,1>{id},std::array<float,8>{},100,[&] {
            ++captures;auto payload=std::make_shared<Payload>(active);
            if(drop_after_capture) available=99;
            return std::function<void()>([payload=std::move(payload)]{});
        },false,id==0);
    };
    require(!put(0) && captures==0,"unknown telemetry captured or evicted state");
    available=0;require(!put(0) && captures==0,"measured zero confused with missing memory guard");
    // Independent charge: 100 payload + 4 token + 8*4 logits + 512 entry.
    available=747;require(!put(0) && captures==0,"checkpoint crossed allocation plus RAM floor");
    available=748;require(put(0) && captures==1 && active==1,"exact allocation plus floor rejected");
    const auto saved=cache.stats();available=99;require(!put(1) && cache.stats().evicted==saved.evicted && active==1,"low RAM evicted a protected root before refusal");
    available=748;drop_after_capture=true;require(!put(1) && captures==2 && active==1,"post-capture pressure published or leaked new state");
    require(cache.stats().entries==1 && cache.stats().memory_skipped==5 && probes==8,"RAM refusal accounting or pre/post probe contract");
    cache.clear();require(!active,"RAM guard retained rejected payload ownership");
}
void reclaim_memory() {
    std::optional<uint64_t> available=UINT64_MAX;uint32_t active=0,captures=0;bool reclaim=false,drop_after_capture=false,lose_reading=false;
    std::vector<int32_t> released;
    struct Payload {
        uint32_t& active;std::optional<uint64_t>& available;bool& reclaim;bool& lose;std::vector<int32_t>& released;int32_t id;
        Payload(uint32_t& n,std::optional<uint64_t>& a,bool& r,bool& l,std::vector<int32_t>& out,int32_t i):active(n),available(a),reclaim(r),lose(l),released(out),id(i){++active;}
        ~Payload(){--active;if(reclaim) {released.push_back(id);if(lose) available.reset();else if(available) *available+=324;}}
    };
    PrefixCache cache(5,10000,[&] {return available;},100);
    auto put=[&](int32_t id,bool pin=false,bool root=false,bool tail=false) {
        return cache.store(std::array<int32_t,1>{id},std::array<float,8>{},100,[&,id] {
            ++captures;auto payload=std::make_shared<Payload>(active,available,reclaim,lose_reading,released,id);
            if(reclaim) {require(available && *available>=748,"producer ran before measured RAM admission");*available-=648;}
            if(drop_after_capture) available=99;
            return std::function<void()>([payload=std::move(payload)]{});
        },pin,root,tail);
    };
    require(put(0,true) && put(1,false,true) && put(2) && put(3,false,false,true),"RAM reclaim initial capture");
    reclaim=true;available=100;
    require(put(4) && released==std::vector<int32_t>({3,2}) && active==3 && captures==5,"measured reclaim did not release tail then LRU before capture");
    require(cache.stats().memory_evicted==2 && !cache.stats().memory_skipped && cache.stats().entries==3,"measured reclaim stats or shared budget");
    const auto prior=cache.stats();available.reset();require(!put(5) && cache.stats().evicted==prior.evicted && captures==5,"unknown reading destroyed old snapshots or captured new state");
    available=0;require(!put(5) && active==2 && captures==5 && cache.stats().memory_evicted==3,"RAM admission credited estimated bytes or evicted protected root/pin");
    require(cache.find(std::array<int32_t,1>{0}) && cache.find(std::array<int32_t,1>{1}),"RAM reclaim discarded protected states");
    available=748;drop_after_capture=true;require(!put(6) && active==2 && captures==6 && cache.stats().entries==2,"post-capture pressure published or leaked reclaimed payload");
    drop_after_capture=false;available=748;require(put(7),"reclaim did not recover after optional refusal");
    available=0;lose_reading=true;const auto before=captures;
    require(!put(6) && captures==before && active==2 && cache.stats().memory_evicted==4,"lost telemetry after eviction continued reclaim/capture");
    reclaim=false;cache.clear();require(!active,"RAM reclaim ownership leaked");
}
}
int main() {
    try {policy();generation();pictures();generation_pictures();arithmetic_modes();intervals();memory_floor();reclaim_memory();std::cout<<"PASS bounded prefix LRU, pin protection/replacement, complete victims, release-before-capture, producer OOM and ownership\n"
        "PASS picture identity: exact list below the prefix, distinct identities per key/start, later pictures ignored, find/peek/pin/visit/release filtering and invalid starts\n"
        "PASS generation picture reuse: same pictures resume, another key or an image-free prompt never resumes a picture checkpoint\n"
        "PASS shared-budget arithmetic mode banks, identical-token distinct snapshots/logits, mode-specific pin and joint clear\n"
        "PASS prompt interval snapshots, branch reuse, resume-relative origin, ckpt=0, cancellation and snapshot OOM\n"
        "PASS process memory admission, unknown/zero telemetry, exact allocation+floor, pressure after capture and rejected payload release\n"
        "PASS bounded RAM reclaim, remeasure after each release, tail/LRU order, protected roots/pins and lost telemetry\n"
        "PASS actual generation conversation/shared-prefix restores, exact logits, ckpt=0, disk adoption and fatal failed apply\n";}
    catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
}
