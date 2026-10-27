#include "strata/hpu/generation.hpp"
#include "strata/hpu/protocol.hpp"
#include <algorithm>
#include <array>
#include <iostream>
using namespace strata::hpu;
namespace {
void require(bool ok,std::string why) {if(!ok) throw std::runtime_error(why);}
void selection() {
    PromptBoundaryOptions o;o.turn_token=9;o.root_min_tokens=3;
    const std::vector<int32_t> ids{9,0,1,2,9,3,4,9,8,5,9,6};
    auto b=prompt_boundaries(ids,0,o,true);require(b.root==4 && b.turn==10 && !b.message,"first system root/last turn index contract");
    o.tail_role_token=8;b=prompt_boundaries(ids,0,o,true);require(b.root==4 && b.turn==7,"effort-role turn not excluded from checkpoint");
    b=prompt_boundaries(ids,4,o,true);require(!b.root && b.turn==7,"resumed request created a root or selected a stale turn");
    b=prompt_boundaries(ids,0,o,false);require(b.root==4 && !b.turn && !b.message,"ckpt=0 must retain only eligible shared root");
    o.root_min_tokens=5;b=prompt_boundaries(ids,0,o,true);require(!b.root,"later turn incorrectly substituted for short first root");
    o.turn_token=-1;b=prompt_boundaries(ids,0,o,true);require(!b.root && !b.turn,"disabled turn detection produced a cut");
    o.turn_token=9;o.tail_role_token=-1;o.message_checkpoint=true;
    std::vector<int32_t> long_prompt(9100,0);long_prompt[0]=9;long_prompt[8500]=9;long_prompt[9000]=9;
    b=prompt_boundaries(long_prompt,0,o,true);require(b.root==8500 && b.turn==9000 && b.message==8500,"original message checkpoint helper not reused");
    b=prompt_boundaries(long_prompt,8500,o,true);require(!b.root && !b.message && b.turn==9000,"message checkpoint reused a boundary at/before resume");
}
void retention() {
    PrefixCache cache(3,5000);auto producer=[] {return std::function<void()>([]{});};
    auto put=[&](int32_t id,bool root=false,bool tail=false) {return cache.store(std::array<int32_t,1>{id},std::array<float,16>{},1000,producer,false,root,tail);};
    require(put(0,true) && put(1,false,true) && put(2),"root/tail/leaf initial capture");
    require(cache.find(std::array<int32_t,1>{1}).has_value() && put(3),"tail hit/new leaf");
    require(!cache.find(std::array<int32_t,1>{1}) && cache.find(std::array<int32_t,1>{0}) && cache.find(std::array<int32_t,1>{2}),"recent tail was not first victim or root was evicted");
    require(put(4,true),"root replacement failed");
    const auto old_root=cache.find(std::array<int32_t,1>{0});
    require((!old_root || !cache.entry(*old_root).root) && put(5),"new automatic root left old root protected");
    const auto new_root=cache.find(std::array<int32_t,1>{4});require(new_root && cache.entry(*new_root).root,"rotating leaf evicted current automatic root");
    PrefixCache single(1,2000);require(single.store(std::array<int32_t,1>{0},std::array<float,16>{},1000,producer,false,true),"single root capture");
    require(single.store(std::array<int32_t,1>{1},std::array<float,16>{},1000,producer),"one-slot root was pinned without room for a leaf");
}
void integration() {
    std::vector<int32_t> state;std::vector<size_t> captures;size_t reads=0;
    auto logits=[] {std::vector<float> values(16,-100);values[0]=100;return values;};
    GenerationOperations op;op.reset=[&]{state.clear();};op.available_host_bytes=[] {return std::optional<uint64_t>(UINT64_MAX);};
    // The tool's decode seam is what honours a cancellation (hpu/src/generation.cpp feed passes the predicate
    // through), so the fixture does the same: without it no prompt read here could give way mid-way.
    op.decode=[&](int32_t token,const auto& cancelled) {if(cancelled && cancelled()) throw Cancelled();state.push_back(token);++reads;return logits();};op.checkpoint_bytes=1000;
    op.checkpoint=[&] {captures.push_back(state.size());return std::function<void()>([&,saved=state]{state=saved;});};
    GenerationOptions options;options.cache_slots=3;options.cache_bytes=10000;options.cache_every=0;options.boundaries.turn_token=9;options.boundaries.tail_role_token=8;options.boundaries.root_min_tokens=3;
    Generation engine(16,64,{},op,options);GenerationEvents events;events.token=[](int32_t token){require(token==0,"boundary checkpoint changed sampled token");};
    auto request=parse_generation_request("GEN 1 9,0,1,2,9,3,4,9,8,5,9,6",16,64);
    (void)engine.run(request,events);require(captures==std::vector<size_t>({4,7,12}),"actual native generation did not capture root/effort-excluded turn/full prompt");
    request.prompt={9,0,1,2,9,3,4,9,8,2,9,6};const auto before=reads;const auto switched=engine.run(request,events);
    require(switched.reused==7 && reads-before==5,"changed effort tail reread stable conversation history");
    // Upstream's `resumed_from0` (src/program/generate.cpp:11044, gate :11727): a prompt read that gave way
    // BEFORE the shared root keeps that read's identity, so the retry still elects the root; a resume out of the
    // parked conversation is a different case and must not re-elect one the chain already carries.
    engine.reset();captures.clear();request.checkpoint=true;
    request.prompt={9,0,1,2,9,3,4,9,8,5,9,6};
    size_t fed=0;
    const auto interrupted=engine.run(request,events,[&]{return fed++>=2;});   // give way before position 4
    require(interrupted.finish=="cancelled","the fixture read did not give way mid-prompt");
    // A cancelled read parks what it reached (upstream keeps those tokens as the live conversation, see
    // pp_exact at generate.cpp:8395-8400), but it must not have elected the root it never got to.
    auto listed=[](const std::vector<size_t>& v) {std::string text;for(auto n:v) text+=std::to_string(n)+" ";return text;};
    require(captures.size()==1 && captures[0]>=1 && captures[0]<4,
            "the interrupted read parked "+listed(captures)+" instead of only its cancellation point (<4)");
    const auto continued=engine.run(request,events);
    require(continued.reused>0 && continued.reused<4,"the retry did not continue the interrupted read");
    require(std::find(captures.begin(),captures.end(),4)!=captures.end(),
            "the retry hid the shared root instead of electing it");
    // A second cancellation keeps the marker (upstream stores read_from==0 || resumed_from0 on the yielding slot).
    engine.reset();captures.clear();fed=0;
    (void)engine.run(request,events,[&]{return fed++>=1;});
    captures.clear();fed=0;
    (void)engine.run(request,events,[&]{return fed++>=1;});
    captures.clear();(void)engine.run(request,events);
    require(std::find(captures.begin(),captures.end(),4)!=captures.end(),
            "two interruptions lost the from-0 identity of the read");
    // Interleaving: the marker belongs to the read that set it. Interrupting A must not give a different
    // conversation B the root election, and running B in between must not cost A its from-0 identity.
    engine.reset();captures.clear();fed=0;
    (void)engine.run(request,events,[&]{return fed++>=2;});                     // A gives way after one token
    // B shares no prefix with A, so it is a different conversation rather than a continuation of it.
    const auto other=parse_generation_request("GEN 1 7,7,7,7,7,7,7,7,7,7",16,64);
    captures.clear();(void)engine.run(other,events);                            // B runs fully, from its own 0
    require(std::find(captures.begin(),captures.end(),4)==captures.end(),
            "a different conversation reached A's root position: "+listed(captures));
    captures.clear();(void)engine.run(request,events);                          // A comes back to its own read
    require(std::find(captures.begin(),captures.end(),4)!=captures.end(),
            "interleaving B lost A's from-0 identity: "+listed(captures));
    engine.reset();captures.clear();request.checkpoint=false;(void)engine.run(request,events);
    require(captures==std::vector<size_t>{4} && engine.cache_stats().entries==1,"ckpt=0 saved a one-shot turn or skipped shared root");
    request.prompt={9,0,1,2,9,6,5,9,7};const auto reused=engine.run(request,events);
    require(reused.reused==4 && engine.cache_stats().entries==1,"one-shot request did not reuse retained system root");
}
}
void stage_park_refusal() {
    // Upstream parks one image PER layer-split stage (src/program/generate.cpp:8244-8285) and refuses to park a
    // split that shares one GPU (:2382-2386). On a single card the stage count can therefore only be 0 or 1; a
    // caller presenting more must be refused by name, not quietly parked as one stage's image.
    std::vector<int32_t> state;std::vector<size_t> captures;std::vector<uint32_t> refused;uint32_t stages=0;
    auto logits=[] {std::vector<float> values(16,-100);values[0]=100;return values;};
    GenerationOperations op;op.reset=[&]{state.clear();};
    op.available_host_bytes=[] {return std::optional<uint64_t>(UINT64_MAX);};
    op.decode=[&](int32_t token,const auto&) {state.push_back(token);return logits();};
    op.checkpoint_bytes=1000;
    op.checkpoint=[&] {captures.push_back(state.size());return std::function<void()>([&,saved=state]{state=saved;});};
    op.parking_stages=[&]{return stages;};
    GenerationOptions options;options.cache_slots=2;options.cache_bytes=10000;options.cache_every=0;
    Generation engine(16,64,{},op,options);
    GenerationEvents events;events.token=[](int32_t) {};events.park_refused=[&](uint32_t n) {refused.push_back(n);};
    auto request=parse_generation_request("GEN 1 0,1,2,3",16,64);request.checkpoint=true;
    stages=0;(void)engine.run(request,events);
    require(!captures.empty() && refused.empty() && engine.park_refused_stages()==0,
            "an unsplitted session did not park");
    const auto parked_once=captures.size();
    engine.reset();captures.clear();refused.clear();
    stages=2;(void)engine.run(request,events);
    require(captures.empty() && !refused.empty() &&
            std::all_of(refused.begin(),refused.end(),[](uint32_t n){return n==2;}) &&
            engine.park_refused_stages()==2,
            "a two-stage session parked anyway instead of refusing and naming the stage count");
    engine.reset();captures.clear();refused.clear();
    stages=1;(void)engine.run(request,events);
    require(captures.size()==parked_once && refused.empty() && engine.park_refused_stages()==0,
            "a one-stage split is not the single image upstream's stage 0 parks");
}
int main() {
    try {selection();retention();integration();
    stage_park_refusal();std::cout<<"PASS prompt root/turn/effort/message selection, root and tail retention, one-slot fallback and actual ckpt=0/shared-root generation reuse\n";}
    catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
