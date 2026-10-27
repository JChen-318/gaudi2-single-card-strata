#include "strata/hpu/generation.hpp"
#include <algorithm>
#include <iostream>
#include <memory>

using namespace strata::hpu;
namespace {
void require(bool ok,const char* message) {if(!ok) throw std::runtime_error(message);}
std::vector<float> logits(std::span<const int32_t> ids) {
    uint32_t hash=0;for(auto id:ids) hash=(hash*7+uint32_t(id))%16;
    std::vector<float> result(16,-100);result[hash]=100;return result;
}
struct Harness {
    std::vector<int32_t> state,emitted;std::vector<size_t> captured,windows;
    bool inside=false;uint32_t partial=0;bool fail=false,omit=false,bad=false;
    GenerationOperations operations() {
        GenerationOperations op;op.reset=[&]{state.clear();};op.available_host_bytes=[] {return std::optional<uint64_t>(UINT64_MAX);};
        op.decode=[&](int32_t id,const auto& cancelled) {if(cancelled && cancelled()) throw Cancelled();state.push_back(id);return logits(state);};
        op.prefill=[&](std::span<const int32_t> ids,const PromptCommit& commit,const auto& cancelled) {
            struct Guard {bool& flag;explicit Guard(bool& x):flag(x){flag=true;}~Guard(){flag=false;}} guard(inside);
            windows.push_back(ids.size());
            const auto count=partial?std::min<size_t>(partial,ids.size()):ids.size();
            for(size_t i=0;i<count;++i) {if(cancelled && cancelled()) throw Cancelled();state.push_back(ids[i]);}
            commit(bad?uint32_t(ids.size()+1):uint32_t(count),logits(state));
            if(fail) throw std::runtime_error("injected prefill failure after committed prefix");
            if(partial && !omit) throw Cancelled();
        };
        op.checkpoint_bytes=100;op.checkpoint=[&] {
            require(!inside,"snapshot callback reentered locked model window");captured.push_back(state.size());
            return std::function<void()>([&,saved=state]{state=saved;});
        };return op;
    }
    GenerationEvents events() {GenerationEvents out;out.token=[&](int32_t id){emitted.push_back(id);};return out;}
};
GenerationOptions options() {
    GenerationOptions out;out.prefill_chunk=8;out.short_read=0;out.cache_every=10;out.cache_tail=true;
    out.cache_slots=8;out.cache_bytes=20000;out.boundaries.turn_token=-1;return out;
}
GenerationRequest request(size_t count) {GenerationRequest out;out.max_new=1;for(size_t i=0;i<count;++i) out.prompt.push_back(int32_t(i%16));return out;}
void chunks() {
    Harness h;const auto config=options();Generation engine(16,128,{},h.operations(),config);const auto first=request(30);
    const auto result=engine.run(first,h.events());
    require(result.generated==1 && h.state==first.prompt && h.windows==std::vector<size_t>{8,8,8,6},"prefill window changed prompt chunk geometry");
    require(h.captured==std::vector<size_t>{16,24,30},"periodic checkpoints split chunks or missed near-tail");
    const auto expected=logits(first.prompt);require(h.emitted.back()==std::max_element(expected.begin(),expected.end())-expected.begin(),"windowed prompt changed target token");
    auto branch=first;branch.prompt[26]=15;h.captured.clear();h.windows.clear();
    const auto changed=engine.run(branch,h.events());
    require(changed.reused==24 && h.windows==std::vector<size_t>{6} && h.state==branch.prompt,"near-tail did not restore complete branch prefix");
    Harness short_h;auto short_config=config;short_config.short_read=6;
    Generation short_engine(16,128,{},short_h.operations(),short_config);(void)short_engine.run(first,short_h.events());
    require(short_h.windows==std::vector<size_t>{8,8,8} && short_h.captured==std::vector<size_t>{16,30},"short-read suffix used prompt batch or parked a tail");
    Harness no_h;auto no_request=first;no_request.checkpoint=false;
    Generation no_engine(16,128,{},no_h.operations(),config);(void)no_engine.run(no_request,no_h.events());
    require(no_h.captured.empty(),"ckpt=0 captured window state");
}
void partial_and_contracts() {
    auto config=options();const auto input=request(20);
    Harness h;h.partial=3;Generation engine(16,128,{},h.operations(),config);
    const auto cancelled=engine.run(input,h.events());require(cancelled.finish=="cancelled" && h.state.size()==3 && h.captured==std::vector<size_t>{3},"partial prefill cancellation lost accepted prefix");
    h.partial=0;const auto resumed=engine.run(input,h.events());require(resumed.reused==3 && h.state==input.prompt,"partial prompt resumed from uncommitted state");
    Harness failed;failed.partial=3;failed.fail=true;Generation recover(16,128,{},failed.operations(),config);
    bool error=false;try {(void)recover.run(input,failed.events());} catch(const std::runtime_error&) {error=true;}
    require(error && failed.state.size()==3,"failed window lost accepted token transactions");
    failed.partial=0;failed.fail=false;require(recover.run(input,failed.events()).reused==3,"ordinary failed window could not resume accepted prefix");
    for(bool malformed:{false,true}) {
        Harness broken;broken.partial=3;broken.omit=true;broken.bad=malformed;
        Generation bad(16,128,{},broken.operations(),config);bool fatal=false;
        try {(void)bad.run(input,broken.events());} catch(const SessionFatal&) {fatal=true;}
        require(fatal && !bad.cache_stats().entries,"invalid/missing completion left a reusable window");
        broken.partial=0;broken.omit=false;broken.bad=false;
        require(!bad.run(input,broken.events()).reused && broken.state==input.prompt,"broken window contract reused mismatched model state");
    }
    Harness pinned;Generation pin_engine(16,128,{},pinned.operations(),config);auto pin=input;pin.pin=5;
    auto events=pinned.events();events.progress=[](uint32_t done,uint32_t,double) {if(done==5) throw Cancelled();};
    const auto stop=pin_engine.run(pin,events);require(stop.finish=="cancelled" && pinned.captured==std::vector<size_t>{5},"pin cut was not captured before progress cancellation");
}
void tail_replacement() {
    PrefixCache cache(8,20000);uint32_t active=0,peak=0;
    struct Payload {uint32_t& active;explicit Payload(uint32_t& x):active(x){++active;}~Payload(){--active;}};
    auto put=[&](int32_t token,bool tail) {
        return cache.store(std::vector<int32_t>{token},std::vector<float>(16),100,[&] {
            auto owned=std::make_shared<Payload>(active);peak=std::max(peak,active);return std::function<void()>([owned]{});
        },false,false,tail);
    };
    require(put(0,false) && put(1,true) && put(2,true),"tail capture failed");
    require(active==2 && peak==2 && !cache.find(std::vector<int32_t>{1}) && cache.find(std::vector<int32_t>{0}),"old disposable tail survived replacement with spare slots");
    require(cache.pin(std::vector<int32_t>{2}) && put(3,true) && cache.find(std::vector<int32_t>{2}),"pin promotion was still marked disposable tail");
    require(put(3,false) && put(4,true) && cache.find(std::vector<int32_t>{3}),"periodic promotion was erased by next tail");
    // Root promotion dominates a caller's tail hint, including an existing
    // entry. A later tail must not erase the shared root through that hint.
    require(cache.store(std::vector<int32_t>{4},std::vector<float>(16),100,[]{return std::function<void()>([]{});},false,true,true),"tail root promotion failed");
    require(put(5,true) && cache.find(std::vector<int32_t>{4}),"root promotion was still marked disposable tail");
}
}
int main() {
    try {chunks();partial_and_contracts();tail_replacement();std::cout<<"PASS actual generation prefill callbacks, unchanged chunk geometry, periodic/near-tail/short-read, branch logits, partial failure/cancel, contract guards, pin timing and bounded one-tail replacement\n";}
    catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
