#include "strata/hpu/protocol.hpp"
#include <algorithm>
#include <iostream>
using namespace strata::hpu;
namespace {
void require(bool ok,const char* why) {if(!ok) throw std::runtime_error(why);}
}
int main() {
    try {
        std::vector<int32_t> state,emitted;bool on=true,fail=false;uint32_t reads=0,switches=0,restores=0;
        auto logits=[&] {std::vector<float> result(16,-100);result[on?1:2]=100;return result;};
        GenerationOperations op;op.reset=[&]{state.clear();};op.available_host_bytes=[] {return std::optional<uint64_t>(UINT64_MAX);};
        op.decode=[&](int32_t id,const auto&) {state.push_back(id);++reads;return logits();};
        op.control_vector=[&](bool value) {++switches;on=value;if(fail) throw std::runtime_error("injected device mode failure");};
        op.checkpoint_bytes=1000;op.checkpoint=[&] {
            return std::function<void()>([&,saved=state,mode=on] {require(on==mode,"cross-mode cached state restored");state=saved;++restores;});
        };
        GenerationOptions options;options.cache_slots=4;options.cache_bytes=20000;Generation engine(16,64,{},op,options);
        GenerationEvents events;events.token=[&](int32_t id){emitted.push_back(id);};
        auto request=parse_generation_request("GEN 1 cvec=1 0,3,4,5",16,64);
        require(engine.run(request,events).reused==0 && emitted.back()==1,"initial steered generation");
        require(engine.run(request,events).reused==4 && switches==0,"same-mode live prefix not reused");
        request.control_vector=false;require(engine.run(request,events).reused==0 && emitted.back()==2 && switches==1,"mode change reused stale steered prefix");
        const auto saved_off=state;require(engine.run(request,events).reused==4 && switches==1,"unsteered continuation lost prefix");
        const auto before_switch_back=reads;
        request.control_vector=true;require(engine.run(request,events).reused==4 && emitted.back()==1 && switches==2 && restores==1 && reads==before_switch_back,"steered parked prefix/logits lost across mode switch");
        request.control_vector=false;require(engine.run(request,events).reused==4 && emitted.back()==2 && switches==3 && restores==2 && reads==before_switch_back,"unsteered parked prefix/logits lost across mode switch");
        on=false;state=saved_off;engine.adopt(state,logits(),false);request.control_vector=false;
        const auto before=reads;require(engine.run(request,events).reused==4 && reads==before && switches==3 && emitted.back()==2,"disk-adopted unsteered prefix reset or sampled wrong logits");
        // Invalid and cancelled requests must not change the arithmetic mode.
        for(const auto* line:{"GEN 1 cvec=2 0,1","GEN 1 cvec=-1 0,1","GEN 1 cvec=1 cvec=0 0,1"}) {
            bool rejected=false;try {(void)parse_generation_request(line,16,64);} catch(const std::invalid_argument&) {rejected=true;}require(rejected,"invalid cvec request accepted");
        }
        request.control_vector=true;require(engine.run(request,events,[]{return true;}).finish=="cancelled" && switches==3,"pre-cancelled request switched control mode");
        fail=true;bool fatal=false;try {(void)engine.run(request,events);} catch(const SessionFatal&) {fatal=true;}
        require(fatal && !engine.cache_stats().entries,"partially failed mode switch kept reusable model state");
        auto reset_failure=op;reset_failure.reset=[] {throw std::runtime_error("partial reset failure");};
        Generation failed_reset(16,64,{},reset_failure,options);request.control_vector=false;fatal=false;
        try {(void)failed_reset.run(request,events);} catch(const SessionFatal&) {fatal=true;}
        require(fatal,"failed mode-switch reset was treated as a recoverable request error");
        // Without a loaded vector the request flag is harmless, as upstream.
        op.control_vector={};Generation plain(16,64,{},op,options);request.control_vector=false;
        (void)plain.run(request,events);request.control_vector=true;require(plain.run(request,events).reused==4,"absent vector changed request cache identity");
        std::cout<<"PASS actual cvec protocol/generation parked on/off modes, exact same-token prefix/logits reuse, restored-mode adoption, no-vector behavior and fatal failed switch\n";
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
