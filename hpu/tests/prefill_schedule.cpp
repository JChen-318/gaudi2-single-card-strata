#include "strata/hpu/prefill_schedule.hpp"
#include "strata/hpu/generation.hpp"
#include <algorithm>
#include <iostream>
#include <stdexcept>

using namespace strata::hpu;
namespace {
void require(bool ok,const char* message) {if(!ok) throw std::runtime_error(message);}
template<class F> void rejects(F f) {bool caught=false;try {f();} catch(const std::invalid_argument&) {caught=true;}require(caught,"bad chunk geometry accepted");}
}
int main() {
    try {
        for(uint32_t remaining=1;remaining<=4096;++remaining) for(uint32_t requested:{32u,64u,128u,256u}) {
            uint32_t rest=remaining,batched=0;
            while(rest>64) {const auto n=std::min(rest,32u);rest-=n;batched+=n;}
            require(canonical_prefill_count(remaining,requested,64,32)==std::min(requested,batched),"canonical 32-row scalar suffix changed");
            require(canonical_prefill_count(remaining,requested,64,0)==std::min(remaining,requested),"default schedule changed");
        }
        for(uint32_t length:{63u,64u,65u,95u,96u,100u,127u,128u,512u,513u,2055u,2304u}) for(uint32_t chunk:{64u,128u}) {
            std::vector<bool> actual,expected;std::vector<int32_t> state;
            const std::vector<float> logits{-100,0,-100,-100};
            GenerationOperations op;op.reset=[&]{state.clear();actual.clear();};
            op.decode=[&](int32_t token,const std::function<bool()>&) {state.push_back(token);actual.push_back(false);return logits;};
            op.prefill=[&](std::span<const int32_t> tokens,const PromptCommit& commit,const std::function<bool()>&) {
                state.insert(state.end(),tokens.begin(),tokens.end());actual.insert(actual.end(),tokens.size(),true);commit(uint32_t(tokens.size()),logits);
            };
            GenerationOptions options;options.prefill_chunk=chunk;options.canonical_prefill_rows=32;
            Generation generator(4,4096,{},op,options);GenerationRequest request;request.prompt.assign(length,0);request.max_new=1;request.sampling.temperature=0;
            GenerationEvents events;events.token=[](int32_t){};(void)generator.run(request,events);
            // The model follows the options' own tail length: the default moved from 64 to 1 (round 32, the 64-row
            // tail cost 40% of a 512-token prefill), and this check is about generation matching the schedule, not
            // about which tail length is the default.
            for(uint32_t at=0;at<length;) {const bool batch=length-at>options.short_read;const auto n=batch?std::min(32u,length-at):1u;expected.insert(expected.end(),n,batch);at+=n;}
            require(actual==expected && state.size()==length,"generation did not preserve the canonical scalar suffix");
        }
        PromptCheckpointSchedule schedule(3,40,10,true,2);
        require(!schedule.complete(11,8).periodic,"interval rounded to absolute token count");
        require(schedule.complete(19,8).periodic,"chunk crossing interval not captured");
        require(!schedule.complete(27,8).periodic,"interval did not advance from actual completed chunk");
        const auto overlap=schedule.complete(36,9);require(overlap.periodic && overlap.tail,"periodic/tail overlap missing");
        const auto final=schedule.complete(40,4);require(!final.periodic && !final.tail,"tail or interval repeated at final chunk");
        PromptCheckpointSchedule tail(0,28,100,true,2);
        require(!tail.complete(10,10).tail && tail.complete(20,10).tail && !tail.complete(28,8).tail,"one near-tail at existing chunk geometry");
        PromptCheckpointSchedule short_read(0,1088,10000,true,64);
        require(!short_read.complete(512,512).tail && !short_read.complete(1024,512).tail,"short read boundary became a tail");
        PromptCheckpointSchedule disabled(0,28,0,true,0);
        require(!disabled.complete(20,20).tail,"disabled interval still parked a tail");
        PromptCheckpointSchedule variable(0,28,100,true,2);
        require(!variable.complete(12,12).tail && variable.complete(20,8).tail,"tail used configured chunk instead of actual chunk size");
        rejects([]{PromptCheckpointSchedule bad(2,1,3,false,0);});
        rejects([&]{(void)tail.complete(28,0);});
        PromptCheckpointSchedule invalid(2,20,4,false,0);
        rejects([&]{(void)invalid.complete(10,9);});rejects([&]{(void)invalid.complete(21,19);});
        std::cout<<"PASS upstream chunk interval geometry, resume-relative overshoot, one tail, short read, variable chunk and invalid boundary guards\n";
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
