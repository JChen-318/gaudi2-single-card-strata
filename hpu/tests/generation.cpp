#include "strata/hpu/generation.hpp"
#include "strata/hpu/protocol.hpp"
#include <array>
#include <iostream>

using namespace strata::hpu;
namespace {
void require(bool ok,const char* message) {if(!ok) throw std::runtime_error(message);}
template<class F> void rejects(F&& f) {bool caught=false;try {f();} catch(const std::exception&) {caught=true;}require(caught,"invalid request accepted");}
}
int main() {
    try {
        uint32_t resets=0,decoded=0;std::vector<int32_t> state,emitted;int32_t fail_token=-1;bool cancel=false;
        GenerationOperations op;
        op.reset=[&]{++resets;state.clear();};
        op.decode=[&](int32_t token,const std::function<bool()>& cancelled) {
            if(cancelled && cancelled()) throw Cancelled();
            if(token==fail_token) throw std::runtime_error("injected atomic decode failure");
            ++decoded;state.push_back(token);std::vector<float> logits(5,-10);logits[(token+1)%5]=10;return logits;
        };
        Generation generator(5,16,{4},op);GenerationEvents events;events.token=[&](int32_t token){emitted.push_back(token);};
        auto request=parse_generation_request("GEN 3 0,1",5,16);auto result=generator.run(request,events);
        require(emitted==std::vector<int32_t>({2,3,4}) && result.finish=="stop" && result.generated==3,"known next-token generation chain");
        require(state==std::vector<int32_t>({0,1,2,3}),"last emitted token must be fed back only when needed");
        emitted.clear();request=parse_generation_request("GEN 1 0,1,2,3,4,0",5,16);result=generator.run(request,events);
        require(result.reused==4 && emitted==std::vector<int32_t>{1} && resets==0,"prefix continuation reuse");
        request=parse_generation_request("GEN 1 2,0",5,16);result=generator.run(request,events);require(result.reused==0 && resets==1,"branch reset/replay");
        request=parse_generation_request("GEN 4 2,0",5,16);cancel=false;emitted.clear();events.token=[&](int32_t token){emitted.push_back(token);cancel=true;};
        result=generator.run(request,events,[&]{return cancel;});require(result.generated==1 && result.finish=="cancelled" && state==std::vector<int32_t>({2,0}),"cancel during fed-back decode");
        cancel=false;events.token=[&](int32_t token){emitted.push_back(token);};
        request=parse_generation_request("GEN 1 1,3,0",5,16);fail_token=3;rejects([&]{(void)generator.run(request,events);});
        require(state==std::vector<int32_t>{1},"failed prefill lost accepted prefix");fail_token=-1;result=generator.run(request,events);require(result.reused==1,"resume accepted prefill prefix after failure");
        const auto before=decoded;
        for(const auto* line:{"GEN 0 1","GEN 1 -1","GEN 1 5","GEN 1 1,","GEN 1 1 2","GEN 1 seed=-1 1","GEN 1 top_p=nan 1","GEN 1 top_k=2x 1","GEN 1 seed=1 seed=2 1","GENI 1 1","GEN 1 unknown=1 1","GEN 1 logit_bias=0:101 1","GEN 20 1"})
            rejects([&]{(void)parse_generation_request(line,5,16);});
        auto biased=parse_generation_request("GEN 1 logit_bias=0:-100,3:10 seed=18446744073709551615 1",5,16);
        require(biased.bias.size()==5 && biased.sampling.seed==UINT64_MAX,"bias/seed protocol parse");
        require(decoded==before,"parser validation performed model work");
        generator.reset();require(state.empty(),"explicit generation reset");
        const std::array<int32_t,2> restored{0,1};std::vector<float> restored_logits(5,-10);restored_logits[3]=10;
        generator.adopt(restored,restored_logits);state.assign(restored.begin(),restored.end());emitted.clear();
        request=parse_generation_request("GEN 1 0,1",5,16);result=generator.run(request,events);
        require(result.reused==2 && emitted==std::vector<int32_t>{3},"restored logits were not adopted with prefix");
        generator.invalidate();result=generator.run(request,events);require(result.reused==0,"invalidated prefix reused old model state");
        std::cout<<"PASS actual generation loop, known token chain, EOS, prompt reuse/branch replay, cancellation and failed prefill resume\n"
                   "PASS original text request grammar, bias/uint64 seed and strict malformed/unsupported key rejection\n";
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
