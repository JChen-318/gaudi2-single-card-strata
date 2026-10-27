#include "strata/hpu/generation.hpp"
#include "strata/spec/draft_source.hpp"
#include <iostream>
using namespace strata::hpu;
namespace {
void require(bool ok,const char* message) {if(!ok) throw std::runtime_error(message);}
struct Source: strata::spec::DraftSource {
    uint32_t resets=0;std::vector<int32_t> history;
    const char* name() const override {return "independent-cycle-source";}
    void reset() override {++resets;history.clear();}
    void append(const int32_t* tokens,size_t n) override {history.insert(history.end(),tokens,tokens+n);}
    int propose(const int32_t* tail,int size,int pending,int maximum,int32_t* out,int* match) override {
        require(size>pending && pending==1,"chained source lost committed context/pending boundary");
        for(int i=0;i<maximum;++i) out[i]=(tail[size-1]+i+1)%11;
        *match=64;return maximum;
    }
};
}
int main() {
    try {
        std::vector<int32_t> state,emitted;GenerationOperations op;op.reset=[&]{state.clear();};
        op.checkpoint=[&] {return std::function<void()>([&,saved=state]{state=saved;});};
        op.decode=[&](int32_t token,const auto&) {state.push_back(token);std::vector<float> logits(11,-10);logits[(token+1)%11]=10;return logits;};
        op.draft=[](const DraftContext& context,uint32_t,const auto&) {return std::vector<DraftRow>{{(context.history.back()+1)%11,{},{}}};};
        GenerationOptions options;options.lookup_chain=7;
        Generation engine(11,64,{},op,options);GenerationRequest request;request.prompt={0,1,2,3,4,5,6,0,1,2};request.max_new=8;request.sampling.greedy=true;
        GenerationEvents events;events.token=[&](int32_t token){emitted.push_back(token);};
        const auto own=engine.run(request,events);const std::vector<int32_t> expected{3,4,5,6,7,8,9,10};
        require(emitted==expected && own.chain_proposed>0 && own.chain_accepted>=3,"original lookup did not extend/verify external draft prefix");
        auto source=std::make_unique<Source>();auto* observed=source.get();strata::spec::extra_draft_sources().push_back(std::move(source));
        engine.reset();emitted.clear();const auto extra=engine.run(request,events);
        require(emitted==expected && extra.chain_proposed==6 && extra.chain_accepted==6,"longest registered source did not win chain selection");
        require(observed->resets==1 && observed->history.size()==request.prompt.size()+8,"registered source did not see prompt/accepted outputs");
        request.prompt={6,7,8};request.max_new=3;emitted.clear();(void)engine.run(request,events);
        require(emitted==std::vector<int32_t>({9,10,0}) && observed->resets==2 && observed->history.size()==6,"new request retained stale source context");
        strata::spec::extra_draft_sources().clear();
        std::cout<<"PASS original lookup-chain and DraftSource registry inside actual generation, pending-draft boundary, longest-match choice, target mismatch and request reset\n";
    } catch(const std::exception& error) {strata::spec::extra_draft_sources().clear();std::cerr<<error.what()<<'\n';return 1;}
}
