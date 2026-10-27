#include "strata/hpu/generation.hpp"
#include "strata/spec/draft_source.hpp"
#include <cmath>
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <optional>

using namespace strata::hpu;
namespace {
void need(bool ok,const char* what) {if(!ok) throw std::runtime_error(what);}
struct ChainEnvironment {
    std::optional<std::string> old;
    ChainEnvironment() {if(const char* value=std::getenv("STRATA_LOOKUP_CHAIN_FIXED")) old=value;set(nullptr);}
    void set(const char* value) {
#ifdef _WIN32
        need(_putenv_s("STRATA_LOOKUP_CHAIN_FIXED",value?value:"")==0,"test environment update failed");
#else
        need((value?setenv("STRATA_LOOKUP_CHAIN_FIXED",value,1):unsetenv("STRATA_LOOKUP_CHAIN_FIXED"))==0,"test environment update failed");
#endif
    }
    ~ChainEnvironment() {
#ifdef _WIN32
        (void)_putenv_s("STRATA_LOOKUP_CHAIN_FIXED",old?old->c_str():"");
#else
        if(old) (void)setenv("STRATA_LOOKUP_CHAIN_FIXED",old->c_str(),1);else (void)unsetenv("STRATA_LOOKUP_CHAIN_FIXED");
#endif
    }
};
struct Model {
    std::vector<int32_t> state,emitted;
    std::vector<size_t> verified,draft_prefixes;
    double time=0;float confidence=1;bool wrong=false,cancel_window=false,nan_clock=false;
    std::vector<float> logits() const {
        std::vector<float> out(16,-20);out[(state.back()+1)%16]=20;return out;
    }
    GenerationOperations operations(bool native,bool mtp) {
        GenerationOperations op;op.reset=[&]{state.clear();};
        op.decode=[&](int32_t token,const auto&) {time+=1;state.push_back(token);return logits();};
        op.checkpoint=[&]{return std::function<void()>([&,saved=state]{state=saved;});};
        op.policy_clock_ms=[&]{return nan_clock?NAN:time;};
        if(mtp) op.draft=[&](const DraftContext& context,uint32_t maximum,const auto&) {
            time+=1;draft_prefixes.push_back(context.history.size());
            if(!maximum) return std::vector<DraftRow>{};
            if(!context.confidence.empty()) context.confidence[0]=confidence;
            return std::vector<DraftRow>{{(context.history.back()+(wrong?2:1))%16,{},{}}};
        };
        if(native) op.verify=[&](const DraftContext& context,std::span<const DraftRow> drafts,const auto& cancelled) {
            if(cancel_window) throw Cancelled();
            need(std::equal(state.begin(),state.end(),context.history.begin(),context.history.end()),"adaptive window lost target ownership");
            time+=50;verified.push_back(drafts.size());
            const auto original=state;std::vector<float> after;
            for(const auto& row:drafts) {state.push_back(row.token);const auto next=logits();after.insert(after.end(),next.begin(),next.end());}
            state=original;
            auto result=select_verified_candidates(context.logits,after,drafts,context.history,context.sampling,{},context.mode,{},cancelled);
            for(uint32_t i=0;i<result.retained;++i) state.push_back(drafts[i].token);
            result.logits=logits();return result;
        };
        return op;
    }
};
GenerationRequest request() {
    GenerationRequest out;for(int i=0;i<111;++i) out.prompt.push_back(i%16);
    out.max_new=32;out.sampling.greedy=true;out.checkpoint=false;return out;
}
GenerationResult run(Generation& gen,Model& model,const GenerationRequest& req,bool foreign=false) {
    model.emitted.clear();GenerationEvents events;events.token=[&](int32_t token){model.emitted.push_back(token);};
    if(foreign) {events.cooperate=[&]{model.time+=1000000;};events.has_foreign_work=[] {return true;};}
    const auto result=gen.run(req,events);
    for(size_t i=0;i<model.emitted.size();++i)
        need(model.emitted[i]==(req.prompt.back()+1+int32_t(i))%16,"adaptive policy changed target token semantics");
    need(result.generated==model.emitted.size(),"adaptive publication/count mismatch");return result;
}
struct Chain: strata::spec::DraftSource {
    const char* name() const override {return "adaptive-cycle-chain";}
    void reset() override {}
    void append(const int32_t*,size_t) override {}
    int propose(const int32_t* tail,int n,int pending,int maximum,int32_t* out,int* match) override {
        need(pending==1 && n>pending,"adaptive chain lost pending/context boundary");
        for(int i=0;i<maximum;++i) out[i]=(tail[n-1]+i+1)%16;
        *match=64;return maximum;
    }
};
}
int main() {
    try {
        ChainEnvironment environment;
        const auto req=request();
        for(bool native:{false,true}) for(bool mtp:{false,true}) {
            Model model;GenerationOptions options;options.adaptive_drafts=true;options.suffix_min_match=3;
            Generation gen(16,256,{},model.operations(native,mtp),options);
            const auto result=run(gen,model,req,native);
            need(result.generated==32 && result.drafts_accepted>0,"adaptive lookup never verified its continuation");
            if(mtp) need(!model.draft_prefixes.empty() && model.draft_prefixes.front()==req.prompt.size()+1,"adaptive first window did not prime one output");
            if(native) need(!model.verified.empty() && model.verified.front()>1,"lookup failed to replace the shorter agreeing MTP/plain window");
            need(gen.adaptive_policy().lookup_rate(64)>.96,"successful lookups did not update their match bucket");
            const auto learned=gen.adaptive_policy().cost_ms(8);
            need(learned>3.6,"actual complete-window time was not learned");
            if(native) need(std::abs(learned-(mtp?52.0:51.0))<1e-6,"native policy timed the wrong round boundaries");
            gen.reset();need(gen.adaptive_policy().cost_ms(8)==learned,"reset discarded process-owned policy learning");
        }
        {
            Model model;model.wrong=true;GenerationOptions options;options.adaptive_drafts=true;options.suffix_min_match=3;
            Generation gen(16,256,{},model.operations(true,true),options);(void)run(gen,model,req);
            need(!model.verified.empty() && std::all_of(model.verified.begin(),model.verified.end(),[](size_t n){return n==1;}),
                 "lookup replaced an MTP window whose first guess disagreed");
            need(gen.adaptive_policy().lookup_rate(64)==.96,"an unused lookup was taught MTP failures");
        }
        {
            Model model;model.cancel_window=true;GenerationOptions options;options.adaptive_drafts=true;options.suffix_min_match=3;
            Generation gen(16,256,{},model.operations(true,true),options);const auto result=run(gen,model,req);
            need(result.finish=="cancelled" && result.generated==1 && gen.adaptive_policy().cost_ms(8)==3.6,
                 "cancelled uncommitted round polluted policy learning");
        }
        {
            Model model;model.nan_clock=true;GenerationOptions options;options.adaptive_drafts=true;options.suffix_min_match=3;
            Generation gen(16,256,{},model.operations(true,true),options);(void)run(gen,model,req);
            need(gen.adaptive_policy().cost_ms(8)==3.6,"invalid timing was treated as a performance measurement");
        }
        strata::spec::extra_draft_sources().push_back(std::make_unique<Chain>());
        for(float confidence:{.01f,1.f}) {
            Model model;model.confidence=confidence;GenerationOptions options;options.adaptive_drafts=true;options.lookup_chain=7;
            Generation gen(16,256,{},model.operations(true,true),options);const auto result=run(gen,model,req);
            need(confidence<.1f?result.chain_proposed==0:result.chain_proposed>0,"actual drafter confidence did not gate lookup-chain policy");
        }
        {
            environment.set("1");Model model;model.confidence=.01f;GenerationOptions options;
            options.adaptive_drafts=true;options.lookup_chain=7;
            Generation gen(16,256,{},model.operations(true,true),options);const auto result=run(gen,model,req);
            need(result.chain_proposed>0,"original fixed-chain environment override was ignored");environment.set(nullptr);
        }
        strata::spec::extra_draft_sources().clear();
        {
            Model model;GenerationOptions options; // default fixed mode must not acquire the new primer/timing path
            options.suffix_min_match=3;Generation gen(16,256,{},model.operations(true,true),options);(void)run(gen,model,req);
            need(!model.draft_prefixes.empty() && model.draft_prefixes.front()==req.prompt.size(),"fixed mode changed its first proposal boundary");
            need(gen.adaptive_policy().cost_ms(8)==3.6,"fixed mode silently enabled policy learning");
        }
        std::cout<<"PASS actual adaptive Generation: native/scalar lookup/MTP selection, primer, first-token agreement, per-round timing/learning/reset, cancellation/invalid clock, chain confidence and unchanged fixed default; synthetic host clock, no HPU performance claim\n";
    } catch(const std::exception& error) {strata::spec::extra_draft_sources().clear();std::cerr<<error.what()<<'\n';return 1;}
}
