#include "strata/hpu/generation.hpp"
#include "strata/core/spec_prob.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>

using namespace strata::hpu;
namespace {
void require(bool ok,const char* message) {if(!ok) throw std::runtime_error(message);}
template<class F> void rejects(F&& f) {bool caught=false;try {f();} catch(const std::exception&) {caught=true;}require(caught,"invalid verify input accepted");}
std::vector<float> logits(std::span<const int32_t> history,bool simple) {
    std::vector<float> out(11,-20);
    if(simple) out[(history.back()+1)%11]=20;
    else {
        uint32_t hash=1;for(auto token:history) hash=hash*31+uint32_t(token);
        for(uint32_t i=0;i<out.size();++i) out[i]=float(int((hash+i*37)%41)-20)/9;
    }
    return out;
}
struct Run {std::vector<int32_t> emitted,state;GenerationResult result;uint32_t native_windows=0;};
Run chain(uint64_t seed,bool gumbel,int mismatch,bool simple,bool lookup=false,uint32_t stop=99,bool native=false,bool probabilistic=false,bool full_q=false) {
    Run run;GenerationOperations op;
    op.reset=[&]{run.state.clear();};
    op.checkpoint=[&] {return std::function<void()>([&,saved=run.state]{run.state=saved;});};
    op.decode=[&](int32_t token,const std::function<bool()>& cancelled) {
        if(cancelled && cancelled()) throw Cancelled();
        run.state.push_back(token);return logits(run.state,simple);
    };
    if(native) op.verify=[&](const DraftContext& context,std::span<const DraftRow> rows,const auto& cancelled) {
        ++run.native_windows;auto speculative=run.state;std::vector<float> after;
        for(const auto& row:rows) {speculative.push_back(row.token);const auto values=logits(speculative,simple);after.insert(after.end(),values.begin(),values.end());}
        const auto stops=stop<11?std::vector<int32_t>{int32_t(stop)}:std::vector<int32_t>{};
        auto selected=select_verified_candidates(context.logits,after,rows,context.history,context.sampling,context.bias,context.mode,stops,cancelled);
        for(uint32_t i=0;i<selected.retained;++i) run.state.push_back(rows[i].token);
        selected.logits=logits(run.state,simple);return selected;
    };
    if(mismatch>=0) op.draft=[&](const DraftContext& context,uint32_t maximum,const auto&) {
        const auto history=context.history;
        require(context.sampling.counter==123+history.size()-14 && context.sampling.seed==seed && context.sampling.gumbel==gumbel,"draft context counter/seed is stale");
        const auto expected=logits(history,simple);require(std::equal(expected.begin(),expected.end(),context.logits.begin(),context.logits.end()),"draft context logits do not match committed target");
        std::vector<DraftRow> rows;
        for(uint32_t i=0;i<maximum;++i) {
            const auto id=(history.back()+1+int32_t(i)+(int(i)==mismatch?5:0))%11;
            if(full_q) {
                const auto u=philox_uniform(seed,(context.sampling.counter+i)|strata::core::kSpecCtrDraft);
                rows.push_back({u<0.2f?0:u<0.5f?4:9,{0,4,9},{0.2f,0.3f,0.5f}});
            } else rows.push_back({id,{}, {}});
        }
        return rows;
    };
    GenerationOptions options;options.max_draft=(mismatch>=0 || lookup)?7:0;options.suffix_min_match=lookup?3:0;
    options.verify_mode=probabilistic?VerifyMode::Probabilistic:VerifyMode::Exact;
    Generation engine(11,128,stop<11?std::vector<int32_t>{int32_t(stop)}:std::vector<int32_t>{},std::move(op),options);
    GenerationRequest request;request.prompt={0,1,2,3,4,5,6,7,8,9,10,0,1,2};request.max_new=42;
    auto& p=request.sampling;p.seed=seed;p.counter=123;p.gumbel=gumbel;p.temperature=simple?0:0.8f;
    p.top_k=7;p.top_p=0.8f;p.min_p=0.07f;p.penalty_last_n=9;p.penalty_repeat=1.2f;p.penalty_freq=0.08f;p.penalty_present=0.1f;
    request.bias.assign(11,0);request.bias[10]=-INFINITY;
    if(simple) request.bias.clear();
    GenerationEvents events;events.token=[&](int32_t token){run.emitted.push_back(token);};
    run.result=engine.run(request,events);return run;
}
void exact_paths() {
    for(bool simple:{false,true}) for(bool gumbel:{false,true}) for(uint64_t seed=0;seed<30;++seed) {
        const auto ordinary=chain(seed,gumbel,-1,simple);
        for(int mismatch=0;mismatch<8;++mismatch) {
            const auto verified=chain(seed,gumbel,mismatch,simple);
            const auto batched=chain(seed,gumbel,mismatch,simple,false,99,true);
            require(verified.emitted==ordinary.emitted && verified.state==ordinary.state,"draft changed target sampling/counter/history or committed a rejected row");
            require(batched.native_windows && batched.emitted==ordinary.emitted && batched.state==ordinary.state,"native candidate window changed scalar target/RNG/penalty path");
            require(batched.result.drafts_checked==verified.result.drafts_checked && batched.result.drafts_accepted==verified.result.drafts_accepted,"native window changed verify counters");
            require(verified.result.drafts_checked>0 && verified.result.drafts_proposed>=verified.result.drafts_checked,"verify was bypassed");
            if(simple) require(verified.result.drafts_accepted>0 || mismatch==0,"known correct drafts not accepted");
        }
        const auto lookup=chain(seed,gumbel,-1,simple,true);
        const auto native_lookup=chain(seed,gumbel,-1,simple,true,99,true);
        require(lookup.emitted==ordinary.emitted && lookup.state==ordinary.state,"prompt lookup changed target stream");
        require(native_lookup.emitted==ordinary.emitted && native_lookup.state==ordinary.state,"native suffix verification changed target stream");
        if(simple) require(lookup.result.drafts_accepted>0,"original suffix source did not reach real generation verify path");
    }
    const auto stopped=chain(0,false,7,true,false,6);
    require(stopped.emitted==std::vector<int32_t>({3,4,5,6}) && stopped.result.finish=="stop" && stopped.state.back()==5,"EOS fed back or window continued past stop");
    for(uint32_t stop=3;stop<=10;++stop) {
        const auto scalar=chain(0,false,7,true,false,stop),native=chain(0,false,7,true,false,stop,true);
        require(native.emitted==scalar.emitted && native.state==scalar.state && native.result.finish==scalar.result.finish,"native verify EOS prefix not retained exactly");
    }
}
void probabilities() {
    for(bool full_q:{false,true}) for(uint64_t seed=0;seed<200;++seed) {
        const auto scalar=chain(seed,seed%2,3,false,false,6,false,true,full_q),native=chain(seed,seed%2,3,false,false,6,true,true,full_q);
        require(native.native_windows && scalar.emitted==native.emitted && scalar.state==native.state &&
            scalar.result.drafts_checked==native.result.drafts_checked && scalar.result.drafts_accepted==native.result.drafts_accepted,
            "native probabilistic acceptance/residual RNG or penalty history differs from sequential reference");
    }
    const std::array<float,3> values{float(std::log(0.6)),float(std::log(0.3)),float(std::log(0.1))};
    for(bool point:{false,true}) for(bool ban:{false,true}) {
        std::array<uint32_t,3> counts{};uint32_t accepted=0;
        const std::array<float,3> bias{0,0,ban?-INFINITY:0};
        for(uint64_t seed=0;seed<40000;++seed) {
            strata::kernels::SamplerParams p;p.seed=seed;p.counter=17;p.temperature=1;p.top_p=1;p.top_k=3;
            DraftRow draft;
            if(point) draft.token=2;
            else {
                draft.ids={0,1,2};draft.probabilities={0.15f,0.75f,0.1f};
                const auto u=philox_uniform(seed,p.counter|strata::core::kSpecCtrDraft);
                draft.token=u<0.15f?0:u<0.9f?1:2;
            }
            const auto pick=verify_pick(values,{},p,bias,&draft,VerifyMode::Probabilistic);
            ++counts[size_t(pick.token)];accepted+=pick.accepted;
            require(!pick.accepted || pick.token==draft.token,"accepted wrong draft");
        }
        const double denominator=ban?0.9:1;
        require(std::abs(double(counts[0])/40000-0.6/denominator)<0.012 &&
                std::abs(double(counts[1])/40000-0.3/denominator)<0.012,"rejection/residual output differs from independent target probabilities");
        if(ban) require(!counts[2],"probabilistic verify emitted hard-banned token");
        require(accepted>0 || (point && ban),"probabilistic source was never accepted");
    }
}
void failures() {
    for(auto row:std::vector<DraftRow>{{11,{},{}},{0,{0},{0.5f}},{0,{1},{1}},{0,{0,0},{0.5f,0.5f}},
                                      {0,{0},{NAN}},{0,{0,1},{1.1f,-0.1f}},{0,{0},{}}})
        rejects([&]{validate_drafts(std::span(&row,1),11,1);});
    std::array<DraftRow,2> too_many{DraftRow{0,{},{}},DraftRow{1,{},{}}};rejects([&]{validate_drafts(too_many,11,1);});
    uint32_t calls=0;std::vector<int32_t> state;
    GenerationOperations op;op.reset=[&]{state.clear();};
    op.checkpoint=[&] {return std::function<void()>([&,saved=state]{state=saved;});};
    op.decode=[&](int32_t token,const auto& cancelled){if(cancelled && cancelled()) throw Cancelled();++calls;state.push_back(token);return logits(state,true);};
    op.draft=[](auto,uint32_t,const auto&){return std::vector<DraftRow>{{4,{},{}},{5,{},{}},{6,{},{}}};};
    GenerationOptions options;options.max_draft=2;Generation engine(11,32,{},op,options);
    GenerationRequest request;request.prompt={0,1};request.max_new=8;request.sampling.temperature=0;
    GenerationEvents events;std::vector<int32_t> output;events.token=[&](int32_t token){output.push_back(token);};
    rejects([&]{(void)engine.run(request,events);});
    require(state==request.prompt && output.empty(),"invalid proposal committed a token");
    request.sampling.counter=UINT64_MAX;const auto before=calls;rejects([&]{(void)engine.run(request,events);});
    require(calls==before,"counter overflow performed model work");request.sampling.counter=0;
    op.draft=[](const DraftContext& context,uint32_t maximum,const auto&) {
        const auto history=context.history;
        std::vector<DraftRow> rows;for(uint32_t i=0;i<maximum;++i) rows.push_back({(history.back()+1+int32_t(i))%11,{}, {}});return rows;
    };
    state.clear();Generation cancelled(11,32,{},op,options);bool stop=false;events.token=[&](int32_t token){output.push_back(token);stop=stop || output.size()==2;};
    const auto result=cancelled.run(request,events,[&]{return stop;});
    require(result.generated==3 && result.finish=="cancelled" && output==std::vector<int32_t>({2,3,4}) && state==std::vector<int32_t>({0,1,2,3}),"scalar committed-window cancellation differs from native atomic publication");
    options.verify_mode=VerifyMode::Probabilistic;Generation probability(11,32,{},op,options);
    request.sampling.counter=1ull<<61;rejects([&]{(void)probability.run(request,events);});
    options.max_draft=8;rejects([&]{validate_generation_options(options);});
}
void native_failures() {
    std::vector<int32_t> state,output;bool stop=false,early=false,bad=false;
    GenerationOperations op;op.reset=[&]{state.clear();};
    op.decode=[&](int32_t token,const auto& cancel) {if(cancel && cancel()) throw Cancelled();state.push_back(token);return logits(state,true);};
    op.draft=[](const DraftContext& context,uint32_t maximum,const auto&) {
        std::vector<DraftRow> rows;for(uint32_t i=0;i<maximum;++i) rows.push_back({(context.history.back()+1+int32_t(i))%11,{}, {}});return rows;
    };
    op.verify=[&](const DraftContext& context,std::span<const DraftRow> rows,const auto& cancel) {
        auto proposed=state;std::vector<float> after;
        for(const auto& row:rows) {proposed.push_back(row.token);const auto v=logits(proposed,true);after.insert(after.end(),v.begin(),v.end());}
        if(early) stop=true;
        auto result=select_verified_candidates(context.logits,after,rows,context.history,context.sampling,context.bias,context.mode,{},cancel);
        if(bad) {result.picks.clear();result.logits=logits(state,true);return result;}
        for(uint32_t i=0;i<result.retained;++i) state.push_back(rows[i].token);
        result.logits=logits(state,true);return result;
    };
    GenerationOptions options;options.max_draft=2;Generation engine(11,32,{},op,options);
    GenerationRequest request;request.prompt={0,1};request.max_new=8;request.sampling.temperature=0;
    GenerationEvents events;events.token=[&](int32_t id){output.push_back(id);stop=true;};
    early=true;auto result=engine.run(request,events,[&]{return stop;});
    require(result.finish=="cancelled" && !result.generated && output.empty() && state==request.prompt,"native cancel before commit published candidates");
    stop=false;early=false;engine.reset();result=engine.run(request,events,[&]{return stop;});
    require(result.finish=="cancelled" && result.generated==3 && output==std::vector<int32_t>({2,3,4}) && state==std::vector<int32_t>({0,1,2,3}),"native atomic publication or cancelled bonus fed an uncommitted token");
    stop=false;bad=true;output.clear();engine.reset();rejects([&]{engine.run(request,events);});require(output.empty(),"malformed native result emitted tokens");
    const std::array<float,2> before{1,0},after{0,1};const std::array<DraftRow,1> draft{DraftRow{0,{},{}}};
    strata::kernels::SamplerParams p;p.counter=UINT64_MAX;
    rejects([&]{select_verified_candidates(before,after,draft,{},p,{},VerifyMode::Exact,{});});
    p.counter=0;rejects([&]{select_verified_candidates(before,{},draft,{},p,{},VerifyMode::Exact,{});});
}
void follow_selection() {
    const std::array<float,3> before{20,-20,-20};
    std::array<float,6> after{-20,20,-20,20,-20,-20};
    const std::array<DraftRow,2> drafts{DraftRow{2,{},{}},DraftRow{1,{},{}}};
    strata::kernels::SamplerParams p;p.greedy=true;
    const std::array<int32_t,3> follow{2,1,2};
    const auto selected=select_verified_candidates(before,after,drafts,{},p,{},VerifyMode::Exact,{}, {},follow);
    require(selected.picks.size()==3 && selected.retained==2 && selected.follow_differ==2 && selected.follow_emitted==3 &&
            selected.picks[0].accepted && selected.picks[1].accepted && !selected.picks[2].accepted && selected.picks[2].token==2,
            "native follow must affect acceptance, bonus and emitted-only statistics");
    const std::array<int32_t,1> stops{2};
    const auto eos=select_verified_candidates(before,after,drafts,{},p,{},VerifyMode::Exact,stops,{},follow);
    require(eos.stopped && eos.picks.size()==1 && eos.retained==0 && eos.follow_emitted==1 && eos.follow_differ==1,
            "follow-forced EOS was retained or later candidate rows were counted as emitted");
    const std::array<int32_t,3> mismatch{0,1,2};
    const auto rejected=select_verified_candidates(before,after,drafts,{},p,{},VerifyMode::Exact,{}, {},mismatch);
    require(rejected.picks.size()==1 && !rejected.retained && !rejected.follow_differ && rejected.follow_emitted==1,"follow mismatch did not stop native accepted prefix");
    uint32_t polls=0;
    rejects([&]{select_verified_candidates(before,after,drafts,{},p,{},VerifyMode::Exact,{},[&]{return ++polls==2;},follow);});
    auto invalid=follow;invalid[2]=3;
    rejects([&]{select_verified_candidates(before,after,drafts,{},p,{},VerifyMode::Exact,{}, {},invalid);});
    after.back()=NAN;
    rejects([&]{select_verified_candidates(before,after,drafts,{},p,{},VerifyMode::Exact,stops,{},follow);});
}
void dflash_sampling() {
    // DFlash under --spec-prob: the drafted rows are point masses, so a sampled request's window must accept each
    // row with p(d), carry the residual p-d of a rejected row into the next window (NOT re-draw from p), and never
    // take the greedy argmax shortcut. Every case below uses a fixed target distribution, so the one legitimate
    // draw of a rejected row can be recomputed independently.
    const std::vector<int32_t> prompt{0,1,2,3,4,5,6,7,8,9,10,0,1,2};
    struct DFlashRun {std::vector<int32_t> emitted,state;GenerationResult result;uint32_t rounds=0,first_keep=0;int32_t first_pending=-1;};
    const auto run_window=[&](uint64_t seed,int32_t first_draft,const std::vector<float>& target) {
        DFlashRun run;GenerationOperations op;
        op.reset=[&]{run.state.clear();};
        op.checkpoint=[&]{return std::function<void()>([&,saved=run.state]{run.state=saved;});};
        op.decode=[&](int32_t token,const std::function<bool()>& cancelled) {
            if(cancelled && cancelled()) throw Cancelled();
            run.state.push_back(token);return target;
        };
        op.dflash=[&](int32_t pending,uint32_t maximum,const DflashSelection& select,const std::function<bool()>&) {
            ++run.rounds;
            const std::vector<int32_t> ids{pending,first_draft,1,2};
            const std::array<int32_t,4> argmax{0,1,2,0};
            const auto keep=select(DflashRows{std::span<const int32_t>(ids),std::span<const int32_t>(argmax),
                [&](uint32_t){return std::span<const float>(target);}});
            if(run.rounds==1) {run.first_keep=keep;run.first_pending=pending;}
            const uint32_t accepted=std::min<uint32_t>(keep,maximum);
            for(uint32_t row=0;row<accepted;++row) run.state.push_back(ids[row]);
            DflashRound round;round.tokens.assign(ids.begin(),ids.begin()+accepted);round.logits=target;round.proposed=3;
            return round;
        };
        GenerationOptions options;options.verify_mode=VerifyMode::Probabilistic;
        Generation engine(11,64,{},std::move(op),options);
        GenerationRequest request;request.prompt=prompt;request.max_new=6;
        auto& p=request.sampling;p.seed=seed;p.counter=0;p.temperature=1;p.top_k=3;p.top_p=1;p.min_p=0;
        GenerationEvents events;events.token=[&](int32_t token){run.emitted.push_back(token);};
        run.result=engine.run(request,events);
        // The plain tail does not feed the very last emitted token (nothing follows it); every other committed
        // token must appear exactly once, in order, and no emission may lack its commit.
        const size_t fed=run.state.size()-prompt.size();
        require(fed==run.emitted.size() || (run.result.generated==request.max_new && fed+1==run.emitted.size()),
                "DFlash sampling emitted a token the target never committed or dropped one");
        require(std::equal(run.state.begin()+prompt.size(),run.state.end(),run.emitted.begin()),
                "DFlash sampling emitted tokens out of commit order");
        return run;
    };
    // p = {0.6,0.3,0.1}; the draft is the argmax, so its row rejects with probability 0.4 and the residual must
    // never be token 0 (q puts all its mass there). A fresh draw from p would emit 0 on rejects.
    {
        std::vector<float> target(11,-20);target[0]=float(std::log(0.6));target[1]=float(std::log(0.3));target[2]=float(std::log(0.1));
        uint32_t rejected=0;
        for(uint64_t seed=0;seed<400;++seed) {
            const auto run=run_window(seed,0,target);
            require(run.rounds && run.result.drafts_checked>0 && run.emitted.size()>=2,"DFlash sampling window never ran");
            if(run.first_keep!=1) continue;
            ++rejected;
            strata::kernels::SamplerParams p;p.seed=seed;p.counter=1;p.temperature=1;p.top_k=3;p.top_p=1;p.min_p=0;
            std::vector<int32_t> seen=prompt;seen.push_back(run.first_pending);
            const DraftRow point{0,{},{}};
            const auto expected=verify_pick(target,seen,p,{},&point,VerifyMode::Probabilistic);
            require(!expected.accepted && run.first_pending==run.emitted[0],"DFlash row accepted a token it rejected");
            require(run.emitted[1]==expected.token,"DFlash rejection did not carry the residual draw as the next token");
            require(run.emitted[1]!=0,"DFlash residual emitted the rejected point-mass token");
        }
        require(rejected>100 && rejected<250,"DFlash point-mass acceptance rate is not p(d)");
    }
    // p(0) = 0.05: with the greedy argmax shortcut the sampled row would accept its draft every time. Point-mass
    // sampling must reject ~95% of them.
    {
        std::vector<float> target(11,-20);target[0]=float(std::log(0.05));target[1]=float(std::log(0.475));target[2]=float(std::log(0.475));
        uint32_t rejected=0;
        for(uint64_t seed=0;seed<200;++seed) {
            const auto run=run_window(seed,0,target);
            if(run.first_keep==1) {++rejected;require(run.emitted[1]!=0,"DFlash sampled row took the argmax shortcut");}
        }
        require(rejected>=180,"DFlash sampled rows accepted a 5% draft too often (argmax shortcut still armed)");
    }
}
}
int main() {
    try {
        exact_paths();probabilities();failures();native_failures();follow_selection();dflash_sampling();
        std::cout<<"PASS real generation scalar/native candidate verify, every mismatch boundary, seed/counter/penalty/Gumbel parity, original suffix, bonus/EOS/cancel\n"
                   "PASS upstream probabilistic acceptance/residual with independent target histogram, point-mass drafts, bans and malformed sources\n"
                   "PASS DFlash sampled windows: point-mass p(d) acceptance, residual carry instead of a fresh p draw, no argmax shortcut\n";
    } catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
}
