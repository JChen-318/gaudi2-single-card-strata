#include "strata/hpu/slot_session.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
using namespace strata::hpu;
namespace {
void need(bool yes,const char* message) {if(!yes) throw std::runtime_error(message);}
template<class F> void rejects(F f) {bool failed=false;try {f();} catch(const std::exception&) {failed=true;}need(failed,"bad slot window accepted");}
std::vector<float> target(std::span<const int32_t> ids,bool control) {
    uint32_t hash=control?17:5;for(auto id:ids) hash=(hash*3+uint32_t(id))%31;
    std::vector<float> row(32);for(uint32_t i=0;i<32;++i) row[i]=float((hash*13+i*i*3+i)%47)/9;
    return row;
}
struct Fixture {
    struct Image {std::vector<int32_t> ids;bool control;};
    std::vector<int32_t> ids;bool control=true;
    std::vector<int32_t> stops;
    uint32_t calls=0,decodes=0,mode=0; // 0 accepts, 1 rejects, 2 alternates
    bool unavailable=false,bad_row=false,bad_contract=false,fail_park=false;
    SlotSessionOperations operations() {
        SlotSessionOperations op;
        op.reset=[&]{ids.clear();};op.control_vector=[&](bool c){control=c;};
        op.prefill=[&](std::span<const int32_t> input){ids.insert(ids.end(),input.begin(),input.end());return target(ids,control);};
        op.decode=[&](int32_t input){++decodes;ids.push_back(input);return target(ids,control);};
        op.checkpoint=[&]()->std::function<void()> {
            if(fail_park) throw std::runtime_error("injected park failure");
            auto image=std::make_shared<Image>(Image{ids,control});
            return [&,image]{ids=image->ids;control=image->control;};
        };
        op.speculate=[&](int32_t pending,std::span<const int32_t> history,const strata::kernels::SamplerParams& sampling)->std::optional<SlotCandidates> {
            ++calls;need(std::equal(ids.begin(),ids.end(),history.begin(),history.end()),"slot snapshot/history not restored before MTP");
            if(unavailable) return std::nullopt;
            auto tentative=ids;tentative.push_back(pending);auto before=target(tentative,control);
            auto token=sample(before,tentative,sampling,{});
            if(mode==1 || (mode==2 && calls%2==0)) token=(token+1)%32;
            SlotCandidates result;result.draft.token=token;
            tentative.push_back(token);auto after=target(tentative,control);
            if(bad_row) after[0]=NAN; // even a rejected physical row must be finite
            before.insert(before.end(),after.begin(),after.end());
            result.selected=select_slot_candidates(before,result.draft,pending,ids,sampling,stops);
            ids.push_back(pending);if(result.selected.retained) ids.push_back(token);
            result.selected.logits=target(ids,control);
            if(bad_contract) ++result.selected.retained;
            return result;
        };
        return op;
    }
};
GenerationRequest request(bool c,uint64_t seed) {
    GenerationRequest r;r.prompt=c?std::vector<int32_t>{3,4,5}:std::vector<int32_t>{7,9};r.control_vector=c;
    auto& p=r.sampling;p.seed=seed;p.counter=19;p.top_k=19;p.top_p=.83f;p.min_p=.03f;
    p.temperature=.8f;p.penalty_repeat=1.17f;p.penalty_freq=.09f;p.penalty_present=.2f;p.penalty_last_n=7;return r;
}
std::vector<int32_t> oracle(const GenerationRequest& r,uint32_t count) {
    auto history=r.prompt;auto p=r.sampling;std::vector<int32_t> out;
    for(uint32_t i=0;i<count;++i) {auto token=sample(target(history,r.control_vector),history,p,{});++p.counter;out.push_back(token);history.push_back(token);}
    return out;
}
void paths() {
    uint32_t dual=0,single=0;
    for(bool gumbel:{false,true}) for(uint32_t mode=0;mode<3;++mode) for(uint64_t seed=0;seed<10;++seed) {
        Fixture f;f.mode=mode;SlotSession slots(2,32,64,{},f.operations(),true);
        auto a=request(false,seed),b=request(true,100+seed);a.sampling.gumbel=b.sampling.gumbel=gumbel;
        const std::array<GenerationRequest,2> requests{a,b};
        std::array<std::vector<int32_t>,2> emitted,expected{oracle(a,21),oracle(b,21)};
        for(int slot=0;slot<2;++slot) slots.admit(slot,requests[slot],21,[&](int32_t id){emitted[slot].push_back(id);});
        while(emitted[0].size()<21 || emitted[1].size()<21) for(int slot=0;slot<2;++slot) {
            auto& out=emitted[slot];if(out.size()==21) continue;
            auto window=slots.step_window(slot,uint32_t(std::min<size_t>(2,21-out.size())));
            need(window.count>=1 && window.count<=2,"window size");
            if(window.count==2) ++dual;else ++single;
            for(uint32_t i=0;i<window.count;++i) {need(!window.rows[i].nonfinite && !window.rows[i].eos,"unexpected stop");out.push_back(window.rows[i].token);}
            need(std::equal(out.begin(),out.end(),expected[slot].begin()),"MTP changed exact Philox/Gumbel/penalty stream");
            auto fed=requests[slot].prompt;fed.insert(fed.end(),out.begin(),out.end()-1);
            need(f.ids==fed && f.control==requests[slot].control_vector,"pending/accepted prefix or control mode diverged");
        }
        slots.finish(0);slots.finish(1);
    }
    need(dual && single,"accept/reject/scalar paths bypassed");
}
void bounds_and_errors() {
    auto r=request(true,9);
    Fixture f;SlotSession e(1,32,64,{},f.operations(),false);e.admit(0,r,8,[](int32_t){});
    need(e.step_window(0,1).count==1 && !f.calls,"one-output budget ran a hidden speculative commit");
    rejects([&]{e.step_window(0,0);});rejects([&]{e.step_window(0,3);});
    f.unavailable=true;need(e.step_window(0,2).count==1 && f.calls==1,"unavailable proposal did not fall back scalar");
    Fixture near;SlotSession last(1,32,uint32_t(r.prompt.size())+2,{},near.operations(),false);
    last.admit(0,r,8,[](int32_t){});need(last.step_window(0,2).count==1 && !near.calls,"context p+3 cutoff violated");
    auto final=r;final.sampling.counter=UINT64_MAX-1;Fixture wide;SlotSession counters(1,32,64,{},wide.operations(),false);
    counters.admit(0,final,2,[](int32_t){});need(counters.step_window(0,2).count==1 && !wide.calls,"last legal draw rejected or wrapped into speculative bonus");
    for(int failure=0;failure<3;++failure) {
        Fixture broken;SlotSession slots(2,32,64,{},broken.operations(),false);
        slots.admit(0,r,8,[](int32_t){});auto other=request(false,71);int32_t first=-1;
        slots.admit(1,other,8,[&](int32_t id){first=id;});
        broken.bad_row=failure==0;broken.bad_contract=failure==1;broken.fail_park=failure==2;
        if(failure==0) {const auto bad=slots.step_window(0,2);need(bad.count==1 && bad.rows[0].nonfinite,"non-finite MTP row published a token");}
        else rejects([&]{slots.step_window(0,2);});
        rejects([&]{slots.step(0);});
        broken.bad_row=broken.bad_contract=broken.fail_park=false;
        need(first==oracle(other,3)[0] && slots.step_window(1,2).rows[0].token==oracle(other,3)[1],"failed slot damaged another owned image");
    }
}
void eos() {
    // Find distinct adjacent outputs so the designated stop cannot end admission.
    GenerationRequest r;std::vector<int32_t> expected;
    for(uint64_t seed=0;seed<100;++seed) {r=request(true,seed);expected=oracle(r,4);if(expected[0]!=expected[1] && expected[0]!=expected[2] && expected[1]!=expected[2]) break;}
    need(expected[0]!=expected[1] && expected[0]!=expected[2] && expected[1]!=expected[2],"EOS fixture");
    for(uint32_t at:{1u,2u}) for(uint32_t mode:{0u,1u}) {
        Fixture f;f.mode=mode;f.stops.assign(1,expected[at]);
        SlotSession slots(1,32,64,std::vector<int32_t>(1,expected[at]),f.operations(),false);
        slots.admit(0,r,8,[](int32_t){});const auto window=slots.step_window(0,2);
        const uint32_t count=mode==1?1:at;need(window.count==count,"EOS/window acceptance cutoff");
        if(mode==0 || at==1) need(window.rows[count-1].eos,"first/bonus EOS lost");
        auto fed=r.prompt;fed.push_back(expected[0]);if(mode==0 && at==2) fed.push_back(expected[1]);
        need(f.ids==fed,"EOS or rejected draft was fed");
    }
}
}
int main() {
    try {paths();bounds_and_errors();eos();std::cout<<"PASS production per-slot two-output MTP orchestration, exact target oracle, interleaved state/counters, accepted/rejected/bonus/EOS, hard bounds and unpublished failures\n";}
    catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
