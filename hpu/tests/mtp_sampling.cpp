#include "strata/hpu/mtp_sampling.hpp"
#include "strata/core/spec_prob.hpp"
#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
using namespace strata::hpu;
namespace {
void require(bool ok,const char* message) {if(!ok) throw std::runtime_error(message);}
template<class F> void rejects(F&& f) {bool caught=false;try {f();} catch(const std::exception&) {caught=true;}require(caught,"invalid MTP sampling accepted");}
}
int main() {
    try {
        const std::array<float,3> logits{float(std::log(0.6)),float(std::log(0.3)),float(std::log(0.1))};
        strata::kernels::SamplerParams p;p.temperature=1;p.top_k=3;p.top_p=1;
        auto raw=pick_mtp_draft(logits,{},p,{},VerifyMode::Exact);
        require(raw.row.token==0 && raw.row.ids.empty() && std::abs(raw.confidence-0.6f)<1e-6,"raw argmax/full-vocabulary gate oracle");
        const std::array<int32_t,3> history{0,0,1};const std::array<float,3> bias{-INFINITY,0,0};
        p.penalty_last_n=3;p.penalty_repeat=1.2f;p.penalty_freq=0.5f;
        for(bool gumbel:{false,true}) for(uint64_t seed=0;seed<1000;++seed) {
            p.seed=seed;p.counter=42;p.gumbel=gumbel;
            auto coupled=pick_mtp_draft(logits,history,p,bias,VerifyMode::Exact,{true,true,1,0});
            require(coupled.row.token==sample(logits,history,p,bias) && coupled.row.ids.empty(),"coupled draft uses different noise/penalty/bias chain");
            auto probability=pick_mtp_draft(logits,history,p,bias,VerifyMode::Probabilistic);
            auto independent=p;independent.counter|=strata::core::kSpecCtrDraft;
            require(probability.row.token==sample(logits,history,independent,bias) && probability.row.token!=0,"probability draft stream or banned token");
            validate_drafts(std::span(&probability.row,1),3,1);
            const auto expected=sampling_distribution(logits,history,independent,bias);
            require(probability.row.ids==expected.ids,"drafter q candidate ids differ from its own cut");
            for(size_t i=0;i<expected.ids.size();++i) require(std::abs(probability.row.probabilities[i]-expected.probabilities[i])<1e-6,"drafter q probabilities differ");
            auto gated=pick_mtp_draft(logits,history,p,bias,VerifyMode::Probabilistic,{false,false,0.5f,0});
            independent.temperature*=0.5f;const auto cold=sampling_distribution(logits,history,independent,bias);
            require(std::abs(gated.confidence-cold.probabilities.front())<1e-6,"temperature-scale/top-probability gate");
        }
        p.temperature=0.1f;raw=pick_mtp_draft(logits,history,p,bias,VerifyMode::Probabilistic,{true,true,1,0.7f});
        require(raw.row.token==0 && raw.row.ids.empty(),"minimum-temperature argmax fallback must remain point mass");
        const std::array<float,4> subset{-INFINITY,0,-INFINITY,float(std::log(3))};
        p.temperature=1;p.counter=5;p.penalty_last_n=0;
        auto partial=pick_mtp_draft(subset,{},p,{},VerifyMode::Probabilistic);
        require(partial.row.token==1 || partial.row.token==3,"sparse draft head emitted token outside subset");
        for(auto id:partial.row.ids) require(id==1 || id==3,"q contains a missing head row");
        raw=pick_mtp_draft(subset,{},p,{},VerifyMode::Exact);
        require(raw.row.token==3 && std::abs(raw.confidence-0.75f)<1e-6,"subset full-softmax confidence includes absent rows");
        const std::array<float,4> banned{0,-INFINITY,0,-INFINITY};
        partial=pick_mtp_draft(subset,{},p,banned,VerifyMode::Probabilistic);
        require(partial.row.token<0,"all-banned subset failed to fall back to target");
        rejects([&]{(void)pick_mtp_draft(std::array<float,1>{-INFINITY},{},p,{},VerifyMode::Exact);});
        rejects([&]{(void)pick_mtp_draft(logits,{},p,{},VerifyMode::Exact,{true,true,NAN,0});});
        p.temperature=1;p.counter=1ull<<61;rejects([&]{(void)pick_mtp_draft(logits,{},p,{},VerifyMode::Probabilistic);});
        std::cout<<"PASS MTP raw/full-softmax confidence oracle, coupled target chain, independent probability stream/q, gate and temperature policies\n";
    } catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
}
