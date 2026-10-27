#include "strata/hpu/sampler.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <numeric>

using namespace strata::hpu;
namespace {
void require(bool ok,const char* message) {if(!ok) throw std::runtime_error(message);}
template<class F> void rejects(F&& f) {bool caught=false;try {f();} catch(const std::exception&) {caught=true;}require(caught,"invalid sampler input accepted");}
}
int main() {
    try {
        struct Draw {uint64_t seed,counter;uint32_t mantissa;};
        for(const Draw d:std::array<Draw,5>{{{0,0,16079353},{1,0,5765649},{0,1,5293434},{42,123,14070705},{UINT64_MAX,UINT64_MAX,10791978}}})
            require(philox_uniform(d.seed,d.counter)==float(d.mantissa)/16777216,"frozen integer Philox oracle");
        strata::kernels::SamplerParams p;p.top_p=1;p.top_k=64;p.greedy=true;
        const std::array<float,3> logits{2,1,-1};const std::array<int32_t,3> history{0,0,2};
        require(sample(logits,{},p)==0,"greedy argmax");p.penalty_last_n=3;p.penalty_repeat=2;p.penalty_freq=.1f;p.penalty_present=.2f;
        require(sample(logits,history,p)==1,"repeat sign/frequency/presence penalty");
        p.penalty_last_n=1;require(sample(logits,history,p)==0,"penalty window tail");
        p={};p.greedy=true;const std::array<float,3> bias{0,-INFINITY,10};require(sample(logits,{},p,bias)==2,"bias before greedy");
        p={};p.top_k=3;p.top_p=.8f;p.min_p=0;p.temperature=100;
        const std::array<float,4> ordered{4,3,2,1};auto d=sampling_distribution(ordered,{},p);
        require(d.ids==std::vector<int32_t>({0,1}),"top-p must precede temperature");
        p.min_keep=3;p.min_p=.5f;d=sampling_distribution(ordered,{},p);require(d.ids==std::vector<int32_t>{0},"min-p must follow top-p/min-keep");
        p={};p.top_k=0;p.top_p=1;p.temperature=1;std::vector<float> ties(100);
        d=sampling_distribution(ties,{},p);require(d.ids.size()==64 && d.ids.front()==0 && d.ids.back()==63,"top-k cap/tie ordering");
        for(auto probability:d.probabilities) require(probability==1.0/64,"uniform distribution");
        const std::array<float,2> pair{0,float(std::log(3.0))};p.top_k=2;d=sampling_distribution(pair,{},p);
        require(d.ids==std::vector<int32_t>({1,0}) && std::abs(d.probabilities[0]-.75)<1e-7,"independent softmax probability");
        for(bool gumbel:{false,true}) {
            p.gumbel=gumbel;uint32_t ones=0;
            for(uint64_t draw=0;draw<20000;++draw) {p.counter=draw;ones+=sample(pair,{},p)==1;}
            require(ones>14600 && ones<15400,"sampled distribution statistical check");
        }
        p.gumbel=false;p.temperature=1e-40f;require(sample(pair,{},p)==1,"tiny temperature overflow");
        const std::array<float,2> banned{-INFINITY,-INFINITY};rejects([&]{(void)sample(banned,{},p);});
        const std::array<float,2> nan{NAN,0};rejects([&]{(void)sample(nan,{},p);});
        p.top_p=NAN;rejects([&]{(void)sample(pair,{},p);});p={};p.penalty_repeat=0;rejects([&]{(void)sample(pair,{},p);});
        p={};p.logit_bias=pair.data();rejects([&]{(void)sample(pair,{},p);});
        p={};p.greedy=true;
        for(const size_t width:{1u,7u,8u,9u,15u,16u,17u,257u,248320u}) {
            std::vector<float> row(width,-INFINITY);
            for(size_t a=0;a<std::min<size_t>(width,19);++a) {
                std::fill(row.begin(),row.end(),-INFINITY);row[a]=-0.0f;row[width-1]=0.0f;
                require(sample(row,history,p)==int32_t(std::min(a,width-1)),"greedy SIMD block tie/signed-zero order");
            }
            for(size_t i=0;i<width;++i) row[i]=float(int((i*37+13)%307)-153)/11;
            const auto expected=int32_t(std::max_element(row.begin(),row.end())-row.begin());
            require(sample(row,history,p)==expected,"greedy full-vocabulary scalar oracle");
            row.back()=NAN;rejects([&]{(void)sample(row,history,p);});
            row.back()=INFINITY;rejects([&]{(void)sample(row,history,p);});
            std::fill(row.begin(),row.end(),-INFINITY);rejects([&]{(void)sample(row,history,p);});
        }
        std::cout<<"PASS sampling order, sign-correct penalties/window, bias/bans, ties, Philox frozen draws, Gumbel/Philox distributions and numeric guards\n";
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
