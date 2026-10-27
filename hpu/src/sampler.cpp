#include "strata/hpu/sampler.hpp"
#include "strata/hpu/performance_modes.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#include <immintrin.h>
#endif

namespace strata::hpu {
float philox_uniform(uint64_t seed,uint64_t counter) {
    uint32_t c0=uint32_t(counter),c1=uint32_t(counter>>32),c2=uint32_t(seed),c3=uint32_t(seed>>32);
    for(uint32_t i=0;i<10;++i) {
        const uint64_t a=uint64_t(0x9E3779B9u)*c0,b=uint64_t(0xBB67AE85u)*c2;
        const auto n0=uint32_t(b>>32)^c1^i,n2=uint32_t(a>>32)^c3;
        c0=n0;c1=uint32_t(b);c2=n2;c3=uint32_t(a);
    }
    return float(c0>>8)*(1.0f/16777216.0f);
}
namespace {
double gumbel_exp(uint64_t seed,uint64_t counter,uint32_t token) {
    uint64_t z=seed^(counter*0x9E3779B97F4A7C15ull)^(uint64_t(token)*0xD1B54A32D192ED03ull);
    z+=0x9E3779B97F4A7C15ull;z=(z^(z>>30))*0xBF58476D1CE4E5B9ull;z=(z^(z>>27))*0x94D049BB133111EBull;z^=z>>31;
    return -std::log((double(z>>11)+0.5)*(1.0/9007199254740992.0));
}
struct Candidate {int32_t id;float logit;};
int32_t plain_argmax_scalar(std::span<const float> values) {
    int32_t best=-1;float maximum=-INFINITY;
    for(size_t i=0;i<values.size();++i) {
        const float x=values[i]+0.0f;
        if(std::isnan(x) || x==INFINITY) throw std::invalid_argument("invalid host logits/bias");
        if(x>maximum) {maximum=x;best=int32_t(i);}
    }
    if(best<0) throw std::invalid_argument("all token logits are banned");
    return best;
}
#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
__attribute__((target("avx")))
int32_t plain_argmax_native(std::span<const float> values) {
    __m256 maxima=_mm256_set1_ps(-INFINITY);const auto infinity=_mm256_set1_ps(INFINITY);
    size_t i=0;
    for(;i+8<=values.size();i+=8) {
        const auto x=_mm256_add_ps(_mm256_loadu_ps(values.data()+i),_mm256_setzero_ps());
        if(_mm256_movemask_ps(_mm256_or_ps(_mm256_cmp_ps(x,x,_CMP_UNORD_Q),_mm256_cmp_ps(x,infinity,_CMP_EQ_OQ))))
            throw std::invalid_argument("invalid host logits/bias");
        maxima=_mm256_max_ps(maxima,x);
    }
    float lanes[8];_mm256_storeu_ps(lanes,maxima);float maximum=-INFINITY;
    for(const float x:lanes) maximum=std::max(maximum,x);
    for(;i<values.size();++i) {
        const float x=values[i]+0.0f;
        if(std::isnan(x) || x==INFINITY) throw std::invalid_argument("invalid host logits/bias");
        maximum=std::max(maximum,x);
    }
    if(maximum==-INFINITY) throw std::invalid_argument("all token logits are banned");
    const auto target=_mm256_set1_ps(maximum);
    // A second pass finds the lowest ID even when equal maxima occur in different SIMD lanes/blocks.
    i=0;
    for(;i+8<=values.size();i+=8) {
        const auto x=_mm256_add_ps(_mm256_loadu_ps(values.data()+i),_mm256_setzero_ps());
        const auto mask=unsigned(_mm256_movemask_ps(_mm256_cmp_ps(x,target,_CMP_EQ_OQ)));
        if(mask) return int32_t(i+unsigned(__builtin_ctz(mask)));
    }
    for(;i<values.size();++i) if(values[i]+0.0f==maximum) return int32_t(i);
    throw std::logic_error("greedy argmax lost its maximum");
}
#endif
// The sampled path's candidate order: the larger score first, the lower id among equals.
bool before(const Candidate& a,const Candidate& b) {return a.logit>b.logit || (a.logit==b.logit && a.id<b.id);}
// Enters `at` into the sorted top-`width` list. Ids arrive in increasing order, so an equal score never displaces a
// lower id: the list a full partial sort of every candidate keeps.
void offer(std::vector<Candidate>& list,size_t width,int32_t at,float value) {
    const Candidate c{at,value};
    if(list.size()==width) {if(!before(c,list.back())) return;list.pop_back();}
    list.insert(std::upper_bound(list.begin(),list.end(),c,before),c);
}
// values[i] is token base + i.
void top_logits_scalar(std::span<const float> values,size_t base,size_t width,std::vector<Candidate>& list) {
    for(size_t i=0;i<values.size();++i) {
        const float x=values[i]+0.0f;
        if(std::isnan(x) || x==INFINITY) throw std::invalid_argument("invalid host logits/bias");
        if(x!=-INFINITY) offer(list,width,int32_t(base+i),x);
    }
}
#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
// Unbiased, unpenalized logits: the scalar list, skipping every 8 values that cannot enter it.
__attribute__((target("avx")))
void top_logits_native(std::span<const float> values,size_t width,std::vector<Candidate>& list) {
    const auto infinity=_mm256_set1_ps(INFINITY);size_t i=0;
    for(;i+8<=values.size();i+=8) {
        const auto x=_mm256_loadu_ps(values.data()+i);
        if(_mm256_movemask_ps(_mm256_or_ps(_mm256_cmp_ps(x,x,_CMP_UNORD_Q),_mm256_cmp_ps(x,infinity,_CMP_EQ_OQ))))
            throw std::invalid_argument("invalid host logits/bias");
        if(list.size()==width && !_mm256_movemask_ps(_mm256_cmp_ps(x,_mm256_set1_ps(list.back().logit),_CMP_GT_OQ))) continue;
        top_logits_scalar(values.subspan(i,8),i,width,list);
    }
    top_logits_scalar(values.subspan(i),i,width,list);
}
#endif
void top_logits(std::span<const float> values,size_t width,std::vector<Candidate>& list) {
#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
    if(__builtin_cpu_supports("avx")) return top_logits_native(values,width,list);
#endif
    top_logits_scalar(values,0,width,list);
}
int32_t plain_argmax(std::span<const float> values) {
#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
    if(__builtin_cpu_supports("avx")) return plain_argmax_native(values);
#endif
    return plain_argmax_scalar(values);
}
}
SampleDistribution sampling_distribution(std::span<const float> logits,std::span<const int32_t> history,
    const strata::kernels::SamplerParams& p,std::span<const float> bias) {
    if(logits.empty() || logits.size()>uint64_t(INT32_MAX) || (!bias.empty() && bias.size()!=logits.size()) || p.logit_bias ||
       p.penalty_last_n<0 || p.min_keep<1 || !std::isfinite(p.temperature) || !std::isfinite(p.top_p) || p.top_p<=0 || p.top_p>1 ||
       !std::isfinite(p.min_p) || p.min_p<0 || p.min_p>1 || !std::isfinite(p.penalty_repeat) || p.penalty_repeat<=0 ||
       !std::isfinite(p.penalty_freq) || !std::isfinite(p.penalty_present)) throw std::invalid_argument("invalid host sampler parameters");
    if(fast_greedy_enabled() && (p.greedy || p.temperature<=0) && bias.empty() &&
       (history.empty() || !p.penalty_last_n || (p.penalty_repeat==1.0f && p.penalty_freq==0.0f && p.penalty_present==0.0f))) {
        SampleDistribution result;result.ids.push_back(plain_argmax(logits));result.probabilities.push_back(1.0);return result;
    }
    // One zeroed count table per thread, reused: only the history's entries are ever non-zero, and they are
    // cleared again on the way out (also when a check below throws).
    thread_local std::vector<uint32_t> counts;
    if(counts.size()<logits.size()) counts.assign(logits.size(),0);
    const size_t window=history.size()-std::min<size_t>(history.size(),p.penalty_last_n);
    struct Clear {
        std::span<const int32_t> ids;std::vector<uint32_t>& counts;size_t size;
        ~Clear() {for(const auto id:ids) if(id>=0 && uint32_t(id)<size) counts[size_t(id)]=0;}
    } clear{history.subspan(window),counts,logits.size()};
    for(size_t i=window;i<history.size();++i)
        if(history[i]>=0 && uint32_t(history[i])<logits.size()) ++counts[history[i]];
    const auto score=[&](size_t i) {
        const auto b=bias.empty()?0:bias[i];
        if(std::isnan(logits[i]) || logits[i]==INFINITY || std::isnan(b) || b==INFINITY) throw std::invalid_argument("invalid host logits/bias");
        float value=logits[i]+b;
        if(counts[i]) {value=value<=0?value*p.penalty_repeat:value/p.penalty_repeat;value-=float(counts[i])*p.penalty_freq+p.penalty_present;}
        if(value==INFINITY || std::isnan(value)) throw std::overflow_error("sampler penalty overflow");
        return value;
    };
    const bool greedy=p.greedy || p.temperature<=0;
    if(greedy) {
        // One candidate, as the top-k list below keeps it: the largest score, the lowest id among equals.
        // Ids are visited in increasing order, so a strictly larger score is the only replacement.
        int32_t best=-1;float best_value=-INFINITY;
        for(size_t i=0;i<logits.size();++i) {
            const float value=score(i);
            if(value!=-INFINITY && (best<0 || value>best_value)) {best=int32_t(i);best_value=value;}
        }
        if(best<0) throw std::invalid_argument("all token logits are banned");
        SampleDistribution result;result.ids.push_back(best);result.probabilities.push_back(1.0);return result;
    }
    // The top-k list in one pass (offer above); without bias or penalties every score is its logit.
    const size_t width=p.top_k>0 && p.top_k<64?size_t(p.top_k):64;
    std::vector<Candidate> candidates;candidates.reserve(width+1);
    if(bias.empty() && (window==history.size() || (p.penalty_repeat==1.0f && p.penalty_freq==0.0f && p.penalty_present==0.0f)))
        top_logits(logits,width,candidates);
    else for(size_t i=0;i<logits.size();++i) {
        const float value=score(i);
        if(value!=-INFINITY && (candidates.size()<width || value>candidates.back().logit)) offer(candidates,width,int32_t(i),value);
    }
    if(candidates.empty()) throw std::invalid_argument("all token logits are banned");
    const auto keep=candidates.size();
    if(!greedy) {
        if(p.top_p<1) {
            double sum=0;for(const auto& c:candidates) sum+=std::exp(double(c.logit)-candidates[0].logit);
            double cumulative=0;size_t cut=keep;
            for(size_t i=0;i<keep;++i) {cumulative+=std::exp(double(candidates[i].logit)-candidates[0].logit)/sum;if(cumulative>=p.top_p) {cut=i+1;break;}}
            candidates.resize(std::min(keep,std::max<size_t>(cut,p.min_keep)));
        }
        if(p.min_p>0) {
            const float threshold=candidates[0].logit+std::log(p.min_p);
            const auto stop=std::find_if(candidates.begin()+1,candidates.end(),[&](const Candidate& c){return c.logit<threshold;});candidates.erase(stop,candidates.end());
        }
    }
    SampleDistribution result;result.ids.reserve(candidates.size());result.probabilities.reserve(candidates.size());double sum=0;
    for(const auto& c:candidates) {
        // Subtract the maximum before division: this remains finite for tiny
        // positive temperatures, where float inverse-temperature would overflow.
        const double probability=greedy?1:std::exp((double(c.logit)-candidates[0].logit)/double(p.temperature));
        result.ids.push_back(c.id);result.probabilities.push_back(probability);sum+=probability;
    }
    for(auto& probability:result.probabilities) probability/=sum;
    return result;
}
int32_t sample(std::span<const float> logits,std::span<const int32_t> history,const strata::kernels::SamplerParams& p,std::span<const float> bias) {
    const auto d=sampling_distribution(logits,history,p,bias);
    if(p.gumbel) {
        double maximum=-1;int32_t id=d.ids.front();
        for(size_t i=0;i<d.ids.size();++i) {const auto score=d.probabilities[i]/gumbel_exp(p.seed,p.counter,uint32_t(d.ids[i]));if(score>maximum) {maximum=score;id=d.ids[i];}}
        return id;
    }
    const double uniform=philox_uniform(p.seed,p.counter);double cumulative=0;
    for(size_t i=0;i<d.ids.size();++i) {cumulative+=d.probabilities[i];if(uniform<cumulative) return d.ids[i];}
    return d.ids.back();
}
}
