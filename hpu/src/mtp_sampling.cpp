#include "strata/hpu/mtp_sampling.hpp"
#include "strata/core/spec_prob.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace strata::hpu {
MtpPick pick_mtp_draft(std::span<const float> logits,std::span<const int32_t> history,
    strata::kernels::SamplerParams parameters,std::span<const float> bias,VerifyMode mode,MtpSamplingOptions options) {
    if(logits.empty() || logits.size()>INT32_MAX || !std::isfinite(options.temperature_scale) || options.temperature_scale<=0 ||
       !std::isfinite(options.min_temperature) || options.min_temperature<0 || (mode!=VerifyMode::Exact && mode!=VerifyMode::Probabilistic))
        throw std::invalid_argument("invalid MTP sampling options");
    if(!bias.empty() && bias.size()!=logits.size()) throw std::invalid_argument("MTP bias shape mismatch");
    bool active=false,allowed=false;
    for(size_t i=0;i<logits.size();++i) {
        if(std::isnan(logits[i]) || logits[i]==INFINITY || (!bias.empty() && (std::isnan(bias[i]) || bias[i]==INFINITY))) throw std::invalid_argument("invalid MTP logits/bias");
        if(logits[i]!=-INFINITY) {active=true;if(bias.empty() || bias[i]!=-INFINITY) allowed=true;}
    }
    if(!active) throw std::invalid_argument("empty MTP head support");
    const bool sampled=!parameters.greedy && parameters.temperature>0 && parameters.temperature>=options.min_temperature;
    if(!sampled || (mode==VerifyMode::Exact && !options.coupled)) {
        // Original default proposals use raw argmax and full-vocabulary softmax
        // confidence; a greedy sampler's one-entry distribution would report 1
        // and incorrectly disable the draft-extension gate.
        const auto selected=std::max_element(logits.begin(),logits.end());double sum=0;
        for(float x:logits) sum+=std::exp(double(x)-*selected);
        return {{int32_t(selected-logits.begin()),{},{}},float(1/sum)};
    }
    if(!allowed) return {}; // All subset tokens banned: use ordinary target decode.
    if(mode==VerifyMode::Probabilistic) {
        if(parameters.counter>=core::kSpecCtrDraft) throw std::invalid_argument("MTP draft counter overlaps reserved streams");
        parameters.counter|=core::kSpecCtrDraft;parameters.temperature*=options.temperature_scale;
    }
    const auto distribution=sampling_distribution(logits,history,parameters,bias);
    const auto token=sample(logits,history,parameters,bias);MtpPick result;result.row.token=token;
    const auto index=size_t(std::find(distribution.ids.begin(),distribution.ids.end(),token)-distribution.ids.begin());
    result.confidence=float(distribution.probabilities[mode==VerifyMode::Probabilistic && !options.gate_pick?0:index]);
    if(mode==VerifyMode::Probabilistic) {
        result.row.ids=distribution.ids;result.row.probabilities.reserve(distribution.ids.size());
        for(double p:distribution.probabilities) result.row.probabilities.push_back(float(p));
    }
    return result;
}
}
