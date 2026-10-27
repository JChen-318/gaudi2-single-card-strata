#include "strata/hpu/verify.hpp"
#include "strata/core/spec_prob.hpp"
#include "strata/hpu/session_flow.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace strata::hpu {
bool valid_verified_candidates(const VerifiedCandidates& window,std::span<const DraftRow> drafts,
    uint32_t vocabulary,uint32_t limit,std::span<const int32_t> stops) {
    if(window.picks.empty() || window.picks.size()>drafts.size()+1 || window.picks.size()>limit || window.logits.size()!=vocabulary) return false;
    uint32_t retained=0;bool stopped=false;
    for(size_t row=0;row<window.picks.size();++row) {
        const auto pick=window.picks[row];const bool draft=row<drafts.size();
        const bool eos=std::find(stops.begin(),stops.end(),pick.token)!=stops.end();
        if(pick.token<0 || uint32_t(pick.token)>=vocabulary || (pick.accepted && (!draft || pick.token!=drafts[row].token))) return false;
        if(pick.accepted && !eos) ++retained;
        if((eos || !draft || !pick.accepted)!=(row+1==window.picks.size())) return false;
        stopped=eos;
    }
    for(float value:window.logits) if(!std::isfinite(value)) return false;
    return retained==window.retained && stopped==window.stopped && retained<=drafts.size();
}
VerifiedCandidates select_slot_candidates(std::span<const float> after,const DraftRow& draft,int32_t pending,
    std::span<const int32_t> history,const strata::kernels::SamplerParams& sampling,std::span<const int32_t> stops) {
    if(after.empty() || after.size()%2 || pending<0 || size_t(pending)>=after.size()/2 ||
       std::find(stops.begin(),stops.end(),pending)!=stops.end()) throw std::invalid_argument("invalid pending slot window");
    std::vector<int32_t> fed(history.begin(),history.end());fed.push_back(pending);
    return select_verified_candidates(after.first(after.size()/2),after.subspan(after.size()/2),std::span(&draft,1),
                                      fed,sampling,{},VerifyMode::Exact,stops);
}
void validate_drafts(std::span<const DraftRow> rows,uint32_t vocabulary,uint32_t maximum) {
    if(rows.size()>maximum) throw std::invalid_argument("draft source exceeded verify window");
    for(const auto& row:rows) {
        if(row.token<0 || uint32_t(row.token)>=vocabulary || row.ids.size()!=row.probabilities.size() || row.ids.size()>64)
            throw std::invalid_argument("invalid draft token/distribution shape");
        if(row.ids.empty()) continue;
        double sum=0;bool found=false;
        for(size_t i=0;i<row.ids.size();++i) {
            if(row.ids[i]<0 || uint32_t(row.ids[i])>=vocabulary || !std::isfinite(row.probabilities[i]) || row.probabilities[i]<0)
                throw std::invalid_argument("invalid draft distribution entry");
            for(size_t j=0;j<i;++j) if(row.ids[i]==row.ids[j]) throw std::invalid_argument("duplicate draft distribution token");
            sum+=row.probabilities[i];if(row.ids[i]==row.token && row.probabilities[i]>0) found=true;
        }
        if(!found || std::abs(sum-1)>1e-5) throw std::invalid_argument("draft distribution is not normalized or lacks drawn token");
    }
}
VerifyPick verify_pick(std::span<const float> logits,std::span<const int32_t> history,
    const strata::kernels::SamplerParams& p,std::span<const float> bias,const DraftRow* draft,VerifyMode mode) {
    if(mode!=VerifyMode::Exact && mode!=VerifyMode::Probabilistic) throw std::invalid_argument("invalid verify mode");
    if(draft) validate_drafts(std::span(draft,1),uint32_t(logits.size()),1);
    if(mode==VerifyMode::Exact || !draft || p.greedy || p.temperature<=0) {
        const auto token=sample(logits,history,p,bias);return {token,draft && token==draft->token};
    }
    if(p.counter>=core::kSpecCtrDraft) throw std::invalid_argument("probabilistic verify counter overlaps reserved streams");
    const auto target=sampling_distribution(logits,history,p,bias);
    const auto result=core::spec_verify_row(
        {target.ids.data(),target.probabilities.data(),int(target.ids.size())},draft->token,
        {draft->ids.data(),draft->probabilities.data(),draft->ids.empty()?-1:int(draft->ids.size())},
        philox_uniform(p.seed,p.counter|core::kSpecCtrAccept),philox_uniform(p.seed,p.counter|core::kSpecCtrResid));
    return {result.token,bool(result.accepted)};
}
VerifiedCandidates select_verified_candidates(std::span<const float> before,std::span<const float> after,
    std::span<const DraftRow> drafts,std::span<const int32_t> history,const strata::kernels::SamplerParams& parameters,
    std::span<const float> bias,VerifyMode mode,std::span<const int32_t> stops,const std::function<bool()>& cancelled,
    std::span<const int32_t> follow) {
    if(before.empty() || before.size()>INT32_MAX || drafts.empty() || drafts.size()>7 || after.size()!=drafts.size()*before.size() ||
       follow.size()>drafts.size()+1 || drafts.size()>UINT64_MAX-parameters.counter ||
       (mode==VerifyMode::Probabilistic && parameters.counter+drafts.size()>=core::kSpecCtrDraft))
        throw std::invalid_argument("invalid candidate verification geometry/counter");
    validate_drafts(drafts,uint32_t(before.size()),7);
    for(auto values:{before,after}) for(float value:values) if(!std::isfinite(value)) throw NonFiniteOutput("non-finite candidate logits");
    for(auto id:history) if(id<0 || size_t(id)>=before.size()) throw std::invalid_argument("candidate history outside vocabulary");
    for(auto id:stops) if(id<0 || size_t(id)>=before.size()) throw std::invalid_argument("candidate stop outside vocabulary");
    for(auto id:follow) if(id<0 || size_t(id)>=before.size()) throw std::invalid_argument("candidate follow outside vocabulary");
    VerifiedCandidates result;result.picks.reserve(drafts.size()+1);
    std::vector<int32_t> previous(history.begin(),history.end());previous.reserve(previous.size()+drafts.size());
    for(size_t row=0;row<=drafts.size();++row) {
        if(cancelled && cancelled()) throw Cancelled();
        auto p=parameters;p.counter+=row;const auto* draft=row<drafts.size()?&drafts[row]:nullptr;
        const auto values=row?after.subspan((row-1)*before.size(),before.size()):before;
        auto pick=verify_pick(values,previous,p,bias,draft,mode);
        if(!follow.empty()) {
            if(row<follow.size()) {
                result.follow_differ+=pick.token!=follow[row];pick.token=follow[row];
                pick.accepted=draft && pick.token==draft->token;
            }
            ++result.follow_emitted;
        }
        result.picks.push_back(pick);
        result.stopped=std::find(stops.begin(),stops.end(),pick.token)!=stops.end();
        if(result.stopped || !draft || !pick.accepted) break;
        ++result.retained;previous.push_back(pick.token);
    }
    return result;
}
}
