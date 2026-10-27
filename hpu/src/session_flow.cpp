#include "strata/hpu/session_flow.hpp"
#include "strata/hpu/tensor.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <exception>
#include <limits>

namespace strata::hpu {
namespace {
void validate(std::span<const float> values,size_t elements) {
    if(values.size()!=elements) throw std::runtime_error("session stage output shape mismatch");
    if(!all_finite(values)) throw NonFiniteOutput("non-finite session stage output");
}
void check_cancelled(const std::function<bool()>& cancelled) {if(cancelled && cancelled()) throw Cancelled();}
[[noreturn]] void propagate(std::exception_ptr error,bool& poisoned) {
    try {std::rethrow_exception(error);}
    catch(const SessionFatal&) {poisoned=true;throw;}
}
}
SessionFlow::SessionFlow(uint32_t h,uint32_t r,uint32_t l,uint32_t v,uint32_t c,SessionOperations op)
    :hidden_(h),streams_(r),layers_(l),vocabulary_(v),capacity_(c),operations_(std::move(op)) {
    // operations_.ple is optional: its absence is the --no-ple ablation, not an incomplete session.
    if(!h || !r || !l || l>1024 || !v || v>INT32_MAX || !c || uint64_t(h)*r>std::numeric_limits<size_t>::max()/4 ||
       !operations_.embedding || !operations_.checkpoint || !operations_.layer || !operations_.head || !operations_.reset_layer)
        throw std::invalid_argument("incomplete session geometry/operations");
    history_.reserve(c); // Token commit below cannot allocate after layer state commits.
}
std::vector<float> SessionFlow::decode(int32_t token,const std::function<bool()>& cancelled) {
    if(poisoned_) throw std::logic_error("session rollback failed; reset required");
    if(token<0 || uint32_t(token)>=vocabulary_ || history_.size()>=capacity_) throw std::invalid_argument("session token/capacity mismatch");
    check_cancelled(cancelled);
    // STRATA_HPU_DECODE_TIMING=1 (diagnostic): the host phases of a decoded token, averaged every 64 tokens.
    static const bool timing=[] {const char* v=std::getenv("STRATA_HPU_DECODE_TIMING");return v && v[0]=='1';}();
    using Clock=std::chrono::steady_clock;static std::array<double,5> spent{};static uint64_t tokens=0;
    auto mark=Clock::now();
    const auto lap=[&](size_t i) {if(!timing) return;const auto now=Clock::now();spent[i]+=std::chrono::duration<double,std::milli>(now-mark).count();mark=now;};
    auto embedded=operations_.embedding(token);validate(embedded,hidden_);lap(0);
    std::vector<float> ple;
    if(operations_.ple) {ple=operations_.ple(token,history_);validate(ple,hidden_);}
    lap(1);
    std::vector<float> residual(size_t(hidden_)*streams_);
    for(uint32_t stream=0;stream<streams_;++stream) std::copy(embedded.begin(),embedded.end(),residual.begin()+size_t(stream)*hidden_);
    std::vector<std::function<void()>> rewind;rewind.reserve(layers_);
    size_t completed=0;
    try {
        if(operations_.chain && !operations_.after_layer) {
            SessionOperations::ChainHooks hooks;
            hooks.begin=[&](uint32_t layer) {
                if(layer!=rewind.size() || layer>=layers_) throw std::logic_error("layer chain out of order");
                check_cancelled(cancelled);
                auto point=operations_.checkpoint(layer);
                if(!point) throw std::logic_error("empty layer rewind closure");
                rewind.push_back(std::move(point));
            };
            hooks.done=[&](uint32_t layer) {if(layer!=completed || layer+1!=rewind.size()) throw std::logic_error("layer chain commit out of order");++completed;};
            residual=operations_.chain(residual,ple,hooks);lap(2);
            if(completed!=layers_) throw std::logic_error("layer chain skipped layers");
            validate(residual,size_t(hidden_)*streams_);
        } else for(uint32_t layer=0;layer<layers_;++layer) {
            check_cancelled(cancelled);
            auto point=operations_.checkpoint(layer);
            if(!point) throw std::logic_error("empty layer rewind closure");
            rewind.push_back(std::move(point));
            residual=operations_.layer(layer,residual,layer==1?std::span<const float>(ple):std::span<const float>{});
            ++completed;validate(residual,size_t(hidden_)*streams_);
            if(operations_.after_layer) {residual=operations_.after_layer(layer,residual);validate(residual,size_t(hidden_)*streams_);}
        }
        check_cancelled(cancelled);
        auto logits=operations_.head(residual);validate(logits,vocabulary_);lap(3);
        if(timing && ++tokens%64==0) {
            std::fprintf(stderr,"strata-hpu: decode host ms/token embedding=%.3f ple=%.3f chain=%.3f head=%.3f\n",
                         spent[0]/64,spent[1]/64,spent[2]/64,spent[3]/64);
            spent={};
        }
        check_cancelled(cancelled);history_.push_back(token);return logits;
    } catch(...) {
        const auto error=std::current_exception();bool failed=false;
        while(completed) {--completed;try {rewind[completed]();} catch(...) {failed=true;}}
        if(failed) {poisoned_=true;throw SessionFatal("session rollback failed; reset required");}
        propagate(error,poisoned_);
    }
}
std::vector<float> SessionFlow::prefill(std::span<const int32_t> tokens,const std::function<bool()>& cancelled) {
    if(tokens.empty() || tokens.size()>capacity_-history_.size()) throw std::invalid_argument("empty/oversized prefill");
    for(auto token:tokens) if(token<0 || uint32_t(token)>=vocabulary_) throw std::invalid_argument("prefill contains invalid token");
    std::vector<float> result;
    for(auto token:tokens) result=decode(token,cancelled);
    return result;
}
std::vector<float> SessionFlow::prefill_window(std::span<const int32_t> tokens,const std::function<bool()>& cancelled) {
    return run_window(tokens,{},cancelled).logits;
}
SessionVerifyResult SessionFlow::verify_window(std::span<const int32_t> tokens,uint64_t budget,const CandidateSelection& select,const std::function<bool()>& cancelled) {
    if(!select || tokens.empty() || tokens.size()>7 ||
       uint64_t(vocabulary_)*tokens.size()>budget/sizeof(float) ||
       uint64_t(vocabulary_)*tokens.size()>std::numeric_limits<size_t>::max()/sizeof(float))
        throw std::invalid_argument("candidate window requires1..7 rows and a sufficient explicit logits budget");
    return run_window(tokens,select,cancelled);
}
SessionVerifyResult SessionFlow::run_window(std::span<const int32_t> tokens,const CandidateSelection& select,const std::function<bool()>& cancelled) {
    if(poisoned_ || !operations_.window_layer || tokens.empty() || tokens.size()>128 || tokens.size()>capacity_-history_.size())
        throw std::invalid_argument("session window state/geometry/operations mismatch");
    for(auto token:tokens) if(token<0 || uint32_t(token)>=vocabulary_) throw std::invalid_argument("window token out of vocabulary");
    const auto n=size_t(hidden_)*streams_,rows=tokens.size();
    std::vector<float> residual(rows*n),embeddings(rows*hidden_),ple(operations_.ple?rows*hidden_:0);
    std::vector<float> candidate_logits(select?rows*vocabulary_:0);
    std::vector<int32_t> previous(history_);previous.reserve(history_.size()+rows);
    for(size_t row=0;row<rows;++row) {
        check_cancelled(cancelled);auto embedded=operations_.embedding(tokens[row]);validate(embedded,hidden_);
        if(operations_.ple) {
            auto gathered=operations_.ple(tokens[row],previous);validate(gathered,hidden_);
            std::copy(gathered.begin(),gathered.end(),ple.begin()+row*hidden_);
        }
        previous.push_back(tokens[row]);
        std::copy(embedded.begin(),embedded.end(),embeddings.begin()+row*hidden_);
        for(uint32_t stream=0;stream<streams_;++stream) std::copy(embedded.begin(),embedded.end(),residual.begin()+row*n+stream*hidden_);
    }
    std::vector<std::function<void()>> rewind;rewind.reserve(layers_);size_t completed=0;
    auto rollback=[&] {
        bool failed=false;
        while(completed) {--completed;try {rewind[completed]();} catch(...) {failed=true;}}
        if(failed) {poisoned_=true;throw SessionFatal("session window rollback failed; reset required");}
    };
    try {
        for(uint32_t layer=0;layer<layers_;++layer) {
            check_cancelled(cancelled);auto point=operations_.checkpoint(layer);
            if(!point) throw std::logic_error("empty window layer rewind closure");
            rewind.push_back(std::move(point));
            residual=operations_.window_layer(layer,residual,embeddings,layer==1?std::span<const float>(ple):std::span<const float>{},cancelled);
            ++completed;validate(residual,rows*n);
            if(operations_.after_layer) for(size_t row=0;row<rows;++row) {
                check_cancelled(cancelled);auto value=operations_.after_layer(layer,std::span<const float>(residual).subspan(row*n,n));validate(value,n);
                std::copy(value.begin(),value.end(),residual.begin()+row*n);
            }
        }
        std::vector<float> logits;
        if(select) {
            if(operations_.window_head) {
                check_cancelled(cancelled);auto output=operations_.window_head(residual,cancelled);validate(output,rows*vocabulary_);
                std::copy(output.begin(),output.end(),candidate_logits.begin());
                logits.assign(output.end()-vocabulary_,output.end());
            } else for(size_t row=0;row<rows;++row) {
                check_cancelled(cancelled);logits=operations_.head(std::span<const float>(residual).subspan(row*n,n));validate(logits,vocabulary_);
                std::copy(logits.begin(),logits.end(),candidate_logits.begin()+row*vocabulary_);
            }
            check_cancelled(cancelled);const auto accepted=select(candidate_logits);
            if(accepted>rows) throw std::invalid_argument("candidate selector retained more than the proposed window");
            check_cancelled(cancelled);
            if(accepted<rows) {
                // Lightweight GDN checkpoints become stale after restore.
                // Consume these roots once, then let replay take fresh roots
                // so a replay failure can independently restore its start.
                rollback();rewind.clear();
                if(!accepted) return {};
                return {accepted,prefill_window(tokens.first(accepted),cancelled)};
            }
        } else {
            check_cancelled(cancelled);logits=operations_.head(std::span<const float>(residual).last(n));validate(logits,vocabulary_);
        }
        check_cancelled(cancelled);history_.insert(history_.end(),tokens.begin(),tokens.end());return {uint32_t(rows),std::move(logits)};
    } catch(...) {
        const auto error=std::current_exception();rollback();
        propagate(error,poisoned_);
    }
}
void SessionFlow::reset() {
    poisoned_=true;
    // Even when one reset fails, try every other layer before returning.
    std::exception_ptr error;
    for(uint32_t layer=0;layer<layers_;++layer) try {operations_.reset_layer(layer);} catch(...) {if(!error) error=std::current_exception();}
    if(error) std::rethrow_exception(error);
    history_.clear();poisoned_=false;
}
void SessionFlow::restore_history(std::span<const int32_t> tokens) {
    if(tokens.size()>capacity_) throw std::invalid_argument("snapshot history exceeds capacity");
    for(auto token:tokens) if(token<0 || uint32_t(token)>=vocabulary_) throw std::invalid_argument("invalid snapshot token ID");
    history_.assign(tokens.begin(),tokens.end());poisoned_=false;
}
}
