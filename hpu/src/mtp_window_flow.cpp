#include "strata/hpu/mtp_window_flow.hpp"
#include "strata/hpu/qsa_state.hpp"
#include "strata/hpu/session_flow.hpp"
#include <algorithm>
#include <cmath>
namespace strata::hpu {
namespace {
void valid(std::span<const float> x,uint64_t count) {
    if(x.size()!=count) throw std::invalid_argument("MTP window shape mismatch");
    for(float v:x) if(!std::isfinite(v)) throw NonFiniteOutput("non-finite MTP window value");
}
}
MtpPairs pair_mtp_prefix(std::span<const float> target,std::span<const float> embedding,std::span<const float> root,uint32_t hidden,uint32_t streams) {
    const auto n=checked_product(hidden,streams);
    if(!hidden || hidden>65536 || !streams || streams>32 || embedding.empty() || embedding.size()%hidden || embedding.size()/hidden>128 || target.size()!=embedding.size()*streams || (!root.empty() && root.size()!=n))
        throw std::invalid_argument("MTP target prefix pair geometry mismatch");
    valid(target,target.size());valid(embedding,embedding.size());valid(root,root.size());
    const size_t start=root.empty()?1:0,rows=embedding.size()/hidden;MtpPairs pairs;
    pairs.residual.reserve((rows-start)*n);pairs.embedding.assign(embedding.begin()+start*hidden,embedding.end());
    if(!root.empty()) pairs.residual.insert(pairs.residual.end(),root.begin(),root.end());
    pairs.residual.insert(pairs.residual.end(),target.begin(),target.begin()+(rows-1)*n);return pairs;
}
MtpOutput execute_mtp_window(std::span<const float> residual,std::span<const float> embedding,uint32_t hidden,uint32_t streams,uint32_t vocabulary,MtpInputNorms norms,const MtpWindowOperations& op,const std::function<bool()>& cancelled) {
    const auto n=checked_product(hidden,streams);
    if(!hidden || hidden>65536 || !streams || streams>32 || !vocabulary || vocabulary>1048576 || embedding.empty() || embedding.size()%hidden || embedding.size()/hidden>128 ||
       residual.size()!=checked_product(embedding.size()/hidden,n) || !op.embedding || !op.hidden || !op.head || !op.rollback)
        throw std::invalid_argument("MTP window requires1..128 complete paired rows and all stages");
    valid(residual,residual.size());valid(embedding,embedding.size());
    if(norms.embedding.size()!=hidden || norms.hidden.size()!=n) throw std::invalid_argument("MTP pre-FC norm shape mismatch");
    auto poll=[&]{if(cancelled && cancelled()) throw Cancelled();};poll();
    const auto rows=embedding.size()/hidden;MtpOutput result;
    try {
        for(size_t row=0;row<rows;) {
            poll();const auto count=std::min<size_t>(kMtpWindowRows,rows-row);
            std::vector<float> e(embedding.begin()+row*hidden,embedding.begin()+(row+count)*hidden),h(residual.begin()+row*n,residual.begin()+(row+count)*n);
            rms_norm_rows(e,hidden,norms.embedding,norms.epsilon);
            for(size_t i=0;i<count;++i) rms_norm_rows(std::span<float>(h).subspan(i*n,n),norms.stream_norm?hidden:uint32_t(n),norms.hidden,norms.epsilon);
            poll();const auto projected_embedding=op.embedding(e);valid(projected_embedding,count*hidden);poll();
            std::vector<float> projected_hidden(h.size());
            for(size_t at=0;at<count*streams;) {
                poll();const auto cells=std::min<size_t>(128,count*streams-at);const auto projected=op.hidden(std::span<const float>(h).subspan(at*hidden,cells*hidden));
                valid(projected,cells*hidden);std::copy(projected.begin(),projected.end(),projected_hidden.begin()+at*hidden);at+=cells;
            }
            std::vector<float> initial(h.size());for(size_t i=0;i<initial.size();++i) initial[i]=projected_hidden[i]+projected_embedding[(i/n)*hidden+i%hidden];
            poll();const auto output=execute_layer_window(initial,hidden,streams,op.layer,cancelled);valid(output,count*n);poll();
            row+=count;if(row==rows) result.residual.assign(output.end()-n,output.end());
        }
        poll();result.logits=op.head(result.residual);
        if(result.logits.size()!=vocabulary) throw std::invalid_argument("MTP final vocabulary shape mismatch");
        bool finite=false;for(float v:result.logits) {if(std::isnan(v) || v==INFINITY) throw NonFiniteOutput("invalid MTP subset head logits");finite|=std::isfinite(v);}
        if(!finite) throw NonFiniteOutput("MTP head has no supported tokens");
        poll();return result;
    } catch(...) {
        try {op.rollback();}
        catch(const std::exception& error) {throw SessionFatal(std::string("MTP window rollback failed: ")+error.what());}
        throw;
    }
}
}
