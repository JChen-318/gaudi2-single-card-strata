#include "strata/hpu/layer_flow.hpp"
#include "strata/hpu/session_flow.hpp"
#include <algorithm>
#include <cmath>

namespace strata::hpu {
namespace {
void check(std::span<const float> values,uint64_t elements,const char* stage) {
    if(values.size()!=elements) throw std::invalid_argument(std::string("layer shape mismatch: ")+stage);
    for(float x:values) if(!std::isfinite(x)) throw NonFiniteOutput(std::string("non-finite layer values: ")+stage);
}
}
std::vector<float> execute_layer(std::span<const float> input,uint32_t hidden,uint32_t streams,const LayerOperations& op) {
    if(!hidden || !streams || !op.read || !op.mixer || !op.moe || !op.write || !op.rollback_mixer)
        throw std::invalid_argument("incomplete layer operations/geometry");
    const auto elements=checked_product(hidden,streams);check(input,elements,"input");
    std::vector<float> residual(input.begin(),input.end());bool mixer_complete=false;
    try {
        if(op.ple) {residual=op.ple(residual);check(residual,elements,"PLE");}
        const auto attention=op.read(residual,false);check(attention.mixed,hidden,"HC attention mixed");check(attention.inject,streams,"HC attention inject");
        const auto mixed=op.mixer(attention.mixed);mixer_complete=true;check(mixed,hidden,"GDN/QSA");
        residual=op.write(residual,mixed,attention.inject,false);check(residual,elements,"HC attention write");
        const auto ffn=op.read(residual,true);check(ffn.mixed,hidden,"HC FFN mixed");check(ffn.inject,streams,"HC FFN inject");
        const auto expert=op.moe(ffn.mixed);check(expert,hidden,"MoE");
        residual=op.write(residual,expert,ffn.inject,true);check(residual,elements,"HC FFN write");
        return residual;
    } catch(...) {
        if(mixer_complete) try {op.rollback_mixer();}
        catch(const std::exception& error) {throw SessionFatal(std::string("layer mixer rollback failed: ")+error.what());}
        throw;
    }
}
std::vector<float> execute_layer_window(std::span<const float> input,uint32_t hidden,uint32_t streams,const LayerOperations& op,const std::function<bool()>& cancelled) {
    if(!hidden || !streams || !op.read || !op.mixer || !op.moe || !op.write || !op.rollback_mixer)
        throw std::invalid_argument("incomplete layer window operations/geometry");
    if(bool(op.window_read)!=bool(op.window_write)) throw std::invalid_argument("both window HC callbacks are required");
    if(op.window_mixer && !op.window_read) throw std::invalid_argument("window mixer requires paired window HC");
    if(op.window_ple && !op.window_read) throw std::invalid_argument("window PLE requires paired window HC");
    const auto n=checked_product(hidden,streams);
    if(input.empty() || input.size()%n || input.size()/n>128) throw std::invalid_argument("layer window requires 1..128 complete rows");
    check(input,input.size(),"window input");const auto rows=input.size()/n;
    auto poll=[&] {if(cancelled && cancelled()) throw Cancelled();};
    std::vector<float> residual(input.size()),mixed(rows*hidden),inject(rows*streams);bool changed=false;
    try {
        if(op.window_read) {
            std::copy(input.begin(),input.end(),residual.begin());
            if(op.window_ple) {
                poll();residual=op.window_ple(residual);check(residual,rows*n,"batched PLE");
            } else if(op.ple) for(size_t row=0;row<rows;++row) {
                poll();auto value=op.ple(std::span<const float>(residual).subspan(row*n,n));check(value,n,"batched HC PLE");
                std::copy(value.begin(),value.end(),residual.begin()+row*n);
            }
            poll();auto read=op.window_read(residual,false);check(read.mixed,rows*hidden,"batched HC attention");check(read.inject,rows*streams,"batched attention inject");
            if(op.window_mixer) {
                poll();mixed=op.window_mixer(read.mixed);changed=true;check(mixed,rows*hidden,"batched window mixer");
            } else for(size_t row=0;row<rows;++row) {
                poll();auto value=op.mixer(std::span<const float>(read.mixed).subspan(row*hidden,hidden));changed=true;check(value,hidden,"batched HC causal mixer");
                std::copy(value.begin(),value.end(),mixed.begin()+row*hidden);
            }
            poll();residual=op.window_write(residual,mixed,read.inject,false);check(residual,rows*n,"batched attention write");
            poll();read=op.window_read(residual,true);check(read.mixed,rows*hidden,"batched HC FFN");check(read.inject,rows*streams,"batched FFN inject");
            poll();const auto expert=op.moe(read.mixed);check(expert,rows*hidden,"batched HC grouped MoE");
            poll();residual=op.window_write(residual,expert,read.inject,true);check(residual,rows*n,"batched FFN write");poll();return residual;
        }
        for(size_t row=0;row<rows;++row) {
            poll();std::vector<float> current(input.begin()+row*n,input.begin()+(row+1)*n);
            if(op.ple) {current=op.ple(current);check(current,n,"window PLE");}
            auto read=op.read(current,false);check(read.mixed,hidden,"window HC attention");check(read.inject,streams,"window attention inject");
            poll();auto attention=op.mixer(read.mixed);changed=true;check(attention,hidden,"window mixer");
            current=op.write(current,attention,read.inject,false);check(current,n,"window attention write");
            read=op.read(current,true);check(read.mixed,hidden,"window HC FFN");check(read.inject,streams,"window FFN inject");
            std::copy(current.begin(),current.end(),residual.begin()+row*n);
            std::copy(read.mixed.begin(),read.mixed.end(),mixed.begin()+row*hidden);
            std::copy(read.inject.begin(),read.inject.end(),inject.begin()+row*streams);
        }
        poll();const auto experts=op.moe(mixed);check(experts,rows*hidden,"window grouped MoE");
        for(size_t row=0;row<rows;++row) {
            poll();auto out=op.write(std::span<const float>(residual).subspan(row*n,n),
                std::span<const float>(experts).subspan(row*hidden,hidden),std::span<const float>(inject).subspan(row*streams,streams),true);
            check(out,n,"window FFN write");std::copy(out.begin(),out.end(),residual.begin()+row*n);
        }
        poll();return residual;
    } catch(...) {
        if(changed) try {op.rollback_mixer();}
        catch(const std::exception& error) {throw SessionFatal(std::string("layer window mixer rollback failed: ")+error.what());}
        throw;
    }
}
}
