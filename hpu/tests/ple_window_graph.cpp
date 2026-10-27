#include "strata/hpu/hc_ple.hpp"
#include "strata/hpu/ple_window_flow.hpp"
#include "graph_interpreter.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <map>
#include <set>
using namespace strata::hpu;
namespace {
struct Fixture {
    HcConfig config;std::map<std::string,std::vector<float>> weights;std::vector<float> initial;
    Fixture(uint32_t hidden,uint32_t streams):config{hidden,streams,8,1e-6f},initial(9*hidden*streams) {
        const uint64_t h=hidden,n=h*streams;
        weights["w_key"].resize(n*h);weights["w_value"].resize(h*h);weights["conv_weights"].resize(4*n);
        for(const auto* name:{"norm_key","norm_query","norm_conv"}) weights[name].resize(n);
        for(auto& [name,values]:weights) for(size_t i=0;i<values.size();++i) {
            if(name.starts_with("norm")) values[i]=.5f+float(i%11)/16;
            else values[i]=float(int((i*17+name.size()*11)%31)-15)/(name=="conv_weights"?127:77);
        }
        for(size_t i=0;i<initial.size();++i) initial[i]=float(int((i*11+3)%41)-20)/64;
    }
    std::vector<float> project(std::span<const float> x,const char* name) const {
        const auto& w=weights.at(name);const auto h=config.hidden;std::vector<float> result(w.size()/h);
        for(size_t row=0;row<result.size();++row) for(uint32_t col=0;col<h;++col) result[row]+=from_bf16(to_bf16(x[col]))*from_bf16(to_bf16(w[row*h+col]));
        return result;
    }
    std::vector<float> norm(std::span<const float> x,const char* name) const {
        const auto& gamma=weights.at(name);std::vector<float> result(x.size());const auto h=config.hidden;
        for(uint32_t row=0;row<config.streams;++row) {
            double sum=0;for(uint32_t col=0;col<h;++col) {const double value=x[uint64_t(row)*h+col];sum+=value*value;}
            const float inverse=float(1/std::sqrt(sum/h+double(config.epsilon)));
            for(uint32_t col=0;col<h;++col) {const auto i=uint64_t(row)*h+col;result[i]=(x[i]*inverse)*gamma[i];}
        }
        return result;
    }
    PleWindowResult reference(std::span<const float> input,std::span<const float> embedding,std::span<const float> previous) const {
        const auto h=config.hidden;const uint64_t n=uint64_t(h)*config.streams;PleWindowResult out;out.history.assign(previous.begin(),previous.end());out.output.resize(input.size());
        for(size_t row=0;row<input.size()/n;++row) {
            const auto hidden=input.subspan(row*n,n),emb=embedding.subspan(row*h,h);const auto key=norm(project(emb,"w_key"),"norm_key"),query=norm(hidden,"norm_query"),value=project(emb,"w_value");
            std::vector<float> gated(n);
            for(uint32_t stream=0;stream<config.streams;++stream) {
                double dot=0;for(uint32_t col=0;col<h;++col) dot+=double(key[uint64_t(stream)*h+col])*query[uint64_t(stream)*h+col];
                const auto score=dot/std::sqrt(double(h)),signed_root=(score>0?1:score<0?-1:0)*std::sqrt(std::max(std::abs(score),1e-6));
                const float gate=float(1/(1+std::exp(-signed_root)));
                for(uint32_t col=0;col<h;++col) gated[uint64_t(stream)*h+col]=value[col]*gate;
            }
            const auto normalized=norm(gated,"norm_conv");const auto& conv=weights.at("conv_weights");
            for(uint64_t channel=0;channel<n;++channel) {
                float sum=0;for(uint32_t tap=0;tap<4;++tap) sum+=(tap==3?normalized[channel]:out.history[uint64_t(tap)*3*n+channel])*conv[uint64_t(tap)*n+channel];
                out.output[row*n+channel]=(hidden[channel]+gated[channel])+sum*float(1/(1+std::exp(-double(sum))));
            }
            std::move(out.history.begin()+n,out.history.end(),out.history.begin());std::copy(normalized.begin(),normalized.end(),out.history.end()-n);
        }
        return out;
    }
    PleWindowResult graph(std::span<const float> input,std::span<const float> embedding,std::span<const float> previous,bool singleton=false) const {
        GraphBuilder builder(OfflineCompilation{});const auto rows=uint32_t(embedding.size()/config.hidden);
        if(singleton) define_ple_block(builder,config);else define_ple_window(builder,config,rows);auto& g=*oracle::current;
        g.set("hidden",input);g.set("embedding",embedding);g.set("history",previous);for(const auto& [name,values]:weights) g.set(name,values);
        g.run();
        if(!singleton) {
            uint64_t pinned=0;for(const auto& slot:g.slots) if(slot.desc.persistent && !weights.contains(slot.desc.name)) pinned+=slot.desc.bytes();
            oracle::require(pinned==ple_window_pinned_bytes(config.hidden,config.streams,rows),"PLE graph persistent tensors exceed admitted transfer slots");
            oracle::require(g.get("key_matrix").desc.dtype==DType::F32 && g.get("value_matrix").desc.dtype==DType::F32,"PLE projections prematurely rounded to BF16");
        }
        return {g.get("out").values,g.get("history_out").values};
    }
};
void close(std::span<const float> actual,std::span<const float> expected,float tolerance) {
    oracle::require(actual.size()==expected.size(),"PLE graph reference shape mismatch");
    for(size_t i=0;i<actual.size();++i) oracle::require(std::isfinite(actual[i]) && std::abs(actual[i]-expected[i])<=tolerance,"PLE graph lost singleton/TPC/FP64 normalization reference");
}
void run(uint32_t hidden,uint32_t streams,uint32_t count,bool zero_embedding=false) {
    Fixture f(hidden,streams);const uint64_t n=uint64_t(hidden)*streams;std::vector<float> input(count*n),embedding(count*hidden);
    for(size_t i=0;i<input.size();++i) input[i]=float(int((i*11+3)%37)-18)/32;
    if(!zero_embedding) for(size_t i=0;i<embedding.size();++i) embedding[i]=float(int((i*13+7)%31)-15)/41;
    const auto actual=execute_ple_window(input,embedding,f.initial,hidden,streams,[&](auto x,auto e,auto history) {return f.graph(x,e,history);});
    const auto reference=f.reference(input,embedding,f.initial);close(actual.output,reference.output,5e-5f);close(actual.history,reference.history,5e-5f);
    auto history=f.initial;std::vector<float> scalar;
    for(uint32_t row=0;row<count;++row) {
        auto next=f.graph(std::span<const float>(input).subspan(row*n,n),std::span<const float>(embedding).subspan(uint64_t(row)*hidden,hidden),history,true);
        scalar.insert(scalar.end(),next.output.begin(),next.output.end());history=std::move(next.history);
    }
    close(actual.output,scalar,0);close(actual.history,history,0);
    oracle::require(f.initial==Fixture(hidden,streams).initial,"graph execution overwrote accepted history");
}
}
int main() {
    try {
        for(const auto& [hidden,streams]:{std::pair{64u,3u},std::pair{128u,2u}}) {
            for(uint32_t rows:{1u,3u,8u,9u,17u,128u}) run(hidden,streams,rows);
            run(hidden,streams,17,true);
        }
        for(uint32_t rows:{0u,9u}) {
            GraphBuilder g(OfflineCompilation{});bool rejected=false;try {define_ple_window(g,HcConfig{64,3,8,1e-6f},rows);} catch(const std::invalid_argument&) {rejected=true;}
            oracle::require(rejected,"out-of-range PLE native graph accepted");
        }
        std::cout<<"PASS actual PLE singleton/window factories and original TPC C under CPU graph interpreter:14cases, bitwise singleton continuity1..128rows, cap8, FP64 oracle, BF16/F32 boundaries, dilation3/history9, non-power streams and zero embeddings; no SDK/device execution\n";
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
