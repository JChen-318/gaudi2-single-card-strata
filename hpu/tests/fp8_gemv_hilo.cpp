#include "strata/hpu/host_recipe.hpp"
#include "strata/hpu/model_abi.hpp"
#include "strata/hpu/fp8.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>
using namespace strata::hpu;
int main() {
    try {
        Device device;Stream copy(device);constexpr uint32_t rows=64,experts=11;
        for(uint32_t columns:{256u,640u,2560u}) for(bool shared:{false,true}) {
            const uint32_t padded=(columns+255)/256*256,xrows=shared?1:experts;
            GraphBuilder g(device);std::vector<std::unique_ptr<DeviceBuffer>> resident;std::vector<Binding> bound;
            std::vector<std::vector<uint8_t>> codes(experts);std::vector<std::vector<float>> scales(experts);
            const auto hi=g.tensor({"hi",DType::BF16,{xrows,padded}}),lo=g.tensor({"lo",DType::BF16,{xrows,padded}});
            const auto xs=g.tensor({"scale",DType::F32,{xrows,1}});std::vector<TensorId> ins{hi,lo,xs};
            for(uint32_t e=0;e<experts;++e) {
                const std::string name="weight_"+std::to_string(e);const uint64_t bytes=uint64_t(rows)*columns+rows*4;
                ins.push_back(g.tensor({name,DType::U8,{bytes}}));resident.push_back(std::make_unique<DeviceBuffer>(device,bytes));
                bound.push_back({name,resident.back().get()});HostBuffer staging(device,bytes);auto* raw=static_cast<uint8_t*>(staging.data());
                codes[e].resize(uint64_t(rows)*columns);scales[e].resize(rows);
                for(uint32_t r=0;r<rows;++r) {
                    scales[e][r]=float(1+(r+e)%7)/64;
                    for(uint32_t c=0;c<columns;++c) codes[e][uint64_t(r)*columns+c]=fp8_143_from_f32_reference(float(int((c*13+r*7+e*3)%43)-21)/8);
                }
                std::memcpy(raw,codes[e].data(),codes[e].size());std::memcpy(raw+codes[e].size(),scales[e].data(),rows*4);
                upload(copy,*resident.back(),staging,bytes);copy.synchronize();
            }
            const auto out=g.tensor({"out",DType::F32,{experts,rows}});const TensorId outputs[]={out};
            abi::Fp8Gemv args{int32_t(columns),int32_t(rows),shared?1:0};g.op("strata_fp8_gemv_hilo_f32",ins,outputs,&args,sizeof(args));
            HostRecipe graph(device,g.compile("fp8_gemv_hilo_test"),bound,{"out"});
            for(int phase=0;phase<4;++phase) {
                std::vector<float> high(uint64_t(xrows)*padded),low(high.size()),row_scales(xrows),plain_hi(uint64_t(xrows)*columns),plain_lo(plain_hi.size());
                for(uint32_t r=0;r<xrows;++r) {
                    std::vector<float> x(columns);float peak=0;
                    for(uint32_t c=0;c<columns;++c) {x[c]=phase==0?0:float(int((c*17+r*11)%71)-35)/13*(phase==2?1e-20f:phase==3?1e20f:1);peak=std::max(peak,std::fabs(x[c]));}
                    row_scales[r]=peak*(1.0f/240)+1e-30f;
                    for(uint32_t c=0;c<columns;++c) {
                        const float normalized=x[c]/row_scales[r];const float h=f32_from_fp8_143(fp8_143_from_f32_reference(normalized));
                        const float l=f32_from_fp8_143(fp8_143_from_f32_reference((normalized-h)*16));
                        const auto index=uint64_t(r)*padded+(c/256)*256+(c%2)*128+(c%256)/2;
                        high[index]=h;low[index]=l;plain_hi[uint64_t(r)*columns+c]=h;plain_lo[uint64_t(r)*columns+c]=l;
                    }
                }
                graph.set("hi",high);graph.set("lo",low);graph.set("scale",row_scales);graph.run();const auto actual=graph.output("out");
                double worst=0;
                for(uint32_t e=0;e<experts;++e) for(uint32_t r=0;r<rows;++r) {
                    const auto xr=shared?0:e;double main=0,lower=0;
                    for(uint32_t c=0;c<columns;++c) {const auto w=f32_from_fp8_143(codes[e][uint64_t(r)*columns+c]);main+=double(plain_hi[uint64_t(xr)*columns+c])*w;lower+=double(plain_lo[uint64_t(xr)*columns+c])*w;}
                    const double expected=((main+lower/16)*row_scales[xr])*scales[e][r];const auto value=actual[e*rows+r];
                    const auto error=std::abs(double(value)-expected);worst=std::max(worst,error/std::max(1e-35,std::abs(expected)));
                    if(!std::isfinite(value) || error>1e-5*std::abs(expected)+1e-6*row_scales[xr])
                        throw std::runtime_error("TPC hi/lo versus FP64 oracle mismatch");
                }
                std::cout<<"PASS columns="<<columns<<" shared="<<shared<<" phase="<<phase<<" max_relative_error="<<worst<<'\n';
            }
        }
    } catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
}
