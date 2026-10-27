#include "strata/hpu/host_recipe.hpp"
#include "strata/hpu/model_abi.hpp"
#include "strata/hpu/fp8.hpp"
#include "strata/hpu/fp8_colpair.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
using namespace strata::hpu;
int main() {
    try {
        Device device;Stream copy(device);constexpr uint32_t experts=11;
        for(const auto& [columns,rows]:{std::pair{256u,128u},std::pair{2560u,1280u},std::pair{640u,2560u}}) for(bool shared:{true,false}) for(int phase=0;phase<4;++phase) {
            const uint32_t padded=(columns+255)/256*256,xrows=shared?1:experts;
            std::vector<float> plain(uint64_t(xrows)*columns),permuted(uint64_t(xrows)*padded);
            for(uint32_t e=0;e<xrows;++e) for(uint32_t c=0;c<columns;++c) {
                const float v=phase==0?0:phase==1?float(c==(e*11+17)%columns):
                    phase==2?float(int((c*17+e*11)%71)-35)/13:
                    std::ldexp(float(int((c*17+e*11)%71)-35)/13,int((c*7+e*3)%41)-20);
                const auto value=from_bf16(to_bf16(v));
                plain[uint64_t(e)*columns+c]=value;
                permuted[uint64_t(e)*padded+(c/256)*256+(c%2)*128+(c%256)/2]=value;
            }
            std::vector<std::vector<uint8_t>> codes(experts);std::vector<std::vector<float>> scales(experts);
            std::vector<std::unique_ptr<DeviceBuffer>> resident;
            std::vector<std::unique_ptr<HostRecipe>> graphs;
            for(bool paired:{false,true}) {
                GraphBuilder g(device);std::vector<Binding> bound;
                auto x=g.tensor({"x",DType::BF16,{xrows,paired?columns:padded}});std::vector<TensorId> in{x};
                for(uint32_t e=0;e<experts;++e) {
                    const std::string name="weight_"+std::to_string(e);const uint64_t count=uint64_t(rows)*columns,bytes=count+rows*4;
                    in.push_back(g.tensor({name,DType::U8,{bytes}}));resident.push_back(std::make_unique<DeviceBuffer>(device,bytes));
                    bound.push_back({name,resident.back().get()});HostBuffer staging(device,bytes);auto* raw=static_cast<uint8_t*>(staging.data());
                    if(!paired) {
                        codes[e].resize(count);scales[e].resize(rows);
                        for(uint32_t r=0;r<rows;++r) {
                            scales[e][r]=float(1+(r+e)%7)/64;
                            for(uint32_t c=0;c<columns;++c) codes[e][uint64_t(r)*columns+c]=fp8_143_from_f32_reference(std::ldexp(float(int((c*13+r*7+e*3)%43)-21)/8,phase==3?int((c+r*3+e)%13)-6:0));
                        }
                        std::memcpy(raw,codes[e].data(),count);
                    } else {std::memcpy(raw,codes[e].data(),count);pack_fp8_colpair({raw,size_t(count)},rows,columns);}
                    std::memcpy(raw+count,scales[e].data(),rows*4);upload(copy,*resident.back(),staging,bytes);copy.synchronize();
                }
                const auto out=g.tensor({"out",DType::F32,{experts,rows}});const TensorId outs[]={out};
                abi::Fp8Gemv args{int32_t(columns),int32_t(rows),shared?1:0};
                g.op(paired?"strata_fp8_gemv_colpair_f32":"strata_fp8_gemv_f32",in,outs,&args,sizeof(args));
                graphs.push_back(std::make_unique<HostRecipe>(device,g.compile(paired?"colpair":"rowmajor"),bound,std::vector<std::string>{"out"}));
                graphs.back()->set("x",paired?plain:permuted);graphs.back()->run();
            }
            double worst=0;size_t bad=0;
            const auto old=graphs[0]->output("out"),actual=graphs[1]->output("out");
            for(uint32_t e=0;e<experts;++e) for(uint32_t r=0;r<rows;++r) {
                double sum=0,magnitude=0;
                for(uint32_t c=0;c<columns;++c) {
                    const double term=double(plain[uint64_t(shared?0:e)*columns+c])*f32_from_fp8_143(codes[e][uint64_t(r)*columns+c]);
                    sum+=term;magnitude+=std::abs(term);
                }
                const double expected=sum*scales[e][r],error=std::abs(double(actual[e*rows+r])-expected);
                worst=std::max(worst,error/std::max(1e-35,magnitude*scales[e][r]));
                if(!std::isfinite(actual[e*rows+r]) || error>2e-5*magnitude*scales[e][r]+1e-6) {
                    if(bad++<8) std::cout<<"mismatch "<<e<<' '<<r<<" expected="<<expected<<" actual="<<actual[e*rows+r]<<" old="<<old[e*rows+r]<<'\n';
                }
            }
            if(bad) throw std::runtime_error("column-pair GEMV versus FP64 mismatch");
            std::cout<<"PASS columns="<<columns<<" rows="<<rows<<" shared="<<shared<<" phase="<<phase<<" normalized_error="<<worst<<'\n';
            if(phase==2) for(size_t mode=0;mode<graphs.size();++mode) {
                for(int i=0;i<10;++i) graphs[mode]->run();
                std::vector<double> samples;
                for(int i=0;i<101;++i) {
                    const auto begin=std::chrono::steady_clock::now();graphs[mode]->run();
                    samples.push_back(std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-begin).count());
                }
                std::sort(samples.begin(),samples.end());std::cout<<"wall_us_median mode="<<mode<<' '<<samples[50]<<" p90="<<samples[90]<<'\n';
            }

        }
    } catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
}
