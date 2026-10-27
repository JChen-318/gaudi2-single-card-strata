#include "strata/hpu/host_recipe.hpp"
#include "strata/hpu/model_abi.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>
using namespace strata::hpu;

int main(int argc,char** argv) {
    try {
        const bool scalar=argc==2 && std::string(argv[1])=="--scalar-reference";
        if(argc>1 && !scalar) throw std::invalid_argument("expected --scalar-reference or no arguments");
        Device device;
        for(uint32_t width:{64u,128u,2560u,65536u}) for(int mode:{0,1,2}) {
            const uint32_t rows=6,gamma_rows=mode==0?rows:mode==1?1:3;
            GraphBuilder g(device);
            const auto x=g.tensor({"x",DType::F32,{rows,width}});
            const auto gamma=g.tensor({"gamma",DType::F32,mode==1?std::vector<uint64_t>{width}:std::vector<uint64_t>{gamma_rows,width}});
            const auto out=g.tensor({"out",DType::F32,{rows,width}});
            abi::RowNorm p{int(width),int(rows),mode==0?0:mode==1?1:-3,1e-6f};
            const TensorId ins[]={x,gamma},outs[]={out};
            g.op(scalar?"strata_row_norm_f32":"strata_row_norm_simd_f32",ins,outs,&p,sizeof(p));
            HostRecipe recipe(device,g.compile("simd_norm_parity"),{}, {"out"},uint64_t(64)<<20);
            std::vector<float> values(uint64_t(rows)*width),weights(uint64_t(gamma_rows)*width);
            for(size_t i=0;i<weights.size();++i) weights[i]=float(int(i%23)-11)/8;
            for(int phase:{0,1,2,3}) {
                const float scale=phase==0?0.0f:phase==1?1.0f:phase==2?1e30f:1e-30f;
                for(size_t i=0;i<values.size();++i) values[i]=float(int((i*17+3)%71)-35)*scale;
                recipe.set("x",values);recipe.set("gamma",weights);recipe.run();
                const auto actual=recipe.output("out");double worst=0;
                for(uint32_t row=0;row<rows;++row) {
                    double sum=0;
                    for(uint32_t col=0;col<width;++col) {const double v=values[uint64_t(row)*width+col];sum+=v*v;}
                    const double inverse=1/std::sqrt(sum/width+double(p.epsilon));
                    for(uint32_t col=0;col<width;++col) {
                        const size_t i=uint64_t(row)*width+col;
                        const double expected=(double(values[i])*inverse)*weights[uint64_t(row%gamma_rows)*width+col];
                        const double error=std::abs(double(actual[i])-expected);
                        if(!std::isfinite(actual[i]) || error>1e-35+3e-5*std::abs(expected))
                            throw std::runtime_error("SIMD RMS versus independent FP64 reference mismatch");
                        worst=std::max(worst,error);
                    }
                }
                std::cout<<"PASS width="<<width<<" gamma_mode="<<mode<<" phase="<<phase<<" max_abs_error="<<worst<<'\n';
            }
        }
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
