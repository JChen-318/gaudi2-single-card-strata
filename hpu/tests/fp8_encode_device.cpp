#include "strata/hpu/host_recipe.hpp"
#include "strata/hpu/model_abi.hpp"
#include "strata/hpu/fp8.hpp"
#include <perf_lib_layer_params.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>
using namespace strata::hpu;
int main() {
    try {
        Device device;
        for(uint32_t columns:{64u,128u,640u,2560u,65536u}) for(uint32_t rows:{1u,11u}) {
            GraphBuilder g(device);
            const auto t=[&](const char* name,DType dtype,std::vector<uint64_t> shape,bool persist=false) {return g.tensor({name,dtype,std::move(shape),persist});};
            const auto x=t("x",DType::F32,{rows,columns},true);
            const auto hi=t("hi",DType::F8,{rows,columns},true),lo=t("lo",DType::F8,{rows,columns});
            const auto scale=t("scale",DType::F32,{rows,1},true);
            abi::Fp8Encode args{int32_t(columns),int32_t(rows)};const TensorId ins[]={x},outs[]={hi,lo,scale};
            g.op("strata_fp8_encode_f32",ins,outs,&args,sizeof(args));
            const auto hf=t("hf",DType::F32,{rows,columns},true),lf=t("lf",DType::F32,{rows,columns},true);
            g.cast(hi,hf);g.cast(lo,lf);
            HostRecipe recipe(device,g.compile("fp8_encode_device"),{}, {"hf","lf","scale"},uint64_t(64)<<20);
            std::vector<float> values(uint64_t(rows)*columns);
            for(int phase=0;phase<5;++phase) {
                for(uint32_t row=0;row<rows;++row) for(uint32_t col=0;col<columns;++col) {
                    const float base=float(int((col*17+row*11)%73)-36)/13;
                    values[uint64_t(row)*columns+col]=phase==0?0:phase==1?base:phase==2?base*1e-30f:phase==3?base*1e30f:
                        std::ldexp(float(int(col%31)-15),int(col%13)-6);
                }
                recipe.set("x",values);recipe.run();
                const auto high=recipe.output("hf"),low=recipe.output("lf"),scales=recipe.output("scale");
                double worst=0;size_t hi_different=0;
                for(uint32_t row=0;row<rows;++row) {
                    float peak=0;for(uint32_t col=0;col<columns;++col) peak=std::max(peak,std::fabs(values[uint64_t(row)*columns+col]));
                    const float expected_scale=peak*(1.0f/240)+1e-30f;
                    if(!std::isfinite(scales[row]) || std::abs(double(scales[row])-expected_scale)>1e-36+2e-6*expected_scale)
                        throw std::runtime_error("FP8 scale mismatch");
                    for(uint32_t col=0;col<columns;++col) {
                        const size_t i=uint64_t(row)*columns+col;const float normalized=values[i]/expected_scale;
                        const float eh=f32_from_fp8_143(fp8_143_from_f32_reference(normalized));
                        const float el=f32_from_fp8_143(fp8_143_from_f32_reference((normalized-eh)*16));
                        hi_different+=high[i]!=eh;
                        const double actual=high[i]+low[i]/16,expected=eh+el/16;
                        const double error=std::abs(actual-expected);worst=std::max(worst,error);
                        // TPC reciprocal multiplication and CPU division can land on opposite
                        // sides of an FP8 midpoint. The complementary low code then reverses
                        // sign: e.g. -3.25 - 2/16 == -3.5 + 2/16. Check the represented value,
                        // allowing half a low-code subnormal plus F32 division roundoff.
                        if(!std::isfinite(actual) || error>1.0/8192+2e-6*std::abs(expected))
                            throw std::runtime_error("FP8 hi/lo reconstruction mismatch at row="+std::to_string(row)+" col="+std::to_string(col)+" actual="+std::to_string(actual)+" expected="+std::to_string(expected)+" hi="+std::to_string(high[i])+" expected_hi="+std::to_string(eh)+" lo="+std::to_string(low[i])+" expected_lo="+std::to_string(el)+" scale="+std::to_string(scales[row]));
                    }
                }
                std::cout<<"PASS columns="<<columns<<" rows="<<rows<<" phase="<<phase<<" reconstruction_error="<<worst<<" high_code_differences="<<hi_different<<'\n';
            }
        }
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
