// Host only: real unchanged original AVX2 NVFP4 kernel, independent float dots.
#include "strata/hpu/expert_scales.hpp"
#include "strata/kernels/cpu/nvfp4_avx2.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include <ggml.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
using namespace strata::hpu;
namespace cpu=strata::kernels::cpu;
namespace {
constexpr uint32_t H=64,T=3;
void require(bool ok,const char* why){if(!ok) throw std::runtime_error(why);}
void close(float actual,double expected,const char* why) {require(std::isfinite(actual) && std::abs(double(actual)-expected)<=std::max(0.00001,std::abs(expected)*0.00005),why);}
}
int main() {
    try {
        if(!cpu::cpu_avx2_ok()) {std::cout<<"SKIP NVFP4 AVX2 scale test: ISA unavailable\n";return 77;}
#if STRATA_HPU_NATIVE_ENTRY_TEST
        cpu::NativeFmt format;std::string error;require(cpu::native_fmt(40,40,H,H,format,error),"actual NVFP4 native format rejected");
        std::array<float,H> bad{};bad[0]=NAN;std::fill(bad.begin()+32,bad.end(),1e-38f);
        std::vector<uint8_t> guarded(format.act_bytes);
        cpu::native_quant_act(format,bad.data(),guarded.data());
        require(guarded[0]==0 && guarded[1]==0x7e && std::all_of(guarded.begin()+34,guarded.end(),[](auto v){return v==0;}),"actual native NVFP4 activation did not guard NaN/tiny Q8 conversion");
        cpu::native_quant_h(format,bad.data(),guarded.data());
        require(guarded[0]==0 && guarded[1]==0x7e,"actual native NVFP4 hidden quantization missed guard");
#endif
        std::mt19937 rng(1221318);std::array<WeightSlice,3> slices;std::array<std::vector<float>,3> decoded;
        for(size_t role=0;role<3;++role) {
            auto raw=std::make_shared<std::vector<uint8_t>>(H*36);for(auto& byte:*raw) byte=uint8_t(rng());
            for(size_t at=0;at<raw->size();at+=36) for(size_t j=0;j<4;++j) (*raw)[at+j]=uint8_t(0x20+(role+j)*3);
            slices[role]=WeightSlice::raw_rows(raw,*raw,40,H,H);decoded[role].resize(H*H);
            ggml_get_type_traits(GGML_TYPE_NVFP4)->to_float(raw->data(),decoded[role].data(),H*H);
        }
        const auto up=H*36,down=up*2,tail=up*3;
        auto blob=native_expert_blob(slices,up,down,tail+16,tail,{0.125f,3.5f,1.0f/32768});
        std::array<float,4> scales;std::memcpy(scales.data(),blob.data()+tail,16);
        const auto qbytes=H/32*34;std::array<std::vector<uint8_t>,T> qx,qh;
        std::array<std::vector<float>,T> x,h,quantized_x,quantized_h,out;
        std::array<const void*,T> xp{},hp{};std::array<float*,T> hidden{},output{};
        for(uint32_t t=0;t<T;++t) {
            x[t].resize(H);h[t].resize(H);out[t].resize(H);quantized_x[t].resize(H);quantized_h[t].resize(H);
            for(auto& v:x[t]) v=float(int(rng()%2001)-1000)/2048;
            qx[t].resize(qbytes);qh[t].resize(qbytes);ggml_quantize_chunk(GGML_TYPE_Q8_0,x[t].data(),qx[t].data(),0,1,H,nullptr);
            ggml_get_type_traits(GGML_TYPE_Q8_0)->to_float(qx[t].data(),quantized_x[t].data(),H);xp[t]=qx[t].data();hidden[t]=h[t].data();output[t]=out[t].data();
        }
        cpu::nvfp4_256_gu_rows(blob.data(),36,up,H,xp.data(),T,hidden.data(),0,H,scales[0],scales[1]);
        for(uint32_t t=0;t<T;++t) {
            for(uint32_t row=0;row<H;++row) {
                double g=0,u=0;for(uint32_t c=0;c<H;++c) {g+=double(decoded[0][row*H+c])*quantized_x[t][c];u+=double(decoded[1][row*H+c])*quantized_x[t][c];}
                g*=scales[0];u*=scales[1];close(h[t][row],g/(1+std::exp(-g))*u,"real original AVX2 NVFP4 gate/up scales differ from independent decoded FP64 dots");
            }
            ggml_quantize_chunk(GGML_TYPE_Q8_0,h[t].data(),qh[t].data(),0,1,H,nullptr);hp[t]=qh[t].data();
            ggml_get_type_traits(GGML_TYPE_Q8_0)->to_float(qh[t].data(),quantized_h[t].data(),H);
        }
        for(float sd:{1.0f,scales[2],4.0f}) {
            cpu::nvfp4_256_rows(blob.data()+down,36,H,hp.data(),T,output.data(),0,H,sd);
            for(uint32_t t=0;t<T;++t) for(uint32_t row=0;row<H;++row) {
                double dot=0;for(uint32_t c=0;c<H;++c) dot+=double(decoded[2][row*H+c])*quantized_h[t][c];
                close(out[t][row],dot*sd,"real original AVX2 NVFP4 down scale differs or changed pre-down activation");
            }
        }
        std::cout<<"PASS actual unchanged AVX2 NVFP4 gate/up/down with native scale tail, three rows and three post-down scales against independent decoded FP64 dots; no SDK/device\n";
    } catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
