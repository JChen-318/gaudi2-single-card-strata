// Host-only comparison against unchanged original AVX2 quantizer and Q2 rows.
#include "strata/hpu/packed_quant.hpp"
#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "packed_emulator.hpp"
#include <cstring>
#include <iostream>
#include <memory>
#include <random>
using namespace strata::hpu;
namespace cpu=strata::kernels::cpu;
namespace {
void require(bool ok,const char* why){if(!ok) throw std::runtime_error(why);}
#ifndef NDEBUG
// Debug's original scalar correction may not contract to FMA. Bound that
// build-dependent difference using the absolute products, not a relative
// tolerance on a cancellation-small output. Release below requires equality.
double correction_bound(std::span<const uint8_t> row,const Q8Activation& activation,uint32_t columns,uint32_t token) {
    double magnitude=0;
    for(uint32_t block=0;block<columns/64;++block) {
        const uint16_t bits=uint16_t(row[block*18])|(uint16_t(row[block*18+1])<<8);
        const double weight=std::abs(strata::kernels::f32_from_f16(bits));
        for(uint32_t j=0;j<64;++j) {
            const int code=(row[block*18+2+j/4]>>(2*(j%4)))&3;
            const auto raw=activation.codes[uint64_t(token)*columns+block*64+j];const int q=raw<128?raw:int(raw)-256;
            magnitude+=weight*std::abs(activation.metadata[(uint64_t(token)*(columns/32)+block*2+j/32)*2])*std::abs(code*q);
        }
        for(uint32_t half=0;half<2;++half) magnitude+=weight*std::abs(activation.metadata[(uint64_t(token)*(columns/32)+block*2+half)*2+1]);
    }
    const double gamma=(2*(columns/64)+8)*std::numeric_limits<float>::epsilon();
    return std::max(0.0002,2*gamma*magnitude);
}
#endif
void compare(uint32_t columns,uint32_t batch,uint32_t corpus,double& peak) {
    constexpr uint32_t rows=7;std::mt19937 rng(columns+batch+corpus*318);
    std::vector<float> input(columns*batch);
    for(size_t i=0;i<input.size();++i) {
        float x=float(int(rng()%20001)-10000)/2048;
        if(corpus==0) x=0;
        if(corpus==1) x=i%32==0?127.0f:float(int(i%9)-4)*0.5f;
        if(corpus==3) x*=1e-5f;
        if(corpus==4) x*=1e8f;
        input[i]=x;
    }
    auto activation=quantize_s2_activation(input,columns);
    std::vector<std::unique_ptr<cpu::ActQ>> native;std::vector<const cpu::ActQ*> pointers;
    for(uint32_t t=0;t<batch;++t) {
        auto q=std::make_unique<cpu::ActQ>();cpu::act_quant_q8_1_avx2(input.data()+t*columns,columns,*q);
        require(!std::memcmp(q->q,activation.codes.data()+t*columns,columns),"HPU host routed codes differ from original AVX2 ActQ");
        for(uint32_t c=0;c<columns/32;++c) {
            const auto at=(uint64_t(t)*(columns/32)+c)*2;
            require(std::bit_cast<uint32_t>(q->scale[c])==std::bit_cast<uint32_t>(activation.metadata[at]) &&
                    std::bit_cast<uint32_t>(q->hx[c])==std::bit_cast<uint32_t>(activation.metadata[at+1]),"HPU FP32 routed scale/hx differ from original AVX2 ActQ");
        }
        pointers.push_back(q.get());native.push_back(std::move(q));
    }
    const auto row_bytes=packed_row_bytes(42,columns);std::vector<uint8_t> weights(row_bytes*rows);
    for(auto& byte:weights) byte=uint8_t(rng());
    for(uint32_t row=0;row<rows;++row) for(uint64_t at=row*row_bytes;at<(row+1)*row_bytes;at+=18) {
        const auto scale=strata::kernels::f16_from_f32(float(int(rng()%31)-15)/512);
        weights[at]=uint8_t(scale);weights[at+1]=uint8_t(scale>>8);
    }
    std::vector<float> expected(rows*batch),actual(expected.size());std::vector<float*> output;
    for(uint32_t t=0;t<batch;++t) output.push_back(expected.data()+t*rows);
    cpu::q2_0_gguf_rows_multi_avx2_legacy(weights.data(),row_bytes,columns/64,pointers.data(),batch,output.data(),0,rows);
    using namespace packed_emulator;
    tensor w{weights.data(),{row_bytes,rows},1},a{activation.codes.data(),{columns,batch},1};
    tensor m{activation.metadata.data(),{2,columns/32,batch},4},o{actual.data(),{rows,batch},4};
    offset={0,0,0,0,0};extent={int(rows),int(batch),1,1,1};packed_kernel(w,a,m,o,columns,rows,batch,42,1);
    for(size_t i=0;i<actual.size();++i) {
        const double error=std::abs(double(actual[i])-expected[i]);peak=std::max(peak,error);
#ifdef NDEBUG
        const double allowed=0;
#else
        const auto row=i%rows,token=i/rows;
        const double allowed=correction_bound(std::span<const uint8_t>(weights).subspan(row*row_bytes,row_bytes),activation,columns,uint32_t(token));
#endif
        if(!(std::isfinite(actual[i]) && error<=allowed))
            std::cerr<<"routed parity K="<<columns<<" T="<<batch<<" corpus="<<corpus<<" element="<<i<<" actual="<<actual[i]<<" expected="<<expected[i]<<" error="<<error<<" allowed="<<allowed<<'\n';
        require(std::isfinite(actual[i]) && error<=allowed,"actual TPC C routed dot differs from unchanged original AVX2 Q2 row kernel");
    }
}
}
int main() {
    try {
        if(!cpu::cpu_avx2_ok()) {std::cout<<"SKIP original AVX2 routed parity: ISA unavailable\n";return 77;}
        double peak=0;for(auto columns:{64u,640u,2560u}) for(auto batch:{1u,3u}) for(uint32_t corpus=0;corpus<5;++corpus) compare(columns,batch,corpus,peak);
        std::cout<<"PASS 30 actual AVX2 ActQ/row cases: routed codes/FP32 scale/hx bit-exact on CPU; TPC C dots "
#ifdef NDEBUG
                 <<"equal Release AVX2"
#else
                 <<"within condition-aware Debug nonfused-correction bound"
#endif
                 <<", max_abs="<<peak<<" (large-value corpus included; no HPU execution)\n";
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
