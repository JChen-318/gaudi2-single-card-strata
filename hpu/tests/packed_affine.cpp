// Encode independent logical codes, then execute the actual production TPC C.
// No runtime, SDK or device. Does not assert native ISA reduction parity.
#include "strata/hpu/packed_quant.hpp"
#include "strata/hpu/expert_format.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "packed_emulator.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
using namespace strata::hpu;
namespace {
void require(bool ok,const char* why){if(!ok) throw std::runtime_error(why);}
template<class F> void rejects(F f){bool failed=false;try{f();}catch(const std::exception&){failed=true;}require(failed,"invalid affine packed payload accepted");}
int signed_code(uint8_t code){return code<128?code:int(code)-256;}
void half(std::vector<uint8_t>& bytes,size_t at,float value){const auto bits=strata::kernels::f16_from_f32(value);bytes[at]=uint8_t(bits);bytes[at+1]=uint8_t(bits>>8);}
std::vector<float> kernel(std::vector<uint8_t>& weights,uint32_t type,uint32_t columns,uint32_t rows,Q8Activation& q) {
    const auto batch=uint32_t(q.codes.size()/columns);std::vector<float> result(batch*rows,NAN);
    using namespace packed_emulator;
    tensor w{weights.data(),{packed_row_bytes(type,columns),rows},1},a{q.codes.data(),{columns,batch},1},m{q.metadata.data(),{2,columns/32,batch},4},o{result.data(),{rows,batch},4};
    for(uint32_t token=0;token<batch;++token) {offset={0,int(token),0,0,0};extent={int(rows),1,1,1,1};packed_kernel(w,a,m,o,columns,rows,batch,type,0);}
    return result;
}
void run(uint32_t type,uint32_t columns,uint32_t batch) {
    // Exhaust all32 independent high bits once per format. Repeating those
    // same rows in every large shape only multiplies scalar-emulator work,
    // especially under ASan. All90 width/batch combinations still execute.
    const uint32_t rows=columns==32 && batch==1?35:5;
    const uint32_t blocks=columns/32,size=type==3?20:type==6?22:24;
    const bool affine=type!=6,high=type!=3;std::vector<uint8_t> weights(uint64_t(rows)*blocks*size);
    std::vector<int> logical(uint64_t(rows)*columns);std::vector<float> scales(rows*blocks),minimum(rows*blocks);
    for(uint32_t row=0;row<rows;++row) for(uint32_t block=0;block<blocks;++block) {
        const auto at=(uint64_t(row)*blocks+block)*size;const float d=float(int(row%7)-3)/16,m=float(int(row%5)-2)/8;
        scales[row*blocks+block]=d;minimum[row*blocks+block]=affine?m:0;half(weights,at,d);if(affine) half(weights,at+2,m);
        const auto qh=at+(affine?4:2),qs=qh+(high?4:0);
        for(uint32_t j=0;j<32;++j) {
            const int code=int((j*7+block*3+row*11)%16)+(high && (row<32?j==row:(j+row+block)%3==0)?16:0);
            logical[uint64_t(row)*columns+block*32+j]=code;
            weights[qs+j%16]|=uint8_t((code&15)<<(j<16?0:4));
            if(high && code>=16) weights[qh+j/8]|=uint8_t(1u<<(j%8));
        }
    }
    validate_packed_rows(weights,type,columns,rows);std::vector<float> input(uint64_t(batch)*columns);
    for(size_t i=0;i<input.size();++i) input[i]=float(int((i*19+batch)%127)-63)*.03125f+.0140625f;
    auto q=quantize_q8_activation(input,columns);const auto actual=kernel(weights,type,columns,rows,q);
    for(uint32_t token=0;token<batch;++token) for(uint32_t row=0;row<rows;++row) {
        long double expected=0;
        for(uint32_t block=0;block<blocks;++block) {
            int dot=0,sum=0;for(uint32_t j=0;j<32;++j) {const auto code=signed_code(q.codes[uint64_t(token)*columns+block*32+j]);dot+=logical[uint64_t(row)*columns+block*32+j]*code;sum+=code;}
            const long double scale=q.metadata[(uint64_t(token)*blocks+block)*2],raw=q.metadata[(uint64_t(token)*blocks+block)*2+1],d=scales[row*blocks+block],m=minimum[row*blocks+block];
            expected+=affine?d*scale*dot+m*scale*sum:d*(scale*dot-16*raw);
        }
        require(std::isfinite(actual[uint64_t(token)*rows+row]) && std::abs(actual[uint64_t(token)*rows+row]-expected)<=std::max(.004L,std::abs(expected)*.00002L),"Q4_1/Q5 packed TPC differs from independent logical-code oracle");
    }
}
void correction_contracts() {
    for(uint32_t type:{3u,7u}) {
        std::vector<uint8_t> weights(packed_block_bytes(type));half(weights,0,0);half(weights,2,1);
        Q8Activation q;q.codes.assign(32,1);q.metadata={.5f,123};const auto actual=kernel(weights,type,32,1,q);
        require(actual[0]==16,"affine learned minimum consumed raw input sum instead of quantized codes");
        for(uint16_t bad:{uint16_t(0x7c00),uint16_t(0x7e00)}) {weights[2]=uint8_t(bad);weights[3]=uint8_t(bad>>8);rejects([&]{validate_packed_rows(weights,type,32,1);});}
    }
    std::vector<uint8_t> weights(22);half(weights,0,1);Q8Activation q;q.codes.assign(32,1);q.metadata={.5f,5};
    require(kernel(weights,6,32,1,q)[0]==-80,"Q5_0 lost original-input Q8_1 raw-sum correction");
    const auto mixed=expert_format(128,64,-1,{3,6,7});
    require(mixed.packed && mixed.bytes==128/32*20*64+128/32*22*64+64/32*24*128,"shared affine expert payload counted as decoded BF16");
    const auto routed=expert_format(128,64,0,{3,6,7});
    require(routed.packed && routed.activation==PackedActivation::RoutedQ8 && routed.bytes==mixed.bytes,"routed affine expert policy/payload mismatch");
    for(uint32_t type:{3u,6u,7u}) rejects([&]{std::vector<uint8_t> short_row(packed_block_bytes(type)-1);validate_packed_rows(short_row,type,32,1);});
}
}
int main(){try {uint32_t cases=0;for(uint32_t type:{3u,6u,7u}) for(uint32_t columns:{32u,64u,128u,640u,2560u}) for(uint32_t batch:{1u,3u,7u,8u,32u,128u}) {run(type,columns,batch);++cases;}correction_contracts();std::cout<<"PASS actual packed affine TPC "<<cases<<" Q4_1/Q5_0/Q5_1 cases, 32 independent high-bit one-hots, logical integer/long-double oracle, raw versus quantized correction counterexamples, finite minima and shared payload budgets; no device\n";}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
