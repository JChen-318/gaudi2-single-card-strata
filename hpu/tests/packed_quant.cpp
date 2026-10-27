#include "strata/hpu/packed_quant.hpp"
#include "strata/kernels/f16_bits.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>

#include "packed_emulator.hpp"
using namespace strata::hpu;
namespace {
void require(bool ok,const char* why){if(!ok) throw std::runtime_error(why);}
template<class F> void rejects(F call){bool rejected=false;try {call();}catch(const std::exception&){rejected=true;}require(rejected,"invalid packed contract accepted");}
template<class T> T read(std::ifstream& stream){T value;stream.read(reinterpret_cast<char*>(&value),sizeof(value));if(!stream) throw std::runtime_error("truncated quant oracle");return value;}
void quant_oracle(const char* path) {
    std::ifstream stream(path,std::ios::binary);const auto cases=read<uint32_t>(stream);require(cases==18,"quant oracle case count changed");
    for(uint32_t c=0;c<cases;++c) {
        const auto columns=read<uint32_t>(stream),count=read<uint32_t>(stream);
        std::vector<float> input(count),metadata(count/16);std::vector<uint8_t> codes(count);
        stream.read(reinterpret_cast<char*>(input.data()),input.size()*4);stream.read(reinterpret_cast<char*>(codes.data()),codes.size());
        stream.read(reinterpret_cast<char*>(metadata.data()),metadata.size()*4);require(bool(stream),"truncated quant oracle payload");
        const auto actual=quantize_q8_activation(input,columns);
        require(actual.codes==codes,"Q8_1 codes differ from independent Python float32/away-round oracle");
        require(actual.metadata.size()==metadata.size() && !std::memcmp(actual.metadata.data(),metadata.data(),metadata.size()*4),"Q8_1 FP16 scale/raw sum differs from independent struct half oracle");
    }
    require(stream.peek()==std::char_traits<char>::eof(),"oracle trailing data");
}
int signed_code(uint8_t q){return q<128?q:int(q)-256;}
void dot_oracle(uint32_t type,uint32_t columns) {
    constexpr uint32_t rows=5,batch=3;const auto row_bytes=packed_row_bytes(type,columns);
    std::vector<uint8_t> weights(row_bytes*rows);std::mt19937 rng(type+columns);
    for(auto& byte:weights) byte=uint8_t(rng());
    const std::array<uint16_t,5> scales{0x3400,0xb000,0x0001,0x3c00,0};
    for(uint32_t row=0;row<rows;++row) for(uint64_t block=0;block<row_bytes;block+=(type==8?34:18)) {
        const auto at=row*row_bytes+block;weights[at]=uint8_t(scales[row]);weights[at+1]=uint8_t(scales[row]>>8);
    }
    validate_packed_rows(weights,type,columns,rows);
    std::vector<float> input(columns*batch);for(auto& x:input) x=float(int(rng()%10000)-5000)/1000;
    auto q=quantize_q8_activation(input,columns);std::vector<float> actual(rows*batch,-INFINITY);
    using namespace packed_emulator;
    tensor w{weights.data(),{row_bytes,rows},1},a{q.codes.data(),{columns,batch},1},m{q.metadata.data(),{2,columns/32,batch},4},o{actual.data(),{rows,batch},4};
    // Disjoint partial index spaces prove kernel row/batch offsets, not just a
    // single all-space invocation. Out-of-bounds scalar accesses throw.
    for(uint32_t t=0;t<batch;++t) for(uint32_t row=0;row<rows;++row) {
        offset={int(row),int(t),0,0,0};extent={1,1,1,1,1};packed_kernel(w,a,m,o,columns,rows,batch,type,0);
    }
    for(uint32_t t=0;t<batch;++t) for(uint32_t row=0;row<rows;++row) {
        long double expected=0;
        for(uint32_t block=0;block<columns/32;++block) {
            const uint64_t at=row*row_bytes+(type==42?block/2*18:block*(type==8?34:18));
            const auto bits=uint16_t(weights[at])|(uint16_t(weights[at+1])<<8);
            const long double d=strata::kernels::f32_from_f16(bits),scale=q.metadata[(t*(columns/32)+block)*2],sum=q.metadata[(t*(columns/32)+block)*2+1];
            int dot=0;
            for(uint32_t j=0;j<32;++j) {
                int code;
                if(type==2) code=(weights[at+2+j%16]>>(j<16?0:4))&15;
                else if(type==8) code=signed_code(weights[at+2+j]);
                else {const auto index=block%2*32+j;code=((weights[at+2+index/4]>>((index%4)*2))&3)-1;}
                dot+=code*signed_code(q.codes[t*columns+block*32+j]);
            }
            expected+=d*(dot*scale-(type==2?8*sum:0));
        }
        require(std::isfinite(actual[t*rows+row]) && std::abs(actual[t*rows+row]-expected)<std::max(0.002L,std::abs(expected)*0.00002L),"TPC packed dot differs from independent long-double format/code/raw-sum oracle");
    }
}
}
int main(int argc,char** argv) {
    if(argc!=2) return 2;
    try {
        quant_oracle(argv[1]);for(auto type:{2u,8u,42u}) for(auto columns:{64u,128u,640u,2560u}) dot_oracle(type,columns);
        require(element_bytes(DType::U8)==1,"packed dtype byte size");
        rejects([]{(void)element_bytes(static_cast<DType>(99));});rejects([]{(void)packed_row_bytes(0,64);});rejects([]{(void)packed_row_bytes(42,32);});
        std::array<float,32> values{};values[0]=NAN;rejects([&]{(void)quantize_q8_activation(values,32);});values[0]=INFINITY;rejects([&]{(void)quantize_q8_activation(values,32);});
        values[0]=std::numeric_limits<float>::denorm_min();rejects([&]{(void)quantize_q8_activation(values,32);});
        std::vector<uint8_t> invalid(34);invalid[0]=0;invalid[1]=0x7c;rejects([&]{validate_packed_rows(invalid,8,32,1);});
        std::cout<<"PASS independent IEEE-half/raw-sum Q8_1 oracle, actual Q2/Q4/Q8 TPC C semantic emulation and typed/bounded invalid inputs\n";
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
