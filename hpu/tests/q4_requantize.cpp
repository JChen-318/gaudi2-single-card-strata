#include "strata/hpu/q4_requantize.hpp"
#include "strata/hpu/weights.hpp"
#include "strata/artifact/dequant.hpp"
#include <ggml.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
using namespace strata::hpu;
namespace {
void require(bool ok,const char* why) {if(!ok) throw std::runtime_error(why);}
template<class F> void rejects(F call) {bool bad=false;try {call();}catch(const std::exception&){bad=true;}require(bad,"unsafe Q4 requantization accepted");}
template<class T> T read(std::ifstream& stream) {T v;stream.read(reinterpret_cast<char*>(&v),sizeof(v));if(!stream) throw std::runtime_error("truncated Q4 oracle");return v;}
void oracle(const char* path) {
    std::ifstream file(path,std::ios::binary);const auto count=read<uint32_t>(file);require(count==18,"Q4 oracle count");
    for(uint32_t c=0;c<count;++c) {
        const auto columns=read<uint32_t>(file),size=read<uint32_t>(file);
        auto values=std::make_shared<std::vector<float>>(size);std::vector<uint8_t> expected(size/32*18),actual(expected.size());
        file.read(reinterpret_cast<char*>(values->data()),size*4);file.read(reinterpret_cast<char*>(expected.data()),expected.size());require(bool(file),"Q4 oracle payload");
        quantize_q4_rows(*values,columns,actual);require(actual==expected,"ggml Q4 bytes differ from independent float32/half oracle");
        auto mapped=WeightSlice::raw_rows(values,{reinterpret_cast<const uint8_t*>(values->data()),size*4},0,columns,size/columns);
        values.reset();std::fill(actual.begin(),actual.end(),0xcd);std::vector<float> scratch(columns);
        mapped.convert_rows_q4(0,actual,scratch);require(actual==expected,"bounded owning-slice Q4 conversion changed rows or lifetime");
        for(uint32_t row=0;row<mapped.rows();++row) {
            auto destination=std::span(actual).subspan(uint64_t(row)*columns/32*18,columns/32*18);
            mapped.convert_rows_q4(row,destination,scratch);require(std::equal(destination.begin(),destination.end(),expected.begin()+uint64_t(row)*columns/32*18),"selected Q4 row order changed");
        }
        rejects([&]{mapped.convert_rows_q4(mapped.rows(),actual,scratch);});
        rejects([&]{mapped.convert_rows_q4(0,std::span(actual).first(actual.size()-1),scratch);});
        rejects([&]{mapped.convert_rows_q4(0,actual,std::span(scratch).first(columns-1));});
    }
    require(file.peek()==std::char_traits<char>::eof(),"Q4 oracle trailing data");
}
void formats() {
    for(uint32_t type:{0u,1u,2u,3u,6u,7u,8u,10u,11u,12u,13u,14u,16u,17u,18u,19u,20u,21u,22u,23u,29u,30u,34u,35u,39u,40u,41u,42u}) {
        const int block=type==42?64:int(ggml_blck_size(ggml_type(type))),bytes=type==42?18:int(ggml_type_size(ggml_type(type)));
        const uint64_t columns=std::max(32,block)*3,rows=3;
        auto raw=std::make_shared<std::vector<uint8_t>>(columns/block*bytes*rows,0x3f);
        auto slice=WeightSlice::raw_rows(raw,*raw,type,columns,rows);
        std::vector<float> expected(columns*rows);const auto* traits=type==42?nullptr:ggml_get_type_traits(ggml_type(type));
        if(type==0) std::memcpy(expected.data(),raw->data(),raw->size());
        else if(type==42) {
            for(uint64_t b=0;b<columns*rows/64;++b) for(uint32_t j=0;j<64;++j) {
                const auto* p=raw->data()+b*18;const float scale=ggml_fp16_to_fp32(uint16_t(p[0])|(uint16_t(p[1])<<8));
                expected[b*64+j]=float(((p[2+j/4]>>(2*(j%4)))&3)-1)*scale;
            }
        } else {require(traits && traits->to_float,"source lacks original ggml decoder");traits->to_float(raw->data(),expected.data(),int64_t(expected.size()));}
        std::vector<uint8_t> reference(expected.size()/32*18),actual(reference.size());
        require(ggml_quantize_chunk(GGML_TYPE_Q4_0,expected.data(),reference.data(),0,rows,columns,nullptr)==reference.size(),"original Q4 reference row bytes");
        slice.convert_rows_q4(0,actual);require(actual==reference,"mapped format -> Q4 differs from original decode + reference quantizer");
    }
}
void invalid() {
    std::array<float,32> values{};std::array<uint8_t,18> output;output.fill(0xcd);
    for(float value:{INFINITY,-INFINITY,NAN,std::numeric_limits<float>::max(),1e-38f}) {
        values[0]=value;rejects([&]{quantize_q4_rows(values,32,output);});require(std::all_of(output.begin(),output.end(),[](auto x){return x==0xcd;}),"invalid Q4 input changed output before validation");
    }
    // Original Q4 explicitly uses inverse=0 when scale underflows to zero;
    // unlike an infinite reciprocal, that case has a defined all-zero result.
    values.fill(0);values[0]=std::numeric_limits<float>::denorm_min();quantize_q4_rows(values,32,output);
    require(output[0]==0 && output[1]==0x80 && std::all_of(output.begin()+2,output.end(),[](auto x){return x==0x88;}),"Q4 zero-scale underflow did not preserve original defined encoding");
    values.fill(0);rejects([&]{quantize_q4_rows(values,31,output);});rejects([&]{quantize_q4_rows(values,32,std::span(output).first(17));});
    rejects([&]{WeightSlice{}.convert_rows_q4(0,output);});
}
}
int main(int argc,char** argv) {
    if(argc!=2) return 2;
    try {oracle(argv[1]);formats();invalid();std::cout<<"PASS 18 independent Q4 row corpora, 28 original source decoders, bounded row/subset order and ownership, unsafe input/output guards; no SDK/device\n";}
    catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
