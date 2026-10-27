#include "strata/hpu/expert_scales.hpp"
#include "strata/hpu/expert_format.hpp"
#include "strata/hpu/packed_expert_flow.hpp"
#include "strata/hpu/cpu_quant_guard.hpp"
#include <ggml.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
using namespace strata::hpu;
namespace {
void require(bool ok,const char* why) {if(!ok) throw std::runtime_error(why);}
template<class F> void rejects(F call) {bool bad=false;try{call();}catch(const std::exception&){bad=true;}require(bad,"invalid expert scale contract accepted");}
std::span<const uint8_t> bytes(std::span<const float> values) {return {reinterpret_cast<const uint8_t*>(values.data()),values.size_bytes()};}
WeightSlice matrix(uint32_t type) {
    const auto n=uint64_t(64*64/ggml_blck_size(ggml_type(type)))*ggml_type_size(ggml_type(type));
    auto owner=std::make_shared<std::vector<uint8_t>>(n,0x38);return WeightSlice::raw_rows(owner,*owner,type,64,64);
}
void tails() {
    const std::array<WeightSlice,3> mismatched{matrix(40),matrix(2),matrix(8)};
    const uint64_t mixed_up=mismatched[0].source_bytes(),mixed_end=mixed_up*2+mismatched[2].source_bytes();
    rejects([&]{(void)native_expert_blob(mismatched,mixed_up,mixed_up*2,mixed_end+16,mixed_end,{1,1,1});});
    for(auto types:{std::array<uint32_t,3>{40,40,40},{40,40,8},{8,8,40},{8,8,8}}) {
        const std::array<WeightSlice,3> slices{matrix(types[0]),matrix(types[1]),matrix(types[2])};
        const uint64_t up=slices[0].source_bytes(),down=up*2,end=down+slices[2].source_bytes();
        const bool nv=types[0]==40 || types[2]==40;
        const std::array<float,3> scales{types[0]==40?0.125f:1,types[1]==40?3.5f:1,types[2]==40?1.0f/32768:1};
        const auto blob=native_expert_blob(slices,up,down,end+(nv?16:0),nv?end:0,scales);
        require(blob.size()==end+(nv?16:0),"native scale tail not charged in whole blob");
        for(size_t role=0;role<3;++role) require(!std::memcmp(blob.data()+(role==0?0:role==1?up:down),slices[role].bytes().data(),slices[role].source_bytes()),"adding native scale tail changed projection bytes/order");
        if(nv) {std::array<float,4> tail;std::memcpy(tail.data(),blob.data()+end,16);require(tail==std::array<float,4>{scales[0],scales[1],scales[2],0},"native sg/su/sd/padding tail layout differs from original packer");}
        rejects([&]{(void)native_expert_blob(slices,up,down+1,blob.size(),nv?end:0,scales);});
        rejects([&]{(void)native_expert_blob(slices,up,down,blob.size()+1,nv?end:0,scales);});
    }
}
void metadata() {
    std::array<float,3> gate{0.125f,2,4},down{0.00003125f,0.5f,3};
    std::array<std::span<const uint8_t>,3> data{bytes(gate),{},bytes(down)};
    require(decode_expert_scales({40,8,40},data,1,3)==std::array<float,3>{2,1,0.5f},"expert scale role/index or mixed-format unit scale");
    for(float bad:{0.0f,-1.0f,NAN,INFINITY}) {gate[2]=bad;rejects([&]{(void)decode_expert_scales({40,8,40},data,0,3);});} // even a nonselected expert validates
    gate[2]=4;
    rejects([&]{(void)decode_expert_scales({40,8,40},data,3,3);});
    rejects([&]{(void)decode_expert_scales({40,8,40},data,0,2);});
    rejects([&]{(void)decode_expert_scales({8,8,40},data,0,3);});
    data[0]={};rejects([&]{(void)decode_expert_scales({40,8,40},data,0,3);});
    require(scaled_expert_mutable_bytes(2560,640)==(8*2560+10*640)*(1+8+32+128),"scaled F32 expert HBM slot reserve");
}
void placement() {
    const std::array<float,4> input{1,2,3,4};const std::array<float,3> scales{2,3,0.125f};int calls=0;
    const auto result=execute_packed_expert(input,2,1,[&](uint32_t role,auto x) {
        ++calls;
        if(role==0) {require(x.size()==4 && x[0]==1,"gate input scaled before dot");return std::vector<float>{-1,2};}
        if(role==1) return std::vector<float>{4,-2};
        const std::array<float,2> expected{-2.0f/(1+std::exp(2.0f))*12,4.0f/(1+std::exp(-4.0f))*-6};
        require(x.size()==2 && x[0]==expected[0] && x[1]==expected[1],"scales not placed after gate/up dots or down scale folded into hidden");
        return std::vector<float>{x[0],2*x[0],x[1],2*x[1]};
    },scales);
    require(calls==3 && std::abs(result[0]-float((-2/(1+std::exp(2.0)))*12*.125))<1e-6,"global down scale not applied to final F32 projection");
    for(auto bad:{std::array<float,3>{NAN,1,1},{1,0,1},{1,1,-1}}) {
        calls=0;rejects([&]{(void)execute_packed_expert(input,2,1,[&](auto,auto){++calls;return std::vector<float>{};},bad);});require(!calls,"invalid global scale reached a projection");
    }
}
int original_calls=0;
void original_quant(const float* x,void* y,int64_t n) {++original_calls;ggml_get_type_traits(GGML_TYPE_Q8_0)->from_float_ref(x,y,n);}
void exceptional_activations() {
    std::array<float,96> values{};for(size_t i=0;i<32;++i) values[i]=i?float(int(i%7)-3)*0.5f:127;
    std::array<uint8_t,102> output;output.fill(0xcd);original_calls=0;
    require(!guard_q8_0(values.data(),output.data(),values.size(),original_quant) && !original_calls && output[0]==0xcd,"ordinary Q8 input did not preserve original whole-vector quantizer");
    values[32]=NAN;for(size_t i=64;i<96;++i) values[i]=1e-38f;
    std::array<uint8_t,34> expected{};original_quant(values.data(),expected.data(),32);original_calls=0;
    require(guard_q8_0(values.data(),output.data(),values.size(),original_quant) && original_calls==1,"exceptional Q8 blocks reached original integer conversion");
    require(std::equal(expected.begin(),expected.end(),output.begin()),"ordinary Q8 block lost caller's original rounding in guarded vector");
    require(output[34]==0 && output[35]==0x7e,"invalid Q8 input missing NaN marker");
    require(std::all_of(output.begin()+68,output.end(),[](auto v){return v==0;}),"tiny reciprocal-overflow Q8 input not represented by finite zero block");
    values.fill(0);values[0]=std::numeric_limits<float>::max();original_calls=0;
    require(guard_q8_0(values.data(),output.data(),32,original_quant) && !original_calls && output[1]==0x7e,"Q8 FP16 scale overflow not marked before integer quantization");
}
}
int main() {
    try{metadata();tails();placement();exceptional_activations();std::cout<<"PASS NVFP4 scale metadata, native tail bytes, bounded blob assembly, F32 projection scale placement, reserve and exceptional Q8 guard with original-rounding passthrough; no SDK/device\n";}
    catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
