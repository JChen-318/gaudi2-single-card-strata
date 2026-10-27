#include "strata/hpu/packed_quant.hpp"
#include "strata/hpu/expert_format.hpp"
#include "strata/hpu/packed_expert_flow.hpp"
#include "strata/hpu/resident_cache.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "packed_emulator.hpp"
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
using namespace strata::hpu;
namespace {
void require(bool ok,const char* why){if(!ok) throw std::runtime_error(why);}
template<class F> void rejects(F call){bool bad=false;try{call();}catch(const std::exception&){bad=true;}require(bad,"invalid S2 contract accepted");}
template<class T> T read(std::ifstream& stream){T value;stream.read(reinterpret_cast<char*>(&value),sizeof(value));if(!stream) throw std::runtime_error("truncated S2 quant oracle");return value;}
void quant_oracle(const char* path) {
    std::ifstream stream(path,std::ios::binary);const auto cases=read<uint32_t>(stream);require(cases==18,"S2 oracle case count");
    for(uint32_t c=0;c<cases;++c) {
        const auto columns=read<uint32_t>(stream),count=read<uint32_t>(stream);std::vector<float> input(count),metadata(count/16);std::vector<uint8_t> codes(count);
        stream.read(reinterpret_cast<char*>(input.data()),input.size()*4);stream.read(reinterpret_cast<char*>(codes.data()),codes.size());
        stream.read(reinterpret_cast<char*>(metadata.data()),metadata.size()*4);require(bool(stream),"truncated S2 quant payload");
        const auto actual=quantize_s2_activation(input,columns);require(actual.codes==codes,"routed reciprocal/.5 codes differ from independent float32 oracle");
        require(!std::memcmp(actual.metadata.data(),metadata.data(),metadata.size()*4),"routed FP32 scale/correction differs from independent oracle");
    }
    require(stream.peek()==std::char_traits<char>::eof(),"S2 oracle trailing data");
}
std::vector<float> project(std::vector<uint8_t>& w,uint32_t columns,uint32_t rows,std::span<const float> input,PackedActivation mode) {
    auto activation=mode==PackedActivation::RoutedS2?quantize_s2_activation(input,columns):quantize_q8_activation(input,columns);
    const auto batch=input.size()/columns;std::vector<float> result(batch*rows);
    using namespace packed_emulator;
    tensor weights{w.data(),{packed_row_bytes(42,columns),rows},1},codes{activation.codes.data(),{columns,batch},1};
    tensor metadata{activation.metadata.data(),{2,columns/32,batch},4},out{result.data(),{rows,batch},4};
    offset={0,0,0,0,0};extent={int(rows),int(batch),1,1,1};packed_kernel(weights,codes,metadata,out,columns,rows,int(batch),42,int(mode));return result;
}
void flow() {
    constexpr uint32_t h=128,ff=64;std::mt19937 rng(448);
    std::array<std::vector<uint8_t>,3> weights;
    for(size_t role=0;role<3;++role) {
        const uint32_t columns=role==2?ff:h,rows=role==2?h:ff;
        weights[role].resize(packed_row_bytes(42,columns)*rows);for(auto& byte:weights[role]) byte=uint8_t(rng());
        for(size_t at=0;at<weights[role].size();at+=18) {weights[role][at]=0;weights[role][at+1]=0x20;} // d=1/128
    }
    for(auto mode:{PackedActivation::RoutedS2,PackedActivation::Q8_1}) {
        std::vector<float> input(3*h);for(auto& x:input) x=float(int(rng()%1000)-500)/512;
        std::vector<uint32_t> roles;
        const auto result=execute_packed_expert(input,h,ff,[&](uint32_t role,auto values){roles.push_back(role);return project(weights[role],role==2?ff:h,role==2?h:ff,values,mode);});
        require(roles==std::vector<uint32_t>({0,1,2}) && result.size()==input.size(),"actual packed expert projection order/shape");
        for(uint32_t row=0;row<3;++row) {
            const auto one=execute_packed_expert(std::span<const float>(input).subspan(row*h,h),h,ff,[&](uint32_t role,auto values){return project(weights[role],role==2?ff:h,role==2?h:ff,values,mode);});
            require(!std::memcmp(result.data()+row*h,one.data(),h*4),"batched routed/shared expert changed single-row arithmetic or crossed rows");
        }
    }
    std::array<float,128> input{};int calls=0;
    rejects([&]{(void)execute_packed_expert(input,h,ff,[&](uint32_t,auto){++calls;return std::vector<float>(1);});});require(calls==2,"shape failure bypassed projection guards");
    input[0]=NAN;calls=0;rejects([&]{(void)execute_packed_expert(input,h,ff,[&](uint32_t,auto){++calls;return std::vector<float>{};});});require(!calls,"non-finite expert input reached projection");
}
void formats() {
    const auto routed=expert_format(2560,640,0,{42,42,42}),shared=expert_format(2560,640,-1,{8,8,8}),mixed=expert_format(2560,640,-1,{2,8,42});
    require(routed.packed && routed.activation==PackedActivation::RoutedS2 && routed.bytes==1382400,"routed Q2 format/charge");
    require(shared.packed && shared.activation==PackedActivation::Q8_1 && shared.bytes==5222400,"shared Q8 format/charge");
    require(mixed.packed && mixed.bytes==1638400/32*18+1638400/32*34+1638400/64*18,"mixed shared packed role charge");
    // Q2 plus an integer superblock is packable and keeps Q2's CPU ActQ contract on the
    // Q2 projections only; a superblock whose 256-value block cannot divide n_ff 640
    // still decodes the whole expert.
    const auto super_mixed=expert_format(2560,640,0,{42,23,42});
    require(super_mixed.packed && super_mixed.activation==PackedActivation::RoutedQ8
        && super_mixed.bytes==uint64_t(720)*640+uint64_t(1360)*640+uint64_t(180)*2560,"mixed Q2/IQ4_XS routed expert policy or charge wrong");
    require(expert_projection_activation(super_mixed,0)==PackedActivation::RoutedS2
        && expert_projection_activation(super_mixed,1)==PackedActivation::RoutedQ8
        && expert_projection_activation(super_mixed,2)==PackedActivation::RoutedS2,"mixed Q2/IQ4_XS projections lost their per-role contracts");
    const auto super_down=expert_format(2560,640,0,{23,23,23});
    require(!super_down.packed && super_down.bytes==9830400,"a 256-value superblock at n_ff 640 was not rejected by layout");
    require(!expert_format(2560,640,-1,{2,19,8}).packed,"partly unsupported shared expert changed only some projections");
    ResidentCache<ExpertFormat> cache(routed.bytes+shared.bytes);uint32_t resolved=0;
    auto put=[&](const char* key,const ExpertFormat& f) {if(auto hit=cache.find(key)) return hit;++resolved;return cache.acquire(key,f.bytes,[&]{return std::make_unique<const ExpertFormat>(f);});};
    auto first=put("routed",routed);const auto generation=first.generation();first={};
    {auto hit=put("routed",routed);require(hit.generation()==generation && resolved==1,"resident hit resolved/relayouted source again");}
    auto held=put("shared",shared);{auto other=put("routed-2",routed);require(cache.stats().resident_bytes==routed.bytes+shared.bytes && cache.stats().evictions==1,"mixed-size cache budget/leased selection");}
    held={};cache.clear();require(!cache.stats().resident_bytes,"mixed cache clear ownership");
}
}
int main(int argc,char** argv) {
    if(argc!=2) return 2;
    try {quant_oracle(argv[1]);flow();formats();std::array<float,32> tiny{};tiny[0]=std::numeric_limits<float>::denorm_min();rejects([&]{(void)quantize_s2_activation(tiny,32);});
        std::cout<<"PASS routed S2 FP32 scale/reciprocal quantization, actual TPC CPU-order source, shared/routed expert flow, row batching and mixed cache format budgets\n";
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
