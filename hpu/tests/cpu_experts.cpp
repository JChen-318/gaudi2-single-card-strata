#include "strata/hpu/cpu_experts.hpp"
#include "strata/hpu/expert_scales.hpp"
#include "strata/hpu/native_pack.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/f16_bits.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>

using namespace strata::hpu;
namespace {
namespace cpu=strata::kernels::cpu;
void require(bool ok,const char* message) {if(!ok) throw std::runtime_error(message);}
std::vector<float> quantized(std::span<const float> input) {
    std::vector<float> result(input.size());
    for(size_t start=0;start<input.size();start+=32) {
        float maximum=0;for(size_t i=0;i<32;++i) maximum=std::max(maximum,std::abs(input[start+i]));
        const float scale=maximum/127,inv=scale?1/scale:0;
        for(size_t i=0;i<32;++i) result[start+i]=std::clamp(std::round(input[start+i]*inv),-127.0f,127.0f)*scale;
    }
    return result;
}
std::vector<float> multiply(const WeightSlice& weight,std::span<const float> x) {
    std::vector<float> row(weight.columns()),result(weight.rows());
    for(uint64_t r=0;r<weight.rows();++r) {
        weight.convert_rows_f32(r,row);double sum=0;for(size_t c=0;c<x.size();++c) sum+=double(row[c])*x[c];result[r]=float(sum);
    }
    return result;
}
std::vector<float> nv_activation(const cpu::NativeFmt& format,std::span<const float> input,bool hidden) {
    require((hidden?format.d_act:format.gu_act)==8,"NVFP4 test activation format changed");
    std::vector<uint8_t> bytes(hidden?format.h_bytes:format.act_bytes);
    if(hidden) cpu::native_quant_h(format,input.data(),bytes.data());else cpu::native_quant_act(format,input.data(),bytes.data());
    std::vector<float> values(input.size());
    for(size_t block=0;block<input.size()/32;++block) {
        const auto* p=bytes.data()+block*34;const float scale=strata::kernels::f32_from_f16(uint16_t(p[0])|(uint16_t(p[1])<<8));
        for(size_t j=0;j<32;++j) values[block*32+j]=(p[2+j]<128?int(p[2+j]):int(p[2+j])-256)*scale;
    }
    return values;
}
void compare(std::span<const float> actual,std::span<const float> expected) {
    require(actual.size()==expected.size(),"CPU expert output shape");double error=0,signal=0;
    for(size_t i=0;i<actual.size();++i) {error+=std::pow(double(actual[i])-expected[i],2);signal+=double(expected[i])*expected[i];}
    require(std::sqrt(error/std::max(signal,1e-24))<2e-4,"CPU expert differs from independent ActQ + decoded-weight dot oracle");
}
}
int main(int argc,char** argv) {
    if(argc!=2 && argc!=3) return 2;
    try {
        WeightTable table(argv[1]);const bool nv=table.info("blk.0.ffn_gate_exps.weight").ggml_type==40;
        cpu::NativeFmt format;std::string error;
        require(cpu::native_fmt(int(table.info("blk.0.ffn_gate_exps.weight").ggml_type),int(table.info("blk.0.ffn_down_exps.weight").ggml_type),2560,640,format,error),"CPU fixture format invalid");
        std::shared_ptr<const NativeExpertPack> pack;
        if(argc==3) pack=std::make_shared<NativeExpertPack>(argv[2],table);
        const uint64_t blob=format.bytes;CpuExperts executor(table,blob*2,2,pack);
        std::vector<float> x(2560);
        for(size_t i=0;i<x.size();++i) x[i]=i%32==0?127.0f/128:float(int(i%19)-9)/128;
        const auto q=nv?nv_activation(format,x,false):quantized(x);const auto gate=table.expert("blk.0.ffn_gate_exps.weight",0),up=table.expert("blk.0.ffn_up_exps.weight",0),down=table.expert("blk.0.ffn_down_exps.weight",0);
        const auto scales=nv?expert_global_scales(table,0,0):std::array<float,3>{1,1,1};
        auto hidden=multiply(gate,q);const auto upper=multiply(up,q);
        for(size_t i=0;i<hidden.size();++i) {const auto value=hidden[i]*scales[0];hidden[i]=(value/(1+std::exp(-value)))*(upper[i]*scales[1]);}
        auto expected=multiply(down,nv?nv_activation(format,hidden,true):quantized(hidden));for(auto& value:expected) value*=scales[2];
        compare(executor.run(0,0,x),expected);compare(executor.run(0,0,x),expected);
        std::vector<float> batch;for(int t=0;t<8;++t) batch.insert(batch.end(),x.begin(),x.end());const auto grouped=executor.run(0,0,batch);
        for(size_t t=0;t<8;++t) compare(std::span<const float>(grouped).subspan(t*2560,2560),expected);
        (void)executor.run(0,1,x);(void)executor.run(0,2,x);compare(executor.run(0,0,x),expected);
        const auto stats=executor.stats();require(stats.hits>=2 && stats.evictions>=1 && stats.resident_bytes<=blob*2,"CPU raw cache bounds/eviction");
        bool rejected=false;try {(void)executor.run(0,-1,x);} catch(const std::exception&) {rejected=true;}require(rejected,"shared expert accidentally accepted by routed CPU executor");
        std::cout<<"PASS real original CPU ActQ and pool against independent scalar quantization/dot oracle, 1/8 rows, raw GGUF slices and bounded cache eviction\n";
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
