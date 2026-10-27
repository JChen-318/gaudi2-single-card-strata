// Hardware-only generation/lease/rebind and mixed-format expert acceptance.
#include "strata/hpu/moe.hpp"
#include "strata/hpu/packed_expert_flow.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "packed_emulator.hpp"
#include <cmath>
#include <cstring>
#include <iostream>
#include <map>
using namespace strata::hpu;
namespace {
constexpr uint32_t H=128,FF=64,KH=256;
void require(bool ok,const char* why){if(!ok) throw std::runtime_error(why);}
ExpertSlices source(int32_t expert,std::array<uint32_t,3> shared_types={8,8,8},std::array<uint32_t,3> routed_types={42,42,42},uint32_t hidden=H,uint32_t ff=FF) {
    std::array<WeightSlice,3> slices;
    for(uint32_t role=0;role<3;++role) {
        const uint32_t type=expert==-1?shared_types[role]:expert==4?30:expert==5?40:routed_types[role],columns=role==2?ff:hidden,rows=role==2?hidden:ff;
        const uint64_t row_bytes=type==30?columns*2:type==40?columns/64*36:packed_row_bytes(type,columns);
        auto bytes=std::make_shared<std::vector<uint8_t>>(row_bytes*rows);
        for(size_t i=0;i<bytes->size();++i) (*bytes)[i]=uint8_t(i*13+role*31+uint32_t(expert+2)*17);
        if(type==30) for(size_t i=0;i<bytes->size();i+=2) {const auto bits=to_bf16(float(int(i%19)-9)/128);(*bytes)[i]=uint8_t(bits);(*bytes)[i+1]=uint8_t(bits>>8);}
        else if(type==40) for(size_t i=0;i<bytes->size();i+=36) for(size_t j=0;j<4;++j) (*bytes)[i+j]=0x30;
        else for(size_t i=0;i<bytes->size();i+=packed_block_bytes(type)) {
            const size_t scale_at=type==11?108:type==14?208:0; // Q3_K/Q6_K keep d last
            (*bytes)[i+scale_at]=0;(*bytes)[i+scale_at+1]=0x20; // d=1/128
            if(type==3 || type==7 || type==12 || type==13) {(*bytes)[i+2]=0;(*bytes)[i+3]=0x18;} // finite learned minimum/dmin
        }
        slices[role]=WeightSlice::raw_rows(bytes,*bytes,type,columns,rows);
    }
    return {slices[0],slices[1],slices[2],expert==5?std::array<float,3>{0.25f,2,0.125f}:std::array<float,3>{1,1,1},expert==5};
}
std::vector<float> reference(int32_t expert,std::span<const float> input,std::array<uint32_t,3> shared_types={8,8,8},std::array<uint32_t,3> routed_types={42,42,42},uint32_t hidden=H,uint32_t ff=FF) {
    const auto slices=source(expert,shared_types,routed_types);const std::array<WeightSlice,3> weights{slices.gate,slices.up,slices.down};
    const auto format=expert_format(hidden,ff,expert,{weights[0].ggml_type(),weights[1].ggml_type(),weights[2].ggml_type()});
    return execute_packed_expert(input,hidden,ff,[&](uint32_t role,auto values) {
        const auto& weight=weights[role];const auto cols=uint32_t(weight.columns()),rows=uint32_t(weight.rows()),batch=uint32_t(values.size()/cols);
        if(expert==5 && !format.packed) {
            std::vector<float> decoded(uint64_t(rows)*cols),result(uint64_t(batch)*rows);weight.convert_rows_f32(0,decoded);
            for(uint32_t t=0;t<batch;++t) for(uint32_t row=0;row<rows;++row) {
                double sum=0;for(uint32_t c=0;c<cols;++c) sum+=double(from_bf16(to_bf16(decoded[uint64_t(row)*cols+c])))*from_bf16(to_bf16(values[uint64_t(t)*cols+c]));
                result[uint64_t(t)*rows+row]=float(sum);
            }
            return result;
        }
        const auto mode=expert_projection_activation(format,role);
        auto activation=mode==PackedActivation::RoutedS2?quantize_s2_activation(values,cols):quantize_q8_activation(values,cols);std::vector<float> result(rows*batch);
        using namespace packed_emulator;
        tensor w{const_cast<uint8_t*>(weight.bytes().data()),{weight.row_bytes(),rows},1},a{activation.codes.data(),{cols,batch},1};
        tensor m{activation.metadata.data(),{2,cols/32,batch},4},o{result.data(),{rows,batch},4};
        offset={0,0,0,0,0};extent={int(rows),int(batch),1,1,1};packed_kernel(w,a,m,o,cols,rows,batch,int(weight.ggml_type()),int(mode));return result;
    },slices.output_scales);
}
void compare(std::span<const float> actual,std::span<const float> expected) {
    require(actual.size()==expected.size(),"packed expert output length");
    for(size_t i=0;i<actual.size();++i) require(std::isfinite(actual[i]) && std::abs(actual[i]-expected[i])<=std::max(0.003f,std::abs(expected[i])*0.00005f),"real HPU packed expert differs from CPU TPC C oracle");
}
}
int main() {
    try {
        Device device;device.allocation_limits(64ull<<20,8ull<<20);
        strata::core::ModelGeometry g;g.n_embd=H;g.n_ff=FF;g.n_expert=8;g.n_expert_used=2;g.n_layers=1;
        std::map<int32_t,uint32_t> resolved;
        ExpertCache cache(device,g,[&](uint32_t,int32_t expert){++resolved[expert];return source(expert);},6*H*FF,512);
        ExpertExecutor executor(device,g,8);std::vector<float> input(3*H);for(size_t i=0;i<input.size();++i) input[i]=float(int((i*17)%31)-15)/32;
        auto held=cache.acquire(0,0);const auto generation=held.generation();const auto first=executor.run(input,held);compare(first,reference(0,input));
        {auto hit=cache.acquire(0,0);require(hit.generation()==generation && resolved[0]==1,"packed cache hit reloaded/relayouted weights");}
        for(auto expert:{1,-1,2,3,6,7}) {
            auto lease=cache.acquire(0,expert);const auto& w=lease.get();
            require(lease.bytes()==w.gate.size()+w.up.size()+w.down.size(),"packed cache charged decoded or partial payload");
            compare(executor.run(input,std::move(lease)),reference(expert,input));
        }
        require(cache.resident(0,0).generation()==generation,"mixed cache evicted a live consumer generation");held={};
        {auto legacy=cache.acquire(0,4);require(!legacy->format.packed,"unsupported source did not select BF16 reference");const auto result=executor.run(input,std::move(legacy));for(float x:result) require(std::isfinite(x),"legacy format switch produced invalid output");}
        {auto nvfp4=cache.acquire(0,5);require(nvfp4->scaled_outputs && nvfp4->format.packed,"NVFP4 expert lost packed arithmetic or its global scale contract");compare(executor.run(input,std::move(nvfp4)),reference(5,input));}
        auto reloaded=cache.acquire(0,0);require(reloaded.generation()>generation && resolved[0]==2,"packed expert ABA reload retained old generation");
        compare(executor.run(input,std::move(reloaded)),first);
        require(cache.stats().resident_bytes<=cache.capacity() && cache.stats().evictions>0,"combined packed/BF16 budget or eviction missing");
        for(auto types:{std::array<uint32_t,3>{3,3,3},std::array<uint32_t,3>{6,6,6},std::array<uint32_t,3>{7,7,7},std::array<uint32_t,3>{3,6,7}}) {
            ExpertCache affine_cache(device,g,[&](uint32_t,int32_t expert){return source(expert,types);},6*H*FF,512);
            ExpertExecutor affine(device,g,8);auto lease=affine_cache.acquire(0,-1);
            require(lease->format.packed && lease.bytes()==expert_format(H,FF,-1,types).bytes,"shared affine cache decoded or mischarged payload");
            compare(affine.run(input,std::move(lease)),reference(-1,input,types));
        }
        // Keep the executor across destroyed/recreated caches and source mode
        // changes. Every call must rebind current leased buffers before use.
        ExpertExecutor routed(device,g,8);
        for(auto types:{std::array<uint32_t,3>{2,2,2},{3,6,7},{8,8,8},{42,2,8},{3,42,7},{6,8,42},{42,42,42},{2,2,2}}) {
            ExpertCache mixed(device,g,[&](uint32_t,int32_t expert){return source(expert,{8,8,8},types);},6*H*FF,512);
            auto lease=mixed.acquire(0,0);const auto format=expert_format(H,FF,0,types);
            require(lease->format==format && lease.bytes()==format.bytes,"routed mixed cache policy/payload mismatch");
            compare(routed.run(input,std::move(lease)),reference(0,input,{8,8,8},types));
            auto shared=mixed.acquire(0,-1);compare(routed.run(input,std::move(shared)),reference(-1,input));
        }
        // Routed K-quant gate/up needs a 256-value-aligned hidden length. The down
        // projection stays Q8_0 because n_ff is never a 256 multiple here, matching
        // upstream native_expert_supported's n_ff % 256 requirement. Q3_K runs the
        // same three role slots with its own sub-block scale layout.
        for(auto types:std::array<std::array<uint32_t,3>,12>{std::array<uint32_t,3>{12,13,8},std::array<uint32_t,3>{11,12,8},std::array<uint32_t,3>{14,11,8},std::array<uint32_t,3>{20,14,8},std::array<uint32_t,3>{23,20,8},std::array<uint32_t,3>{16,23,8},std::array<uint32_t,3>{18,16,8},std::array<uint32_t,3>{17,18,8},std::array<uint32_t,3>{21,17,8},std::array<uint32_t,3>{22,21,8},std::array<uint32_t,3>{29,22,8},std::array<uint32_t,3>{40,40,8}}) {
            strata::core::ModelGeometry kg;kg.n_embd=KH;kg.n_ff=FF;kg.n_expert=8;kg.n_expert_used=2;kg.n_layers=1;
            uint32_t kresolved=0;
            ExpertCache kcache(device,kg,[&](uint32_t,int32_t expert){++kresolved;return source(expert,types,types,KH,FF);},6*KH*FF,4096);
            ExpertExecutor kexecutor(device,kg,8);
            std::vector<float> kinput(3*KH);for(size_t i=0;i<kinput.size();++i) kinput[i]=float(int((i*29)%37)-18)/32;
            const auto format=expert_format(KH,FF,0,types),shared_format=expert_format(KH,FF,-1,types);
            require(format.packed && format.activation==PackedActivation::RoutedQ8,"routed K-quant expert lost packed arithmetic at real geometry");
            require(shared_format.packed && shared_format.activation==PackedActivation::Q8_1,"shared K-quant expert did not take the head contract");
            require(!expert_format(KH,FF,-1,{8,8,13}).packed && !expert_format(KH,FF,-1,{8,8,11}).packed,
                "K-quant down role was packed despite a 64-column projection");
            auto lease=kcache.acquire(0,0);const auto kgeneration=lease.generation();
            require(lease->format==format && lease.bytes()==format.bytes,"routed K-quant cache charged decoded or partial payload");
            const auto expected=reference(0,kinput,types,types,KH,FF);
            compare(kexecutor.run(kinput,std::move(lease)),expected);
            auto hit=kcache.acquire(0,0);
            require(hit.generation()==kgeneration && kresolved==1,"routed K-quant hit reloaded or relayouted source bytes");
            compare(kexecutor.run(kinput,std::move(hit)),expected);
            auto shared=kcache.acquire(0,-1);
            require(shared->format==shared_format && shared.bytes()==shared_format.bytes,"shared K-quant lease mischarged the payload");
            compare(kexecutor.run(kinput,std::move(shared)),reference(-1,kinput,types,types,KH,FF));
        }
        std::cout<<"PASS real HPU routed/shared packed experts, effective batch rows, one combined payload budget, no-I/O hits, leased victim protection, BF16 format switch, ABA rebind and routed/shared Q3_K, Q4_K, Q5_K, Q6_K, IQ4_NL, IQ4_XS, IQ2_XXS, IQ3_XXS, IQ2_XS, IQ3_S, IQ2_S, IQ1_M and NVFP4 role sets\n";
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
