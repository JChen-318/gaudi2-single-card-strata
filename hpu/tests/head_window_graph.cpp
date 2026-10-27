// Actual production graph factories with the test-only CPU graph backend.
// No libSynapse/runtime/device; this is not an MME or ISA numeric claim.
#include "strata/hpu/model_head.hpp"
#include "strata/hpu/linear_plan.hpp"
#include "strata/hpu/packed_quant.hpp"
#include "strata/hpu/expert_format.hpp"
#include "strata/hpu/hc_ple.hpp"
#include "strata/hpu/window_limits.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "graph_interpreter.hpp"
#include <algorithm>
#include <array>
#include <iostream>
using namespace strata::hpu;
namespace {
template<class F> void rejects(F call) {bool rejected=false;try {call();} catch(const std::exception&) {rejected=true;}oracle::require(rejected,"invalid linear graph/budget accepted");}
std::vector<float> linear(uint32_t h,uint32_t v,uint32_t type,PackedActivation mode,std::span<const float> weights,std::span<const float> input,bool device_quantizer=false) {
    const auto batch=uint32_t(input.size()/h);GraphBuilder builder(OfflineCompilation{});
    if(type==30) define_model_head(builder,h,v,batch);else define_packed_matmul(builder,h,v,batch,type,mode,device_quantizer);
    auto& graph=*oracle::current;graph.set("weight",weights);
    if(type==30) graph.set("x",input);
    else if(device_quantizer) graph.set("activation_input",input);
    else {
        const auto q=mode==PackedActivation::RoutedS2?quantize_s2_activation(input,h):quantize_q8_activation(input,h);
        const std::vector<float> codes(q.codes.begin(),q.codes.end());graph.set("activation",codes);graph.set("metadata",q.metadata);
    }
    uint64_t slots=0;for(const auto& slot:graph.slots) if(slot.desc.persistent && slot.desc.name!="weight") slots+=slot.desc.bytes();
    oracle::require(slots==linear_pinned_bytes(h,v,batch,type,mode,device_quantizer),"linear budget differs from actual persistent non-weight slots");
    graph.run();auto logits=graph.get("logits").values;
    if(!device_quantizer && type!=30 && mode==PackedActivation::Q8_1) {
        // Same weights, same input, but the Q8_1 codes and metadata are produced by the device
        // kernel inside the graph. The two chains must agree bit for bit, because the quantizer is
        // the code the Gaudi2 build actually ships.
        const auto chained=linear(h,v,type,mode,weights,input,true);
        oracle::require(chained.size()==logits.size(),"device-quantized chain returned a different logit count");
        size_t differing=0;
        for(size_t i=0;i<logits.size()&&i<chained.size();++i) if(logits[i]!=chained[i]) ++differing;
        oracle::require(differing==0,"device-quantized chain is not bit-identical to the host quantizer");
    }
    return logits;
}
void heads(uint32_t h,uint32_t v,uint32_t batch,uint32_t type,PackedActivation mode) {
    const auto stride=type==30?h:packed_row_bytes(type,h);std::vector<float> weights(uint64_t(v)*stride),input(uint64_t(batch)*h);
    for(size_t i=0;i<input.size();++i) input[i]=float(int((i*17+batch)%37)-18)/19;
    if(type==30) for(size_t i=0;i<weights.size();++i) weights[i]=float(int((i*11)%31)-15)/33;
    else {
        for(size_t i=0;i<weights.size();++i) weights[i]=float((i*19+7)%256);
        const uint32_t block=packed_block_bytes(type);
        for(uint32_t row=0;row<v;++row) for(uint64_t b=0;b<stride;b+=block) {
            const auto scale=strata::kernels::f16_from_f32(float(int(row%5)-2)/8);
            weights[row*stride+b]=float(scale&255);weights[row*stride+b+1]=float(scale>>8);
            if(type==3 || type==7) {const auto minimum=strata::kernels::f16_from_f32(float(int(row%3)-1)/16);weights[row*stride+b+2]=float(minimum&255);weights[row*stride+b+3]=float(minimum>>8);}
        }
    }
    const auto actual=linear(h,v,type,mode,weights,input);std::vector<float> singleton;
    for(uint32_t row=0;row<batch;++row) {
        auto value=linear(h,v,type,mode,weights,std::span<const float>(input).subspan(uint64_t(row)*h,h));singleton.insert(singleton.end(),value.begin(),value.end());
    }
    oracle::require(actual==singleton,"batched head changed singleton quantization/graph arithmetic or row order");
}
void hc(uint32_t h,uint32_t streams,uint32_t batch) {
    const HcConfig config{h,streams,8,1e-6f};const uint64_t n=uint64_t(h)*streams;
    std::vector<float> input(batch*n),norm(n),down(n*8),up(n*8),inject(n*streams);
    auto fill=[](std::vector<float>& values) {for(size_t i=0;i<values.size();++i) values[i]=float(int((i*7+3)%31)-15)/41;};
    fill(input);fill(down);fill(up);fill(inject);for(size_t i=0;i<norm.size();++i) norm[i]=.5f+float(i%7)/8;
    auto run=[&](std::span<const float> residual,bool single,bool final) {
        GraphBuilder builder(OfflineCompilation{});if(single) define_hc_read(builder,config,final);else define_hc_read_window(builder,config,uint32_t(residual.size()/n),final);
        auto& graph=*oracle::current;graph.set("residual",residual);graph.set("norm",norm);graph.set("w_down",down);graph.set("w_up",up);
        const std::array<float,1> scale{1/float(streams)};graph.set("inverse_streams",scale);
        if(!final) graph.set("w_inject",inject);
        if(final) for(const auto& slot:graph.slots) oracle::require(slot.desc.name!="w_inject" && slot.desc.name!="inject","final HC retained injector tensors");
        graph.run();return graph.get("mixed").values;
    };
    const auto actual=run(input,false,true);oracle::require(actual==run(input,false,false),"final-only HC changed normal HC mixed output");
    std::vector<float> singleton;for(uint32_t row=0;row<batch;++row) {
        auto next=run(std::span<const float>(input).subspan(row*n,n),true,true);singleton.insert(singleton.end(),next.begin(),next.end());
    }
    oracle::require(actual==singleton,"batched final HC changed singleton stream normalization/mix arithmetic");
}
}
void expert_context_slots() {
    // Inspect the actual graph factories instead of repeating the planner's
    // formula: three projection roles x three contexts x four row buckets.
    for(auto geometry:{std::array<uint32_t,2>{128,64},std::array<uint32_t,2>{2560,640}}) {
        uint64_t slots=0;
        for(uint32_t batch:{1u,8u,32u,128u}) for(auto mode:{PackedActivation::Q8_1,PackedActivation::RoutedS2,PackedActivation::RoutedQ8}) for(uint32_t role=0;role<3;++role) {
            const uint32_t h=geometry[0],ff=geometry[1];
            GraphBuilder builder(OfflineCompilation{});
            define_packed_matmul(builder,role==2?ff:h,role==2?h:ff,batch,mode==PackedActivation::RoutedS2?42:2,mode,mode==PackedActivation::Q8_1);
            for(const auto& slot:oracle::current->slots) if(slot.desc.persistent && slot.desc.name!="weight") slots+=slot.desc.bytes();
        }
        oracle::require(slots==packed_expert_mutable_bytes(geometry[0],geometry[1],1),"expert reserve differs from actual codebook/quantizer/context slots");
        oracle::require(slots*49==packed_expert_mutable_bytes(geometry[0],geometry[1],49),"target plus MTP context reserve mismatch");
    }
}
int main() {
    try {
        expert_context_slots();
        uint32_t cases=0;
        for(uint32_t h:{64u,128u}) for(uint32_t v:{5u,17u}) for(uint32_t batch:{1u,3u,7u,8u,32u,128u}) {
            for(uint32_t type:{30u,2u,3u,6u,7u,8u,42u}) {heads(h,v,batch,type,PackedActivation::Q8_1);++cases;}
            heads(h,v,batch,42,PackedActivation::RoutedS2);++cases;
            for(uint32_t type:{2u,3u,6u,7u,8u}) {heads(h,v,batch,type,PackedActivation::RoutedQ8);++cases;}
        }
        for(uint32_t h:{64u,128u}) for(uint32_t streams:{3u,4u}) for(uint32_t batch:{1u,3u,7u,8u,32u,128u}) hc(h,streams,batch);
        for(uint32_t batch:{0u,kResidentWindowRows+1}) {
            rejects([&]{GraphBuilder b(OfflineCompilation{});define_model_head(b,64,5,batch);});
            rejects([&]{GraphBuilder b(OfflineCompilation{});define_packed_matmul(b,64,5,batch,42);});
            rejects([&]{(void)linear_pinned_bytes(64,5,batch,30);});
        }
        rejects([]{(void)linear_pinned_bytes(63,5,7,42);});rejects([]{(void)linear_pinned_bytes(0,5,7,30);});
        rejects([]{GraphBuilder b(OfflineCompilation{});define_packed_matmul(b,64,5,3,2,PackedActivation::RoutedS2);});
        rejects([]{GraphBuilder b(OfflineCompilation{});define_packed_matmul(b,64,5,3,42,PackedActivation::RoutedQ8);});
        rejects([]{GraphBuilder b(OfflineCompilation{});define_packed_matmul(b,64,5,3,2,static_cast<PackedActivation>(3));});
        oracle::require(linear_pinned_bytes(2560,248320,7,30)==6988800,"target head transfer budget changed");
        std::cout<<"PASS actual head graph factories "<<cases<<" BF16/Q2/Q4/Q5/Q8/Q8_1/S2 batch cases, exact singleton rows, 24 final-HC cases, non-power streams, slot budgets and invalid geometry; CPU semantics only, no device\n";
    } catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
}
