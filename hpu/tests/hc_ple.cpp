#include "strata/hpu/hc_ple.hpp"
#include "strata/hpu/host_recipe.hpp"
#include "strata/hpu/hc_window.hpp"
#include "strata/hpu/ple_window.hpp"
#include "strata/hpu/session_flow.hpp"
#include <array>
#include <cmath>
#include <iostream>
using namespace strata::hpu;
namespace {
void compare(std::span<const float> a,const std::vector<float>& b,float tolerance,const char* name) {
    if(a.size()!=b.size()) throw std::runtime_error("HC/PLE output shape mismatch");
    double peak=0;
    for(size_t i=0;i<a.size();++i) {
        const auto error=std::abs(double(a[i])-b[i]);peak=std::max(peak,error);
        if(!std::isfinite(a[i]) || error>tolerance) throw std::runtime_error(std::string(name)+" hardware parity failure at "+std::to_string(i));
    }
    std::cout<<"PASS real HPU "<<name<<" max_abs_error="<<peak<<'\n';
}
float sigmoid(float x) {return float(1/(1+std::exp(-double(x))));}
void run_window(const Device& device) {
    const HcConfig config{64,3,8,1e-6f};const uint32_t h=config.hidden,r=config.streams,n=h*r;
    struct Weights {std::vector<std::unique_ptr<DeviceBuffer>> owned;std::vector<Binding> bindings;};
    std::array<Weights,2> weights;Stream copy(device);
    const std::array<const char*,4> names{"w_down","w_up","w_inject","norm"};
    const std::array<uint64_t,4> elements{n*config.rank,n*config.rank,n*r,n};
    for(size_t phase=0;phase<weights.size();++phase) for(size_t role=0;role<4;++role) {
        const auto bytes=elements[role]*(role==3?4:2);HostBuffer source(device,bytes);auto target=std::make_unique<DeviceBuffer>(device,bytes);
        for(size_t i=0;i<elements[role];++i) {
            const float value=role==3?float((i+phase)%9+7)/16:float(int((i+phase*3)%13)-6)/128;
            if(role==3) static_cast<float*>(source.data())[i]=value;else static_cast<uint16_t*>(source.data())[i]=to_bf16(value);
        }
        upload(copy,*target,source,bytes);copy.synchronize();weights[phase].bindings.push_back({names[role],target.get()});weights[phase].owned.push_back(std::move(target));
    }
    // The executor and scalar recipes die before the borrowed weights. Alternate
    // weights at equal row counts, then change row geometry to cover rebinds.
    HcWindow window(device,config,128);size_t iteration=0;
    for(uint32_t rows:{5u,5u,2u,8u}) {
        const auto& bindings=weights[iteration++%2].bindings;
        HostRecipe scalar_read(device,build_hc_read(device,config),bindings,{"mixed","inject"});
        HostRecipe scalar_write(device,build_hc_write(device,config),{}, {"out"});
        const std::array<float,1> scale{1/float(r)};scalar_read.set("inverse_streams",scale);
        std::vector<float> input(uint64_t(rows)*n),block(uint64_t(rows)*h),expected_mixed,expected_inject,expected_out;
        for(size_t i=0;i<input.size();++i) input[i]=float(int(i%31)-15)/32;
        for(size_t i=0;i<block.size();++i) block[i]=float(int(i%19)-9)/64;
        const auto read=window.read(input,bindings);
        for(uint32_t row=0;row<rows;++row) {
            scalar_read.set("residual",std::span<const float>(input).subspan(uint64_t(row)*n,n));scalar_read.run();
            const auto mixed=scalar_read.output("mixed"),inject=scalar_read.output("inject");
            expected_mixed.insert(expected_mixed.end(),mixed.begin(),mixed.end());expected_inject.insert(expected_inject.end(),inject.begin(),inject.end());
            scalar_write.set("residual",std::span<const float>(input).subspan(uint64_t(row)*n,n));
            scalar_write.set("block",std::span<const float>(block).subspan(uint64_t(row)*h,h));scalar_write.set("inject",inject);scalar_write.run();
            const auto out=scalar_write.output("out");expected_out.insert(expected_out.end(),out.begin(),out.end());
        }
        compare(read.mixed,expected_mixed,4e-4f,"window HC MME mix versus scalar");compare(read.inject,expected_inject,4e-4f,"window HC MME inject versus scalar");
        const std::array<Binding,3> final_bindings{bindings[0],bindings[1],bindings[3]};
        const auto final=window.read(input,final_bindings,true);
        compare(final.mixed,expected_mixed,4e-4f,"window final HC mix versus scalar");
        if(!final.inject.empty()) throw std::runtime_error("final HC exposed injector output");
        const auto normal_again=window.read(input,bindings);
        compare(normal_again.mixed,expected_mixed,4e-4f,"same-row final-to-normal HC mode switch");
        compare(normal_again.inject,expected_inject,4e-4f,"HC mode switch lost injector binding");
        compare(window.write(input,block,read.inject),expected_out,6e-4f,"window HC write versus scalar");
    }
}
void run_ple_window(const Device& device,uint32_t h,uint32_t streams) {
    const HcConfig config{h,streams,8,1e-6f};const uint64_t n=uint64_t(h)*streams;
    std::vector<std::unique_ptr<DeviceBuffer>> owned;std::vector<Binding> bindings;Stream copy(device);
    const std::array<const char*,6> names{"w_key","w_value","norm_key","norm_query","norm_conv","conv_weights"};
    const std::array<uint64_t,6> elements{n*h,uint64_t(h)*h,n,n,n,4*n};
    for(size_t role=0;role<names.size();++role) {
        const auto bytes=elements[role]*(role<2?2:4);HostBuffer source(device,bytes);auto target=std::make_unique<DeviceBuffer>(device,bytes);
        for(size_t i=0;i<elements[role];++i) {
            const float value=role>=2 && role<=4?.5f+float(i%11)/16:float(int((i*17+role*11)%31)-15)/128;
            if(role<2) static_cast<uint16_t*>(source.data())[i]=to_bf16(value);else static_cast<float*>(source.data())[i]=value;
        }
        upload(copy,*target,source,bytes);copy.synchronize();bindings.push_back({names[role],target.get()});owned.push_back(std::move(target));
    }
    HostRecipe scalar(device,build_ple_block(device,config),bindings,{"out","history_out"});PleWindow window(device,config,bindings);
    std::vector<float> scalar_history(9*n),window_history(9*n);
    for(size_t i=0;i<scalar_history.size();++i) scalar_history[i]=float(int((i*11+3)%41)-20)/64;
    window_history=scalar_history;
    auto run_rows=[&](uint32_t rows) {
        std::vector<float> input(uint64_t(rows)*n),embedding(uint64_t(rows)*h),expected;
        for(size_t i=0;i<input.size();++i) input[i]=float(int((i*11+rows)%37)-18)/32;
        for(size_t i=0;i<embedding.size();++i) embedding[i]=float(int((i*13+7)%31)-15)/41;
        for(uint32_t row=0;row<rows;++row) {
            scalar.set("hidden",std::span<const float>(input).subspan(uint64_t(row)*n,n));scalar.set("embedding",std::span<const float>(embedding).subspan(uint64_t(row)*h,h));
            scalar.set("history",scalar_history);scalar.run();const auto output=scalar.output("out");expected.insert(expected.end(),output.begin(),output.end());scalar_history=scalar.values("history_out");
        }
        auto actual=window.run(input,embedding,window_history);compare(actual.output,expected,6e-4f,"PLE native window versus singleton output");
        compare(actual.history,scalar_history,3e-4f,"PLE native window causal history");window_history=std::move(actual.history);
    };
    for(uint32_t rows:{1u,5u,8u,17u,2u}) run_rows(rows);
    const auto before=window_history;int polls=0;bool cancelled=false;
    try {(void)window.run(std::vector<float>(17*n,.125f),std::vector<float>(17ull*h,.25f),window_history,[&] {return ++polls==3;});} catch(const Cancelled&) {cancelled=true;}
    if(!cancelled || before!=window_history) throw std::runtime_error("cancelled native PLE window changed accepted caller history");
    run_rows(1);
    std::cout<<"PASS real HPU native PLE window H="<<h<<" streams="<<streams<<" cap8, nonzero weights/history, geometry changes, chunk continuation and cancellation\n";
}
void run(const Device& device,uint32_t h) {
    const uint32_t streams=4,n=streams*h;
    GraphBuilder g(device);
    const auto x=g.tensor({"x",DType::F32,{streams,h}}),gamma=g.tensor({"gamma",DType::F32,{streams,h}});
    const auto norm=g.tensor({"norm",DType::F32,{streams,h}}),gate=g.tensor({"gate",DType::F32,{streams,h}});
    const auto mixed=g.tensor({"mixed",DType::F32,{h}}),inject=g.tensor({"inject",DType::F32,{streams}});
    const auto out=g.tensor({"out",DType::F32,{streams,h}});
    g.row_norm(x,gamma,norm,1e-6f);g.hc_mix(norm,gate,mixed);g.hc_write(x,mixed,inject,out);
    HostRecipe hc(device,g.compile("hc_postops_hardware"),{}, {"norm","mixed","out"});
    std::vector<float> residual(n),weight(n),gates(n),injection{-4,-1,1,4},expected_norm(n),expected_mix(h),expected_out(n);
    for(uint32_t i=0;i<n;++i) {residual[i]=float(int(i%31)-15)/32;weight[i]=float(i%9+4)/8;gates[i]=float(int(i%13)-6)/4;}
    for(uint32_t row=0;row<streams;++row) {
        double sum=0;for(uint32_t d=0;d<h;++d) sum+=double(residual[row*h+d])*residual[row*h+d];
        const auto inverse=float(1/std::sqrt(sum/h+1e-6));
        for(uint32_t d=0;d<h;++d) expected_norm[row*h+d]=(residual[row*h+d]*inverse)*weight[row*h+d];
    }
    for(uint32_t d=0;d<h;++d) for(uint32_t row=0;row<streams;++row) expected_mix[d]+=expected_norm[row*h+d]*sigmoid(gates[row*h+d])/streams;
    for(uint32_t row=0;row<streams;++row) for(uint32_t d=0;d<h;++d) expected_out[row*h+d]=residual[row*h+d]+expected_mix[d]*(2*sigmoid(injection[row]/streams));
    hc.set("x",residual);hc.set("gamma",weight);hc.set("gate",gates);hc.set("inject",injection);hc.run();
    compare(hc.output("norm"),expected_norm,2e-4f,"HC normalized");compare(hc.output("mixed"),expected_mix,2e-4f,"HC mixed");compare(hc.output("out"),expected_out,4e-4f,"HC write");
    GraphBuilder p(device);
    const auto key=p.tensor({"key",DType::F32,{streams,h}}),query=p.tensor({"query",DType::F32,{streams,h}}),value=p.tensor({"value",DType::F32,{h}});
    const auto gated=p.tensor({"gated",DType::F32,{streams,h}}),norm_weight=p.tensor({"gamma",DType::F32,{streams,h}});
    const auto normalized=p.tensor({"normalized",DType::F32,{streams,h}}),nf=p.tensor({"normalized_flat",DType::F32,{n},false});
    const auto gf=p.tensor({"gated_flat",DType::F32,{n},false}),hist=p.tensor({"history",DType::F32,{9,n}});
    const auto wc=p.tensor({"weights",DType::F32,{4,n}}),hidden=p.tensor({"hidden",DType::F32,{n}});
    const auto next=p.tensor({"history_out",DType::F32,{9,n}}),result=p.tensor({"out",DType::F32,{n}});
    p.ple_gate(key,query,value,gated);p.row_norm(gated,norm_weight,normalized,1e-6f);p.reshape(gated,gf);p.reshape(normalized,nf);
    p.ple_conv(hist,nf,wc,hidden,gf,next,result,3);
    HostRecipe ple(device,p.compile("ple_postops_hardware"),{}, {"gated","normalized","history_out","out"});
    std::vector<float> keys(n),queries(n,1),values(h,0.25f),history(9*n),conv_weight(4*n),expected_gated(n),expected_normalized(n),expected_history(9*n),expected_result(n);
    for(uint32_t row=0;row<streams;++row) for(uint32_t d=0;d<h;++d) keys[row*h+d]=row==0?0:row==1?0.01f:row==2?-0.01f:1e-10f;
    for(size_t i=0;i<history.size();++i) history[i]=float(int(i%23)-11)/64;
    for(size_t i=0;i<conv_weight.size();++i) conv_weight[i]=float(int(i%11)-5)/32;
    auto reference_history=history;
    for(uint32_t row=0;row<streams;++row) {
        const float s=keys[row*h]*float(h)/std::sqrt(float(h));
        const float root=(s>0?1.0f:(s<0?-1.0f:0))*std::sqrt(std::max(std::abs(s),1e-6f));
        double sum=0;for(uint32_t d=0;d<h;++d) {const auto v=values[d]*sigmoid(root);expected_gated[row*h+d]=v;sum+=double(v)*v;}
        const auto inverse=float(1/std::sqrt(sum/h+1e-6));
        for(uint32_t d=0;d<h;++d) expected_normalized[row*h+d]=(expected_gated[row*h+d]*inverse)*weight[row*h+d];
    }
    ple.set("key",keys);ple.set("query",queries);ple.set("value",values);ple.set("gamma",weight);ple.set("weights",conv_weight);ple.set("hidden",residual);
    for(int step=0;step<5;++step) {
        for(uint32_t channel=0;channel<n;++channel) {
            float sum=0;for(uint32_t tap=0;tap<4;++tap) sum+=(tap==3?expected_normalized[channel]:reference_history[tap*3*n+channel])*conv_weight[tap*n+channel];
            expected_result[channel]=(residual[channel]+expected_gated[channel])+sum*sigmoid(sum);
            for(uint32_t row=0;row<9;++row) expected_history[row*n+channel]=row==8?expected_normalized[channel]:reference_history[(row+1)*n+channel];
        }
        ple.set("history",history);ple.run();
        compare(ple.output("gated"),expected_gated,2e-4f,"PLE signed-root gate");compare(ple.output("normalized"),expected_normalized,2e-4f,"PLE conv input");
        compare(ple.output("history_out"),expected_history,2e-4f,"PLE history");compare(ple.output("out"),expected_result,4e-4f,"PLE result");
        // Preserve actual normalized history for the next hardware launch. The
        // oracle follows its own expected history to expose cumulative drift.
        history.assign(ple.output("history_out").begin(),ple.output("history_out").end());
        reference_history=expected_history;
    }
    std::cout<<"PASS real HPU HC/PLE postops H="<<h<<" five PLE steps\n";
}
}
int main() {
    try {Device device;run(device,64);run(device,2560);run_window(device);run_ple_window(device,64,3);run_ple_window(device,2560,4);}
    catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
