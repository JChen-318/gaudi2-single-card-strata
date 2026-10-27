#include "strata/hpu/moe.hpp"
#include "strata/hpu/fp8.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
using namespace strata::hpu;
namespace {
constexpr uint32_t H=128,F=64,E=4;
float weight(uint32_t role,int id,size_t i) {return float(int((i*13+role*7+uint32_t(id+2)*11)%23)-11)/128;}
ExpertSlices source(int id) {
    std::array<WeightSlice,3> slices;
    for(uint32_t role=0;role<3;++role) {
        auto bytes=std::make_shared<std::vector<uint8_t>>(H*F*2);
        for(size_t i=0;i<H*F;++i) {const uint16_t b=to_bf16(weight(role,id,i));std::memcpy(bytes->data()+2*i,&b,2);}
        slices[role]=WeightSlice::raw_rows(bytes,*bytes,30,role==2?F:H,role==2?H:F);
    }
    return {slices[0],slices[1],slices[2]};
}
std::vector<float> projection(std::span<const float> x,uint32_t role,int id,uint32_t rows,uint32_t cols) {
    float peak=0;for(float v:x) peak=std::max(peak,std::fabs(v));const float scale=peak*(1.0f/240)+1e-30f;
    std::vector<float> high(cols),low(cols),out(rows),w(cols);std::vector<uint8_t> codes(cols);
    for(uint32_t c=0;c<cols;++c) {const float a=x[c]/scale;high[c]=f32_from_fp8_143(fp8_143_from_f32_reference(a));low[c]=f32_from_fp8_143(fp8_143_from_f32_reference((a-high[c])*16));}
    for(uint32_t r=0;r<rows;++r) {
        for(uint32_t c=0;c<cols;++c) w[c]=weight(role,id,uint64_t(r)*cols+c);
        const float ws=quantize_fp8_row(w,codes.data());float main=0,lower=0;
        for(uint32_t c=0;c<cols;++c) {const float q=f32_from_fp8_143(codes[c]);main+=high[c]*q;lower+=low[c]*q;}
        out[r]=((main+lower/16)*scale)*ws;
    }
    return out;
}
std::vector<float> expert(std::span<const float> x,int id) {
    auto gate=projection(x,0,id,F,H),up=projection(x,1,id,F,H);
    for(uint32_t c=0;c<F;++c) gate[c]=(gate[c]/(1+std::exp(-gate[c])))*up[c];
    return projection(gate,2,id,H,F);
}
void compare(std::span<const float> actual,std::span<const float> expected) {
    if(actual.size()!=expected.size()) throw std::runtime_error("FP8 expert output size");
    double worst=0;for(size_t i=0;i<actual.size();++i) {const double error=std::abs(double(actual[i])-expected[i]);worst=std::max(worst,error);
        if(!std::isfinite(actual[i]) || error>1e-5+2e-4*std::abs(expected[i])) throw std::runtime_error("FP8 expert versus independent oracle mismatch");}
    std::cout<<"PASS FP8 expert max_abs_error="<<worst<<'\n';
}
}
int main() {
    try {
        Device device;strata::core::ModelGeometry g;g.n_embd=H;g.n_ff=F;g.n_expert=E;g.n_expert_used=2;g.n_layers=1;
        ExpertCache cache(device,g,[](uint32_t,int32_t id){return source(id);},1<<20,1<<16);cache.set_fp8(true);
        const auto before=device.hbm_allocations().used;
        {
            auto held=cache.acquire(0,1);const auto& w=held.get();
            if(held.bytes()!=w.gate.size()+w.up.size()+w.down.size() || device.hbm_allocations().used-before!=held.bytes())
                throw std::runtime_error("FP8 storage/cache/allocation accounting mismatch");
            ExpertExecutor executor(device,g,8);
            for(uint32_t rows:{1u,5u}) {
                std::vector<float> input(rows*H),expected;
                for(size_t i=0;i<input.size();++i) input[i]=float(int((i*7+3)%29)-14)/32;
                for(uint32_t r=0;r<rows;++r) {auto y=expert(std::span<const float>(input).subspan(r*H,H),1);expected.insert(expected.end(),y.begin(),y.end());}
                compare(executor.run(input,held),expected);
            }
        }
        Stream copy(device);HostBuffer zeros(device,E*H*2),shared(device,H*2);DeviceBuffer router(device,zeros.size()),shared_gate(device,shared.size());
        std::memset(zeros.data(),0,zeros.size());std::memset(shared.data(),0,shared.size());
        upload(copy,router,zeros,zeros.size());upload(copy,shared_gate,shared,shared.size());copy.synchronize();
        MoeBlock block(device,g,0,cache,{"w_router",&router},{"w_shared_gate",&shared_gate});
        for(uint32_t phase=0;phase<4;++phase) {
            std::vector<float> input(H),logits{float(phase),-.5f,1.5f,.7f};
            for(uint32_t c=0;c<H;++c) input[c]=phase==0?0:float(int((c*7+phase*3)%29)-14)/32;
            auto routes=route_reference(logits,1,E,2);std::vector<float> combined(H);
            for(uint32_t i=0;i<2;++i) {auto y=expert(input,routes.ids[i]);for(uint32_t c=0;c<H;++c) combined[c]+=y[c]*routes.weights[i];}
            auto y=expert(input,-1);for(uint32_t c=0;c<H;++c) combined[c]+=.3f*y[c];
            HcConfig hc{H,3,8,1e-6f};std::vector<float> residual(3*H),inject{-.7f,0,.8f},expected(3*H);
            for(size_t i=0;i<residual.size();++i) residual[i]=float(int(i%17)-8)/64;
            for(uint32_t r=0;r<3;++r) for(uint32_t c=0;c<H;++c) expected[r*H+c]=residual[r*H+c]+combined[c]*(2/(1+std::exp(-inject[r]/3)));
            compare(block.fused_ffn(logits,.3f,input,residual,inject,hc),expected);
            const std::vector<float> zero_logits(E,0);const auto ordinary_routes=route_reference(zero_logits,1,E,2);
            std::vector<float> ordinary(H);
            for(uint32_t i=0;i<2;++i) {auto value=expert(input,ordinary_routes.ids[i]);for(uint32_t c=0;c<H;++c) ordinary[c]+=value[c]*ordinary_routes.weights[i];}
            for(uint32_t c=0;c<H;++c) ordinary[c]+=.5f*y[c];
            compare(block.run(input),ordinary);
        }
        std::cout<<"PASS joined storage, single/multi-row executor, routed/shared fused HC, changing routes\n";
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
