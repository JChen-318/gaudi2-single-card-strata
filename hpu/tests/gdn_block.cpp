#include "strata/hpu/gdn.hpp"
#include "strata/hpu/gdn_block.hpp"
#include "strata/hpu/gdn_window.hpp"
#include "strata/hpu/session_flow.hpp"
#include "gdn_fixture.hpp"
#include <cstring>
#include <map>

using namespace strata::hpu;
namespace {
struct Buffer {
    HostBuffer host;
    std::unique_ptr<DeviceBuffer> device;
    bool bf16;
    uint64_t elements;
    Buffer(const Device& d,uint64_t count,bool is_bf16=false)
        : host(d,count*(is_bf16?2:4)),device(std::make_unique<DeviceBuffer>(d,host.size())),bf16(is_bf16),elements(count) {}
    void set(const std::vector<float>& values) {
        if(values.size()!=elements) throw std::logic_error("GDN block input size mismatch");
        if(bf16) for(size_t i=0;i<values.size();++i) static_cast<uint16_t*>(host.data())[i]=to_bf16(values[i]);
        else std::memcpy(host.data(),values.data(),values.size()*4);
    }
    std::vector<float> values() const {
        std::vector<float> result(elements);
        if(bf16) for(uint64_t i=0;i<elements;++i) result[i]=from_bf16(static_cast<const uint16_t*>(host.data())[i]);
        else std::memcpy(result.data(),host.data(),elements*4);
        return result;
    }
};
std::vector<float> weight(uint64_t rows,uint64_t columns,int seed) {
    std::vector<float> result(rows*columns);
    for(uint64_t row=0;row<rows;++row) for(uint64_t col=0;col<columns;++col)
        result[row*columns+col]=float(int((row*13+col*17+uint64_t(seed))%19)-9)/256;
    return result;
}
std::vector<float> projection(const std::vector<float>& x,const std::vector<float>& w,bool round_output=true) {
    const auto h=x.size(),n=w.size()/h;
    std::vector<float> output(n);
    for(size_t row=0;row<n;++row) {
        float sum=0;
        for(size_t col=0;col<h;++col) sum+=x[col]*w[row*h+col];
        output[row]=round_output?from_bf16(to_bf16(sum)):sum;
    }
    return output;
}
void run(const Device& device,int h,int s,int hk,int hv) {
    gdn_test::Fixture f(s,hk,hv);
    strata::core::ModelGeometry geometry;
    geometry.n_embd=h; geometry.ssm_state_size=s; geometry.ssm_k_heads=hk; geometry.ssm_v_heads=hv;
    geometry.ssm_conv_channels=f.c; geometry.ssm_value_dim=s*hv;
    auto recipe=build_gdn_block(device,geometry);
    const auto wq=weight(f.c,h,1),wz=weight(hv*s,h,7),wa=weight(hv,h,11),wb=weight(hv,h,19),wo=weight(h,hv*s,23);
    std::map<std::string,Buffer> buffers;
    auto add=[&](const char* name,uint64_t count,bool bf16=false) {
        buffers.emplace(std::piecewise_construct,std::forward_as_tuple(name),std::forward_as_tuple(device,count,bf16));
    };
    add("x",h,true); add("w_qkv",wq.size(),true); add("w_gate",wz.size(),true);
    add("w_alpha",wa.size(),true); add("w_beta",wb.size(),true); add("w_out",wo.size(),true);
    add("state",f.state.size()); add("state_out",f.state.size());
    add("conv_state",f.conv.size()); add("conv_state_out",f.conv.size());
    add("conv_weights",f.weights.size()); add("norm",s); add("dt",hv); add("ssm_a",hv); add("out",h,true);
    buffers.at("w_qkv").set(wq); buffers.at("w_gate").set(wz); buffers.at("w_alpha").set(wa);
    buffers.at("w_beta").set(wb); buffers.at("w_out").set(wo);
    buffers.at("state").set(f.state); buffers.at("conv_state").set(f.time_major_conv());
    buffers.at("conv_weights").set(f.time_major_weights()); buffers.at("norm").set(f.norm);
    buffers.at("dt").set(f.dt); buffers.at("ssm_a").set(f.a);
    Stream input(device),compute(device),readback(device);
    Event ready(device),done(device);
    for(auto& [name,b]:buffers) if(name!="x" && name!="out" && name!="state_out" && name!="conv_state_out")
        upload(input,*b.device,b.host,b.host.size());
    ready.record(input); ready.synchronize();
    for(int phase=0;phase<4;++phase) {
        std::vector<float> x(h);
        for(int col=0;col<h;++col) x[col]=float((col*7+phase*11)%17-8)/32;
        f.qkv=projection(x,wq); f.z=projection(x,wz); f.alpha=projection(x,wa); f.raw_beta=projection(x,wb);
        buffers.at("x").set(x); auto& bx=buffers.at("x"); upload(input,*bx.device,bx.host,bx.host.size());
        ready.record(input); compute.wait(ready);
        std::vector<Binding> bindings;
        for(auto& [name,b]:buffers) bindings.push_back({name,b.device.get()});
        recipe->enqueue(compute,bindings); readback.wait(recipe->completion());
        for(auto name:{"state_out","conv_state_out","out"}) { auto& b=buffers.at(name); download(readback,b.host,*b.device,b.host.size()); }
        done.record(readback);
        f.reference_step();
        auto y=f.out;
        for(auto& value:y) value=from_bf16(to_bf16(value));
        const auto expected=projection(y,wo);
        done.synchronize();
        gdn_test::compare(buffers.at("conv_state_out").values(),f.time_major_conv(),"conv_state",0);
        const auto se=gdn_test::compare(buffers.at("state_out").values(),f.state,"block state",2e-4);
        const auto actual=buffers.at("out").values();
        double error2=0,signal2=0,max_error=0;
        for(size_t i=0;i<actual.size();++i) {
            if(!std::isfinite(actual[i])) throw std::runtime_error("non-finite GDN block output");
            const double delta=double(actual[i])-expected[i]; error2+=delta*delta; signal2+=double(expected[i])*expected[i];
            max_error=std::max(max_error,std::abs(delta));
        }
        const double relative=std::sqrt(error2/std::max(signal2,1e-24));
        if(relative>0.02 && max_error>0.002) throw std::runtime_error("BF16 GDN block output parity failed");
        std::cout << "PASS real HPU GDN BF16 block H=" << h << " S=" << s << " phase=" << phase
                  << " state_error=" << se << " output_relative_l2=" << relative << " max_output_error=" << max_error << '\n';
        buffers.at("state").device.swap(buffers.at("state_out").device);
        buffers.at("conv_state").device.swap(buffers.at("conv_state_out").device);
    }
    std::vector<Binding> weights;
    for(const auto* name:{"w_qkv","w_gate","w_alpha","w_beta","w_out","conv_weights","norm","dt","ssm_a"}) weights.push_back({name,buffers.at(name).device.get()});
    GdnBlock block(device,geometry,weights);std::vector<float> x(h,.125f);
    (void)block.decode(x);const auto point=block.checkpoint(),snapshot=block.snapshot();
    const auto expected=block.decode(x);block.restore(point);
    gdn_test::compare(block.decode(x),expected,"GDN runtime rewind",0);
    block.reset();block.restore(snapshot);gdn_test::compare(block.decode(x),expected,"GDN runtime persistent restore",0);
    block.reset();block.restore(snapshot);gdn_test::compare(block.decode(x),expected,"GDN runtime repeated restore",0);
    x[0]=NAN;bool rejected=false;try {(void)block.decode(x);} catch(const std::exception&) {rejected=true;}
    if(!rejected || block.size()!=2) throw std::runtime_error("GDN runtime invalid input changed position");
    std::cout<<"PASS real HPU GDN runtime rewind, repeated snapshot restore/reset and invalid input atomicity H="<<h<<'\n';
    GdnWindow shared(device,geometry);GdnBlock sequential(device,geometry,weights),batched(device,geometry,weights,&shared);
    auto compare=[&](GdnBlock& scalar,GdnBlock& window,uint32_t rows) {
        std::vector<float> input(uint64_t(rows)*h),expected;
        for(size_t i=0;i<input.size();++i) input[i]=float(int((i*7+scalar.size()*11)%29)-14)/32;
        for(uint32_t row=0;row<rows;++row) {
            const auto value=scalar.decode(std::span<const float>(input).subspan(uint64_t(row)*h,h));
            expected.insert(expected.end(),value.begin(),value.end());
        }
        const auto actual=window.prefill(input);gdn_test::compare(actual,expected,"native batched GDN versus singleton",.004);
        if(scalar.size()!=window.size()) throw std::runtime_error("GDN window position differs from singleton");
        std::vector<float> next(h,.125f);gdn_test::compare(window.decode(next),scalar.decode(next),"GDN window state continuation",.004);
    };
    for(auto rows:{1u,5u,8u,17u}) compare(sequential,batched,rows);
    const auto saved=batched.size();int polls=0;bool cancelled=false;
    try {(void)batched.prefill(std::vector<float>(17*h,.125f),[&] {return ++polls==3;});}
    catch(const Cancelled&) {cancelled=true;}
    if(!cancelled || batched.size()!=saved) throw std::runtime_error("GDN partial native window was accepted after cancellation");
    compare(sequential,batched,2);
    if(h==32) {
        // Same T with a different layer's borrowed weights, then return to the
        // first owner: no executor-local accepted state or stale binding.
        Buffer alternate(device,wo.size(),true);auto other=wo;for(auto& value:other) value=-value;alternate.set(other);
        upload(input,*alternate.device,alternate.host,alternate.host.size());ready.record(input);ready.synchronize();
        auto other_weights=weights;for(auto& binding:other_weights) if(binding.name=="w_out") binding.buffer=alternate.device.get();
        GdnBlock other_scalar(device,geometry,other_weights),other_window(device,geometry,other_weights,&shared);
        compare(other_scalar,other_window,2);compare(sequential,batched,2);
    }
    std::cout<<"PASS real HPU shared native GDN window projections/state H="<<h<<" subwindow_cap=8 cancellation/rebind/continuation\n";
}
}
int main() {
    try { Device device; run(device,32,64,2,6); run(device,2560,128,16,48); }
    catch(const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
