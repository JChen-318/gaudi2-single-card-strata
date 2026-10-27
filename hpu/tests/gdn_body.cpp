#include "strata/hpu/gdn.hpp"
#include "gdn_fixture.hpp"
#include <cstring>
#include <map>

using namespace strata::hpu;
namespace {
struct Buffer {
    std::unique_ptr<HostBuffer> host;
    std::unique_ptr<DeviceBuffer> device;
    uint64_t elements;
    Buffer(const Device& d,uint64_t count) : host(std::make_unique<HostBuffer>(d,(count+64)*4)),
        device(std::make_unique<DeviceBuffer>(d,(count+64)*4)),elements(count) {
        std::fill_n(static_cast<float*>(host->data()),count+64,12345.0f);
    }
    void set(const std::vector<float>& data) {
        if(data.size()!=elements) throw std::logic_error("GDN input size mismatch");
        std::memcpy(host->data(),data.data(),data.size()*4);
    }
    std::vector<float> values() const {
        auto* p=static_cast<const float*>(host->data());
        for(uint64_t i=elements;i<elements+64;++i) if(p[i]!=12345.0f) throw std::runtime_error("TPC write exceeded tensor extent");
        return {p,p+elements};
    }
};
void run(const Device& device,int s,int hk,int hv) {
    gdn_test::Fixture f(s,hk,hv);
    strata::core::ModelGeometry geometry;
    geometry.ssm_state_size=s; geometry.ssm_k_heads=hk; geometry.ssm_v_heads=hv;
    geometry.ssm_conv_channels=f.c; geometry.ssm_value_dim=s*hv;
    auto recipe=build_gdn_body(device,geometry,true);
    std::map<std::string,Buffer> buffers;
    auto add=[&](const char* name,uint64_t count) { buffers.emplace(std::piecewise_construct,std::forward_as_tuple(name),std::forward_as_tuple(device,count)); };
    add("state",f.state.size()); add("state_out",f.state.size());
    add("conv_state",f.conv.size()); add("conv_state_out",f.conv.size());
    add("qkv",f.qkv.size()); add("alpha",hv); add("raw_beta",hv); add("z",hv*s);
    add("conv_weights",f.weights.size()); add("norm",s); add("dt",hv); add("ssm_a",hv);
    add("activated",f.c); add("q",hk*s); add("k",hk*s); add("v",hv*s); add("gate",hv); add("beta",hv);
    add("recurrent",hv*s); add("out",hv*s);
    buffers.at("state").set(f.state); buffers.at("conv_state").set(f.time_major_conv());
    buffers.at("conv_weights").set(f.time_major_weights()); buffers.at("norm").set(f.norm);
    buffers.at("dt").set(f.dt); buffers.at("ssm_a").set(f.a);
    Stream input(device),compute(device),readback(device);
    Event ready(device),done(device);
    for(auto& [name,b]:buffers) { (void)name; upload(input,*b.device,*b.host,b.host->size()); }
    ready.record(input); ready.synchronize();
    for(int phase=0;phase<4;++phase) {
        f.inputs(phase);
        buffers.at("qkv").set(f.qkv); buffers.at("alpha").set(f.alpha);
        buffers.at("raw_beta").set(f.raw_beta); buffers.at("z").set(f.z);
        for(auto name:{"qkv","alpha","raw_beta","z"}) {
            auto& b=buffers.at(name); upload(input,*b.device,*b.host,b.host->size());
        }
        ready.record(input); compute.wait(ready);
        std::vector<Binding> bindings;
        for(auto& [name,b]:buffers) bindings.push_back({name,b.device.get()});
        recipe->enqueue(compute,bindings); readback.wait(recipe->completion());
        for(auto name:{"state_out","conv_state_out","out","activated","q","k","v","gate","beta","recurrent"}) {
            auto& b=buffers.at(name); download(readback,*b.host,*b.device,b.host->size());
        }
        done.record(readback); f.reference_step(); done.synchronize();
        gdn_test::compare(buffers.at("conv_state_out").values(),f.time_major_conv(),"conv_state",0);
        for(auto pair:{std::pair{"activated",&f.activated},{"q",&f.q},{"k",&f.k},{"v",&f.v},
                       {"gate",&f.gate},{"beta",&f.beta},{"recurrent",&f.recurrent}})
            gdn_test::compare(buffers.at(pair.first).values(),*pair.second,pair.first,1e-4);
        const auto se=gdn_test::compare(buffers.at("state_out").values(),f.state,"state",1e-4);
        const auto oe=gdn_test::compare(buffers.at("out").values(),f.out,"out",2e-4);
        std::cout << "PASS real HPU GDN body S=" << s << " hk=" << hk << " hv=" << hv << " phase=" << phase
                  << " state_error=" << se << " output_error=" << oe << " tensor_canaries=ok\n";
        buffers.at("state").device.swap(buffers.at("state_out").device);
        buffers.at("conv_state").device.swap(buffers.at("conv_state_out").device);
    }
}
}
int main() {
    try { Device device; run(device,64,1,3); run(device,128,16,48); }
    catch(const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
