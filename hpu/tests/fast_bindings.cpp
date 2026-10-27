#include "strata/hpu/host_recipe.hpp"
#include <algorithm>
#include <array>
#include <iostream>
#include <vector>
using namespace strata::hpu;
template<class F> void rejects(F f) {bool rejected=false;try {f();} catch(const std::exception&) {rejected=true;}if(!rejected) throw std::runtime_error("invalid binding accepted");}
int main() {
    try {
        Device device;Stream copy(device),compute(device);
        HostBuffer ha(device,256),hb(device,256),hc(device,256),result(device,256);
        DeviceBuffer a(device,256),b(device,256),c(device,256),out(device,256),padding(device,512);
        for(size_t i=0;i<64;++i) {static_cast<float*>(ha.data())[i]=float(i);static_cast<float*>(hb.data())[i]=2;static_cast<float*>(hc.data())[i]=-3;}
        upload(copy,a,ha,256);upload(copy,b,hb,256);upload(copy,c,hc,256);copy.synchronize();
        const auto graph=[&] {auto g=std::make_unique<GraphBuilder>(device);
            auto x=g->tensor({"x",DType::F32,{1,64}}),y=g->tensor({"y",DType::F32,{1,64}}),z=g->tensor({"out",DType::F32,{1,64}});
            g->add(x,y,z);return g->compile("binding_parity");};
        auto r=graph();std::array<Binding,3> bindings{{{"x",&a},{"y",&b},{"out",&out}}};
        const auto check_output=[&](float bias) {download(copy,result,out,256);copy.synchronize();for(size_t i=0;i<64;++i) if(static_cast<float*>(result.data())[i]!=float(i)+bias) throw std::runtime_error("stale cached binding address/order");};
        r->enqueue(compute,bindings);compute.synchronize();check_output(2);
        bindings[1].buffer=&c;r->enqueue(compute,bindings);compute.synchronize();check_output(-3);
        std::rotate(bindings.begin(),bindings.begin()+1,bindings.end());r->enqueue(compute,bindings);compute.synchronize();check_output(-3);
        auto bad=bindings;bad[1].name=bad[0].name;rejects([&]{r->enqueue(compute,bad);});
        bad=bindings;bad[0].name="absent";rejects([&]{r->enqueue(compute,bad);});
        bad=bindings;bad[0].buffer=nullptr;rejects([&]{r->enqueue(compute,bad);});
        bad=bindings;bad[0].buffer=&padding;bad[0].offset=1;rejects([&]{r->enqueue(compute,bad);});
        bad=bindings;bad[0].offset=256;rejects([&]{r->enqueue(compute,bad);});
        r->enqueue(compute,bindings);compute.synchronize();check_output(-3);
        HostRecipe host(device,graph(),std::array<Binding,1>{{{"y",&b}}},{"out"});
        std::vector<float> values(64);for(size_t i=0;i<64;++i) values[i]=float(i);
        host.set("x",values);host.run();
        for(size_t i=0;i<64;++i) if(host.output("out")[i]!=float(i)+2) throw std::runtime_error("host initial binding mismatch");
        host.bind_external({"y",&c});host.run();
        for(size_t i=0;i<64;++i) if(host.output("out")[i]!=float(i)-3) throw std::runtime_error("host rebind mismatch");
        rejects([&]{host.bind_external({"x",&a});});rejects([&]{host.bind_external({"out",&out});});
        rejects([&]{host.bind_external({"y",nullptr});});rejects([&]{host.bind_external({"y",&c,256});});
        rejects([&]{host.bind_external({"y",&padding,1});});
        rejects([&]{host.set("absent",values);});
        host.bind_external({"y",&b});host.set("x",values);host.run();
        for(size_t i=0;i<64;++i) if(host.output("out")[i]!=float(i)+2) throw std::runtime_error("host recovery failed");
        std::cout<<"PASS cached recipe reorder/rebind/duplicate/unknown/null/range/alignment and host-slot recovery\n";
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
