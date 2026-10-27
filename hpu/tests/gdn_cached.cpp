#include "strata/hpu/host_recipe.hpp"
#include "strata/hpu/gdn_abi.hpp"
#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <iostream>
using namespace strata::hpu;
int main() {
    try {
        Device device;
        struct Shape {uint32_t width,key_heads,value_heads;};
        for(const auto& c:{Shape{64,1,8},Shape{128,2,16},Shape{128,4,48},Shape{128,16,48},Shape{256,4,16}}) {
            const uint64_t s=c.width,hk=c.key_heads,hv=c.value_heads;
            std::vector<std::unique_ptr<HostRecipe>> graphs;
            for(const char* guid:{"strata_gdn_step_f32","strata_gdn_step_cached_f32"}) {
                GraphBuilder g(device);std::vector<TensorId> inputs;
                inputs.push_back(g.tensor({"state",DType::F32,{s,hv,s}}));
                inputs.push_back(g.tensor({"q",DType::F32,{hk,s}}));inputs.push_back(g.tensor({"k",DType::F32,{hk,s}}));
                inputs.push_back(g.tensor({"v",DType::F32,{hv,s}}));inputs.push_back(g.tensor({"gate",DType::F32,{hv}}));
                inputs.push_back(g.tensor({"beta",DType::F32,{hv}}));
                const TensorId outputs[]={g.tensor({"state_out",DType::F32,{s,hv,s}}),g.tensor({"out",DType::F32,{hv,s}})};
                abi::GdnStep args{int32_t(s),int32_t(hk)};g.op(guid,inputs,outputs,&args,sizeof(args));
                graphs.push_back(std::make_unique<HostRecipe>(device,g.compile(guid),std::span<const Binding>{},std::vector<std::string>{"state_out","out"}));
            }
            for(int phase=0;phase<4;++phase) {
                std::vector<float> state(s*hv*s),q(hk*s),k(q.size()),v(hv*s),gate(hv),beta(hv);
                for(uint64_t i=0;i<state.size();++i) state[i]=phase==0?0:std::ldexp(float(int((i*13+3)%41)-20)/512,phase==3?int(i%11)-5:0);
                for(uint64_t head=0;head<hk;++head) for(uint64_t j=0;j<s;++j) {
                    q[head*s+j]=phase==1?float(j==(head*7+3)%s):float(int((head*17+j*7)%31)-15)/(float(s)*11);
                    k[head*s+j]=phase==1?float(j==(head*11+7)%s):float(int((head*11+j*13)%37)-18)/(float(s)*3);
                }
                for(uint64_t i=0;i<v.size();++i) v[i]=float(int((i*19+1)%43)-21)/128;
                for(uint64_t head=0;head<hv;++head) {gate[head]=phase==0?0:float(-int(1+head%13))/17;beta[head]=phase==0?0:phase==1?1:float(1+head%7)/9;}
                for(auto& graph:graphs) {
                    graph->set("state",state);graph->set("q",q);graph->set("k",k);graph->set("v",v);graph->set("gate",gate);graph->set("beta",beta);graph->run();
                }
                for(const char* name:{"state_out","out"}) {
                    const auto old=graphs[0]->output(name),cached=graphs[1]->output(name);
                    if(old.size()!=cached.size()) throw std::runtime_error("cached GDN output shape mismatch");
                    for(size_t i=0;i<old.size();++i) if(std::bit_cast<uint32_t>(old[i])!=std::bit_cast<uint32_t>(cached[i])) {
                        std::cout<<"mismatch "<<name<<' '<<i<<" old="<<old[i]<<" cached="<<cached[i]<<'\n';
                        throw std::runtime_error("cached GDN is not bit exact with original scalar-load recurrence");
                    }
                }
                std::cout<<"PASS width="<<s<<" hk="<<hk<<" hv="<<hv<<" phase="<<phase<<" bit_exact=1\n";
            }
        }
    } catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
}
