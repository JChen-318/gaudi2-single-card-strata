#include "strata/hpu/expert.hpp"
#include <iostream>
using namespace strata::hpu;
int main() {
    bool initialized=false;
    try {
        configure_kernel_libraries();check(synInitialize(),"offline synInitialize");initialized=true;
        const strata::core::ModelGeometry geometry;
        for(uint32_t tokens:{1,8,32,128}) for(bool router:{false,true}) {
            GraphBuilder graph(OfflineCompilation{});
            if(router) define_moe_router(graph,geometry,tokens);else define_bf16_expert(graph,geometry,tokens);
            const auto name=std::string(router?"strata_moe_router_":"strata_moe_expert_")+std::to_string(tokens);
            std::cout<<"PASS offline "<<name<<" workspace_bytes="<<graph.compile_offline(name)<<" no_device_acquired\n";
        }
        check(synDestroy(),"offline synDestroy");initialized=false;
    } catch(const std::exception& e) {
        if(initialized) (void)synDestroy();
        std::cerr<<e.what()<<'\n';return 1;
    }
}
