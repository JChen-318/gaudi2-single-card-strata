#include "strata/hpu/control_graph.hpp"
#include <iostream>
using namespace strata::hpu;
int main() {
    bool initialized=false;
    try {
        configure_kernel_libraries();check(synInitialize(),"offline control synInitialize");initialized=true;
        for(uint32_t hidden:{64u,2560u}) for(auto mode:{ControlVectorMode::Add,ControlVectorMode::Project}) {
            GraphBuilder graph(OfflineCompilation{});define_control_vector(graph,hidden,4,mode);
            std::cout<<"PASS F32 control vector hidden="<<hidden<<" mode="<<uint32_t(mode)<<" workspace="<<graph.compile_offline("strata_control_"+std::to_string(hidden)+"_"+std::to_string(uint32_t(mode)))<<" no_device_acquired\n";
        }
        check(synDestroy(),"offline control synDestroy");initialized=false;
    } catch(const std::exception& e) {if(initialized) (void)synDestroy();std::cerr<<e.what()<<'\n';return 1;}
}
