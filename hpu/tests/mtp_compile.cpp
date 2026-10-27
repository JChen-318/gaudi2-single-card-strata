#include "strata/hpu/mtp.hpp"
#include "strata/hpu/qsa.hpp"
#include "strata/hpu/model_head.hpp"
#include <iostream>
using namespace strata::hpu;
int main() {
    bool initialized=false;
    try {
        configure_kernel_libraries();check(synInitialize(),"offline MTP synInitialize");initialized=true;
        {GraphBuilder graph(OfflineCompilation{});define_mtp_input(graph,2560,4);
         std::cout<<"PASS MTP double FC/broadcast input workspace="<<graph.compile_offline("strata_mtp_input")<<" no_device_acquired\n";}
        QsaConfig c;c.top_k=c.capacity;
        {GraphBuilder graph(OfflineCompilation{});define_mtp_projections(graph,c,2560);
         std::cout<<"PASS MTP Q/gate/K/V projections workspace="<<graph.compile_offline("strata_mtp_projections")<<" no_device_acquired\n";}
        for(uint32_t bucket:{1u,8u,2048u,32768u}) {
            GraphBuilder graph(OfflineCompilation{});define_qsa_attention(graph,c,bucket);
            std::cout<<"PASS MTP dense causal attention bucket="<<bucket<<" workspace="<<graph.compile_offline("strata_mtp_dense_"+std::to_string(bucket))<<" no_device_acquired\n";
        }
        {GraphBuilder graph(OfflineCompilation{});define_model_head(graph,2560,3);
         std::cout<<"PASS MTP subset head non-tile output width=3 workspace="<<graph.compile_offline("strata_mtp_subset_head")<<" no_device_acquired\n";}
        check(synDestroy(),"offline MTP synDestroy");initialized=false;
    } catch(const std::exception& error) {if(initialized) (void)synDestroy();std::cerr<<error.what()<<'\n';return 1;}
}
