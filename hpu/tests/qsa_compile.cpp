#include "strata/hpu/qsa.hpp"
#include "strata/hpu/window_limits.hpp"
#include <iostream>

using namespace strata::hpu;
int main() {
    bool initialized=false;
    try {
        configure_kernel_libraries();check(synInitialize(),"offline synInitialize");initialized=true;
        const QsaConfig config;
        for(uint32_t bucket:{1,8,128,512,2048,2051}) {
            GraphBuilder graph(OfflineCompilation{});define_qsa_attention(graph,config,bucket);
            auto bytes=graph.compile_offline("strata_qsa_attention_"+std::to_string(bucket));
            std::cout<<"PASS offline QSA attention HQ24 HKV2 D256 bucket="<<bucket<<" workspace_bytes="<<bytes<<" no_device_acquired\n";
            GraphBuilder tapped(OfflineCompilation{});define_qsa_attention(tapped,config,bucket,true);
            const auto tap_bytes=tapped.compile_offline("strata_qsa_half_attention_"+std::to_string(bucket));
            if(tap_bytes>bytes+(4ull<<20)) throw std::runtime_error("QSA half tap workspace exceeds incremental allowance");
            std::cout<<"PASS offline QSA pre-gate half output bucket="<<bucket<<" workspace_bytes="<<tap_bytes<<" no_device_acquired\n";
        }
        {GraphBuilder graph(OfflineCompilation{});define_qsa_projections(graph,config,2560);
         std::cout<<"PASS offline QSA projections BF16 inputs/F32 output workspace_bytes="<<graph.compile_offline("strata_qsa_projections")<<'\n';}
        {GraphBuilder graph(OfflineCompilation{});define_qsa_output(graph,config,2560);
         std::cout<<"PASS offline QSA output projection workspace_bytes="<<graph.compile_offline("strata_qsa_output")<<'\n';}
        // The window factories accept up to kResidentWindowRows after the fused-window widening, so every
        // power-of-two row count up to it is legal and the refusal boundary sits one row above.
        for(uint32_t rows:{1u,3u,8u,16u,32u,64u,128u,256u}) {
            {GraphBuilder graph(OfflineCompilation{});define_qsa_projections_window(graph,config,2560,rows);
             const auto bytes=graph.compile_offline("strata_qsa_projection_window_"+std::to_string(rows));
             std::cout<<"PASS offline QSA window projections rows="<<rows<<" workspace_bytes="<<bytes<<" no_device_acquired\n";}
            {GraphBuilder graph(OfflineCompilation{});define_qsa_output_window(graph,config,2560,rows);
             const auto bytes=graph.compile_offline("strata_qsa_output_window_"+std::to_string(rows));
             std::cout<<"PASS offline QSA window output rows="<<rows<<" workspace_bytes="<<bytes<<" no_device_acquired\n";}
            for(uint32_t bucket:{1u,8u,2051u}) {
                GraphBuilder graph(OfflineCompilation{});define_qsa_attention_window(graph,config,bucket,rows);
                const auto bytes=graph.compile_offline("strata_qsa_attention_window_"+std::to_string(rows)+"_"+std::to_string(bucket));
                std::cout<<"PASS offline QSA window attention rows="<<rows<<" bucket="<<bucket<<" workspace_bytes="<<bytes<<" no_device_acquired\n";
            }
        }
        for(uint32_t rows:{0u,kResidentWindowRows+1}) {
            auto rejects=[](auto call) {bool failed=false;try {call();} catch(const std::exception&) {failed=true;}if(!failed) throw std::runtime_error("invalid QSA graph window accepted");};
            GraphBuilder graph(OfflineCompilation{});
            rejects([&] {define_qsa_projections_window(graph,config,2560,rows);});
            rejects([&] {define_qsa_output_window(graph,config,2560,rows);});
            rejects([&] {define_qsa_attention_window(graph,config,8,rows);});
        }
        auto small=config;small.query_heads=6;small.kv_heads=2;small.width=64;small.rotary=32;small.index_heads=2;small.index_width=64;small.capacity=32;small.top_k=4;
        for(uint32_t rows:{2u,3u}) for(uint32_t kind:{0u,1u,2u}) {
            GraphBuilder graph(OfflineCompilation{});
            if(kind==0) define_qsa_projections_window(graph,small,64,rows);
            if(kind==1) define_qsa_attention_window(graph,small,7,rows);
            if(kind==2) define_qsa_output_window(graph,small,64,rows);
            const auto bytes=graph.compile_offline("strata_qsa_window_rebind_"+std::to_string(kind)+"_"+std::to_string(rows));
            std::cout<<"PASS offline QSA window rebind fixture rows="<<rows<<" kind="<<kind<<" workspace_bytes="<<bytes<<" no_device_acquired\n";
        }
        for(uint32_t rows:{1u,3u,8u}) for(uint32_t bucket:{128u,2048u,32768u}) {
            QsaConfig dense;dense.top_k=dense.capacity;GraphBuilder graph(OfflineCompilation{});define_qsa_attention_window(graph,dense,bucket,rows);
            const auto bytes=graph.compile_offline("strata_mtp_dense_window_"+std::to_string(rows)+"_"+std::to_string(bucket));
            const uint64_t allowance=(64ull<<20)+uint64_t(rows)*dense.query_heads*bucket*16+uint64_t(rows)*dense.query_heads*dense.width*24+uint64_t(rows)*dense.kv_heads*bucket*dense.width*8;
            if(bytes>allowance) throw std::runtime_error("MTP dense window workspace exceeds allowance");
            std::cout<<"PASS offline dense MTP attention rows="<<rows<<" bucket="<<bucket<<" workspace_bytes="<<bytes<<" no_device_acquired\n";
        }
        check(synDestroy(),"offline synDestroy");initialized=false;
    } catch(const std::exception& e) {
        if(initialized) (void)synDestroy();
        std::cerr<<e.what()<<'\n';return 1;
    }
}
