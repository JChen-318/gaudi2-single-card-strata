#include "strata/hpu/hc_ple.hpp"
#include "strata/hpu/model_head.hpp"
#include <iostream>
using namespace strata::hpu;
int main() {
    bool initialized=false;
    try {
        configure_kernel_libraries();check(synInitialize(),"offline synInitialize");initialized=true;
        for(uint32_t h:{64,2560}) for(int kind=0;kind<5;++kind) {
            HcConfig c;c.hidden=h;GraphBuilder graph(OfflineCompilation{});
            const auto name="strata_hc_ple_"+std::to_string(h)+"_"+std::to_string(kind);
            if(kind<2) define_hc_read(graph,c,kind==1);
            else if(kind==2) define_hc_write(graph,c);
            else define_ple_block(graph,c,kind==4);
            auto bytes=graph.compile_offline(name);
            std::cout<<"PASS offline HC/PLE H="<<h<<" kind="<<kind<<" workspace_bytes="<<bytes<<" no_device_acquired\n";
        }
        for(const uint32_t vocabulary:{17u,248320u}) {
            GraphBuilder graph(OfflineCompilation{});define_model_head(graph,2560,vocabulary);
            const auto bytes=graph.compile_offline("strata_head_"+std::to_string(vocabulary));
            std::cout<<"PASS offline model head vocab="<<vocabulary<<" workspace_bytes="<<bytes<<" no_device_acquired\n";
        }
        for(uint32_t h:{64u,2560u}) for(uint32_t rows:{1u,8u,128u}) for(bool write:{false,true}) {
            HcConfig config;config.hidden=h;GraphBuilder graph(OfflineCompilation{});
            if(write) define_hc_write_window(graph,config,rows);else define_hc_read_window(graph,config,rows);
            const auto bytes=graph.compile_offline("strata_hc_window_"+std::to_string(h)+"_"+std::to_string(rows)+(write?"_write":"_read"));
            std::cout<<"PASS offline batched HC H="<<h<<" rows="<<rows<<" write="<<write<<" workspace_bytes="<<bytes<<" no_device_acquired\n";
        }
        for(uint32_t rows:{2u,5u}) for(bool write:{false,true}) {
            HcConfig config{64,3,8,1e-6f};GraphBuilder graph(OfflineCompilation{});
            if(write) define_hc_write_window(graph,config,rows);else define_hc_read_window(graph,config,rows);
            (void)graph.compile_offline("strata_hc_window_rebind_fixture_"+std::to_string(rows)+(write?"_write":"_read"));
            std::cout<<"PASS offline HC hardware rebind fixture rows="<<rows<<" write="<<write<<" no_device_acquired\n";
        }
        for(uint32_t h:{64u,2560u}) for(uint32_t rows:{1u,3u,8u}) {
            HcConfig config;config.hidden=h;GraphBuilder graph(OfflineCompilation{});define_ple_window(graph,config,rows);
            const auto bytes=graph.compile_offline("strata_ple_window_"+std::to_string(h)+"_"+std::to_string(rows));
            std::cout<<"PASS offline batched PLE H="<<h<<" rows="<<rows<<" workspace_bytes="<<bytes<<" no_device_acquired\n";
        }
        for(uint32_t rows:{2u,5u}) {
            GraphBuilder graph(OfflineCompilation{});define_ple_window(graph,HcConfig{64,3,8,1e-6f},rows);
            const auto bytes=graph.compile_offline("strata_ple_window_non_power_"+std::to_string(rows));
            std::cout<<"PASS offline batched PLE non-power streams3 rows="<<rows<<" workspace_bytes="<<bytes<<" no_device_acquired\n";
        }
        for(uint32_t h:{64u,2560u}) for(uint32_t rows:{1u,3u,7u}) {
            HcConfig config;config.hidden=h;GraphBuilder graph(OfflineCompilation{});define_hc_read_window(graph,config,rows,true);
            const auto bytes=graph.compile_offline("strata_hc_final_window_"+std::to_string(h)+"_"+std::to_string(rows));
            std::cout<<"PASS offline final HC H="<<h<<" rows="<<rows<<" workspace_bytes="<<bytes<<" no_device_acquired\n";
        }
        check(synDestroy(),"offline synDestroy");initialized=false;
    } catch(const std::exception& e) {
        if(initialized) (void)synDestroy();
        std::cerr<<e.what()<<'\n';return 1;
    }
}
