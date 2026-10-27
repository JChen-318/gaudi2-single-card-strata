#include "strata/hpu/model_head.hpp"
#include <array>
#include <iostream>
using namespace strata::hpu;
int main() {
    bool initialized=false;
    try {
        configure_kernel_libraries();check(synInitialize(),"offline packed synInitialize");initialized=true;
        for(auto type:{2u,3u,6u,7u,8u,20u,42u}) for(auto batch:{1u,8u,128u}) {
            GraphBuilder graph(OfflineCompilation{});define_packed_matmul(graph,2560,batch==1?248320:640,batch,type);
            const auto bytes=graph.compile_offline("strata_packed_"+std::to_string(type)+"_"+std::to_string(batch));
            std::cout<<"PASS packed offline type="<<type<<" batch="<<batch<<" workspace="<<bytes<<" no_device_acquired\n";
        }
        // The Q8_1 activations are quantized on the device now, so one recipe holds the
        // quantizer and the dot, and the codes/metadata are produced and consumed inside the
        // graph. This is the only device-free proof that Synapse will compile that chain.
        for(auto type:{2u,8u,11u,14u,16u,23u,29u,40u,42u}) for(auto batch:{1u,8u,128u}) {
            GraphBuilder graph(OfflineCompilation{});
            define_packed_matmul(graph,2560,batch==1?248320:640,batch,type,PackedActivation::Q8_1,true);
            const auto bytes=graph.compile_offline("strata_packed_quantized_"+std::to_string(type)+"_"+std::to_string(batch));
            if(bytes>(16ull<<20)) throw std::runtime_error("chained packed graph workspace exceeds allowance");
            std::cout<<"PASS packed offline with device quantizer type="<<type<<" batch="<<batch
                <<" workspace="<<bytes<<" no_device_acquired\n";
        }
        for(auto dimensions:{std::array<uint32_t,2>{2560,2560},std::array<uint32_t,2>{2560,12288},std::array<uint32_t,2>{2560,512},std::array<uint32_t,2>{6144,2560}}) for(auto type:{2u,8u}) {
            GraphBuilder graph(OfflineCompilation{});define_packed_matmul(graph,dimensions[0],dimensions[1],1,type);
            (void)graph.compile_offline("strata_mtp_packed_projection_"+std::to_string(type)+"_"+std::to_string(dimensions[0])+"_"+std::to_string(dimensions[1]));
        }
        for(auto batch:{1u,8u,128u}) for(bool down:{false,true}) {
            GraphBuilder graph(OfflineCompilation{});define_packed_matmul(graph,down?640:2560,down?2560:640,batch,42,PackedActivation::RoutedS2);
            (void)graph.compile_offline("strata_routed_s2_"+std::to_string(batch)+(down?"_down":"_gate"));
        }
        for(auto type:{2u,3u,6u,7u,8u,20u}) for(auto batch:{1u,8u,32u,128u}) for(bool down:{false,true}) {
            GraphBuilder graph(OfflineCompilation{});define_packed_matmul(graph,down?640:2560,down?2560:640,batch,type,PackedActivation::RoutedQ8);
            const auto bytes=graph.compile_offline("strata_routed_q8_"+std::to_string(type)+"_"+std::to_string(batch)+(down?"_down":"_gate"));
            if(bytes>(16ull<<20)) throw std::runtime_error("routed packed workspace exceeds allowance");
            std::cout<<"PASS offline routed Q8 type="<<type<<" batch="<<batch<<" workspace="<<bytes<<" no_device_acquired\n";
        }
        for(auto batch:{1u,8u,32u,128u}) for(bool down:{false,true}) {
            GraphBuilder graph(OfflineCompilation{});define_model_head(graph,down?640:2560,down?2560:640,batch);
            (void)graph.compile_offline("strata_scaled_expert_f32_"+std::to_string(batch)+(down?"_down":"_gate"));
        }
        // Routed K-quant gate/up graphs (the K families need 256-value blocks, so a
        // 640-column projection is never packed) and the shared/head rejection.
        for(auto type:{11u,12u,13u,14u,16u,17u,18u,21u,22u,23u,29u,40u}) for(auto batch:{1u,8u,32u,128u}) {
            GraphBuilder graph(OfflineCompilation{});define_packed_matmul(graph,2560,640,batch,type,PackedActivation::RoutedQ8);
            const auto bytes=graph.compile_offline("strata_routed_k_"+std::to_string(type)+"_"+std::to_string(batch));
            if(bytes>(16ull<<20)) throw std::runtime_error("routed K-quant workspace exceeds allowance");
            std::cout<<"PASS offline routed K-quant type="<<type<<" batch="<<batch<<" workspace="<<bytes<<" no_device_acquired\n";
        }
        for(uint32_t type:{11u,12u,13u,14u,16u,17u,18u,21u,22u,23u,29u}) {
            GraphBuilder graph(OfflineCompilation{});bool layout=false;
            try{define_packed_matmul(graph,640,2560,1,type,PackedActivation::RoutedQ8);}catch(const std::invalid_argument&){layout=true;}
            if(!layout) throw std::runtime_error("K-quant block that does not divide the projection length was accepted");
            // The same superblock schedule serves the shared-expert and output-head
            // graphs: native_mmvq q3_q8_dot_impl and q4_q8_dot_impl both multiply the
            // group terms by the quantized activation code sums.
            for(auto dimensions:{std::array<uint32_t,2>{2560,640},std::array<uint32_t,2>{2560,248320}}) for(uint32_t mode:{0u,2u}) {
                GraphBuilder head(OfflineCompilation{});
                define_packed_matmul(head,dimensions[0],dimensions[1],1,type,static_cast<PackedActivation>(mode));
                const auto bytes=head.compile_offline("strata_k_"+std::to_string(type)+"_"+std::to_string(dimensions[1])+"_"+std::to_string(mode));
                if(bytes>(64ull<<20)) throw std::runtime_error("K-quant head workspace exceeds allowance");
                std::cout<<"PASS offline K-quant type="<<type<<" rows="<<dimensions[1]<<" activation="<<mode<<" workspace="<<bytes<<" no_device_acquired\n";
            }
        }
        for(auto type:{0u,2u,42u}) {
            GraphBuilder graph(OfflineCompilation{});bool rejected=false;
            try {define_packed_matmul(graph,type==42?32:63,4,1,type);}catch(const std::invalid_argument&){rejected=true;}
            if(!rejected) throw std::runtime_error("invalid packed geometry accepted");
        }
        for(uint32_t batch:{3u,7u}) for(uint32_t type:{30u,2u,3u,6u,7u,8u,11u,14u,16u,17u,18u,20u,21u,22u,23u,29u,42u}) {
            GraphBuilder graph(OfflineCompilation{});
            if(type==30) define_model_head(graph,2560,248320,batch);else define_packed_matmul(graph,2560,248320,batch,type);
            const auto bytes=graph.compile_offline("strata_target_head_window_"+std::to_string(type)+"_"+std::to_string(batch));
            if(bytes>(64ull<<20)) throw std::runtime_error("target head workspace exceeds candidate compiler allowance");
            std::cout<<"PASS offline target head type="<<type<<" rows="<<batch<<" workspace_bytes="<<bytes<<" no_device_acquired\n";
        }
        for(uint32_t batch:{3u,8u,32u}) for(uint32_t type:{2u,8u}) for(auto dimensions:{std::array<uint32_t,2>{2560,2560},std::array<uint32_t,2>{2560,12288},std::array<uint32_t,2>{2560,512},std::array<uint32_t,2>{6144,2560}}) {
            GraphBuilder graph(OfflineCompilation{});define_packed_matmul(graph,dimensions[0],dimensions[1],batch,type);
            const auto bytes=graph.compile_offline("strata_mtp_window_projection_"+std::to_string(type)+"_"+std::to_string(batch)+"_"+std::to_string(dimensions[0])+"_"+std::to_string(dimensions[1]));
            if(bytes>(16ull<<20)) throw std::runtime_error("MTP window projection workspace exceeds allowance");
            std::cout<<"PASS offline MTP packed projection batch="<<batch<<" type="<<type<<" workspace_bytes="<<bytes<<" no_device_acquired\n";
        }
        check(synDestroy(),"offline packed synDestroy");initialized=false;
    }catch(const std::exception& e){if(initialized) (void)synDestroy();std::cerr<<e.what()<<'\n';return 1;}
}
