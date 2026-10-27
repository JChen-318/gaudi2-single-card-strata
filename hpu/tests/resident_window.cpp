// Offline shape coverage; --device compares one wide MoE with 8-row chunks.
#include "strata/hpu/resident_experts.hpp"
#include "strata/hpu/host_recipe.hpp"
#include "strata/hpu/hc_ple.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <chrono>
#include <cstring>
#include <iostream>
using namespace strata::hpu;
namespace {
strata::core::ModelGeometry geometry() {
    strata::core::ModelGeometry g{};g.n_embd=1280;g.n_ff=1280;g.n_expert=64;g.n_expert_used=10;return g;
}
void define(GraphBuilder& g,uint32_t rows) {
    const auto c=geometry();const uint64_t h=c.n_embd;
    const auto x=g.tensor({"x",DType::F32,{rows,h}}),logits=g.tensor({"logits",DType::F32,{rows,uint64_t(c.n_expert)}}),gate=g.tensor({"gate",DType::F32,{rows,1}});
    g.set_scope("moe.");const auto block=define_resident_moe(g,c,x,logits,gate,rows);g.set_scope({});
    const auto out=g.tensor({"out",DType::F32,{rows,h}});g.reshape(block,out);
}
}
int main(int argc,char** argv) {
    try {
        const bool bench=argc==2 && std::string(argv[1])=="--bench";
        if(!bench && (argc!=2 || std::string(argv[1])!="--device")) {
            configure_kernel_libraries();check(synInitialize(),"resident window offline init");
            for(uint32_t rows:{9u,16u,32u,64u,128u}) {
                GraphBuilder g(OfflineCompilation{});define(g,rows);
                std::cout<<"PASS resident window compile rows="<<rows<<" workspace="<<g.compile_offline("resident_window_offline")<<'\n';
                GraphBuilder hc(OfflineCompilation{});define_hc_read_window(hc,{1280,4,64,1e-6f},rows);
                std::cout<<"PASS HC window compile rows="<<rows<<" workspace="<<hc.compile_offline("resident_hc_registry")<<'\n';
            }
            check(synDestroy(),"resident window offline destroy");return 0;
        }
        const auto c=geometry();Device d;Stream copy(d);const uint64_t h=c.n_embd,ff=c.n_ff;
        const auto gu_bytes=iq4_resident_bytes(2*ff,h),down_bytes=iq4_resident_bytes(h,ff);
        DeviceBuffer gu(d,64*gu_bytes),down(d,64*down_bytes),shared_gu(d,gu_bytes),shared_down(d,down_bytes),table(d,256);
        HostBuffer bounce(d,gu_bytes);
        auto upload_weights=[&](DeviceBuffer& buffer,uint64_t rows,uint64_t columns,uint32_t experts) {
            const auto bytes=iq4_resident_bytes(rows,columns),codes=rows*columns/2;
            auto* raw=static_cast<uint8_t*>(bounce.data());
            for(uint32_t e=0;e<experts;++e) {
                for(uint64_t i=0;i<codes;++i) raw[i]=uint8_t((i*13+e*17)%256);
                for(uint64_t i=codes;i<bytes;i+=4) {const float s=float(1+(i/4+e)%7)/4096;std::memcpy(raw+i,&s,4);}
                upload(copy,buffer,bounce,bytes,e*bytes);copy.synchronize();
            }
        };
        upload_weights(gu,2*ff,h,64);upload_weights(down,h,ff,64);
        upload_weights(shared_gu,2*ff,h,1);upload_weights(shared_down,h,ff,1);
        const auto lut=iq4_codebook_table();std::memcpy(bounce.data(),lut.data(),256);upload(copy,table,bounce,256);copy.synchronize();
        const std::array<Binding,5> bindings{{{"moe.w_gu",&gu},{"moe.w_down",&down},{"moe.s_gu",&shared_gu},{"moe.s_down",&shared_down},{"moe.table",&table}}};
        GraphBuilder small(d);define(small,8);HostRecipe reference(d,small.compile("resident_window_reference8"),bindings,{"out"});
        GraphBuilder single(d);define(single,1);HostRecipe tail(d,single.compile("resident_window_reference1"),bindings,{"out"});
        for(uint32_t rows:{9u,16u,32u,64u,128u}) {
            GraphBuilder graph(d);define(graph,rows);HostRecipe wide(d,graph.compile("resident_window_device"),bindings,{"out"});
            for(uint32_t phase=0;phase<3;++phase) {
                std::vector<float> x(rows*h),logits(rows*64),gate(rows,.25f);
                for(uint64_t i=0;i<x.size();++i) x[i]=phase==1?0:float(int((i*19)%257)-128)/512;
                for(uint32_t r=0;r<rows;++r) for(uint32_t e=0;e<64;++e) logits[r*64+e]=float((e*11+(phase==2?0:r*7))%64)/16;
                wide.set("x",x);wide.set("logits",logits);wide.set("gate",gate);wide.run();const auto actual=wide.values("out");
                std::vector<float> expected;
                for(uint32_t r=0;r<rows;) {
                    const uint32_t count=rows-r>=8?8:1;auto& control=count==8?reference:tail;
                    control.set("x",std::span<const float>(x).subspan(r*h,count*h));
                    control.set("logits",std::span<const float>(logits).subspan(r*64,count*64));control.set("gate",std::span<const float>(gate).subspan(r,count));control.run();
                    const auto y=control.values("out");expected.insert(expected.end(),y.begin(),y.end());r+=count;
                }
                if(actual.size()!=expected.size() || std::memcmp(actual.data(),expected.data(),actual.size()*4)) throw std::runtime_error("wide MoE differs from 8-row reference");
                std::cout<<"PASS resident window device rows="<<rows<<" phase="<<phase<<" bit_exact=1\n"<<std::flush;
                if(bench) {
                    // Same recipe and already-clean inputs. Includes graph submit,
                    // output download and wait; this is not raw kernel time.
                    std::vector<double> samples;
                    for(uint32_t pass=0;pass<5;++pass) {
                        const auto begin=std::chrono::steady_clock::now();
                        for(uint32_t run=0;run<4;++run) wide.run();
                        samples.push_back(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count()/4);
                    }
                    std::sort(samples.begin(),samples.end());
                    std::cout<<"BENCH resident window rows="<<rows<<" phase="<<phase<<" host_run_ms="<<samples[2]<<" trials=5 repeats=4\n"<<std::flush;
                }
            }
        }
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
