#include "strata/hpu/qsa.hpp"
#include "strata/hpu/host_recipe.hpp"
#include "strata/hpu/qsa_window.hpp"
#include "strata/hpu/qsa_window_flow.hpp"
#include <algorithm>
#include <array>
#include <cstring>
#include <iostream>
#include <string>
using namespace strata::hpu;

static void define(GraphBuilder& g,const QsaConfig& c,uint32_t rows,uint32_t bucket,bool decode) {
    const auto k=g.tensor({"key",DType::F32,{uint64_t(rows)*c.kv_heads,bucket,c.width}});
    const auto v=g.tensor({"value",DType::F32,{uint64_t(rows)*c.kv_heads,bucket,c.width}});
    const auto m=g.tensor({"mask",DType::F32,decode?std::vector<uint64_t>{bucket}:
                          std::vector<uint64_t>{uint64_t(rows)*c.kv_heads,1,bucket}});
    define_qsa_device_gather(g,c,bucket,rows,k,v,m,decode?1:c.kv_heads);
}
int main(int argc,char** argv) {
    try {
        QsaConfig c;c.capacity=4096;c.page_size=7;c.top_k=32;
        constexpr uint32_t bucket=35;
        const bool on_device=argc==2 && std::string(argv[1])=="--device";
        if(!on_device) {
            configure_kernel_libraries();check(synInitialize(),"gather offline init");
            for(uint32_t variant:{0u,1u,3u,4u,8u,32u,64u,128u}) {
                const uint32_t rows=variant?variant:1;const bool decode=variant==0;
                GraphBuilder g(OfflineCompilation{});define(g,c,rows,bucket,decode);
                std::cout<<"PASS offline gather rows="<<rows<<" workspace="<<g.compile_offline("qsa_gather_offline")<<'\n';
            }
            check(synDestroy(),"gather offline destroy");return 0;
        }
        Device d;Stream stream(d);QsaState state(c);
        const uint64_t row=uint64_t(c.kv_heads)*c.width;
        std::vector<float> k(row),v(row),index(c.index_width),gamma(c.index_width,1);
        for(uint32_t cell=0;cell<c.capacity;++cell) {
            for(uint64_t i=0;i<row;++i) {k[i]=float(int((cell*13+i*7)%2048)-1024)/64;v[i]=-k[i];}
            state.append(k,v,index,gamma);
        }
        HostBuffer host(d,uint64_t(c.capacity)*row*8);
        auto* data=static_cast<float*>(host.data());
        state.fp16_cells(0,c.capacity,{data,c.capacity*row},{data+c.capacity*row,c.capacity*row});
        DeviceBuffer keys(d,c.capacity*row*4),values(d,c.capacity*row*4);
        upload(stream,keys,host,c.capacity*row*4);upload(stream,values,host,c.capacity*row*4,0,c.capacity*row*4);stream.synchronize();
        const std::array<Binding,2> bindings{{{"kv.cache_k",&keys},{"kv.cache_v",&values}}};
        // Inspect the real composed recipe: device-produced KV/mask/out must
        // not acquire pinned mirrors when the finish has no host outputs.
        for(uint32_t rows:{1u,3u,8u,32u,64u,128u}) {
            GraphBuilder g(d);g.set_scope("att.");define_qsa_attention_window(g,c,bucket,rows);g.set_scope({});
            define_qsa_device_gather(g,c,bucket,rows,g.find("att.key"),g.find("att.value"),g.find("att.mask"),c.kv_heads);
            const auto expected=qsa_window_gather_pinned_bytes(c,rows,bucket);
            HostRecipe recipe(d,g.compile("qsa_gather_pinned_schema"),bindings,{},expected);
            if(recipe.pinned_bytes()!=expected) throw std::runtime_error("gather pinned plan differs from actual composed graph slots");
            std::cout<<"PASS gather pinned schema rows="<<rows<<" bytes="<<expected<<'\n';
        }
        for(uint32_t variant:{0u,1u,3u,4u,8u,32u,64u,128u}) {
            const uint32_t rows=variant?variant:1;const bool decode=variant==0;
            GraphBuilder g(d);define(g,c,rows,bucket,decode);
            HostRecipe recipe(d,g.compile("qsa_gather_device"),bindings,{"key","value","mask"});
            // Read unrelated, noncontiguous cells beyond the old device-cache edge.
            for(uint32_t pass=0;pass<3;++pass) {
                std::vector<float> ids(uint64_t(rows)*bucket,-1),ek,ev,em;
                for(uint32_t r=0;r<rows;++r) {
                    std::vector<int32_t> selected{int32_t(4095-pass),0,2052,int32_t(3000+r)};
                    if(pass==1) std::reverse(selected.begin(),selected.end());
                    for(size_t i=0;i<selected.size();++i) ids[uint64_t(r)*bucket+i]=float(selected[i]);
                    const auto expected=state.gather(selected,bucket);
                    ek.insert(ek.end(),expected.key.begin(),expected.key.end());
                    ev.insert(ev.end(),expected.value.begin(),expected.value.end());
                    for(uint32_t head=0;head<(decode?1:c.kv_heads);++head)
                        em.insert(em.end(),expected.mask.begin(),expected.mask.end());
                }
                recipe.set("kv.ids",ids);recipe.run();
                const auto same=[&](const char* name,const std::vector<float>& expected) {
                    auto got=recipe.output(name);
                    if(got.size()!=expected.size() || std::memcmp(got.data(),expected.data(),got.size_bytes()))
                        throw std::runtime_error(std::string("gather differs from independent paged host oracle: ")+name);
                };
                same("key",ek);same("value",ev);same("mask",em);
            }
            std::cout<<"PASS device gather bit-identical to paged host oracle rows="<<rows<<" decode_mask="<<decode<<"\n";
        }
        QsaWindow window(d,c,2560);
        for(uint32_t rows:{1u,3u,8u}) {
            QsaWindowBatch batch;batch.rows=rows;batch.bucket=bucket;
            batch.query.assign(uint64_t(rows)*c.query_heads*c.width,.01f);batch.gate.assign(batch.query.size(),.5f);
            batch.ids.assign(uint64_t(rows)*bucket,-1);
            for(uint32_t r=0;r<rows;++r) {
                const std::vector<int32_t> selected{4095,0,2052,int32_t(3000+r)};
                std::copy(selected.begin(),selected.end(),batch.ids.begin()+uint64_t(r)*bucket);
                const auto kv=state.gather(selected,bucket);
                batch.key.insert(batch.key.end(),kv.key.begin(),kv.key.end());batch.value.insert(batch.value.end(),kv.value.begin(),kv.value.end());
                for(uint32_t head=0;head<c.kv_heads;++head) batch.mask.insert(batch.mask.end(),kv.mask.begin(),kv.mask.end());
            }
            const auto expected=window.attention(batch);
            batch.key.clear();batch.value.clear();batch.mask.clear();
            const auto actual=window.attention(batch,bindings);
            if(actual.size()!=expected.size() || std::memcmp(actual.data(),expected.data(),actual.size()*4))
                throw std::runtime_error("window attention gather differs from uploaded KV");
            std::cout<<"PASS window attention host KV vs device gather rows="<<rows<<" bit_exact=1\n";
        }
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
