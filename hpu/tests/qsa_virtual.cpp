#include "strata/hpu/host_recipe.hpp"
#include "strata/hpu/model_abi.hpp"
#include "strata/hpu/performance_modes.hpp"
#include <array>
#include <cstring>
#include <iostream>
using namespace strata::hpu;
void define(GraphBuilder& g,uint32_t first,uint32_t bucket) {
    const auto k=g.tensor({"cache_k",DType::F32,{128,2,64}}),v=g.tensor({"cache_v",DType::F32,{128,2,64}});
    const auto nk=g.tensor({"now_k",DType::F32,{128,2,64}}),nv=g.tensor({"now_v",DType::F32,{128,2,64}}),cell=g.tensor({"cell",DType::F32,{1}});
    const auto out=g.tensor({"key",DType::F32,{2,bucket+32,64}}),value=g.tensor({"value",DType::F32,{2,bucket+32,64}});
    abi::QsaVirtual args{64,2,128,128,int32_t(first),32,int32_t(bucket)};const TensorId in[]={k,v,nk,nv,cell},o[]={out,value};
    g.op("strata_qsa_virtual_f32",in,o,&args,sizeof(args));
}
int main(int argc,char** argv) {
 try {
    const bool device=argc==2 && std::string(argv[1])=="--device";
    if(!device) {
        configure_kernel_libraries();check(synInitialize(),"virtual init");
        GraphBuilder g(OfflineCompilation{});define(g,96,512);std::cout<<"PASS QSA virtual compile workspace="<<g.compile_offline("qsa_virtual_offline")<<'\n';
        check(synDestroy(),"virtual destroy");return 0;
    }
    Device d;
    for(const auto a:std::array<std::array<uint32_t,3>,5>{{{0,0,1},{0,32,32},{32,32,128},{64,64,128},{127,96,512}}}) {
        const auto cell=a[0],first=a[1],bucket=a[2];GraphBuilder g(d);define(g,first,bucket);HostRecipe recipe(d,g.compile("qsa_virtual_device"),{}, {"key","value"});
        std::vector<float> cache(128*2*64),now(cache.size()),value(cache.size()),next(cache.size());
        for(uint32_t row=0;row<128;++row) for(uint32_t head=0;head<2;++head) for(uint32_t col=0;col<64;++col) {
            const uint64_t at=(row*2+head)*64+col;cache[at]=float(10000+row*128+head*64+col);value[at]=cache[at]+.5f;
            now[at]=-float(1+row*128+head*64+col);next[at]=now[at]-.25f;
        }
        recipe.set("cache_k",cache);recipe.set("cache_v",value);recipe.set("now_k",now);recipe.set("now_v",next);const std::array<float,1> position{float(cell)};recipe.set("cell",position);recipe.run();
        const auto actual=recipe.values("key"),values=recipe.values("value");
        for(uint32_t head=0;head<2;++head) for(uint32_t item=0;item<bucket+32;++item) for(uint32_t col=0;col<64;++col) {
            int source=-1;bool from_now=false;
            if(item<cell) source=int(item);
            else if(item<cell+first) {source=int(item-cell);from_now=true;}
            else if(item>=bucket) {source=int(first+item-bucket);from_now=true;}
            const auto at=(head*(bucket+32)+item)*64+col;float k=0,v=0;
            if(source>=0) {const auto src=(uint32_t(source)*2+head)*64+col;k=from_now?now[src]:cache[src];v=from_now?next[src]:value[src];}
            if(actual[at]!=k || values[at]!=v) throw std::runtime_error("QSA virtual prefix differs at "+std::to_string(at));
        }
        std::cout<<"PASS QSA virtual bit_exact cell="<<cell<<" first="<<first<<" bucket="<<bucket<<'\n';
    }
    if(prefill_final_enabled()) {
        const auto before=d.pinned_allocations().used;
        {
            GraphBuilder g(d);define(g,32,128);HostRecipe recipe(d,g.compile("device_only_internal"),{},{});
            constexpr uint64_t input_bytes=(4*128*2*64+1)*4;
            if(recipe.pinned_bytes()!=input_bytes || d.pinned_allocations().used-before>input_bytes+5*127)
                throw std::runtime_error("internal device tensors consumed pinned host buffers");
            for(auto name:{"cache_k","cache_v","now_k","now_v","cell"}) {auto x=recipe.input(name);std::fill(x.begin(),x.end(),0.0f);}
            recipe.run();bool rejected=false;try {(void)recipe.values("key");} catch(const std::logic_error&) {rejected=true;}
            if(!rejected) throw std::runtime_error("internal-only tensor exposed a host output");
        }
        if(d.pinned_allocations().used!=before) throw std::runtime_error("device-only internal pinned memory leak");
        std::cout<<"PASS device-only internal tensors: pinned inputs only, output contract, release exact\n";
    }

 } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
