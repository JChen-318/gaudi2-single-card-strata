// Complete isolated U4 MoE graph: compare actual device kernels and MME against
// an independent scalar reconstruction of the stored affine weights. Not a model quality gate.
#include "strata/hpu/u4_moe.hpp"
#include "strata/hpu/host_recipe.hpp"
#include "strata/hpu/model_abi.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <numeric>
#include <map>
#include "strata/hpu/u4_quant.hpp"
using namespace strata::hpu;
namespace {
strata::core::ModelGeometry geometry() {
    strata::core::ModelGeometry c{};c.n_embd=128;c.n_ff=64;c.n_expert=64;c.n_expert_used=2;return c;
}
void define(GraphBuilder& g,uint32_t rows,const strata::core::ModelGeometry& c=geometry()) {
    const auto x=g.tensor({"x",DType::F32,{rows,uint64_t(c.n_embd)}});
    const auto logits=g.tensor({"logits",DType::F32,{rows,uint64_t(c.n_expert)}});
    const auto gate=g.tensor({"shared_gate",DType::F32,{rows,1}});
    define_resident_u4_moe(g,c,U4LayerStore::make(c.n_expert,c.n_embd,c.n_ff),x,logits,gate,rows);
}
void require(bool yes,const char* why) {if(!yes) throw std::runtime_error(why);}
void gather_check(const Device& d) {
    GraphBuilder g(d);
    const auto x=g.tensor({"x",DType::F32,{1,128}}),map=g.tensor({"map",DType::I32,{1}});
    const auto a=g.tensor({"a",DType::BF16,{1,1,128}}),sum=g.tensor({"sum",DType::F32,{1}});
    abi::U4MoeGather p{1,128,1,3};const TensorId in[]={x,map},out[]={a,sum};g.op("strata_u4_moe_gather_f32",in,out,&p,sizeof(p));
    DeviceBuffer mapbuf(d,128);HostBuffer mb(d,128);std::memset(mb.data(),0,128);Stream stream(d);upload(stream,mapbuf,mb,128);stream.synchronize();
    const std::vector<Binding> bindings{{"map",&mapbuf,0}};
    HostRecipe host(d,g.compile("u4_gather_sum_check"),bindings,{"a","sum"});
    std::vector<float> input(128);for(size_t i=0;i<input.size();++i) input[i]=float(int(i*13%97)-48)/173;
    host.set("x",input);host.run();const auto actual=host.values("a"),total=host.values("sum");
    double expected=0;for(size_t i=0;i<128;++i) {
        require(actual[i]==from_bf16(to_bf16(input[i])),"gather BF16 lane mismatch");expected+=actual[i];
    }
    std::cout<<"GATHER sum="<<total[0]<<" expected="<<expected<<std::endl;
    require(std::abs(total[0]-expected)<1e-6,"gather affine sum does not match actual BF16 A");
}
struct Fixture {
    strata::core::ModelGeometry config;
    U4LayerStore store;
    std::vector<uint8_t> layer;
    DeviceBuffer payload,map;
    std::vector<Binding> bindings;
    std::map<uint32_t,std::pair<std::vector<float>,std::vector<float>>> originals;
    Fixture(const Device& d,const strata::core::ModelGeometry& c=geometry()):config(c),
        store(U4LayerStore::make(c.n_expert,c.n_embd,c.n_ff)),layer(store.bytes()),payload(d,store.bytes()),map(d,1024) {
        for(uint64_t expert=0;expert<store.blocks();++expert) {
            for(bool gu:{true,false}) {
                const auto groups=gu?store.gu_groups:store.down_groups,n=gu?store.gu_rows:store.down_rows,k=gu?store.gu_group:store.down_group;
                for(uint64_t g=0;g<groups;++g) for(uint64_t row=0;row<n;++row) {
                    const auto at=gu?store.gu_code_at(expert,g,row):store.down_code_at(expert,g,row);
                    for(uint64_t j=0;j<k;j+=2) {
                        const uint8_t a=uint8_t((expert+3*row+j+g)%16),b=uint8_t((expert+3*row+j+1+g)%16);
                        layer[at+j/2]=uint8_t(a|(b<<4));
                    }
                    const float scale=float(1+(expert+row+g)%3)/1024;
                    const float offset=-scale*(6.25f+float(row%3)*.125f);
                    const auto sa=gu?store.gu_scale_at(expert,g,row):store.down_scale_at(expert,g,row);
                    const auto oa=gu?store.gu_offset_at(expert,g,row):store.down_offset_at(expert,g,row);
                    std::memcpy(layer.data()+sa,&scale,4);std::memcpy(layer.data()+oa,&offset,4);
                }
            }
        }
        HostBuffer staging(d,layer.size());std::memcpy(staging.data(),layer.data(),layer.size());Stream copy(d);
        upload(copy,payload,staging,layer.size());copy.synchronize();
        HostBuffer mh(d,1024);auto* ids=static_cast<int32_t*>(mh.data());for(int i=0;i<256;++i) ids[i]=(config.n_expert_used+1)*i+config.n_expert_used;
        upload(copy,map,mh,1024);copy.synchronize();
        const auto add=[&](const char* name,uint64_t start,uint64_t block) {
            bindings.push_back({name,&payload,start});bindings.push_back({std::string("shared_")+name,&payload,start+store.experts*block});
        };
        add("gu_codes",store.gu_codes_start,store.gu_codes_block);
        add("gu_scale",store.gu_scale_start,store.gu_affine_block);add("gu_offset",store.gu_offset_start,store.gu_affine_block);
        add("down_codes",store.down_codes_start,store.down_codes_block);
        add("down_scale",store.down_scale_start,store.down_affine_block);add("down_offset",store.down_offset_start,store.down_affine_block);
        bindings.push_back({"shared_map",&map,0});
    }
    void load_real(const Device& d,const WeightTable& table,uint32_t index) {
        require(index<uint32_t(table.geometry().n_layers),"real layer out of range");
        const auto prefix="blk."+std::to_string(index)+".";
        const auto source=[](const WeightSlice& x) {return U4Source{x.bytes(),x.row_bytes(),x.rows(),x.columns(),x.ggml_type()};};
        for(const uint32_t e:{0u,1u,uint32_t(store.experts)}) {
            const bool shared=e==store.experts;
            const auto get=[&](const char* name) {return shared?table.matrix(prefix+"ffn_"+name+"_shexp.weight"):
                table.expert(prefix+"ffn_"+name+"_exps.weight",e);};
            const auto gate=get("gate"),up=get("up"),down=get("down");
            std::vector<uint8_t> block(store.block_bytes());
            pack_u4_block(source(gate),source(up),source(down),block,store);place_u4_block(block,layer,store,e);
            auto& original=originals[e];original.first.resize(2*store.ff*store.hidden);original.second.resize(store.hidden*store.ff);
            gate.convert_rows_f32(0,std::span<float>(original.first).first(store.ff*store.hidden));
            up.convert_rows_f32(0,std::span<float>(original.first).subspan(store.ff*store.hidden));
            down.convert_rows_f32(0,original.second);
            std::cout<<"REAL WEIGHTS layer="<<index<<" expert="<<e<<" source_types="<<gate.ggml_type()<<'/'<<down.ggml_type()
                     <<" gu_rel_rms="<<u4_relative_rms(original.first,dequantize_u4_matrix(layer,store,e,true))
                     <<" down_rel_rms="<<u4_relative_rms(original.second,dequantize_u4_matrix(layer,store,e,false))<<std::endl;
        }
        HostBuffer staging(d,layer.size());std::memcpy(staging.data(),layer.data(),layer.size());Stream copy(d);
        upload(copy,payload,staging,layer.size());copy.synchronize();
    }
    std::vector<float> expert(uint32_t e,std::span<const float> input,bool original=false) const {
        const auto gu=original?originals.at(e).first:dequantize_u4_matrix(layer,store,e,true);
        const auto down=original?originals.at(e).second:dequantize_u4_matrix(layer,store,e,false);
        const size_t h=store.hidden,ff=store.ff;
        std::vector<float> a(ff),y(h);
        for(size_t j=0;j<ff;++j) {
            double gate=0,up=0;
            for(size_t k=0;k<h;++k) {
                const double x=from_bf16(to_bf16(input[k]));gate+=gu[j*h+k]*x;up+=gu[(ff+j)*h+k]*x;
            }
            const float gf=float(gate),uf=float(up);
            a[j]=from_bf16(to_bf16((gf/(1+std::exp(-gf)))*uf));
        }
        for(size_t j=0;j<h;++j) {double v=0;for(size_t k=0;k<ff;++k) v+=double(down[j*ff+k])*a[k];y[j]=float(v);}
        return y;
    }
    void zero_device_offsets(const Device& d) {
        HostBuffer zeros(d,std::max(store.gu_affine_block,store.down_affine_block)*store.blocks());
        std::memset(zeros.data(),0,zeros.size());Stream stream(d);
        upload(stream,payload,zeros,store.gu_affine_block*store.blocks(),store.gu_offset_start);
        upload(stream,payload,zeros,store.down_affine_block*store.blocks(),store.down_offset_start);
        stream.synchronize();
    }
};
void run(const Device& d,Fixture& f,uint32_t rows) {
    const size_t h=f.store.hidden,e_count=f.store.experts;
    require(f.config.n_expert_used==2,"numeric fixture requires top-k 2");
    GraphBuilder g(d);define(g,rows,f.config);
    HostRecipe host(d,g.compile("u4_moe_fixture_"+std::to_string(rows)),f.bindings,{"expert_block","expert_rows","route_weights","overflow_count"});
    for(int phase=0;phase<2;++phase) {
        std::vector<float> x(rows*h),logits(rows*e_count,-20),shared(rows,.25f);
        for(size_t i=0;i<x.size();++i) x[i]=float(int((i*13+phase*7)%97)-48)/173;
        for(uint32_t r=0;r<rows;++r) {
            const uint32_t e=phase?0:(r*7)%e_count;
            logits[r*e_count+e]=2;logits[r*e_count+(e+1)%e_count]=1;
        }
        host.set("x",x);host.set("logits",logits);host.set("shared_gate",shared);host.run();
        const auto y=host.values("expert_block"),slots=host.values("expert_rows"),weights=host.values("route_weights"),spill=host.values("overflow_count");
        const float total_spill=std::accumulate(spill.begin(),spill.end(),0.f);
        require(total_spill==(phase && rows>64?float(2*(rows-64)):0.f),"overflow count did not match forced hot routing");
        double worst=0,peak=0,original_error=0,original_energy=0;bool nonzero=false;
        for(uint32_t r=0;r<rows;++r) {
            const auto in=std::span<const float>(x).subspan(r*h,h);
            const uint32_t e=phase?0:(r*7)%e_count;
            const auto a=f.expert(e,in),b=f.expert((e+1)%e_count,in),s=f.expert(uint32_t(e_count),in);
            std::vector<float> oa,ob,os;
            if(!f.originals.empty()) {oa=f.expert(e,in,true);ob=f.expert((e+1)%e_count,in,true);os=f.expert(uint32_t(e_count),in,true);}
            for(size_t j=0;j<h;++j) {
                const float ar=phase && r>=64?0:a[j],br=phase && r>=64?0:b[j];
                const double ref=ar*weights[r*3]+br*weights[r*3+1]+s[j]*weights[r*3+2];
                if(!oa.empty()) {
                    const double original=oa[j]*weights[r*3]+ob[j]*weights[r*3+1]+os[j]*weights[r*3+2];
                    original_error+=(ref-original)*(ref-original);original_energy+=original*original;
                }
                require(std::isfinite(y[r*h+j]),"nonfinite MoE output");
                worst=std::max(worst,std::abs(y[r*h+j]-ref));peak=std::max(peak,std::abs(ref));nonzero|=std::abs(ref)>1e-7;
                if(phase && r>=64) require(slots[(r*3)*h+j]==0 && slots[(r*3+1)*h+j]==0,"stale overflow slot became active");
            }
        }
        require(nonzero,"fixture cannot distinguish broken arithmetic from zero output");
        require(worst<2e-9+peak*2e-3,"U4 MoE differs from affine/BF16 scalar reference");
        std::cout<<"PASS U4 MoE rows="<<rows<<" h="<<h<<" groups="<<f.store.gu_groups<<"/"<<f.store.down_groups<<" phase="<<phase<<" max_abs="<<worst<<" peak="<<peak<<" spill="<<total_spill
                 <<" (overflow is diagnostic, not served)"<<std::endl;
        if(!f.originals.empty()) std::cout<<"REQUANTIZATION ONLY layer_output_rel_rms="<<std::sqrt(original_error/std::max(original_energy,1e-30))
                                         <<" (same BF16 activations; not a full-model logits gate)"<<std::endl;
    }
}
strata::core::ModelGeometry production_geometry(uint32_t experts=512,uint32_t topk=10) {
    auto c=geometry();c.n_embd=2560;c.n_ff=640;c.n_expert=experts;c.n_expert_used=topk;return c;
}
struct TimedEvent {
    synEventHandle handle=nullptr;
    explicit TimedEvent(const Device& d) {check(synEventCreate(&handle,d.info().id,EVENT_COLLECT_TIME),"event create");}
    ~TimedEvent() {if(handle) (void)synEventDestroy(handle);}
};
void time_production(const Device& d,uint32_t rows) {
    const auto c=production_geometry();Fixture f(d,c);GraphBuilder g(d);define(g,rows,c);
    auto recipe=g.compile("u4_moe_production_timing_"+std::to_string(rows));const auto workspace=recipe->workspace_bytes();
    HostRecipe host(d,std::move(recipe),f.bindings,{},256ull<<20);
    std::vector<float> x(rows*2560),logits(rows*512,-20),shared(rows,.25f);
    for(size_t i=0;i<x.size();++i) x[i]=float(int(i*13%97)-48)/173;
    for(uint32_t r=0;r<rows;++r) for(uint32_t k=0;k<10;++k) logits[r*512+(r*7+k*37)%512]=float(10-k);
    host.set("x",x);host.set("logits",logits);host.set("shared_gate",shared);host.run();
    Stream stream(d);TimedEvent start(d),end(d);std::vector<double> samples;
    for(int pass=0;pass<5;++pass) {
        check(synEventRecord(start.handle,stream.handle()),"record start");
        for(int i=0;i<8;++i) host.launch(stream);
        check(synEventRecord(end.handle,stream.handle()),"record end");check(synEventSynchronize(end.handle),"wait event");host.wait();
        uint64_t ns=0;check(synEventElapsedTime(&ns,start.handle,end.handle),"elapsed");require(ns>0,"zero timing interval");
        samples.push_back(double(ns)/8e6);
    }
    std::sort(samples.begin(),samples.end());
    std::cout<<"TIMING full U4 MoE rows="<<rows<<" ms="<<samples[2]<<" workspace="<<workspace<<" samples=";
    for(auto ms:samples) std::cout<<ms<<',';
    std::cout<<std::endl;
}

}
int main(int argc,char** argv) {
    try {
        if(argc==2 && std::string(argv[1])=="--device-gather") {
            Device d;gather_check(d);
        } else if(argc==2 && std::string(argv[1])=="--device-mutate-offset") {
            Device d;Fixture f(d);f.zero_device_offsets(d);bool rejected=false;
            try {run(d,f,3);} catch(const std::runtime_error& e) {
                if(std::string(e.what())!="U4 MoE differs from affine/BF16 scalar reference") throw;
                rejected=true;
            }
            require(rejected,"offset mutation unexpectedly survived");
            std::cout<<"PASS negative control: zeroed offset terms rejected by independent MoE reference\n";
        } else if(argc==2 && std::string(argv[1])=="--device") {
            Device d;Fixture f(d);for(uint32_t rows:{1u,3u,8u,65u}) run(d,f,rows);
        } else if(argc==2 && std::string(argv[1])=="--device-production-shape") {
            Device d;Fixture f(d,production_geometry(64,2));run(d,f,3);
        } else if(argc==4 && std::string(argv[1])=="--device-real-shape") {
            WeightTable table(argv[2]);Device d;Fixture f(d,production_geometry(64,2));
            f.load_real(d,table,uint32_t(std::stoul(argv[3])));run(d,f,1);
        } else if(argc==3 && std::string(argv[1])=="--device-production-timing") {
            const auto rows=std::stoul(argv[2]);require(rows>0 && rows<=256,"production rows must be in 1..256");
            Device d;time_production(d,uint32_t(rows));
        } else if(argc==3 && std::string(argv[1])=="--production-compile") {
            const auto rows=std::stoul(argv[2]);
            require(rows>0 && rows<=256,"production rows must be in 1..256");
            const auto c=production_geometry();
            configure_kernel_libraries();check(synInitialize(),"init");
            {
                GraphBuilder g(OfflineCompilation{});define(g,uint32_t(rows),c);
                const auto workspace=g.compile_offline("u4_moe_production_"+std::to_string(rows));
                std::cout<<"PASS production U4 MoE compile rows="<<rows<<" workspace="<<workspace<<std::endl;
            }
            check(synDestroy(),"destroy");
        } else if(argc==1) {
            configure_kernel_libraries();check(synInitialize(),"init");
            for(uint32_t rows:{1u,8u,65u,256u}) {
                GraphBuilder g(OfflineCompilation{});define(g,rows);g.compile_offline("u4_moe_fixture");
                std::cout<<"PASS isolated U4 MoE compile rows="<<rows<<std::endl;
            }
            check(synDestroy(),"destroy");std::cout<<"PASS isolated U4 MoE offline graphs\n";
        } else throw std::invalid_argument("unknown U4 MoE probe mode");
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
