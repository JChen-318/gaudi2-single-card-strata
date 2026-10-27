// Hardware diagnostic: exact multi-row recurrence against linked production steps, and the cost of the three
// ways to run the same window. chain = one gdn_step node per row linked by the host (the production path with
// STRATA_HPU_GDN_WINDOW off); local = the same steps with each member's state slice kept on the TPC
// (gdn_step_final.c); window = the whole window in one launch (gdn_step_window.c).
// --device also times the production geometry (s=128, hk=6, hv=48) at one row (decode step) and 64 rows.
#include "strata/hpu/gdn.hpp"
#include "strata/hpu/gdn_abi.hpp"
#include "strata/hpu/host_recipe.hpp"
#include <algorithm>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
using namespace strata::hpu;
namespace {
enum class Variant {chain,local,window};
const char* name(Variant variant) {return variant==Variant::chain?"chain":variant==Variant::local?"local":"window";}

void define_step(GraphBuilder& g,uint32_t w,uint32_t hk,uint32_t hv,uint32_t rows,Variant variant) {
    const auto T=[&](std::string n,std::vector<uint64_t> shape,bool persist=false){return g.tensor({std::move(n),DType::F32,std::move(shape),persist});};
    auto state=T("state",{w,hv,w},true);const auto q=T("q",{rows,hk,w},true),k=T("k",{rows,hk,w},true),v=T("v",{rows,hv,w},true);
    const auto gate=T("gate",{rows,hv},true),beta=T("beta",{rows,hv},true);
    if(variant!=Variant::chain) {
        const auto next=T("state_out",{w,hv,w},true),out=T("out",{rows,hv,w},true);
        abi::GdnStepRows a{int32_t(w),int32_t(hk),int32_t(rows)};const TensorId in[]={state,q,k,v,gate,beta},o[]={next,out};
        g.op(variant==Variant::local?"strata_gdn_step_final_f32":"strata_gdn_step_window_f32",in,o,&a,sizeof(a));return;
    }
    const auto slices=[&](TensorId input,const char* name,std::vector<uint64_t> shape) {
        std::vector<TensorId> pieces,values;auto one=shape;one.insert(one.begin(),1);
        for(uint32_t r=0;r<rows;++r) {
            const auto prefix=std::string(name)+std::to_string(r);
            pieces.push_back(T(prefix+"slice",one));values.push_back(T(prefix,shape));
        }
        synSplitParams p{};p.axis=unsigned(shape.size());const TensorId in[]={input};g.op("split",in,pieces,&p,sizeof(p));
        for(uint32_t r=0;r<rows;++r) g.reshape(pieces[r],values[r]);
        return values;
    };
    const auto qs=slices(q,"q",{hk,w}),ks=slices(k,"k",{hk,w}),vs=slices(v,"v",{hv,w}),gs=slices(gate,"g",{hv}),bs=slices(beta,"b",{hv});
    std::vector<TensorId> results;
    for(uint32_t r=0;r<rows;++r) {
        const auto next=T(r+1==rows?"state_out":"state"+std::to_string(r+1),{w,hv,w},r+1==rows);
        const auto out=T("out"+std::to_string(r),{hv,w}),row=T("row"+std::to_string(r),{1,hv,w});
        g.gdn_step(state,qs[r],ks[r],vs[r],gs[r],bs[r],next,out);g.reshape(out,row);results.push_back(row);state=next;
    }
    g.concat(results,T("out",{rows,hv,w},true),0);
}

struct Case {uint32_t w,hk,hv,rows;bool timed;};

// Same fill for every variant of a case: small normals, a per-head decay of exp(-0.1), and a 0.3 write rate, so
// the state stays normal over the window and a bit-exact comparison is meaningful.
void fill_inputs(HostRecipe& recipe) {
    for(const char* tensor:{"state","q","k","v","gate","beta"}) {
        auto x=recipe.input(tensor);
        const bool gate=!std::strcmp(tensor,"gate"),beta=!std::strcmp(tensor,"beta");
        for(size_t i=0;i<x.size();++i) x[i]=gate?-.1f:beta?.3f:float(int((i*13)%127)-63)/1024;
    }
}

std::unique_ptr<HostRecipe> build(const Device& device,const Case& c,Variant variant,const char* role,bool readback) {
    GraphBuilder g(device);define_step(g,c.w,c.hk,c.hv,c.rows,variant);
    const auto graph="gdn_final_"+std::string(name(variant))+"_"+std::to_string(c.w)+"x"+std::to_string(c.hv)+"x"+
                     std::to_string(c.rows)+"_"+role;
    return std::make_unique<HostRecipe>(device,g.compile(graph),std::span<const Binding>{},
        readback?std::vector<std::string>{"state_out","out"}:std::vector<std::string>{});
}

void compare(const Case& c,Variant variant,HostRecipe& chain,HostRecipe& candidate) {
    for(const char* tensor:{"state_out","out"}) {
        const auto x=chain.values(tensor),y=candidate.values(tensor);
        if(x.size()!=y.size() || std::memcmp(x.data(),y.data(),x.size()*sizeof(float))) {
            const size_t n=std::min(x.size(),y.size());
            for(size_t i=0;i<std::min<size_t>(16,n);++i) std::cerr<<i<<":"<<x[i]<<"/"<<y[i]<<" ";
            std::cerr<<"\n";
            double delta=0;size_t worst=0;
            for(size_t i=0;i<n;++i) if(std::abs(x[i]-y[i])>delta) {delta=std::abs(x[i]-y[i]);worst=i;}
            std::cerr<<"worst="<<worst<<" chain="<<(n?x[worst]:0.f)<<" "<<name(variant)<<"="<<(n?y[worst]:0.f)<<"\n";
            throw std::runtime_error("GDN mismatch w="+std::to_string(c.w)+" hk="+std::to_string(c.hk)+
                " hv="+std::to_string(c.hv)+" rows="+std::to_string(c.rows)+" tensor="+tensor+
                " variant="+name(variant)+" max_abs="+std::to_string(delta));
        }
    }
}

struct TimedEvent {
    synEventHandle handle=nullptr;
    explicit TimedEvent(const Device& device) {check(synEventCreate(&handle,device.info().id,EVENT_COLLECT_TIME),"timing event create");}
    ~TimedEvent() {if(handle) (void)synEventDestroy(handle);}
};

// Device-interval time of one window with no output DMA: a recipe with no host outputs, queued launches, events
// recorded around the batch. This is the number the layer chain sees, not a host round trip.
double queued_ms(const Device& device,HostRecipe& recipe) {
    recipe.run(); // compile and upload the inputs once
    Stream stream(device);TimedEvent start(device),end(device);
    constexpr int iterations=16;std::vector<double> samples;
    for(int pass=0;pass<5;++pass) {
        check(synEventRecord(start.handle,stream.handle()),"timing start record");
        for(int i=0;i<iterations;++i) recipe.launch(stream);
        check(synEventRecord(end.handle,stream.handle()),"timing end record");
        check(synEventSynchronize(end.handle),"timing end wait");recipe.wait();
        uint64_t ns=0;check(synEventElapsedTime(&ns,start.handle,end.handle),"timing event elapsed");
        if(!ns) throw std::runtime_error("zero device interval");
        samples.push_back(double(ns)/1e6/iterations);
    }
    std::sort(samples.begin(),samples.end());return samples[samples.size()/2];
}
}
int main(int argc,char** argv) {
 try {
    const bool device=argc==2 && std::string(argv[1])=="--device";
    if(!device) {
        configure_kernel_libraries();check(synInitialize(),"init");
        for(uint32_t rows:{1u,8u,32u,64u,128u}) {
            GraphBuilder g(OfflineCompilation{});strata::core::ModelGeometry c{};
            c.n_embd=256;c.ssm_state_size=128;c.ssm_k_heads=2;c.ssm_v_heads=4;c.ssm_d_conv=4;
            c.ssm_conv_channels=128*8;c.ssm_value_dim=128*4;c.rms_eps=1e-6;
            define_gdn_prefill_final(g,c,rows,true);
            std::cout<<"PASS final GDN compile rows="<<rows<<" workspace="<<g.compile_offline("gdn_final_offline")<<'\n';
        }
        check(synDestroy(),"destroy");return 0;
    }
    Device d;
    std::vector<Case> cases;
    for(const uint32_t w:{64u,128u}) for(const uint32_t rows:{1u,8u,32u,64u,128u}) cases.push_back({w,2,4,rows,false});
    // The production GDN geometry: one row is the decode step, 64 rows the prefill window the plan's GDN
    // figure refers to (36 GDN layers per window). Only these two cases pay for a second set of recipes.
    cases.push_back({128,6,48,1,true});
    cases.push_back({128,6,48,64,true});
    for(const auto& c:cases) {
        auto chain=build(d,c,Variant::chain,"verify",true);
        auto local=build(d,c,Variant::local,"verify",true);
        auto window=build(d,c,Variant::window,"verify",true);
        for(auto* recipe:{chain.get(),local.get(),window.get()}) fill_inputs(*recipe);
        // values() requires a completed run: run all three, then compare both candidates against the chain.
        for(auto* recipe:{chain.get(),local.get(),window.get()}) recipe->run();
        compare(c,Variant::local,*chain,*local);
        compare(c,Variant::window,*chain,*window);
        // Both the published state and every row output must match the per-row chain to the bit.
        std::cout<<"PASS GDN w="<<c.w<<" hk="<<c.hk<<" hv="<<c.hv<<" rows="<<c.rows
                 <<" chain/local/window state+output bit_exact=1\n"<<std::flush;
        if(!c.timed) continue;
        auto timed_chain=build(d,c,Variant::chain,"timing",false);
        auto timed_local=build(d,c,Variant::local,"timing",false);
        auto timed_window=build(d,c,Variant::window,"timing",false);
        for(auto* recipe:{timed_chain.get(),timed_local.get(),timed_window.get()}) fill_inputs(*recipe);
        const double chain_ms=queued_ms(d,*timed_chain),local_ms=queued_ms(d,*timed_local),window_ms=queued_ms(d,*timed_window);
        std::cout<<std::fixed<<std::setprecision(3)
                 <<"TIMING GDN w="<<c.w<<" hk="<<c.hk<<" hv="<<c.hv<<" rows="<<c.rows
                 <<" ms_per_window chain="<<chain_ms<<" local="<<local_ms<<" window="<<window_ms
                 <<" us_per_row chain="<<chain_ms*1000/c.rows<<" local="<<local_ms*1000/c.rows
                 <<" window="<<window_ms*1000/c.rows
                 <<" speedup window_vs_chain="<<chain_ms/window_ms<<"x local_vs_chain="<<chain_ms/local_ms<<"x"
                 <<" (device events, "<<16<<" launches/pass, median of 5)"<<std::defaultfloat<<"\n"<<std::flush;
    }
 } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
