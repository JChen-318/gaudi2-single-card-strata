#include "strata/hpu/mtp_window_flow.hpp"
#include "strata/hpu/mtp_window_plan.hpp"
#include "strata/hpu/dense_kv.hpp"
#include "strata/hpu/session_flow.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <memory>
#include <numeric>
using namespace strata::hpu;
namespace {
void require(bool ok,const char* why){if(!ok) throw std::runtime_error(why);}
template<class F> void rejects(F call){bool failed=false;try {call();}catch(const std::exception&){failed=true;}require(failed,"invalid MTP window accepted");}
struct Fixture {
    uint32_t h=4,r=3;DenseKvState kv{256,1,1};std::vector<float> eg{.5f,.75f,1.25f,1.5f},hg;
    int bad=0,calls=0;uint32_t heads=0,moes=0,hidden_calls=0,largest_hidden=0;
    explicit Fixture(uint32_t streams=3):r(streams),hg(h*r){for(size_t i=0;i<hg.size();++i) hg[i]=float(i%9+4)/8;}
    std::vector<float> fc(std::span<const float> x) const {std::vector<float> out(x.size());for(size_t row=0;row<x.size()/h;++row) for(uint32_t d=0;d<h;++d) out[row*h+d]=x[row*h+d]*float(d+1)/8+x[row*h+(d+1)%h]/4;return out;}
    HcReadResult read(std::span<const float> x,bool ffn) const {HcReadResult out;out.mixed.assign(h,ffn?.125f:0);out.inject.resize(r);for(uint32_t s=0;s<r;++s) {out.inject[s]=float(s+1)/64;for(uint32_t d=0;d<h;++d) out.mixed[d]+=x[s*h+d]/r;}return out;}
    std::vector<float> write(std::span<const float> x,std::span<const float> block,std::span<const float> inject) const {std::vector<float> out(x.begin(),x.end());for(uint32_t s=0;s<r;++s) for(uint32_t d=0;d<h;++d) out[s*h+d]+=block[d]*(1+inject[s]);return out;}
    std::vector<float> mixer(std::span<const float> x) {
        const std::array<float,1> key{x[0]},value{x[1]};kv.append(key,value);auto all=kv.gather(kv.size());
        const float sum=std::accumulate(all.value.begin(),all.value.end(),0.0f)/32;std::vector<float> out(x.begin(),x.end());for(auto& v:out) v=v/4+sum;return out;
    }
    std::vector<float> head(std::span<const float> x){++heads;std::vector<float> out(16,-INFINITY);out[2]=std::accumulate(x.begin(),x.end(),0.0f);out[5]=-out[2]/2;return out;}
    MtpWindowOperations operations() {
        const auto point=kv.checkpoint();auto rewound=std::make_shared<bool>(false);MtpWindowOperations op;
        op.rollback=[this,point,rewound]{if(!*rewound){*rewound=true;kv.restore(point);}};
        op.embedding=[this](auto x){++calls;if(bad==1 && calls==2) throw std::runtime_error("FC embedding failed");return fc(x);};
        op.hidden=[this](auto x){++hidden_calls;largest_hidden=std::max(largest_hidden,uint32_t(x.size()/h));auto out=fc(x);if(bad==2 && calls==2) out.pop_back();return out;};
        op.head=[this](auto x){auto out=head(x);if(bad==7) out.pop_back();if(bad==8) out[2]=INFINITY;return out;};
        auto& layer=op.layer;
        layer.read=[this](auto x,bool ffn){return read(x,ffn);};layer.write=[this](auto x,auto b,auto i,bool){return write(x,b,i);};
        layer.mixer=[this](auto x){return mixer(x);};layer.moe=[this](auto x){++moes;if(bad==5 && calls==2) throw std::runtime_error("MoE failed");std::vector<float> out(x.begin(),x.end());for(auto& v:out) v=v/8+.25f;return out;};layer.rollback_mixer=op.rollback;
        layer.window_read=[this](auto x,bool ffn){if(bad==3 && calls==2) throw std::runtime_error("HC read failed");HcReadResult result;for(size_t row=0;row<x.size()/(h*r);++row){auto out=read(x.subspan(row*h*r,h*r),ffn);result.mixed.insert(result.mixed.end(),out.mixed.begin(),out.mixed.end());result.inject.insert(result.inject.end(),out.inject.begin(),out.inject.end());}return result;};
        layer.window_write=[this](auto x,auto b,auto i,bool){if(bad==6 && calls==2) throw std::runtime_error("HC write failed");std::vector<float> result;for(size_t row=0;row<x.size()/(h*r);++row){auto out=write(x.subspan(row*h*r,h*r),b.subspan(row*h,h),i.subspan(row*r,r));result.insert(result.end(),out.begin(),out.end());}return result;};
        layer.window_mixer=[this](auto x){const auto point=kv.checkpoint();try {std::vector<float> result;for(size_t at=0;at<x.size();at+=h){auto out=mixer(x.subspan(at,h));if(bad==4 && calls==2) throw std::runtime_error("native mixer failed after append");result.insert(result.end(),out.begin(),out.end());}return result;}catch(...){kv.restore(point);throw;}};
        return op;
    }
    MtpOutput scalar(std::span<const float> residual,std::span<const float> embedding,bool stream) {
        std::vector<float> e(embedding.begin(),embedding.end()),hidden(residual.begin(),residual.end());
        auto norm=[](std::span<float> x,uint32_t width,std::span<const float> gamma){for(size_t row=0;row<x.size()/width;++row){double sum=0;for(uint32_t i=0;i<width;++i) sum+=double(x[row*width+i])*x[row*width+i];const float inv=float(1/std::sqrt(sum/width+double(1e-6f)));for(uint32_t i=0;i<width;++i) x[row*width+i]=(x[row*width+i]*inv)*gamma[gamma.size()==width?i:row*width+i];}};
        norm(e,h,eg);norm(hidden,stream?h:h*r,hg);const auto pe=fc(e);std::vector<float> initial(h*r);
        for(uint32_t s=0;s<r;++s){const auto ph=fc(std::span<const float>(hidden).subspan(s*h,h));for(uint32_t d=0;d<h;++d) initial[s*h+d]=ph[d]+pe[d];}
        const auto a=read(initial,false);const auto mixed=mixer(a.mixed);auto next=write(initial,mixed,a.inject);const auto f=read(next,true);auto expert=f.mixed;for(auto& v:expert) v=v/8+.25f;
        auto result=write(next,expert,f.inject);auto logits=head(result);return {std::move(result),std::move(logits)};
    }
};
void run(uint32_t rows,uint32_t streams,bool stream) {
    Fixture a(streams),b(streams);std::vector<float> x(rows*4*streams),e(rows*4);
    for(size_t i=0;i<x.size();++i) x[i]=float(int(i*7%29)-14)/11;
    for(size_t i=0;i<e.size();++i) e[i]=float(int(i*13%31)-15)/13;
    MtpOutput expected;for(uint32_t row=0;row<rows;++row) expected=b.scalar(std::span<const float>(x).subspan(row*4*streams,4*streams),std::span<const float>(e).subspan(row*4,4),stream);
    const auto actual=execute_mtp_window(x,e,4,streams,16,{a.eg,a.hg,1e-6f,stream},a.operations());
    require(actual.residual==expected.residual && actual.logits==expected.logits,"native MTP orchestration differs from independent singleton norm/FC/HC/MoE oracle");
    require(a.kv.gather(rows).key==b.kv.gather(rows).key && a.kv.gather(rows).value==b.kv.gather(rows).value,"MTP window private KV diverged");
    require(a.heads==1 && a.moes==(rows+7)/8 && a.largest_hidden<=128,"MTP failed final-only head/grouped MoE/bounded FC execution");
}
void failures() {
    Fixture a;const std::vector<float> x(17*12,.5f),e(17*4,.75f);const auto start=a.kv.checkpoint();
    for(int bad=1;bad<=8;++bad){a.bad=bad;a.calls=0;rejects([&]{(void)execute_mtp_window(x,e,4,3,16,{a.eg,a.hg},a.operations());});require(!a.kv.size(),"failed MTP stage retained KV");}a.bad=0;
    uint32_t polls=0;(void)execute_mtp_window(x,e,4,3,16,{a.eg,a.hg},a.operations(),[&]{++polls;return false;});a.kv.restore(start);
    for(uint32_t boundary=1;boundary<=polls;++boundary){uint32_t n=0;rejects([&]{(void)execute_mtp_window(x,e,4,3,16,{a.eg,a.hg},a.operations(),[&]{return ++n==boundary;});});require(!a.kv.size(),"MTP cancellation retained known-prefix cells");}
    std::cout<<"PASS MTP window all FC/HC/mixer/MoE/head failures, subset logits and "<<polls<<" cancel boundaries restore root\n";
}
void pairs_and_budget() {
    for(uint32_t rows:{1u,3u,8u,17u,128u}) for(bool resumed:{false,true}) {
        std::vector<float> x(rows*12),e(rows*4),root(12,-7);std::iota(x.begin(),x.end(),1.0f);std::iota(e.begin(),e.end(),101.0f);
        const auto paired=pair_mtp_prefix(x,e,resumed?std::span<const float>(root):std::span<const float>{},4,3);const auto count=rows-(resumed?0:1);
        require(paired.embedding.size()==count*4 && paired.residual.size()==count*12,"prefix pairing count mismatch");
        for(uint32_t row=0;row<count;++row){const auto original=row+(resumed?0:1);for(uint32_t d=0;d<12;++d) require(paired.residual[row*12+d]==(original?x[(original-1)*12+d]:root[d]),"known-prefix previous target residual misaligned");for(uint32_t d=0;d<4;++d) require(paired.embedding[row*4+d]==e[original*4+d],"known-prefix next embedding misaligned");}
    }
    const auto& g=strata::core::kCompiledGeometry;require(!plan_mtp_window(g,0,128).hbm_bytes,"disabled MTP window consumes budget");
    for(uint32_t context:{1u,7u,128u,32768u}) for(uint32_t rows:{1u,3u,7u,8u,9u,128u}) {
        const auto p=plan_mtp_window(g,rows,context);require(p.rows==std::min({rows,context,8u}) && p.pinned_bytes>0 && p.hbm_bytes>p.pinned_bytes && p.host_bytes>p.pinned_bytes,"MTP window budget missing slots/workspace/copies");
        const auto narrow=plan_mtp_window(g,rows,context,1);require(narrow.hbm_bytes<=p.hbm_bytes && narrow.host_bytes<=p.host_bytes,"dense window size not reflected in budget");
    }
    rejects([&]{(void)plan_mtp_window(g,129,128);});rejects([&]{(void)plan_mtp_window(g,1,0);});
    std::cout<<"PASS MTP root/new-session pairing10 cases and window budget24 context/chunk cases; no vocabulary slab\n";
}
void admission() {
    const auto& g=strata::core::kCompiledGeometry;uint32_t cases=0;
    for(uint32_t context:{1u,3u,8u,17u,128u}) for(uint32_t window:{0u,1u}) for(uint32_t row=1;row<=std::min(context,128u);++row) {
        const auto target=plan_prefill_window(g,row,context);const auto draft=plan_mtp_window(g,row,context,window);
        const auto host=target.host_bytes+draft.host_bytes,hbm=target.hbm_bytes+draft.hbm_bytes;
        for(const auto& limits:{std::array<uint64_t,3>{host,hbm,draft.pinned_bytes},std::array<uint64_t,3>{host-1,hbm,UINT64_MAX},std::array<uint64_t,3>{host,hbm-1,UINT64_MAX},std::array<uint64_t,3>{host,hbm,draft.pinned_bytes-1}}) {
            const auto selected=select_model_prefill_window(g,context,limits[0],limits[1],limits[2],true,window);uint32_t expected=0;
            // Independent descending exhaustive admission against explicit costs.
            for(uint32_t n=std::min(context,128u);n>0;--n){const auto a=plan_prefill_window(g,n,context);const auto b=plan_mtp_window(g,n,context,window);if(a.host_bytes+b.host_bytes<=limits[0] && a.hbm_bytes+b.hbm_bytes<=limits[1] && b.pinned_bytes<=limits[2]){expected=n;break;}}
            require(selected.target.rows==expected,"joint target/MTP auto window violates resource maximality");++cases;
        }
        const auto simple=select_prefill_window(g,context,host,hbm);const auto no_mtp=select_model_prefill_window(g,context,host,hbm,0,false);
        require(simple.rows==no_mtp.target.rows && !no_mtp.mtp.rows,"joint planner changed target-only admission");
    }
    std::cout<<"PASS joint target/MTP auto admission "<<cases<<" exact/one-byte host/HBM/pinned cases vs independent descending search, dense window and scalar fallback\n";
}
void diagnostic_admission() {
    const auto& g=strata::core::kCompiledGeometry;uint32_t cases=0;
    // Fixed ladder plus row-dependent capture with a seven-row scalar envelope.
    const uint64_t ladder=(uint64_t(g.n_layers)+1)*uint64_t(g.hc)*uint64_t(g.n_embd)*4;
    const WindowHostCost extra=[&](uint32_t rows){return ladder+4096+uint64_t(g.n_layers)*std::max(7u,rows)*uint64_t(g.n_expert_used)*16;};
    for(bool mtp:{false,true}) for(uint32_t context:{7u,17u,128u}) for(uint32_t minimum:{0u,7u}) {
        for(uint32_t row:{0u,1u,7u,8u,17u,128u}) {
            if(row>context) continue;
            const auto a=plan_prefill_window(g,row,context);const auto b=mtp?plan_mtp_window(g,row,context,1):MtpWindowMemoryPlan{};
            const auto host=a.host_bytes+b.host_bytes+extra(row),hbm=a.hbm_bytes+b.hbm_bytes;
            for(const auto& limits:{std::array<uint64_t,3>{host,hbm,b.pinned_bytes},std::array<uint64_t,3>{host-1,UINT64_MAX,UINT64_MAX},std::array<uint64_t,3>{host,hbm?hbm-1:0,UINT64_MAX},std::array<uint64_t,3>{host,hbm,b.pinned_bytes?b.pinned_bytes-1:0}}) {
                int expected=-1;
                for(int n=int(context);n>=int(minimum);--n) {
                    const auto t=plan_prefill_window(g,uint32_t(n),context);const auto m=mtp?plan_mtp_window(g,uint32_t(n),context,1):MtpWindowMemoryPlan{};
                    if(t.host_bytes+m.host_bytes+extra(uint32_t(n))<=limits[0] && t.hbm_bytes+m.hbm_bytes<=limits[1] && m.pinned_bytes<=limits[2]) {expected=n;break;}
                }
                if(expected<0) rejects([&]{(void)select_model_prefill_window(g,context,limits[0],limits[1],limits[2],mtp,1,extra,minimum);});
                else {
                    const auto p=select_model_prefill_window(g,context,limits[0],limits[1],limits[2],mtp,1,extra,minimum);
                    require(p.target.rows==uint32_t(expected) && p.additional_host_bytes==extra(uint32_t(expected)),"diagnostic joint budget maximum/cost mismatch");
                    require(p.target.rows>=minimum && p.target.host_bytes+p.mtp.host_bytes+p.additional_host_bytes<=limits[0],"diagnostic cost omitted or verify window disabled");
                }
                ++cases;
            }
        }
    }
    // Disabled graphs still require the diagnostic buffers. The zero-row result
    // must carry its real charge, and fixed features cannot silently disappear.
    const auto scalar=select_model_prefill_window(g,128,extra(0),0,0,true,0,extra);
    require(!scalar.target.rows && !scalar.mtp.rows && scalar.additional_host_bytes==extra(0),"scalar fallback lost diagnostic charge");
    rejects([&]{(void)select_model_prefill_window(g,128,extra(0)-1,UINT64_MAX,UINT64_MAX,false,0,extra);});
    rejects([&]{(void)select_model_prefill_window(g,128,UINT64_MAX,0,0,false,0,extra,7);});
    rejects([&]{(void)select_model_prefill_window(g,7,UINT64_MAX,UINT64_MAX,UINT64_MAX,false,0,extra,8);});
    rejects([&]{(void)select_model_prefill_window(g,128,UINT64_MAX,UINT64_MAX,UINT64_MAX,false,0,[](uint32_t){return UINT64_MAX;},1);});
    // Search all candidates rather than assuming the callback is monotonic.
    const WindowHostCost stepped=[](uint32_t rows){return rows==17?0:UINT64_MAX;};
    const auto unusual=select_model_prefill_window(g,32,plan_prefill_window(g,17,32).host_bytes,UINT64_MAX,UINT64_MAX,false,0,stepped);
    require(unusual.target.rows==17,"joint selector stopped before a later feasible cost tier");
    std::cout<<"PASS joint diagnostics "<<cases<<" boundary cases, fixed/scalar costs, verify floor, MTP/pinned/HBM, overflow-safe refusal and nonmonotonic cost tiers; no device\n";
}
}
int main(){try {uint32_t cases=0;for(uint32_t rows:{1u,3u,7u,8u,9u,17u,128u}) for(uint32_t streams:{3u,32u}) for(bool norm:{false,true}) {run(rows,streams,norm);++cases;}failures();pairs_and_budget();admission();diagnostic_admission();std::cout<<"PASS production MTP window "<<cases<<" singleton-oracle cases, stream/flat norm, FC128 bound, final-only head and grouped layer execution; no device\n";}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
