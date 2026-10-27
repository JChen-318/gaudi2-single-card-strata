// A verify window's GDN recurrent steps on random inputs: four chained gdn_step launches (the per-row chain, with
// the flags of the environment, e.g. STRATA_HPU_CACHED_GDN_STEP + GDN_STEP_PIPE) against one
// strata_gdn_step_rows4_f32 node (STRATA_HPU_GDN_STEP_ROWS). Prints the device time of each graph and the largest
// differences of the four row states and outputs. Hardware only, not CTest.
//   strata-hpu-gdn-step-probe [rows 4] [iters 50]
#include "strata/hpu/recipe.hpp"
#include "strata/hpu/gdn_abi.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <string>
#include <vector>
using namespace strata::hpu;
int main(int argc,char** argv) {
    try {
        const uint32_t rows=argc>1?uint32_t(std::stoul(argv[1])):4,iters=argc>2?uint32_t(std::stoul(argv[2])):50;
        const uint64_t s=128,hk=16,hv=48,state_bytes=s*hv*s*4;
        Device device;Stream stream(device);
        DeviceBuffer state(device,state_bytes),q(device,rows*hk*s*4),k(device,rows*hk*s*4),v(device,rows*hv*s*4),gate(device,rows*hv*4),beta(device,rows*hv*4);
        std::vector<std::unique_ptr<DeviceBuffer>> a_states,b_states;
        for(int i=0;i<4;++i) {a_states.push_back(std::make_unique<DeviceBuffer>(device,state_bytes));b_states.push_back(std::make_unique<DeviceBuffer>(device,state_bytes));}
        DeviceBuffer a_out(device,rows*hv*s*4),b_out(device,rows*hv*s*4);
        std::vector<std::unique_ptr<DeviceBuffer>> row_gate,row_beta; // the chain's per-row copies (row offsets are not 128-byte aligned)
        for(uint32_t r=0;r<rows;++r) {row_gate.push_back(std::make_unique<DeviceBuffer>(device,hv*4));row_beta.push_back(std::make_unique<DeviceBuffer>(device,hv*4));}
        {
            std::mt19937 rng(7);std::normal_distribution<float> n(0.0f,1.0f);std::uniform_real_distribution<float> u(0.0f,1.0f);
            const auto fill=[&](DeviceBuffer& b,uint64_t count,auto make) {
                HostBuffer h(device,count*4);auto* p=static_cast<float*>(h.data());for(uint64_t i=0;i<count;++i) p[i]=make(i);
                upload(stream,b,h,count*4);stream.synchronize();
            };
            fill(state,s*hv*s,[&](uint64_t){return 0.1f*n(rng);});
            // q, k: unit rows per head (q also scaled by 1/sqrt(s)), as gdn_prepare gives them.
            for(auto* b:{&q,&k}) {
                std::vector<float> x(rows*hk*s);for(auto& e:x) e=n(rng);
                for(uint64_t h=0;h<rows*hk;++h) {double z=0;for(uint64_t i=0;i<s;++i) z+=double(x[h*s+i])*x[h*s+i];
                    const float f=float(1.0/std::sqrt(z))*(b==&q?float(1.0/std::sqrt(double(s))):1.0f);for(uint64_t i=0;i<s;++i) x[h*s+i]*=f;}
                fill(*b,x.size(),[&](uint64_t i){return x[i];});
            }
            fill(v,rows*hv*s,[&](uint64_t){return n(rng);});
            std::vector<float> gv(rows*hv),bv(rows*hv);for(auto& e:gv) e=-2.0f*u(rng);for(auto& e:bv) e=u(rng);
            fill(gate,rows*hv,[&](uint64_t i){return gv[i];});fill(beta,rows*hv,[&](uint64_t i){return bv[i];});
            for(uint32_t r=0;r<rows;++r) {fill(*row_gate[r],hv,[&](uint64_t i){return gv[r*hv+i];});fill(*row_beta[r],hv,[&](uint64_t i){return bv[r*hv+i];});}
        }
        const auto time=[&](Recipe& r,const std::vector<Binding>& b) {
            r.enqueue(stream,b);stream.synchronize();
            const auto begin=std::chrono::steady_clock::now();
            for(uint32_t i=0;i<iters;++i) r.enqueue(stream,b);
            stream.synchronize();
            return std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-begin).count()/iters;
        };
        // A: the chain, one gdn_step per row.
        std::unique_ptr<Recipe> chain;std::vector<Binding> ab;
        {
            GraphBuilder g(device);
            auto prev=g.tensor({"state",DType::F32,{s,hv,s},true});ab.push_back({"state",&state});
            std::vector<TensorId> outs;
            for(uint32_t r=0;r<rows;++r) {
                const auto R=std::to_string(r);
                const auto qt=g.tensor({"q"+R,DType::F32,{hk,s},true}),kt=g.tensor({"k"+R,DType::F32,{hk,s},true}),vt=g.tensor({"v"+R,DType::F32,{hv,s},true});
                const auto gt=g.tensor({"g"+R,DType::F32,{hv},true}),bt=g.tensor({"b"+R,DType::F32,{hv},true});
                const auto next=g.tensor({"s"+R,DType::F32,{s,hv,s},true}),out=g.tensor({"o"+R,DType::F32,{hv,s},true});
                g.gdn_step(prev,qt,kt,vt,gt,bt,next,out);prev=next;
                ab.push_back({"q"+R,&q,r*hk*s*4});ab.push_back({"k"+R,&k,r*hk*s*4});ab.push_back({"v"+R,&v,r*hv*s*4});
                ab.push_back({"g"+R,row_gate[r].get()});ab.push_back({"b"+R,row_beta[r].get()});
                ab.push_back({"s"+R,a_states[r].get()});ab.push_back({"o"+R,&a_out,r*hv*s*4});
            }
            chain=g.compile("strata_gdn_step_probe_chain");
        }
        // B: one rows4 node.
        std::unique_ptr<Recipe> merged;std::vector<Binding> bb;
        {
            GraphBuilder g(device);
            const auto st=g.tensor({"state",DType::F32,{s,hv,s},true}),qt=g.tensor({"q",DType::F32,{rows,hk,s},true}),kt=g.tensor({"k",DType::F32,{rows,hk,s},true});
            const auto vt=g.tensor({"v",DType::F32,{rows,hv,s},true}),gt=g.tensor({"g",DType::F32,{rows,hv},true}),bt=g.tensor({"b",DType::F32,{rows,hv},true});
            std::vector<TensorId> outs;
            for(int r=0;r<4;++r) outs.push_back(g.tensor({"s"+std::to_string(r),DType::F32,{s,hv,s},true}));
            outs.push_back(g.tensor({"o",DType::F32,{rows,hv,s},true}));
            abi::GdnStepRows a{int32_t(s),int32_t(hk),int32_t(rows)};const TensorId in[]={st,qt,kt,vt,gt,bt};
            g.op(std::getenv("PROBE_GUID")?std::getenv("PROBE_GUID"):"strata_gdn_step_rows4_f32",in,outs,&a,sizeof(a));
            merged=g.compile("strata_gdn_step_probe_rows");
            bb={{"state",&state},{"q",&q},{"k",&k},{"v",&v},{"g",&gate},{"b",&beta},{"o",&b_out}};
            for(int r=0;r<4;++r) bb.push_back({"s"+std::to_string(r),b_states[size_t(r)].get()});
        }
        // Device time: each graph repeated `reps` times back to back in one launch (state_out_3 / the last row's
        // state feeding the next copy), the launch overhead divided out.
        const uint32_t reps=std::getenv("PROBE_REPS")?uint32_t(std::stoul(std::getenv("PROBE_REPS"))):8;
        std::unique_ptr<Recipe> chain_reps,merged_reps;
        {
            GraphBuilder g(device);auto prev=g.tensor({"state",DType::F32,{s,hv,s},true});
            for(uint32_t c=0;c<reps;++c) for(uint32_t r=0;r<rows;++r) {
                const auto R=std::to_string(r),C=std::to_string(c)+"_"+R;
                const auto qt=c?g.find("q"+R):g.tensor({"q"+R,DType::F32,{hk,s},true}),kt=c?g.find("k"+R):g.tensor({"k"+R,DType::F32,{hk,s},true});
                const auto vt=c?g.find("v"+R):g.tensor({"v"+R,DType::F32,{hv,s},true});
                const auto gt=c?g.find("g"+R):g.tensor({"g"+R,DType::F32,{hv},true}),bt=c?g.find("b"+R):g.tensor({"b"+R,DType::F32,{hv},true});
                const bool last=c+1==reps;
                const auto next=last?g.tensor({"s"+R,DType::F32,{s,hv,s},true}):g.tensor({"s"+C,DType::F32,{s,hv,s},false});
                const auto out=last?g.tensor({"o"+R,DType::F32,{hv,s},true}):g.tensor({"o"+C,DType::F32,{hv,s},false});
                g.gdn_step(prev,qt,kt,vt,gt,bt,next,out);prev=next;
            }
            chain_reps=g.compile("strata_gdn_step_probe_chain_reps");
        }
        {
            GraphBuilder g(device);
            auto st=g.tensor({"state",DType::F32,{s,hv,s},true});const auto qt=g.tensor({"q",DType::F32,{rows,hk,s},true}),kt=g.tensor({"k",DType::F32,{rows,hk,s},true});
            const auto vt=g.tensor({"v",DType::F32,{rows,hv,s},true}),gt=g.tensor({"g",DType::F32,{rows,hv},true}),bt=g.tensor({"b",DType::F32,{rows,hv},true});
            for(uint32_t c=0;c<reps;++c) {
                const bool last=c+1==reps;const auto C=last?std::string():"_"+std::to_string(c);
                std::vector<TensorId> outs;
                for(int r=0;r<4;++r) outs.push_back(g.tensor({"s"+std::to_string(r)+C,DType::F32,{s,hv,s},last}));
                outs.push_back(g.tensor({"o"+C,DType::F32,{rows,hv,s},last}));
                abi::GdnStepRows a{int32_t(s),int32_t(hk),int32_t(rows)};const TensorId in[]={st,qt,kt,vt,gt,bt};
                g.op(std::getenv("PROBE_GUID")?std::getenv("PROBE_GUID"):"strata_gdn_step_rows4_f32",in,outs,&a,sizeof(a));
                st=outs[3];
            }
            merged_reps=g.compile("strata_gdn_step_probe_rows_reps");
        }
        const double ta=time(*chain_reps,ab)/reps,tb=time(*merged_reps,bb)/reps;
        // The single copies, for the comparison below.
        (void)time(*chain,ab);(void)time(*merged,bb);
        HostBuffer ha(device,state_bytes),hb(device,state_bytes);
        const auto diff=[&](const DeviceBuffer& x,const DeviceBuffer& y,uint64_t bytes) {
            download(stream,ha,x,bytes);download(stream,hb,y,bytes);stream.synchronize();
            const auto* p=static_cast<const float*>(ha.data());const auto* r=static_cast<const float*>(hb.data());
            double worst=0,scale=0;for(uint64_t i=0;i<bytes/4;++i) {worst=std::max(worst,double(std::fabs(p[i]-r[i])));scale=std::max(scale,double(std::fabs(p[i])));}
            return std::pair{worst,scale};
        };
        std::cout<<"rows "<<rows<<": chain "<<ta<<" us, rows4 "<<tb<<" us per window (device, "<<reps<<" copies per launch)\n";
        for(uint32_t r=0;r<rows;++r) {const auto [w,m]=diff(*a_states[r],*b_states[r],state_bytes);std::cout<<"  state "<<r<<" max|diff| "<<w<<" (max|x| "<<m<<")\n";}
        const auto [w,m]=diff(a_out,b_out,rows*hv*s*4);std::cout<<"  outputs max|diff| "<<w<<" (max|x| "<<m<<")\n";
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
