#include "tpc_emulator.hpp"
#include "gdn_fixture.hpp"
namespace emulated {
#define __global__
#define main conv_kernel
#include "../tpc/gdn_conv.c"
#undef main
#define main prepare_kernel
#include "../tpc/gdn_prepare.c"
#undef main
#define main recurrence_kernel
#include "../tpc/gdn_step.c"
#undef main
#define main norm_kernel
#include "../tpc/gdn_norm.c"
#undef main
#undef __global__
}
using namespace emulated;
void run(int s,int hk,int hv,int taps,float amplitude=1) {
    gdn_test::Fixture f(s,hk,hv,taps);
    for(auto& value:f.conv) value*=amplitude;
    auto state=f.state,conv=f.time_major_conv(),weights=f.time_major_weights();
    std::vector<float> next(state.size()),conv_next(conv.size()),activated(f.c),q(hk*s),k(hk*s),v(hv*s),gate(hv),beta(hv),recurrent(hv*s),out(hv*s);
    auto vector=[](std::vector<float>& data) { return tensor{&data,{data.size()}}; };
    for(int phase=0;phase<4;++phase) {
        f.inputs(phase);
        for(auto& value:f.qkv) value*=amplitude;
        offset={}; extent={(f.c+63)/64,1,1,1,1};
        conv_kernel({&conv,{uint64_t(f.c),uint64_t(taps-1)}},vector(f.qkv),{&weights,{uint64_t(f.c),uint64_t(taps)}},
                    {&conv_next,{uint64_t(f.c),uint64_t(taps-1)}},vector(activated),f.c,taps);
        // Execute head-zero partition last: gate/beta have exactly one writer.
        for(int head=hv;head-- >0;) {
            offset={head,0,0,0,0}; extent={1,1,1,1,1};
            prepare_kernel(vector(activated),vector(f.alpha),vector(f.dt),vector(f.a),vector(f.raw_beta),
                {&q,{uint64_t(s),uint64_t(hk)}},{&k,{uint64_t(s),uint64_t(hk)}},{&v,{uint64_t(s),uint64_t(hv)}},
                vector(gate),vector(beta),s,hk,hv,f.eps);
        }
        for(int head=hv;head-- >0;) for(int block=0;block<s/64;++block) {
            offset={block,head,0,0,0}; extent={1,1,1,1,1};
            recurrence_kernel({&state,{uint64_t(s),uint64_t(hv),uint64_t(s)}},{&q,{uint64_t(s),uint64_t(hk)}},
                {&k,{uint64_t(s),uint64_t(hk)}},{&v,{uint64_t(s),uint64_t(hv)}},vector(gate),vector(beta),
                {&next,{uint64_t(s),uint64_t(hv),uint64_t(s)}},{&recurrent,{uint64_t(s),uint64_t(hv)}},s,hk);
        }
        offset={}; extent={hv,1,1,1,1};
        norm_kernel({&recurrent,{uint64_t(s),uint64_t(hv)}},{&f.z,{uint64_t(s),uint64_t(hv)}},vector(f.norm),
                    {&out,{uint64_t(s),uint64_t(hv)}},s,hv,f.eps);
        f.reference_step();
        gdn_test::compare(conv_next,f.time_major_conv(),"conv_state",0);
        auto tolerance=[](const std::vector<float>& values) {
            double peak=0; for(auto value:values) peak=std::max(peak,double(std::abs(value)));
            return std::max(1e-5,peak*5e-6);
        };
        for(auto pair : {std::pair{&activated,&f.activated}, {&q,&f.q}, {&k,&f.k}, {&v,&f.v},
                        {&gate,&f.gate}, {&beta,&f.beta}, {&recurrent,&f.recurrent}})
            gdn_test::compare(*pair.first,*pair.second,"intermediate",tolerance(*pair.second));
        const auto state_error=gdn_test::compare(next,f.state,"state",tolerance(f.state));
        const auto out_error=gdn_test::compare(out,f.out,"out",3e-5);
        std::cout << "PASS CPU emulation of original TPC sources S=" << s << " hk=" << hk << " hv=" << hv
                  << " taps=" << taps << " amplitude=" << amplitude << " phase=" << phase << " state_error=" << state_error << " output_error=" << out_error << '\n';
        state.swap(next); conv.swap(conv_next);
    }
}
int main() {
    try { run(64,1,3,2); run(128,16,48,4); run(256,2,6,4); run(64,2,6,4,1e24f); }
    catch(const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
