#include "tpc_emulator.hpp"
#include <algorithm>
#include <iostream>
namespace emulated {
#define __global__
#define main row_norm_kernel
#include "../tpc/row_norm.c"
#undef main
#define main hc_mix_kernel
#include "../tpc/hc_mix.c"
#undef main
#define main hc_write_kernel
#include "../tpc/hc_write.c"
#undef main
#define main ple_gate_kernel
#include "../tpc/ple_gate.c"
#undef main
#define main ple_conv_kernel
#include "../tpc/ple_conv.c"
#undef main
#undef __global__
}
using namespace emulated;
namespace {
void require(bool ok,const char* message) {if(!ok) throw std::runtime_error(message);}
double close(const std::vector<float>& actual,const std::vector<float>& expected,float tolerance) {
    require(actual.size()==expected.size(),"shape mismatch");double peak=0;
    for(size_t i=0;i<actual.size();++i) {
        const double delta=std::abs(double(actual[i])-expected[i]);peak=std::max(peak,delta);
        if(!std::isfinite(actual[i]) || delta>tolerance)
            throw std::runtime_error("HC/PLE semantic parity index="+std::to_string(i)+" actual="+std::to_string(actual[i])+" expected="+
                std::to_string(expected[i])+" error="+std::to_string(delta)+" tolerance="+std::to_string(tolerance));
    }
    return peak;
}
tensor vec(std::vector<float>& x) {return {&x,{x.size()}};}
tensor mat(std::vector<float>& x,uint64_t width,uint64_t rows) {return {&x,{width,rows}};}
float sigmoid(float x) {return float(1/(1+std::exp(-double(x))));}
std::vector<float> norm(const std::vector<float>& x,const std::vector<float>& gamma,int width) {
    auto out=x;
    for(size_t base=0;base<x.size();base+=width) {
        double sum=0;for(int d=0;d<width;++d) sum+=double(x[base+d])*x[base+d];
        const float inverse=float(1/std::sqrt(sum/width+1e-6));
        for(int d=0;d<width;++d) out[base+d]=(x[base+d]*inverse)*gamma[gamma.size()==size_t(width)?d:base+d];
    }
    return out;
}
void run(int h,int streams) {
    const int channels=h*streams;const float eps=1e-6f;
    std::vector<float> residual(channels),gamma(channels),gate(channels),inject(streams),block(h),xn(channels),mixed(h),expected(h),out(channels);
    for(int i=0;i<channels;++i) {residual[i]=float(i%17-8)/16;gamma[i]=float(i%7+5)/8;gate[i]=float(i%13-6)/4;}
    for(int i=0;i<h;++i) block[i]=float(i%9-4)/32;
    for(int c=0;c<streams;++c) inject[c]=float(c-1)*2;
    offset={};extent={streams,1,1,1,1};
    row_norm_kernel(mat(residual,h,streams),mat(gamma,h,streams),mat(xn,h,streams),h,streams,0,eps);
    const auto norm_error=close(xn,norm(residual,gamma,h),5e-6f);
    offset={};extent={h/64,1,1,1,1};
    hc_mix_kernel(mat(xn,h,streams),mat(gate,h,streams),vec(mixed),h,streams);
    for(int d=0;d<h;++d) for(int c=0;c<streams;++c) expected[d]+=xn[c*h+d]*sigmoid(gate[c*h+d])/streams;
    close(mixed,expected,5e-6f);
    offset={};extent={h/64,streams,1,1,1};
    hc_write_kernel(mat(residual,h,streams),vec(block),vec(inject),mat(out,h,streams),h,streams);
    expected.resize(channels);
    for(int c=0;c<streams;++c) for(int d=0;d<h;++d) expected[c*h+d]=residual[c*h+d]+block[d]*(2*sigmoid(inject[c]/streams));
    close(out,expected,5e-6f);
    // Zero, positive, negative and nearly-zero signed roots across streams.
    std::vector<float> key(channels),query(channels,1),gated(channels),value(h,0.25f),normalized(channels);
    for(int c=0;c<streams;++c) for(int d=0;d<h;++d) key[c*h+d]=c%4==0?0:(c%4==1?0.01f:(c%4==2?-0.01f:1e-10f));
    offset={};extent={streams,1,1,1,1};
    ple_gate_kernel(mat(key,h,streams),mat(query,h,streams),vec(value),mat(gated,h,streams),h,streams,1/std::sqrt(float(h)));
    for(int c=0;c<streams;++c) {
        double dot=0;for(int d=0;d<h;++d) dot+=double(key[c*h+d])*query[c*h+d];
        const float score=float(dot)/std::sqrt(float(h));
        const float root=(score>0?1.0f:(score<0?-1.0f:0))*std::sqrt(std::max(std::abs(score),1e-6f));
        for(int d=0;d<h;++d) expected[c*h+d]=value[d]*sigmoid(root);
    }
    close(gated,expected,3e-6f);
    row_norm_kernel(mat(gated,h,streams),mat(gamma,h,streams),mat(normalized,h,streams),h,streams,0,eps);
    close(normalized,norm(gated,gamma,h),5e-6f);
    // History values identify row/channel, exposing either transposition error
    // or a wrong tap offset. Subsequent steps must append normalized, not gated.
    std::vector<float> history(9*channels),weights(4*channels),next(history.size()),reference_next(history.size());
    for(size_t i=0;i<history.size();++i) history[i]=float(int(i%23)-11)/64;
    for(size_t i=0;i<weights.size();++i) weights[i]=float(int(i%11)-5)/32;
    double worst=0;
    for(int step=0;step<5;++step) {
        offset={};extent={channels/64,1,1,1,1};
        ple_conv_kernel(mat(history,channels,9),vec(normalized),mat(weights,channels,4),vec(residual),vec(gated),
                        mat(next,channels,9),vec(out),channels,4,3);
        for(int channel=0;channel<channels;++channel) {
            float sum=0;for(int tap=0;tap<4;++tap) sum+=(tap==3?normalized[channel]:history[tap*3*channels+channel])*weights[tap*channels+channel];
            expected[channel]=(residual[channel]+gated[channel])+sum*sigmoid(sum);
            for(int row=0;row<9;++row) reference_next[row*channels+channel]=row==8?normalized[channel]:history[(row+1)*channels+channel];
        }
        worst=std::max(worst,close(out,expected,5e-6f));close(next,reference_next,0);history.swap(next);
    }
    // Shared gamma and huge inputs exercise the scaled normalization branch.
    std::vector<float> shared(h,1);for(auto& v:residual) v*=1e24f;
    offset={};extent={streams,1,1,1,1};
    row_norm_kernel(mat(residual,h,streams),vec(shared),mat(xn,h,streams),h,streams,1,eps);
    close(xn,norm(residual,shared,h),1e-5f);
    std::cout<<"PASS CPU original TPC HC/PLE H="<<h<<" HC="<<streams<<" norm_error="<<norm_error<<" five_step_PLE_error="<<worst<<'\n';
}
}
int main() {
    try {run(64,4);run(2560,4);run(128,3);}
    catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
