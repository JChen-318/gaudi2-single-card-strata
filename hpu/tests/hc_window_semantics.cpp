#include "tpc_emulator.hpp"
#include <algorithm>
#include <iostream>
#include <limits>
namespace emulated {
#define __global__
#define main row_norm_window_kernel
#include "../tpc/row_norm.c"
#undef main
#define main hc_mix_window_kernel
#include "../tpc/hc_mix_window.c"
#undef main
#define main hc_write_window_kernel
#include "../tpc/hc_write_window.c"
#undef main
#undef __global__
}
using namespace emulated;
namespace {
void close(float actual,double expected,double tolerance,const char* name) {
    if(!std::isfinite(actual) || std::abs(double(actual)-expected)>tolerance*std::max(1.0,std::abs(expected)))
        throw std::runtime_error(std::string(name)+" actual="+std::to_string(actual)+" expected="+std::to_string(expected));
}
void run(int h,int r,int t,float amplitude=1) {
    const auto n=h*r;const auto size=size_t(n)*t;
    std::vector<float> x(size),gamma(n),gate(size),block(size_t(t)*h),inject(t*r),xn(size,NAN),mixed(size_t(t)*h,NAN),output(size,NAN);
    for(size_t i=0;i<size;++i) {x[i]=float(int(i%29)-14)*amplitude/16;gate[i]=float(int((i+17*(i/n))%37)-18)/8;}
    for(int i=0;i<n;++i) gamma[i]=float(i%13+7)/16;
    for(size_t i=0;i<block.size();++i) block[i]=float(int(i%19)-9)/32;
    for(size_t i=0;i<inject.size();++i) inject[i]=float(int(i%23)-11)*8;
    tensor input{&x,{uint64_t(h),uint64_t(r),uint64_t(t)}},gates{&gate,input.dims},normalized{&xn,input.dims};
    // Reverse partitions catch accidentally treating token as stream/row, or
    // broadcasting one output across windows. Every output starts as NaN.
    for(int token=t;token-->0;) {
        offset={token*r,0,0,0,0};extent={r,1,1,1,1};
        row_norm_window_kernel({&x,{uint64_t(h),uint64_t(t*r)}},{&gamma,{uint64_t(h),uint64_t(r)}},
            {&xn,{uint64_t(h),uint64_t(t*r)}},h,t*r,-r,1e-6f);
    }
    for(int token=t;token-->0;) for(int tile=0;tile<h/64;++tile) {
        offset={tile,token,0,0,0};extent={1,1,1,1,1};
        hc_mix_window_kernel(normalized,gates,{&mixed,{uint64_t(h),uint64_t(t)}},h,r,t);
        for(int stream=r;stream-->0;) {
            offset={tile,stream,token,0,0};extent={1,1,1,1,1};
            hc_write_window_kernel(input,{&block,{uint64_t(h),uint64_t(t)}},{&inject,{uint64_t(r),uint64_t(t)}},
                {&output,input.dims},h,r,t);
        }
    }
    for(int token=0;token<t;++token) {
        for(int stream=0;stream<r;++stream) {
            double sum=0;for(int d=0;d<h;++d) {const auto value=x[size_t(token)*n+stream*h+d];sum+=double(value)*value;}
            const auto inverse=1/std::sqrt(sum/h+1e-6);
            for(int d=0;d<h;++d) {
                const auto index=size_t(token)*n+stream*h+d;
                close(xn[index],double(x[index])*inverse*gamma[stream*h+d],5e-6,"repeated stream gamma RMS");
                const auto weight=2/(1+std::exp(-double(inject[token*r+stream])/r));
                close(output[index],double(x[index])+block[token*h+d]*weight,5e-6,"token-specific HC write");
            }
        }
        for(int d=0;d<h;++d) {
            double sum=0;for(int stream=0;stream<r;++stream) {
                const auto index=size_t(token)*n+stream*h+d;sum+=xn[index]/(1+std::exp(-double(gate[index])));
            }
            close(mixed[token*h+d],sum/r,5e-6,"token-specific HC mix");
        }
    }
    std::cout<<"PASS actual TPC C semantics HC window H="<<h<<" streams="<<r<<" rows="<<t<<" amplitude="<<amplitude<<'\n';
}
}
int main() {
    try {run(64,1,1);run(64,3,5);run(2560,4,8);run(64,32,128);run(2560,4,3,1e24f);}
    catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
}
