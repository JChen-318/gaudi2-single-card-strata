#pragma once
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace gdn_test {
inline float sigmoid(float value) {
    if(value>=0) return 1/(1+std::exp(-value));
    const float e=std::exp(value); return e/(1+e);
}
struct Fixture {
    int s,hk,hv,d,c;
    float eps=1e-6f;
    std::vector<float> state,conv,weights,qkv,alpha,raw_beta,z,norm,dt,a;
    std::vector<float> activated,q,k,v,gate,beta,recurrent,out;
    Fixture(int width,int keys,int values,int taps=4)
        : s(width),hk(keys),hv(values),d(taps),c(s*(2*hk+hv)),state(s*hv*s),conv(c*(d-1)),weights(c*d),
          qkv(c),alpha(hv),raw_beta(hv),z(hv*s),norm(s),dt(hv),a(hv),
          activated(c),q(hk*s),k(hk*s),v(hv*s),gate(hv),beta(hv),recurrent(hv*s),out(hv*s) {
        for(size_t i=0;i<state.size();++i) state[i]=float(int((i*13+3)%41)-20)/512;
        for(int channel=0;channel<c;++channel) {
            for(int tap=0;tap<d-1;++tap) conv[channel*(d-1)+tap]=float((channel*7+tap*11)%31-15)/64;
            for(int tap=0;tap<d;++tap) weights[channel*d+tap]=float((channel*3+tap*7)%19-9)/32;
        }
        // One zero Q row and one zero K row pin the epsilon floor.
        if(hk>1) {
            for(int channel=0;channel<s;++channel) for(int tap=0;tap<d;++tap) weights[channel*d+tap]=0;
            for(int channel=hk*s;channel<(hk+1)*s;++channel) for(int tap=0;tap<d;++tap) weights[channel*d+tap]=0;
        }
        for(int col=0;col<s;++col) norm[col]=0.5f+float(col%13)/16;
        for(int head=0;head<hv;++head) { dt[head]=float(head%7-3)/4; a[head]=-0.05f*float(head%5); }
        if(hv>1) { dt[0]=-100; dt[1]=100; }
    }
    void inputs(int phase) {
        for(int channel=0;channel<c;++channel) qkv[channel]=float((channel*17+phase*13)%47-23)/32;
        for(int head=0;head<hv;++head) {
            alpha[head]=float((head*11+phase*3)%29-14)/4;
            raw_beta[head]=head%3==0?-100:head%3==1?100:float(head%9-4);
        }
        for(int i=0;i<hv*s;++i) z[i]=float((i*19+phase*7)%43-21)/4;
    }
    std::vector<float> time_major_conv() const {
        std::vector<float> packed(conv.size());
        for(int channel=0;channel<c;++channel) for(int tap=0;tap<d-1;++tap)
            packed[tap*c+channel]=conv[channel*(d-1)+tap];
        return packed;
    }
    std::vector<float> time_major_weights() const {
        std::vector<float> packed(weights.size());
        for(int channel=0;channel<c;++channel) for(int tap=0;tap<d;++tap) packed[tap*c+channel]=weights[channel*d+tap];
        return packed;
    }
    void reference_step() {
        // Independent oracle follows upstream channel-major conv storage and
        // FP64 norm accumulation, rather than the HPU execution layout/math.
        for(int channel=0;channel<c;++channel) {
            float sum=0;
            for(int tap=0;tap<d-1;++tap) sum+=conv[channel*(d-1)+tap]*weights[channel*d+tap];
            sum+=qkv[channel]*weights[channel*d+d-1];
            activated[channel]=sum*sigmoid(sum);
            for(int tap=0;tap<d-2;++tap) conv[channel*(d-1)+tap]=conv[channel*(d-1)+tap+1];
            conv[channel*(d-1)+d-2]=qkv[channel];
        }
        for(int head=0;head<hk;++head) {
            double qs=0,ks=0;
            for(int i=0;i<s;++i) {
                const double qvalue=activated[head*s+i],kvalue=activated[hk*s+head*s+i];
                qs+=qvalue*qvalue; ks+=kvalue*kvalue;
            }
            const float qi=float(1/std::sqrt(qs+double(eps))),ki=float(1/std::sqrt(ks+double(eps)));
            const float scale=1/std::sqrt(float(s));
            for(int i=0;i<s;++i) { q[head*s+i]=(activated[head*s+i]*qi)*scale; k[head*s+i]=activated[hk*s+head*s+i]*ki; }
        }
        std::copy(activated.begin()+2*hk*s,activated.end(),v.begin());
        for(int head=0;head<hv;++head) {
            const float sum=alpha[head]+dt[head];
            gate[head]=(sum>20?sum:std::log1p(std::exp(sum)))*a[head];
            beta[head]=sigmoid(raw_beta[head]);
            const float decay=std::exp(gate[head]);
            for(int j=0;j<s;++j) {
                float contraction=0;
                for(int i=0;i<s;++i) { auto& cell=state[(i*hv+head)*s+j]; cell*=decay; contraction+=cell*k[(head%hk)*s+i]; }
                const float delta=(v[head*s+j]-contraction)*beta[head];
                float answer=0;
                for(int i=0;i<s;++i) { auto& cell=state[(i*hv+head)*s+j]; cell+=k[(head%hk)*s+i]*delta; answer+=cell*q[(head%hk)*s+i]; }
                recurrent[head*s+j]=answer;
            }
            double sum2=0;
            for(int i=0;i<s;++i) { const double value=recurrent[head*s+i]; sum2+=value*value; }
            const float inv=float(1/std::sqrt(sum2/s+double(eps)));
            for(int i=0;i<s;++i) out[head*s+i]=((recurrent[head*s+i]*inv)*norm[i])*sigmoid(z[head*s+i]);
        }
    }
};
inline double compare(const std::vector<float>& actual,const std::vector<float>& expected,const char* name,double tolerance) {
    if(actual.size()!=expected.size()) throw std::runtime_error("GDN result shape mismatch");
    double error=0;
    for(size_t i=0;i<actual.size();++i) {
        if(!std::isfinite(actual[i])) throw std::runtime_error(std::string("non-finite GDN ")+name);
        error=std::max(error,double(std::abs(actual[i]-expected[i])));
    }
    if(error>tolerance) throw std::runtime_error(std::string("GDN parity ")+name+" max_abs_error="+std::to_string(error));
    return error;
}
}
