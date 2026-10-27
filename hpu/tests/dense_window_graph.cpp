#include "strata/hpu/dense_window_flow.hpp"
#include "strata/hpu/qsa.hpp"
#include "strata/hpu/session_flow.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "graph_interpreter.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
using namespace strata::hpu;
namespace {
template<class F> void rejects(F call) {bool failed=false;try {call();} catch(const std::exception&) {failed=true;}oracle::require(failed,"invalid dense transaction accepted");}
QsaConfig config(uint32_t capacity=37) {QsaConfig c;c.query_heads=6;c.kv_heads=2;c.width=4;c.rotary=4;c.index_heads=1;c.index_width=4;c.block=3;c.top_k=capacity;c.capacity=capacity;c.position_base=11;c.validate();return c;}
std::vector<float> gamma{.8f,1.1f,.9f,1.2f};
DenseProjected project(std::span<const float> x) {
    const auto rows=x.size()/4;DenseProjected p;p.q_gate.resize(rows*48);p.key.resize(rows*8);p.value.resize(rows*8);
    for(size_t row=0;row<rows;++row) {
        for(size_t i=0;i<48;++i) p.q_gate[row*48+i]=x[row*4+i%4]*float(i%7+1)/19+float(i%3)/9;
        for(size_t i=0;i<8;++i) {p.key[row*8+i]=x[row*4+i%4]*float(i%3+1)/7;p.value[row*8+i]=x[row*4+(i+1)%4]*float(i%5+1)/13;}
    }
    return p;
}
std::vector<float> graph_attention(const QsaConfig& c,const QsaWindowBatch& b) {
    GraphBuilder builder(OfflineCompilation{});define_qsa_attention_window(builder,c,b.bucket,b.rows);auto& g=*oracle::current;
    g.set("query",b.query);g.set("q_gate",b.gate);g.set("key",b.key);g.set("value",b.value);g.set("mask",b.mask);
    const std::vector<float> scale{1/std::sqrt(float(c.width))};g.set("scale",scale);g.run();return g.get("out").values;
}
std::vector<float> output(std::span<const float> x) {
    std::vector<float> out(x.size()/6);for(size_t row=0;row<out.size()/4;++row) for(size_t d=0;d<4;++d) for(size_t h=0;h<6;++h) out[row*4+d]+=x[row*24+h*4+d]/6;return out;
}
// Independent dense causal softmax, FP16 cache storage, gamma RMS and RoPE.
std::vector<float> reference(const QsaConfig& c,DenseKvState& kv,std::span<const float> x,uint32_t window) {
    const auto p=project(x);std::vector<float> query(24),gate(24),key=p.key;
    for(size_t h=0;h<6;++h) for(size_t d=0;d<4;++d) {query[h*4+d]=p.q_gate[h*8+d];gate[h*4+d]=p.q_gate[h*8+4+d];}
    for(auto* a:{&query,&key}) for(size_t row=0;row<a->size()/4;++row) {
        double ss=0;for(size_t d=0;d<4;++d) ss+=double((*a)[row*4+d])*(*a)[row*4+d];const float inv=float(1/std::sqrt(ss/4+double(c.epsilon)));
        for(size_t d=0;d<4;++d) (*a)[row*4+d]=((*a)[row*4+d]*inv)*gamma[d];
        for(size_t d=0;d<2;++d) {
            const double theta=double(c.position_base+kv.size())*std::pow(c.rope.freq_base,-2.0*double(d)/4);const float cs=float(std::cos(theta)),sn=float(std::sin(theta));
            const float l=(*a)[row*4+d],r=(*a)[row*4+d+2];(*a)[row*4+d]=l*cs-r*sn;(*a)[row*4+d+2]=l*sn+r*cs;
        }
    }
    kv.append(key,p.value);const auto b=dense_attention_bucket(c.capacity,kv.size(),window);const auto selected=kv.gather(b,window);
    std::vector<float> attended(24);
    for(size_t h=0;h<6;++h) {
        std::vector<double> scores(selected.count);double maximum=-INFINITY;
        for(size_t cell=0;cell<selected.count;++cell) {double dot=0;for(size_t d=0;d<4;++d) dot+=double(query[h*4+d])*selected.key[((h/3)*b+cell)*4+d];scores[cell]=dot/2;maximum=std::max(maximum,scores[cell]);}
        double denominator=0;for(auto& s:scores) {s=std::exp(s-maximum);denominator+=s;}
        for(size_t d=0;d<4;++d) {double sum=0;for(size_t cell=0;cell<selected.count;++cell) sum+=scores[cell]/denominator*selected.value[((h/3)*b+cell)*4+d];attended[h*4+d]=float(sum/(1+std::exp(-double(gate[h*4+d]))));}
    }
    return output(attended);
}
void close(std::span<const float> a,std::span<const float> b) {oracle::require(a.size()==b.size(),"dense output shape mismatch");for(size_t i=0;i<a.size();++i) oracle::require(std::isfinite(a[i]) && std::abs(a[i]-b[i])<3e-5f,"dense window differs from independent causal FP16/softmax oracle");}
void run(uint32_t root,uint32_t rows,uint32_t window) {
    const auto c=config(149);DenseKvState state(c.capacity,2,4),scalar(c.capacity,2,4);
    std::vector<float> input(rows*4);for(size_t i=0;i<input.size();++i) input[i]=float(int(i*13%29)-14)/17;
    const std::vector<float> seed{.4f,-.3f,.2f,.8f};for(uint32_t i=0;i<root;++i) {reference(c,state,seed,window);reference(c,scalar,seed,window);}
    DenseWindowOperations op{project,[&](const auto& batch){return graph_attention(c,batch);},output};
    const auto actual=execute_dense_window(state,c,input,4,window,{gamma,gamma},op);std::vector<float> expected;
    for(uint32_t row=0;row<rows;++row) {const auto value=reference(c,scalar,std::span<const float>(input).subspan(row*4,4),window);expected.insert(expected.end(),value.begin(),value.end());}
    close(actual,expected);oracle::require(state.size()==root+rows && state.gather(state.size()).key==scalar.gather(scalar.size()).key,"dense private KV differs from singleton oracle");
    // Causal prefix gather must not reveal any of the later appended rows.
    for(uint32_t end=1;end<=state.size();++end) {
        const auto prefix=state.gather_prefix(end,state.size(),window);oracle::require(prefix.count==(window?std::min(end,window):end),"prefix gather count includes future rows");
        for(uint32_t cell=prefix.count;cell<state.size();++cell) oracle::require(prefix.mask[cell]==-INFINITY,"future causal cell unmasked");
    }
}
void failures() {
    const auto c=config();DenseKvState state(c.capacity,2,4);const std::vector<float> seed{.4f,-.3f,.2f,.8f};reference(c,state,seed,3);
    const auto original=state.gather(8);const auto start=state.checkpoint();std::vector<float> input(17*4,.5f);
    auto unchanged=[&]{oracle::require(state.size()==1 && state.gather(8).key==original.key && state.gather(8).value==original.value,"failed dense window retained speculative KV");};
    for(int bad=0;bad<5;++bad) {
        uint32_t calls=0;DenseWindowOperations op;
        op.project=[&](auto x){auto p=project(x);if(++calls==2) {if(bad==0) p.q_gate.pop_back();if(bad==1) p.value[0]=65536;if(bad==2) p.key[0]=NAN;}return p;};
        op.attention=[&](const auto& b){if(bad==3 && calls==2) throw std::runtime_error("attention failed after KV append");return graph_attention(c,b);};
        op.output=[&](auto x){auto result=output(x);if(bad==4 && calls==2) result[0]=INFINITY;return result;};
        rejects([&]{(void)execute_dense_window(state,c,input,4,3,{gamma,gamma},op);});unchanged();
    }
    DenseWindowOperations op{project,[&](const auto& b){return graph_attention(c,b);},output};
    uint32_t polls=0;(void)execute_dense_window(state,c,input,4,3,{gamma,gamma},op,[&]{++polls;return false;});state.restore(start);
    for(uint32_t at=1;at<=polls;++at) {uint32_t n=0;rejects([&]{(void)execute_dense_window(state,c,input,4,3,{gamma,gamma},op,[&]{return ++n==at;});});unchanged();}
    rejects([&]{(void)state.gather_prefix(0,8);});rejects([&]{(void)state.gather_prefix(2,8);});
    auto bad=c;bad.top_k=1;rejects([&]{(void)execute_dense_window(state,bad,input,4,3,{gamma,gamma},op);});unchanged();
    std::cout<<"PASS dense failure/FP16 overflow/shape/nonfinite guards and "<<polls<<" cancellation boundaries, prefix guards and nested rewind\n";
}
}
int main() {try {
    uint32_t cases=0;for(uint32_t root:{0u,3u,8u}) for(uint32_t rows:{1u,3u,7u,8u,9u,17u,128u}) for(uint32_t window:{0u,1u,5u,19u}) {run(root,rows,window);++cases;}
    failures();std::cout<<"PASS actual dense MTP window/attention graph "<<cases<<" root/row/window cases vs independent causal FP16/RMS/RoPE/softmax oracle, group3, private KV and future masks; no device\n";
} catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
