#include "strata/hpu/ple_window_flow.hpp"
#include "strata/hpu/layer_flow.hpp"
#include "strata/hpu/session_flow.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <numeric>
using namespace strata::hpu;
namespace {
void require(bool ok,const char* message) {if(!ok) throw std::runtime_error(message);}
template<class F> void rejects(F call) {bool failed=false;try {call();} catch(const std::exception&) {failed=true;}require(failed,"invalid PLE window accepted");}
struct Fixture {
    int calls=0,fail=-1,bad=-1,kind=0;
    std::vector<size_t> batches;
    PleWindowResult step(std::span<const float> x,std::span<const float> embedding,std::span<const float> previous) {
        const auto call=calls++;if(call==fail) throw std::runtime_error("injected PLE native subwindow failure");
        batches.push_back(x.size()/4);PleWindowResult out;out.history.assign(previous.begin(),previous.end());out.output.resize(x.size());
        for(size_t row=0;row<x.size()/4;++row) {
            std::array<float,4> current{};
            for(size_t i=0;i<4;++i) {
                current[i]=x[row*4+i]+embedding[row*2+i%2]/4;
                const auto sum=out.history[i]/8+out.history[3*4+i]/4+out.history[6*4+i]/2+current[i];
                out.output[row*4+i]=(x[row*4+i]+current[i])+sum/(1+std::exp(-sum));
            }
            std::move(out.history.begin()+4,out.history.end(),out.history.begin());std::copy(current.begin(),current.end(),out.history.end()-4);
        }
        if(call==bad) {
            if(kind==0) out.output.pop_back();
            if(kind==1) out.history.pop_back();
            if(kind==2) out.output[0]=NAN;
            if(kind==3) out.history[0]=INFINITY;
        }
        return out;
    }
    PleWindowRunner runner() {return [this](auto x,auto e,auto history) {return step(x,e,history);};}
};
std::vector<float> history() {std::vector<float> h(36);for(size_t i=0;i<h.size();++i) h[i]=float(int(i%17)-8)/32;return h;}
std::vector<float> rows(size_t count,size_t width) {std::vector<float> x(count*width);for(size_t i=0;i<x.size();++i) x[i]=float(int(i*11%31)-15)/32;return x;}
void transactions() {
    const auto accepted=history();
    for(size_t count:{1u,2u,8u,9u,17u,128u}) {
        const auto x=rows(count,4),embedding=rows(count,2);Fixture one,batch;
        const auto expected=one.step(x,embedding,accepted),actual=execute_ple_window(x,embedding,accepted,2,2,batch.runner());
        require(actual.output==expected.output && actual.history==expected.history,"PLE causal dilation/history changed at chunk boundary");
        require(batch.batches.size()==(count+7)/8 && std::all_of(batch.batches.begin(),batch.batches.end(),[](auto n) {return n>=1 && n<=8;}),"unbounded PLE native graph");
        for(size_t at=0;at<batch.batches.size();++at) for(int kind=-1;kind<4;++kind) {
            Fixture broken;if(kind<0) broken.fail=int(at);else {broken.bad=int(at);broken.kind=kind;}
            rejects([&] {(void)execute_ple_window(x,embedding,accepted,2,2,broken.runner());});require(accepted==history(),"failed PLE window mutated accepted history");
        }
        for(size_t boundary=1;boundary<=2*batch.batches.size()+2;++boundary) {
            Fixture broken;size_t polls=0;rejects([&] {(void)execute_ple_window(x,embedding,accepted,2,2,broken.runner(),[&] {return ++polls==boundary;});});
            require(accepted==history(),"cancelled PLE window mutated accepted history");
        }
    }
    Fixture invalid;const auto x=rows(2,4),embedding=rows(2,2);auto bad=accepted;bad[0]=NAN;
    rejects([&] {(void)execute_ple_window({},embedding,accepted,2,2,invalid.runner());});
    rejects([&] {(void)execute_ple_window(x,{},accepted,2,2,invalid.runner());});
    rejects([&] {(void)execute_ple_window(x,embedding,bad,2,2,invalid.runner());});
    rejects([&] {(void)execute_ple_window(rows(129,4),rows(129,2),accepted,2,2,invalid.runner());});require(!invalid.calls,"invalid PLE input reached native runner");
    require(ple_window_pinned_bytes(2560,4,8)==1433600,"target PLE transfer-slot byte contract changed");
}
void layer() {
    const auto input=rows(17,4),embedding=rows(17,2),root=history();auto accepted=root,pending=root;int position=3,rollbacks=0,ple_calls=0;bool cancelled=false;
    int failure=-1;LayerOperations op;
    op.ple=[](auto)->std::vector<float> {throw std::runtime_error("batched PLE called row fallback");};
    op.read=[](auto,bool)->HcReadResult {throw std::runtime_error("batched PLE called row HC");};
    op.write=[](auto,auto,auto,bool)->std::vector<float> {throw std::runtime_error("batched PLE called row write");};
    op.mixer=[](auto)->std::vector<float> {throw std::runtime_error("batched PLE called row mixer");};
    op.window_ple=[&](auto x) {
        ++ple_calls;Fixture f;f.fail=failure==0?1:-1;auto next=execute_ple_window(x,embedding,pending,2,2,f.runner());pending=std::move(next.history);
        if(failure==5) next.output[0]=NAN;
        return next.output;
    };
    op.window_read=[&](std::span<const float> x,bool ffn) {
        if(failure==1 && !ffn) throw std::runtime_error("injected first HC failure");
        HcReadResult out;for(size_t row=0;row<x.size()/4;++row) for(size_t i=0;i<2;++i) {out.mixed.push_back(x[row*4+i]);out.inject.push_back(1);}return out;
    };
    op.window_mixer=[&](std::span<const float> x) {position+=int(x.size()/2);return std::vector<float>(x.begin(),x.end());};
    op.window_write=[&](std::span<const float> x,auto,auto,bool ffn) {
        if(failure==4 && ffn) throw std::runtime_error("injected final HC write failure");
        return std::vector<float>(x.begin(),x.end());
    };
    op.moe=[&](std::span<const float> x) {
        if(failure==2) throw std::runtime_error("injected MoE failure");
        if(failure==3) cancelled=true;
        return std::vector<float>(x.begin(),x.end());
    };
    op.rollback_mixer=[&] {position=3;++rollbacks;};
    auto execute=[&] {pending=accepted;auto out=execute_layer_window(input,2,2,op,[&] {return cancelled;});accepted=std::move(pending);return out;};
    Fixture reference;const auto expected=reference.step(input,embedding,root);require(execute()==expected.output && accepted==expected.history && position==20 && ple_calls==1,"full layer PLE window commit/order mismatch");
    for(failure=0;failure<6;++failure) {
        accepted=root;position=3;cancelled=false;rollbacks=0;rejects(execute);
        require(accepted==root && position==3 && rollbacks==((failure>=2 && failure<=4)?1:0),"failed layer published staged PLE history or missed mixer rollback");
    }
    auto missing=op;missing.window_read={};missing.window_write={};missing.window_mixer={};bool guarded=false;
    try {(void)execute_layer_window(input,2,2,missing);} catch(const std::invalid_argument&) {guarded=true;}
    require(guarded,"window PLE was silently dropped without window HC");
}
}
int main() {
    try {transactions();layer();std::cout<<"PASS production PLE windows:1..128rows cap8, causal dilation3/history9 across chunks, every failure/export/cancel, staged whole-layer history and mixer rollback; target_pinned_bytes=1433600\n";}
    catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
