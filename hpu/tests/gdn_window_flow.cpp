#include "strata/hpu/gdn_window_flow.hpp"
#include "strata/hpu/layer_flow.hpp"
#include "strata/hpu/session_flow.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <numeric>
using namespace strata::hpu;
namespace {
void require(bool ok,const char* reason) {if(!ok) throw std::runtime_error(reason);}
template<class F> void rejects(F call) {bool failed=false;try {call();} catch(const std::exception&) {failed=true;}require(failed,"invalid/failed GDN window accepted");}
struct Fixture {
    std::vector<size_t> batches;int bad=-1,kind=0;
    GdnWindowResult step(std::span<const float> x,std::span<const float> s,std::span<const float> c) {
        const auto index=int(batches.size());batches.push_back(x.size()/2);
        if(index==bad && kind==0) throw std::runtime_error("injected GDN chunk failure");
        GdnWindowResult out;out.state.assign(s.begin(),s.end());out.conv.assign(c.begin(),c.end());
        for(size_t row=0;row<x.size()/2;++row) {
            out.state[0]=out.state[0]*.5f+x[row*2];out.conv[0]=out.conv[0]*.25f+x[row*2+1];
            out.output.push_back(out.state[0]+out.conv[0]);out.output.push_back(out.state[0]-out.conv[0]);
        }
        if(index==bad) {
            if(kind==1) out.output.pop_back();
            if(kind==2) out.state.pop_back();
            if(kind==3) out.conv.pop_back();
            if(kind==4) out.output.back()=NAN;
            if(kind==5) out.state[0]=INFINITY;
            if(kind==6) out.conv[0]=NAN;
        }
        return out;
    }
    GdnWindowRunner runner() {return [this](auto x,auto s,auto c) {return step(x,s,c);};}
};
void transactions() {
    const std::array<float,2> state{3,9},conv{5,11};
    for(size_t rows:{1u,2u,8u,9u,17u,128u}) {
        std::vector<float> x(rows*2);std::iota(x.begin(),x.end(),-8.0f);Fixture batch,single;
        const auto expected=single.step(x,state,conv),actual=execute_gdn_window(x,2,state,conv,batch.runner());
        require(actual.output==expected.output && actual.state==expected.state && actual.conv==expected.conv,"chunk boundary changed causal recurrence");
        require(batch.batches.size()==(rows+7)/8 && std::all_of(batch.batches.begin(),batch.batches.end(),[](auto n) {return n>=1 && n<=8;}),"GDN native graph rows are unbounded");
        for(size_t at=0;at<batch.batches.size();++at) for(int kind=0;kind<7;++kind) {
            Fixture broken;broken.bad=int(at);broken.kind=kind;
            rejects([&] {(void)execute_gdn_window(x,2,state,conv,broken.runner());});
            require(state==std::array<float,2>{3,9} && conv==std::array<float,2>{5,11},"failed chunk mutated accepted state");
        }
        for(size_t boundary=1;boundary<=batch.batches.size()*2+2;++boundary) {
            Fixture cancel;size_t polls=0;rejects([&] {(void)execute_gdn_window(x,2,state,conv,cancel.runner(),[&] {return ++polls==boundary;});});
            require(state[0]==3 && conv[0]==5,"cancelled chunk mutated accepted state");
        }
    }
    Fixture f;const auto runner=f.runner();
    rejects([&] {(void)execute_gdn_window({},2,state,conv,runner);});
    rejects([&] {(void)execute_gdn_window(std::vector<float>(258),2,state,conv,runner);});
    rejects([&] {(void)execute_gdn_window(std::vector<float>(3),2,state,conv,runner);});
    const std::array<float,2> bad{NAN,0};rejects([&] {(void)execute_gdn_window(bad,2,state,conv,runner);});
    require(f.batches.empty(),"invalid window reached runner");
}
void layer() {
    const std::vector<float> input(17*4,1);std::array<float,1> state{3},conv{5};
    const auto initial_state=state,initial_conv=conv;size_t mixer_calls=0,rollbacks=0;bool fail_after=false,bad_export=false,fail_mixer=false;
    LayerOperations op;
    op.read=[](auto,bool)->HcReadResult {throw std::runtime_error("singleton HC used");};
    op.write=[](auto,auto,auto,bool)->std::vector<float> {throw std::runtime_error("singleton HC write used");};
    op.mixer=[](auto)->std::vector<float> {throw std::runtime_error("singleton mixer used");};
    op.window_read=[](std::span<const float> x,bool) {return HcReadResult{std::vector<float>(x.size()/2,2),std::vector<float>(x.size()/2,1)};};
    op.window_write=[&](std::span<const float> x,auto,auto,bool ffn) {
        if(fail_after && ffn) throw std::runtime_error("injected later HC failure");
        return std::vector<float>(x.begin(),x.end());
    };
    op.moe=[](std::span<const float> x) {return std::vector<float>(x.begin(),x.end());};
    op.window_mixer=[&](auto x) {
        ++mixer_calls;Fixture f;f.bad=fail_mixer?1:-1;
        auto staged=execute_gdn_window(x,2,state,conv,f.runner());state[0]=staged.state[0];conv[0]=staged.conv[0];
        if(bad_export) staged.output[0]=NAN;
        return staged.output;
    };
    op.rollback_mixer=[&] {state=initial_state;conv=initial_conv;++rollbacks;};
    require(execute_layer_window(input,2,2,op)==input && mixer_calls==1 && rollbacks==0,"native window mixer was not called once");
    state=initial_state;conv=initial_conv;fail_after=true;
    rejects([&] {(void)execute_layer_window(input,2,2,op);});require(state==initial_state && conv==initial_conv && rollbacks==1,"post-mixer stage did not rewind whole window");
    fail_after=false;bad_export=true;
    rejects([&] {(void)execute_layer_window(input,2,2,op);});require(rollbacks==2 && state==initial_state,"bad mixer export did not rewind");
    bad_export=false;fail_mixer=true;
    rejects([&] {(void)execute_layer_window(input,2,2,op);});require(rollbacks==2 && state==initial_state,"atomic failed mixer was redundantly restored");
    op.window_read={};op.window_write={};rejects([&] {(void)execute_layer_window(input,2,2,op);});
}
}
int main() {
    try {transactions();layer();std::cout<<"PASS actual GDN chunk transaction and layer window dispatch:1..128rows, cap8, causal continuity, every chunk failure/export/cancel, later HC rollback\n";}
    catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
