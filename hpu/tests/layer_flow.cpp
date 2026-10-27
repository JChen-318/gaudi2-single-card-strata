#include "strata/hpu/layer_flow.hpp"
#include "strata/hpu/session_flow.hpp"
#include <iostream>
#include <limits>
using namespace strata::hpu;
namespace {
void require(bool ok,const char* what) {if(!ok) throw std::runtime_error(what);}
template<class F> void rejects(F&& f) {bool rejected=false;try {f();} catch(const std::exception&) {rejected=true;}require(rejected,"invalid layer accepted");}
struct Fixture {
    int state=10,rollbacks=0,fail=-1,stage=0;
    std::vector<std::string> trace;
    void visit(const char* name) {trace.push_back(name);if(stage++==fail) throw std::runtime_error("injected stage failure");}
    LayerOperations operations() {
        LayerOperations op;
        op.ple=[&](std::span<const float> x) {visit("ple");std::vector<float> r(x.begin(),x.end());for(auto& f:r) f+=10;return r;};
        op.read=[&](std::span<const float> x,bool ffn) {
            visit(ffn?"read_ffn":"read_attn");return HcReadResult{{(x[0]+x[2])/2,(x[1]+x[3])/2},ffn?std::vector<float>{2,3}:std::vector<float>{0,1}};
        };
        op.mixer=[&](std::span<const float> x) {visit("mixer");++state;return std::vector<float>{x[0]*2,x[1]*2};};
        op.moe=[&](std::span<const float> x) {visit("moe");return std::vector<float>{x[0]+5,x[1]+5};};
        op.write=[&](std::span<const float> x,std::span<const float> block,std::span<const float> inject,bool ffn) {
            visit(ffn?"write_ffn":"write_attn");std::vector<float> r(x.begin(),x.end());
            for(size_t c=0;c<2;++c) for(size_t d=0;d<2;++d) r[c*2+d]+=block[d]*(inject[c]+1);
            return r;
        };
        op.rollback_mixer=[&] {state=10;++rollbacks;};return op;
    }
};
}
int main() {
    try {
        const std::vector<float> input{1,2,3,4};
        Fixture good;const auto out=execute_layer(input,2,2,good.operations());
        require(out==std::vector<float>({194,209,273,294}),"full layer order or injection lifetime changed");
        require(good.trace==std::vector<std::string>({"ple","read_attn","mixer","write_attn","read_ffn","moe","write_ffn"}),"stage skipped/reordered");
        require(good.state==11 && good.rollbacks==0 && input==std::vector<float>({1,2,3,4}),"success commit or input mutation");
        for(int fail=0;fail<7;++fail) {
            Fixture f;f.fail=fail;auto operations=f.operations();rejects([&]{(void)execute_layer(input,2,2,operations);});
            require(f.state==10 && f.rollbacks==(fail>2?1:0),"mixer rollback stage boundary");require(f.trace.size()==size_t(fail+1),"execution continued after failure");
        }
        Fixture malformed;auto bad=malformed.operations();bad.mixer=[&](std::span<const float>) {++malformed.state;return std::vector<float>{};};
        rejects([&]{(void)execute_layer(input,2,2,bad);});require(malformed.state==10 && malformed.rollbacks==1,"invalid mixer export must roll back");
        Fixture nan;bad=nan.operations();bad.moe=[](std::span<const float>) {return std::vector<float>{0,std::numeric_limits<float>::quiet_NaN()};};
        rejects([&]{(void)execute_layer(input,2,2,bad);});require(nan.rollbacks==1,"non-finite downstream must restore mixer");
        Fixture rollback;bad=rollback.operations();bad.moe=[](std::span<const float>)->std::vector<float>{throw NonFiniteOutput("injected MoE export");};
        bad.rollback_mixer=[] {throw std::runtime_error("injected physical rewind failure");};bool fatal=false;
        try {(void)execute_layer(input,2,2,bad);} catch(const SessionFatal&) {fatal=true;}
        require(fatal,"failed physical mixer rollback was downgraded to a numerical failure");
        Fixture ordinary;auto no_ple=ordinary.operations();no_ple.ple={};(void)execute_layer(input,2,2,no_ple);require(ordinary.trace.front()=="read_attn","ordinary layer invoked PLE");
        Fixture missing;bad=missing.operations();bad.write={};rejects([&]{(void)execute_layer(input,2,2,bad);});require(missing.trace.empty(),"incomplete wiring reached callbacks");
        rejects([&]{(void)execute_layer(std::vector<float>{1},2,2,good.operations());});
        std::cout<<"PASS actual native layer flow: PLE before HC, GDN/QSA, both HC writes and MoE\n"
                   "PASS injection lifetime, no caller mutation, all seven stage failures, atomic mixer boundary and finite exports\n";
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
