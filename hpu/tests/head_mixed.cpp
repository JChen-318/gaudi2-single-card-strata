#include "strata/hpu/head_mixed.hpp"
#include "strata/hpu/session_flow.hpp"
#include <array>
#include <iostream>
#include <limits>

using namespace strata::hpu;
namespace {
void require(bool ok,const char* message) {if(!ok) throw std::runtime_error(message);}
template<class F> void rejects(F&& f) {bool caught=false;try {f();} catch(const std::exception&) {caught=true;}require(caught,"invalid mixed transaction accepted");}
std::vector<float> row(int32_t token) {return {float(token),-float(token)};}
std::vector<float> values(const HeadMixed& output) {std::vector<float> copy(output.width());std::copy(output.current().begin(),output.current().end(),copy.begin());return copy;}
struct Fixture {
    HeadMixed output{2};
    uint32_t cells=0;
    bool staged=false,fail_head=false;
    SessionOperations operations() {
        SessionOperations op;
        op.embedding=[](int32_t token){return row(token);};
        op.ple=[](int32_t,std::span<const int32_t>){return std::vector<float>(2);};
        op.checkpoint=[this](uint32_t){return [this,old=cells]{cells=old;};};
        op.layer=[this](uint32_t,std::span<const float> input,std::span<const float>){++cells;return std::vector<float>(input.begin(),input.end());};
        op.window_layer=[this](uint32_t,std::span<const float> input,std::span<const float>,std::span<const float>,const auto&){cells+=uint32_t(input.size()/2);return std::vector<float>(input.begin(),input.end());};
        op.head=[this](std::span<const float> input){output.stage(input,1);staged=true;if(fail_head) throw std::runtime_error("injected replay head failure");return std::vector<float>(16,input[0]);};
        op.window_head=[this](std::span<const float> input,const auto&){output.stage(input,input.size()/2);staged=true;std::vector<float> logits;for(size_t i=0;i<input.size();i+=2) logits.insert(logits.end(),16,input[i]);return logits;};
        op.reset_layer=[this](uint32_t){cells=0;};return op;
    }
};
void transactions() {
    const std::array<int32_t,3> candidates{4,5,6};
    for(uint32_t accepted=0;accepted<=3;++accepted) {
        Fixture f;SessionFlow flow(2,1,1,16,64,f.operations());
        flow.decode(1);f.output.commit();require(values(f.output)==row(1),"singleton head publication");
        const auto result=flow.verify_window(candidates,4096,[&](auto){return accepted;});
        if(result.accepted) f.output.commit();
        require(values(f.output)==row(accepted?candidates[accepted-1]:1),"accepted-prefix head input differs from final retained token");
        require(f.cells==1+accepted && flow.tokens().size()==1+accepted,"candidate rollback/replay position mismatch");
        flow.prefill_window(std::array<int32_t,2>{7,8});f.output.commit();
        require(values(f.output)==row(8),"prefill did not publish its last row");
        const auto previous=values(f.output);const auto extent=f.cells;
        f.staged=false;
        rejects([&]{(void)flow.verify_window(candidates,4096,[](auto){return 3u;},[&]{return f.staged;});});
        require(f.staged && values(f.output)==previous && f.cells==extent && flow.tokens().size()==extent,"cancel after candidate head exposed staged output");
        rejects([&]{(void)flow.verify_window(candidates,4096,[](auto)->uint32_t{throw std::runtime_error("selector failure");});});
        require(values(f.output)==previous && f.cells==extent,"throwing selector exposed staged output");
        f.fail_head=true;
        rejects([&]{(void)flow.verify_window(candidates,4096,[](auto){return 1u;});});
        require(values(f.output)==previous && f.cells==extent && flow.tokens().size()==extent,"failed prefix replay exposed staged output");
    }
}
void parking() {
    HeadMixed output(2);std::vector<float> a,b,empty;
    output.stage(row(1),1);output.commit();output.capture(a,true);
    output.stage(row(9),1);output.commit();output.capture(b,true);
    output.restore(a,true);require(values(output)==row(1),"conversation A restored another conversation's head");
    output.restore(b,true);require(values(output)==row(9),"conversation B restored another conversation's head");
    output.capture(a,true);require(a==row(9),"retained image did not overwrite its head row");
    output.capture(empty,false);require(empty.empty(),"empty conversation captured a valid head");
    output.restore(empty,false);require(values(output)==row(0),"empty restore retained published values");
    output.restore(a,true);const auto prior=values(output);
    for(const auto& bad: {std::vector<float>{1},std::vector<float>{1,2,3},std::vector<float>{NAN,0},std::vector<float>{0,INFINITY}})
        rejects([&]{output.restore(bad,true);});
    rejects([&]{output.restore(a,false);});
    require(values(output)==prior,"invalid image partially published");
    rejects([&]{output.stage(row(2),2);});rejects([&]{output.stage({},0);});
    rejects([&]{output.stage(std::array<float,4>{NAN,0,3,4},2);});
    rejects([&]{output.stage(row(2),std::numeric_limits<size_t>::max());});
    output.commit(); // Invalid staging must leave the previous pending row intact.
    require(values(output)==row(9),"invalid staging mutated the pending row");
}
}
int main() {
    try {transactions();parking();std::cout<<"PASS production mixed staging/publication, native SessionFlow full/partial/zero acceptance, replay and cancellation\nPASS distinct conversation restore, retained capture, empty image and malformed/non-finite refusal\n";}
    catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
