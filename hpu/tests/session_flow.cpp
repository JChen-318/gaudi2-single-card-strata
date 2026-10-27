#include "strata/hpu/session_flow.hpp"
#include "strata/hpu/allocation_budget.hpp"
#include <array>
#include <atomic>
#include <cmath>
#include <iostream>
#include <numeric>
#include <thread>

using namespace strata::hpu;
namespace {
void require(bool ok,const char* message) {if(!ok) throw std::runtime_error(message);}
template<class F> void rejects(F&& f) {bool caught=false;try {f();} catch(const std::exception&) {caught=true;}require(caught,"expected failure");}
struct Fixture {
    std::array<uint32_t,48> positions{};int fail=-1;bool bad_head=false,bad_layer=false,fail_rewind=false,fatal_layer=false;
    size_t ple_previous=0,rollbacks=0;int fail_after=-1;bool bad_after=false,steer=false;
    SessionOperations operations() {
        SessionOperations op;
        op.embedding=[](int32_t token) {return std::vector<float>{float(token),float(token+1)};};
        op.ple=[&](int32_t token,std::span<const int32_t> previous) {ple_previous=previous.size();return std::vector<float>{float(token),float(previous.size())};};
        op.checkpoint=[&](uint32_t layer) {
            const auto saved=positions[layer];
            return std::function<void()>([&,layer,saved] {++rollbacks;if(fail_rewind && layer==2) throw std::runtime_error("rewind");positions[layer]=saved;});
        };
        op.layer=[&](uint32_t layer,std::span<const float> input,std::span<const float> ple) {
            require(input.size()==4,"layer input shape");require(layer==1?ple.size()==2:ple.empty(),"PLE layer placement");
            if(int(layer)==fail) {if(fatal_layer) throw SessionFatal("layer failed its own rollback");throw std::runtime_error("injected layer failure");}
            std::vector<float> out(input.begin(),input.end());
            for(auto& x:out) x+=float(layer+1+positions[layer])+(ple.empty()?0:ple[0]);
            ++positions[layer];if(bad_layer && layer==2) out[0]=NAN;
            return out;
        };
        op.head=[&](std::span<const float> input) {
            const auto sum=std::accumulate(input.begin(),input.end(),0.0f);std::vector<float> logits(5,sum);
            if(bad_head) logits[0]=INFINITY;
            return logits;
        };
        op.after_layer=[&](uint32_t layer,std::span<const float> input) {
            if(int(layer)==fail_after) throw std::runtime_error("injected post-FFN control failure");
            std::vector<float> out(input.begin(),input.end());
            if(steer && layer==47) for(auto& value:out) value+=1;
            if(bad_after && layer==47) out[0]=NAN;
            return out;
        };
        op.reset_layer=[&](uint32_t layer) {positions[layer]=0;};return op;
    }
    void all(uint32_t value) const {for(auto p:positions) require(p==value,"layer positions diverged");}
};
void chain_regressions() {
    Fixture f,reference;auto op=f.operations();op.after_layer={};
    const auto layer=op.layer;
    int omit=-1;bool bad_shape=false;
    op.chain=[&,layer](std::span<const float> input,std::span<const float> ple,const SessionOperations::ChainHooks& hooks) {
        std::vector<float> residual(input.begin(),input.end());
        for(uint32_t i=0;i<48;++i) {
            if(int(i)==omit) break;
            hooks.begin(i);residual=layer(i,residual,i==1?ple:std::span<const float>{});hooks.done(i);
        }
        if(bad_shape) residual.pop_back();
        return residual;
    };
    SessionFlow chain(2,2,48,5,128,std::move(op)),plain(2,2,48,5,128,reference.operations());
    require(chain.decode(2)==plain.decode(2),"chain versus scalar oracle");f.all(1);
    for(int failed=0;failed<48;++failed) {
        f.fail=failed;rejects([&]{(void)chain.decode(3);});f.all(1);
        require(chain.tokens().size()==1 && !chain.poisoned(),"chain failure committed or poisoned");
    }
    f.fail=-1;
    for(int boundary=1;boundary<=51;++boundary) {
        int polls=0;bool cancelled=false;
        try {(void)chain.decode(3,[&]{return ++polls==boundary;});} catch(const Cancelled&) {cancelled=true;}
        require(cancelled,"chain cancellation boundary missed");f.all(1);
        require(chain.tokens().size()==1 && !chain.poisoned(),"chain cancellation committed");
    }
    f.bad_head=true;bool numerical=false;
    try {(void)chain.decode(3);} catch(const NonFiniteOutput&) {numerical=true;}
    require(numerical && !chain.poisoned(),"chain nonfinite head type/recovery");f.all(1);f.bad_head=false;
    f.bad_layer=true;numerical=false;
    try {(void)chain.decode(3);} catch(const NonFiniteOutput&) {numerical=true;}
    require(numerical,"chain final residual nonfinite not rejected");f.all(1);f.bad_layer=false;
    for(int missing:{0,1,47}) {omit=missing;rejects([&]{(void)chain.decode(3);});f.all(1);}
    omit=-1;bad_shape=true;rejects([&]{(void)chain.decode(3);});f.all(1);bad_shape=false;
    require(chain.decode(3)==plain.decode(3),"chain retry after failures changed state/logits");f.all(2);
    f.fail=7;f.fail_rewind=true;bool fatal=false;
    try {(void)chain.decode(4);} catch(const SessionFatal&) {fatal=true;}
    require(fatal && chain.poisoned(),"chain rollback failure must poison");rejects([&]{(void)chain.decode(4);});
    f.fail=-1;f.fail_rewind=false;chain.reset();f.all(0);
    f.fail=3;f.fatal_layer=true;fatal=false;
    try {(void)chain.decode(2);} catch(const SessionFatal&) {fatal=true;}
    require(fatal && chain.poisoned(),"chain layer fatal type lost");
    f.fail=-1;f.fatal_layer=false;chain.reset();require(chain.decode(2)==std::vector<float>(5,4722),"chain reset recovery");
    Fixture gated;auto fallback=gated.operations();
    fallback.chain=[](auto,auto,const auto&)->std::vector<float>{throw std::runtime_error("chain must be bypassed for after_layer");};
    SessionFlow controlled(2,2,48,5,8,std::move(fallback));require(controlled.decode(2)[0]==4722,"after_layer did not bypass chain");
    std::cout<<"PASS chain oracle,48 failure sites,51 cancel boundaries,nonfinite/shape/skipped layers,retry,poison/reset,control fallback\n";
}
}
int main() {
    try {
        chain_regressions();
        Fixture f;SessionFlow flow(2,2,48,5,8,f.operations());
        auto logits=flow.decode(2);require(logits.size()==5 && logits[0]==4722,"embedding broadcast/layer/head oracle");f.all(1);
        require(flow.tokens().size()==1 && flow.tokens()[0]==2 && f.ple_previous==0,"first token commit");
        for(int fail=0;fail<48;++fail) {
            f.fail=fail;rejects([&]{(void)flow.decode(3);});f.all(1);require(flow.tokens().size()==1,"failed token committed");
        }
        f.fail=-1;f.bad_head=true;bool numerical=false;try {(void)flow.decode(3);} catch(const NonFiniteOutput&) {numerical=true;}
        require(numerical && !flow.poisoned(),"head nonfinite did not retain typed failure after successful rollback");f.all(1);f.bad_head=false;
        for(int fail=0;fail<48;++fail) {f.fail_after=fail;rejects([&]{(void)flow.decode(3);});f.all(1);}
        f.fail_after=-1;f.bad_after=true;rejects([&]{(void)flow.decode(3);});f.all(1);f.bad_after=false;
        f.bad_layer=true;rejects([&]{(void)flow.decode(3);});f.all(1);f.bad_layer=false;
        // Cancel before any work, during the trunk, and after head completion.
        for(int boundary:{1,6,51}) {
            int polls=0;rejects([&]{(void)flow.decode(3,[&]{return ++polls==boundary;});});f.all(1);
        }
        const std::array<int32_t,2> prompt{3,4};(void)flow.prefill(prompt);f.all(3);
        require(flow.tokens().size()==3 && f.ple_previous==2,"prefill history continuity");
        const std::array<int32_t,2> invalid{1,5};rejects([&]{(void)flow.prefill(invalid);});f.all(3);
        rejects([&]{(void)flow.decode(-1);});rejects([&]{(void)flow.decode(5);});
        f.fail=7;f.fail_rewind=true;bool fatal=false;try {(void)flow.decode(1);} catch(const SessionFatal&) {fatal=true;}
        require(fatal && flow.poisoned(),"rollback failure did not poison session with fatal type");
        rejects([&]{(void)flow.decode(1);});f.fail=-1;f.fail_rewind=false;flow.reset();f.all(0);require(flow.tokens().empty() && !flow.poisoned(),"reset recovery");
        f.steer=true;require(flow.decode(2)[0]==4726,"control vector hook not after FFN and before final head");flow.reset();f.steer=false;
        f.fail=3;f.fatal_layer=true;fatal=false;try {(void)flow.decode(2);} catch(const SessionFatal&) {fatal=true;}
        require(fatal && flow.poisoned(),"fatal throwing-layer state was lost after earlier layers rewound");f.fail=-1;f.fatal_layer=false;flow.reset();
        SessionOperations empty;rejects([&]{SessionFlow invalid_flow(2,2,48,5,8,empty);});
        AllocationBudget budget;budget.configure(8);budget.charge(5);rejects([&]{budget.charge(4);});require(budget.stats().used==5,"rejected allocation mutated ledger");
        rejects([&]{budget.configure(12);});budget.release(5);
        std::atomic<bool> failed=false;std::array<std::thread,4> workers;
        for(auto& thread:workers) thread=std::thread([&] {for(int i=0;i<1000;++i) {try {budget.charge(1);budget.release(1);} catch(...) {failed=true;}}});
        for(auto& thread:workers) thread.join();
        require(!failed && budget.stats().used==0 && budget.stats().peak<=8,"concurrent budget ledger");
        budget.configure(UINT64_MAX);budget.charge(UINT64_MAX);rejects([&]{budget.charge(1);});budget.release(UINT64_MAX);
        std::cout<<"PASS actual 48-layer session orchestration, independent logits oracle, failure at every layer, cancellation, prefill, reset\n"
                   "PASS global allocation cap, rejected-allocation atomicity, concurrent accounting and overflow guard\n";
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
