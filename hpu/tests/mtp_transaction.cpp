#include "strata/hpu/dense_kv.hpp"
#include "strata/hpu/session_flow.hpp"
#include <array>
#include <iostream>
#include <numeric>
using namespace strata::hpu;
namespace {
void require(bool ok,const char* message) {if(!ok) throw std::runtime_error(message);}
template<class F> void rejects(F&& f) {bool caught=false;try {f();} catch(const std::exception&) {caught=true;}require(caught,"injected failure did not propagate");}
}
int main() {
    try {
        DenseKvState mtp(8,1,1);std::array<uint32_t,48> positions{};
        SessionFlow* active=nullptr;float previous=0,pending=0;int32_t next_token=0;bool bad_head=false,bad_mtp=false;
        SessionOperations op;
        op.embedding=[&](int32_t token){next_token=token;return std::vector<float>{float(token)};};
        op.ple=[](auto,auto){return std::vector<float>{0};};
        op.checkpoint=[&](uint32_t layer) {
            if(layer==48) {auto point=mtp.checkpoint();return std::function<void()>([&,point]{mtp.restore(point);});}
            const auto point=positions[layer];return std::function<void()>([&,layer,point]{positions[layer]=point;});
        };
        op.layer=[&](uint32_t layer,std::span<const float> residual,auto) {
            std::vector<float> out(residual.begin(),residual.end());
            if(layer<48) {++positions[layer];out[0]+=1;return out;}
            if(!active->tokens().empty()) {
                const auto point=mtp.checkpoint();
                try {
                    const std::array<float,1> key{previous},value{float(next_token)};mtp.append(key,value);
                    if(bad_mtp) throw std::runtime_error("failure after draft KV append");
                } catch(...) {mtp.restore(point);throw;}
            }
            return out;
        };
        op.head=[&](std::span<const float> residual){pending=residual[0];if(bad_head) throw std::runtime_error("target head failed");return std::vector<float>(16,pending);};
        op.reset_layer=[&](uint32_t layer){if(layer==48) mtp.reset();else positions[layer]=0;};
        SessionFlow flow(1,1,49,16,8,op);active=&flow;
        auto commit=[&](int32_t token,const std::function<bool()>& cancelled={}) {auto result=flow.decode(token,cancelled);previous=pending;return result;};
        (void)commit(1);require(previous==49 && !mtp.size(),"first token invented an MTP pair");
        (void)commit(2);auto kv=mtp.gather(1);require(kv.key[0]==49 && kv.value[0]==2 && previous==50,"MTP paired the new residual instead of previous R with current token");
        bad_head=true;rejects([&]{(void)commit(3);});bad_head=false;
        require(mtp.size()==1 && previous==50 && flow.tokens().size()==2,"head failure published MTP/target output");
        for(int boundary:{50,51,52}) {int polls=0;rejects([&]{(void)commit(3,[&]{return ++polls==boundary;});});require(mtp.size()==1 && previous==50,"auxiliary/post-head cancel leaked MTP state");}
        bad_mtp=true;rejects([&]{(void)commit(3);});bad_mtp=false;
        require(mtp.size()==1 && !flow.poisoned(),"atomic MTP-stage error broke target rollback");
        for(auto p:positions) require(p==2,"target/MTP positions diverged after cancellation");
        (void)commit(4);kv=mtp.gather(2);require(kv.key[1]==50 && kv.value[1]==4 && previous==52,"retry after rollback used speculative residual/token");
        // Nested proposal steps: an inner failure rewind must leave the outer
        // accepted-prefix checkpoint restorable without a whole-KV copy.
        const auto outer=mtp.checkpoint();mtp.append(std::array<float,1>{7},std::array<float,1>{8});
        const auto inner=mtp.checkpoint();mtp.append(std::array<float,1>{9},std::array<float,1>{10});mtp.restore(inner);mtp.restore(outer);
        require(mtp.size()==2,"nested draft rollback lost accepted prefix");
        flow.reset();require(!mtp.size() && flow.tokens().empty(),"joint reset left MTP state");
        std::cout<<"PASS actual SessionFlow target+MTP auxiliary transaction, independent residual/token pairing, post-head cancel, retry and nested prefix rollback\n";
    } catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
}
