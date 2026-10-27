#include "strata/hpu/qsa_state.hpp"
#include "strata/hpu/dense_kv.hpp"
#include <algorithm>
#include <array>
#include <iostream>
using namespace strata::hpu;
namespace {
void require(bool ok,const char* what) {if(!ok) throw std::runtime_error(what);}
template<class F> void rejects(F f) {bool bad=false;try {f();} catch(const std::exception&) {bad=true;}require(bad,"stale branch accepted");}
void append(QsaState& state,int index) {
    const auto& c=state.config();std::vector<float> key(c.kv_heads*c.width),value(key.size()),raw(c.index_width),gamma(c.index_width,1);
    for(size_t i=0;i<key.size();++i) {key[i]=float(index*13+int(i)%11)/128;value[i]=float(index*7-int(i)%5)/64;}
    for(size_t i=0;i<raw.size();++i) raw[i]=float(index+int(i)%3)/16;
    state.append(key,value,raw,gamma);
}
}
int main() {
    try {
        QsaConfig c;c.width=8;c.rotary=4;c.index_width=4;c.index_heads=2;c.query_heads=4;c.kv_heads=1;c.block=4;c.top_k=5;c.capacity=32;
        QsaState state(c),reference(c);const auto bytes=state.allocated_bytes();
        for(int i=0;i<5;++i) {append(state,i);append(reference,i);}
        const auto parent=state.checkpoint();for(int i=5;i<9;++i) append(state,i);
        const auto child=state.checkpoint();for(int i=9;i<14;++i) append(state,i);
        state.restore(child);state.restore(parent);state.restore(parent);
        require(state.size()==5,"nested QSA rollback lost parent");
        for(int i=5;i<10;++i) {append(state,100+i);append(reference,100+i);}
        rejects([&]{state.restore(child);});
        require(std::equal(state.pooled_rows().begin(),state.pooled_rows().end(),reference.pooled_rows().begin(),reference.pooled_rows().end()),"QSA pooling after nested branch rewind");
        for(int32_t first:{0,5}) {
            const std::array<int32_t,5> ids{first,first+1,first+2,first+3,first+4};
            const auto a=state.gather(ids,5),b=reference.gather(ids,5);require(a.key==b.key && a.value==b.value,"QSA accepted/replayed FP16 KV differs");
        }
        require(state.allocated_bytes()==bytes,"QSA rewind ancestry grew beyond reserved capacity");
        const auto snapshot=state.snapshot();const auto before=state.checkpoint();state.restore(snapshot);rejects([&]{state.restore(before);});
        const auto reset=state.checkpoint();state.reset();rejects([&]{state.restore(reset);});
        DenseKvState draft(32,1,8);std::array<float,8> row{};draft.append(row,row);const auto outer=draft.checkpoint();
        draft.append(row,row);const auto inner=draft.checkpoint();draft.append(row,row);draft.restore(inner);draft.restore(outer);
        require(draft.size()==1,"MTP child rewind invalidated outer window");draft.append(row,row);rejects([&]{draft.restore(inner);});
        std::cout<<"PASS actual QSA/MTP nested rewind ancestry, pooling-boundary replay, stale branch/snapshot/reset rejection and fixed memory\n";
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
