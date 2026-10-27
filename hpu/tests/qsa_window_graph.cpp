// Test-only CPU interpreter for the ACTUAL src/qsa.cpp graph factories. This
// target never links runtime.cpp/recipe.cpp or libSynapse. It models logical
// tensor math and BF16 boundaries, not SDK compilation, ISA accuracy or memory.
#include "strata/hpu/qsa.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <iostream>
#include <numeric>
#include "graph_interpreter.hpp"
using namespace strata::hpu;
namespace {
QsaConfig config() {QsaConfig c;c.query_heads=6;c.kv_heads=2;c.width=8;c.rotary=4;c.index_heads=2;c.index_width=4;c.capacity=32;c.top_k=4;return c;}
void scalar_half_tap() {
    const auto c=config();std::vector<float> baseline;size_t nodes=0;uint64_t persistent=0;
    for(bool tap:{false,true}) {
        GraphBuilder graph(OfflineCompilation{});define_qsa_attention(graph,c,7,tap);auto& g=*oracle::current;
        std::vector<float> query(c.query_heads*c.width),gate(query.size());SelectedKv kv{3,7,std::vector<float>(c.kv_heads*7*c.width),std::vector<float>(c.kv_heads*7*c.width),std::vector<float>(7,-INFINITY)};
        std::fill_n(kv.mask.begin(),3,0);
        for(size_t i=0;i<query.size();++i) {query[i]=float(int(i%13)-6)/17;gate[i]=float(int(i%7)-3)/4;}
        for(uint32_t head=0;head<c.kv_heads;++head) for(uint32_t cell=0;cell<3;++cell) for(uint32_t d=0;d<c.width;++d) {
            const auto at=(head*7+cell)*c.width+d;kv.key[at]=float(int((head*11+cell*7+d)%17)-8)/13;kv.value[at]=float(head*23+cell*5+d+1)/19;
        }
        g.set("query",query);g.set("q_gate",gate);g.set("key",kv.key);g.set("value",kv.value);g.set("mask",kv.mask);const std::array<float,1> scale{1/std::sqrt(float(c.width))};g.set("scale",scale);g.run();
        const auto pre=qsa_attention_reference(query,kv,c.query_heads,c.kv_heads,c.width);auto gated=pre;
        for(size_t i=0;i<gated.size();++i) gated[i]*=float(1/(1+std::exp(-double(gate[i]))));
        oracle::close(g.get("context_flat").values,pre);oracle::close(g.get("out").values,gated);
        oracle::require(g.get("context_flat").desc.persistent==tap,"QSA pre-gate output always/never allocated");
        uint64_t bytes=0;for(const auto& slot:g.slots) if(slot.desc.persistent) bytes+=slot.desc.bytes();
        if(!tap) {baseline=g.get("out").values;nodes=g.nodes.size();persistent=bytes;}
        else {oracle::close(g.get("out").values,baseline);oracle::require(g.nodes.size()==nodes && bytes==persistent+query.size()*4,"half tap re-ran attention or changed unrelated persistent schemas");}
    }
}
void attention(uint32_t rows,uint32_t bucket) {
    const auto c=config();GraphBuilder graph(OfflineCompilation{});define_qsa_attention_window(graph,c,bucket,rows);auto& g=*oracle::current;
    const uint64_t qsize=uint64_t(c.query_heads)*c.width;std::vector<float> query(rows*qsize),gate(query.size()),key,value,mask,expected;
    for(size_t i=0;i<query.size();++i) {query[i]=float(int(i*7%29)-14)/32;gate[i]=float(int(i*11%19)-9)/8;}
    for(uint32_t row=0;row<rows;++row) {
        const auto count=1+(row*3)%bucket;SelectedKv kv{count,bucket,std::vector<float>(c.kv_heads*bucket*c.width),std::vector<float>(c.kv_heads*bucket*c.width),std::vector<float>(bucket,-INFINITY)};
        std::fill_n(kv.mask.begin(),count,0);
        for(uint32_t h=0;h<c.kv_heads;++h) for(uint32_t cell=0;cell<count;++cell) for(uint32_t d=0;d<c.width;++d) {
            const auto i=(h*bucket+cell)*c.width+d;kv.key[i]=float(int((row*13+h*7+cell*5+d*3)%31)-15)/32;kv.value[i]=float(int((row*7+h*11+cell*17+d*5)%37)-18)/32;
        }
        key.insert(key.end(),kv.key.begin(),kv.key.end());value.insert(value.end(),kv.value.begin(),kv.value.end());
        for(uint32_t h=0;h<c.kv_heads;++h) mask.insert(mask.end(),kv.mask.begin(),kv.mask.end());
        auto result=qsa_attention_reference(std::span<const float>(query).subspan(row*qsize,qsize),kv,c.query_heads,c.kv_heads,c.width);
        for(size_t i=0;i<result.size();++i) result[i]*=float(1/(1+std::exp(-double(gate[row*qsize+i]))));
        expected.insert(expected.end(),result.begin(),result.end());
    }
    g.set("query",query);g.set("q_gate",gate);g.set("key",key);g.set("value",value);g.set("mask",mask);const std::array<float,1> scale{1/std::sqrt(float(c.width))};g.set("scale",scale);g.run();
    oracle::close(g.get("out").values,expected);oracle::require(g.get("scores").desc.shape==std::vector<uint64_t>{uint64_t(rows)*c.kv_heads,c.query_heads/c.kv_heads,bucket},"factory batch/head axes changed");
}
void projections(uint32_t rows) {
    const auto c=config();const uint32_t hidden=4;GraphBuilder graph(OfflineCompilation{});define_qsa_projections_window(graph,c,hidden,rows);auto& g=*oracle::current;
    std::vector<float> input(rows*hidden);for(size_t i=0;i<input.size();++i) input[i]=float(int(i*11%23)-11)/17;g.set("x",input);
    bool f32_preserved=false;
    for(const auto* name:{"q_gate","k","v","index_query","index_key"}) {
        const auto count=g.get(name).values.size()/rows;std::vector<float> weights(count*hidden);
        for(size_t i=0;i<weights.size();++i) weights[i]=float(int(i*13%29)-14)/71;
        g.set(std::string("w_")+name,weights);
    }
    g.run();
    for(const auto* name:{"q_gate","k","v","index_query","index_key"}) {
        const auto& actual=g.get(name);const auto& weights=g.get(std::string("w_")+name).values;const auto count=actual.values.size()/rows;std::vector<float> expected(actual.values.size());
        for(uint32_t row=0;row<rows;++row) for(size_t out=0;out<count;++out) for(uint32_t i=0;i<hidden;++i) expected[row*count+out]+=from_bf16(to_bf16(input[row*hidden+i]))*weights[out*hidden+i];
        oracle::close(actual.values,expected);oracle::require(actual.desc.dtype==DType::F32,"QSA projection output prematurely rounded to BF16");
        for(float value:actual.values) f32_preserved|=value!=from_bf16(to_bf16(value));
    }
    oracle::require(f32_preserved,"projection oracle failed to exercise F32 output precision");
}
void output(uint32_t rows) {
    const auto c=config();const uint32_t hidden=4;const auto inner=c.query_heads*c.width;GraphBuilder graph(OfflineCompilation{});define_qsa_output_window(graph,c,hidden,rows);auto& g=*oracle::current;
    std::vector<float> input(rows*inner),weights(hidden*inner),expected(rows*hidden);
    for(size_t i=0;i<input.size();++i) input[i]=float(int(i*11%29)-14)/31;
    for(size_t i=0;i<weights.size();++i) weights[i]=float(int(i*17%31)-15)/53;
    g.set("attention",input);g.set("w_out",weights);g.run();
    for(uint32_t row=0;row<rows;++row) for(uint32_t o=0;o<hidden;++o) for(uint32_t i=0;i<inner;++i)
        expected[row*hidden+o]+=from_bf16(to_bf16(input[row*inner+i]))*from_bf16(to_bf16(weights[o*inner+i]));
    oracle::close(g.get("out").values,expected);oracle::require(g.get("out").desc.dtype==DType::F32,"QSA output accumulation was rounded");
}
}
int main() {
    try {scalar_half_tap();for(uint32_t rows:{1u,3u,8u}) {projections(rows);output(rows);for(uint32_t bucket:{1u,7u}) attention(rows,bucket);}
        std::cout<<"PASS CPU interpreter of actual QSA window graph factories:12cases, merged batch/KV-head axes, group3, row masks/padding, BF16 inputs and F32 projection/output contract; no SDK/device execution\n";}
    catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
