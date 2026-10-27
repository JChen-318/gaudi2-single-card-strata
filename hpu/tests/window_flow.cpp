#include "strata/hpu/layer_flow.hpp"
#include "strata/hpu/session_flow.hpp"
#include "strata/hpu/prefill_plan.hpp"
#include "strata/hpu/window_limits.hpp"
#include "strata/hpu/mtp_window_flow.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <memory>
#include <numeric>

using namespace strata::hpu;
namespace {
void require(bool ok,const char* message) {if(!ok) throw std::runtime_error(message);}
template<class F> void rejects(F call) {bool failed=false;try {call();} catch(const std::exception&) {failed=true;}require(failed,"invalid window accepted");}
struct Fixture {
    std::array<uint32_t,48> positions{},moe_calls{};
    std::array<uint64_t,49> epochs{};bool epoch_check=false;
    std::array<size_t,48> moe_rows{};
    std::array<uint32_t,48> hc_reads{},hc_writes{},mixer_windows{};bool batched_hc=true,batched_mixer=false,batched_ple=false;uint32_t ple_windows=0;
    std::vector<std::pair<uint32_t,uint32_t>> order;
    std::vector<float> previous{0,0,0,0};uint32_t draft_cells=0;double draft_sum=0;
    int fail_layer=-1,fail_row=1;bool bad_head=false,bad_export=false,rewind_fail=false,steer=true;
    int heads=0,bad_head_at=-1;size_t ple_history=0;
    bool batched_head=false,short_head=false,throw_head=false;uint32_t head_windows=0;
    LayerOperations layer_ops(uint32_t layer,std::span<const float> ple) {
        LayerOperations out;const auto start=positions[layer];
        if(layer==1) out.ple=[ple](std::span<const float> input) {
            std::vector<float> result(input.begin(),input.end());for(auto& x:result) x+=ple[0]+ple[1];return result;
        };
        out.read=[](std::span<const float> x,bool ffn) {return HcReadResult{{(x[0]+x[2])/2,(x[1]+x[3])/2},ffn?std::vector<float>{2,3}:std::vector<float>{0,1}};};
        out.mixer=[this,layer,start](std::span<const float> x) {
            if(int(layer)==fail_layer && positions[layer]-start==uint32_t(fail_row)) throw std::runtime_error("injected later mixer failure");
            order.emplace_back(layer,positions[layer]);const auto add=float(++positions[layer]);return std::vector<float>{x[0]+add,x[1]+add};
        };
        out.moe=[this,layer](std::span<const float> x) {++moe_calls[layer];moe_rows[layer]=x.size()/2;std::vector<float> result(x.begin(),x.end());for(auto& f:result) f=f/8+1;return result;};
        out.write=[](std::span<const float> x,std::span<const float> block,std::span<const float> inject,bool) {
            std::vector<float> result(x.begin(),x.end());for(size_t c=0;c<2;++c) for(size_t d=0;d<2;++d) result[c*2+d]+=block[d]*(inject[c]+1)/16;return result;
        };
        out.rollback_mixer=[this,layer,start] {positions[layer]=start;};return out;
    }
    void draft(std::span<const float> residual,std::span<const float> embedding) {
        if(draft_cells || positions[0]>1) {for(size_t i=0;i<previous.size();++i) draft_sum+=double(previous[i])*(embedding[i%2]+1);++draft_cells;}
        previous.assign(residual.begin(),residual.end());
    }
    SessionOperations operations() {
        SessionOperations out;auto last_embedding=std::make_shared<std::vector<float>>();
        out.embedding=[last_embedding](int32_t token) {*last_embedding={float(token),float(token+1)};return *last_embedding;};
        out.ple=[this](int32_t token,std::span<const int32_t> ids) {ple_history=ids.size();return std::vector<float>{float(token)/8,float(ids.size())/8};};
        out.checkpoint=[this](uint32_t layer) {
            const auto state=positions;const auto prev=previous;const auto cells=draft_cells;const auto sum=draft_sum;
            const auto epoch=epochs[layer];
            return std::function<void()>([this,layer,state,prev,cells,sum,epoch] {
                if(rewind_fail && layer==2) throw std::runtime_error("injected window rewind failure");
                if(epoch_check && epoch!=epochs[layer]) throw std::runtime_error("stale epoch checkpoint reused");
                ++epochs[layer];
                if(layer==48) {previous=prev;draft_cells=cells;draft_sum=sum;} else positions[layer]=state[layer];
            });
        };
        out.layer=[this,last_embedding](uint32_t layer,std::span<const float> input,std::span<const float> ple) {
            if(layer==48) {draft(input,*last_embedding);return std::vector<float>(input.begin(),input.end());}
            return execute_layer(input,2,2,layer_ops(layer,ple));
        };
        out.window_layer=[this](uint32_t layer,std::span<const float> input,std::span<const float> embeddings,std::span<const float> ple,const auto& cancelled) {
            const auto rows=input.size()/4;
            if(layer==48) {
                // Independent auxiliary pairing oracle: row0 on the first
                // window has no preceding target row. Later windows do.
                const auto saved_previous=previous;const auto saved_cells=draft_cells;const auto saved_sum=draft_sum;
                const bool resumed=positions[0]>rows;
                const auto pairs=pair_mtp_prefix(input,embeddings,resumed?std::span<const float>(previous):std::span<const float>{},2,2);
                try {for(size_t row=0;row<rows;++row) {
                    if(cancelled && cancelled()) throw Cancelled();
                    if(resumed || row) {
                        const auto pair=row-(resumed?0:1);
                        for(size_t i=0;i<4;++i) draft_sum+=double(pairs.residual[pair*4+i])*(pairs.embedding[pair*2+i%2]+1);
                        ++draft_cells;
                    }
                    previous.assign(input.begin()+row*4,input.begin()+(row+1)*4);
                }} catch(...) {previous=saved_previous;draft_cells=saved_cells;draft_sum=saved_sum;throw;}
                return std::vector<float>(input.begin(),input.end());
            }
            auto op=layer_ops(layer,{});size_t row=0;
            if(layer==1) op.ple=[ple,row](std::span<const float> input) mutable {
                std::vector<float> result(input.begin(),input.end());for(auto& x:result) x+=ple[row*2]+ple[row*2+1];++row;return result;
            };
            if(batched_hc) {
                if(batched_ple && layer==1) op.window_ple=[this,ple_row=op.ple](std::span<const float> input) {
                    ++ple_windows;std::vector<float> result;
                    for(size_t at=0;at<input.size();at+=4) {const auto value=ple_row(input.subspan(at,4));result.insert(result.end(),value.begin(),value.end());}
                    return result;
                };
                if(batched_mixer) op.window_mixer=[this,layer,start=positions[layer],mixer=op.mixer](std::span<const float> input) {
                    ++mixer_windows[layer];std::vector<float> result;
                    try {for(size_t at=0;at<input.size();at+=2) {
                        const auto value=mixer(input.subspan(at,2));result.insert(result.end(),value.begin(),value.end());
                    }} catch(...) {positions[layer]=start;throw;}
                    return result;
                };
                op.window_read=[this,layer,read=op.read](std::span<const float> input,bool ffn) {
                    ++hc_reads[layer];HcReadResult result;
                    for(size_t at=0;at<input.size();at+=4) {auto value=read(input.subspan(at,4),ffn);result.mixed.insert(result.mixed.end(),value.mixed.begin(),value.mixed.end());result.inject.insert(result.inject.end(),value.inject.begin(),value.inject.end());}
                    return result;
                };
                op.window_write=[this,layer,write=op.write](std::span<const float> input,std::span<const float> block,std::span<const float> inject,bool ffn) {
                    ++hc_writes[layer];std::vector<float> result;
                    for(size_t at=0;at<input.size()/4;++at) {auto value=write(input.subspan(at*4,4),block.subspan(at*2,2),inject.subspan(at*2,2),ffn);result.insert(result.end(),value.begin(),value.end());}
                    return result;
                };
            }
            auto result=execute_layer_window(input,2,2,op,cancelled);if(bad_export && layer==47) result[0]=NAN;return result;
        };
        out.after_layer=[this](uint32_t layer,std::span<const float> input) {
            std::vector<float> result(input.begin(),input.end());if(steer && layer<48) for(auto& x:result) x+=float(layer%3)/32;return result;
        };
        out.head=[this](std::span<const float> input) {++heads;return std::vector<float>(16,(bad_head || heads==bad_head_at)?INFINITY:std::accumulate(input.begin(),input.end(),0.0f));};
        if(batched_head) out.window_head=[this,head=out.head](std::span<const float> input,const auto& cancelled) {
            ++head_windows;if(throw_head) throw std::runtime_error("injected native head failure");
            std::vector<float> output;for(size_t at=0;at<input.size();at+=4) {
                if(cancelled && cancelled()) throw Cancelled();
                auto row=head(input.subspan(at,4));output.insert(output.end(),row.begin(),row.end());
            }
            if(short_head) output.pop_back();
            return output;
        };
        out.reset_layer=[this](uint32_t layer) {if(layer<48) positions[layer]=0;else {draft_cells=0;draft_sum=0;previous.assign(4,0);}};return out;
    }
    void at(uint32_t position) {for(auto value:positions) require(value==position,"window layer positions diverged");}
};
void windows(bool batched_hc,bool batched_mixer=false,bool batched_ple=false) {
    const std::array<int32_t,5> input{1,2,3,4,5};Fixture seq,batch;batch.batched_hc=batched_hc;batch.batched_mixer=batched_mixer;batch.batched_ple=batched_ple;SessionFlow reference(2,2,49,16,32,seq.operations()),window(2,2,49,16,32,batch.operations());
    const auto expected=reference.prefill(input),actual=window.prefill_window(input);batch.at(5);
    require(actual==expected && seq.previous==batch.previous && seq.draft_cells==batch.draft_cells && seq.draft_sum==batch.draft_sum,"layer-major window changed scalar target or MTP pairing oracle");
    require(batch.heads==1 && seq.heads==5 && batch.ple_history==4,"window materialized per-token target heads or lost PLE history");
    for(size_t layer=0;layer<48;++layer) require(batch.moe_calls[layer]==1 && batch.moe_rows[layer]==5,"window did not group real FFN rows into one MoE call");
    for(size_t layer=0;layer<48;++layer) require(batch.hc_reads[layer]==(batched_hc?2u:0u) && batch.hc_writes[layer]==(batched_hc?2u:0u),"window did not use exactly two batched HC read/write calls per layer");
    for(size_t layer=0;layer<48;++layer) require(batch.mixer_windows[layer]==(batched_mixer?1u:0u),"window mixer did not execute once per layer");
    require(batch.ple_windows==(batched_ple?1u:0u),"window PLE was skipped or called more than once");
    require(batch.order[0]==std::pair<uint32_t,uint32_t>{0,0} && batch.order[4]==std::pair<uint32_t,uint32_t>{0,4} && batch.order[5]==std::pair<uint32_t,uint32_t>{1,0},"window remained token-major");
    const std::array<int32_t,2> continuation{6,7};require(reference.prefill(continuation)==window.prefill_window(continuation),"resumed window logits mismatch");
    require(seq.previous==batch.previous && seq.draft_cells==batch.draft_cells && seq.draft_sum==batch.draft_sum,"resumed MTP previous residual/next embedding mismatch");
    for(int layer=0;layer<48;++layer) {batch.fail_layer=layer;rejects([&]{(void)window.prefill_window(continuation);});batch.at(7);require(window.tokens().size()==7,"failed window committed token prefix");}
    batch.fail_layer=-1;batch.bad_head=true;rejects([&]{(void)window.prefill_window(continuation);});batch.at(7);batch.bad_head=false;
    // A cancellation after the auxiliary layer must rewind both trunk and MTP.
    const auto sum=batch.draft_sum;const auto cells=batch.draft_cells;const auto previous=batch.previous;
    const int shift=(batched_mixer?48:0)+(batched_ple?1:0);
    for(int boundary:std::array<int,9>{1,5,24,100,240,400,(batched_hc?535:533)-shift,(batched_hc?538:536)-shift,(batched_hc?539:537)-shift}) {
        int polls=0;try {(void)window.prefill_window(continuation,[&]{return ++polls==boundary;});} catch(const Cancelled&) {}
        if(polls>=boundary) {batch.at(7);require(window.tokens().size()==7 && batch.draft_sum==sum && batch.draft_cells==cells && batch.previous==previous,"cancelled window failed full target/MTP rollback");}
        else throw std::runtime_error("cancellation boundary did not execute");
    }
    batch.bad_export=true;rejects([&]{(void)window.prefill_window(continuation);});batch.at(7);batch.bad_export=false;
    batch.fail_layer=7;batch.rewind_fail=true;rejects([&]{(void)window.prefill_window(continuation);});require(window.poisoned(),"window rewind failure did not poison session");
    batch.fail_layer=-1;batch.rewind_fail=false;window.reset();batch.at(0);require(!window.poisoned() && window.tokens().empty(),"window failed reset recovery");
    rejects([&]{(void)window.prefill_window({});});const std::array<int32_t,2> invalid{1,16};rejects([&]{(void)window.prefill_window(invalid);});
}
void memory() {
    const auto& g=strata::core::kCompiledGeometry;
    require(!plan_prefill_window(g,0,128).host_bytes,"disabled window reserves rows");
    const auto a=plan_prefill_window(g,8,128),b=plan_prefill_window(g,16,128),small=plan_prefill_window(g,128,4);
    require(a.rows==8 && b.host_bytes>a.host_bytes && b.host_bytes<2*a.host_bytes && b.hbm_bytes>a.hbm_bytes && small.rows==4,"bounded shared GDN/window memory geometry");
    auto different=g;different.n_vocab=INT32_MAX;require(plan_prefill_window(different,8,128).host_bytes==a.host_bytes,"prefill reserved a token*vocabulary slab");
    rejects([&]{(void)plan_prefill_window(g,kResidentWindowRows+1,128);});rejects([&]{(void)plan_prefill_window(g,1,0);});
    different.n_embd=INT64_MAX;rejects([&]{(void)plan_prefill_window(different,128,128);});
}
void candidates(bool batched_head) {
    uint32_t cases=0;const std::array<int32_t,7> proposed{1,2,3,4,5,6,7};
    for(const uint32_t root:{0u,3u}) for(uint32_t count=1;count<=7;++count) for(uint32_t keep=0;keep<=count;++keep) {
        Fixture oracle,batch;batch.batched_hc=true;batch.batched_mixer=true;batch.batched_ple=true;batch.epoch_check=true;batch.batched_head=batched_head;
        SessionFlow reference(2,2,49,16,32,oracle.operations()),window(2,2,49,16,32,batch.operations());
        for(uint32_t i=0;i<root;++i) {reference.decode(8+int32_t(i));window.decode(8+int32_t(i));}
        std::vector<float> all;
        for(auto id:std::span(proposed).first(count)) {auto row=reference.decode(id);all.insert(all.end(),row.begin(),row.end());}
        // A separate scalar oracle contains only the chosen prefix, including
        // root MTP residual/next-embedding alignment and control transforms.
        Fixture prefix;SessionFlow expected(2,2,49,16,32,prefix.operations());std::vector<float> expected_logits;
        for(uint32_t i=0;i<root;++i) expected.decode(8+int32_t(i));
        for(auto id:std::span(proposed).first(keep)) expected_logits=expected.decode(id);
        const auto heads=batch.heads;
        const auto result=window.verify_window(std::span(proposed).first(count),count*16*4,[&](std::span<const float> values) {
            require(std::equal(values.begin(),values.end(),all.begin(),all.end()),"candidate row logits differ from scalar target oracle");
            require(window.tokens().size()==root,"unselected candidates published to session history");return keep;
        });
        require(result.accepted==keep && result.logits==expected_logits,"retained candidate prefix/logits mismatch");
        require(batch.heads-heads==int(count+(keep && keep<count?1:0)),"full acceptance replayed target or partial replay computed all heads");
        require(batch.head_windows==(batched_head?1u:0u),"candidate head was not invoked once for the whole window");
        batch.at(root+keep);require(std::equal(window.tokens().begin(),window.tokens().end(),expected.tokens().begin(),expected.tokens().end()),"candidate history retained rejected tail");
        require(batch.previous==prefix.previous && batch.draft_cells==prefix.draft_cells && batch.draft_sum==prefix.draft_sum,"candidate MTP state retained rejected tail");
        require(window.decode(12)==expected.decode(12),"candidate prefix continuation changed target output");
        require(batch.previous==prefix.previous && batch.draft_sum==prefix.draft_sum,"candidate prefix continuation changed MTP alignment");++cases;
    }
    Fixture batch;batch.batched_mixer=true;batch.batched_ple=true;batch.epoch_check=true;batch.batched_head=batched_head;
    SessionFlow window(2,2,49,16,32,batch.operations());window.decode(9);
    const auto previous=batch.previous;const auto sum=batch.draft_sum;
    auto root=[&] {batch.at(1);require(window.tokens().size()==1 && batch.previous==previous && batch.draft_sum==sum && !window.poisoned(),"candidate failure did not restore healthy root");};
    rejects([&]{window.verify_window(proposed,7*16*4-1,[](auto){return 7;});});root();
    rejects([&]{window.verify_window({},0,[](auto){return 0;});});root();
    rejects([&]{window.verify_window(proposed,UINT64_MAX,{});});root();
    const std::array<int32_t,8> too_many{};rejects([&]{window.verify_window(too_many,UINT64_MAX,[](auto){return 0;});});root();
    rejects([&]{window.verify_window(proposed,UINT64_MAX,[](auto)->uint32_t{throw std::runtime_error("selector failure");});});root();
    rejects([&]{window.verify_window(proposed,UINT64_MAX,[](auto){return 8;});});root();
    bool cancel=false;
    rejects([&]{window.verify_window(proposed,UINT64_MAX,[&](auto){cancel=true;return 3;},[&]{return cancel;});});root();cancel=false;
    // Replay has already consumed the original GDN-like epoch checkpoint.
    // Its own failure must use fresh roots, not try the stale original again.
    rejects([&]{window.verify_window(proposed,UINT64_MAX,[&](auto){batch.fail_layer=17;return 3;});});root();batch.fail_layer=-1;
    batch.bad_head=true;rejects([&]{window.verify_window(proposed,UINT64_MAX,[](auto){return 7;});});root();batch.bad_head=false;
    if(batched_head) {
        batch.short_head=true;rejects([&]{window.verify_window(proposed,UINT64_MAX,[](auto){return 7;});});root();batch.short_head=false;
        batch.throw_head=true;rejects([&]{window.verify_window(proposed,UINT64_MAX,[](auto){return 7;});});root();batch.throw_head=false;
    }
    batch.bad_head_at=batch.heads+4;rejects([&]{window.verify_window(proposed,UINT64_MAX,[](auto){return 7;});});root();batch.bad_head_at=-1;
    const auto head_cancel=batch.heads+3;
    rejects([&]{window.verify_window(proposed,UINT64_MAX,[](auto){return 7;},[&]{return batch.heads>=head_cancel;});});root();
    for(int layer=0;layer<48;++layer) {batch.fail_layer=layer;rejects([&]{window.verify_window(proposed,UINT64_MAX,[](auto){return 7;});});root();}batch.fail_layer=-1;
    batch.rewind_fail=true;rejects([&]{window.verify_window(proposed,UINT64_MAX,[](auto){return 0;});});require(window.poisoned(),"candidate failed rewind not poisoned");
    batch.rewind_fail=false;window.reset();batch.at(0);require(!window.poisoned(),"candidate poison reset did not recover");
    std::cout<<"PASS candidate window "<<cases<<" scalar 48-layer target/MTP/control/PLE prefix cases, bounded all-row logits, full-accept state reuse, fresh-epoch partial replay, selection/cancel/layer/head/replay failure and poison/reset\n";
}
}
int main() {
    try {windows(false);windows(true);windows(true,true);windows(true,true,true);memory();candidates(false);candidates(true);std::cout<<"PASS production layer-major 48-layer windows, batched PLE/HC/mixer/head/fallback, one grouped MoE call/layer, scalar target/MTP/control/PLE oracles, every layer failure, cancellation, head/export failure, poison/reset and bounded memory\n";}
    catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
}
