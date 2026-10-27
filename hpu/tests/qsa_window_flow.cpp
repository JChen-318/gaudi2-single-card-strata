#include "strata/hpu/qsa_window_flow.hpp"
#include "strata/hpu/session_flow.hpp"
#include "strata/hpu/layer_flow.hpp"
#include "strata/hpu/window_limits.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <numeric>
using namespace strata::hpu;
namespace {
void require(bool ok,const char* what) {if(!ok) throw std::runtime_error(what);}
template<class F> void rejects(F call) {bool failed=false;try {call();} catch(const std::exception&) {failed=true;}require(failed,"invalid QSA window accepted");}
void equal(std::span<const float> a,std::span<const float> b) {
    require(a.size()==b.size(),"QSA oracle shape mismatch");
    for(size_t i=0;i<a.size();++i) require(a[i]==b[i] || std::abs(a[i]-b[i])<1e-6f,"QSA window changed causal projection/KV/attention/output oracle");
}
QsaConfig config() {
    QsaConfig c;c.query_heads=6;c.kv_heads=2;c.width=8;c.rotary=4;c.index_heads=2;c.index_width=4;
    c.capacity=256;c.page_size=3;c.block=4;c.top_k=4;c.position_base=7;return c;
}
struct OracleRow {std::vector<float> query,gate;SelectedKv kv;};
struct Fixture {
    QsaConfig c=config();std::vector<float> qgamma,kgamma,igamma,ikgamma;
    int stage=0,fail=-1,bad=-1;std::vector<uint32_t> batches;
    std::vector<OracleRow> expected;size_t checked=0;
    Fixture():qgamma(c.query_heads*c.width),kgamma(c.kv_heads*c.width),igamma(c.index_heads*c.index_width),ikgamma(c.index_width) {
        for(auto* values:{&qgamma,&kgamma,&igamma,&ikgamma}) for(size_t i=0;i<values->size();++i) (*values)[i]=.5f+float(i%7)/8;
    }
    QsaNorms norms() const {return {qgamma,kgamma,igamma,ikgamma};}
    void visit() {if(stage++==fail) throw std::runtime_error("injected QSA native stage failure");}
    std::vector<float> matrix(std::span<const float> input,uint32_t in,uint32_t out,int seed,bool round_input=true) {
        std::vector<float> values(input.size()/in*out);
        for(size_t row=0;row<input.size()/in;++row) for(uint32_t o=0;o<out;++o) for(uint32_t i=0;i<in;++i) {
            const auto x=round_input?from_bf16(to_bf16(input[row*in+i])):input[row*in+i];
            values[row*out+o]+=x*float(int((o*11+i*7+seed)%19)-9)/64;
        }
        return values;
    }
    QsaProjected project(std::span<const float> x) {
        return {matrix(x,4,c.query_heads*c.width*2,3),matrix(x,4,c.kv_heads*c.width,7),matrix(x,4,c.kv_heads*c.width,11),
                matrix(x,4,c.index_heads*c.index_width,13),matrix(x,4,c.index_width,17)};
    }
    std::vector<float> output(std::span<const float> x) {return matrix(x,c.query_heads*c.width,4,23);}
    std::vector<float> gated(std::span<const float> q,std::span<const float> gate,const SelectedKv& kv) {
        auto out=qsa_attention_reference(q,kv,c.query_heads,c.kv_heads,c.width);
        for(size_t i=0;i<out.size();++i) out[i]*=float(1/(1+std::exp(-double(gate[i]))));
        return out;
    }
    // Independent singleton control oracle, retaining the exact prefix gather
    // before any subsequent token or completed compression block can appear.
    std::vector<float> singleton(QsaState& state,std::span<const float> x,bool capture) {
        auto p=project(x);std::vector<float> query(c.query_heads*c.width),gate(query.size());
        for(uint32_t h=0;h<c.query_heads;++h) for(uint32_t d=0;d<c.width;++d) {
            query[h*c.width+d]=p.q_gate[h*2*c.width+d];gate[h*c.width+d]=p.q_gate[(h*2+1)*c.width+d];
        }
        rms_norm_rows(query,c.width,qgamma,c.epsilon);rms_norm_rows(p.key,c.width,kgamma,c.epsilon);rms_norm_rows(p.index_query,c.index_width,igamma,c.epsilon);
        const auto rope=rope_row(c.rotary,c.position_base+state.size(),c.rope);
        rotate_neox(query,c.width,rope);rotate_neox(p.key,c.width,rope);rotate_neox(p.index_query,c.index_width,rope);
        state.append(p.key,p.value,p.index_key,ikgamma);const auto ids=state.select(p.index_query);const auto kv=state.gather(ids,uint32_t(ids.size()));
        auto result=output(gated(query,gate,kv));if(capture) expected.push_back({std::move(query),std::move(gate),kv});return result;
    }
    QsaWindowOperations operations() {
        QsaWindowOperations op;
        op.project=[&](auto x) {visit();auto p=project(x);batches.push_back(uint32_t(x.size()/4));if(stage-1==bad) p.index_key.pop_back();return p;};
        op.attention=[&](const QsaWindowBatch& b) {
            visit();std::vector<float> result;const uint64_t q=uint64_t(c.query_heads)*c.width,kvrow=uint64_t(c.kv_heads)*b.bucket*c.width;
            require(b.rows<=8 && b.bucket<=c.top_k+c.block-1,"unbounded QSA batch allocation");
            for(uint32_t row=0;row<b.rows;++row) {
                SelectedKv kv{0,b.bucket,{b.key.begin()+row*kvrow,b.key.begin()+(row+1)*kvrow},{b.value.begin()+row*kvrow,b.value.begin()+(row+1)*kvrow},
                              {b.mask.begin()+uint64_t(row)*c.kv_heads*b.bucket,b.mask.begin()+(uint64_t(row)*c.kv_heads+1)*b.bucket}};
                for(float x:kv.mask) if(x==0) ++kv.count;
                if(!expected.empty()) {
                    const auto& e=expected.at(checked++);equal(std::span<const float>(b.query).subspan(row*q,q),e.query);equal(std::span<const float>(b.gate).subspan(row*q,q),e.gate);
                    require(kv.count==e.kv.count,"future token/completed pool changed prior selection count");
                    for(uint32_t h=0;h<c.kv_heads;++h) for(uint32_t item=0;item<b.bucket;++item) {
                        const auto mask=b.mask[(uint64_t(row)*c.kv_heads+h)*b.bucket+item];require(item<kv.count?mask==0:mask==-INFINITY,"per-row/head mask broadcast mismatch");
                        for(uint32_t d=0;d<c.width;++d) {
                            const auto at=(uint64_t(h)*b.bucket+item)*c.width+d;
                            const auto old=(uint64_t(h)*e.kv.bucket+item)*c.width+d;
                            require(kv.key[at]==(item<kv.count?e.kv.key[old]:0) && kv.value[at]==(item<kv.count?e.kv.value[old]:0),"window gather leaked a future row or mixed KV heads");
                        }
                    }
                }
                const auto values=gated(std::span<const float>(b.query).subspan(row*q,q),std::span<const float>(b.gate).subspan(row*q,q),kv);
                result.insert(result.end(),values.begin(),values.end());
            }
            if(stage-1==bad) result[0]=NAN;
            return result;
        };
        op.output=[&](auto x) {visit();auto values=output(x);if(stage-1==bad) values.pop_back();return values;};return op;
    }
};
std::vector<float> input(size_t rows,int start=0) {
    std::vector<float> values(rows*4);
    for(size_t row=0;row<rows;++row) for(size_t i=0;i<4;++i) values[row*4+i]=float(int(((row+start)*17+i*11)%37)-18)/16;
    return values;
}
void parity() {
    for(size_t rows:{1u,2u,7u,8u,9u,17u,128u}) {
        Fixture f;std::vector<uint32_t> pages((f.c.capacity+f.c.page_size-1)/f.c.page_size);std::iota(pages.rbegin(),pages.rend(),0);
        QsaState reference(f.c),state(f.c,pages);const auto prefix=input(3),x=input(rows,3);std::vector<float> expected;
        for(size_t row=0;row<3;++row) {const auto one=std::span<const float>(prefix).subspan(row*4,4);(void)f.singleton(reference,one,false);(void)f.singleton(state,one,false);}
        const auto point=state.checkpoint();
        for(size_t row=0;row<rows;++row) {const auto value=f.singleton(reference,std::span<const float>(x).subspan(row*4,4),true);expected.insert(expected.end(),value.begin(),value.end());}
        const auto actual=execute_qsa_window(state,x,4,f.norms(),f.operations());equal(actual,expected);equal(state.pooled_rows(),reference.pooled_rows());
        require(state.size()==rows+3 && f.checked==rows && f.batches.size()==(rows+7)/8,"QSA chunk count or state commit mismatch");
        state.restore(point);require(state.size()==3,"successful QSA window lost outer checkpoint ancestry");
    }
}
void failures() {
    Fixture f;QsaState state(f.c);const auto prefix=input(3),x=input(17,3);
    for(size_t row=0;row<3;++row) (void)f.singleton(state,std::span<const float>(prefix).subspan(row*4,4),false);
    const auto point=state.checkpoint();const auto saved_pool=std::vector<float>(state.pooled_rows().begin(),state.pooled_rows().end());const std::array<int32_t,3> ids{0,1,2};const auto saved=state.gather(ids,7);
    auto intact=[&] {require(state.size()==3,"failed/cancelled QSA window accepted a prefix");equal(state.pooled_rows(),saved_pool);const auto kv=state.gather(ids,7);equal(kv.key,saved.key);equal(kv.value,saved.value);state.restore(point);};
    for(int bad=0;bad<9;++bad) for(bool malformed:{false,true}) {
        Fixture broken;if(malformed) broken.bad=bad;else broken.fail=bad;
        rejects([&] {(void)execute_qsa_window(state,x,4,broken.norms(),broken.operations());});intact();
    }
    size_t total=0;Fixture good;(void)execute_qsa_window(state,x,4,good.norms(),good.operations(),[&] {++total;return false;});state.restore(point);
    for(size_t boundary=1;boundary<=total;++boundary) {
        Fixture cancelled;size_t polls=0;rejects([&] {(void)execute_qsa_window(state,x,4,cancelled.norms(),cancelled.operations(),[&] {return ++polls==boundary;});});intact();
    }
    Fixture invalid;auto operations=invalid.operations();rejects([&] {(void)execute_qsa_window(state,{},4,invalid.norms(),operations);});
    rejects([&] {(void)execute_qsa_window(state,input(kResidentWindowRows+1),4,invalid.norms(),operations);});
    auto bad=input(1);bad[0]=NAN;rejects([&] {(void)execute_qsa_window(state,bad,4,invalid.norms(),operations);});require(invalid.stage==0,"invalid input reached native projections");
}
void ids_only() {
    Fixture f;QsaState reference(f.c),state(f.c);const auto x=input(128);
    std::vector<float> expected;
    for(uint32_t r=0;r<128;++r) {
        const auto y=f.singleton(reference,std::span<const float>(x).subspan(r*4,4),true);
        expected.insert(expected.end(),y.begin(),y.end());
    }
    auto op=f.operations();const auto original=op.attention;op.selected_ids_only=true;
    op.attention=[&](const QsaWindowBatch& batch) {
        require(batch.key.empty() && batch.value.empty() && batch.mask.empty(),"ids-only stage retained KV slabs");
        require(batch.ids.size()==uint64_t(batch.rows)*batch.bucket,"ids-only stage lost selected rows");
        auto host=batch;const auto kv=uint64_t(f.c.kv_heads)*batch.bucket*f.c.width;
        host.key.resize(batch.rows*kv);host.value.resize(batch.rows*kv);host.mask.resize(uint64_t(batch.rows)*f.c.kv_heads*batch.bucket);
        for(uint32_t r=0;r<batch.rows;++r) {
            std::vector<int32_t> ids;bool padded=false;
            for(uint32_t i=0;i<batch.bucket;++i) {
                const auto id=batch.ids[uint64_t(r)*batch.bucket+i];
                if(id<0) {require(id==-1,"noncanonical id padding");padded=true;}
                else {require(!padded && id==int32_t(id),"invalid id/padding order");ids.push_back(int32_t(id));}
            }
            auto mask=std::span<float>(host.mask).subspan(uint64_t(r)*f.c.kv_heads*batch.bucket,f.c.kv_heads*batch.bucket);
            state.gather_into(ids,batch.bucket,std::span<float>(host.key).subspan(r*kv,kv),std::span<float>(host.value).subspan(r*kv,kv),mask.first(batch.bucket));
            for(uint32_t h=1;h<f.c.kv_heads;++h) std::copy_n(mask.begin(),batch.bucket,mask.begin()+h*batch.bucket);
        }
        return original(host);
    };
    const auto point=state.checkpoint();equal(execute_qsa_window(state,x,4,f.norms(),op),expected);
    require(f.checked==128 && state.size()==128,"ids-only whole-window extent mismatch");state.restore(point);
    op.attention=[](const QsaWindowBatch&)->std::vector<float> {throw std::runtime_error("injected ids-only gather failure");};
    rejects([&]{(void)execute_qsa_window(state,x,4,f.norms(),op);});require(state.size()==0,"ids-only failure accepted rows");
}
void layer_rewind() {
    Fixture f;QsaState state(f.c);const auto prefix=input(3),window=input(17);
    for(size_t row=0;row<3;++row) (void)f.singleton(state,std::span<const float>(prefix).subspan(row*4,4),false);
    const auto parent=state.checkpoint(),layer_point=state.checkpoint();size_t rollbacks=0;bool fail_after=false,cancel_after=false,cancelled=false,bad_export=false;
    LayerOperations op;
    op.read=[](auto,bool)->HcReadResult {throw std::runtime_error("singleton HC was used");};
    op.write=[](auto,auto,auto,bool)->std::vector<float> {throw std::runtime_error("singleton HC write was used");};
    op.mixer=[](auto)->std::vector<float> {throw std::runtime_error("singleton QSA was used");};
    op.window_read=[](std::span<const float> values,bool) {return HcReadResult{std::vector<float>(values.size()/2,.125f),std::vector<float>(values.size()/4,1)};};
    op.window_write=[&](std::span<const float> values,auto,auto,bool ffn) {
        if(ffn && fail_after) throw std::runtime_error("injected HC failure after native QSA window");
        return std::vector<float>(values.begin(),values.end());
    };
    op.window_mixer=[&](auto values) {
        auto out=execute_qsa_window(state,values,4,f.norms(),f.operations());if(bad_export) out[0]=NAN;return out;
    };
    op.moe=[&](std::span<const float> values) {if(cancel_after) cancelled=true;return std::vector<float>(values.begin(),values.end());};
    op.rollback_mixer=[&] {state.restore(layer_point);++rollbacks;};const std::vector<float> residual(window.size()*2,.25f);
    require(execute_layer_window(residual,4,2,op)==residual && state.size()==20 && !rollbacks,"layer did not accept real QSA whole window");state.restore(parent);
    fail_after=true;rejects([&] {(void)execute_layer_window(residual,4,2,op);});require(state.size()==3 && rollbacks==1,"later HC failed to restore real QSA window");state.restore(parent);fail_after=false;
    bad_export=true;rejects([&] {(void)execute_layer_window(residual,4,2,op);});require(state.size()==3 && rollbacks==2,"invalid QSA layer export failed rewind");state.restore(parent);bad_export=false;
    cancel_after=true;rejects([&] {(void)execute_layer_window(residual,4,2,op,[&] {return cancelled;});});require(state.size()==3 && rollbacks==3,"post-QSA MoE cancellation failed nested parent rewind");state.restore(parent);
}
void device_buckets() {
    auto c=config();c.capacity=4096;c.top_k=2048;c.block=4;   // the production geometry: selection width 2051
    require(qsa_attention_bucket(c,2051)==2051,"QSA selection bucket fixture");
    require(qsa_device_bucket(c,0,1)==1 && qsa_device_bucket(c,513,1)==2048,"QSA device bucket gaps changed");
    require(qsa_device_bucket(c,2048,1)==2048,"QSA device bucket lost the top power");
    require(qsa_device_bucket(c,2049,1)==2050 && qsa_device_bucket(c,2050,1)==2050,"QSA device bucket did not clamp the top bucket");
    require(qsa_device_bucket(c,2047,4)==2047 && qsa_device_bucket(c,2048,3)==2048,"QSA verify bucket did not leave room for its rows");
    rejects([&] {(void)qsa_device_bucket(c,2051,1);});
    rejects([&] {(void)qsa_device_bucket(c,2048,4);});
    rejects([&] {(void)qsa_device_bucket(c,0,0);});
    for(uint32_t cells=0;cells<2051;++cells) {
        const auto bucket=qsa_device_bucket(c,cells,1);
        require(bucket>=cells && bucket+1<=2051,"QSA device bucket escaped the selection width");
    }
    std::cout<<"PASS QSA device buckets cover the cache and reserve their current rows at width 2051\n";
}
}
int main() {
    try {
        device_buckets();parity();ids_only();failures();layer_rewind();const QsaConfig target;
        auto same_size=target;same_size.query_heads=12;same_size.kv_heads=1;same_size.width=512;
        require(!qsa_window_geometry_equal(target,same_size) && qsa_window_geometry_equal(target,target),"same flattened sizes hid incompatible shared QSA head grouping");
        const auto pinned=qsa_window_attention_pinned_bytes(target,8,2051);
        require(pinned==(3ull*8*24*256+2ull*8*2*2051*256+8ull*2*2051+1)*4 && pinned>64ull*1024*1024,"target QSA attention slots silently exceed default64MiB recipe budget");
        std::cout<<"PASS production QSA window:1..128rows cap8, independent causal singleton/FP16 KV oracle, per-head masks/padding, permuted pages, compression/RoPE, every native stage failure/export/cancel, nested rewind; target_pinned_bytes="<<pinned<<'\n';
    }
    catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
