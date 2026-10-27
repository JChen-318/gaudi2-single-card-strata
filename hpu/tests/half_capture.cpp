#include "strata/hpu/half_capture.hpp"
#include "strata/hpu/qsa_state.hpp"
#include "strata/kernels/f16_bits.hpp"
#include <bit>
#include <iostream>
using namespace strata::hpu;
namespace core=strata::core;
namespace {
void need(bool ok,const char* why) {if(!ok) throw std::runtime_error(why);}
template<class F> void rejects(F call) {bool failed=false;try {call();} catch(const std::exception&) {failed=true;}need(failed,"invalid half capture accepted");}
core::ModelGeometry geometry() {auto g=core::kCompiledGeometry;g.n_layers=4;g.n_embd=3;g.hc=2;g.n_head=2;g.n_head_kv=1;g.head_dim=4;return g;}
void complete(HalfCapture& capture,const core::ModelGeometry& g,std::vector<float>& expected) {
    const auto h=uint64_t(g.n_embd),r=uint64_t(g.hc),q=uint64_t(g.n_head*g.head_dim),v=uint64_t(g.n_head_kv*g.head_dim);
    const auto stride=2*h+2*r+q+5*v+8;expected.assign(g.n_layers*stride,0.0f);
    for(uint32_t layer=0;layer<uint32_t(g.n_layers);++layer) {
        auto* out=expected.data()+layer*stride;
        for(uint32_t field=0;field<4;++field) {
            std::vector<float> values(field<2?h:r);
            for(size_t at=0;at<values.size();++at) values[at]=float(layer*100+field*10+at+1);
            capture.half(layer,HalfField(field),values);
            const auto offset=field<2?field*h:2*h+(field-2)*r;std::copy(values.begin(),values.end(),out+offset);
        }
        if(!core::is_qsa_layer(g,layer)) continue;
        std::vector<float> attention(q),raw(v),post(v),key(v);
        for(size_t at=0;at<q;++at) attention[at]=float(at+1)/7;
        for(size_t at=0;at<v;++at) {raw[at]=float(at+2)/11;post[at]=raw[at]+1;key[at]=-raw[at];}
        // Unused selected-ID sentinel -1 has a NaN F32 bit pattern. All
        // selected FP16 values here are finite. Do not F32-check raw bits.
        const std::array<uint16_t,4> selected{0x3c00,0xbc00,0x0001,0x8000},pool{0x3e00,0xbe00,0x0001,0x8000};
        QsaHalfView view{attention,raw,post,key,selected,pool};view.ids={0,1,-1,-1};view.step={1,2,0,2};
        capture.qsa(layer,view);
        // Independent ordered memcpy transcription of upstream, including
        // the early K prefix whose EVERY byte is later overwritten.
        const auto a=2*h+2*r+q;
        std::copy(attention.begin(),attention.end(),out+2*h+2*r);std::copy(raw.begin(),raw.end(),out+a);
        const std::array<uint16_t,4> dead_k{0x1111,0x2222,0x3333,0x4444};
        std::memcpy(out+a+2*v+8,dead_k.data(),v*2);std::memcpy(out+a+2*v+8+v/2,pool.data(),v*2);
        std::copy(post.begin(),post.end(),out+a+3*v+8);std::copy(key.begin(),key.end(),out+a+4*v+8);
        std::memcpy(out+a+2*v,selected.data(),v*2);std::memcpy(out+a+2*v+v/2,view.ids.data(),16);std::memcpy(out+a+2*v+v/2+4,view.step.data(),16);
    }
}
void state_bits() {
    QsaConfig c;c.query_heads=2;c.kv_heads=2;c.width=4;c.rotary=2;c.index_heads=1;c.index_width=4;c.capacity=16;c.page_size=4;c.top_k=2;
    QsaState state(c,{2,0,3,1});const auto allocated=state.allocated_bytes();
    for(uint32_t cell=0;cell<8;++cell) {
        std::array<float,8> values{},keys{};const std::array<float,4> raw{1,2,3,4},gamma{1,1,1,1};
        for(uint32_t at=0;at<8;++at) values[at]=float(cell*13+at+1)/17;
        state.append(keys,values,raw,gamma);
    }
    const auto prefix=state.value_pool_prefix();std::array<uint16_t,8> selected{};state.selected_value_bits(7,selected);
    for(uint32_t at=0;at<8;++at) {
        const auto logical=4+at/4,dim=at%4;
        need(prefix[at]==strata::kernels::f16_from_f32(float(logical*13+dim+1)/17),"half pool view used logical rather than physical prefix");
        need(selected[at]==strata::kernels::f16_from_f32(float(7*13+at+1)/17),"half selected view lost token-major heads or FP16 bits");
    }
    need(state.allocated_bytes()==allocated,"half bit views grew state buffers");state.reset();rejects([&]{(void)state.value_pool_prefix();});rejects([&]{state.selected_value_bits(0,selected);});
}
}
int main() {
    try {
        state_bits();const auto g=geometry();const HalfLayout layout(g);HalfCapture capture,off;
        const auto& full=core::kCompiledGeometry;const auto planned=plan_half_producer(full);
        const uint64_t stride=2*2560+2*4+24*256+5*2*256+8;
        need(planned.host_bytes==2*48*stride*4+96+4096+12*2*256*2+13*(24*256*4+16384) && planned.pinned_bytes==13*24*256*4 && planned.hbm_bytes==13*(24*256*4+(4ull<<20)),"half producer plan omitted staging/export/tap/scratch/workspace or live recipe replacement costs");
        auto no_qsa=g;no_qsa.n_layers=3;
        const auto no_tap=plan_half_producer(no_qsa);
        need(no_tap.host_bytes==HalfCapture::host_bytes(no_qsa) && !no_tap.pinned_bytes && !no_tap.hbm_bytes,"GDN-only half producer reserved a nonexistent QSA replacement");
        off.half(900,HalfField::FfnBlock,{});off.qsa(900,{});{auto round=off.round();round.finish();}
        need(!off.storage_bytes() && off.records().empty(),"disabled half capture allocated");
        rejects([&]{capture.configure(g,layout.host_bytes()-1);});need(!capture.enabled(),"rejected half budget configured storage");
        capture.configure(g,layout.host_bytes());const auto storage=capture.storage_bytes();std::vector<float> expected;
        for(int pass=0;pass<3;++pass) {
            {auto round=capture.round();complete(capture,g,expected);round.finish();}
            const auto exported=capture.records();need(exported.size()==expected.size() && std::memcmp(exported.data(),expected.data(),expected.size()*4)==0,"half wire differs from original ordered float/bit writes");
            need(storage+exported.capacity()*4<=layout.host_bytes() && storage==capture.storage_bytes(),"half export/storage exceeds admitted budget");
        }
        const auto old=capture.records();rejects([&]{capture.configure(g,layout.host_bytes());});need(capture.records().size()==old.size(),"reconfiguration changed a live capture or doubled its budget");
        rejects([&]{auto round=capture.round();capture.half(0,HalfField::AttentionBlock,std::vector<float>(3,1));round.finish();});
        need(capture.records().empty(),"partial half frame exposed old data");
        rejects([&]{auto round=capture.round();complete(capture,g,expected);capture.half(0,HalfField::AttentionBlock,std::vector<float>(3));});
        rejects([&]{auto round=capture.round();capture.half(4,HalfField::FfnInject,{});});
        rejects([&]{auto round=capture.round();capture.half(0,HalfField::AttentionBlock,std::array<float,3>{1,INFINITY,2});});
        for(int fault=0;fault<8;++fault) rejects([&] {
            auto round=capture.round();std::vector<float> attention(8,1),value(4,1);const std::array<uint16_t,4> bits{};
            QsaHalfView view{attention,value,value,value,bits,bits};view.ids={0,1,-1,-1};view.step={1,2,0,2};uint32_t layer=3;
            if(fault==0) view.attention=view.attention.first(7);
            if(fault==1) view.selected_value=view.selected_value.first(3);
            if(fault==2) view.step[0]=0;
            if(fault==3) view.step[2]=1;
            if(fault==4) view.step[3]=1;
            if(fault==5) view.ids[1]=0;
            if(fault==6) value[0]=NAN;
            if(fault==7) layer=0;
            capture.qsa(layer,view);
        });
        rejects([&]{auto round=capture.round();complete(capture,g,expected);round.finish();throw std::runtime_error("post-finish observer");});
        need(capture.records().empty() && capture.storage_bytes()==storage,"post-finish half failure retained old view or grew buffers");
        auto invalid=g;invalid.qsa_interval=0;rejects([&]{(void)HalfCapture::host_bytes(invalid);});invalid=g;invalid.head_dim=3;rejects([&]{(void)HalfCapture::host_bytes(invalid);});
        {auto round=capture.round();complete(capture,g,expected);round.finish();}
        need(!capture.view().empty(),"half capture failed to recover");
        {auto grouped=capture.round(false);capture.half(0,HalfField::AttentionBlock,{});grouped.finish();}
        need(capture.view().empty(),"unsupported grouped pass returned an old scalar frame");capture.clear();need(capture.view().empty(),"half reset retained a stale view");
        std::cout<<"PASS half capture actual upstream ordered byte layout, overwritten K slice, packed NaN bits, zero padding, completeness, failure/recovery, bounded export and disabled allocation; producer plumbing only\n";
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
