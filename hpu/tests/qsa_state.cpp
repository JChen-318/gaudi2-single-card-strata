#include "strata/hpu/qsa_state.hpp"
#include "strata/hpu/state_io.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/kv_stream.hpp"   // kKvHybrid, the tag the parked header must carry
#include "strata/kernels/qsa.hpp"         // qsa_real_shapes(): the page size the header writes
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <chrono>
#include <filesystem>
#include <source_location>

using namespace strata::hpu;
namespace {
void require(bool ok,const std::string& message) { if(!ok) throw std::runtime_error(message); }
void close(std::span<const float> a,std::span<const float> b,float tolerance=1e-6f,std::source_location at=std::source_location::current()) {
    const auto what=std::string(at.function_name())+":"+std::to_string(at.line());
    require(a.size()==b.size(),what+": size mismatch "+std::to_string(a.size())+" vs "+std::to_string(b.size()));
    for(size_t i=0;i<a.size();++i) require(std::abs(a[i]-b[i])<=tolerance,
        what+" numeric mismatch at "+std::to_string(i)+": "+std::to_string(a[i])+" vs "+std::to_string(b[i]));
}
template<class F> void rejects(F&& f) {
    bool rejected=false;try { f(); } catch(const std::exception&) {rejected=true;}
    require(rejected,"invalid QSA operation accepted");
}
QsaConfig config() {
    QsaConfig c;c.query_heads=6;c.kv_heads=2;c.width=4;c.rotary=2;
    c.index_heads=2;c.index_width=4;c.block=4;c.top_k=2;c.capacity=32;c.page_size=4;return c;
}
void append(QsaState& s,int n) {
    const auto& c=s.config();std::vector<float> k(c.kv_heads*c.width),v(k.size()),idx(c.index_width),gamma(idx.size(),1);
    for(size_t i=0;i<k.size();++i) { k[i]=float(n+1)*0.0137f+float(i)*0.021f;v[i]=float(n-int(i))*0.137f; }
    for(size_t i=0;i<idx.size();++i) idx[i]=float((n+3)*(int(i)+2)%11)-4;
    s.append(k,v,idx,gamma);
}
void pages_and_attention() {
    auto c=config();QsaState contiguous(c),permuted(c,{7,2,5,0,6,3,1,4});
    for(int i=0;i<21;++i) {append(contiguous,i);append(permuted,i);}
    const std::vector<int32_t> ids{0,3,4,12,20};
    auto a=contiguous.gather(ids,5),b=permuted.gather(ids,5);
    require(a.key==b.key && a.value==b.value,"physical page permutation changed KV values");
    for(uint32_t h=0;h<c.kv_heads;++h) for(size_t i=0;i<ids.size();++i) for(uint32_t d=0;d<c.width;++d) {
        const auto source=float(ids[i]+1)*0.0137f+float(h*c.width+d)*0.021f;
        const auto expected=strata::kernels::f32_from_f16(strata::kernels::f16_from_f32(source));
        require(a.key[(h*ids.size()+i)*c.width+d]==expected,"KV must round to FP16 before attention");
    }
    // Zero queries make every selected value equiprobable. Distinct KV heads
    // detect contiguous GQA mapping versus the incorrect head%kv_heads mapping.
    std::vector<float> query(c.query_heads*c.width);
    auto output=qsa_attention_reference(query,a,c.query_heads,c.kv_heads,c.width);
    for(uint32_t h=0;h<c.query_heads;++h) for(uint32_t d=0;d<c.width;++d) {
        float expected=0;
        for(size_t i=0;i<ids.size();++i) expected+=a.value[((h/3)*ids.size()+i)*c.width+d]/float(ids.size());
        require(std::abs(output[h*c.width+d]-expected)<2e-6f,"GQA attention mismatch");
    }
    const auto padded=contiguous.gather(std::span<const int32_t>(ids).first(3),5);
    require(padded.mask==std::vector<float>({0,0,0,-INFINITY,-INFINITY}),"padding mask");
    auto bad=padded;bad.mask[3]=0;
    rejects([&]{(void)qsa_attention_reference(query,bad,c.query_heads,c.kv_heads,c.width);});
    rejects([&]{(void)contiguous.gather(std::vector<int32_t>{0,0},5);});
    rejects([&]{(void)contiguous.gather(std::vector<int32_t>{21},5);});
    auto empty=contiguous.gather({},0);
    require(qsa_attention_reference(query,empty,c.query_heads,c.kv_heads,c.width)==query,"empty attention must be zero");
    const auto before=permuted.allocated_bytes();std::vector<uint16_t> selected(c.kv_heads*c.width);
    for(auto cell:{0,4,20}) {
        permuted.selected_value_bits(cell,selected);const std::array<int32_t,1> pick{cell};const auto gathered=permuted.gather(pick,1);
        for(size_t i=0;i<selected.size();++i) require(selected[i]==strata::kernels::f16_from_f32(gathered.value[i]),"half selected bits did not match actual FP16 gather input");
    }
    const auto prefix=permuted.value_pool_prefix();
    // Physical page0 belongs to logical page3 in the permutation: the prefix
    // is head0 cells12 and13, NOT logical cell0 and not one cell/all heads.
    for(size_t i=0;i<prefix.size();++i) {
        const auto cell=12+int(i/c.width),dim=int(i%c.width);
        require(prefix[i]==strata::kernels::f16_from_f32(float(cell-dim)*.137f),"half pool prefix invented a logical/cache layout");
    }
    require(permuted.allocated_bytes()==before,"half diagnostic state grew buffers");
    rejects([&]{permuted.selected_value_bits(-1,selected);});rejects([&]{permuted.selected_value_bits(21,selected);});
    rejects([&]{permuted.selected_value_bits(0,std::span(selected).first(selected.size()-1));});
    permuted.reset();rejects([&]{(void)permuted.value_pool_prefix();});
}
void pooling_and_selection() {
    auto c=config();c.position_base=17;QsaState state(c);
    std::vector<float> k(8,0),v(8,0),gamma(4,1),raw{1,2,3,4};
    state.append(k,v,raw,gamma);
    // First spare key deliberately has position zero, independent of base.
    auto expected=raw;rms_norm_rows(expected,4,gamma,c.epsilon);
    close(state.pooled_rows(),expected);
    std::vector<double> mean(4);
    for(uint32_t i=0;i<4;++i) {
        if(i) {raw={float(i+1),float(2*i+2),float(3-i),float(i+4)};state.append(k,v,raw,gamma);}
        for(size_t d=0;d<4;++d) mean[d]+=double(raw[d])/4;
    }
    double sum=0;for(auto x:mean) sum+=x*x;
    std::vector<float> pooled(4);for(size_t d=0;d<4;++d) pooled[d]=float(mean[d]/std::sqrt(sum/4+double(c.epsilon)));
    rotate_neox(pooled,4,rope_row(2,17,c.rope));
    close(state.pooled_rows().first(4),pooled);
    close(state.pooled_rows().subspan(4),expected);
    // Per-head ReLU must precede summation: opposite heads do not cancel.
    std::vector<float> query(8);std::copy(pooled.begin(),pooled.end(),query.begin());
    for(size_t d=0;d<4;++d) query[4+d]=-pooled[d];
    const auto scores=state.cell_scores(query);
    double square=0;for(float x:pooled) square+=double(x)*x;
    for(float score:scores) require(std::abs(score-float(square))<1e-5f,"per-head indexer ReLU");
    for(int i=4;i<9;++i) append(state,i);
    std::fill(query.begin(),query.end(),0);
    // 9 cells, top_k+(block-1)=5; tail is preferred, remaining ties by ID.
    require(state.select(query)==std::vector<int32_t>({0,1,2,3,8}),"stable cell top-k/tail/ascending winners");
    const auto tail=state.cell_scores(query);
    require(tail[0]==0 && tail[8]==1e9f,"finite incomplete tail bias");
    // Finite tail bias is not an unconditional forced inclusion.
    require(state.select(query,std::vector<float>{2e9f,2e9f,0})==std::vector<int32_t>({0,1,2,3,4}),"tail bias incorrectly treated as infinite");
}
void rollback_and_failures() {
    auto c=config();QsaState state(c),replay(c),foreign(c);
    for(int i=0;i<5;++i) {append(state,i);append(replay,i);}
    auto point=state.checkpoint();
    for(int i=5;i<14;++i) append(state,i);
    rejects([&]{foreign.restore(point);});
    state.restore(point);state.restore(point); // same accepted prefix is still an ancestor
    for(int i=5;i<10;++i) {append(state,100+i);append(replay,100+i);}
    close(state.pooled_rows(),replay.pooled_rows(),0);
    std::vector<int32_t> ids(10);std::iota(ids.begin(),ids.end(),0);
    // A small selection budget cannot gather all cells, so compare chunks.
    for(size_t start=0;start<ids.size();start+=5) {
        auto span=std::span<const int32_t>(ids).subspan(start,5);
        auto a=state.gather(span,5),b=replay.gather(span,5);
        require(a.key==b.key && a.value==b.value,"rollback changed accepted prefix/replayed KV");
    }
    const auto before=std::vector<float>(state.pooled_rows().begin(),state.pooled_rows().end());
    std::vector<float> huge(8,1e30f),zero(8),idx(4,1),gamma(4,1);
    rejects([&]{state.append(huge,zero,idx,gamma);});
    require(state.size()==10 && before==std::vector<float>(state.pooled_rows().begin(),state.pooled_rows().end()),"failed append mutated state");
    point=state.checkpoint();state.reset();rejects([&]{state.restore(point);});require(!state.size(),"reset failed");
    rejects([&]{auto bad=c;bad.query_heads=5;(void)QsaState(bad);});
    rejects([&]{(void)QsaState(c,{0,0,1,2,3,4,5,6});});
    for(int n=0;n<32;++n) append(state,n);
    rejects([&]{append(state,33);});
}
void nested_rewinds() {
    const auto c=config();QsaState state(c),reference(c);
    for(int i=0;i<5;++i) {append(state,i);append(reference,i);}
    const auto parent=state.checkpoint();for(int i=5;i<9;++i) append(state,i);
    const auto child=state.checkpoint();for(int i=9;i<14;++i) append(state,i);
    state.restore(child);state.restore(parent);require(state.size()==5,"child rewind invalidated outer QSA window");
    for(int i=5;i<10;++i) {append(state,100+i);append(reference,100+i);}
    rejects([&]{state.restore(child);}); // same length, different generation
    close(state.pooled_rows(),reference.pooled_rows(),0);
    for(int32_t begin=0;begin<10;begin+=5) {
        std::array<int32_t,5> ids{begin,begin+1,begin+2,begin+3,begin+4};
        const auto a=state.gather(ids,5),b=reference.gather(ids,5);require(a.key==b.key && a.value==b.value,"nested rewind corrupted original or rewritten KV");
    }
}
void rope() {
    strata::kernels::RopeScaling scaling;
    for(auto type:{strata::kernels::RopeScalingType::None,strata::kernels::RopeScalingType::Linear,strata::kernels::RopeScalingType::YaRN}) {
        scaling.type=type;scaling.factor=4;scaling.ext_factor=type==strata::kernels::RopeScalingType::YaRN?1:0;
        auto row=rope_row(64,12345,scaling);
        require(row.cosine.size()==32 && row.sine.size()==32,"RoPE table shape");
        std::vector<float> vector(256,1);rotate_neox(vector,256,row);
        for(size_t i=64;i<vector.size();++i) require(vector[i]==1,"RoPE altered unrotated tail");
        if(type!=strata::kernels::RopeScalingType::YaRN) {
            const double angle=12345.0/(type==strata::kernels::RopeScalingType::Linear?4:1);
            require(row.cosine[0]==float(std::cos(angle)) && row.sine[0]==float(std::sin(angle)),"RoPE position/scaling");
        }
    }
}
// Independent INT8 oracle for the storage-only pool: fp16(amax/127) as the scale, rint(x/stored scale) ties to
// even, clamped to [-127,127], decoded as fp16(code*stored scale). Written from kv_q8.hpp's sentence, not from
// kv_cell.hpp, so a mutation of the cell law has to disagree with this file.
struct Int8Row {
    std::vector<int8_t> codes;
    std::vector<uint16_t> scales;
};
Int8Row int8_oracle(std::span<const float> values) {
    Int8Row out;
    out.codes.resize(values.size());
    out.scales.resize(values.size() / 64);
    for (size_t group = 0; group < out.scales.size(); ++group) {
        float amax = 0;
        for (size_t t = 0; t < 64; ++t) amax = std::max(amax, std::fabs(values[group * 64 + t]));
        out.scales[group] = strata::kernels::f16_from_f32(amax / 127.0f);
        const double scale = double(strata::kernels::f32_from_f16(out.scales[group]));
        for (size_t t = 0; t < 64; ++t) {
            int code = 0;
            if (scale > 0) code = std::clamp(int(std::rint(double(values[group * 64 + t]) / scale)), -127, 127);
            out.codes[group * 64 + t] = int8_t(code);
        }
    }
    return out;
}
uint16_t int8_bit(const Int8Row& row, const size_t index) {
    return strata::kernels::f16_from_f32(float(row.codes[index]) * strata::kernels::f32_from_f16(row.scales[index / 64]));
}
QsaConfig int8_config() {
    QsaConfig c;c.query_heads=4;c.kv_heads=2;c.width=64;c.rotary=2;
    c.index_heads=2;c.index_width=8;c.block=4;c.top_k=2;c.capacity=16;c.page_size=4;
    c.kv=KvMode::Int8;return c;
}
void int8_rows(const QsaConfig& c,int n,std::vector<float>& k,std::vector<float>& v) {
    k.resize(c.kv_heads*c.width);v.resize(k.size());
    for(uint32_t head=0;head<c.kv_heads;++head) for(uint32_t d=0;d<c.width;++d) {
        const auto i=uint64_t(head)*c.width+d;
        k[i]=float(((n*37+int(d)*13+int(head)*29)%101)-50)/17.0f;
        v[i]=float(((n*17+int(d)*5+int(head)*11)%71)-35)/9.0f;
        if(d==7) k[i]=float(n+1)*0.9f;   // a per-cell outlier: each cell/head has its own scale
    }
}
void append_int8(QsaState& s,int n) {
    const auto& c=s.config();std::vector<float> k,v,idx(c.index_width,0.5f),gamma(c.index_width,1);
    int8_rows(c,n,k,v);s.append(k,v,idx,gamma);
}
void int8_storage() {
    const auto c=int8_config();
    QsaState state(c),permuted(c,{1,2,3,0}),reference(c);   // logical page3 -> physical page0
    for(int n=0;n<16;++n) {append_int8(state,n);append_int8(permuted,n);append_int8(reference,n);}
    const std::vector<int32_t> ids{0,3,4,12,15};
    const auto got=state.gather(ids,5),perm=permuted.gather(ids,5);
    require(perm.key==got.key && perm.value==got.value,"INT8 page permutation changed KV values");
    // Every gathered value must be the oracle's decode of that cell/head's own group.
    for(uint32_t head=0;head<c.kv_heads;++head) {
        std::vector<Int8Row> oracle_k(ids.size()),oracle_v(ids.size());
        for(size_t item=0;item<ids.size();++item) {
            std::vector<float> k,v;int8_rows(c,ids[item],k,v);
            oracle_k[item]=int8_oracle(std::span<const float>(k).subspan(uint64_t(head)*c.width,c.width));
            oracle_v[item]=int8_oracle(std::span<const float>(v).subspan(uint64_t(head)*c.width,c.width));
        }
        for(size_t item=0;item<ids.size();++item) for(uint32_t d=0;d<c.width;++d) {
            const auto at=(uint64_t(head)*ids.size()+item)*c.width+d;
            require(got.key[at]==strata::kernels::f32_from_f16(int8_bit(oracle_k[item],d)),"INT8 K differs from the storage law");
            require(got.value[at]==strata::kernels::f32_from_f16(int8_bit(oracle_v[item],d)),"INT8 V differs from the storage law");
        }
    }
    // The half-dump accessors expose the same decoded bits the attention scratch sees.
    std::vector<uint16_t> selected(uint64_t(c.kv_heads)*c.width);
    for(const int32_t cell:{0,4,12,15}) {
        state.selected_value_bits(cell,selected);
        std::vector<float> k,v;int8_rows(c,cell,k,v);
        for(uint32_t head=0;head<c.kv_heads;++head) {
            const auto row=int8_oracle(std::span<const float>(v).subspan(uint64_t(head)*c.width,c.width));
            for(uint32_t d=0;d<c.width;++d) require(selected[uint64_t(head)*c.width+d]==int8_bit(row,d),"INT8 selected V bits differ from the law");
        }
    }
    const auto before=permuted.allocated_bytes();const auto prefix=permuted.value_pool_prefix();
    // Physical page0 belongs to logical page3: the first kv_heads*width values are head0's cells 12 and 13.
    std::vector<float> k12,v12;int8_rows(c,12,k12,v12);
    const auto row12=int8_oracle(std::span<const float>(v12).first(c.width));
    for(uint32_t i=0;i<c.width;++i) require(prefix[i]==int8_bit(row12,i),"INT8 pool prefix invented a logical/cache layout");
    std::vector<float> k13,v13;int8_rows(c,13,k13,v13);
    const auto row13=int8_oracle(std::span<const float>(v13).first(c.width));
    for(uint32_t i=0;i<c.width;++i) require(prefix[c.width+i]==int8_bit(row13,i),"INT8 pool prefix second cell");
    require(permuted.allocated_bytes()==before,"INT8 half diagnostic state grew buffers");
    // The constructor's exact charge: codes (capacity/heads/width, one byte) + scales (one uint16 per 64 values)
    // + the fixed prefix buffer and the indexer/pages arrays.
    require(state.allocated_bytes()==4920,"INT8 QSA host byte bound");
    // Snapshot parts are upstream's order and byte counts: K codes, V codes, K scales, V scales, pooled rows.
    const auto full=reference.snapshot(),state_full=state.snapshot();uint64_t sizes[5]{};
    reference.snapshot_sizes(full,sizes);
    require(sizes[0]==16*2*64 && sizes[1]==16*2*64 && sizes[2]==16*2*2 && sizes[3]==16*2*2 && sizes[4]==(16/4+1)*8*4,
            "INT8 parked part sizes");
    reference.reset();reference.restore(full);
    for(const int32_t first:{0,5,10}) {
        std::vector<int32_t> part;for(int32_t cell=first;cell<std::min(first+5,16);++cell) part.push_back(cell);
        const auto a=reference.gather(part,5),b=state.gather(part,5);require(a.key==b.key && a.value==b.value,"INT8 snapshot restore changed KV");
    }
    // Grow-in-place reuse: the retained 8-cell image is grown to 16 with the live cells' codes and scales.
    QsaState live(c),expected(c);
    for(int n=0;n<8;++n) {append_int8(live,n);append_int8(expected,n);}
    auto prior=live.snapshot();
    for(int n=8;n<16;++n) {append_int8(live,n);append_int8(expected,n);}
    auto reused=live.snapshot_reused(std::move(prior),8,2);
    live.reset();live.restore(reused);
    const auto live_kv=live.gather(ids,5),expected_kv=expected.gather(ids,5);
    require(live_kv.key==expected_kv.key && live_kv.value==expected_kv.value,"INT8 reused snapshot differs from the live pool");
    close(live.pooled_rows(),expected.pooled_rows(),0);
    QsaState short_prior(c);
    append_int8(short_prior,0);append_int8(short_prior,1);append_int8(short_prior,2);append_int8(short_prior,3);
    auto kept=short_prior.snapshot();append_int8(short_prior,4);
    rejects([&]{ (void)short_prior.snapshot_reused(std::move(kept),1,3); });   // only 4/4+1 = 2 pooled rows exist
    const auto path=std::filesystem::temp_directory_path()/("strata-hpu-qsa-int8-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())+".bin");
    const auto f16path=std::filesystem::temp_directory_path()/("strata-hpu-qsa-int8-f16-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())+".bin");
    struct Cleanup {std::filesystem::path a,b;~Cleanup(){std::error_code error;std::filesystem::remove(a,error);std::filesystem::remove(b,error);}} cleanup{path,f16path};
    {StateWriter writer(path.string(),1<<20);state.write_snapshot(writer,state_full);writer.finish();}
    QsaState imported(c,{2,0,3,1});
    {StateReader reader(path.string(),1<<20);const auto loaded=imported.read_snapshot(reader,16);reader.finish();imported.restore(loaded);}
    for(const int32_t first:{0,5,10}) {
        std::vector<int32_t> part;for(int32_t cell=first;cell<std::min(first+5,16);++cell) part.push_back(cell);
        const auto a=imported.gather(part,5),b=state.gather(part,5);require(a.key==b.key && a.value==b.value,"INT8 disk restoration changed KV");
    }
    // The parked format tag keeps the two images apart in both directions.
    auto f16c=int8_config();f16c.kv=KvMode::Fp16;QsaState f16(f16c);
    std::vector<float> fk(uint64_t(c.kv_heads)*c.width,0.25f),fv(fk.size(),-0.5f),fidx(c.index_width,0.1f),fgamma(c.index_width,1);
    for(int n=0;n<4;++n) f16.append(fk,fv,fidx,fgamma);
    const auto f16full=f16.snapshot();
    {StateWriter writer(f16path.string(),1<<20);f16.write_snapshot(writer,f16full);writer.finish();}
    {StateReader reader(f16path.string(),1<<20);rejects([&]{ (void)imported.read_snapshot(reader,4); });}
    {StateReader reader(path.string(),1<<20);rejects([&]{ (void)f16.read_snapshot(reader,16); });}
    // The int8 append validates like the FP16 one: a non-finite row publishes nothing.
    auto bad_k=fk,bad_v=fv;bad_k[3]=NAN;
    const auto count=state.size();
    rejects([&]{state.append(bad_k,bad_v,fidx,fgamma);});require(state.size()==count,"failed INT8 append published a cell");
    auto bad=int8_config();bad.width=4;
    rejects([&]{ (void)QsaState(bad); });
}
// Independent Q4_0 oracle: the Hadamard transform and the ggml q4_0 group law from src/kernels/cuda/kv_q4.cu:
// 78-91 - largest |x| with the larger signed value breaking ties, the unrounded 1/d, truncation toward zero, and
// elements t / t+16 in the low / high nibble. q4 code choices flip on near-zero rotated blocks, so the rotation
// below repeats upstream's FP32 butterfly order to predict the stored bytes exactly; hadamard_signs() separately
// checks that butterfly against the mathematical sign-sum matrix on a well-conditioned row.
struct Q4Row { std::array<uint8_t,144> stored{}; std::array<uint16_t,256> rotated{}; };
std::vector<float> hadamard256(std::span<const float> row) {
    require(row.size()==256,"Q4 rotation oracle shape");
    std::vector<float> out(row.begin(),row.end());
    for(auto& x:out) x*=1.0f/16.0f;
    for(uint32_t h=1;h<256;h*=2)
        for(uint32_t i=0;i<256;i+=2*h)
            for(uint32_t j=0;j<h;++j) {
                const float x=out[i+j],y=out[i+j+h];
                out[i+j]=x+y;out[i+j+h]=x-y;
            }
    return out;
}
std::vector<float> hadamard_signs(std::span<const float> row) {
    std::vector<float> out(256);
    for(uint32_t k=0;k<256;++k) {
        double sum=0;
        for(uint32_t j=0;j<256;++j) sum+=std::popcount(k&j)%2?-double(row[j]):double(row[j]);
        out[k]=float(sum/16.0);
    }
    return out;
}
Q4Row q4_oracle(std::span<const float> row) {
    const auto rotated=hadamard256(row);
    Q4Row out;
    for(uint32_t b=0;b<8;++b) {
        const auto* values=rotated.data()+b*32;
        float amax=0,mval=0;bool first=true;
        for(uint32_t t=0;t<32;++t) {
            const float x=values[t];require(std::isfinite(x),"Q4 oracle needs finite values");
            const float a=std::fabs(x);
            if(first || a>amax || (a==amax && x>mval)) {amax=a;mval=x;first=false;}
        }
        const float d=mval/-8.0f,id=d!=0?1.0f/d:0.0f;
        const uint16_t scale=strata::kernels::f16_from_f32(d);
        const float stored=strata::kernels::f32_from_f16(scale);
        out.stored[b*18]=uint8_t(scale&0xff);out.stored[b*18+1]=uint8_t(scale>>8);
        std::array<uint8_t,32> q{};
        for(uint32_t t=0;t<32;++t) {
            const float scaled=values[t]*id+8.5f;
            int code=0;
            if(std::isfinite(scaled) && scaled>float(std::numeric_limits<int>::min()) &&
               scaled<float(std::numeric_limits<int>::max())) code=int(std::trunc(scaled));
            q[t]=uint8_t(code<0?0:(code>15?15:code));
            out.rotated[b*32+t]=strata::kernels::f16_from_f32(float(q[t]-8)*stored);
        }
        for(uint32_t j=0;j<16;++j) out.stored[b*18+2+j]=uint8_t(q[j]|(q[j+16]<<4));
    }
    return out;
}
std::vector<float> q4_unrotated(const Q4Row& row) {
    std::array<float,256> decoded{};
    for(uint32_t i=0;i<256;++i) decoded[i]=strata::kernels::f32_from_f16(row.rotated[i]);
    return hadamard256(decoded);
}
QsaConfig q4_config() {
    QsaConfig c;c.query_heads=4;c.kv_heads=2;c.width=256;c.rotary=2;
    c.index_heads=2;c.index_width=8;c.block=4;c.top_k=2;c.capacity=8;c.page_size=4;
    c.kv=KvMode::Q4;return c;
}
void q4_rows(const QsaConfig& c,int n,std::vector<float>& k,std::vector<float>& v) {
    k.resize(c.kv_heads*c.width);v.resize(k.size());
    for(uint32_t head=0;head<c.kv_heads;++head) for(uint32_t d=0;d<c.width;++d) {
        const auto i=uint64_t(head)*c.width+d;
        k[i]=float(((n*37+int(d)*13+int(head)*29)%101)-50)/17.0f;
        v[i]=float(((n*17+int(d)*5+int(head)*11)%71)-35)/9.0f;
        if(d==7) k[i]=float(n+1)*0.9f;   // a per-cell outlier inside the first block
    }
}
void append_q4(QsaState& s,int n) {
    const auto& c=s.config();std::vector<float> k,v,idx(c.index_width,0.5f),gamma(c.index_width,1);
    q4_rows(c,n,k,v);s.append(k,v,idx,gamma);
}
void q4_storage() {
    const auto c=q4_config();
    { // The FP32 butterfly must still be the Hadamard matrix: agree with the sign-sum form.
        std::vector<float> k,v;q4_rows(c,2,k,v);
        const auto probe=std::span<const float>(k).first(c.width);
        const auto butterfly=hadamard256(probe),signs=hadamard_signs(probe);
        for(uint32_t i=0;i<256;++i)
            require(std::abs(butterfly[i]-signs[i])<=2e-5f*(1+std::abs(signs[i])),"Q4 rotation is not the Hadamard matrix");
    }
    QsaState state(c),permuted(c,{1,0}),reference(c);   // logical page1 -> physical page0
    for(int n=0;n<8;++n) {append_q4(state,n);append_q4(permuted,n);append_q4(reference,n);}
    const std::vector<int32_t> ids{0,3,4,7};
    constexpr uint32_t bucket=5;
    const auto got=state.gather(ids,bucket),perm=permuted.gather(ids,bucket);
    require(perm.key==got.key && perm.value==got.value,"Q4 page permutation changed KV values");
    for(uint32_t head=0;head<c.kv_heads;++head) for(size_t item=0;item<ids.size();++item) {
        std::vector<float> k,v;q4_rows(c,ids[item],k,v);
        const auto want_k=q4_unrotated(q4_oracle(std::span<const float>(k).subspan(uint64_t(head)*c.width,c.width)));
        const auto want_v=q4_unrotated(q4_oracle(std::span<const float>(v).subspan(uint64_t(head)*c.width,c.width)));
        for(uint32_t d=0;d<c.width;++d) {
            const auto at=(uint64_t(head)*bucket+item)*c.width+d;
            if(got.key[at]!=want_k[d]) throw std::runtime_error("Q4 K differs at head "+std::to_string(head)+" cell "+std::to_string(ids[item])+" dim "+std::to_string(d)+": "+std::to_string(got.key[at])+" vs "+std::to_string(want_k[d]));
            if(got.value[at]!=want_v[d]) throw std::runtime_error("Q4 V differs at head "+std::to_string(head)+" cell "+std::to_string(ids[item])+" dim "+std::to_string(d)+": "+std::to_string(got.value[at])+" vs "+std::to_string(want_v[d]));
        }
    }
    for(uint32_t head=0;head<c.kv_heads;++head) for(uint32_t d=0;d<c.width;++d)
        require(got.key[(uint64_t(head)*bucket+ids.size())*c.width+d]==0 &&
                got.value[(uint64_t(head)*bucket+ids.size())*c.width+d]==0,"Q4 gather padding is not zero");
    // The half-dump accessors expose the decoded rotated rows: exactly the bits kv_gather_q4_kernel writes.
    std::vector<uint16_t> selected(uint64_t(c.kv_heads)*c.width);
    for(const int32_t cell:{0,4,7}) {
        state.selected_value_bits(cell,selected);
        std::vector<float> k,v;q4_rows(c,cell,k,v);
        for(uint32_t head=0;head<c.kv_heads;++head) {
            const auto row=q4_oracle(std::span<const float>(v).subspan(uint64_t(head)*c.width,c.width));
            for(uint32_t d=0;d<c.width;++d)
                require(selected[uint64_t(head)*c.width+d]==row.rotated[d],"Q4 selected V bits differ from the law");
        }
    }
    const auto before=permuted.allocated_bytes();const auto prefix=permuted.value_pool_prefix();
    // Physical page0 is logical page1: head0's first kv_heads storage rows are cells 4 and 5, still rotated.
    for(uint32_t slot=0;slot<c.kv_heads;++slot) {
        std::vector<float> k,v;q4_rows(c,4+int(slot),k,v);
        const auto row=q4_oracle(std::span<const float>(v).first(c.width));
        for(uint32_t d=0;d<c.width;++d)
            require(prefix[uint64_t(slot)*c.width+d]==row.rotated[d],"Q4 pool prefix invented a layout or unrotated early");
    }
    require(permuted.allocated_bytes()==before,"Q4 half diagnostic state grew buffers");
    // Two page-rounded pools (2 pages * 4 slots * 2 heads * 144 B), the 512-value prefix buffer and the indexer.
    require(state.allocated_bytes()==5936,"Q4 QSA host byte bound");
    // Parked parts: K and V blocks, no separate scales, the indexer rows.
    const auto full=reference.snapshot();uint64_t sizes[5]{};
    reference.snapshot_sizes(full,sizes);
    require(sizes[0]==8*2*144 && sizes[1]==8*2*144 && sizes[2]==0 && sizes[3]==0 && sizes[4]==(8/4+1)*8*4,
            "Q4 parked part sizes");
    reference.reset();reference.restore(full);
    for(const int32_t first:{0,4}) {
        std::vector<int32_t> part;for(int32_t cell=first;cell<std::min(first+4,8);++cell) part.push_back(cell);
        const auto a=reference.gather(part,5),b=state.gather(part,5);
        require(a.key==b.key && a.value==b.value,"Q4 snapshot restore changed KV");
    }
    // Grow-in-place: the retained 4-cell image is grown to 8 with the live blocks.
    QsaState live(c),expected(c);
    for(int n=0;n<4;++n) {append_q4(live,n);append_q4(expected,n);}
    auto prior=live.snapshot();
    for(int n=4;n<8;++n) {append_q4(live,n);append_q4(expected,n);}
    auto reused=live.snapshot_reused(std::move(prior),4,1);
    live.reset();live.restore(reused);
    const auto live_kv=live.gather(ids,5),expected_kv=expected.gather(ids,5);
    require(live_kv.key==expected_kv.key && live_kv.value==expected_kv.value,"Q4 reused snapshot differs from the live pool");
    close(live.pooled_rows(),expected.pooled_rows(),0);
    const auto path=std::filesystem::temp_directory_path()/("strata-hpu-qsa-q4-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())+".bin");
    const auto f16path=std::filesystem::temp_directory_path()/("strata-hpu-qsa-q4-f16-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())+".bin");
    struct Cleanup {std::filesystem::path a,b;~Cleanup(){std::error_code error;std::filesystem::remove(a,error);std::filesystem::remove(b,error);}} cleanup{path,f16path};
    {StateWriter writer(path.string(),1<<20);reference.write_snapshot(writer,full);writer.finish();}
    QsaState imported(c,{0,1});
    {StateReader reader(path.string(),1<<20);const auto loaded=imported.read_snapshot(reader,8);reader.finish();imported.restore(loaded);}
    for(const int32_t first:{0,4}) {
        std::vector<int32_t> part;for(int32_t cell=first;cell<std::min(first+4,8);++cell) part.push_back(cell);
        const auto a=imported.gather(part,5),b=state.gather(part,5);
        require(a.key==b.key && a.value==b.value,"Q4 disk restoration changed KV");
    }
    // The parked format tag keeps Q4 and FP16 images apart in both directions at the same geometry.
    auto f16c=q4_config();f16c.kv=KvMode::Fp16;QsaState f16(f16c);
    std::vector<float> fk(uint64_t(c.kv_heads)*c.width,0.25f),fv(fk.size(),-0.5f),fidx(c.index_width,0.1f),fgamma(c.index_width,1);
    for(int n=0;n<8;++n) f16.append(fk,fv,fidx,fgamma);
    const auto f16full=f16.snapshot();
    {StateWriter writer(f16path.string(),1<<20);f16.write_snapshot(writer,f16full);writer.finish();}
    {StateReader reader(f16path.string(),1<<20);rejects([&]{ (void)imported.read_snapshot(reader,8); });}
    {StateReader reader(path.string(),1<<20);rejects([&]{ (void)f16.read_snapshot(reader,8); });}
    // Non-finite input and a non-256 geometry are refused before publishing.
    auto bad_k=fk,bad_v=fv;bad_k[3]=NAN;
    const auto count=state.size();
    rejects([&]{state.append(bad_k,bad_v,fidx,fgamma);});require(state.size()==count,"failed Q4 append published a cell");
    auto bad=q4_config();bad.width=64;
    rejects([&]{ (void)QsaState(bad); });
}
// Hybrid K8V4: K is the unrotated INT8 group pair and V the rotated Q4_0 half. The oracle composes the two
// independent ones above; a rotated K or an unrotated V has to disagree with this file. Written from the
// upstream split (src/core/layer.cpp:649-661,1080-1103), not from qsa_state.cpp.
QsaConfig hybrid_config() {
    auto c=q4_config();c.kv=KvMode::K8V4;return c;
}
void append_hybrid(QsaState& s,int n) {
    const auto& c=s.config();std::vector<float> k,v,idx(c.index_width,0.5f),gamma(c.index_width,1);
    q4_rows(c,n,k,v);s.append(k,v,idx,gamma);
}
void hybrid_storage() {
    const auto c=hybrid_config();
    require(QsaState(c).allocated_bytes()==7856,"K8V4 QSA host byte bound");
    QsaState state(c),permuted(c,{1,0}),reference(c);   // logical page1 -> physical page0
    for(int n=0;n<8;++n) {append_hybrid(state,n);append_hybrid(permuted,n);append_hybrid(reference,n);}
    const std::vector<int32_t> ids{0,3,4,7};
    constexpr uint32_t bucket=5;
    const auto got=state.gather(ids,bucket),perm=permuted.gather(ids,bucket);
    require(perm.key==got.key && perm.value==got.value,"K8V4 page permutation changed KV values");
    for(uint32_t head=0;head<c.kv_heads;++head) for(size_t item=0;item<ids.size();++item) {
        std::vector<float> k,v;q4_rows(c,ids[item],k,v);
        const auto want_k=int8_oracle(std::span<const float>(k).subspan(uint64_t(head)*c.width,c.width));
        const auto want_v=q4_unrotated(q4_oracle(std::span<const float>(v).subspan(uint64_t(head)*c.width,c.width)));
        for(uint32_t d=0;d<c.width;++d) {
            const auto at=(uint64_t(head)*bucket+item)*c.width+d;
            require(got.key[at]==strata::kernels::f32_from_f16(int8_bit(want_k,d)),"K8V4 K differs from the unrotated INT8 half");
            require(got.value[at]==want_v[d],"K8V4 V differs from the rotated Q4_0 half");
        }
    }
    for(uint32_t head=0;head<c.kv_heads;++head) for(uint32_t d=0;d<c.width;++d)
        require(got.key[(uint64_t(head)*bucket+ids.size())*c.width+d]==0 &&
                got.value[(uint64_t(head)*bucket+ids.size())*c.width+d]==0,"K8V4 gather padding is not zero");
    // Half-dump accessors expose the rotated V bits; K is not part of them (upstream's V pool only).
    std::vector<uint16_t> selected(uint64_t(c.kv_heads)*c.width);
    for(const int32_t cell:{0,4,7}) {
        state.selected_value_bits(cell,selected);
        std::vector<float> k,v;q4_rows(c,cell,k,v);
        for(uint32_t head=0;head<c.kv_heads;++head) {
            const auto row=q4_oracle(std::span<const float>(v).subspan(uint64_t(head)*c.width,c.width));
            for(uint32_t d=0;d<c.width;++d)
                require(selected[uint64_t(head)*c.width+d]==row.rotated[d],"K8V4 selected V bits differ from the rotated half");
        }
    }
    const auto before=permuted.allocated_bytes();const auto prefix=permuted.value_pool_prefix();
    // Physical page0 is logical page1: head0's first kv_heads storage rows are cells 4 and 5, still rotated.
    for(uint32_t slot=0;slot<c.kv_heads;++slot) {
        std::vector<float> k,v;q4_rows(c,4+int(slot),k,v);
        const auto row=q4_oracle(std::span<const float>(v).first(c.width));
        for(uint32_t d=0;d<c.width;++d)
            require(prefix[uint64_t(slot)*c.width+d]==row.rotated[d],"K8V4 pool prefix invented a layout or unrotated early");
    }
    require(permuted.allocated_bytes()==before,"K8V4 half diagnostic state grew buffers");
    // Parked parts in upstream order: K codes, V q4_0 blocks, K scales, no V scales, pooled rows.
    const auto full=reference.snapshot();uint64_t sizes[5]{};
    reference.snapshot_sizes(full,sizes);
    require(sizes[0]==8*2*256 && sizes[1]==8*2*144 && sizes[2]==8*2*4*2 && sizes[3]==0 && sizes[4]==(8/4+1)*8*4,
            "K8V4 parked part sizes");
    reference.reset();reference.restore(full);
    for(const int32_t first:{0,4}) {
        std::vector<int32_t> part;for(int32_t cell=first;cell<std::min(first+4,8);++cell) part.push_back(cell);
        const auto a=reference.gather(part,5),b=state.gather(part,5);
        require(a.key==b.key && a.value==b.value,"K8V4 snapshot restore changed KV");
    }
    // Grow-in-place: the retained 4-cell image is grown to 8 with both halves' live cells.
    QsaState live(c),expected(c);
    for(int n=0;n<4;++n) {append_hybrid(live,n);append_hybrid(expected,n);}
    auto prior=live.snapshot();
    for(int n=4;n<8;++n) {append_hybrid(live,n);append_hybrid(expected,n);}
    auto reused=live.snapshot_reused(std::move(prior),4,1);
    live.reset();live.restore(reused);
    const auto live_kv=live.gather(ids,5),expected_kv=expected.gather(ids,5);
    require(live_kv.key==expected_kv.key && live_kv.value==expected_kv.value,"K8V4 reused snapshot differs from the live pool");
    close(live.pooled_rows(),expected.pooled_rows(),0);
    const auto path=std::filesystem::temp_directory_path()/("strata-hpu-qsa-k8v4-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())+".bin");
    const auto q4path=std::filesystem::temp_directory_path()/("strata-hpu-qsa-k8v4-q4-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())+".bin");
    struct Cleanup {std::filesystem::path a,b;~Cleanup(){std::error_code error;std::filesystem::remove(a,error);std::filesystem::remove(b,error);}} cleanup{path,q4path};
    {StateWriter writer(path.string(),1<<20);reference.write_snapshot(writer,full);writer.finish();}
    // The header the writer actually put on disk, field by field: upstream's put_kv_header list
    // (src/core/conversation_file.cpp:676-678) with kKvHybrid as the format tag and no +16, because that tag
    // already means rotated (src/kernels/kv_stream.hpp:45-57). A writer that priced hybrid as plain INT8 or Q4
    // is caught here rather than by a reader that made the same mistake.
    {StateReader header(path.string(),1<<20);
     const auto shapes=strata::kernels::qsa_real_shapes();
     require(header.integer()==uint64_t(strata::kernels::kKvHybrid),"K8V4 parked format tag");
     require(header.integer()==8,"K8V4 parked cells field");
     require(header.integer()==uint64_t(c.kv_heads) && header.integer()==uint64_t(c.width),"K8V4 parked head geometry");
     require(header.integer()==uint64_t(shapes.page_size),"K8V4 parked page size from the pinned shapes");
     require(header.integer()==8/uint64_t(c.block)+1,"K8V4 parked pooled rows");
     require(header.integer()==uint64_t(c.index_width),"K8V4 parked index dim");}
    QsaState imported(c,{0,1});
    {StateReader reader(path.string(),1<<20);const auto loaded=imported.read_snapshot(reader,8);reader.finish();imported.restore(loaded);}
    for(const int32_t first:{0,4}) {
        std::vector<int32_t> part;for(int32_t cell=first;cell<std::min(first+4,8);++cell) part.push_back(cell);
        const auto a=imported.gather(part,5),b=state.gather(part,5);
        require(a.key==b.key && a.value==b.value,"K8V4 disk restoration changed KV");
    }
    // The parked format tag keeps hybrid and q4_0 (same geometry, same width) apart in both directions.
    auto q4c=hybrid_config();q4c.kv=KvMode::Q4;QsaState q4(q4c);
    for(int n=0;n<8;++n) append_q4(q4,n);
    const auto q4full=q4.snapshot();
    {StateWriter writer(q4path.string(),1<<20);q4.write_snapshot(writer,q4full);writer.finish();}
    {StateReader reader(q4path.string(),1<<20);rejects([&]{ (void)imported.read_snapshot(reader,8); });}
    {StateReader reader(path.string(),1<<20);rejects([&]{ (void)q4.read_snapshot(reader,8); });}
    // Non-finite input and non-256 geometry are refused before publishing.
    std::vector<float> fk(uint64_t(c.kv_heads)*c.width,0.25f),fv(fk.size(),-0.5f),fidx(c.index_width,0.1f),fgamma(c.index_width,1);
    auto bad_k=fk,bad_v=fv;bad_k[3]=NAN;
    const auto count=state.size();
    rejects([&]{state.append(bad_k,bad_v,fidx,fgamma);});require(state.size()==count,"failed K8V4 append published a cell");
    for(const uint32_t width:{64u,128u}) {auto bad=hybrid_config();bad.width=width;rejects([&]{ (void)QsaState(bad); });}
}
void all_finite_half_patterns() {
    QsaConfig c;c.capacity=128;QsaState state(c);
    std::vector<float> k(512),v(512),index(128),gamma(128,1);
    for(unsigned base=0;base<63488;base+=512) {
        for(unsigned i=0;i<512;++i) {
            const auto n=base+i;const uint16_t bits=uint16_t(n<31744?n:(n-31744)|0x8000);
            k[i]=strata::kernels::f32_from_f16(bits);v[511-i]=k[i];
        }
        state.append(k,v,index,gamma);const int32_t id=int32_t(state.size()-1);
        const auto out=state.gather(std::span<const int32_t>(&id,1),1);
        for(unsigned i=0;i<512;++i)
            require(std::bit_cast<uint32_t>(out.key[i])==std::bit_cast<uint32_t>(k[i]) &&
                    std::bit_cast<uint32_t>(out.value[i])==std::bit_cast<uint32_t>(v[i]),"FP16 widening changed finite bits");
    }
    // Nonmonotone public gathers retain caller order, including the fallback's duplicate rejection.
    const std::vector<int32_t> ids{3,1,2};const auto out=state.gather(ids,3);
    for(size_t i=0;i<ids.size();++i) {
        const auto single=state.gather(std::span<const int32_t>(&ids[i],1),1);
        for(unsigned head=0;head<2;++head) for(unsigned d=0;d<256;++d)
            require(std::bit_cast<uint32_t>(out.key[(head*3+i)*256+d])==std::bit_cast<uint32_t>(single.key[head*256+d]),"gather changed nonmonotone order");
    }
    rejects([&]{(void)state.gather(std::vector<int32_t>{3,1,3},3);});
}
void half_rounding_boundaries() {
    // Every finite positive half interval's exact midpoint and its adjacent f32 values, both signs.
    // These include signed zero, half subnormals, carries into the next exponent and tie-to-even.
    std::vector<float> probes;
    for(uint32_t bits=0;bits<0x7bff;++bits) {
        const float a=strata::kernels::f32_from_f16(uint16_t(bits));
        const float b=strata::kernels::f32_from_f16(uint16_t(bits+1));
        const float midpoint=float((double(a)+double(b))*0.5);
        for(const float x:{std::nextafter(midpoint,-INFINITY),midpoint,std::nextafter(midpoint,INFINITY)}) {
            probes.push_back(x);probes.push_back(-x);
        }
    }
    QsaConfig c;c.capacity=512;QsaState state(c);
    std::vector<float> k(512),v(512),index(128),gamma(128,1);
    for(size_t start=0;start<probes.size();start+=k.size()) {
        for(size_t i=0;i<k.size();++i) {k[i]=probes[std::min(start+i,probes.size()-1)];v[k.size()-1-i]=-k[i];}
        state.append(k,v,index,gamma);const int32_t id=int32_t(state.size()-1);
        const auto out=state.gather(std::span<const int32_t>(&id,1),1);
        for(size_t i=0;i<k.size();++i) {
            const auto expected_k=strata::kernels::f32_from_f16(strata::kernels::f16_from_f32(k[i]));
            const auto expected_v=strata::kernels::f32_from_f16(strata::kernels::f16_from_f32(v[i]));
            require(std::bit_cast<uint32_t>(out.key[i])==std::bit_cast<uint32_t>(expected_k) &&
                    std::bit_cast<uint32_t>(out.value[i])==std::bit_cast<uint32_t>(expected_v),"FP16 midpoint rounding changed KV bits");
        }
    }
    // Overflow is refused atomically even when it occurs in a later SIMD lane or the V row.
    const auto before=state.size();
    std::fill(k.begin(),k.end(),0);std::fill(v.begin(),v.end(),0);
    k[511]=65520.0f;rejects([&]{state.append(k,v,index,gamma);});require(state.size()==before,"FP16 K overflow published cache");
    k[511]=65504.0f;v[509]=-65520.0f;rejects([&]{state.append(k,v,index,gamma);});require(state.size()==before,"FP16 V overflow published cache");
}
void persistent_snapshots() {    const auto c=config();QsaState state(c,{7,6,5,4,3,2,1,0}),reference(c),foreign(c);
    for(int i=0;i<11;++i) {append(state,i);append(reference,i);}
    const auto snapshot=state.snapshot();const auto stale=state.checkpoint();
    const auto expected_pool=std::vector<float>(reference.pooled_rows().begin(),reference.pooled_rows().end());
    auto compare=[&] {
        require(state.size()==11,"snapshot position restore");close(state.pooled_rows(),expected_pool,0);
        for(int32_t first=0;first<11;first+=5) {
            std::vector<int32_t> ids;for(int32_t cell=first;cell<std::min(first+5,11);++cell) ids.push_back(cell);
            const auto a=state.gather(ids,5),b=reference.gather(ids,5);
            require(a.key==b.key && a.value==b.value,"snapshot changed KV through physical page permutation");
        }
    };
    for(int branch=0;branch<3;++branch) {
        state.reset();for(int i=0;i<16;++i) append(state,100+branch*20+i);
        state.restore(snapshot);compare();rejects([&]{state.restore(stale);});
        for(int i=11;i<20;++i) append(state,300+i);
    }
    state.reset();state.restore(snapshot);compare();rejects([&]{foreign.restore(snapshot);});
    state.reset();const auto empty=state.snapshot();append(state,7);state.restore(empty);require(!state.size() && state.pooled_rows().empty(),"empty snapshot");
    const auto path=std::filesystem::temp_directory_path()/("strata-hpu-qsa-state-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())+".bin");
    struct Cleanup {std::filesystem::path path;~Cleanup(){std::error_code error;std::filesystem::remove(path,error);}} cleanup{path};
    {StateWriter writer(path.string(),65536);state.write_snapshot(writer,snapshot);writer.finish();}
    // Disk parsing re-owns state for a fresh instance and a different physical
    // page map; the caller verifies the complete checksum before commit.
    QsaState imported(c);
    {StateReader reader(path.string(),65536);const auto loaded=imported.read_snapshot(reader,11);reader.finish();imported.restore(loaded);}
    close(imported.pooled_rows(),expected_pool,0);
    for(int32_t first=0;first<11;first+=5) {
        std::vector<int32_t> ids;for(int32_t cell=first;cell<std::min(first+5,11);++cell) ids.push_back(cell);
        const auto a=imported.gather(ids,5),b=reference.gather(ids,5);require(a.key==b.key && a.value==b.value,"serialized KV page-map independence");
    }
    rejects([&]{StateReader reader(path.string(),65536);(void)imported.read_snapshot(reader,10);});
}
}
int main() {
    try {
        pages_and_attention();pooling_and_selection();rollback_and_failures();nested_rewinds();rope();persistent_snapshots();int8_storage();q4_storage();hybrid_storage();all_finite_half_patterns();half_rounding_boundaries();
        QsaState target(QsaConfig{});
        std::cout<<"PASS QSA FP16 paged KV, contiguous GQA, pooling/RoPE, per-head ReLU, finite tail bias, stable top-k\n"
                   "PASS checkpoint across pooling boundaries, replay, reset, page permutation, atomic append failures\n"
                   "PASS QSA INT8 paged KV: stored-scale decode against an independent oracle, half-dump bits, snapshot parts,\n"
                   "     grow-in-place reuse, disk restoration, F16/INT8 format-tag separation, atomic non-finite refusal\n"
                   "PASS QSA Q4_0 paged KV: Hadamard + q4_0 decode against an independent sign-sum oracle, rotated half-dump\n"
                   "     bits, snapshot parts, grow-in-place reuse, disk restoration, F16/Q4 shape separation, geometry refusal\n"
                   "PASS QSA K8V4 paged KV: unrotated INT8 K + rotated Q4_0 V against the composed independent oracles,\n"
                   "     rotated V half-dump bits, [K codes, V blocks, K scales, 0, pooled] parts, the put_kv_header fields\n"
                   "     read back off the published file (kKvHybrid tag, no +16), grow-in-place reuse, disk restoration,\n"
                   "     K8V4/Q4 format-tag separation, non-finite and non-256 refusal\n"
                 <<"target_capacity="<<target.config().capacity<<" cpu_state_bytes="<<target.allocated_bytes()<<'\n';
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
