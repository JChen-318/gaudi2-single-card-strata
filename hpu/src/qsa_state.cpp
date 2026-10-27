#include "strata/hpu/qsa_state.hpp"
#include "strata/hpu/qsa_index_exact.hpp"
#include <optional>
#include "strata/hpu/kv_cell.hpp"
#include "strata/hpu/snapshot_memory.hpp"
#include "strata/hpu/performance_modes.hpp"
#include "strata/kernels/f16_bits.hpp"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <exception>
#include <numeric>
#include <set>
#include <thread>
#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#include <immintrin.h>
#endif

namespace strata::hpu {
namespace {
std::atomic<uint64_t> next_owner{1};
// Every FP16 bit pattern's f32_from_f16 value, built once: the gather converts each selected K/V row through
// this table instead of the scalar bit-manipulation, with identical results.
const float* f16_table() {
    static const std::vector<float> table=[] {
        std::vector<float> values(65536);
        for(uint32_t bits=0;bits<65536;++bits) values[bits]=strata::kernels::f32_from_f16(uint16_t(bits));
        return values;
    }();
    return table.data();
}
void widen_f16_table(const uint16_t* input,float* output,size_t count) {
    const auto* table=f16_table();for(size_t i=0;i<count;++i) output[i]=table[input[i]];
}
#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
__attribute__((target("avx,f16c")))
void widen_f16_native(const uint16_t* input,float* output,size_t count) {
    size_t i=0;
    for(;i+8<=count;i+=8) _mm256_storeu_ps(output+i,_mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(input+i))));
    if(i<count) widen_f16_table(input+i,output+i,count-i);
}
#endif
using F16Widener=void (*)(const uint16_t*,float*,size_t);
F16Widener f16_widener() {
#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
    static const F16Widener function=__builtin_cpu_supports("avx") && __builtin_cpu_supports("f16c")?widen_f16_native:widen_f16_table;
    return function;
#else
    return widen_f16_table;
#endif
}
bool pack_f16_scalar(const float* input,uint16_t* output,size_t count) {
    bool finite=true;
    for(size_t i=0;i<count;++i) {
        output[i]=strata::kernels::f16_from_f32(input[i]);
        finite=finite && (output[i]&0x7c00u)!=0x7c00u;
    }
    return finite;
}
#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
__attribute__((target("avx,f16c")))
bool pack_f16_native(const float* input,uint16_t* output,size_t count) {
    size_t i=0;int invalid=0;
    for(;i+8<=count;i+=8) {
        const auto bits=_mm256_cvtps_ph(_mm256_loadu_ps(input+i),_MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(output+i),bits);
        invalid|=_mm_movemask_epi8(_mm_cmpeq_epi16(_mm_and_si128(bits,_mm_set1_epi16(0x7c00)),_mm_set1_epi16(0x7c00)));
    }
    return pack_f16_scalar(input+i,output+i,count-i) && !invalid;
}
#endif
bool pack_f16(const float* input,uint16_t* output,size_t count) {
#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
    if(__builtin_cpu_supports("avx") && __builtin_cpu_supports("f16c")) return pack_f16_native(input,output,count);
#endif
    return pack_f16_scalar(input,output,count);
}
// The indexer's block dot: every float product is exact in F64, so the two variants differ only in how the
// 128-term sum is associated.
double index_dot_scalar(const float* query,const float* pooled,uint32_t width) {
    double dot=0;for(uint32_t d=0;d<width;++d) dot+=double(query[d])*double(pooled[d]);return dot;
}
#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
__attribute__((target("avx2")))
double index_dot_avx2(const float* query,const float* pooled,uint32_t width) {
    __m256d a0=_mm256_setzero_pd(),a1=_mm256_setzero_pd(),a2=_mm256_setzero_pd(),a3=_mm256_setzero_pd();
    uint32_t d=0;
    for(;d+16<=width;d+=16) {
        a0=_mm256_add_pd(a0,_mm256_mul_pd(_mm256_cvtps_pd(_mm_loadu_ps(query+d)),_mm256_cvtps_pd(_mm_loadu_ps(pooled+d))));
        a1=_mm256_add_pd(a1,_mm256_mul_pd(_mm256_cvtps_pd(_mm_loadu_ps(query+d+4)),_mm256_cvtps_pd(_mm_loadu_ps(pooled+d+4))));
        a2=_mm256_add_pd(a2,_mm256_mul_pd(_mm256_cvtps_pd(_mm_loadu_ps(query+d+8)),_mm256_cvtps_pd(_mm_loadu_ps(pooled+d+8))));
        a3=_mm256_add_pd(a3,_mm256_mul_pd(_mm256_cvtps_pd(_mm_loadu_ps(query+d+12)),_mm256_cvtps_pd(_mm_loadu_ps(pooled+d+12))));
    }
    for(;d+4<=width;d+=4)
        a0=_mm256_add_pd(a0,_mm256_mul_pd(_mm256_cvtps_pd(_mm_loadu_ps(query+d)),_mm256_cvtps_pd(_mm_loadu_ps(pooled+d))));
    const auto sum=_mm256_add_pd(_mm256_add_pd(a0,a1),_mm256_add_pd(a2,a3));
    const auto pair=_mm_add_pd(_mm256_castpd256_pd128(sum),_mm256_extractf128_pd(sum,1));
    double dot=_mm_cvtsd_f64(pair)+_mm_cvtsd_f64(_mm_unpackhi_pd(pair,pair));
    for(;d<width;++d) dot+=double(query[d])*double(pooled[d]);
    return dot;
}
#endif
using IndexDot=double (*)(const float*,const float*,uint32_t);
IndexDot index_dot() {
#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
    static const IndexDot function=__builtin_cpu_supports("avx2") && qsa_index_simd_enabled()?index_dot_avx2:index_dot_scalar;
    return function;
#else
    return index_dot_scalar;
#endif
}
// Block scores are independent, so a split cannot change any score. Fork-join with one countdown atomic and
// no shared lock: a pool of waiting workers needs condition variables, and the split only pays for its thread
// setup once a select covers many blocks.
template<class Body>
void parallel_blocks(uint32_t blocks,uint32_t threads,const Body& body) {
    const uint32_t chunk=blocks/threads,extra=blocks%threads;
    std::vector<std::exception_ptr> errors(threads);
    std::atomic<uint32_t> remaining{threads-1};
    std::vector<std::thread> workers;workers.reserve(threads-1);
    try {
        for(uint32_t index=1;index<threads;++index) {
            const uint32_t begin=index*chunk+std::min(index,extra),end=begin+chunk+(index<extra?1:0);
            workers.emplace_back([&body,&errors,&remaining,index,begin,end] {
                try {body(begin,end);} catch(...) {errors[index]=std::current_exception();}
                remaining.fetch_sub(1,std::memory_order_release);
            });
        }
    } catch(...) {
        // A joinable thread abandoned on the stack would terminate the process; the started slices are still
        // valid work, so let them finish before reporting the spawn failure.
        for(auto& worker:workers) if(worker.joinable()) worker.join();
        throw;
    }
    try {body(0,chunk);} catch(...) {errors[0]=std::current_exception();}
    while(remaining.load(std::memory_order_acquire)) std::this_thread::yield();
    for(auto& worker:workers) worker.join();
    for(const auto& error:errors) if(error) std::rethrow_exception(error);
}
void finite(std::span<const float> values) {
    if(!all_finite(values)) throw std::invalid_argument("QSA requires finite input values");
}
std::vector<float> pooled_key(std::span<const double> raw,std::span<const float> gamma,float epsilon) {
    double sum=0;
    for(auto value:raw) sum+=value*value;
    const double inverse=1/std::sqrt(sum/double(raw.size())+double(epsilon));
    std::vector<float> result(raw.size());
    for(size_t i=0;i<raw.size();++i) result[i]=float(raw[i]*inverse*double(gamma[i]));
    return result;
}
}
void QsaConfig::validate() const {
    if(!query_heads || !kv_heads || query_heads%kv_heads || query_heads>1024 || !width || width>65536 ||
       !index_heads || index_heads>32 || !index_width || index_width>65536 || block<2 || block>64 || !top_k ||
       !capacity || capacity>uint32_t(INT32_MAX) || !page_size || !rotary || rotary%2 || rotary>width || rotary>index_width ||
       !std::isfinite(epsilon) || epsilon<=0 || uint64_t(top_k)+block-1>uint64_t(INT32_MAX) ||
       position_base>uint64_t(INT32_MAX)-capacity)
        throw std::invalid_argument("unsupported QSA geometry/position/epsilon");
    if(kv==KvMode::Int8) (void)kv_int8_groups(width);
    if(kv==KvMode::Q4 || kv==KvMode::K8V4) (void)kv_q4_bytes_per_head(width);
    if(kv==KvMode::K8V4) (void)kv_int8_groups(width);
    if(auto error=strata::kernels::rope_scaling_invalid(rope)) throw std::invalid_argument(error);
}
uint32_t QsaConfig::selection_width(uint32_t cells) const {
    if(cells>capacity) throw std::out_of_range("QSA cells exceed capacity");
    return uint32_t(std::min(uint64_t(cells),uint64_t(top_k)+block-1));
}
QsaConfig qsa_config(const core::ModelGeometry& g,uint32_t capacity,uint32_t page_size) {
    const int64_t dims[]={g.n_head,g.n_head_kv,g.head_dim,g.n_rot,g.idx_q_heads,g.idx_key_dim,g.idx_block,g.idx_top_k};
    for(auto d:dims) if(d<=0 || d>INT32_MAX) throw std::invalid_argument("invalid model QSA geometry");
    QsaConfig c;
    c.query_heads=g.n_head;c.kv_heads=g.n_head_kv;c.width=g.head_dim;c.rotary=g.n_rot;
    c.index_heads=g.idx_q_heads;c.index_width=g.idx_key_dim;c.block=g.idx_block;c.top_k=g.idx_top_k;
    c.capacity=capacity;c.page_size=page_size;c.epsilon=float(g.rms_eps);c.validate();return c;
}
namespace {
template<class PositionOfPair>
RopeRow rope_row_with(uint32_t rotary,const PositionOfPair& position_of_pair,const strata::kernels::RopeScaling& sc) {
    if(!rotary || rotary%2 || rotary>65536) throw std::invalid_argument("invalid RoPE geometry/position");
    if(auto error=strata::kernels::rope_scaling_invalid(sc)) throw std::invalid_argument(error);
    const uint32_t half=rotary/2;
    RopeRow out{std::vector<float>(half),std::vector<float>(half)};
    double correction[2]{};
    if(sc.type!=strata::kernels::RopeScalingType::None) sc.corr_dims(int(rotary),correction);
    for(uint32_t i=0;i<half;++i) {
        const double angle=double(position_of_pair(i))*std::pow(sc.freq_base,-2.0*double(i)/rotary);
        double theta=angle,magnitude=1;
        if(sc.type!=strata::kernels::RopeScalingType::None) {
            theta=sc.freq_scale()*angle;magnitude=sc.mscale();
            if(sc.ext_factor!=0) {
                const double ramp=double(strata::kernels::rope_yarn_ramp(float(correction[0]),float(correction[1]),int(i)))*sc.ext_factor;
                theta=theta*(1-ramp)+angle*ramp;
            }
        }
        out.cosine[i]=float(std::cos(theta)*magnitude);out.sine[i]=float(std::sin(theta)*magnitude);
    }
    return out;
}
}
RopeRow rope_row(uint32_t rotary,uint64_t position,const strata::kernels::RopeScaling& sc) {
    if(position>uint64_t(INT32_MAX)) throw std::invalid_argument("invalid RoPE geometry/position");
    return rope_row_with(rotary,[position](uint32_t){return position;},sc);
}
RopeRow rope_row_sectors(uint32_t rotary,uint64_t t,uint64_t h,uint64_t w,const strata::kernels::RopeScaling& sc) {
    if(t>uint64_t(INT32_MAX) || h>uint64_t(INT32_MAX) || w>uint64_t(INT32_MAX)) throw std::invalid_argument("invalid RoPE geometry/position");
    return rope_row_with(rotary,[t,h,w](uint32_t pair){return pair%3==0?t:pair%3==1?h:w;},sc);
}
RopePositions::RopePositions(uint64_t cells):cells_(cells) {
    if(cells>uint64_t(INT32_MAX)) throw std::invalid_argument("image position table exceeds the int32 cell range");
    positions_.resize(std::size_t(cells*3));
    reset_identity();
}
void RopePositions::reset_identity() {
    for(uint64_t cell=0;cell<cells_;++cell) {
        if(cell>uint64_t(INT32_MAX)) throw std::invalid_argument("image position table exceeds the int32 cell range");
        positions_[std::size_t(cell*3)]=positions_[std::size_t(cell*3+1)]=positions_[std::size_t(cell*3+2)]=int32_t(cell);
    }
}
void RopePositions::install(std::span<const int32_t> table) {
    if(table.size()!=positions_.size()) throw std::invalid_argument("image position table shape mismatch");
    for(const auto value:table) if(value<0) throw std::invalid_argument("image position table contains a negative position");
    std::copy(table.begin(),table.end(),positions_.begin());
}
int32_t RopePositions::position(uint64_t cell,uint32_t pair) const {
    if(cell>=cells_) throw std::out_of_range("image position cell out of range");
    return positions_[std::size_t(cell*3+pair%3)];
}
RopeRow rope_row_at(const RopePositions* positions,uint32_t rotary,uint64_t position,
                    const strata::kernels::RopeScaling& sc) {
    if(!positions || !positions->installed() || position>=positions->cells()) return rope_row(rotary,position,sc);
    return rope_row_sectors(rotary,uint64_t(positions->position(position,0)),uint64_t(positions->position(position,1)),
                            uint64_t(positions->position(position,2)),sc);
}
void rotate_neox(std::span<float> rows,uint32_t width,const RopeRow& rope) {
    if(!width || rows.size()%width || rope.cosine.empty() || rope.cosine.size()!=rope.sine.size() || rope.cosine.size()*2>width)
        throw std::invalid_argument("RoPE tensor shape mismatch");
    finite(rows);finite(rope.cosine);finite(rope.sine);
    const auto half=rope.cosine.size();
    for(size_t row=0;row<rows.size()/width;++row) for(size_t pair=0;pair<half;++pair) {
        auto& left=rows[row*width+pair];auto& right=rows[row*width+half+pair];
        const float x=left,y=right,c=rope.cosine[pair],s=rope.sine[pair];
        left=x*c-y*s;right=x*s+y*c;
    }
    finite(rows);
}
void rms_norm_rows(std::span<float> rows,uint32_t width,std::span<const float> gamma,float epsilon) {
    if(!width || rows.size()%width || (gamma.size()!=width && gamma.size()!=rows.size()) || !std::isfinite(epsilon) || epsilon<=0)
        throw std::invalid_argument("RMS tensor shape/epsilon mismatch");
    finite(rows);finite(gamma);
    for(size_t row=0;row<rows.size()/width;++row) {
        double sum=0;for(uint32_t i=0;i<width;++i) { double value=rows[row*width+i];sum+=value*value; }
        const float inverse=float(1/std::sqrt(sum/width+double(epsilon)));
        for(uint32_t i=0;i<width;++i) rows[row*width+i]=(rows[row*width+i]*inverse)*gamma[gamma.size()==width?i:row*width+i];
    }
}
QsaState::QsaState(QsaConfig config,std::vector<uint32_t> pages) : config_(std::move(config)),owner_(next_owner++) {
    config_.validate();const auto& c=config_;
    const uint64_t count=(uint64_t(c.capacity)+c.page_size-1)/c.page_size;
    if(pages.empty()) { pages.resize(count);std::iota(pages.begin(),pages.end(),0); }
    if(pages.size()!=count) throw std::invalid_argument("QSA page table size mismatch");
    std::vector<uint8_t> seen(count);
    for(auto page:pages) { if(page>=count || seen[page]) throw std::invalid_argument("QSA page table must be a permutation");seen[page]=1; }
    pages_=std::move(pages);
    const auto elements=checked_product(checked_product(count,c.page_size),checked_product(c.kv_heads,c.width));
    if(c.kv==KvMode::Int8) {
        key_codes_.resize(elements);value_codes_.resize(elements);
        const auto scales=checked_product(checked_product(count,c.page_size),checked_product(c.kv_heads,groups()));
        key_scales_.resize(scales);value_scales_.resize(scales);
        value_prefix_bits_.resize(checked_product(c.kv_heads,c.width));
    } else if(c.kv==KvMode::Q4) {
        const auto bytes=checked_product(checked_product(count,c.page_size),
                                         checked_product(c.kv_heads,kv_q4_bytes_per_head(c.width)));
        q4_keys_.resize(bytes);q4_values_.resize(bytes);
        value_prefix_bits_.resize(checked_product(c.kv_heads,c.width));
    } else if(c.kv==KvMode::K8V4) {
        key_codes_.resize(elements);
        key_scales_.resize(checked_product(checked_product(count,c.page_size),checked_product(c.kv_heads,groups())));
        q4_values_.resize(checked_product(checked_product(count,c.page_size),
                                          checked_product(c.kv_heads,kv_q4_bytes_per_head(c.width))));
        value_prefix_bits_.resize(checked_product(c.kv_heads,c.width));
    } else {keys_.resize(elements);values_.resize(elements);}
    tail_.resize(checked_product(c.block-1,c.index_width));spare_.resize(c.index_width);
    pooled_.resize(checked_product(uint64_t(c.capacity)/c.block+1,c.index_width));
    prefixes_.resize(uint64_t(c.capacity)+1);
}
uint64_t QsaState::allocated_bytes() const {
    const uint64_t parts[]{snapshot_vector_bytes(keys_),snapshot_vector_bytes(values_),
        snapshot_vector_bytes(key_codes_),snapshot_vector_bytes(value_codes_),
        snapshot_vector_bytes(key_scales_),snapshot_vector_bytes(value_scales_),
        snapshot_vector_bytes(q4_keys_),snapshot_vector_bytes(q4_values_),
        snapshot_vector_bytes(value_prefix_bits_),snapshot_vector_bytes(pages_),
        snapshot_vector_bytes(tail_),snapshot_vector_bytes(spare_),snapshot_vector_bytes(pooled_),
        snapshot_vector_bytes(prefixes_)};
    uint64_t bytes=0;for(const auto part:parts) bytes=snapshot_add_bytes(bytes,part);
    return bytes;
}
uint64_t QsaState::row(uint32_t cell,uint32_t head) const {
    return ((uint64_t(pages_[cell/config_.page_size])*config_.kv_heads+head)*config_.page_size+cell%config_.page_size)*config_.width;
}
uint64_t QsaState::scale_row(uint32_t cell,uint32_t head) const {
    return ((uint64_t(pages_[cell/config_.page_size])*config_.kv_heads+head)*config_.page_size+cell%config_.page_size)*groups();
}
uint64_t QsaState::q4_row(uint32_t cell,uint32_t head) const {
    return ((uint64_t(pages_[cell/config_.page_size])*config_.kv_heads+head)*config_.page_size+cell%config_.page_size)*q4_bytes();
}
void QsaState::append(std::span<const float> k,std::span<const float> v,std::span<const float> raw,std::span<const float> gamma) {
    const auto& c=config_;
    if(size_>=c.capacity || generation_==UINT64_MAX) throw std::out_of_range("QSA cache capacity/generation exhausted");
    if(k.size()!=uint64_t(c.kv_heads)*c.width || v.size()!=k.size() || raw.size()!=c.index_width || gamma.size()!=raw.size())
        throw std::invalid_argument("QSA append shape mismatch");
    finite(k);finite(v);finite(raw);finite(gamma);
    // Upstream quantizes against the stored FP16 scale and clamps to [-127,127]; the FP16 path rounds the values
    // themselves and refuses an overflow of the 2-byte image; Q4_0 rotates the rows first and stores the ggml
    // blocks; k8v4 stores K unrotated INT8 groups and rotates only V into q4_0 blocks. Every path commits only
    // after all checks passed.
    std::vector<uint16_t> kh,vh;std::vector<int8_t> kc,vc;std::vector<uint16_t> ks,vs;std::vector<uint8_t> kq,vq;
    if(int8()) {
        kc.resize(k.size());vc.resize(v.size());ks.resize(k.size()/kv_int8_group);vs.resize(v.size()/kv_int8_group);
        kv_int8_quantize(k,kc,ks);kv_int8_quantize(v,vc,vs);
    } else if(q4()) {
        // Upstream rotates K/V before append (src/core/layer.cpp:1080-1163, q4 branch); the port rotates here so
        // the attention recipes can keep reading unrotated FP32 rows (see kv_q4_cell.hpp).
        const auto row_bytes=q4_bytes();
        std::vector<float> kr(k.size()),vr(v.size());
        kv_q4_rotate_rows(k,kr);kv_q4_rotate_rows(v,vr);
        kq.resize(uint64_t(c.kv_heads)*row_bytes);vq.resize(kq.size());
        for(uint32_t head=0;head<c.kv_heads;++head) {
            const auto at=uint64_t(head)*c.width;
            kv_q4_quantize_row(std::span<const float>(kr).subspan(at,c.width),
                               std::span<uint8_t>(kq).subspan(uint64_t(head)*row_bytes,row_bytes));
            kv_q4_quantize_row(std::span<const float>(vr).subspan(at,c.width),
                               std::span<uint8_t>(vq).subspan(uint64_t(head)*row_bytes,row_bytes));
        }
    } else if(hybrid()) {
        // k8v4 keeps K unrotated as the INT8 group pair and rotates only V (src/core/layer.cpp:1080-1103).
        kc.resize(k.size());ks.resize(k.size()/kv_int8_group);
        kv_int8_quantize(k,kc,ks);
        const auto row_bytes=q4_bytes();
        std::vector<float> vr(v.size());
        kv_q4_rotate_rows(v,vr);
        vq.resize(uint64_t(c.kv_heads)*row_bytes);
        for(uint32_t head=0;head<c.kv_heads;++head)
            kv_q4_quantize_row(std::span<const float>(vr).subspan(uint64_t(head)*c.width,c.width),
                               std::span<uint8_t>(vq).subspan(uint64_t(head)*row_bytes,row_bytes));
    } else {
        kh.resize(k.size());vh.resize(v.size());
        if(qsa_host_pack_enabled()) {
            const bool key_ok=pack_f16(k.data(),kh.data(),k.size());
            const bool value_ok=pack_f16(v.data(),vh.data(),v.size());
            if(!key_ok || !value_ok) throw std::invalid_argument("QSA FP16 cache overflow");
        } else for(size_t i=0;i<k.size();++i) {
            kh[i]=strata::kernels::f16_from_f32(k[i]);vh[i]=strata::kernels::f16_from_f32(v[i]);
            if(!std::isfinite(strata::kernels::f32_from_f16(kh[i])) || !std::isfinite(strata::kernels::f32_from_f16(vh[i])))
                throw std::invalid_argument("QSA FP16 cache overflow");
        }
    }
    const auto slot=size_%c.block,completed=size_/c.block;
    std::vector<float> first,pooled;
    if(size_==0) {
        std::vector<double> promoted(raw.begin(),raw.end());first=pooled_key(promoted,gamma,c.epsilon);
        // Upstream's spare keeps angle zero regardless of pos_base (native_qsa_indexer.cu:95,
        // `rope_pos = pos == 0 ? 0 : pos_base + R*b`); a table still supplies cell 0's sectors.
        rotate_neox(first,c.index_width,rope_row_at(c.positions,c.rotary,0,c.rope));
    }
    if(slot==c.block-1) {
        std::vector<double> mean(c.index_width);
        for(uint32_t dim=0;dim<c.index_width;++dim) {
            double sum=0;for(uint32_t r=0;r<c.block-1;++r) sum+=double(tail_[uint64_t(r)*c.index_width+dim]);
            mean[dim]=(sum+double(raw[dim]))/c.block;
        }
        pooled=pooled_key(mean,gamma,c.epsilon);
        rotate_neox(pooled,c.index_width,rope_row_at(c.positions,c.rotary,c.position_base+uint64_t(completed)*c.block,c.rope));
    }
    // Commit only after every conversion/allocation/validation above succeeded.
    for(uint32_t head=0;head<c.kv_heads;++head) {
        if(int8()) {
            std::copy_n(kc.begin()+uint64_t(head)*c.width,c.width,key_codes_.begin()+row(size_,head));
            std::copy_n(vc.begin()+uint64_t(head)*c.width,c.width,value_codes_.begin()+row(size_,head));
            std::copy_n(ks.begin()+uint64_t(head)*groups(),groups(),key_scales_.begin()+scale_row(size_,head));
            std::copy_n(vs.begin()+uint64_t(head)*groups(),groups(),value_scales_.begin()+scale_row(size_,head));
        } else if(q4()) {
            const auto row_bytes=q4_bytes(),at=uint64_t(head)*row_bytes;
            std::copy_n(kq.begin()+at,row_bytes,q4_keys_.begin()+q4_row(size_,head));
            std::copy_n(vq.begin()+at,row_bytes,q4_values_.begin()+q4_row(size_,head));
        } else if(hybrid()) {
            std::copy_n(kc.begin()+uint64_t(head)*c.width,c.width,key_codes_.begin()+row(size_,head));
            std::copy_n(ks.begin()+uint64_t(head)*groups(),groups(),key_scales_.begin()+scale_row(size_,head));
            const auto row_bytes=q4_bytes(),at=uint64_t(head)*row_bytes;
            std::copy_n(vq.begin()+at,row_bytes,q4_values_.begin()+q4_row(size_,head));
        } else {
            std::copy_n(kh.begin()+uint64_t(head)*c.width,c.width,keys_.begin()+row(size_,head));
            std::copy_n(vh.begin()+uint64_t(head)*c.width,c.width,values_.begin()+row(size_,head));
        }
    }
    if(size_==0) { spare_=std::move(first);std::copy(spare_.begin(),spare_.end(),pooled_.begin()); }
    if(slot<c.block-1) std::copy(raw.begin(),raw.end(),tail_.begin()+uint64_t(slot)*c.index_width);
    else {
        std::copy(pooled.begin(),pooled.end(),pooled_.begin()+uint64_t(completed)*c.index_width);
        std::copy(spare_.begin(),spare_.end(),pooled_.begin()+uint64_t(completed+1)*c.index_width);
    }
    prefixes_[++size_]=++generation_;
}
std::span<const float> QsaState::pooled_rows() const {
    return std::span<const float>(pooled_).first(size_?uint64_t(size_/config_.block+1)*config_.index_width:0);
}
std::span<const float> QsaState::tail_rows() const {
    return std::span<const float>(tail_).first(uint64_t(size_%config_.block)*config_.index_width);
}
std::vector<float> QsaState::block_scores(std::span<const float> query,std::span<const float> bias) const {
    const auto& c=config_;
    if(query.size()!=uint64_t(c.index_heads)*c.index_width || (!bias.empty() && bias.size()!=uint64_t(size_/c.block)+1))
        throw std::invalid_argument("QSA indexer query/bias shape mismatch");
    finite(query);finite(bias);
    const uint32_t blocks=size_?(size_+c.block-1)/c.block:0;
    std::vector<float> scores(blocks);const auto dot=index_dot();
    std::optional<ExactQsaIndex> exact;
    if(qsa_index_exact_enabled() || qsa_index_ilp_enabled()) exact.emplace(query,c.index_width,c.index_heads);
    const auto compiled=[&](uint32_t begin,uint32_t end) {
        std::vector<double> interleaved;
        if(qsa_index_ilp_enabled()) {
            interleaved.resize(end-begin);
            exact->score_blocks(pooled_.data()+uint64_t(begin)*c.index_width,end-begin,interleaved);
        }
        for(uint32_t block=begin;block<end;++block) {
            const auto* pooled=pooled_.data()+uint64_t(block)*c.index_width;
            double score=0;
            if(!interleaved.empty()) score=interleaved[block-begin];
            else if(exact) score=exact->score(pooled);
            else for(uint32_t head=0;head<c.index_heads;++head)
                score+=std::max(dot(query.data()+uint64_t(head)*c.index_width,pooled,c.index_width),0.0);
            if(!bias.empty()) score+=double(bias[block]);
            float value=float(score);if(block+1==blocks && size_%c.block) value+=1e9f;
            if(!std::isfinite(value)) throw std::runtime_error("non-finite QSA indexer score");
            scores[block]=value;
        }
    };
    const auto threads=uint32_t(qsa_index_threads());
    // From 2048 blocks (8k cells) the split pays for its thread starts (~50 us per call; 12 calls a decoded token).
    if(threads>1 && blocks>=2048) parallel_blocks(blocks,threads,compiled);
    else compiled(0,blocks);
    return scores;
}
std::vector<float> QsaState::cell_scores(std::span<const float> query,std::span<const float> bias) const {
    const auto& c=config_;const auto scores=block_scores(query,bias);
    std::vector<float> out(size_);
    for(uint32_t block=0;block<scores.size();++block)
        std::fill(out.begin()+uint64_t(block)*c.block,out.begin()+std::min<uint64_t>(size_,uint64_t(block+1)*c.block),scores[block]);
    return out;
}
std::vector<int32_t> QsaState::select(std::span<const float> query,std::span<const float> bias) const {
    const auto& c=config_;const uint32_t width=config_.selection_width(size_);
    std::vector<int32_t> ids;
    if(width<size_) {
        // Every cell of a block shares its score, so ranking blocks and expanding them yields the same prefix of
        // the (score desc, cell id asc) order as ranking all cells, with 1/block of the comparisons.
        const auto scores=block_scores(query,bias);
        const uint32_t blocks=uint32_t(scores.size());
        std::vector<int32_t> order(blocks);std::iota(order.begin(),order.end(),0);
        // Only the last block can be short, so ceil(width/block) ranked blocks can fall a few cells short of
        // `width`; one more block always covers the worst case.
        const uint32_t keep=std::min<uint32_t>(blocks,uint32_t((uint64_t(width)-1)/c.block+2));
        const auto better=[&](int32_t a,int32_t b){return scores[a]>scores[b] || (scores[a]==scores[b] && a<b);};
        // The comparator is a strict total order, so the top section is unique and nth_element is exact. It reads
        // every block once, where partial_sort keeps a heap of `keep` through the whole scan.
        if(keep<blocks) std::nth_element(order.begin(),order.begin()+keep,order.end(),better);
        std::sort(order.begin(),order.begin()+keep,better);
        ids.clear();ids.reserve(width);
        for(uint32_t i=0;i<keep && ids.size()<width;++i) {
            const auto begin=uint64_t(uint32_t(order[i]))*c.block,end=std::min<uint64_t>(size_,begin+c.block);
            for(auto cell=begin;cell<end && ids.size()<width;++cell) ids.push_back(int32_t(cell));
        }
        std::sort(ids.begin(),ids.end());
    } else { // Validate inputs even for the identity path.
        if(query.size()!=uint64_t(config_.index_heads)*config_.index_width || (!bias.empty() && bias.size()!=uint64_t(size_/config_.block)+1))
            throw std::invalid_argument("QSA identity query/bias shape mismatch");
        finite(query);finite(bias);
        ids.resize(size_);std::iota(ids.begin(),ids.end(),0);
    }
    return ids;
}
void QsaState::check_ids(std::span<const int32_t> ids) const {
    // select() returns sorted IDs. Validate that common case without one tree allocation per cell;
    // the public gather API still accepts an arbitrary order and rejects duplicates in that order.
    bool ordered=true;int32_t previous=-1;
    for(auto id:ids) {
        if(id<0 || uint32_t(id)>=size_) throw std::out_of_range("QSA invalid/duplicate selected cell");
        ordered=ordered && id>previous;previous=id;
    }
    if(!ordered) {
        std::set<int32_t> seen;
        for(auto id:ids) if(!seen.insert(id).second) throw std::out_of_range("QSA invalid/duplicate selected cell");
    }
}
void QsaState::gather_rows(std::span<const int32_t> ids,uint32_t bucket,float* key,float* value) const {
    const auto& c=config_;
    // INT8 K rows decode through their stored scale; rotated Q4_0 rows (the whole q4_0 pool and k8v4's V half)
    // are dequantized and rotated back into FP32; one row of scratch is reused for every cell/head.
    const bool int8_keys=int8()||hybrid();
    const auto widen=f16_widener();
    std::vector<float> unrotated((q4()||hybrid())?c.width:0);
    for(uint32_t head=0;head<c.kv_heads;++head) for(size_t item=0;item<ids.size();++item) {
        const auto source=row(uint32_t(ids[item]),head),dest=(uint64_t(head)*bucket+item)*c.width;
        if(int8_keys) {
            const auto scale=scale_row(uint32_t(ids[item]),head);
            for(uint32_t d=0;d<c.width;++d)
                key[dest+d]=kv_int8_value(key_codes_[source+d],key_scales_[scale+d/kv_int8_group]);
            if(hybrid()) {
                const auto stored=q4_row(uint32_t(ids[item]),head),bytes=q4_bytes();
                kv_q4_dequantize_unrotated(std::span<const uint8_t>(q4_values_).subspan(stored,bytes),unrotated);
                std::copy(unrotated.begin(),unrotated.end(),value+dest);
            } else {
                for(uint32_t d=0;d<c.width;++d)
                    value[dest+d]=kv_int8_value(value_codes_[source+d],value_scales_[scale+d/kv_int8_group]);
            }
        } else if(q4()) {
            const auto stored=q4_row(uint32_t(ids[item]),head),bytes=q4_bytes();
            kv_q4_dequantize_unrotated(std::span<const uint8_t>(q4_keys_).subspan(stored,bytes),unrotated);
            std::copy(unrotated.begin(),unrotated.end(),key+dest);
            kv_q4_dequantize_unrotated(std::span<const uint8_t>(q4_values_).subspan(stored,bytes),unrotated);
            std::copy(unrotated.begin(),unrotated.end(),value+dest);
        } else {
            // Cached FP16 values are finite; F16C widens them exactly, with table fallback on other hosts.
            widen(keys_.data()+source,key+dest,c.width);widen(values_.data()+source,value+dest,c.width);
        }
    }
}
SelectedKv QsaState::gather(std::span<const int32_t> ids,uint32_t bucket) const {
    if(ids.size()>bucket || bucket>config_.selection_width(config_.capacity)) throw std::invalid_argument("QSA gather bucket mismatch");
    check_ids(ids);
    const auto& c=config_;const auto count=checked_product(checked_product(c.kv_heads,bucket),c.width);
    SelectedKv out{uint32_t(ids.size()),bucket,std::vector<float>(count),std::vector<float>(count),std::vector<float>(bucket,-INFINITY)};
    std::fill_n(out.mask.begin(),ids.size(),0);
    gather_rows(ids,bucket,out.key.data(),out.value.data());
    return out;
}
void QsaState::gather_into(std::span<const int32_t> ids,uint32_t bucket,std::span<float> key,std::span<float> value,
                           std::span<float> mask) const {
    if(ids.size()>bucket || bucket>config_.selection_width(config_.capacity)) throw std::invalid_argument("QSA gather bucket mismatch");
    check_ids(ids);
    const auto& c=config_;const auto count=checked_product(checked_product(c.kv_heads,bucket),c.width);
    if(key.size()!=count || value.size()!=count || mask.size()!=bucket)
        throw std::invalid_argument("QSA gather destination shape mismatch");
    std::fill_n(mask.begin(),ids.size(),0.0f);std::fill(mask.begin()+ids.size(),mask.end(),-INFINITY);
    const auto padding=uint64_t(c.width)*(bucket-uint32_t(ids.size()));
    for(uint32_t head=0;head<c.kv_heads;++head) {
        const auto at=(uint64_t(head)*bucket+ids.size())*c.width;
        std::fill_n(key.data()+at,padding,0.0f);std::fill_n(value.data()+at,padding,0.0f);
    }
    gather_rows(ids,bucket,key.data(),value.data());
}
void QsaState::fp16_cells(uint32_t first,uint32_t count,std::span<float> keys,std::span<float> values) const {
    const auto& c=config_;const uint64_t row=uint64_t(c.kv_heads)*c.width;
    if(int8() || q4() || hybrid()) throw std::logic_error("QSA FP16 cell export needs the FP16 pool");
    if(uint64_t(first)+count>size_ || keys.size()!=row*count || values.size()!=keys.size()) throw std::invalid_argument("QSA FP16 cell export range/shape mismatch");
    const auto widen=f16_widener();
    for(uint32_t cell=0;cell<count;++cell) for(uint32_t head=0;head<c.kv_heads;++head) {
        const auto source=this->row(first+cell,head),dest=uint64_t(cell)*row+uint64_t(head)*c.width;
        widen(keys_.data()+source,keys.data()+dest,c.width);widen(values_.data()+source,values.data()+dest,c.width);
    }
}
std::span<const uint16_t> QsaState::value_pool_prefix() const {
    if(!size_) throw std::logic_error("QSA diagnostic pool has no committed cell");
    const auto count=uint64_t(config_.kv_heads)*config_.width;
    if(!int8() && !q4() && !hybrid()) return std::span<const uint16_t>(values_).first(count);
    if(q4() || hybrid()) {
        // Same physical prefix as the FP16 pool: the first kv_heads storage rows, still rotated, as the FP16 bits
        // kv_gather_q4_kernel writes before the attention back-rotation. Nothing is allocated here.
        const auto bytes=q4_bytes();
        for(uint32_t row=0;row<config_.kv_heads;++row)
            kv_q4_dequantize_row(std::span<const uint8_t>(q4_values_).subspan(uint64_t(row)*bytes,bytes),
                                 std::span<uint16_t>(value_prefix_bits_).subspan(uint64_t(row)*config_.width,config_.width));
        return value_prefix_bits_;
    }
    // The INT8 pool's physical prefix is page 0 / head 0 / its first kv_heads cells; decode it to the same FP16
    // bits attention consumes. The buffer was sized in the constructor, so this accessor allocates nothing.
    for(uint64_t i=0;i<count;++i) value_prefix_bits_[i]=kv_int8_bits(value_codes_[i],value_scales_[i/kv_int8_group]);
    return value_prefix_bits_;
}
void QsaState::selected_value_bits(int32_t cell,std::span<uint16_t> output) const {
    if(cell<0 || uint32_t(cell)>=size_ || output.size()!=uint64_t(config_.kv_heads)*config_.width)
        throw std::invalid_argument("QSA diagnostic selected cell/shape mismatch");
    for(uint32_t head=0;head<config_.kv_heads;++head) {
        const auto source=row(uint32_t(cell),head);
        const auto dest=output.begin()+uint64_t(head)*config_.width;
        if(int8()) {
            const auto scale=scale_row(uint32_t(cell),head);
            for(uint32_t d=0;d<config_.width;++d) dest[d]=kv_int8_bits(value_codes_[source+d],value_scales_[scale+d/kv_int8_group]);
        } else if(q4() || hybrid()) {
            const auto bytes=q4_bytes();
            kv_q4_dequantize_row(std::span<const uint8_t>(q4_values_).subspan(q4_row(uint32_t(cell),head),bytes),
                                 output.subspan(uint64_t(head)*config_.width,config_.width));
        } else std::copy_n(values_.begin()+source,config_.width,dest);
    }
}
QsaState::Checkpoint QsaState::checkpoint() const {
    Checkpoint p;p.owner=owner_;p.epoch=epoch_;p.cells=size_;p.prefix=prefixes_[size_];p.tail=tail_;p.spare=spare_;return p;
}
QsaState::Checkpoint QsaState::snapshot() const {
    auto point=checkpoint();point.persistent=true;const auto& c=config_;
    const auto elements=checked_product(uint64_t(size_),checked_product(c.kv_heads,c.width));
    const auto scales=checked_product(uint64_t(size_),checked_product(c.kv_heads,groups()));
    if(int8()) {
        point.key_codes.resize(elements);point.value_codes.resize(elements);
        point.key_scales.resize(scales);point.value_scales.resize(scales);
        for(uint32_t cell=0;cell<size_;++cell) for(uint32_t head=0;head<c.kv_heads;++head) {
            const auto dest=(uint64_t(cell)*c.kv_heads+head)*c.width,source=row(cell,head);
            const auto dest_scale=(uint64_t(cell)*c.kv_heads+head)*groups(),source_scale=scale_row(cell,head);
            std::copy_n(key_codes_.begin()+source,c.width,point.key_codes.begin()+dest);
            std::copy_n(value_codes_.begin()+source,c.width,point.value_codes.begin()+dest);
            std::copy_n(key_scales_.begin()+source_scale,groups(),point.key_scales.begin()+dest_scale);
            std::copy_n(value_scales_.begin()+source_scale,groups(),point.value_scales.begin()+dest_scale);
        }
    } else if(hybrid()) {
        const auto row_bytes=this->q4_bytes();
        const auto q4_bytes=checked_product(uint64_t(size_),checked_product(c.kv_heads,row_bytes));
        point.key_codes.resize(elements);point.key_scales.resize(scales);point.q4_value.resize(q4_bytes);
        for(uint32_t cell=0;cell<size_;++cell) for(uint32_t head=0;head<c.kv_heads;++head) {
            const auto dest=(uint64_t(cell)*c.kv_heads+head)*c.width,source=row(cell,head);
            const auto dest_scale=(uint64_t(cell)*c.kv_heads+head)*groups(),source_scale=scale_row(cell,head);
            const auto dest_q4=(uint64_t(cell)*c.kv_heads+head)*row_bytes,source_q4=q4_row(cell,head);
            std::copy_n(key_codes_.begin()+source,c.width,point.key_codes.begin()+dest);
            std::copy_n(key_scales_.begin()+source_scale,groups(),point.key_scales.begin()+dest_scale);
            std::copy_n(q4_values_.begin()+source_q4,row_bytes,point.q4_value.begin()+dest_q4);
        }
    } else if(q4()) {
        const auto row_bytes=this->q4_bytes();
        const auto q4_bytes=checked_product(uint64_t(size_),checked_product(c.kv_heads,row_bytes));
        point.q4_key.resize(q4_bytes);point.q4_value.resize(q4_bytes);
        for(uint32_t cell=0;cell<size_;++cell) for(uint32_t head=0;head<c.kv_heads;++head) {
            const auto dest=(uint64_t(cell)*c.kv_heads+head)*row_bytes,source=q4_row(cell,head);
            std::copy_n(q4_keys_.begin()+source,row_bytes,point.q4_key.begin()+dest);
            std::copy_n(q4_values_.begin()+source,row_bytes,point.q4_value.begin()+dest);
        }
    } else {
        point.keys.resize(elements);point.values.resize(elements);
        for(uint32_t cell=0;cell<size_;++cell) for(uint32_t head=0;head<c.kv_heads;++head) {
            const auto dest=(uint64_t(cell)*c.kv_heads+head)*c.width,source=row(cell,head);
            std::copy_n(keys_.begin()+source,c.width,point.keys.begin()+dest);std::copy_n(values_.begin()+source,c.width,point.values.begin()+dest);
        }
    }
    const auto pooled=pooled_rows();point.pooled.assign(pooled.begin(),pooled.end());return point;
}
void QsaState::snapshot_sizes(const Checkpoint& point,uint64_t (&sizes)[5]) const {
    // Upstream's parked parts in order: K, V, K scale, V scale, pooled indexer rows. FP16 K/V are two bytes per
    // element with no scale parts; INT8 K/V are one code byte per element plus two scale bytes per 64 values;
    // k8v4 is the INT8 K half plus the rotated Q4_0 V half with no V scale parts (kKvHybrid).
    if(int8()) {
        sizes[0]=uint64_t(point.key_codes.size())*sizeof(int8_t);
        sizes[1]=uint64_t(point.value_codes.size())*sizeof(int8_t);
        sizes[2]=uint64_t(point.key_scales.size())*sizeof(uint16_t);
        sizes[3]=uint64_t(point.value_scales.size())*sizeof(uint16_t);
    } else if(hybrid()) {
        sizes[0]=uint64_t(point.key_codes.size())*sizeof(int8_t);
        sizes[1]=uint64_t(point.q4_value.size());
        sizes[2]=uint64_t(point.key_scales.size())*sizeof(uint16_t);
        sizes[3]=0;   // the q4_0 V block carries its own FP16 scale
    } else if(q4()) {
        sizes[0]=uint64_t(point.q4_key.size());sizes[1]=uint64_t(point.q4_value.size());
        sizes[2]=0;sizes[3]=0;   // the q4_0 block carries its own FP16 scale
    } else {
        sizes[0]=uint64_t(point.keys.size())*sizeof(uint16_t);
        sizes[1]=uint64_t(point.values.size())*sizeof(uint16_t);
        sizes[2]=0;sizes[3]=0;
    }
    sizes[4]=uint64_t(point.pooled.size())*sizeof(float);
}
QsaState::Checkpoint QsaState::snapshot_reused(Checkpoint&& prior,uint32_t whole_cells,uint32_t pooled_rows_kept) const {
    const auto& c=config_;
    // The same ancestry rule a rewind checkpoint is restored under: the image may only be grown when the live
    // prefix at its own extent is still the one it was parked from. Rewind-and-reappend rewrites prefixes_[k]
    // for k past the rewind, so a rewritten conversation is refused here rather than silently reused.
    if(!prior.persistent || prior.owner!=owner_ || prior.cells>size_ || prior.cells>c.capacity ||
       whole_cells>prior.cells || (prior.cells && prior.prefix!=prefixes_[prior.cells]) ||
       prior.tail.size()!=tail_.size() || prior.spare.size()!=spare_.size())
        throw std::invalid_argument("QSA reused snapshot is stale/foreign");
    const auto per_cell=checked_product(c.kv_heads,c.width);
    const auto prior_elements=checked_product(uint64_t(prior.cells),per_cell);
    const auto prior_scales=checked_product(uint64_t(prior.cells),checked_product(c.kv_heads,groups()));
    const auto row_bytes=(q4()||hybrid())?q4_bytes():0;
    const auto prior_q4=checked_product(uint64_t(prior.cells),checked_product(c.kv_heads,row_bytes));
    if(int8()?prior.key_codes.size()!=prior_elements || prior.value_codes.size()!=prior_elements ||
              prior.key_scales.size()!=prior_scales || prior.value_scales.size()!=prior_scales
             :hybrid()?prior.key_codes.size()!=prior_elements || prior.key_scales.size()!=prior_scales ||
                       prior.q4_value.size()!=prior_q4
             :q4()?prior.q4_key.size()!=prior_q4 || prior.q4_value.size()!=prior_q4
                   :prior.keys.size()!=prior_elements || prior.values.size()!=prior_elements)
        throw std::invalid_argument("QSA reused snapshot shape mismatch");
    const auto rows=uint64_t(size_?size_/c.block+1:0);
    if(uint64_t(pooled_rows_kept)*c.index_width>prior.pooled.size())
        throw std::invalid_argument("QSA reused snapshot: missing reusable prefix");
    Checkpoint point=std::move(prior);
    // Growing keeps [0,whole_cells) and [0,pooled_rows_kept) exactly where they are; upstream's
    // ConversationBuffer::resize does the same and then visits only [keep,size).
    if(int8()) {
        snapshot_resize(point.key_codes,checked_product(uint64_t(size_),per_cell));
        snapshot_resize(point.value_codes,point.key_codes.size());
        snapshot_resize(point.key_scales,checked_product(uint64_t(size_),checked_product(c.kv_heads,groups())));
        snapshot_resize(point.value_scales,point.key_scales.size());
        for(uint32_t cell=whole_cells;cell<size_;++cell) for(uint32_t head=0;head<c.kv_heads;++head) {
            const auto dest=(uint64_t(cell)*c.kv_heads+head)*c.width,source=row(cell,head);
            const auto dest_scale=(uint64_t(cell)*c.kv_heads+head)*groups(),source_scale=scale_row(cell,head);
            std::copy_n(key_codes_.begin()+source,c.width,point.key_codes.begin()+dest);
            std::copy_n(value_codes_.begin()+source,c.width,point.value_codes.begin()+dest);
            std::copy_n(key_scales_.begin()+source_scale,groups(),point.key_scales.begin()+dest_scale);
            std::copy_n(value_scales_.begin()+source_scale,groups(),point.value_scales.begin()+dest_scale);
        }
    } else if(hybrid()) {
        snapshot_resize(point.key_codes,checked_product(uint64_t(size_),per_cell));
        snapshot_resize(point.key_scales,checked_product(uint64_t(size_),checked_product(c.kv_heads,groups())));
        const auto grown=checked_product(uint64_t(size_),checked_product(c.kv_heads,row_bytes));
        snapshot_resize(point.q4_value,grown);
        for(uint32_t cell=whole_cells;cell<size_;++cell) for(uint32_t head=0;head<c.kv_heads;++head) {
            const auto dest=(uint64_t(cell)*c.kv_heads+head)*c.width,source=row(cell,head);
            const auto dest_scale=(uint64_t(cell)*c.kv_heads+head)*groups(),source_scale=scale_row(cell,head);
            const auto dest_q4=(uint64_t(cell)*c.kv_heads+head)*row_bytes,source_q4=q4_row(cell,head);
            std::copy_n(key_codes_.begin()+source,c.width,point.key_codes.begin()+dest);
            std::copy_n(key_scales_.begin()+source_scale,groups(),point.key_scales.begin()+dest_scale);
            std::copy_n(q4_values_.begin()+source_q4,row_bytes,point.q4_value.begin()+dest_q4);
        }
    } else if(q4()) {
        const auto grown=checked_product(uint64_t(size_),checked_product(c.kv_heads,row_bytes));
        snapshot_resize(point.q4_key,grown);snapshot_resize(point.q4_value,grown);
        for(uint32_t cell=whole_cells;cell<size_;++cell) for(uint32_t head=0;head<c.kv_heads;++head) {
            const auto dest=(uint64_t(cell)*c.kv_heads+head)*row_bytes,source=q4_row(cell,head);
            std::copy_n(q4_keys_.begin()+source,row_bytes,point.q4_key.begin()+dest);
            std::copy_n(q4_values_.begin()+source,row_bytes,point.q4_value.begin()+dest);
        }
    } else {
        snapshot_resize(point.keys,checked_product(uint64_t(size_),per_cell));snapshot_resize(point.values,point.keys.size());
        for(uint32_t cell=whole_cells;cell<size_;++cell) for(uint32_t head=0;head<c.kv_heads;++head) {
            const auto dest=(uint64_t(cell)*c.kv_heads+head)*c.width,source=row(cell,head);
            std::copy_n(keys_.begin()+source,c.width,point.keys.begin()+dest);
            std::copy_n(values_.begin()+source,c.width,point.values.begin()+dest);
        }
    }
    snapshot_resize(point.pooled,rows*c.index_width);
    for(uint64_t r=pooled_rows_kept;r<rows;++r) {
        const auto at=r*c.index_width;
        std::copy_n(pooled_.begin()+at,c.index_width,point.pooled.begin()+at);
    }
    // The in-progress indexer block is live state, not parked state: its raw rows and the rotating spare row
    // always come from the pools, exactly like the spare row upstream re-copies past the reuse boundary.
    point.tail.assign(tail_.begin(),tail_.end());point.spare.assign(spare_.begin(),spare_.end());
    point.cells=size_;point.prefix=prefixes_[size_];point.epoch=epoch_;
    return point;
}
void QsaState::restore(const Checkpoint& p) {
    if(p.owner!=owner_ || p.cells>config_.capacity || (!p.persistent && (p.epoch!=epoch_ || p.cells>size_ || p.prefix!=prefixes_[p.cells])) || p.tail.size()!=tail_.size() || p.spare.size()!=spare_.size())
        throw std::invalid_argument("QSA stale/foreign checkpoint");
    if(p.persistent) {
        const auto& c=config_;const auto elements=checked_product(uint64_t(p.cells),checked_product(c.kv_heads,c.width));
        const auto scales=checked_product(uint64_t(p.cells),checked_product(c.kv_heads,groups()));
        const auto bytes=(q4()||hybrid())?q4_bytes():0;
        const auto q4_elements=checked_product(uint64_t(p.cells),checked_product(c.kv_heads,bytes));
        const bool shaped=int8()?p.key_codes.size()==elements && p.value_codes.size()==elements &&
                                 p.key_scales.size()==scales && p.value_scales.size()==scales
                          :hybrid()?p.key_codes.size()==elements && p.key_scales.size()==scales &&
                                    p.q4_value.size()==q4_elements
                          :q4()?p.q4_key.size()==q4_elements && p.q4_value.size()==q4_elements
                               :p.keys.size()==elements && p.values.size()==elements;
        if(!shaped || p.pooled.size()!=(p.cells?uint64_t(p.cells/c.block+1)*c.index_width:0))
            throw std::invalid_argument("QSA snapshot shape mismatch");
        const auto generation=std::max(generation_,p.prefix);
        if(uint64_t(p.cells)>UINT64_MAX-generation) throw std::overflow_error("QSA snapshot lineage exhausted");
        for(uint32_t cell=0;cell<p.cells;++cell) for(uint32_t head=0;head<c.kv_heads;++head) {
            const auto source=(uint64_t(cell)*c.kv_heads+head)*c.width,dest=row(cell,head);
            if(int8()) {
                const auto source_scale=(uint64_t(cell)*c.kv_heads+head)*groups(),dest_scale=scale_row(cell,head);
                std::copy_n(p.key_codes.begin()+source,c.width,key_codes_.begin()+dest);
                std::copy_n(p.value_codes.begin()+source,c.width,value_codes_.begin()+dest);
                std::copy_n(p.key_scales.begin()+source_scale,groups(),key_scales_.begin()+dest_scale);
                std::copy_n(p.value_scales.begin()+source_scale,groups(),value_scales_.begin()+dest_scale);
            } else if(hybrid()) {
                const auto source_scale=(uint64_t(cell)*c.kv_heads+head)*groups(),dest_scale=scale_row(cell,head);
                std::copy_n(p.key_codes.begin()+source,c.width,key_codes_.begin()+dest);
                std::copy_n(p.key_scales.begin()+source_scale,groups(),key_scales_.begin()+dest_scale);
                const auto source_bytes=(uint64_t(cell)*c.kv_heads+head)*bytes,dest_bytes=q4_row(cell,head);
                std::copy_n(p.q4_value.begin()+source_bytes,bytes,q4_values_.begin()+dest_bytes);
            } else if(q4()) {
                const auto source_bytes=(uint64_t(cell)*c.kv_heads+head)*bytes,dest_bytes=q4_row(cell,head);
                std::copy_n(p.q4_key.begin()+source_bytes,bytes,q4_keys_.begin()+dest_bytes);
                std::copy_n(p.q4_value.begin()+source_bytes,bytes,q4_values_.begin()+dest_bytes);
            } else {
                std::copy_n(p.keys.begin()+source,c.width,keys_.begin()+dest);
                std::copy_n(p.values.begin()+source,c.width,values_.begin()+dest);
            }
        }
        std::copy(p.pooled.begin(),p.pooled.end(),pooled_.begin());
        // A restored branch's tip retains its identity, which may differ from
        // its cell count after a rewind/replay. Intermediate prefixes get fresh
        // identities: unrelated parked images must not become reusable merely
        // because both were restored/reset at the same length.
        ++epoch_;generation_=generation;prefixes_[0]=0;
        for(uint32_t cell=1;cell<=p.cells;++cell) prefixes_[cell]=++generation_;
        if(p.cells) prefixes_[p.cells]=p.prefix;
    }
    size_=p.cells;std::copy(p.tail.begin(),p.tail.end(),tail_.begin());std::copy(p.spare.begin(),p.spare.end(),spare_.begin());
    std::copy(spare_.begin(),spare_.end(),pooled_.begin()+uint64_t(size_/config_.block)*config_.index_width);
    // A child rewind preserves an ancestor's prefix identity. Appending a new
    // branch gets fresh generations, rejecting any checkpoint of the old tail.
}
void QsaState::reset() { size_=0;++epoch_;prefixes_[0]=0;std::fill(tail_.begin(),tail_.end(),0);std::fill(spare_.begin(),spare_.end(),0);std::fill(pooled_.begin(),pooled_.end(),0); }
std::vector<float> qsa_attention_reference(std::span<const float> query,const SelectedKv& kv,uint32_t hq,uint32_t hk,uint32_t d) {
    if(!hq || !hk || hq%hk || !d || query.size()!=uint64_t(hq)*d || kv.count>kv.bucket ||
       kv.key.size()!=uint64_t(hk)*kv.bucket*d || kv.value.size()!=kv.key.size() || kv.mask.size()!=kv.bucket)
        throw std::invalid_argument("QSA attention shape mismatch");
    finite(query);finite(kv.key);finite(kv.value);
    for(uint32_t cell=0;cell<kv.bucket;++cell)
        if(cell<kv.count?kv.mask[cell]!=0:kv.mask[cell]!=-INFINITY)
            throw std::invalid_argument("QSA attention requires contiguous valid rows and -inf padding");
    std::vector<float> output(uint64_t(hq)*d);if(!kv.count) return output;
    std::vector<float> scores(kv.count);const float scale=1/std::sqrt(float(d));
    for(uint32_t head=0;head<hq;++head) {
        const auto kh=head/(hq/hk);float maximum=-INFINITY;
        for(uint32_t cell=0;cell<kv.count;++cell) {
            float dot=0;for(uint32_t i=0;i<d;++i) dot+=query[uint64_t(head)*d+i]*kv.key[(uint64_t(kh)*kv.bucket+cell)*d+i];
            scores[cell]=dot*scale;
            if(!std::isfinite(scores[cell])) throw std::overflow_error("QSA attention dot overflow");
            maximum=std::max(maximum,scores[cell]);
        }
        float sum=0;for(auto& value:scores) {value=std::exp(value-maximum);sum+=value;}
        const float inverse=1/sum;
        for(uint32_t i=0;i<d;++i) for(uint32_t cell=0;cell<kv.count;++cell)
            output[uint64_t(head)*d+i]+=(scores[cell]*inverse)*kv.value[(uint64_t(kh)*kv.bucket+cell)*d+i];
    }
    return output;
}
}
