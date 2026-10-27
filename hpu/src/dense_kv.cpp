#include "strata/hpu/dense_kv.hpp"
#include "strata/hpu/kv_cell.hpp"
#include "strata/hpu/snapshot_memory.hpp"
#include "strata/kernels/f16_bits.hpp"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <numeric>

namespace strata::hpu {
namespace {std::atomic<uint64_t> next_dense_owner{1};}
DenseKvState::DenseKvState(uint32_t capacity,uint32_t heads,uint32_t width,KvMode kv)
    :capacity_(capacity),heads_(heads),width_(width),kv_(kv),owner_(next_dense_owner++) {
    if(!capacity || capacity>uint32_t(INT32_MAX) || !heads || heads>1024 || !width || width>65536)
        throw std::invalid_argument("invalid dense KV geometry");
    // The drafter has no indexer and no hybrid pool: under --kv k8v4 upstream builds its ring as plain INT8
    // (src/core/mtp.cpp:344-349), so a direct hybrid construction is refused rather than silently stored FP16.
    if(kv_==KvMode::K8V4) throw std::invalid_argument("dense KV has no k8v4 layout; the drafter runs plain INT8");
    if(kv_==KvMode::Int8) (void)kv_int8_groups(width_);
    if(kv_==KvMode::Q4) (void)kv_q4_bytes_per_head(width_);
    const auto count=checked_product(capacity,checked_product(heads,width));
    if(count>SIZE_MAX/2) throw std::overflow_error("dense KV size overflow");
    if(kv_==KvMode::Int8) {
        key_codes_.resize(count);value_codes_.resize(count);
        const auto scales=checked_product(capacity,checked_product(heads,groups()));
        key_scales_.resize(scales);value_scales_.resize(scales);
    } else if(kv_==KvMode::Q4) {
        const auto bytes=checked_product(capacity,checked_product(heads,q4_bytes()));
        q4_key_.resize(bytes);q4_value_.resize(bytes);
    } else {key_.resize(count);value_.resize(count);}
    prefixes_.resize(uint64_t(capacity)+1);
}
void DenseKvState::append(std::span<const float> key,std::span<const float> value) {
    const auto width=uint64_t(heads_)*width_;
    if(size_>=capacity_ || key.size()!=width || value.size()!=width || generation_==UINT64_MAX) throw std::invalid_argument("dense KV append shape/capacity/generation mismatch");
    // Validate both rows before exposing any new cell. No storage is written until every check passed.
    if(int8()) {
        std::vector<int8_t> kc(width),vc(width);
        std::vector<uint16_t> ks(width/kv_int8_group),vs(width/kv_int8_group);
        kv_int8_quantize(key,kc,ks);kv_int8_quantize(value,vc,vs);
        const auto at=uint64_t(size_)*width,scale_at=uint64_t(size_)*heads_*groups();
        std::copy(kc.begin(),kc.end(),key_codes_.begin()+at);std::copy(vc.begin(),vc.end(),value_codes_.begin()+at);
        std::copy(ks.begin(),ks.end(),key_scales_.begin()+scale_at);std::copy(vs.begin(),vs.end(),value_scales_.begin()+scale_at);
    } else if(q4()) {
        // The drafter's dense pool rotates and packs exactly like the trunk (src/core/mtp.cpp:346-364 rotates a Q4
        // drafter); the gathered rows are rotated back for its attention recipe.
        const auto row_bytes=q4_bytes();
        std::vector<float> kr(width),vr(width);
        kv_q4_rotate_rows(key,kr);kv_q4_rotate_rows(value,vr);
        std::vector<uint8_t> kq(uint64_t(heads_)*row_bytes),vq(kq.size());
        for(uint32_t head=0;head<heads_;++head) {
            const auto head_at=uint64_t(head)*width_;
            kv_q4_quantize_row(std::span<const float>(kr).subspan(head_at,width_),
                               std::span<uint8_t>(kq).subspan(uint64_t(head)*row_bytes,row_bytes));
            kv_q4_quantize_row(std::span<const float>(vr).subspan(head_at,width_),
                               std::span<uint8_t>(vq).subspan(uint64_t(head)*row_bytes,row_bytes));
        }
        const auto at=uint64_t(size_)*heads_*row_bytes;
        std::copy(kq.begin(),kq.end(),q4_key_.begin()+at);std::copy(vq.begin(),vq.end(),q4_value_.begin()+at);
    } else {
        for(const auto row:{key,value}) for(float x:row)
            if(!std::isfinite(x) || !std::isfinite(kernels::f32_from_f16(kernels::f16_from_f32(x)))) throw std::invalid_argument("dense KV requires finite FP16 values");
        for(size_t i=0;i<width;++i) {
            key_[uint64_t(size_)*width+i]=kernels::f16_from_f32(key[i]);value_[uint64_t(size_)*width+i]=kernels::f16_from_f32(value[i]);
        }
    }
    prefixes_[++size_]=++generation_;
}
SelectedKv DenseKvState::gather(uint32_t bucket,uint32_t window) const {
    return gather_prefix(size_,bucket,window);
}
SelectedKv DenseKvState::gather_prefix(uint32_t cells,uint32_t bucket,uint32_t window) const {
    if(!cells || cells>size_) throw std::invalid_argument("dense KV prefix is not committed");
    const auto count=window?std::min(cells,window):cells;
    if(!count || bucket<count || bucket>capacity_) throw std::invalid_argument("dense KV gather bucket mismatch");
    const auto elements=checked_product(bucket,checked_product(heads_,width_));
    SelectedKv out{count,bucket,std::vector<float>(elements),std::vector<float>(elements),std::vector<float>(bucket,-INFINITY)};
    std::fill_n(out.mask.begin(),count,0);
    const bool quantized=int8();
    std::vector<float> unrotated(q4()?width_:0);
    for(uint32_t head=0;head<heads_;++head) for(uint32_t cell=0;cell<count;++cell) {
        const auto live=cells-count+cell;
        const auto src=(uint64_t(live)*heads_+head)*width_,dst=(uint64_t(head)*bucket+cell)*width_;
        if(quantized) {
            const auto scale_src=(uint64_t(live)*heads_+head)*groups();
            for(uint32_t i=0;i<width_;++i) {
                out.key[dst+i]=kv_int8_value(key_codes_[src+i],key_scales_[scale_src+i/kv_int8_group]);
                out.value[dst+i]=kv_int8_value(value_codes_[src+i],value_scales_[scale_src+i/kv_int8_group]);
            }
        } else if(q4()) {
            const auto row_bytes=q4_bytes();
            const auto stored=(uint64_t(live)*heads_+head)*row_bytes;
            kv_q4_dequantize_unrotated(std::span<const uint8_t>(q4_key_).subspan(stored,row_bytes),unrotated);
            std::copy(unrotated.begin(),unrotated.end(),out.key.begin()+dst);
            kv_q4_dequantize_unrotated(std::span<const uint8_t>(q4_value_).subspan(stored,row_bytes),unrotated);
            std::copy(unrotated.begin(),unrotated.end(),out.value.begin()+dst);
        } else {
            for(uint32_t i=0;i<width_;++i) {out.key[dst+i]=kernels::f32_from_f16(key_[src+i]);out.value[dst+i]=kernels::f32_from_f16(value_[src+i]);}
        }
    }
    return out;
}
uint64_t DenseKvState::allocated_bytes() const {
    const uint64_t parts[]{snapshot_vector_bytes(key_),snapshot_vector_bytes(value_),
        snapshot_vector_bytes(key_codes_),snapshot_vector_bytes(value_codes_),
        snapshot_vector_bytes(key_scales_),snapshot_vector_bytes(value_scales_),
        snapshot_vector_bytes(q4_key_),snapshot_vector_bytes(q4_value_),
        snapshot_vector_bytes(prefixes_)};
    uint64_t bytes=0;for(const auto part:parts) bytes=snapshot_add_bytes(bytes,part);
    return bytes;
}
DenseKvState::Checkpoint DenseKvState::checkpoint() const {Checkpoint p;p.owner=owner_;p.epoch=epoch_;p.cells=size_;p.prefix=prefixes_[size_];return p;}
uint64_t DenseKvState::parked_bytes() const {
    return parked_bytes(size_);
}
uint64_t DenseKvState::parked_bytes(uint64_t cells) const {
    if(cells>capacity_) throw std::out_of_range("draft parked extent exceeds capacity");
    const uint64_t per_cell=kv_token_bytes(width_,heads_,kv_);
    return checked_product(cells,per_cell);
}
void DenseKvState::snapshot_sizes(const Checkpoint& point,uint64_t (&sizes)[5]) const {
    if(int8()) {
        sizes[0]=uint64_t(point.key_codes.size())*sizeof(int8_t);sizes[1]=uint64_t(point.value_codes.size())*sizeof(int8_t);
        sizes[2]=uint64_t(point.key_scales.size())*sizeof(uint16_t);sizes[3]=uint64_t(point.value_scales.size())*sizeof(uint16_t);
    } else if(q4()) {
        sizes[0]=uint64_t(point.q4_key.size());sizes[1]=uint64_t(point.q4_value.size());
        sizes[2]=0;sizes[3]=0;
    } else {
        sizes[0]=uint64_t(point.key.size())*sizeof(uint16_t);sizes[1]=uint64_t(point.value.size())*sizeof(uint16_t);
        sizes[2]=0;sizes[3]=0;
    }
    sizes[4]=0;   // the draft pool has no indexer rows
}
DenseKvState::Checkpoint DenseKvState::snapshot_reused(Checkpoint&& prior,uint32_t keep_cells) const {
    // Three separate refusals, each named: a foreign/unowned image, an image wider than the live pool, and an
    // image whose ancestry was rewritten by a rewind-and-reappend (the rule restore() already uses).
    if(!prior.persistent || prior.owner!=owner_) throw std::invalid_argument("draft KV reused snapshot is foreign");
    if(prior.cells>size_ || prior.cells>capacity_ || keep_cells>prior.cells)
        throw std::invalid_argument("draft KV reused snapshot exceeds the live extent");
    if(prior.cells && prior.prefix!=prefixes_[prior.cells])
        throw std::invalid_argument("draft KV reused snapshot has a rewritten prefix");
    const auto per_cell=checked_product(heads_,width_);
    const auto prior_elements=checked_product(uint64_t(prior.cells),per_cell);
    const auto prior_scales=checked_product(uint64_t(prior.cells),checked_product(heads_,groups()));
    const auto row_bytes=q4()?q4_bytes():0;
    const auto prior_q4=checked_product(uint64_t(prior.cells),checked_product(heads_,row_bytes));
    const bool shaped=int8()?prior.key_codes.size()==prior_elements && prior.value_codes.size()==prior_elements &&
                             prior.key_scales.size()==prior_scales && prior.value_scales.size()==prior_scales
                           :q4()?prior.q4_key.size()==prior_q4 && prior.q4_value.size()==prior_q4
                                :prior.key.size()==prior_elements && prior.value.size()==prior_elements;
    if(!shaped) throw std::invalid_argument("draft KV reused snapshot shape mismatch");
    Checkpoint point=std::move(prior);
    if(int8()) {
        snapshot_resize(point.key_codes,checked_product(uint64_t(size_),per_cell));
        snapshot_resize(point.value_codes,point.key_codes.size());
        snapshot_resize(point.key_scales,checked_product(uint64_t(size_),checked_product(heads_,groups())));
        snapshot_resize(point.value_scales,point.key_scales.size());
        for(uint32_t cell=keep_cells;cell<size_;++cell) {
            const auto at=uint64_t(cell)*per_cell;
            std::copy_n(key_codes_.begin()+at,per_cell,point.key_codes.begin()+at);
            std::copy_n(value_codes_.begin()+at,per_cell,point.value_codes.begin()+at);
            const auto scale_at=uint64_t(cell)*heads_*groups();
            std::copy_n(key_scales_.begin()+scale_at,heads_*groups(),point.key_scales.begin()+scale_at);
            std::copy_n(value_scales_.begin()+scale_at,heads_*groups(),point.value_scales.begin()+scale_at);
        }
    } else if(q4()) {
        const auto live_bytes=uint64_t(size_)*heads_*row_bytes;
        snapshot_resize(point.q4_key,live_bytes);snapshot_resize(point.q4_value,live_bytes);
        std::copy_n(q4_key_.begin()+uint64_t(keep_cells)*heads_*row_bytes,live_bytes-uint64_t(keep_cells)*heads_*row_bytes,
                    point.q4_key.begin()+uint64_t(keep_cells)*heads_*row_bytes);
        std::copy_n(q4_value_.begin()+uint64_t(keep_cells)*heads_*row_bytes,live_bytes-uint64_t(keep_cells)*heads_*row_bytes,
                    point.q4_value.begin()+uint64_t(keep_cells)*heads_*row_bytes);
    } else {
        snapshot_resize(point.key,uint64_t(size_)*per_cell);snapshot_resize(point.value,point.key.size());
        std::copy_n(key_.begin()+uint64_t(keep_cells)*per_cell,uint64_t(size_-keep_cells)*per_cell,point.key.begin()+uint64_t(keep_cells)*per_cell);
        std::copy_n(value_.begin()+uint64_t(keep_cells)*per_cell,uint64_t(size_-keep_cells)*per_cell,point.value.begin()+uint64_t(keep_cells)*per_cell);
    }
    point.cells=size_;point.prefix=prefixes_[size_];point.epoch=epoch_;
    return point;
}
DenseKvState::Checkpoint DenseKvState::snapshot() const {
    auto p=checkpoint();p.persistent=true;
    const auto count=uint64_t(size_)*heads_*width_;
    if(int8()) {
        const auto scales=uint64_t(size_)*heads_*groups();
        p.key_codes.assign(key_codes_.begin(),key_codes_.begin()+count);p.value_codes.assign(value_codes_.begin(),value_codes_.begin()+count);
        p.key_scales.assign(key_scales_.begin(),key_scales_.begin()+scales);p.value_scales.assign(value_scales_.begin(),value_scales_.begin()+scales);
    } else if(q4()) {
        const auto bytes=uint64_t(size_)*heads_*q4_bytes();
        p.q4_key.assign(q4_key_.begin(),q4_key_.begin()+bytes);p.q4_value.assign(q4_value_.begin(),q4_value_.begin()+bytes);
    } else {
        p.key.assign(key_.begin(),key_.begin()+count);p.value.assign(value_.begin(),value_.begin()+count);
    }
    return p;
}
void DenseKvState::restore(const Checkpoint& p) {
    const auto count=uint64_t(p.cells)*heads_*width_;
    const auto q4_count=q4()?uint64_t(p.cells)*heads_*q4_bytes():0;
    if(p.owner!=owner_ || p.cells>capacity_ || (!p.persistent && (p.epoch!=epoch_ || p.cells>size_ || p.prefix!=prefixes_[p.cells])) ||
       (p.persistent && (int8()?p.key_codes.size()!=count || p.value_codes.size()!=count ||
                                 p.key_scales.size()!=uint64_t(p.cells)*heads_*groups() ||
                                 p.value_scales.size()!=uint64_t(p.cells)*heads_*groups()
                               :q4()?p.q4_key.size()!=q4_count || p.q4_value.size()!=q4_count
                                    :p.key.size()!=count || p.value.size()!=count)))
        throw std::invalid_argument("stale/foreign dense KV checkpoint");
    if(p.persistent) {
        const auto generation=std::max(generation_,p.prefix);
        if(uint64_t(p.cells)>UINT64_MAX-generation) throw std::overflow_error("draft KV snapshot lineage exhausted");
        if(int8()) {
            std::copy(p.key_codes.begin(),p.key_codes.end(),key_codes_.begin());
            std::copy(p.value_codes.begin(),p.value_codes.end(),value_codes_.begin());
            std::copy(p.key_scales.begin(),p.key_scales.end(),key_scales_.begin());
            std::copy(p.value_scales.begin(),p.value_scales.end(),value_scales_.begin());
        } else if(q4()) {
            std::copy(p.q4_key.begin(),p.q4_key.end(),q4_key_.begin());
            std::copy(p.q4_value.begin(),p.q4_value.end(),q4_value_.begin());
        } else {
            std::copy(p.key.begin(),p.key.end(),key_.begin());std::copy(p.value.begin(),p.value.end(),value_.begin());
        }
        ++epoch_;generation_=generation;prefixes_[0]=0;
        for(uint32_t cell=1;cell<=p.cells;++cell) prefixes_[cell]=++generation_;
        if(p.cells) prefixes_[p.cells]=p.prefix;
    }
    size_=p.cells;
}
void DenseKvState::reset() {size_=0;++epoch_;prefixes_[0]=0;}
}
