#pragma once
#include "strata/core/geometry.hpp"
#include "strata/hpu/session_flow.hpp"
#include "strata/hpu/tensor.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <exception>
#include <functional>
#include <span>
#include <vector>

namespace strata::hpu {
enum class HalfField {AttentionBlock,FfnBlock,AttentionInject,FfnInject};
// Borrowed synchronous outputs of a real scalar QSA pass. The selected value
// is one token, all KV heads (the upstream token-major gather prefix); pool
// is the actual physical FP16 V pool prefix. No conversion through float bits.
struct QsaHalfView {
    std::span<const float> attention,raw_value,post_value,key;
    std::span<const uint16_t> selected_value,pool_value;
    std::array<int32_t,4> ids{-1,-1,-1,-1},step{};
};
using QsaHalfSink=std::function<void(const QsaHalfView&)>;
using LayerHalfSink=std::function<void(uint32_t,HalfField,std::span<const float>)>;
using LayerQsaHalfSink=std::function<void(uint32_t,const QsaHalfView&)>;
struct HalfLayout {
    uint32_t layers=0,hidden=0,streams=0,query=0,value=0,interval=0,block=0,selection_max=0;
    uint64_t qsa_offset=0,stride=0;
    HalfLayout()=default;
    explicit HalfLayout(const core::ModelGeometry& g) {
        if(g.n_layers<1 || g.n_layers>4096 || g.n_embd<1 || g.n_embd>65536 || g.hc<1 || g.hc>32 ||
           g.n_head<1 || g.n_head>1024 || g.n_head_kv<1 || g.n_head_kv>g.n_head || g.n_head%g.n_head_kv ||
           g.head_dim<1 || g.head_dim>65536 || g.qsa_interval<1 || g.qsa_interval>4096 || g.idx_block<2 || g.idx_block>64 ||
           g.idx_top_k<1 || uint64_t(g.idx_top_k)+uint64_t(g.idx_block)-1>INT32_MAX)
            throw std::invalid_argument("half capture geometry exceeds bounded layout");
        layers=uint32_t(g.n_layers);hidden=uint32_t(g.n_embd);streams=uint32_t(g.hc);
        query=uint32_t(g.n_head*g.head_dim);value=uint32_t(g.n_head_kv*g.head_dim);
        interval=uint32_t(g.qsa_interval);block=uint32_t(g.idx_block);
        selection_max=uint32_t(g.idx_top_k+g.idx_block-1);
        if(value%2) throw std::invalid_argument("half capture needs complete FP16 pairs");
        qsa_offset=2ull*hidden+2ull*streams;stride=qsa_offset+query+5ull*value+8;
    }
    bool qsa(uint32_t layer) const {return interval && layer%interval==interval-1;}
    uint64_t elements() const {return checked_product(layers,stride);}
    uint64_t host_bytes() const {return checked_product(elements(),8)+2ull*layers+4096;} // staging + ONE export
};
struct HalfProducerMemory {uint64_t host_bytes=0,pinned_bytes=0,hbm_bytes=0;};
// Incremental opt-in envelope, to be charged by Model before device creation:
// staging/export, selected FP16 scratch, persistent pre-gate output/readback,
// descriptor overhead and a separately tested compiler-workspace allowance.
inline HalfProducerMemory plan_half_producer(const core::ModelGeometry& g) {
    const HalfLayout layout(g);const uint64_t qsa=layout.layers/layout.interval;
    const auto tap=uint64_t(layout.query)*4;
    // Model serializes layers, but a bucket replacement constructs its new
    // HostRecipe before releasing the previous one. Price ONE additional live
    // readback/persistent output, descriptors and compiler workspace at peak.
    const auto recipes=qsa+(qsa?1:0);
    return {layout.host_bytes()+qsa*uint64_t(layout.value)*2+recipes*(tap+(16ull<<10)),recipes*tap,recipes*(tap+(4ull<<20))};
}
// One scalar physical target pass. Reserved/unassigned upstream slots are
// deterministic zero padding here, never advertised as computed QSA values.
// Missing required fields are fatal; packed data is NOT finite-checked as F32.
class HalfCapture {
public:
    static uint64_t host_bytes(const core::ModelGeometry& g) {return HalfLayout(g).host_bytes();}
    void configure(const core::ModelGeometry& g,uint64_t budget) {
        if(enabled()) throw std::logic_error("half capture is configured once per model");
        HalfLayout layout(g);if(layout.host_bytes()>budget) throw std::invalid_argument("half capture exceeds host memory policy");
        std::vector<float> data(size_t(layout.elements()),0);std::vector<uint8_t> expected(layout.layers),seen(layout.layers);
        if(data.capacity()!=layout.elements() || expected.capacity()!=layout.layers || seen.capacity()!=layout.layers)
            throw std::length_error("half capture capacities exceed admitted envelope");
        for(uint32_t layer=0;layer<layout.layers;++layer) expected[layer]=layout.qsa(layer)?31:15;
        layout_=layout;data_=std::move(data);expected_=std::move(expected);seen_=std::move(seen);clear();
    }
    bool enabled() const {return !data_.empty();}
    void clear() {active_=ready_=false;std::fill(seen_.begin(),seen_.end(),0);}
    uint64_t storage_bytes() const {return data_.capacity()*4+expected_.capacity()+seen_.capacity();}
    std::span<const float> view() const {return ready_?std::span<const float>(data_):std::span<const float>{};}
    std::vector<float> records() const {
        if(!ready_) return {};
        std::vector<float> out(data_);if(out.capacity()!=out.size()) throw std::length_error("half export exceeds admitted capacity");return out;
    }
    class Round {
    public:
        explicit Round(HalfCapture& owner,bool scalar=true):owner_(owner),capture_(owner.enabled() && scalar) {
            if(!capture_) {owner.clear();return;}
            exceptions_=std::uncaught_exceptions();owner.clear();std::fill(owner.data_.begin(),owner.data_.end(),0.0f);owner.active_=true;
        }
        Round(const Round&)=delete;
        ~Round() {if(!finished_ || (capture_ && std::uncaught_exceptions()>exceptions_)) owner_.clear();}
        void finish() {
            if(capture_ && owner_.seen_!=owner_.expected_) throw SessionFatal("half capture did not observe every required field/layer");
            owner_.active_=false;owner_.ready_=capture_;finished_=true;
        }
    private:
        HalfCapture& owner_;bool capture_=false,finished_=false;int exceptions_=0;
    };
    Round round(bool scalar=true) {return Round(*this,scalar);}
    void half(uint32_t layer,HalfField field,std::span<const float> values) {
        if(!active_) return;
        const auto index=uint32_t(field);if(index>3) throw std::invalid_argument("unknown half field");
        const uint64_t count=index<2?layout_.hidden:layout_.streams;
        const uint64_t offset=index<2?uint64_t(index)*layout_.hidden:2ull*layout_.hidden+uint64_t(index-2)*layout_.streams;
        check_field(layer,uint8_t(1u<<index));finite(values,count);
        std::copy(values.begin(),values.end(),data_.begin()+uint64_t(layer)*layout_.stride+offset);seen_[layer]|=uint8_t(1u<<index);
    }
    void qsa(uint32_t layer,const QsaHalfView& v) {
        if(!active_) return;
        check_field(layer,16);if(!layout_.qsa(layer)) throw std::invalid_argument("QSA half field on a GDN layer");
        finite(v.attention,layout_.query);finite(v.raw_value,layout_.value);finite(v.post_value,layout_.value);finite(v.key,layout_.value);
        if(v.selected_value.size()!=layout_.value || v.pool_value.size()!=layout_.value || v.step[1]<1 || v.step[0]!=v.step[1]-1 ||
           v.step[2]!=v.step[1]/int32_t(layout_.block) || v.step[3]!=int32_t(std::min(uint32_t(v.step[1]),layout_.selection_max))) throw std::invalid_argument("QSA half bit/step geometry mismatch");
        for(uint32_t i=0;i<4;++i) {
            if(i<uint32_t(v.step[3])) {if(v.ids[i]<0 || v.ids[i]>=v.step[1] || (i && v.ids[i]<=v.ids[i-1])) throw std::invalid_argument("QSA half selected IDs invalid");}
            else if(v.ids[i]!=-1) throw std::invalid_argument("QSA half unused IDs need the declared sentinel");
        }
        auto* out=data_.data()+uint64_t(layer)*layout_.stride;
        const auto a=layout_.qsa_offset+layout_.query,n=uint64_t(layout_.value);
        std::copy(v.attention.begin(),v.attention.end(),out+layout_.qsa_offset);
        std::copy(v.raw_value.begin(),v.raw_value.end(),out+a);
        // Upstream's early K-pool debug slice is completely overwritten by
        // the later gathered-V/IDs/step writes. Preserve FINAL byte offsets.
        std::memcpy(out+a+5*n/2+8,v.pool_value.data(),size_t(n*2));
        std::copy(v.post_value.begin(),v.post_value.end(),out+a+3*n+8);
        std::copy(v.key.begin(),v.key.end(),out+a+4*n+8);
        std::memcpy(out+a+2*n,v.selected_value.data(),size_t(n*2));
        std::memcpy(out+a+5*n/2,v.ids.data(),16);std::memcpy(out+a+5*n/2+4,v.step.data(),16);
        seen_[layer]|=16;
    }
private:
    // Default layout has no buffers; dimensions are unused until configure.
    HalfLayout layout_;
    std::vector<float> data_;std::vector<uint8_t> expected_,seen_;bool active_=false,ready_=false;
    void check_field(uint32_t layer,uint8_t bit) const {
        if(layer>=layout_.layers || (seen_[layer]&bit)) throw std::invalid_argument("half field layer/duplicate mismatch");
    }
    static void finite(std::span<const float> values,uint64_t count) {
        if(values.size()!=count) throw std::invalid_argument("half field shape mismatch");
        for(float x:values) if(!std::isfinite(x)) throw std::invalid_argument("non-finite computed half field");
    }
};
}
