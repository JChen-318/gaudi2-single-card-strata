#pragma once
#include "strata/hpu/snapshot_memory.hpp"
#include "strata/hpu/qsa_state.hpp"

namespace strata::hpu {
// MTP's own dense causal KV: no pooled keys or indexer selection. Cells are kept on the host; gather stages only
// the active attention bucket. --kv int8 stores each 64-value group as codes plus one FP16 scale, and --kv q4_0 the
// Hadamard-rotated q4_0 blocks (kv_q4_cell.hpp); every reader still sees the FP32 rows attention consumes. There is
// no hybrid layout here: under --kv k8v4 the drafter is built as plain INT8 (src/core/mtp.cpp:344-349), and a
// direct KvMode::K8V4 construction is refused rather than silently stored as FP16.
class DenseKvState {
public:
    DenseKvState(uint32_t capacity,uint32_t heads,uint32_t width,KvMode kv=KvMode::Fp16);
    DenseKvState(const DenseKvState&)=delete;
    DenseKvState& operator=(const DenseKvState&)=delete;
    void append(std::span<const float> key,std::span<const float> value);
    SelectedKv gather(uint32_t bucket,uint32_t window=0) const;
    // Immutable causal prefix, even after later cells have been appended.
    SelectedKv gather_prefix(uint32_t cells,uint32_t bucket,uint32_t window=0) const;
    uint32_t size() const {return size_;}
    uint32_t capacity() const {return capacity_;}
    uint32_t heads() const {return heads_;}
    uint32_t width() const {return width_;}
    bool int8() const {return kv_==KvMode::Int8;}
    bool q4() const {return kv_==KvMode::Q4;}
    uint64_t allocated_bytes() const;
    struct Checkpoint {
    private:
        uint64_t owner=0,epoch=0,prefix=0;
        uint32_t cells=0;
        bool persistent=false;
        std::vector<uint16_t> key,value;
        std::vector<int8_t> key_codes,value_codes;
        std::vector<uint16_t> key_scales,value_scales;
        std::vector<uint8_t> q4_key,q4_value;
        friend class DenseKvState;
    };
    Checkpoint checkpoint() const;
    Checkpoint snapshot() const;
    static uint64_t snapshot_capacity_bytes(const Checkpoint& p) {
        uint64_t bytes=0;
        for(auto n:{snapshot_vector_bytes(p.key),snapshot_vector_bytes(p.value),snapshot_vector_bytes(p.key_codes),snapshot_vector_bytes(p.value_codes),snapshot_vector_bytes(p.key_scales),snapshot_vector_bytes(p.value_scales),snapshot_vector_bytes(p.q4_key),snapshot_vector_bytes(p.q4_value)}) bytes=snapshot_add_bytes(bytes,n);
        return bytes;
    }
    // The parked share of this pool: the K/V rows a snapshot copies, at the live cell count.
    uint64_t parked_bytes() const;
    uint64_t parked_bytes(uint64_t cells) const;
    // The draft pool's parked parts in upstream's order, and the grow-in-place park (see QsaState): the cells
    // before `keep_cells` stay in the retained image and only the rest is re-read from the live pool.
    void snapshot_sizes(const Checkpoint&,uint64_t (&sizes)[5]) const;
    Checkpoint snapshot_reused(Checkpoint&& prior,uint32_t keep_cells) const;
    void restore(const Checkpoint&);
    void reset();
    void write_snapshot(StateWriter&,const Checkpoint&) const;
    Checkpoint read_snapshot(StateReader&,uint32_t expected_cells) const;
private:
    uint32_t capacity_,heads_,width_,size_=0;
    KvMode kv_=KvMode::Fp16;
    uint64_t owner_,epoch_=0;
    mutable uint64_t generation_=0; // snapshot reads also mint a private lineage identity
    std::vector<uint16_t> key_,value_;
    std::vector<int8_t> key_codes_,value_codes_;
    std::vector<uint16_t> key_scales_,value_scales_;
    std::vector<uint8_t> q4_key_,q4_value_;
    std::vector<uint64_t> prefixes_;
    uint32_t groups() const {return width_/64;}
    uint64_t q4_bytes() const {return kv_q4_bytes_per_head(width_);}
};
}
