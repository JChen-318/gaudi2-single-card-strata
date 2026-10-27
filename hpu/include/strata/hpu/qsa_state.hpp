#pragma once
#include "strata/hpu/snapshot_memory.hpp"
#include "strata/hpu/kv_mode.hpp"
#include "strata/hpu/kv_q4_cell.hpp"
#include "strata/core/geometry.hpp"
#include "strata/kernels/rope_scaling.hpp"
#include "strata/hpu/tensor.hpp"
#include <span>

namespace strata::hpu {
class StateWriter;
class StateReader;
struct RopeRow { std::vector<float> cosine,sine; };
// The engine's request-scoped image position table (upstream's mrope_host, src/program/generate.cpp:3197-3215):
// [cells][3] int32 (t, h, w) read by every rope site. `cells == 0` means the table is not installed, which is
// the default and reproduces rope_row(cell) exactly. With --vision the engine always installs a table: the
// identity until a GENI request, and the identity again for a text request.
struct RopePositions {
    explicit RopePositions(uint64_t cells=0);
    uint64_t cells() const {return cells_;}
    bool installed() const {return cells_!=0;}
    void install(std::span<const int32_t> table);
    void reset_identity();
    int32_t position(uint64_t cell,uint32_t pair) const;
private:
    uint64_t cells_=0;
    std::vector<int32_t> positions_;
};
// The rotation of cell `base` at `rotary`: the table's (t, h, w) sectors when one is installed, else the
// plain position the backend always used.
RopeRow rope_row_at(const RopePositions* positions,uint32_t rotary,uint64_t position,
                    const strata::kernels::RopeScaling&);
struct QsaConfig {
    uint32_t query_heads=24,kv_heads=2,width=256,rotary=64,index_heads=4,index_width=128,block=4,top_k=2048;
    uint32_t capacity=32768,page_size=4;
    // --kv. Storage only: int8 keeps the same FP16 scratch the attention path already consumes and the indexer
    // rows stay FP32, exactly like upstream (include/strata/kernels/kv_q8.hpp:10-12); q4_0 stores the Hadamard-
    // rotated ggml blocks and rotates a gathered row back before attention reads it (kv_q4_cell.hpp); k8v4 stores
    // unrotated INT8 K codes with their scales and rotated Q4_0 V blocks (src/core/layer.cpp:649-661,1080-1103).
    KvMode kv=KvMode::Fp16;
    float epsilon=1e-6f;
    uint64_t position_base=0;
    strata::kernels::RopeScaling rope;
    // Non-owning; the Model owns the table and outlives every state built from this config. Null keeps the
    // original text-only rotation path.
    const RopePositions* positions=nullptr;
    void validate() const;
    uint32_t selection_width(uint32_t cells) const;
};
QsaConfig qsa_config(const core::ModelGeometry&,uint32_t capacity,uint32_t page_size=4);
RopeRow rope_row(uint32_t rotary,uint64_t position,const strata::kernels::RopeScaling&);
// M-RoPE: pair i rotates at `t` (i % 3 == 0), `h` (1) or `w` (2), the table's stream
// (include/strata/kernels/mrope.hpp). t == h == w is exactly rope_row(t).
RopeRow rope_row_sectors(uint32_t rotary,uint64_t t,uint64_t h,uint64_t w,const strata::kernels::RopeScaling&);
void rotate_neox(std::span<float> rows,uint32_t width,const RopeRow&);
void rms_norm_rows(std::span<float> rows,uint32_t width,std::span<const float> gamma,float epsilon);
struct SelectedKv {
    uint32_t count=0,bucket=0;
    // Head-major [kv_heads,bucket,width]. FP16-rounded values widened to FP32
    // for native MME attention. Padding is zero and mask is -inf for padding.
    std::vector<float> key,value,mask;
};
// CPU control plane: exact FP16 KV storage and FP64 pooled-key math. No torch,
// CUDA or device API. Projections and attention use native Synapse recipes.
class QsaState {
public:
    explicit QsaState(QsaConfig,std::vector<uint32_t> physical_pages={});
    QsaState(const QsaState&)=delete;
    QsaState& operator=(const QsaState&)=delete;
    QsaState(QsaState&&)=delete;
    QsaState& operator=(QsaState&&)=delete;
    uint32_t size() const { return size_; }
    const QsaConfig& config() const { return config_; }
    uint64_t allocated_bytes() const;
    // k is already weighted-RMS-normalized and rotated; v is unrotated.
    // raw_index_key is unnormalized/unrotated, index_gamma is multiplicative.
    // Under --kv int8 both rows are stored as 64-value INT8 groups and every reader sees fp16(code*scale).
    // Under --kv q4_0 both rows are Hadamard-rotated here, quantized to ggml q4_0 blocks, and rotated back on gather.
    // Under --kv k8v4 only K is the INT8 group pair (unrotated, as upstream's ring stores it); V is rotated and
    // stored as q4_0 blocks, so it is rotated back on gather exactly like the q4_0 V pool.
    void append(std::span<const float> k,std::span<const float> v,std::span<const float> raw_index_key,
                std::span<const float> index_gamma);
    std::vector<float> cell_scores(std::span<const float> normalized_rotated_index_query,
                                   std::span<const float> block_bias={}) const;
    std::vector<int32_t> select(std::span<const float> normalized_rotated_index_query,
                                std::span<const float> block_bias={}) const;
    SelectedKv gather(std::span<const int32_t> ids,uint32_t bucket) const;
    // Same K/V/mask values as gather(), written straight into caller buffers ([kv_heads,bucket,width] each and
    // the shared [bucket] mask): no temporaries, so a window does not allocate or copy 8 MiB per row. Every
    // destination element is written, including the zero padding behind `ids`.
    void gather_into(std::span<const int32_t> ids,uint32_t bucket,std::span<float> key,std::span<float> value,
                     std::span<float> mask) const;
    // Half-dump source: actual physical V-pool prefix and one selected cell's
    // exact FP16 bits in token-major [HKV,D] order. No new state allocation.
    std::span<const uint16_t> value_pool_prefix() const;
    // STRATA_HPU_QSA_DEVICE: cells [first, first+count) of the FP16 pool, token-major [count, kv_heads, width],
    // widened exactly to F32 (the device cache's layout). FP16 pool only.
    void fp16_cells(uint32_t first,uint32_t count,std::span<float> keys,std::span<float> values) const;
    void selected_value_bits(int32_t cell,std::span<uint16_t> output) const;
    bool int8() const { return config_.kv==KvMode::Int8; }
    bool q4() const { return config_.kv==KvMode::Q4; }
    bool hybrid() const { return config_.kv==KvMode::K8V4; }
    struct Checkpoint {
    private:
        uint64_t owner=0,epoch=0,prefix=0;
        uint32_t cells=0;
        std::vector<float> tail,spare;
        bool persistent=false;
        std::vector<uint16_t> keys,values;
        std::vector<int8_t> key_codes,value_codes;
        std::vector<uint16_t> key_scales,value_scales;
        std::vector<uint8_t> q4_key,q4_value;
        std::vector<float> pooled;
        friend class QsaState;
    };
    // Prefix ancestry allows a child rewind followed by its still-accepted
    // parent, without copying the KV arena. Rewritten branches are rejected.
    Checkpoint checkpoint() const;
    // Owned active KV + pooled state. Unlike a rewind checkpoint, this can
    // be restored repeatedly after reset or after exploring another branch.
    Checkpoint snapshot() const;
    static uint64_t snapshot_capacity_bytes(const Checkpoint& p) {
        uint64_t bytes=0;
        for(auto n:{snapshot_vector_bytes(p.keys),snapshot_vector_bytes(p.values),snapshot_vector_bytes(p.key_codes),snapshot_vector_bytes(p.value_codes),snapshot_vector_bytes(p.key_scales),snapshot_vector_bytes(p.value_scales),snapshot_vector_bytes(p.q4_key),snapshot_vector_bytes(p.q4_value),snapshot_vector_bytes(p.pooled),snapshot_vector_bytes(p.tail),snapshot_vector_bytes(p.spare)}) bytes=snapshot_add_bytes(bytes,n);
        return bytes;
    }
    // The parked image's own part sizes in bytes, in upstream's order (K, V, K scale, V scale, pooled rows):
    // what conversation_kv_save compares its reuse boundary against, and where a keep larger than the answer is
    // reported as a missing reusable prefix rather than a shorter copy (src/core/conversation_snapshot.cpp:173).
    void snapshot_sizes(const Checkpoint&,uint64_t (&sizes)[5]) const;
    // Upstream's reuse branch of the park (src/core/conversation_snapshot.cpp:157-179). `prior` is the retained
    // image of THIS conversation, moved in the way upstream moves `reuse` into the fresh SavedConversation, and
    // is grown in place: the cells before `whole_cells` and the indexer rows before `pooled_rows_kept` are kept
    // as they are, everything from those boundaries on is re-read from the live pools. The two boundaries come
    // from the parked-page law (strata/hpu/parked_pages.hpp), never recomputed here.
    Checkpoint snapshot_reused(Checkpoint&& prior,uint32_t whole_cells,uint32_t pooled_rows_kept) const;
    void write_snapshot(StateWriter&,const Checkpoint&) const;
    Checkpoint read_snapshot(StateReader&,uint32_t expected_cells) const;
    void restore(const Checkpoint&);
    void reset();
    std::span<const float> pooled_rows() const;
    // The raw index keys of the cells after the last complete block (size % block rows).
    std::span<const float> tail_rows() const;
private:
    QsaConfig config_;
    std::vector<uint32_t> pages_;
    // Exactly one of the four pool layouts is populated: FP16 carries the two-bit-per-value image, INT8 the codes
    // and their one FP16 scale per 64 values, Q4 the rotated 18-byte blocks, and K8V4 the INT8 K half (codes +
    // scales) with the rotated Q4_0 V blocks. The prefix decode buffer is allocated up front so the half-dump
    // accessors never grow state.
    std::vector<uint16_t> keys_,values_;
    std::vector<int8_t> key_codes_,value_codes_;
    std::vector<uint16_t> key_scales_,value_scales_;
    std::vector<uint8_t> q4_keys_,q4_values_;
    mutable std::vector<uint16_t> value_prefix_bits_;
    std::vector<float> tail_,spare_,pooled_;
    uint32_t size_=0;
    uint64_t owner_,epoch_=0;
    mutable uint64_t generation_=0; // snapshot reads also mint a private lineage identity
    std::vector<uint64_t> prefixes_; // bounded ancestry, no full KV copies for nested rewind
    uint64_t row(uint32_t cell,uint32_t head) const;
    // One float per indexer block; block b covers cells [b*block, min(size_, (b+1)*block)).
    std::vector<float> block_scores(std::span<const float> query,std::span<const float> bias) const;
    void check_ids(std::span<const int32_t> ids) const;
    void gather_rows(std::span<const int32_t> ids,uint32_t bucket,float* key,float* value) const;
    uint64_t scale_row(uint32_t cell,uint32_t head) const;
    uint64_t q4_row(uint32_t cell,uint32_t head) const;
    uint64_t q4_bytes() const {return kv_q4_bytes_per_head(config_.width);}
    uint32_t groups() const {return config_.width/64;}
};
std::vector<float> qsa_attention_reference(std::span<const float> query,const SelectedKv&,
                                          uint32_t query_heads,uint32_t kv_heads,uint32_t width);
}
