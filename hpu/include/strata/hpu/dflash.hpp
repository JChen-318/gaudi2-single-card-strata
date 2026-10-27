#pragma once
#include "strata/hpu/host_recipe.hpp"
#include "strata/hpu/recipe.hpp"
#include "strata/hpu/runtime.hpp"
#include <array>
#include <map>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace strata::hpu {
// STRATA_HPU_DFLASH=<directory>: the DFlash 2 block-diffusion drafter (incoai/Qwen3.8-Flash-Next-DFlash2,
// model.safetensors + config.json) on the device, after sglang 0.5.22 srt/models/dflash.py (DFlash2DraftModel)
// and srt/speculative/dflash_worker_v2.py.
// Context: for every committed target token, the residual after target layers 5/15/25/35/45, averaged over the
// four hyper-connection streams, concatenated [12800] -> fc -> hidden_norm, becomes every draft layer's K (k_proj,
// k_norm, NeoX RoPE theta 1e4 over the whole head) and V (v_proj) at that token's position.
// Draft: the block [anchor, mask x (block-1)] at positions p..p+block-1 (target embeddings) through 5 layers of
// non-causal attention over [context cells; the block], each sublayer wrapped in a two-tap grouped dynamic
// convolution (16 channels per group) along the block; final norm; rows 1.. through the TARGET lm_head; top-16
// candidates per row; the selector scores score[e,p,c] = unary[e,c] + <pred[e,p] * proj(h[e]), succ[e,c]>
// (pred = the previous row's candidate, the anchor for row 1) and the greedy walk takes argmaxes edge by edge.
// The context sliding window (2048) is a ring of cells: the drafter attends to the last 2048 context positions.
struct DflashConfig {
    // block: rows per draft (anchor + block-1 drafted). config.json trains 8; the model card deploys 4 (three drafts
    // per verification, "the current QSA implementation"; acceptance length 2.83-3.26), and at 8 this port measured
    // 2.9 tokens per round, so 4 is the default. STRATA_HPU_DFLASH_BLOCK=2..8 overrides it.
    uint32_t hidden=2560,layers=5,heads=32,kv_heads=8,head_dim=128,intermediate=7680,block=4;
    uint32_t groups=160,taps=2,rank=256,top_k=16,window=2048;int32_t mask_token=248077;
    float epsilon=1e-6f,theta=10000.0f;
    std::array<uint32_t,5> target_layers{5,15,25,35,45};
};
class DflashDrafter {
public:
    // lm_head: the target's BF16 [vocabulary, hidden] head weight. capacity: the most context cells kept.
    // feature_rows: how many rows one capture pass may carry (a prefill window wider than a draft block is
    // captured once and appended to the context block by block).
    DflashDrafter(const Device&,const std::string& directory,Binding lm_head,uint32_t vocabulary,uint32_t capacity,
                  uint32_t feature_rows=1);
    ~DflashDrafter();
    DflashDrafter(const DflashDrafter&)=delete;
    DflashDrafter& operator=(const DflashDrafter&)=delete;
    const DflashConfig& config() const;
    // Index of `target_layer` among the captured layers, or -1. capture() writes that layer's stream mean of
    // `residual` [rows, streams, hidden] (rows 0.. of the feature slot); it queues on `stream` and never waits.
    int feature_slot(uint32_t target_layer) const;
    void capture(const Stream&,const DeviceBuffer& residual,uint32_t streams,int slot,uint32_t rows=1);
    // Captured rows first_row.. become context positions position.. (position == cells(); kept in a ring of the
    // window's cells, so any context length works); waits for `stream` unless
    // `synchronize` is false (then only work queued on the same stream may read them). first_row+count must fit the
    // capture rows, so a wide window appends its blocks one call at a time.
    void append(const Stream&,uint32_t position,uint32_t first_row=0,uint32_t count=1,bool synchronize=true);
    uint32_t cells() const;
    // Forget the context cells from `cells` on (no draft may be in flight).
    void truncate(uint32_t cells);
    void reset();
    // Draft tokens following `anchor` (already the next token, at `position` == cells()): block-1 of them.
    std::vector<int32_t> draft(std::span<const float> anchor_embedding,std::span<const float> mask_embedding,int32_t anchor,uint32_t position);
    // draft() in two halves: begin_draft() queues it on `stream` (behind that stream's appends), end_draft() waits
    // and walks the selector. One draft at a time; nothing else may use the drafter in between.
    void begin_draft(const Stream&,std::span<const float> anchor_embedding,std::span<const float> mask_embedding,int32_t anchor,uint32_t position);
    std::vector<int32_t> end_draft();
    bool drafting() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
