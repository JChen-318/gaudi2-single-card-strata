#pragma once
#include "strata/hpu/dense_kv.hpp"
#include "strata/hpu/hc_ple.hpp"
#include "strata/hpu/mtp_weights.hpp"
#include "strata/hpu/recipe.hpp"
#include "strata/hpu/mtp_window_flow.hpp"

namespace strata::hpu {
void define_mtp_input(GraphBuilder&,uint32_t hidden,uint32_t streams);
void define_mtp_projections(GraphBuilder&,const QsaConfig&,uint32_t hidden);
struct MtpOptions {
    uint32_t context=32768,window=0;
    uint64_t position_base=0,expert_cache=32ull<<20,staging=1ull<<20;
    bool stream_norm=false,q4_projections=false,q4_head=false;
    // The drafter's dense KV follows the trunk's --kv format (upstream chooses both from the same globals).
    KvMode kv=KvMode::Fp16;
    std::vector<int32_t> draft_vocab;
    strata::kernels::RopeScaling rope;
    // Request-scoped image positions (--vision); null is the text-only rotation.
    const RopePositions* positions=nullptr;
};
// One full native draft layer, with separate dense KV and a bounded expert
// cache. Six FC/QKV/O projections retain Q8_0 bytes or bounded Q4_0 copies,
// controlled by --mtp-q4. Both use host Q8_1
// activation staging plus packed TPC dots. Shared Q8 and routed Q2 experts use
// their distinct packed activation contracts; HC/router remain BF16 reference.
// CUDA/HPU reduction-bitwise parity is unclaimed.
class MtpBlock {
public:
    MtpBlock(const Device&,MtpWeights,Binding shared_output_head,MtpOptions={},WeightSlice output_head_source={});
    ~MtpBlock();
    // Pairs the main/draft residual at cell i with embedding(token[i+1]);
    // the returned logits predict token[i+2]. Caller synchronizes acceptance.
    MtpOutput step(std::span<const float> previous_residual,std::span<const float> next_embedding);
    // Known target prefix only: explicit previous residual/current embedding
    // pairs, atomic1..128 rows, final-row output. Independent of draft chaining.
    MtpOutput prefill(std::span<const float> previous_residuals,std::span<const float> next_embeddings,const std::function<bool()>& cancelled={});
    DenseKvState::Checkpoint checkpoint() const;
    DenseKvState::Checkpoint snapshot() const;
    uint64_t parked_bytes() const;
    uint64_t parked_bytes(uint64_t cells) const;
    // The draft pool's reuse-aware park. `unchanged_tokens` is the count the TARGET conversation reports; this
    // call applies upstream's draft-layer cap `max(0, unchanged - 1)` itself, because the draft layer always
    // re-refreshes the page it is still writing (src/core/conversation_state.cpp:291-294) and a caller that
    // forgot the cap would trust a page that is about to be rewritten. Returns the bytes carried over.
    uint64_t snapshot_reused(DenseKvState::Checkpoint&& prior,uint64_t unchanged_tokens,
                             DenseKvState::Checkpoint& out) const;
    void restore(const DenseKvState::Checkpoint&);
    void reset();
    uint32_t size() const;
    bool healthy() const;
    void write_snapshot(StateWriter&,const DenseKvState::Checkpoint&) const;
    DenseKvState::Checkpoint read_snapshot(StateReader&,uint32_t cells) const;
private:
    struct Impl;std::unique_ptr<Impl> impl_;
};
}
