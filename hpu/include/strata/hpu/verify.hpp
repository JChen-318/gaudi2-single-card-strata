#pragma once
#include "strata/hpu/sampler.hpp"
#include <functional>

namespace strata::hpu {
enum class VerifyMode { Exact, Probabilistic };
// Empty q means a point mass, as for prompt lookup. Otherwise this is the
// drafter's own post-sampling-chain distribution, never the target's cut.
struct DraftRow {
    int32_t token=-1;
    std::vector<int32_t> ids;
    std::vector<float> probabilities;
};
struct VerifyPick {int32_t token;bool accepted;};
struct VerifiedCandidates {
    std::vector<VerifyPick> picks; // accepted draft prefix then mismatch/bonus, or EOS
    uint32_t retained=0; // target-fed draft prefix; EOS is never fed
    bool stopped=false;
    std::vector<float> logits; // native hook sets the retained prefix's logits
    uint32_t follow_differ=0,follow_emitted=0; // only the selected, published prefix
};
void validate_drafts(std::span<const DraftRow>,uint32_t vocabulary,uint32_t maximum);
// Validate a backend's already committed accepted-prefix result before any
// token is published. Shared by ordinary Generation and slot windows.
bool valid_verified_candidates(const VerifiedCandidates&,std::span<const DraftRow>,uint32_t vocabulary,
    uint32_t output_limit,std::span<const int32_t> stop_ids);
struct SlotCandidates {DraftRow draft;VerifiedCandidates selected;};
// Two target rows fed [known pending token, one proposed successor]. Pending
// was emitted in the previous window; select only NEW outputs, in exact mode
// like upstream batch-MTP (generate.cpp:9986-10020). Never re-sample pending.
VerifiedCandidates select_slot_candidates(std::span<const float> after,const DraftRow&,int32_t pending,
    std::span<const int32_t> history,const strata::kernels::SamplerParams&,std::span<const int32_t> stop_ids);
// One target row. Exact mode uses the ordinary sampler, including Gumbel.
// Probabilistic mode reuses upstream spec_verify_row and its disjoint Philox
// streams; it preserves the target distribution, not the exact-mode token path.
VerifyPick verify_pick(std::span<const float> logits,std::span<const int32_t> history,
    const strata::kernels::SamplerParams&,std::span<const float> bias,
    const DraftRow*,VerifyMode);
// Pure selector shared by the native Model and host tests. after is bounded
// [draft_count,vocabulary], before is the committed prefix's distribution.
VerifiedCandidates select_verified_candidates(std::span<const float> before,std::span<const float> after,
    std::span<const DraftRow>,std::span<const int32_t> history,const strata::kernels::SamplerParams&,
    std::span<const float> bias,VerifyMode,std::span<const int32_t> stop_ids,const std::function<bool()>& cancelled={},
    std::span<const int32_t> follow={});
}
