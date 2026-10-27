#pragma once
#include "strata/hpu/layer_flow.hpp"
namespace strata::hpu {
inline constexpr uint32_t kMtpWindowRows=8;
struct MtpOutput {std::vector<float> residual,logits;};
struct MtpPairs {std::vector<float> residual,embedding;};
// Empty root means a new session: row0 has no previous target residual.
MtpPairs pair_mtp_prefix(std::span<const float> target_residual,std::span<const float> embedding,std::span<const float> root,uint32_t hidden,uint32_t streams);
struct MtpInputNorms {std::span<const float> embedding,hidden;float epsilon=1e-6f;bool stream_norm=false;};
struct MtpWindowOperations {
    std::function<std::vector<float>(std::span<const float>)> embedding,hidden,head;
    LayerOperations layer;
    // Idempotent owner closure; also used by layer.rollback_mixer. A failed
    // restore poisons the owner. Restores the entire known-prefix transaction.
    std::function<void()> rollback;
};
// Explicit [T,HC,H] previous target residuals paired with [T,H] current token
// embeddings. Native chunks<=8, FC rows<=128. Only the final full-vocab head
// is materialized. No autoregressive drafts or caller history changes here.
MtpOutput execute_mtp_window(std::span<const float> residual,std::span<const float> embedding,uint32_t hidden,uint32_t streams,uint32_t vocabulary,MtpInputNorms,const MtpWindowOperations&,const std::function<bool()>& cancelled={});
}
