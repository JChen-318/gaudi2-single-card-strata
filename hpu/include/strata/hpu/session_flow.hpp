#pragma once
#include <cstdint>
#include <functional>
#include <span>
#include <stdexcept>
#include <vector>

namespace strata::hpu {
class SessionFatal:public std::runtime_error {public:using std::runtime_error::runtime_error;};
// Numerical failure with a valid shape; native operations rewind before publication.
class NonFiniteOutput:public std::runtime_error {public:using std::runtime_error::runtime_error;};
class Cancelled:public std::runtime_error {
public: Cancelled():std::runtime_error("generation cancelled") {}
};
struct SessionOperations {
    std::function<std::vector<float>(int32_t)> embedding;
    std::function<std::vector<float>(int32_t,std::span<const int32_t>)> ple;
    // Returns an owned rewind closure. A throwing layer must already have
    // rolled itself back; the session rewinds only layers that returned.
    std::function<std::function<void()>(uint32_t)> checkpoint;
    std::function<std::vector<float>(uint32_t,std::span<const float>,std::span<const float>)> layer;
    // Optional atomic layer-major window. Inputs [T,HC,H], [T,H] embeddings,
    // and PLE [T,H] at layer1 only. A throwing callback restores itself; the
    // session restores previously completed layers. No callback may escape.
    std::function<std::vector<float>(uint32_t,std::span<const float>,std::span<const float>,std::span<const float>,const std::function<bool()>&)> window_layer;
    // Optional stateless post-FFN transform. Invoked after recording the layer
    // as completed, so a transform failure rewinds that layer too.
    std::function<std::vector<float>(uint32_t,std::span<const float>)> after_layer;
    std::function<std::vector<float>(std::span<const float>)> head;
    // Optional stateless candidate head: [T,HC,H] -> owned [T,V].
    // Shape/finite/cancellation checks precede candidate selection and commit.
    std::function<std::vector<float>(std::span<const float>,const std::function<bool()>&)> window_head;
    std::function<void(uint32_t)> reset_layer;
    // Optional whole-token layer chain (STRATA_HPU_CHAIN): runs every layer itself from the input residual
    // and returns the last layer's output. It calls begin(layer) before a layer touches its state (the
    // session takes that layer's rewind point there) and done(layer) once the layer has committed, so a
    // failure rewinds exactly the committed layers, as the per-layer loop does. Intermediate residuals are
    // not shape/finite checked one by one; the final one is. Ignored when after_layer is set.
    struct ChainHooks {std::function<void(uint32_t)> begin,done;};
    std::function<std::vector<float>(std::span<const float>,std::span<const float>,const ChainHooks&)> chain;
};
struct SessionVerifyResult {
    uint32_t accepted=0; // retained candidate prefix, excluding a stop token
    std::vector<float> logits; // after retained prefix; empty if accepted==0
};
// Receives [candidate_count,vocabulary] logits AFTER each proposed token.
// The caller owns the distribution before candidate0. Return the candidate
// prefix to retain; no callback may re-enter the session or escape this call.
using CandidateSelection=std::function<uint32_t(std::span<const float>)>;
// Actual shared orchestration used by the native Model. Serialized by owner.
// Each token commits only after all layers and logits validate. Prefill keeps
// its successfully accepted prefix if a subsequent token fails or cancels.
class SessionFlow {
public:
    SessionFlow(uint32_t hidden,uint32_t streams,uint32_t layers,uint32_t vocabulary,uint32_t capacity,SessionOperations);
    std::vector<float> decode(int32_t token,const std::function<bool()>& cancelled={});
    std::vector<float> prefill(std::span<const int32_t>,const std::function<bool()>& cancelled={});
    // One atomic 1..128-token window. Only the final row needs target logits;
    // all token/layer state commits together after the head and cancel check.
    std::vector<float> prefill_window(std::span<const int32_t>,const std::function<bool()>& cancelled={});
    // Atomic1..7 candidate target window. Its explicitly bounded logits slab
    // is allocated before layer state changes. Full acceptance reuses computed
    // state; a shorter prefix rewinds once then replays only that prefix using
    // fresh checkpoints. Failure/cancel restores the starting target/MTP state.
    SessionVerifyResult verify_window(std::span<const int32_t>,uint64_t logits_budget,
                                     const CandidateSelection&,const std::function<bool()>& cancelled={});
    void reset();
    std::span<const int32_t> tokens() const {return history_;}
    bool poisoned() const {return poisoned_;}
private:
    uint32_t hidden_,streams_,layers_,vocabulary_,capacity_;
    SessionOperations operations_;
    std::vector<int32_t> history_;
    bool poisoned_=false;
    SessionVerifyResult run_window(std::span<const int32_t>,const CandidateSelection&,
                                   const std::function<bool()>&);
    void restore_history(std::span<const int32_t>);
    friend class Model;
};
}
