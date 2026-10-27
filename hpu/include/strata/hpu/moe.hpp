#pragma once
#include "strata/hpu/expert.hpp"
#include "strata/hpu/expert_cache.hpp"
#include "strata/hpu/decode_chain.hpp"
#include "strata/hpu/routing.hpp"
#include "strata/hpu/hc_ple.hpp"
#include <array>

namespace strata::hpu {
class CpuExperts;
struct MoeOptions {
    CpuExperts* cpu=nullptr;
    // Routed HPU cache misses loaded per call. Resident hits remain on HPU;
    // additional misses use original CPU ActQ/IQ. Shared expert stays on HPU.
    uint32_t hpu_load_budget=UINT32_MAX;
    // Optional --dump-routing producer: receives this call's routed ids and weights (tokens*top_k), in
    // router order. Upstream surfaces them through its fused dispatch (src/program/generate.cpp:1007-1023);
    // empty means no hook is wired and the dump flag is refused at open instead of writing zeros.
    std::function<void(uint32_t layer,const Routes&)> routing_sink;
};
// Reuses one packed format per activation contract plus the BF16 reference
// context, with bounded pinned slots. The lease is passed by value
// and retained through synchronous readback, so eviction cannot race a launch.
// Returns unweighted F32 packed results or widened BF16 reference rows;
// trailing bucket rows are ignored. Formats may change across cache generations.
class ExpertExecutor {
public:
    ExpertExecutor(const Device&,core::ModelGeometry,uint32_t bucket);
    ~ExpertExecutor();
    ExpertExecutor(const ExpertExecutor&)=delete;
    ExpertExecutor& operator=(const ExpertExecutor&)=delete;
    std::vector<float> run(std::span<const float> rows,ExpertLease);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
// Native MoE, including BF16 router/shared gate projections, CPU
// stable probability top-k, grouped HPU experts, exactly-once combine and the
// gated shared expert. Cache is shared across layers and must outlive blocks.
// Q2 routed experts use FP32-scale ActQ; packed shared Q2/Q4/Q8 uses Q8_1.
// Other formats retain BF16 reference. Optional bounded CPU ActQ/IQ fallback.
class MoeBlock {
public:
    MoeBlock(const Device&,WeightTable,uint32_t layer,ExpertCache&,MoeOptions={});
    MoeBlock(const Device&,core::ModelGeometry,uint32_t layer,ExpertCache&,Binding router,Binding shared_gate,MoeOptions={});
    ~MoeBlock();
    MoeBlock(const MoeBlock&)=delete;
    MoeBlock& operator=(const MoeBlock&)=delete;
    std::vector<float> run(std::span<const float> normalized_rows);
    // Fused single-token decode (--fp8-experts, no CPU fallback): the layer's own graph computes the router
    // with router_bindings() ("w_router", "w_shared_gate"); fused_ffn() then routes on the host (same
    // route_reference, profile counting and dump hook as run()), runs every chosen expert, their weighted
    // sum and the FFN HC write in one launch, and returns the next residual [streams*hidden].
    bool fused_supported() const;
    std::array<Binding,2> router_bindings() const;
    std::vector<float> fused_ffn(std::span<const float> logits,float shared_gate,std::span<const float> x,
                                 std::span<const float> residual,std::span<const float> inject,const HcConfig& hc);
    // STRATA_HPU_CHAIN: the same FFN half on the chain's device buffers (x, mid as residual, inject), writing
    // chain.out() and launched on chain.stream without waiting; the chosen experts' leases go to chain.leases.
    void fused_ffn_chain(std::span<const float> logits,float shared_gate,DecodeChain& chain,const HcConfig& hc);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
