#pragma once
#include "strata/hpu/tensor.hpp"
#include <span>
#include <functional>

namespace strata::hpu {
enum class ExpertPlacement { ResidentHpu,LoadHpu,Cpu };
class PlacementPolicy {
public:
    explicit PlacementPolicy(uint32_t hpu_load_budget):budget_(hpu_load_budget) {}
    ExpertPlacement choose(int32_t expert,bool resident) {
        if(expert< -1) throw std::invalid_argument("invalid expert placement ID");
        if(resident) return ExpertPlacement::ResidentHpu;
        if(expert==-1) return ExpertPlacement::LoadHpu;
        if(loads_>=budget_) return ExpertPlacement::Cpu;
        ++loads_;return ExpertPlacement::LoadHpu;
    }
    uint32_t loads() const {return loads_;}
private:
    uint32_t budget_,loads_=0;
};
struct Routes {
    uint32_t tokens = 0, experts = 0, top_k = 0;
    std::vector<int32_t> ids;
    std::vector<float> weights;
};
// Host reference path for initial native integration. Finite F32 logits only.
// Softmax over all experts, stable probability top-k (lower ID on a tie),
// selected-probability sum clamped at 2^-14, matching Strata's default router.
Routes route_reference(std::span<const float> logits, uint32_t tokens, uint32_t experts, uint32_t top_k);

struct ExpertGroup {
    int32_t expert_id = -1;
    // Original flat token*top_k+rank positions, in ascending order.
    std::vector<uint64_t> routed_indices;
};
class RoutedParts {
public:
    RoutedParts(Routes routes, uint32_t hidden);
    std::span<const ExpertGroup> groups() const { return groups_; }
    // Expert outputs are unweighted. Can be called for CPU and HPU groups in
    // any order, once per group, after the caller's readback completion event.
    // Calls on one object are serialized by the orchestrator.
    void scatter_group(size_t group_index, std::span<const float> unweighted_rows);
    std::vector<float> combine() const;
private:
    Routes routes_;
    uint32_t hidden_;
    std::vector<ExpertGroup> groups_;
    std::vector<float> parts_;
    std::vector<uint8_t> filled_;
};
// Synchronous scheduler shared by native MoE and host contract tests. Callback
// receives contiguous unweighted activation rows for one expert (id=-1 is the
// shared expert), and returns unweighted outputs. Router weights apply once.
using RoutedExecutor=std::function<std::vector<float>(int32_t,std::span<const float>)>;
std::vector<float> dispatch_moe(Routes,std::span<const float> input,uint32_t hidden,
                              std::span<const float> shared_gate,const RoutedExecutor&);
// Runs CPU groups on one background dispatcher while the caller submits HPU
// groups and the shared expert. Resident HPU groups finish before new uploads,
// so their leases cannot make a tiny cache deadlock on an incoming expert.
// All CPU work is joined before return or propagation of an HPU failure.
std::vector<float> dispatch_hybrid_moe(Routes,std::span<const float>,uint32_t hidden,std::span<const float> shared_gate,
    const std::function<ExpertPlacement(int32_t)>& place,const RoutedExecutor& hpu,const RoutedExecutor& cpu);
// Optional batched HPU form: one call receives every HPU group as (expert id, gathered rows), in the order
// the per-expert form would submit them (resident, then loaded), with the shared expert (-1, all rows)
// last, and returns one unweighted output per entry. Each expert's result is the same as its own call;
// only the number of submissions changes. An empty callback keeps the per-expert form.
using RoutedGroupExecutor=std::function<std::vector<std::vector<float>>(std::span<const std::pair<int32_t,std::vector<float>>>)>;
std::vector<float> dispatch_hybrid_moe(Routes,std::span<const float>,uint32_t hidden,std::span<const float> shared_gate,
    const std::function<ExpertPlacement(int32_t)>& place,const RoutedExecutor& hpu,const RoutedExecutor& cpu,
    const RoutedGroupExecutor& hpu_group);
} // namespace strata::hpu
