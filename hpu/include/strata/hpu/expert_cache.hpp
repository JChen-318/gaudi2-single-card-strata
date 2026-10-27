#pragma once
#include "strata/hpu/resident_cache.hpp"
#include "strata/hpu/weight_transfer.hpp"
#include "strata/hpu/expert_format.hpp"
#include "strata/hpu/expert_profile.hpp"
// The upstream residency controls (--resident-experts / --resident-cpu-experts / --expert-cache-per-layer).
// The header is runtime-free, so it also compiles into the host test that exercises ResidentCache::acquire_in.
#include "strata/hpu/expert_residency.hpp"

namespace strata::hpu {
class NativeExpertPack;
struct ExpertSlices {WeightSlice gate,up,down;std::array<float,3> output_scales{1,1,1};bool scaled_outputs=false;};
// The cache key of one (layer, expert) pair, in the one place every caller and every pin reads it.
// The shared expert of a layer is `layer:-1`, exactly as the resolver names it.
inline std::string expert_cache_key(uint32_t layer,int32_t expert) {return std::to_string(layer)+":"+std::to_string(expert);}
struct ExpertWeights {
    DeviceBuffer gate,up,down;
    ExpertFormat format;
    std::array<float,3> output_scales{1,1,1};bool scaled_outputs=false;
    ExpertWeights(DeviceBuffer g,DeviceBuffer u,DeviceBuffer d,ExpertFormat f={},std::array<float,3> scales={1,1,1},bool scaled=false):gate(std::move(g)),up(std::move(u)),down(std::move(d)),format(f),output_scales(scales),scaled_outputs(scaled) {}
};
using ExpertLease=ResidentCache<ExpertWeights>::Lease;
using ExpertResolver=std::function<ExpertSlices(uint32_t layer,int32_t expert)>;
// Resolver is serialized on cold misses and must not reenter acquire().
// expert=-1 names a layer's shared expert. Constructor with WeightTable checks
// [H,FF,E]/[FF,H,E] for routed experts and [H,FF]/[FF,H] for shared experts.
// Budget caps combined packed/BF16 payloads, excluding driver allocation overhead,
// recipe workspace, activations, router weights and transient pinned staging.
class ExpertCache {
public:
    ExpertCache(const Device&,WeightTable,uint64_t hbm_payload_budget,uint64_t pinned_budget=1024*1024,std::shared_ptr<const NativeExpertPack> pack={});
    ExpertCache(const Device&,core::ModelGeometry,ExpertResolver,uint64_t hbm_payload_budget,uint64_t pinned_budget=1024*1024);
    ~ExpertCache();
    ExpertCache(const ExpertCache&)=delete;
    ExpertCache& operator=(const ExpertCache&)=delete;
    ExpertLease acquire(uint32_t layer,int32_t expert);
    ExpertLease resident(uint32_t layer,int32_t expert);
    // --fp8-experts (opt-in, port-only): every expert loaded from now on is dequantized on the host and
    // stored as per-row-scaled FP8 (strata/hpu/fp8.hpp), so ExpertExecutor runs it on the MME. Any global
    // output scale is folded into the row scales. Must be set before the first acquire.
    void set_fp8(bool);
    bool fp8() const;
    void resize(uint64_t hbm_payload_budget);
    uint64_t capacity() const;
    CacheStats stats() const;
    const core::ModelGeometry& geometry() const;
    // ---- Upstream's expert-cache profiling and adaptation knobs (strata/hpu/expert_profile.hpp).
    // Off by default: default options allocate no counters and build no profiler, so every hook below is one
    // null check on the lookup path and the cache behaves exactly as it does today.
    void set_expert_profile_options(ExpertProfileOptions);
    const ExpertProfileOptions& expert_profile_options() const;
    /// The counters, or null when nothing was asked for. Owned by the cache; valid for its lifetime.
    ExpertProfiler* expert_profiler();
    /// The routed expert IDs of one layer-window, which is where upstream counts routing: one increment per
    /// routed entry (src/core/expert_source.cpp:3717-3719). Feeds `usage` only; the hit/miss tallies come
    /// from acquire(). Free when the knobs are off.
    void note_routing(uint32_t layer, const int32_t* ids, int64_t n);
    /// One --adapt-every tick: republishes the resident set from the cache, then runs the upstream round and
    /// hands back at most --adapt-swaps residency changes for the caller to copy. Device work stays with the
    /// caller; call this again once the copies land. False with an empty list when the tier is off.
    bool adapt_experts(std::vector<ExpertSwap>& applied);
    // ---- upstream's expert-cache residency knobs (strata/hpu/expert_residency.hpp) -----------------------
    // Install --expert-cache-per-layer's per-layer byte partition and, when the RAM tier is on and a static
    // profile decides which experts it holds, the profile's admitted set - pinned against eviction when the
    // copy is page-locked (--resident-experts / --resident-budget-gib). Called once at startup with the
    // options the tool parsed. Every field is false/zero by default, so this is a no-op unless asked for and
    // the cache keeps its plain lazy acquire and plain LRU order.
    void set_expert_residency(ExpertResidencyOptions);
    const ExpertResidencyOptions& expert_residency_options() const;
    /// What was actually installed. `per_layer_experts` is the quota q the split bought (0 = the flag could
    /// not give one slot per layer and the uniform cache stands, upstream's fall-through); `admitted`/`pinned`
    /// count the experts the profile filled. Reporting only: no hardware claim.
    struct ResidencyReport {uint64_t per_layer_experts=0,admitted=0,pinned=0;bool per_layer_honored=false;};
    ResidencyReport residency_report() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
