#pragma once
#include "strata/hpu/weights.hpp"
#include "strata/hpu/resident_cache.hpp"
#include "strata/hpu/cpu_affinity.hpp"
#include <vector>
namespace strata::hpu {
class NativeExpertPack;
// Upstream's CpuTopology (include/strata/kernels/cpu/pool.hpp): the physical cores the pool plans its
// workers on and the core the host thread keeps. Detected by the pinned pool code, not re-derived here.
struct CpuPlacement {
    int host_core = -1;
    std::vector<int> worker_cores;
    bool hybrid = false;
    int p_cores = 0, p_threads = 0, e_cores = 0;
};
// Original Strata CPU ActQ/Q2_0 and ggml native IQ kernels with a bounded raw
// expert cache. Routed experts only; shared expert remains on HPU. Serialized
// pool submissions accept 1..8 token rows, unweighted FP32 output.
// NVFP4 reads original per-expert F32 .scale tensors and appends the native
// 16-byte sg/su/sd/zero tail; the raw cache budget includes that tail.
class CpuExperts {
public:
    // Schema/CPU/cache validation only: no pool, workers, weights or device. `workers` is upstream's
    // --pool-workers domain (0 = every physical core, resolved by ExpertPool); `tasks` is --pool-tasks
    // (0 = automatic, otherwise 1..ExpertPool::kMaxTasks).
    static void validate(const WeightTable&,uint64_t raw_cache_budget,uint32_t workers,uint32_t tasks=0);
    // Upstream's worker placement (src/kernels/cpu/pool.cpp:92-380): one worker per physical core,
    // SMT siblings excluded, the skipped host core never in worker_cores.
    static CpuPlacement placement(uint32_t workers, CpuAffinity affinity={});
    // Upstream's --host-core applies before the pool and the session pin any thread (:2293-2299).
    // The constructor calls this; exposed so a caller can plan with the same setting.
    static void apply_host_core(CpuAffinity affinity);
    CpuExperts(WeightTable,uint64_t raw_cache_budget=256ull<<20,uint32_t workers=0,std::shared_ptr<const NativeExpertPack> pack={},CpuAffinity affinity={},uint32_t tasks=0);
    ~CpuExperts();
    CpuExperts(const CpuExperts&)=delete;
    CpuExperts& operator=(const CpuExperts&)=delete;
    std::vector<float> run(uint32_t layer,int32_t expert,std::span<const float> rows);
    CacheStats stats() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
