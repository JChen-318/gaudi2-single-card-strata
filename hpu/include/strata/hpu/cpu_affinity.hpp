#pragma once
#include <string_view>
namespace strata::hpu {
// Upstream's --pool-affinity (src/program/generate.cpp:2006-2013): which physical cores the bounded
// CPU-expert workers may use. All = every core; Auto = P-cores, then E-cores on a hybrid CPU;
// PCores = P-cores only. The worker count is not changed by this knob.
enum class PoolAffinity { All, Auto, PCores };
// Upstream's --host-core first|last (src/program/generate.cpp:1991-1997, :2293-2299): the physical
// core the pool's host worker keeps. Last swaps it with the last planned worker core (pool.cpp:82-92)
// and applies before any thread is pinned.
enum class HostCore { First, Last };
struct CpuAffinity {
    PoolAffinity affinity = PoolAffinity::All; // upstream's default (:440)
    // Upstream constructs the pool with pin=true (src/program/generate.cpp:5051); only the worker
    // threads are pinned (src/kernels/cpu/pool.cpp:500-505), never the caller.
    bool pin = true;
    HostCore host_core = HostCore::First;      // upstream's default (:442)
};
// The spellings upstream accepts (:1993); anything else is refused with upstream's own text.
inline bool parse_host_core(std::string_view text, HostCore& out) {
    if (text == "first") {out = HostCore::First; return true;}
    if (text == "last") {out = HostCore::Last; return true;}
    return false;
}
// The spellings upstream accepts (:2009-2010); anything else is refused with upstream's own text.
inline bool parse_pool_affinity(std::string_view text, PoolAffinity& out) {
    if (text == "all") {out = PoolAffinity::All; return true;}
    if (text == "auto") {out = PoolAffinity::Auto; return true;}
    if (text == "p-cores" || text == "pcores") {out = PoolAffinity::PCores; return true;}
    return false;
}
inline const char* pool_affinity_name(PoolAffinity affinity) {
    switch (affinity) {
        case PoolAffinity::All: return "all";
        case PoolAffinity::Auto: return "auto";
        case PoolAffinity::PCores: return "p-cores";
    }
    return "all";
}
}
