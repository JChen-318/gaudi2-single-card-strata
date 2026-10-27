// Host-only guard for the CPU-expert worker placement the port now wires: upstream's --pool-affinity
// spellings, the pinned default, and the CpuTopology law this build actually detects (one logical
// processor per physical core, the skipped host core never among the workers, --host-core last swap).
#include "strata/hpu/cpu_affinity.hpp"
#include "strata/hpu/cpu_experts.hpp"
#include "strata/kernels/cpu/pool.hpp"
#include <algorithm>
#include <iostream>
#include <set>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {
using namespace strata::hpu;
void require(bool ok, const char* why) { if (!ok) throw std::runtime_error(why); }

void parsing() {
    PoolAffinity affinity = PoolAffinity::Auto;
    require(parse_pool_affinity("auto", affinity) && affinity == PoolAffinity::Auto, "auto is accepted");
    require(parse_pool_affinity("p-cores", affinity) && affinity == PoolAffinity::PCores, "p-cores is accepted");
    require(parse_pool_affinity("pcores", affinity) && affinity == PoolAffinity::PCores, "pcores is accepted");
    require(parse_pool_affinity("all", affinity) && affinity == PoolAffinity::All, "all is accepted");
    require(!parse_pool_affinity("", affinity), "an empty affinity is refused");
    require(!parse_pool_affinity("ALL", affinity), "the spelling is case-sensitive, as upstream's is");
    require(!parse_pool_affinity("p-core", affinity), "p-core (singular) is refused");
    const CpuAffinity defaults;
    require(defaults.affinity == PoolAffinity::All, "upstream's default affinity is all");
    require(defaults.pin, "upstream pins the pool's workers by default");
    require(defaults.host_core == HostCore::First, "upstream's default host core is first");
    HostCore host = HostCore::Last;
    require(parse_host_core("first", host) && host == HostCore::First, "host-core first is accepted");
    require(parse_host_core("last", host) && host == HostCore::Last, "host-core last is accepted");
    require(!parse_host_core("", host) && !parse_host_core("First", host) && !parse_host_core("auto", host),
            "host-core accepts exactly first|last, case-sensitively (generate.cpp:1993)");
    // The constructor applies this before the pool exists; check the same call through the pinned pool's
    // own accessor, then restore first so the placement checks below see the default topology.
    CpuAffinity last_affinity;
    last_affinity.host_core = HostCore::Last;
    CpuExperts::apply_host_core(last_affinity);
    require(strata::kernels::cpu::host_core_setting() == strata::kernels::cpu::HostCore::Last,
            "apply_host_core installs upstream's last setting before any pinning");
    CpuExperts::apply_host_core(CpuAffinity{});
    require(strata::kernels::cpu::host_core_setting() == strata::kernels::cpu::HostCore::First,
            "apply_host_core restores first");
}

void placement() {
    const CpuPlacement all = CpuExperts::placement(0, CpuAffinity{});
    require(!all.worker_cores.empty(), "the detected topology has worker cores");
    require(all.host_core >= 0, "skip_first reserves a host core");
    require(std::find(all.worker_cores.begin(), all.worker_cores.end(), all.host_core) == all.worker_cores.end(),
            "the host core is never also a worker core");
    const std::set<int> unique(all.worker_cores.begin(), all.worker_cores.end());
    require(unique.size() == all.worker_cores.size(), "one logical processor per physical core (no SMT siblings)");
    const unsigned logical = std::thread::hardware_concurrency();
    if (logical) require(all.worker_cores.size() <= logical, "the plan cannot exceed the machine's logical CPUs");

    // Upstream --host-core last (src/kernels/cpu/pool.cpp:82-92): the host takes the last planned core and
    // the previous host core becomes a worker, so the worker set keeps its size. Hybrid CPUs keep first.
    if (!all.hybrid && !all.worker_cores.empty()) {
        strata::kernels::cpu::set_host_core(strata::kernels::cpu::HostCore::Last);
        const CpuPlacement last = CpuExperts::placement(0, CpuAffinity{});
        strata::kernels::cpu::set_host_core(strata::kernels::cpu::HostCore::First);
        require(last.host_core == all.worker_cores.back(), "host-core last puts the host on the last planned core");
        require(std::find(last.worker_cores.begin(), last.worker_cores.end(), all.host_core) != last.worker_cores.end(),
                "the previous host core becomes a worker");
        require(last.worker_cores.size() == all.worker_cores.size(), "the swap keeps the worker count");
        const CpuPlacement restored = CpuExperts::placement(0, CpuAffinity{});
        require(restored.host_core == all.host_core && restored.worker_cores == all.worker_cores,
                "host-core first restores the default placement");
        CpuAffinity pcores;
        pcores.affinity = PoolAffinity::PCores;
        require(CpuExperts::placement(0, pcores).worker_cores == all.worker_cores,
                "on a uniform CPU p-cores has no E-cores to exclude, so it plans the same cores");
    }

    const CpuPlacement two = CpuExperts::placement(2, CpuAffinity{});
    require(two.worker_cores.size() == std::min<size_t>(2, all.worker_cores.size()),
            "a worker count reports exactly the cores the pool pins");
    require(std::equal(two.worker_cores.begin(), two.worker_cores.end(), all.worker_cores.begin()),
            "the pool pins the first planned cores in the plan's order");
}
}

int main() {
    try {
        parsing();
        placement();
        std::cout << "cpu affinity law ok: parsing, pinned default, physical-core placement, host-core last swap\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "cpu affinity test failed: " << error.what() << '\n';
        return 1;
    }
}
