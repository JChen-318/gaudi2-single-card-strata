#pragma once
#include "strata/platform/aux_cpus.hpp"
#include <cstdlib>
#include <string>
#include <vector>

namespace strata::hpu {
// Upstream's --aux-cpus off|auto|LIST (src/program/generate.cpp:1998-2008, :2299-2308, :5071-5076):
// every thread that is neither a pool worker nor the host runs on CPUs no worker uses. The parsing and
// selection laws are the original strata::aux_cpus ones; this wrapper only names the program-level order:
// validate at parse time, snapshot the allowed CPUs before the pool exists, plan once its workers are known.
inline bool parse_aux_cpus(const std::string& text, std::string& spec, std::string& err) {
    strata::aux_cpus::Config cfg;
    if (!strata::aux_cpus::parse_spec(text, cfg, err)) return false;
    spec = text;
    return true;
}

// --aux-cpus when it was given, else STRATA_AUX_CPUS. Call before any thread is pinned: the allowed set is
// the process's CPUs at this moment (generate.cpp:2299-2308).
inline bool start_aux_cpus(const std::string& spec, std::string& err) {
    std::string text = spec;
    if (text.empty())
        if (const char* env = std::getenv("STRATA_AUX_CPUS")) text = env;
    return strata::aux_cpus::configure(text, err);
}

// After the pool exists: host_cpu is the CPU the host keeps (-1 when unknown), worker_cpus the workers'
// CPUs. Returns upstream's one-line report.
inline std::string plan_aux_cpus(int host_cpu, const std::vector<int>& worker_cpus) {
    return strata::aux_cpus::set_topology(host_cpu, worker_cpus);
}
}
