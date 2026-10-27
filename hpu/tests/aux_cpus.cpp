// Host law for --aux-cpus: the original parse/derive/restrict functions plus the port's parse/configure/plan
// wrapper. The sysfs and affinity calls only run on Linux; the selection laws and every refusal are portable.
#include "strata/hpu/aux_cpus.hpp"
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

using namespace strata;
using namespace strata::hpu;

namespace {
int checks = 0;
void require(bool ok, const char* what) {
    ++checks;
    if (!ok) throw std::runtime_error(std::string("aux_cpus: ") + what);
}
void eq(const std::vector<int>& got, const std::vector<int>& want, const char* what) {
    ++checks;
    if (got == want) return;
    auto text = [](const std::vector<int>& values) {
        std::string out;
        for (int value : values) out += (out.empty() ? "" : ",") + std::to_string(value);
        return "{" + out + "}";
    };
    throw std::runtime_error(std::string("aux_cpus: ") + what + " (got " + text(got) + ", want " + text(want) + ")");
}
const std::vector<aux_cpus::LogicalCpu> kAllowed{
    {0, 7, 0}, {1, 7, 0}, {2, 7, 1}, {3, 7, 1}, {4, 8, 0}, {5, 8, 0}, {6, 8, 1}, {7, -1, -1}};
}

int run() {
    std::vector<int> out;

    require(!aux_cpus::parse_cpu_list("", out) && out.empty(), "empty list refused");
    require(aux_cpus::parse_cpu_list("24", out), "single CPU");
    eq(out, {24}, "single CPU value");
    require(aux_cpus::parse_cpu_list("24,26", out), "two CPUs");
    eq(out, {24, 26}, "two CPU values");
    require(aux_cpus::parse_cpu_list("26,24", out), "unsorted list");
    eq(out, {24, 26}, "list sorted");
    require(aux_cpus::parse_cpu_list("24,24,26", out), "duplicate list");
    eq(out, {24, 26}, "duplicates removed");
    require(aux_cpus::parse_cpu_list("24-27", out), "range");
    eq(out, {24, 25, 26, 27}, "range expanded");
    require(aux_cpus::parse_cpu_list("24-24", out), "one-wide range");
    eq(out, {24}, "one-wide range value");
    require(aux_cpus::parse_cpu_list("0-3,8", out), "range then CPU");
    eq(out, {0, 1, 2, 3, 8}, "range then CPU value");
    require(aux_cpus::parse_cpu_list("100000", out) && out.size() == 1, "largest accepted number");
    for (const char* bad : {"24-", "27-24", "abc", "1,,2", ",1", "1,", "-1", " 1", "1 ", "1-2-3", "24.26",
                            "100001", "auto"})
        require(!aux_cpus::parse_cpu_list(bad, out) && out.empty(), bad);

    aux_cpus::Config cfg;
    std::string err;
    require(aux_cpus::parse_spec("", cfg, err) && !cfg.on && !cfg.use_list && cfg.list.empty(), "empty spec is off");
    require(aux_cpus::parse_spec("off", cfg, err) && !cfg.on, "off is off");
    require(aux_cpus::parse_spec("auto", cfg, err) && cfg.on && !cfg.use_list, "auto is on and not a list");
    require(aux_cpus::parse_spec("24,26", cfg, err) && cfg.on && cfg.use_list, "CPU list is an explicit plan");
    eq(cfg.list, {24, 26}, "CPU list value");
    require(!aux_cpus::parse_spec("auto,1", cfg, err), "mixed auto/list refused");
    require(err == "--aux-cpus takes off, auto or a CPU list such as 24,26 or 24-27 (got 'auto,1')",
            "refusal text");

    std::string spec;
    require(parse_aux_cpus("auto", spec, err) && spec == "auto", "wrapper accepts auto");
    require(parse_aux_cpus("", spec, err) && spec.empty(), "wrapper keeps an empty flag empty (env may apply)");
    require(parse_aux_cpus("off", spec, err) && spec == "off", "wrapper keeps off (env must not apply)");
    require(!parse_aux_cpus("nope", spec, err), "wrapper refuses a bad spec");
    require(err == "--aux-cpus takes off, auto or a CPU list such as 24,26 or 24-27 (got 'nope')",
            "wrapper refusal text");

    eq(aux_cpus::derive(kAllowed, 0, {2}), {1, 4, 5, 6}, "host's siblings first, then every free core");
    eq(aux_cpus::derive(kAllowed, 4, {0, 2}), {5, 6}, "sibling of the host, then a core with no worker");
    eq(aux_cpus::derive(kAllowed, -1, {}), {0, 1, 2, 3, 4, 5, 6}, "unknown host: every known core");
    eq(aux_cpus::derive(kAllowed, 4, {2}), {5, 0, 1, 6}, "siblings first, then free cores; a worker core is never taken");
    eq(aux_cpus::derive(kAllowed, 7, {}), {0, 1, 2, 3, 4, 5, 6}, "unknown host CPU excluded, unknown-core CPU skipped");
    eq(aux_cpus::derive(kAllowed, 0, {99}), {1, 2, 3, 4, 5, 6}, "worker CPU outside the allowed set");
    eq(aux_cpus::derive(kAllowed, 0, {0}), {1, 2, 3, 4, 5, 6}, "host also named a worker: siblings still first");
    eq(aux_cpus::derive({}, 0, {}), {}, "no allowed CPUs: no plan");

    eq(aux_cpus::restrict_list({0, 1, 3, 4, 5, 7, 99}, kAllowed, 0, {4}), {1, 3, 5, 7},
       "list minus host/workers/not-allowed");
    eq(aux_cpus::restrict_list({7, 3, 9}, kAllowed, -1, {}), {7, 3}, "list keeps allowed CPUs in list order");
    eq(aux_cpus::restrict_list({1, 3}, kAllowed, 1, {3}), {}, "list entirely host and workers");

    // The program-level sequence: a bad spec is refused before any affinity call; off leaves the feature off.
    require(!start_aux_cpus("nope", err), "start refuses a bad spec");
    require(err == "--aux-cpus takes off, auto or a CPU list such as 24,26 or 24-27 (got 'nope')",
            "start refusal text");
    require(start_aux_cpus("off", err), "start accepts off");
    require(!aux_cpus::enabled(), "off means disabled");
    require(aux_cpus::plan().empty(), "no plan before set_topology");
    const std::string note = plan_aux_cpus(0, {1});
    require(note.rfind("strata aux cpus: ", 0) == 0, "plan returns upstream's report line");
#if defined(__linux__)
    require(note.find("off at start") != std::string::npos, "off stays off at start-up");
#endif
    aux_cpus::set_enabled(false);

    std::printf("PASS aux_cpus law: %d checks (parse list/spec, derive, restrict, parse/configure/plan seam)\n", checks);
    return 0;
}

int main() {
    try {
        return run();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL %s\n", error.what());
        return 1;
    }
}
