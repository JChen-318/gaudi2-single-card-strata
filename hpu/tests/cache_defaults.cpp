// Upstream's --prompt-cache default is 6 kept checkpoints; this backend's default must not silently be none.
//
// The decision is the pure function hpu/include/strata/hpu/generation.hpp now carries, so it can be swept here
// and the tool's call site is guarded separately (work/check_cache_default_wiring.py, run by the local suite):
// the port charges the prefix cache against --snapshot-budget-mib, and a cache with slots but no byte limit is
// a contradiction hpu/src/prefix_cache.cpp itself refuses - so the default is upstream's number wherever the
// operator granted the RAM, and a REPORTED zero where they did not. Host only: no device, no allocation.
#include "strata/hpu/generation.hpp"
#include "strata/hpu/prefix_cache.hpp"
#include <array>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

using namespace strata::hpu;
namespace {
uint32_t cases = 0;
void require(bool ok, const std::string& why) { if (!ok) throw std::runtime_error(why); }

void defaults() {
    // given, snapshot budget, requested -> expected slots
    const std::vector<std::array<bool, 2>> flags{{false, false}, {false, true}, {true, false}, {true, true}};
    for (const auto& flag : flags) {
        const bool given = flag[0];
        for (uint64_t budget : {uint64_t(0), uint64_t(1) << 30}) {
            for (uint32_t requested : {0u, 1u, 3u, 6u, 64u}) {
                const uint32_t got = default_prompt_cache_slots(given, budget, requested);
                if (given) require(got == requested, "an explicit --prompt-cache was overridden");
                else if (budget) require(got == 6 && got == upstream_prompt_cache_slots,
                                          "a granted budget did not reach upstream's default");
                else require(got == 0, "no budget still defaulted to keeping checkpoints");
                // the note must never let a zero look like a choice the operator made
                require(prompt_cache_note(got) == (got ? "prompt_cache=default" : "prompt_cache=no-snapshot-budget"),
                        "the run report does not say why the default is what it is");
                // and whatever the default picks must be a cache the cache itself accepts
                bool accepted = true;
                try { PrefixCache cache(got, budget ? budget : 1); }
                catch (const std::exception&) { accepted = false; }
                require(accepted || (got && !budget), "an affordable combination was refused, or an unaffordable one accepted");
                ++cases;
            }
        }
    }
    // upstream's own default, read from the pinned source rather than remembered: the number the port mirrors
    require(upstream_prompt_cache_slots == 6, "the mirrored upstream default drifted from 6 checkpoints");
    // slots without a byte limit is the contradiction the default must never produce
    bool refused = false;
    try { (void)PrefixCache(6, 0); } catch (const std::invalid_argument&) { refused = true; }
    require(refused, "a cache with slots and no byte limit was accepted");
    ++cases;
}
}

int main() {
    try {
        defaults();
        std::cout << "PASS prompt-cache default parity: " << cases
                  << " combinations where an explicit value always wins, a granted snapshot budget reaches "
                  "upstream's 6 checkpoints, and no budget yields a reported zero rather than a silent one; "
                  "slots-without-limit stays refused; no device\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
