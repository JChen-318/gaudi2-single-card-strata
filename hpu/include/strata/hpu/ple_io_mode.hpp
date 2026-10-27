#pragma once
// Original CLI PLE value domains, with storage admitted by explicit byte plans.
// direct/mmap/ram share the original reader. RAM reserves tensor pages and
// pretouch workers; Direct reserves row-cache data/keys, queues and worker stacks.
// Large caches and 1..1024 inflight are accepted only when the complete plan fits.
// Delay is stored separately from PleIoOptions and reaches the original reader.
#include "strata/kernels/ngram.hpp"

#include <cstdint>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

namespace strata::hpu {
#if defined(__linux__) && defined(__SANITIZE_ADDRESS__)
inline constexpr bool kPleLockResultSimulated=true;
#elif defined(__linux__) && defined(__has_feature)
#if __has_feature(address_sanitizer)
inline constexpr bool kPleLockResultSimulated=true;
#else
inline constexpr bool kPleLockResultSimulated=false;
#endif
#else
inline constexpr bool kPleLockResultSimulated=false;
#endif
// ASan's mlock success does not establish OS residency. Report no trustworthy
// lock in that configuration, including INFO, rather than echoing the stub.
inline bool reported_ple_ram_lock(bool succeeded) {return succeeded && !kPleLockResultSimulated;}
inline constexpr uint32_t kPleRowCacheRows = 1u << 20;   // original default; capacity is admitted by its byte plan
inline constexpr uint32_t kPleInflightMax = 1024;

// The same basic policy is used by preflight and the actual reader. RAM's
// full-table/worker reservation is additionally priced by plan_model and
// checked again by open_ple_source before the original reader touches pages.
inline void validate_ple_io_policy(const strata::kernels::PleIoOptions& io) {
    if(io.mode!=strata::kernels::PleIo::Direct && io.mode!=strata::kernels::PleIo::Mmap)
        throw std::invalid_argument("invalid PLE reader mode");
    if(io.lock && io.mode!=strata::kernels::PleIo::Mmap)
        throw std::invalid_argument("PLE RAM lock requires mmap mode");
    if(!io.max_inflight || io.max_inflight>kPleInflightMax)
        throw std::invalid_argument("PLE reader exceeds bounded RAM policy");
}

// `--ple-io direct|mmap|ram` -> {mode, lock}. Upstream's own message for anything else.
inline strata::kernels::PleIoOptions resolve_ple_io(const std::string_view text, strata::kernels::PleIoOptions options) {
    if (text == "direct") { options.mode = strata::kernels::PleIo::Direct; options.lock = false; return options; }
    if (text == "mmap") { options.mode = strata::kernels::PleIo::Mmap; options.lock = false; return options; }
    if (text == "ram") { options.mode = strata::kernels::PleIo::Mmap; options.lock = true; return options; }
    throw std::invalid_argument("invalid --ple-io/--ple-row-cache/--ple-inflight/--ple-delay-us");
}

inline strata::kernels::PleIoOptions resolve_ple_row_cache(const int64_t rows, strata::kernels::PleIoOptions options) {
    if (rows < 0) throw std::invalid_argument("invalid --ple-io/--ple-row-cache/--ple-inflight/--ple-delay-us");
    options.cache_rows = uint64_t(rows);
    return options;
}

inline strata::kernels::PleIoOptions resolve_ple_inflight(const int64_t reads, strata::kernels::PleIoOptions options) {
    // The original reader supports the complete range; its slab is byte-priced.
    if (reads < 1 || reads > 1024)
        throw std::invalid_argument("invalid --ple-io/--ple-row-cache/--ple-inflight/--ple-delay-us");
    options.max_inflight = uint32_t(reads);
    return options;
}

inline double resolve_ple_delay_us(const double microseconds) {
    if (!(microseconds >= 0.0) || !std::isfinite(microseconds))
        throw std::invalid_argument("invalid --ple-io/--ple-row-cache/--ple-inflight/--ple-delay-us");
    return microseconds;
}
} // namespace strata::hpu
