#pragma once
#include <cstdint>
#include <algorithm>
#include <cerrno>
#include <climits>
#include <cstdlib>
#include <stdexcept>

namespace strata::hpu {
inline constexpr uint64_t kPleReaderBaselineHostBytes=320ull<<20;
struct PleReaderMemory {
    uint64_t cache_rows=0,cache_bytes=0,inflight_bytes=0,worker_bytes=0,host_bytes=0;
};
// Match DirectFile::io_threads's atoi-prefix/clamp law for representable
// integers. Refuse overflow rather than relying on atoi's undefined overflow.
inline uint32_t ple_file_workers(const char* text,uint32_t fallback) {
    if(!text) return fallback;
    errno=0;const auto value=std::strtoll(text,nullptr,10);
    if(errno==ERANGE || value<INT_MIN || value>INT_MAX)
        throw std::invalid_argument("STRATA_IO_THREADS exceeds the upstream integer range");
    return uint32_t(std::clamp(value,1ll,64ll));
}
// The native adapter serializes gather(16 rows): no unbounded batch or public
// prefetch queue reaches this reader. Cache storage follows original RowCache
// exactly (eight-way sets, packed data, uint32 keys and a uint8 per set).
// Inflight allowance includes the 8KiB slab and an extra 8KiB per slot for
// jobs/uses/completions/queue capacity; 4MiB covers bounded foreground staging,
// four latency rings and fixed reader/file/control metadata. Worker stacks
// include DirectFile's pool as well as PleReader's optional coordinator.
inline PleReaderMemory plan_ple_reader(uint64_t rows,uint32_t row_bytes,uint32_t inflight,
                                      uint64_t stack,uint64_t guard,uint32_t workers) {
    auto add=[](uint64_t a,uint64_t b) {
        if(b>UINT64_MAX-a) throw std::overflow_error("PLE reader reservation overflow");
        return a+b;
    };
    auto multiply=[](uint64_t a,uint64_t b) {
        if(b && a>UINT64_MAX/b) throw std::overflow_error("PLE reader reservation overflow");
        return a*b;
    };
    if(!row_bytes || row_bytes>320 || !inflight || inflight>1024 || workers>65 || (workers && !stack))
        throw std::invalid_argument("invalid PLE reader reservation geometry");
    PleReaderMemory memory;memory.cache_rows=(rows/8)*8;
    memory.cache_bytes=add(multiply(memory.cache_rows,uint64_t(row_bytes)+4),rows/8);
    memory.inflight_bytes=multiply(inflight,16384);
    memory.worker_bytes=multiply(workers,add(add(stack,guard),uint64_t(1)<<20));
    memory.host_bytes=add(add(add(memory.cache_bytes,memory.inflight_bytes),memory.worker_bytes),uint64_t(4)<<20);
    return memory;
}
}
