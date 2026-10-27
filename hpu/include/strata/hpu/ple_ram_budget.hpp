#pragma once
#include <cstdint>
#include <stdexcept>

namespace strata::hpu {
struct PleRamMemory {
    uint64_t table_bytes=0,page_bytes=0,worker_bytes=0,host_bytes=0;
};

// Upstream pretouches the mapped tensor with at most sixteen POSIX workers.
// Price their entire default stacks and guards, not just the small touched
// portion. The metadata/TLS allowance is additional to the model's existing
// process/compiler reserves. This is an admission estimate, not measured RSS.
inline PleRamMemory plan_ple_ram(uint64_t offset,uint64_t payload,uint64_t page,
                                 uint64_t stack,uint64_t guard,uint32_t workers) {
    auto add=[](uint64_t a,uint64_t b) {
        if(b>UINT64_MAX-a) throw std::overflow_error("PLE RAM reservation overflow");
        return a+b;
    };
    if(!payload || !page || (page&(page-1)) || page>65536 || workers>16 || (workers && !stack))
        throw std::invalid_argument("invalid PLE RAM reservation geometry");
    (void)add(offset,payload); // The tensor's end must be addressable too.
    PleRamMemory memory;memory.table_bytes=payload;
    const auto span=add(offset&(page-1),payload);
    memory.page_bytes=add(span,page-1)&~(page-1);
    const auto per_worker=add(add(stack,guard),uint64_t(1)<<20);
    if(workers && per_worker>UINT64_MAX/workers) throw std::overflow_error("PLE RAM worker reservation overflow");
    memory.worker_bytes=per_worker*workers;
    memory.host_bytes=add(add(memory.page_bytes,memory.worker_bytes),uint64_t(1)<<20);
    return memory;
}
}
