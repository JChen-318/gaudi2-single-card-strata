#include "strata/hpu/ple_ram_budget.hpp"
#include <iostream>
#include <limits>

using namespace strata::hpu;
int main() {
    auto require=[](bool ok) {if(!ok) throw std::runtime_error("PLE RAM budget oracle mismatch");};
    for(uint64_t page:{4096,65536}) for(uint64_t offset:{0,32,4095,65535,65536})
        for(uint64_t bytes:{1,90,320,4096,65536}) for(uint32_t workers:{0,1,16}) {
            const auto plan=plan_ple_ram(offset,bytes,page,8ull<<20,4096,workers);
            // Independent page-range oracle: every touched byte belongs to
            // one inclusive page interval, including header-sharing edge pages.
            const auto expected=((offset+bytes-1)/page-offset/page+1)*page;
            require(plan.table_bytes==bytes && plan.page_bytes==expected);
            require(plan.worker_bytes==workers*((9ull<<20)+4096));
            require(plan.host_bytes==expected+plan.worker_bytes+(1ull<<20));
        }
    auto refuses=[&](uint64_t offset,uint64_t bytes,uint64_t page,uint64_t stack,uint64_t guard,uint32_t workers) {
        try {(void)plan_ple_ram(offset,bytes,page,stack,guard,workers);}
        catch(const std::exception&) {return;}
        throw std::runtime_error("invalid PLE RAM budget accepted");
    };
    refuses(0,0,4096,1,0,1);refuses(0,1,0,1,0,1);refuses(0,1,4095,1,0,1);
    refuses(0,1,131072,1,0,1);refuses(0,1,4096,0,0,1);refuses(0,1,4096,1,0,17);
    const auto max=std::numeric_limits<uint64_t>::max();
    refuses(max,1,4096,1,0,1);
    refuses(4095,max,4096,1,0,1);refuses(0,max-4095,4096,1,0,1);
    refuses(0,1,4096,max,0,1);refuses(0,1,4096,max/2,0,16);
    std::cout<<"PASS PLE RAM page-range/worker/full-host reservation oracles and overflow guards; no device\n";
}
