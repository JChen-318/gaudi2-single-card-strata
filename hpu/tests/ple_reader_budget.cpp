#include "strata/hpu/ple_reader_budget.hpp"
#include <iostream>
#include <limits>

using namespace strata::hpu;
int main() {
    auto require=[](bool value) {if(!value) throw std::runtime_error("PLE reader independent budget oracle mismatch");};
    for(uint64_t rows:{0,1,7,8,9,1048576,8388608}) for(uint32_t bytes:{90,110,160,170,320})
        for(uint32_t slots:{1,256,512,1024}) for(uint32_t workers:{1,17,65}) {
            const auto p=plan_ple_reader(rows,bytes,slots,8ull<<20,4096,workers);
            const uint64_t admitted=rows-rows%8;
            require(p.cache_rows==admitted && p.cache_bytes==admitted*bytes+admitted*4+admitted/8);
            require(p.inflight_bytes==uint64_t(slots)*16384 && p.worker_bytes==workers*((9ull<<20)+4096));
            require(p.host_bytes==p.cache_bytes+p.inflight_bytes+p.worker_bytes+(4ull<<20));
        }
    require(ple_file_workers(nullptr,16)==16 && ple_file_workers("",16)==1 && ple_file_workers("garbage",16)==1);
    require(ple_file_workers(" -5",16)==1 && ple_file_workers("+17suffix",16)==17 && ple_file_workers("999",16)==64);
    auto refuses=[](auto call) {try {call();} catch(const std::exception&) {return;} throw std::runtime_error("invalid PLE reader budget accepted");};
    refuses([]{(void)plan_ple_reader(UINT64_MAX,320,1024,1,0,1);});
    refuses([]{(void)plan_ple_reader(1,320,1025,1,0,1);});
    refuses([]{(void)plan_ple_reader(1,320,1,UINT64_MAX,1,1);});
    refuses([]{(void)ple_file_workers("999999999999999999999",16);});
    std::cout<<"PASS PLE reader cache-set/key/data, full inflight, 65 worker/env/overflow admission oracles; no device\n";
}
