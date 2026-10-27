#include "strata/hpu/ple_input.hpp"
#include "strata/hpu/ple_io_mode.hpp"
#include <iostream>
#include <cmath>
#include <chrono>
#if defined(__linux__)
#include <sys/syscall.h>
#include <sys/mman.h>
#include <unistd.h>
// TEST DRIVER ONLY. ASan interposes mlock/munlock as successful no-ops. Use
// the real kernel calls here so RLIMIT_MEMLOCK=0 still exercises the original
// reader's failure/fallback branch without weakening any assertion.
extern "C" int mlock(const void* address,size_t length) noexcept {
    return int(::syscall(SYS_mlock,address,length));
}
extern "C" int munlock(const void* address,size_t length) noexcept {
    return int(::syscall(SYS_munlock,address,length));
}
#endif

using namespace strata::hpu;
int main(int argc,char** argv) {
    if(argc!=2 && argc!=3) return 2;
    try {
        const auto source=inspect_ple_source(argv[1]);
        strata::kernels::PleTable reader;auto io=resolve_ple_io(argc==3?argv[2]:"mmap",{});
        const bool is_direct=io.mode==strata::kernels::PleIo::Direct;
        if(is_direct) {io.cache_rows=4096;io.max_inflight=1024;}
        const auto memory=ple_ram_memory(source,io);
        const auto direct=ple_reader_memory(source,io);
        if(is_direct) {
            for(const auto budget:{uint64_t(0),direct.host_bytes-1}) {
                bool refused=false;
                try {open_ple_source(reader,source,io,0,budget,5000);}
                catch(const std::invalid_argument& e) {refused=std::string(e.what()).find("host memory reservation")!=std::string::npos;}
                if(!refused || reader.is_open()) throw std::runtime_error("direct reader ignored host reservation");
            }
        }
        if(io.lock) {
            for(const auto budget:{uint64_t(0),memory.host_bytes-1}) {
                bool refused=false;
                try {open_ple_source(reader,source,io,budget);}
                catch(const std::invalid_argument& e) {refused=std::string(e.what()).find("RAM reservation")!=std::string::npos;}
                if(!refused || reader.is_open()) throw std::runtime_error("RAM reader ignored preflight reservation");
            }
            auto corrupt=source;corrupt.payload_bytes=1;
            bool refused=false;
            try {open_ple_source(reader,corrupt,io,memory.host_bytes);}
            catch(const std::runtime_error& e) {refused=std::string(e.what()).find("source metadata changed")!=std::string::npos;}
            if(!refused || reader.is_open()) throw std::runtime_error("RAM source sizing trusted stale metadata");
        }
        open_ple_source(reader,source,io,memory.host_bytes,direct.host_bytes,is_direct?5000:0);
        if(reader.rows()!=source.rows) throw std::runtime_error("PLE preflight and original reader disagree");
        float values[160];
        double observed_delay=0;
        for(uint32_t row:{uint32_t(0),uint32_t(source.rows-1)}) {
            const auto start=std::chrono::steady_clock::now();
            reader.read_row(row,values);
            const auto elapsed=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count();
            if(is_direct && elapsed<5000) throw std::runtime_error("positive delay did not reach actual reader");
            if(is_direct) observed_delay+=elapsed;
            for(float value:values) if(!std::isfinite(value) || value!=1) throw std::runtime_error("independent constant-one PLE row oracle mismatch");
        }
        if(ple_source_identity(source.path)!=source.identity) throw std::runtime_error("PLE file identity changed");
        std::cout<<source.rows<<'\n'<<source.identity<<'\n';
        std::cout<<source.payload_offset<<' '<<source.payload_bytes<<' '<<memory.page_bytes<<' '
                 <<memory.worker_bytes<<' '<<memory.host_bytes<<' '<<(reader.locked()?1:0)<<'\n';
        std::cout<<direct.cache_rows<<' '<<direct.cache_bytes<<' '<<direct.inflight_bytes<<' '
                 <<direct.worker_bytes<<' '<<direct.host_bytes<<' '<<observed_delay<<'\n';
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
