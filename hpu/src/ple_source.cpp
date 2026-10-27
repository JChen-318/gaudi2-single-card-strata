#include "strata/hpu/ple_input.hpp"
#include "strata/hpu/ple_io_mode.hpp"
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat-truncation"
#endif
#include "strata/artifact/gguf_reader.hpp"
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif
#include <filesystem>
#include <cmath>
#include <limits>
#include <utility>
#include <cstdlib>
#include <sys/stat.h>
#ifndef _WIN32
#include <pthread.h>
#include <unistd.h>
#endif

namespace strata::hpu {
std::string ple_source_identity(const std::string& filename) {
    const auto path=std::filesystem::canonical(filename).string();
    std::string identity="ple-local-file-v1|"+std::to_string(path.size())+":"+path+"|";
#ifdef _WIN32
    const auto handle=CreateFileW(std::filesystem::path(path).c_str(),FILE_READ_ATTRIBUTES,
        FILE_SHARE_READ|FILE_SHARE_WRITE|FILE_SHARE_DELETE,nullptr,OPEN_EXISTING,FILE_ATTRIBUTE_NORMAL,nullptr);
    if(handle==INVALID_HANDLE_VALUE) throw std::runtime_error("PLE file identity unavailable");
    BY_HANDLE_FILE_INFORMATION info{};const auto ok=GetFileInformationByHandle(handle,&info);CloseHandle(handle);
    if(!ok || (info.dwFileAttributes&FILE_ATTRIBUTE_DIRECTORY)) throw std::runtime_error("PLE file identity unavailable");
    identity+=std::to_string(info.dwVolumeSerialNumber)+":"+std::to_string((uint64_t(info.nFileIndexHigh)<<32)|info.nFileIndexLow)+":"+
        std::to_string((uint64_t(info.nFileSizeHigh)<<32)|info.nFileSizeLow)+":"+
        std::to_string((uint64_t(info.ftLastWriteTime.dwHighDateTime)<<32)|info.ftLastWriteTime.dwLowDateTime)+"|";
#else
    struct stat info{};
    if(::stat(path.c_str(),&info)!=0 || !S_ISREG(info.st_mode) || info.st_size<0) throw std::runtime_error("PLE file identity unavailable");
    identity+=std::to_string(uint64_t(info.st_dev))+":"+std::to_string(uint64_t(info.st_ino))+":"+
        std::to_string(uint64_t(info.st_size))+":"+std::to_string(info.st_mtim.tv_sec)+":"+std::to_string(info.st_mtim.tv_nsec)+"|";
#endif
    if(identity.size()>65536) throw std::invalid_argument("PLE file identity exceeds snapshot header bound");
    return identity;
}
PleSourceInfo inspect_ple_source(const std::string& filename) {
    PleSourceInfo source;source.path=std::filesystem::canonical(filename).string();source.identity=ple_source_identity(source.path);
    // A standalone shard need not contain model geometry, split companions or
    // token metadata. Parse just this file, never resolve its split family.
    // GgufFile maps read-only without prefaulting; destruction drops the map.
    const strata::GgufFile file(source.path);
    const auto* tensor=file.find("per_layer_token_embd.weight");
    if(!tensor) throw std::invalid_argument("external/internal PLE tensor not found");
    if(tensor->shape.size()!=2 || tensor->shape[0]!=160 || !tensor->shape[1] || tensor->shape[1]>uint64_t(UINT32_MAX)+1)
        throw std::invalid_argument("PLE table shape must be [160,rows] with addressable rows");
    const auto* format=strata::kernels::ple_format_for_type(tensor->type_name());
    if(!format) throw std::invalid_argument("unsupported PLE table format");
    if(format->needs_scale) {
        const auto* marker=file.get("strata.ple.format");const auto* scale=file.get("strata.ple.scale");
        const double value=scale && scale->is_num() && scale->type!=strata::MetaType::BOOL?scale->num():0;
        if(!marker || marker->type!=strata::MetaType::STRING || marker->s!="f8_e4m3" ||
           !std::isfinite(value) || value<=0 || value>std::numeric_limits<float>::max() || float(value)<=0)
            throw std::invalid_argument("PLE FP8 requires f8_e4m3 marker and positive finite FP32 scale");
    }
    source.rows=tensor->shape[1];const auto bytes=checked_product(source.rows,format->row_bytes);
    if(file.data_start()>file.file_size()) throw std::invalid_argument("PLE data offset out of file");
    const auto available=file.file_size()-file.data_start();
    if(tensor->offset>available || bytes>available-tensor->offset ||
       (file.tensors().size()==1 && (tensor->offset || bytes!=available)))
        throw std::invalid_argument("PLE table payload length/offset mismatch");
    source.payload_offset=file.data_start()+tensor->offset;
    source.payload_bytes=bytes;
    if(ple_source_identity(source.path)!=source.identity) throw std::runtime_error("PLE table changed during preflight");
    return source;
}
namespace {
std::pair<uint64_t,uint64_t> ple_worker_stack() {
#ifdef _WIN32
    // Native HPU execution is Linux-only. Keep Windows host fixture helpers
    // conservative rather than deriving target POSIX stack values there.
    return {uint64_t(16)<<20,65536};
#else
    pthread_attr_t attributes;
    if(pthread_attr_init(&attributes)!=0) throw std::runtime_error("PLE RAM worker attributes unavailable");
    size_t stack=0,guard=0;
    const int stack_error=pthread_attr_getstacksize(&attributes,&stack);
    const int guard_error=pthread_attr_getguardsize(&attributes,&guard);
    pthread_attr_destroy(&attributes);
    if(stack_error || guard_error || !stack) throw std::runtime_error("PLE RAM worker stack reservation unavailable");
    return {stack,guard};
#endif
}
}
PleRamMemory ple_ram_memory(const PleSourceInfo& source,const strata::kernels::PleIoOptions& io) {
    validate_ple_io_policy(io);
    if(!io.lock) return {};
#ifdef _WIN32
    SYSTEM_INFO info{};GetSystemInfo(&info);
    return plan_ple_ram(source.payload_offset,source.payload_bytes,info.dwPageSize,0,0,0);
#else
    const auto page=::sysconf(_SC_PAGESIZE);
    if(page<=0) throw std::runtime_error("PLE RAM page size unavailable");
    const auto [stack,guard]=ple_worker_stack();
    return plan_ple_ram(source.payload_offset,source.payload_bytes,uint64_t(page),stack,guard,16);
#endif
}
PleReaderMemory ple_reader_memory(const PleSourceInfo& source,const strata::kernels::PleIoOptions& io) {
    validate_ple_io_policy(io);
    if(io.mode!=strata::kernels::PleIo::Direct) return {};
    if(!source.rows || source.payload_bytes%source.rows || source.payload_bytes/source.rows>320)
        throw std::invalid_argument("invalid PLE reader row byte geometry");
    const auto [stack,guard]=ple_worker_stack();
#ifdef _WIN32
    constexpr uint32_t fallback=4;
#else
    constexpr uint32_t fallback=16;
#endif
    const auto workers=ple_file_workers(std::getenv("STRATA_IO_THREADS"),fallback)+(io.io_thread?1:0);
    return plan_ple_reader(io.cache_rows,uint32_t(source.payload_bytes/source.rows),io.max_inflight,stack,guard,workers);
}
}
