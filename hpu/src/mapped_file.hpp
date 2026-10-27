#pragma once
#include "strata/hpu/tensor.hpp"
#include <filesystem>
#include <span>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace strata::hpu::detail {
inline std::string local_identity(const std::string& filename,uint64_t expected) {
    const auto path=std::filesystem::canonical(filename).string();
    std::string value=std::to_string(path.size())+":"+path+"|";
#ifdef _WIN32
    const auto handle=CreateFileW(std::filesystem::path(path).c_str(),FILE_READ_ATTRIBUTES,
        FILE_SHARE_READ|FILE_SHARE_WRITE|FILE_SHARE_DELETE,nullptr,OPEN_EXISTING,FILE_ATTRIBUTE_NORMAL,nullptr);
    if(handle==INVALID_HANDLE_VALUE) throw std::runtime_error("mapped file identity unavailable");
    BY_HANDLE_FILE_INFORMATION info{};const auto ok=GetFileInformationByHandle(handle,&info);CloseHandle(handle);
    const auto bytes=(uint64_t(info.nFileSizeHigh)<<32)|info.nFileSizeLow;
    if(!ok || (info.dwFileAttributes&FILE_ATTRIBUTE_DIRECTORY) || bytes!=expected) throw std::runtime_error("mapped file identity/size changed");
    value+=std::to_string(info.dwVolumeSerialNumber)+":"+std::to_string((uint64_t(info.nFileIndexHigh)<<32)|info.nFileIndexLow)+":"+
        std::to_string(bytes)+":"+std::to_string((uint64_t(info.ftLastWriteTime.dwHighDateTime)<<32)|info.ftLastWriteTime.dwLowDateTime)+"|";
#else
    struct stat info{};
    if(::stat(path.c_str(),&info) || !S_ISREG(info.st_mode) || info.st_size<0 || uint64_t(info.st_size)!=expected)
        throw std::runtime_error("mapped file identity/size changed");
    value+=std::to_string(uint64_t(info.st_dev))+":"+std::to_string(uint64_t(info.st_ino))+":"+
        std::to_string(uint64_t(info.st_size))+":"+std::to_string(info.st_mtim.tv_sec)+":"+std::to_string(info.st_mtim.tv_nsec)+"|";
#endif
    return value;
}
class ReadOnlyFile {
public:
    const std::string path;
    const uint64_t bytes;
    explicit ReadOnlyFile(const std::string& filename):path(std::filesystem::canonical(filename).string()),bytes(std::filesystem::file_size(path)) {
        if(!bytes || bytes>SIZE_MAX) throw std::invalid_argument("invalid read-only mapped file size");
        const auto before=local_identity(path,bytes);
#ifdef _WIN32
        file_=CreateFileW(std::filesystem::path(path).c_str(),GENERIC_READ,FILE_SHARE_READ|FILE_SHARE_WRITE|FILE_SHARE_DELETE,
            nullptr,OPEN_EXISTING,FILE_ATTRIBUTE_NORMAL,nullptr);
        if(file_==INVALID_HANDLE_VALUE) throw std::runtime_error("cannot open read-only file");
        mapping_=CreateFileMappingW(file_,nullptr,PAGE_READONLY,0,0,nullptr);
        if(mapping_) data_=static_cast<const uint8_t*>(MapViewOfFile(mapping_,FILE_MAP_READ,0,0,0));
#else
        fd_=::open(path.c_str(),O_RDONLY|O_CLOEXEC);
        if(fd_>=0) {
            auto* address=::mmap(nullptr,size_t(bytes),PROT_READ,MAP_PRIVATE,fd_,0);
            if(address!=MAP_FAILED) data_=static_cast<const uint8_t*>(address);
        }
#endif
        try {if(!data_ || local_identity(path,bytes)!=before) throw std::runtime_error("read-only mapping failed or file changed");}
        catch(...) {close();throw;}
    }
    ~ReadOnlyFile() {close();}
    ReadOnlyFile(const ReadOnlyFile&)=delete;ReadOnlyFile& operator=(const ReadOnlyFile&)=delete;
    std::span<const uint8_t> range(uint64_t offset,uint64_t count) const {check_range(bytes,offset,count);return {data_+offset,size_t(count)};}
private:
    const uint8_t* data_=nullptr;
#ifdef _WIN32
    HANDLE file_=INVALID_HANDLE_VALUE,mapping_=nullptr;
#else
    int fd_=-1;
#endif
    void close() noexcept {
#ifdef _WIN32
        if(data_) UnmapViewOfFile(data_);
        if(mapping_) CloseHandle(mapping_);
        if(file_!=INVALID_HANDLE_VALUE) CloseHandle(file_);
#else
        if(data_) ::munmap(const_cast<uint8_t*>(data_),size_t(bytes));
        if(fd_>=0) ::close(fd_);
#endif
        data_=nullptr;
    }
};
}
