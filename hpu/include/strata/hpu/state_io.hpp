#pragma once
#include "strata/hpu/tensor.hpp"
#include <bit>
#include <span>
#include <string>
#include <type_traits>

namespace strata::hpu {
class StateIoError:public std::runtime_error {
public:
    bool published,storage;
    StateIoError(const std::string& message,bool was_published=false,bool out_of_space=false):std::runtime_error(message),published(was_published),storage(out_of_space) {}
};
// Versioned little-endian streaming state container, with a byte cap and
// CRC64 over the complete payload. Atomic publish occurs only after fsync.
// CRC is for accidental corruption; model identity is checked by its caller.
class StateWriter {
public:
    StateWriter(std::string path,uint64_t max_payload);
    ~StateWriter();
    StateWriter(const StateWriter&)=delete;
    StateWriter& operator=(const StateWriter&)=delete;
    void bytes(std::span<const uint8_t>);
    void integer(uint64_t value);
    void text(const std::string&);
    template<class T> void vector(std::span<const T> values) {
        static_assert(std::is_same_v<T,float> || std::is_same_v<T,uint16_t> || std::is_same_v<T,int32_t> || std::is_same_v<T,int8_t> || std::is_same_v<T,uint8_t>);
        integer(values.size());bytes({reinterpret_cast<const uint8_t*>(values.data()),values.size_bytes()});
    }
    void finish();
private:
    std::string path_,temporary_;
    int fd_=-1;
    uint64_t limit_=0,written_=0,checksum_=0;
    bool published_=false;
};
class StateReader {
public:
    StateReader(const std::string& path,uint64_t max_payload);
    ~StateReader();
    StateReader(const StateReader&)=delete;
    StateReader& operator=(const StateReader&)=delete;
    void bytes(std::span<uint8_t>);
    uint64_t integer();
    std::string text(uint64_t max_bytes=65536);
    template<class T> std::vector<T> vector(uint64_t expected) {
        static_assert(std::is_same_v<T,float> || std::is_same_v<T,uint16_t> || std::is_same_v<T,int32_t> || std::is_same_v<T,int8_t> || std::is_same_v<T,uint8_t>);
        if(integer()!=expected) throw std::invalid_argument("state vector shape mismatch");
        const auto size=checked_product(expected,sizeof(T));
        if(size>remaining_) throw std::invalid_argument("truncated state vector");
        std::vector<T> result(expected);bytes({reinterpret_cast<uint8_t*>(result.data()),size_t(size)});return result;
    }
    void finish();
private:
    int fd_=-1;
    uint64_t remaining_=0,checksum_=0;
    bool finished_=false;
};
}
