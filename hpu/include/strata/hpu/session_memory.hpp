#pragma once
#include "strata/core/conversation_memory.hpp"
#include <functional>
#include <stdexcept>
namespace strata::hpu {
using SessionMemoryProbe=std::function<std::optional<uint64_t>()>;
class SessionMemoryRefused:public std::runtime_error {
public:using std::runtime_error::runtime_error;
};
// The sample is not a reservation. Call before allocating a complete disk
// snapshot and again with allocation=0 before publish/apply. Unknown fails
// closed, including when the configured free-memory floor is zero.
inline void check_session_memory(uint64_t allocation,uint64_t floor,const SessionMemoryProbe& probe={}) {
    const auto available=probe?probe():core::available_host_bytes();
    if(!core::conversation_memory_admit(available,allocation,floor))
        throw SessionMemoryRefused("session host memory guard: need "+std::to_string(allocation)+" bytes plus floor "+std::to_string(floor)+"; available "+(available?std::to_string(*available):"unknown"));
}
}
