#pragma once
#include <cstdint>
#include <stdexcept>
namespace strata::hpu {
inline constexpr uint32_t kMaxGenerationCheckpoints=64;
inline uint64_t cache_metadata_add(uint64_t a,uint64_t b) {
    if(b>UINT64_MAX-a) throw std::overflow_error("cache metadata size overflow");
    return a+b;
}
// Separate from Model image payloads. Per slot: bank token vector, interrupted
// read token vector and one taken/incoming token overlap (3*context*4); bank
// logits and incoming/taken logits (2*vocab*4); descriptors/callables/selection
// scratch (4096). Plan the supported maximum even if a run selects fewer.
inline uint64_t generation_cache_metadata_bytes(uint64_t context,uint64_t vocabulary,uint32_t slots=kMaxGenerationCheckpoints) {
    if(!context || context>INT32_MAX || !vocabulary || vocabulary>INT32_MAX || slots>kMaxGenerationCheckpoints)
        throw std::invalid_argument("invalid cache metadata geometry");
    const auto one=cache_metadata_add(cache_metadata_add(context*12,vocabulary*8),4096);
    return one*slots; // bounded geometry above cannot overflow uint64
}
// Scalar verification holds one root token/logit copy and <=8 picks. The
// root MODEL image belongs to the existing snapshot pool, not this envelope.
inline uint64_t generation_scalar_window_bytes(uint64_t context,uint64_t vocabulary) {
    (void)generation_cache_metadata_bytes(context,vocabulary,0);
    return (context+vocabulary)*4+4096;
}
}
