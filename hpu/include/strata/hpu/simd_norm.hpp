#pragma once
#include <cstdlib>

namespace strata::hpu {
// Process-wide arithmetic choice: frozen on first use so recipes and disk-state identity agree.
inline bool simd_norm_enabled() {
    static const bool enabled=[] {
        const char* value=std::getenv("STRATA_HPU_SIMD_NORM");
        return value && value[0]=='1' && value[1]=='\0';
    }();
    return enabled;
}
}
