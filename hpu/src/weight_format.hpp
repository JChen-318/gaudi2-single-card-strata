#pragma once
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat-truncation"
#endif
#include "strata/artifact/dequant.hpp"
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif
#include <ggml.h>
namespace strata::hpu::detail {
struct Format { uint64_t elements, bytes; ggml_to_float_t decode; };
inline Format format(uint32_t type) {
    int elements = 0, bytes = 0;
    const bool known = strata::block_geometry(type, elements, bytes);
    if (type == 0) return {1, 4, nullptr};
    if (type == 42) return {64, 18, nullptr};
    if (type >= GGML_TYPE_COUNT) return {};
    const auto t = static_cast<ggml_type>(type);
    const auto* traits = ggml_get_type_traits(t);
    if (!traits || traits->blck_size <= 0 || traits->blck_size > 256 || !traits->type_size) return {};
    if (known && (traits->blck_size != elements || traits->type_size != size_t(bytes)))
        return {};
    return {uint64_t(traits->blck_size), traits->type_size, traits->to_float};
}
inline bool convertible(uint32_t type) {
    const auto f = format(type);
    return f.elements && (f.decode || type == 0 || type == 30 || type == 42);
}
}
