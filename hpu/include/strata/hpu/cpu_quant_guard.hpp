#pragma once
#include <cstdint>
namespace strata::hpu {
// Called by verified build copies of the original CPU quantization entry
// points. No allocation/throw: worker errors propagate as a NaN block scale.
// False leaves the original normal-range path byte-for-byte in control.
bool guard_q8k(const float* input,void* output,int64_t elements) noexcept;
using QuantizeRow=void(*)(const float*,void*,int64_t);
// Q8_0 normal blocks use the caller's EXACT original implementation, including
// its tie rounding, when another block needs guarding. Tiny FP16-zero blocks
// become zeros; nonfinite/overflow blocks carry a NaN scale for post-join error.
bool guard_q8_0(const float*,void*,int64_t elements,QuantizeRow original) noexcept;
}
