#pragma once
#include <cstdint>
#include <span>
namespace strata::hpu {
// The pinned ggml reference Q4_0 encoder, matching upstream --mtp-q4. Output
// is caller-owned and row-major; no full-matrix staging or SDK initialization.
// Nonfinite values, unsafe reciprocals and nonfinite FP16 scales reject before
// invoking ggml's integer conversion. All inputs validate before output changes.
void quantize_q4_rows(std::span<const float> input,uint32_t columns,std::span<uint8_t> output);
}
