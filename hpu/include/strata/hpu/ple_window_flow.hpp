#pragma once
#include <cstdint>
#include <functional>
#include <span>
#include <vector>
namespace strata::hpu {
inline constexpr uint32_t kPleWindowRows=8;
struct PleWindowResult {std::vector<float> output,history;};
using PleWindowRunner=std::function<PleWindowResult(std::span<const float> hidden,std::span<const float> embedding,std::span<const float> history)>;
uint64_t ple_window_pinned_bytes(uint32_t hidden,uint32_t streams,uint32_t rows);
// Stage1..128 rows through native chunks<=8. History is9 rows, oldest first,
// and contains normalized gated values. Never mutates accepted caller spans.
PleWindowResult execute_ple_window(std::span<const float> residual,std::span<const float> embedding,std::span<const float> history,
                                   uint32_t hidden,uint32_t streams,const PleWindowRunner&,const std::function<bool()>& cancelled={});
}
