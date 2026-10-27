#pragma once
#include "strata/hpu/window_limits.hpp"
#include <cstdint>
#include <functional>
#include <span>
#include <vector>
namespace strata::hpu {
inline constexpr uint32_t kGdnWindowRows=8;
struct GdnWindowResult {std::vector<float> output,state,conv;};
using GdnWindowRunner=std::function<GdnWindowResult(std::span<const float>,std::span<const float>,std::span<const float>)>;
// Stage all subwindows before the caller commits any state. Failure/cancel
// never mutates the accepted input spans. A runner owns no accepted state.
GdnWindowResult execute_gdn_window(std::span<const float> input,uint32_t hidden,std::span<const float> state,
                                 std::span<const float> conv,const GdnWindowRunner&,const std::function<bool()>& cancelled={});
}
