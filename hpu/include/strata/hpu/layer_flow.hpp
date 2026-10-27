#pragma once
#include "strata/hpu/tensor.hpp"
#include <functional>
#include <span>

namespace strata::hpu {
struct HcReadResult {std::vector<float> mixed,inject;};
struct LayerOperations {
    std::function<std::vector<float>(std::span<const float>)> ple; // Optional, before first read.
    std::function<HcReadResult(std::span<const float>,bool ffn)> read;
    std::function<std::vector<float>(std::span<const float>)> mixer,moe;
    std::function<std::vector<float>(std::span<const float>,std::span<const float>,std::span<const float>,bool ffn)> write;
    // Mixer.decode is atomic on failure. This callback restores its checkpoint
    // only after a successful decode followed by a later failed layer stage.
    std::function<void()> rollback_mixer;
    // Optional paired multi-row HC callbacks for [T,HC,H]. PLE stays causal;
    // window_mixer can replace row mixer calls with one atomic native window.
    std::function<HcReadResult(std::span<const float>,bool ffn)> window_read;
    std::function<std::vector<float>(std::span<const float>,std::span<const float>,std::span<const float>,bool ffn)> window_write;
    // Optional atomic whole-window mixer, used with paired window HC. A failed
    // call must leave accepted mixer state unchanged; later stages use rewind.
    std::function<std::vector<float>(std::span<const float>)> window_mixer;
    // Optional whole-window PLE before paired window HC. Its owner stages
    // history until the entire layer succeeds, like the row PLE callback.
    std::function<std::vector<float>(std::span<const float>)> window_ple;
};
// Actual native layer orchestration also used by device-free contract tests.
// PLE history is staged by its callback and committed by the owner only after
// this function succeeds. Every exported intermediate must have finite values.
std::vector<float> execute_layer(std::span<const float> residual,uint32_t hidden,uint32_t streams,const LayerOperations&);
// Row-major [T,HC,H], T<=128. PLE and fallback mixers run in token order;
// all FFN rows enter one MoE call. The owner stages PLE history and supplies
// one mixer rewind for the entire window, including a later row's failure.
std::vector<float> execute_layer_window(std::span<const float>,uint32_t hidden,uint32_t streams,const LayerOperations&,
                                      const std::function<bool()>& cancelled={});
}
