#pragma once
#include "strata/hpu/hc_ple.hpp"
#include "strata/hpu/ple_window_flow.hpp"
namespace strata::hpu {
// Only layer1 owns PLE. Retains one current bounded graph; immutable borrowed
// weights must outlive it. History stays caller-owned and is returned staged.
class PleWindow {
public:
    PleWindow(const Device&,HcConfig,std::span<const Binding> weights);
    ~PleWindow();
    PleWindow(const PleWindow&)=delete;
    PleWindow& operator=(const PleWindow&)=delete;
    PleWindowResult run(std::span<const float> residual,std::span<const float> embedding,std::span<const float> history,
                        const std::function<bool()>& cancelled={});
private:
    struct Impl;std::unique_ptr<Impl> impl_;
};
}
