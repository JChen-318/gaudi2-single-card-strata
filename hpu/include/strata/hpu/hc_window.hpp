#pragma once
#include "strata/hpu/hc_ple.hpp"
#include "strata/hpu/layer_flow.hpp"
namespace strata::hpu {
// Shared by all target layers under Model serialization. Holds one read and
// one write graph at a time; weights rebind for each HC half/layer. This avoids
// allocating48 copies of window-sized pinned/HBM buffers.
class HcWindow {
public:
    HcWindow(const Device&,HcConfig,uint32_t maximum_rows);
    ~HcWindow();
    HcWindow(const HcWindow&)=delete;
    HcReadResult read(std::span<const float>,std::span<const Binding>,bool final_mix=false);
    std::vector<float> write(std::span<const float>,std::span<const float> block,std::span<const float> inject);
private:
    struct Impl;std::unique_ptr<Impl> impl_;
};
}
