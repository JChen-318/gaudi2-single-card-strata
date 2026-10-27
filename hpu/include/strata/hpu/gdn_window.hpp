#pragma once
#include "strata/hpu/gdn.hpp"
#include "strata/hpu/gdn_window_flow.hpp"
namespace strata::hpu {
// One Model-owned executor shared by all GDN layers. Only the current row
// geometry is retained. External immutable weights are rebound on every call;
// owners keep them alive until this executor is destroyed. Calls are synchronous.
class GdnWindow {
public:
    GdnWindow(const Device&,core::ModelGeometry);
    ~GdnWindow();
    GdnWindow(const GdnWindow&)=delete;
    GdnWindow& operator=(const GdnWindow&)=delete;
    GdnWindowResult run(std::span<const float> input,std::span<const float> state,std::span<const float> conv,
                        std::span<const Binding> weights,const std::function<bool()>& cancelled={});
private:
    struct Impl;std::unique_ptr<Impl> impl_;
};
}
