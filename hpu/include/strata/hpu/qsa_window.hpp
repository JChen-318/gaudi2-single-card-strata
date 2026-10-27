#pragma once
#include "strata/hpu/qsa.hpp"
#include "strata/hpu/qsa_window_flow.hpp"
namespace strata::hpu {
// Model-owned shared projection/attention/output recipes, one current geometry
// each. All returned host values are owned; calls serialize and rebind weights.
// Destroy before the layer weights last borrowed by these recipes.
class QsaWindow {
public:
    QsaWindow(const Device&,QsaConfig,uint32_t hidden);
    ~QsaWindow();
    QsaWindow(const QsaWindow&)=delete;
    QsaWindow& operator=(const QsaWindow&)=delete;
    bool compatible(const QsaConfig&,uint32_t hidden) const;
    QsaProjected project(std::span<const float>,std::span<const Binding>);
    std::vector<float> attention(const QsaWindowBatch&,std::span<const Binding> device_cache={});
    std::vector<float> output(std::span<const float>,Binding);
private:
    struct Impl;std::unique_ptr<Impl> impl_;
};
}
