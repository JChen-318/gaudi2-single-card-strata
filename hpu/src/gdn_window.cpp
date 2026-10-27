#include "strata/hpu/gdn_window.hpp"
#include "strata/hpu/host_recipe.hpp"
#include <mutex>
#include <set>
namespace strata::hpu {
struct GdnWindow::Impl {
    Device device;core::ModelGeometry geometry;uint32_t rows=0;
    std::unique_ptr<HostRecipe> execute;std::mutex mutex;
    Impl(const Device& d,core::ModelGeometry g):device(d),geometry(g) {}
};
GdnWindow::GdnWindow(const Device& device,core::ModelGeometry geometry):impl_(std::make_unique<Impl>(device,geometry)) {}
GdnWindow::~GdnWindow()=default;
GdnWindowResult GdnWindow::run(std::span<const float> input,std::span<const float> state,std::span<const float> conv,
                              std::span<const Binding> weights,const std::function<bool()>& cancelled) {
    auto& p=*impl_;std::lock_guard lock(p.mutex);
    std::set<std::string> names;for(const auto& w:weights) names.insert(w.name);
    if(weights.size()!=9 || names!=std::set<std::string>{"w_qkv","w_gate","w_alpha","w_beta","w_out","conv_weights","norm","dt","ssm_a"})
        throw std::invalid_argument("GDN window requires five projections and four constants");
    const auto& g=p.geometry;
    if(g.n_embd<=0 || uint64_t(g.n_embd)>UINT32_MAX || g.ssm_state_size<=0 || g.ssm_v_heads<=0 || g.ssm_d_conv<2 || g.ssm_conv_channels<=0 ||
       state.size()!=checked_product(checked_product(g.ssm_state_size,g.ssm_v_heads),g.ssm_state_size) ||
       conv.size()!=checked_product(g.ssm_d_conv-1,g.ssm_conv_channels)) throw std::invalid_argument("GDN window state geometry mismatch");
    return execute_gdn_window(input,uint32_t(g.n_embd),state,conv,[&](auto x,auto s,auto c) {
        const auto rows=uint32_t(x.size()/g.n_embd);
        if(p.rows!=rows) {
            p.execute.reset();p.rows=0;
            p.execute=std::make_unique<HostRecipe>(p.device,build_gdn_window(p.device,g,rows),weights,
                                                std::vector<std::string>{"out","state_out","conv_state_out"});p.rows=rows;
        } else for(const auto& binding:weights) p.execute->bind_external(binding);
        p.execute->set("x",x);p.execute->set("state",s);p.execute->set("conv_state",c);p.execute->run();
        return GdnWindowResult{p.execute->values("out"),p.execute->values("state_out"),p.execute->values("conv_state_out")};
    },cancelled);
}
}
