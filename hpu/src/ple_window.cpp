#include "strata/hpu/ple_window.hpp"
#include "strata/hpu/host_recipe.hpp"
#include <mutex>
#include <set>
namespace strata::hpu {
struct PleWindow::Impl {
    Device device;HcConfig config;std::vector<Binding> weights;uint32_t rows=0;
    std::unique_ptr<HostRecipe> execute;std::mutex mutex;
    Impl(const Device& d,HcConfig c,std::span<const Binding> w):device(d),config(c),weights(w.begin(),w.end()) {}
};
PleWindow::PleWindow(const Device& device,HcConfig config,std::span<const Binding> weights) {
    config.validate();std::set<std::string> names;for(const auto& binding:weights) names.insert(binding.name);
    if(weights.size()!=6 || names!=std::set<std::string>{"w_key","w_value","norm_key","norm_query","norm_conv","conv_weights"})
        throw std::invalid_argument("PLE window requires two projections, three norms and conv weights");
    impl_=std::make_unique<Impl>(device,config,weights);
}
PleWindow::~PleWindow()=default;
PleWindowResult PleWindow::run(std::span<const float> residual,std::span<const float> embedding,std::span<const float> history,const std::function<bool()>& cancelled) {
    auto& p=*impl_;std::lock_guard lock(p.mutex);const auto& c=p.config;
    return execute_ple_window(residual,embedding,history,c.hidden,c.streams,[&](auto input,auto embeddings,auto previous) {
        const auto rows=uint32_t(embeddings.size()/c.hidden);
        if(p.rows!=rows) {
            p.execute.reset();p.rows=0;
            p.execute=std::make_unique<HostRecipe>(p.device,build_ple_window(p.device,c,rows),p.weights,std::vector<std::string>{"out","history_out"},
                                                  ple_window_pinned_bytes(c.hidden,c.streams,rows));p.rows=rows;
        }
        p.execute->set("hidden",input);p.execute->set("embedding",embeddings);p.execute->set("history",previous);p.execute->run();
        return PleWindowResult{p.execute->values("out"),p.execute->values("history_out")};
    },cancelled);
}
}
