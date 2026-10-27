#include "strata/hpu/hc_window.hpp"
#include "strata/hpu/host_recipe.hpp"
#include "strata/hpu/window_limits.hpp"
#include <array>
#include <mutex>
#include <set>
namespace strata::hpu {
struct HcWindow::Impl {
    Device device;HcConfig config;uint32_t maximum,read_rows=0,write_rows=0;bool read_final=false;
    std::unique_ptr<HostRecipe> reader,writer;std::mutex mutex;
    Impl(const Device& d,HcConfig c,uint32_t rows):device(d),config(c),maximum(rows) {}
    uint32_t rows(std::span<const float> residual) const {
        const uint64_t n=uint64_t(config.hidden)*config.streams;
        if(residual.empty() || residual.size()%n || residual.size()/n>maximum) throw std::invalid_argument("shared HC window row geometry mismatch");
        return uint32_t(residual.size()/n);
    }
};
HcWindow::HcWindow(const Device& device,HcConfig config,uint32_t maximum) {
    config.validate();if(!maximum || maximum>kResidentWindowRows) throw std::invalid_argument("shared HC window cap must be1.."+std::to_string(kResidentWindowRows));
    impl_=std::make_unique<Impl>(device,config,maximum);
}
HcWindow::~HcWindow()=default;
HcReadResult HcWindow::read(std::span<const float> residual,std::span<const Binding> weights,bool final) {
    auto& p=*impl_;std::lock_guard lock(p.mutex);const auto rows=p.rows(residual);
    std::set<std::string> names;for(const auto& weight:weights) names.insert(weight.name);
    const auto expected=final?std::set<std::string>{"w_down","w_up","norm"}:std::set<std::string>{"w_down","w_up","w_inject","norm"};
    if(weights.size()!=expected.size() || names!=expected) throw std::invalid_argument("HC window read binding contract mismatch");
    if(rows!=p.read_rows || final!=p.read_final) {
        p.reader.reset();p.read_rows=0;GraphBuilder graph(p.device);define_hc_read_window(graph,p.config,rows,final);
        auto next=std::make_unique<HostRecipe>(p.device,graph.compile("strata_hc_read_window_"+std::to_string(rows)+(final?"_final":"")),weights,final?std::vector<std::string>{"mixed"}:std::vector<std::string>{"mixed","inject"});
        const std::array<float,1> scale{1/float(p.config.streams)};next->set("inverse_streams",scale);p.reader=std::move(next);p.read_rows=rows;p.read_final=final;
    } else for(const auto& binding:weights) p.reader->bind_external(binding);
    p.reader->set("residual",residual);p.reader->run();return {p.reader->values("mixed"),final?std::vector<float>{}:p.reader->values("inject")};
}
std::vector<float> HcWindow::write(std::span<const float> residual,std::span<const float> block,std::span<const float> inject) {
    auto& p=*impl_;std::lock_guard lock(p.mutex);const auto rows=p.rows(residual);
    if(block.size()!=uint64_t(rows)*p.config.hidden || inject.size()!=uint64_t(rows)*p.config.streams) throw std::invalid_argument("HC window write input mismatch");
    if(rows!=p.write_rows) {
        p.writer.reset();p.write_rows=0;GraphBuilder graph(p.device);define_hc_write_window(graph,p.config,rows);
        p.writer=std::make_unique<HostRecipe>(p.device,graph.compile("strata_hc_write_window_"+std::to_string(rows)),std::span<const Binding>{},std::vector<std::string>{"out"});p.write_rows=rows;
    }
    p.writer->set("residual",residual);p.writer->set("block",block);p.writer->set("inject",inject);p.writer->run();return p.writer->values("out");
}
}
