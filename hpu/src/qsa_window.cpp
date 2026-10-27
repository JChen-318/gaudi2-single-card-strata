#include "strata/hpu/qsa_window.hpp"
#include "strata/hpu/host_recipe.hpp"
#include <array>
#include <cmath>
#include <mutex>
#include <set>
namespace strata::hpu {
struct QsaWindow::Impl {
    Device device;QsaConfig config;uint32_t hidden,projection_rows=0,attention_rows=0,bucket=0,output_rows=0;
    std::unique_ptr<HostRecipe> projection,attention,output;std::mutex mutex;
    bool device_gather=false;
    Impl(const Device& d,QsaConfig c,uint32_t h):device(d),config(c),hidden(h) {}
    uint32_t rows(size_t elements,uint64_t width) const {
        if(!elements || elements%width || elements/width>kQsaWindowRows) throw std::invalid_argument("QSA native window requires1..8 complete rows");
        return uint32_t(elements/width);
    }
};
QsaWindow::QsaWindow(const Device& device,QsaConfig config,uint32_t hidden) {
    config.validate();if(!hidden) throw std::invalid_argument("QSA shared window hidden width is zero");
    impl_=std::make_unique<Impl>(device,config,hidden);
}
QsaWindow::~QsaWindow()=default;
bool QsaWindow::compatible(const QsaConfig& c,uint32_t hidden) const {
    const auto& p=*impl_;return hidden==p.hidden && qsa_window_geometry_equal(p.config,c);
}
QsaProjected QsaWindow::project(std::span<const float> input,std::span<const Binding> weights) {
    auto& p=*impl_;std::lock_guard lock(p.mutex);const auto rows=p.rows(input.size(),p.hidden);
    std::set<std::string> names;for(const auto& weight:weights) names.insert(weight.name);
    if(weights.size()!=5 || names!=std::set<std::string>{"w_q_gate","w_k","w_v","w_index_query","w_index_key"}) throw std::invalid_argument("QSA window requires five projection weights");
    if(p.projection_rows!=rows) {
        p.projection.reset();p.projection_rows=0;
        p.projection=std::make_unique<HostRecipe>(p.device,build_qsa_projections_window(p.device,p.config,p.hidden,rows),weights,
            std::vector<std::string>{"q_gate","k","v","index_query","index_key"});p.projection_rows=rows;
    } else for(const auto& weight:weights) p.projection->bind_external(weight);
    p.projection->set("x",input);p.projection->run();
    return {p.projection->values("q_gate"),p.projection->values("k"),p.projection->values("v"),p.projection->values("index_query"),p.projection->values("index_key")};
}
std::vector<float> QsaWindow::attention(const QsaWindowBatch& batch,std::span<const Binding> device_cache) {
    auto& p=*impl_;std::lock_guard lock(p.mutex);const auto& c=p.config;
    if(!batch.rows || batch.rows>kQsaWindowRows || !batch.bucket || batch.bucket>c.selection_width(c.capacity)) throw std::invalid_argument("QSA window attention bucket/row mismatch");
    const bool gather=!device_cache.empty();
    if(gather && (device_cache.size()!=2 || batch.ids.size()!=uint64_t(batch.rows)*batch.bucket))
        throw std::invalid_argument("QSA window device gather bindings/ids mismatch");
    if(p.attention_rows!=batch.rows || p.bucket!=batch.bucket || p.device_gather!=gather) {
        p.attention.reset();p.attention_rows=0;p.bucket=0;
        GraphBuilder graph(p.device);define_qsa_attention_window(graph,c,batch.bucket,batch.rows);
        if(gather) define_qsa_device_gather(graph,c,batch.bucket,batch.rows,graph.find("key"),graph.find("value"),graph.find("mask"),c.kv_heads);
        p.attention=std::make_unique<HostRecipe>(p.device,graph.compile("strata_qsa_attention_window_"+std::to_string(batch.rows)+"_"+std::to_string(batch.bucket)+(gather?"_gather":"")),device_cache,std::vector<std::string>{"out"},
                                               qsa_window_attention_pinned_bytes(c,batch.rows,batch.bucket)+
                                               (gather?uint64_t(batch.rows)*batch.bucket*sizeof(float):0));
        const std::array<float,1> scale{1/std::sqrt(float(c.width))};p.attention->set("scale",scale);p.attention_rows=batch.rows;p.bucket=batch.bucket;p.device_gather=gather;
    }
    p.attention->set("query",batch.query);p.attention->set("q_gate",batch.gate);
    if(gather) {
        for(const auto& binding:device_cache) p.attention->bind_external(binding);
        p.attention->set("kv.ids",batch.ids);
    } else {
        p.attention->set("key",batch.key);p.attention->set("value",batch.value);p.attention->set("mask",batch.mask);
    }
    p.attention->run();return p.attention->values("out");
}
std::vector<float> QsaWindow::output(std::span<const float> input,Binding weight) {
    auto& p=*impl_;std::lock_guard lock(p.mutex);const auto rows=p.rows(input.size(),uint64_t(p.config.query_heads)*p.config.width);
    if(weight.name!="w_out") throw std::invalid_argument("QSA window output requires w_out");
    if(p.output_rows!=rows) {
        p.output.reset();p.output_rows=0;const std::array<Binding,1> weights{weight};
        p.output=std::make_unique<HostRecipe>(p.device,build_qsa_output_window(p.device,p.config,p.hidden,rows),weights,std::vector<std::string>{"out"});p.output_rows=rows;
    } else p.output->bind_external(weight);
    p.output->set("attention",input);p.output->run();return p.output->values("out");
}
}
