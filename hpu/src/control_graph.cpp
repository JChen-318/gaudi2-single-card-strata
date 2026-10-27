#include "strata/hpu/control_graph.hpp"
namespace strata::hpu {
void define_control_vector(GraphBuilder& graph,uint32_t hidden,uint32_t streams,ControlVectorMode mode) {
    if(!hidden || hidden>65536 || !streams || streams>32 || (mode!=ControlVectorMode::Add && mode!=ControlVectorMode::Project)) throw std::invalid_argument("invalid control vector graph geometry/mode");
    const auto r=graph.tensor({"residual",DType::F32,{streams,hidden}}),d=graph.tensor({"direction",DType::F32,{1,hidden}});
    const auto out=graph.tensor({"out",DType::F32,{streams,hidden}});
    if(mode==ControlVectorMode::Add) {graph.add(r,d,out);return;}
    const auto s=graph.tensor({"negative_scale",DType::F32,{1}}),dot=graph.tensor({"dot",DType::F32,{streams,1},false});
    const auto factor=graph.tensor({"factor",DType::F32,{streams,1},false}),correction=graph.tensor({"correction",DType::F32,{streams,hidden},false});
    graph.gemm(r,d,dot,true);graph.multiply(dot,s,factor);graph.gemm(factor,d,correction);graph.add(r,correction,out);
}
}
