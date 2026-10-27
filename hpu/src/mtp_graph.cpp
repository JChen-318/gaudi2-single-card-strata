#include "strata/hpu/mtp.hpp"

namespace strata::hpu {
void define_mtp_input(GraphBuilder& graph,uint32_t hidden,uint32_t streams) {
    if(!hidden || hidden%64 || hidden>65536 || !streams || streams>32) throw std::invalid_argument("invalid MTP input geometry");
    const auto e=graph.tensor({"embedding",DType::BF16,{1,hidden}}),h=graph.tensor({"hidden",DType::BF16,{streams,hidden}});
    const auto we=graph.tensor({"w_embedding",DType::BF16,{hidden,hidden}}),wh=graph.tensor({"w_hidden",DType::BF16,{hidden,hidden}});
    const auto ep=graph.tensor({"embedding_projected",DType::F32,{1,hidden},false}),hp=graph.tensor({"hidden_projected",DType::F32,{streams,hidden},false});
    const auto r=graph.tensor({"residual",DType::F32,{streams,hidden}});
    graph.gemm(e,we,ep,true);graph.gemm(h,wh,hp,true);graph.add(hp,ep,r);
}
void define_mtp_projections(GraphBuilder& graph,const QsaConfig& c,uint32_t hidden) {
    c.validate();if(!hidden) throw std::invalid_argument("invalid MTP projection hidden width");
    const auto x=graph.tensor({"x",DType::BF16,{1,hidden}});
    auto project=[&](const std::string& name,uint64_t heads,uint64_t width) {
        const auto rows=checked_product(heads,width);
        const auto w=graph.tensor({"w_"+name,DType::BF16,{rows,hidden}}),m=graph.tensor({name+"_matrix",DType::F32,{1,rows},false});
        const auto out=graph.tensor({name,DType::F32,{heads,width}});graph.gemm(x,w,m,true);graph.reshape(m,out);
    };
    project("q_gate",c.query_heads,uint64_t(c.width)*2);project("k",c.kv_heads,c.width);project("v",c.kv_heads,c.width);
}
}
