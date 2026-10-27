#include "strata/hpu/expert.hpp"
#include "strata/hpu/window_limits.hpp"
#include <perf_lib_layer_params.h>

namespace strata::hpu {
void define_bf16_expert(GraphBuilder& graph,const core::ModelGeometry& geometry,uint64_t tokens) {
    if (!tokens || tokens > 128 || geometry.n_embd <= 0 || geometry.n_ff <= 0)
        throw std::invalid_argument("invalid expert geometry or token count");
    const auto h = static_cast<uint64_t>(geometry.n_embd);
    const auto ff = static_cast<uint64_t>(geometry.n_ff);
    const auto x = graph.tensor({"x", DType::BF16, {tokens, h}});
    const auto wg = graph.tensor({"w_gate", DType::BF16, {ff, h}});
    const auto wu = graph.tensor({"w_up", DType::BF16, {ff, h}});
    const auto wd = graph.tensor({"w_down", DType::BF16, {h, ff}});
    const auto out = graph.tensor({"out", DType::BF16, {tokens, h}});
    const auto gate = graph.tensor({"gate", DType::BF16, {tokens, ff}, false});
    const auto up = graph.tensor({"up", DType::BF16, {tokens, ff}, false});
    const auto sigmoid = graph.tensor({"sigmoid", DType::BF16, {tokens, ff}, false});
    const auto silu = graph.tensor({"silu", DType::BF16, {tokens, ff}, false});
    const auto activation = graph.tensor({"activation", DType::BF16, {tokens, ff}, false});
    graph.gemm(x, wg, gate, true);
    graph.gemm(x, wu, up, true);
    graph.sigmoid(gate, sigmoid);
    graph.multiply(gate, sigmoid, silu);
    graph.multiply(silu, up, activation);
    graph.gemm(activation, wd, out, true);
}
std::unique_ptr<Recipe> build_bf16_expert(const Device& device,const core::ModelGeometry& geometry,uint64_t tokens) {
    GraphBuilder graph(device);define_bf16_expert(graph,geometry,tokens);
    return graph.compile("strata_bf16_expert_T"+std::to_string(tokens));
}
void define_moe_router(GraphBuilder& graph,const core::ModelGeometry& g,uint64_t tokens,bool merged) {
    // tokens is a GEMM shape here, not a fixed array, so the only bound is the widest resident window.
    if(!tokens || tokens>kResidentWindowRows || g.n_embd<=0 || g.n_expert<=0 || g.n_expert>4096) throw std::invalid_argument("invalid MoE router geometry");
    const uint64_t h=g.n_embd,e=g.n_expert;
    if(merged) {
        // MoeBlock's merged buffer: router rows then the shared gate row, one GEMM split into logits and dot.
        const auto x=graph.tensor({"x",DType::BF16,{tokens,h}}),w=graph.tensor({"w_router",DType::BF16,{e+1,h}});
        const auto both=graph.tensor({"router_both",DType::F32,{tokens,e+1},false}),logits=graph.tensor({"logits",DType::F32,{tokens,e}});
        const auto dot=graph.tensor({"shared_dot",DType::F32,{tokens,1},false}),gate=graph.tensor({"shared_gate",DType::F32,{tokens,1}});
        graph.gemm(x,w,both,true);
        synSplitParams p{};p.axis=0;const TensorId in[]={both},out[]={logits,dot};graph.op("split",in,out,&p,sizeof(p));
        graph.sigmoid(dot,gate);return;
    }
    const auto x=graph.tensor({"x",DType::BF16,{tokens,h}}),w=graph.tensor({"w_router",DType::BF16,{e,h}});
    const auto shared=graph.tensor({"w_shared_gate",DType::BF16,{1,h}}),logits=graph.tensor({"logits",DType::F32,{tokens,e}});
    const auto dot=graph.tensor({"shared_dot",DType::F32,{tokens,1},false}),gate=graph.tensor({"shared_gate",DType::F32,{tokens,1}});
    graph.gemm(x,w,logits,true);graph.gemm(x,shared,dot,true);graph.sigmoid(dot,gate);
}
std::unique_ptr<Recipe> build_moe_router(const Device& d,const core::ModelGeometry& g,uint64_t tokens) {
    GraphBuilder graph(d);define_moe_router(graph,g,tokens);return graph.compile("strata_moe_router_T"+std::to_string(tokens));
}
} // namespace strata::hpu
