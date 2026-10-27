#include "strata/hpu/model_head.hpp"
#include "strata/hpu/packed_quant.hpp"
#include "strata/hpu/window_limits.hpp"
#include "../tpc/packed_codebook_layout.inc"
namespace strata::hpu {
void define_model_head(GraphBuilder& graph,uint32_t hidden,uint32_t vocabulary,uint32_t batch) {
    if(!hidden || !vocabulary || hidden>65536 || vocabulary>1048576 || !batch || batch>kResidentWindowRows) throw std::invalid_argument("unsupported output head geometry");
    const auto x=graph.tensor({"x",DType::BF16,{batch,hidden}}),weight=graph.tensor({"weight",DType::BF16,{vocabulary,hidden}});
    const auto output=graph.tensor({"logits",DType::F32,{batch,vocabulary}});graph.gemm(x,weight,output,true);
}
std::unique_ptr<Recipe> build_model_head(const Device& device,uint32_t hidden,uint32_t vocabulary,uint32_t batch) {
    GraphBuilder graph(device);define_model_head(graph,hidden,vocabulary,batch);return graph.compile("strata_lm_head_"+std::to_string(batch));
}
void define_packed_matmul(GraphBuilder& graph,uint32_t columns,uint32_t rows,uint32_t batch,uint32_t type,PackedActivation mode,bool device_quantizer) {
    const auto bytes=packed_row_bytes(type,columns);
    if(!rows || rows>1048576 || !batch || batch>kResidentWindowRows) throw std::invalid_argument("invalid packed matmul rows/batch");
    const auto w=graph.tensor({"weight",DType::U8,{rows,bytes}}),o=graph.tensor({"logits",DType::F32,{batch,rows}});
    TensorId a,m;
    if(device_quantizer) {
        // The FP32 activation stays on the device and the Q8_1 codes and per-block metadata
        // are produced by the quantizer kernel inside this same graph.
        const auto x=graph.tensor({"activation_input",DType::F32,{batch,columns}});
        a=graph.tensor({"activation",DType::U8,{batch,columns}});
        m=graph.tensor({"metadata",DType::F32,{batch,columns/32,2}});
        graph.activation_quant(x,a,m,batch,columns);
    } else {
        a=graph.tensor({"activation",DType::U8,{batch,columns}});
        m=graph.tensor({"metadata",DType::F32,{batch,columns/32,2}});
    }
    // One read-only codebook per graph: the six IQ grids exceed a kernel's static scalar budget,
    // and the dot reads one magnitude byte per value from here.
    const auto c=graph.tensor({"codebook",DType::U8,{1,STRATA_PACKED_CODEBOOK_BYTES}});
    graph.packed_matmul(w,a,m,c,o,columns,type,mode);
}
}
