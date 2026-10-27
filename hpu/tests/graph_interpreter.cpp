#include "graph_interpreter.hpp"
#include "../tpc/packed_codebook_layout.inc"
#include "strata/hpu/recipe.hpp"
#include "strata/hpu/packed_quant.hpp"
#include <algorithm>
#include <cmath>
#include "tpc_emulator.hpp"
#include "packed_emulator.hpp"
namespace emulated {
#define __global__
#define main row_norm_kernel
#include "../tpc/row_norm.c"
#undef main
#define main hc_mix_kernel
#include "../tpc/hc_mix.c"
#undef main
#define main hc_write_kernel
#include "../tpc/hc_write.c"
#undef main
#define main hc_mix_window_kernel
#include "../tpc/hc_mix_window.c"
#undef main
#define main hc_write_window_kernel
#include "../tpc/hc_write_window.c"
#undef main
#define main ple_gate_kernel
#include "../tpc/ple_gate.c"
#undef main
#define main ple_conv_kernel
#include "../tpc/ple_conv.c"
#undef main
#undef __global__
}
namespace oracle {
Graph* current=nullptr;
void require(bool ok,const char* reason) {if(!ok) throw std::runtime_error(reason);}
void close(std::span<const float> a,std::span<const float> b) {
    require(a.size()==b.size(),"graph oracle output shape mismatch");
    for(size_t i=0;i<a.size();++i) require(std::isfinite(a[i]) && std::abs(a[i]-b[i])<2e-5f,"actual QSA factory changed logical math or rounding");
}
}
namespace strata::hpu {
struct GraphBuilder::Impl {oracle::Graph graph;};
GraphBuilder::GraphBuilder(OfflineCompilation):impl_(std::make_unique<Impl>()) {oracle::current=&impl_->graph;}
GraphBuilder::GraphBuilder(const Device&) {throw std::runtime_error("test interpreter forbids device graph construction");}
GraphBuilder::~GraphBuilder()=default;
struct Recipe::Impl {};
Recipe::~Recipe()=default;
std::unique_ptr<Recipe> GraphBuilder::compile(const std::string&) {throw std::runtime_error("CPU graph interpreter cannot compile a native recipe");}
TensorId GraphBuilder::tensor(TensorDesc desc) {
    const auto count=desc.bytes()/element_bytes(desc.dtype);auto& g=impl_->graph;
    for(const auto& slot:g.slots) oracle::require(slot.desc.name!=desc.name,"duplicate interpreted tensor");
    g.slots.push_back({std::move(desc),std::vector<float>(count,NAN)});return TensorId(g.slots.size()-1);
}
void GraphBuilder::reshape(TensorId a,TensorId o) {
    auto& g=impl_->graph;oracle::require(g.slots[a].values.size()==g.slots[o].values.size() && g.slots[a].desc.dtype==g.slots[o].desc.dtype,"interpreted reshape mismatch");
    g.nodes.push_back([&g,a,o] {g.slots[o].values=g.slots[a].values;});
}
void GraphBuilder::cast(TensorId a,TensorId o) {
    auto& g=impl_->graph;oracle::require(g.slots[a].desc.shape==g.slots[o].desc.shape,"interpreted cast shape mismatch");
    g.nodes.push_back([&g,a,o] {for(size_t i=0;i<g.slots[o].values.size();++i) g.slots[o].values[i]=g.slots[o].desc.dtype==DType::BF16?from_bf16(to_bf16(g.slots[a].values[i])):g.slots[a].values[i];});
}
void GraphBuilder::gemm(TensorId a,TensorId b,TensorId o,bool tb) {
    auto& g=impl_->graph;const auto as=g.slots[a].desc.shape,bs=g.slots[b].desc.shape;
    oracle::require(as.size()==2 && bs.size()==2 && as[1]==bs[tb?1:0] && g.slots[o].desc.shape==std::vector<uint64_t>{as[0],bs[tb?0:1]},"interpreted GEMM geometry mismatch");
    g.nodes.push_back([&g,a,b,o,tb] {
        const auto& x=g.slots[a];const auto& w=g.slots[b];auto& out=g.slots[o];const auto m=x.desc.shape[0],k=x.desc.shape[1],n=out.desc.shape[1];
        for(uint64_t r=0;r<m;++r) for(uint64_t c=0;c<n;++c) {
            float sum=0;for(uint64_t i=0;i<k;++i) sum+=x.values[r*k+i]*w.values[tb?c*k+i:i*n+c];
            out.values[r*n+c]=out.desc.dtype==DType::BF16?from_bf16(to_bf16(sum)):sum;
        }
    });
}
void GraphBuilder::packed_matmul(TensorId w,TensorId a,TensorId m,TensorId c,TensorId o,uint32_t columns,uint32_t type,PackedActivation mode) {
    auto& g=impl_->graph;const auto ws=g.slots[w].desc.shape,as=g.slots[a].desc.shape;
    oracle::require(columns>0 && columns<=65536 && ((type==42 && columns%64==0) || ((type==2 || type==3 || type==6 || type==7 || type==8) && columns%32==0)),"interpreted packed format/columns mismatch");
    const uint64_t row_bytes=uint64_t(columns)/(type==42?64:32)*(type==3?20:type==6?22:type==7?24:type==8?34:18);
    oracle::require(packed_activation_supported(type,mode) &&
        g.slots[w].desc.dtype==DType::U8 && g.slots[a].desc.dtype==DType::U8 && g.slots[m].desc.dtype==DType::F32 && g.slots[o].desc.dtype==DType::F32 &&
        g.slots[c].desc.dtype==DType::U8 && g.slots[c].desc.shape==std::vector<uint64_t>{1,STRATA_PACKED_CODEBOOK_BYTES} &&
        ws.size()==2 && as.size()==2 && ws[0]>0 && ws[0]<=1048576 && as[0]>0 && as[0]<=128 && ws[1]==row_bytes && as[1]==columns &&
        g.slots[m].desc.shape==std::vector<uint64_t>{as[0],columns/32,2} && g.slots[o].desc.shape==std::vector<uint64_t>{as[0],ws[0]},"interpreted packed contract mismatch");
    g.nodes.push_back([&g,w,a,m,o,columns,type,mode,row_bytes] {
        std::vector<uint8_t> weights,codes;
        auto bytes=[](const std::vector<float>& input,std::vector<uint8_t>& output) {
            for(float x:input) {oracle::require(std::isfinite(x) && x>=0 && x<=255 && std::floor(x)==x,"invalid interpreted byte");output.push_back(uint8_t(x));}
        };
        bytes(g.slots[w].values,weights);bytes(g.slots[a].values,codes);
        const auto rows=g.slots[w].desc.shape[0],batch=g.slots[a].desc.shape[0];
        using namespace packed_emulator;
        packed_emulator::tensor wt{weights.data(),{row_bytes,rows},1},at{codes.data(),{columns,batch},1},mt{g.slots[m].values.data(),{2,columns/32,batch},4},ot{g.slots[o].values.data(),{rows,batch},4};
        offset={0,0,0,0,0};extent={int(rows),int(batch),1,1,1};packed_kernel(wt,at,mt,ot,columns,int(rows),int(batch),int(type),int(mode));
    });
}
void GraphBuilder::batch_gemm(TensorId a,TensorId b,TensorId o,bool tb) {
    auto& g=impl_->graph;const auto as=g.slots[a].desc.shape,bs=g.slots[b].desc.shape;
    oracle::require(as.size()==3 && bs.size()==3 && as[0]==bs[0] && as[2]==bs[tb?2:1] &&
        g.slots[o].desc.shape==std::vector<uint64_t>{as[0],as[1],bs[tb?1:2]},"interpreted batch GEMM geometry mismatch");
    g.nodes.push_back([&g,a,b,o,tb] {
        const auto& x=g.slots[a];const auto& w=g.slots[b];auto& out=g.slots[o];const auto batches=x.desc.shape[0],m=x.desc.shape[1],k=x.desc.shape[2],n=out.desc.shape[2];
        for(uint64_t batch=0;batch<batches;++batch) for(uint64_t r=0;r<m;++r) for(uint64_t c=0;c<n;++c) {
            float sum=0;for(uint64_t i=0;i<k;++i) sum+=x.values[(batch*m+r)*k+i]*w.values[batch*k*n+(tb?c*k+i:i*n+c)];out.values[(batch*m+r)*n+c]=sum;
        }
    });
}
void GraphBuilder::multiply(TensorId a,TensorId b,TensorId o) {
    auto& g=impl_->graph;oracle::require(g.slots[a].desc.shape==g.slots[o].desc.shape && (g.slots[b].values.size()==1 || g.slots[b].desc.shape==g.slots[a].desc.shape),"interpreted multiply mismatch");
    g.nodes.push_back([&g,a,b,o] {for(size_t i=0;i<g.slots[o].values.size();++i) g.slots[o].values[i]=g.slots[a].values[i]*g.slots[b].values[g.slots[b].values.size()==1?0:i];});
}
void GraphBuilder::add(TensorId a,TensorId b,TensorId o) {
    auto& g=impl_->graph;const auto as=g.slots[a].desc.shape,bs=g.slots[b].desc.shape;
    oracle::require(as==g.slots[o].desc.shape && as.size()>=bs.size(),"interpreted add rank mismatch");
    for(size_t i=0;i<bs.size();++i) oracle::require(bs[bs.size()-1-i]==1 || bs[bs.size()-1-i]==as[as.size()-1-i],"interpreted add broadcast mismatch");
    g.nodes.push_back([&g,a,b,o] {
        const auto& x=g.slots[a];const auto& bias=g.slots[b];
        for(size_t index=0;index<x.values.size();++index) {
            size_t remainder=index,at=0,stride=1;
            for(size_t dim=0;dim<x.desc.shape.size();++dim) {
                const auto coordinate=remainder%x.desc.shape[x.desc.shape.size()-1-dim];remainder/=x.desc.shape[x.desc.shape.size()-1-dim];
                if(dim<bias.desc.shape.size()) {const auto extent=bias.desc.shape[bias.desc.shape.size()-1-dim];at+=(extent==1?0:coordinate)*stride;stride*=extent;}
            }
            g.slots[o].values[index]=x.values[index]+bias.values[at];
        }
    });
}
void GraphBuilder::softmax_last(TensorId a,TensorId o) {
    auto& g=impl_->graph;oracle::require(g.slots[a].desc.shape==g.slots[o].desc.shape,"interpreted softmax shape mismatch");
    g.nodes.push_back([&g,a,o] {
        const auto& x=g.slots[a];auto& out=g.slots[o].values;const auto width=x.desc.shape.back();
        for(size_t row=0;row<x.values.size()/width;++row) {
            const auto begin=x.values.begin()+row*width;const auto peak=*std::max_element(begin,begin+width);float sum=0;
            for(size_t i=0;i<width;++i) {out[row*width+i]=std::exp(x.values[row*width+i]-peak);sum+=out[row*width+i];}
            for(size_t i=0;i<width;++i) out[row*width+i]/=sum;
        }
    });
}
void GraphBuilder::sigmoid(TensorId a,TensorId o) {
    auto& g=impl_->graph;oracle::require(g.slots[a].desc.shape==g.slots[o].desc.shape,"interpreted sigmoid mismatch");
    g.nodes.push_back([&g,a,o] {for(size_t i=0;i<g.slots[o].values.size();++i) g.slots[o].values[i]=1/(1+std::exp(-g.slots[a].values[i]));});
}
namespace {
emulated::tensor tpc_tensor(oracle::Graph& graph,TensorId id) {
    auto& slot=graph.slots[id];auto dims=slot.desc.shape;std::reverse(dims.begin(),dims.end());return {&slot.values,std::move(dims)};
}
}
void GraphBuilder::split_rows(TensorId input,std::span<const TensorId> outputs) {
    auto& g=impl_->graph;const auto shape=g.slots[input].desc.shape;
    oracle::require(shape.size()==2 && outputs.size()==shape[0],"interpreted split geometry mismatch");
    const std::vector<TensorId> ids(outputs.begin(),outputs.end());
    for(auto id:ids) oracle::require(g.slots[id].desc.shape==std::vector<uint64_t>{1,shape[1]},"interpreted split row mismatch");
    g.nodes.push_back([&g,input,ids] {const auto width=g.slots[input].desc.shape[1];for(size_t row=0;row<ids.size();++row) std::copy_n(g.slots[input].values.begin()+row*width,width,g.slots[ids[row]].values.begin());});
}
namespace interpreted_quantizer {
// The oracle executes the same TPC C source the Gaudi2 build ships, through the same adapter the
// packed dot uses, instead of calling the host quantizer the backend is being compared against.
using namespace packed_emulator;
#define __global__
#define main activation_quant_kernel_interpreted
#include "../tpc/activation_quant.c"
#undef main
#undef __global__
}

void GraphBuilder::activation_quant(TensorId input,TensorId codes,TensorId metadata,uint32_t batch,uint32_t columns) {
    auto& g=impl_->graph;const auto& xs=g.slots[input].desc;const auto& cs=g.slots[codes].desc;const auto& ms=g.slots[metadata].desc;
    oracle::require(xs.dtype==DType::F32 && cs.dtype==DType::U8 && ms.dtype==DType::F32 && columns>0 && columns%32==0 &&
        xs.shape==std::vector<uint64_t>{batch,columns} && cs.shape==std::vector<uint64_t>{batch,columns} &&
        ms.shape==std::vector<uint64_t>{batch,columns/32,2} && g.slots[input].values.size()==size_t(batch)*columns,
        "interpreted activation quantizer contract mismatch");
    g.nodes.push_back([&g,input,codes,metadata,batch,columns] {
        auto& source=g.slots[input].values;auto& meta=g.slots[metadata].values;
        std::vector<uint8_t> bytes(size_t(batch)*columns);
        const auto saved_offset=packed_emulator::offset,saved_extent=packed_emulator::extent;
        using namespace interpreted_quantizer;
        packed_emulator::tensor x{source.data(),{columns,batch},4},c{bytes.data(),{columns,batch},1},m{meta.data(),{2,columns/32,batch},4};
        meta.assign(size_t(batch)*(columns/32)*2,0.0f);
        packed_emulator::offset={0,0,0,0,0};packed_emulator::extent={int(columns/32),int(batch),1,1,1};
        activation_quant_kernel_interpreted(x,c,m,int(columns),int(batch));
        packed_emulator::offset=saved_offset;packed_emulator::extent=saved_extent;
        auto& out=g.slots[codes].values;out.clear();out.reserve(bytes.size());
        for(uint8_t byte:bytes) out.push_back(float(byte));
    });
}

void GraphBuilder::concatenate_rows(std::span<const TensorId> inputs,TensorId output) {
    auto& g=impl_->graph;const auto shape=g.slots[output].desc.shape;
    oracle::require(shape.size()==2 && inputs.size()==shape[0],"interpreted concatenate geometry mismatch");
    const std::vector<TensorId> ids(inputs.begin(),inputs.end());
    for(auto id:ids) oracle::require(g.slots[id].desc.shape==std::vector<uint64_t>{1,shape[1]},"interpreted concatenate row mismatch");
    g.nodes.push_back([&g,ids,output] {const auto width=g.slots[output].desc.shape[1];for(size_t row=0;row<ids.size();++row) std::copy(g.slots[ids[row]].values.begin(),g.slots[ids[row]].values.end(),g.slots[output].values.begin()+row*width);});
}
void GraphBuilder::row_norm(TensorId input,TensorId gamma,TensorId output,float epsilon) {
    auto& g=impl_->graph;const auto shape=g.slots[input].desc.shape;oracle::require(shape.size()==2,"interpreted row norm rank mismatch");
    const auto width=shape[1],rows=shape[0],gr=g.slots[gamma].values.size()/width;
    oracle::require(gr && rows%gr==0 && g.slots[gamma].values.size()%width==0,"interpreted norm repeat mismatch");
    const int shared=gr==1?1:gr==rows?0:-int(gr);
    g.nodes.push_back([&g,input,gamma,output,epsilon,width,rows,shared] {
        emulated::offset={};emulated::extent={int(rows),1,1,1,1};
        emulated::row_norm_kernel(tpc_tensor(g,input),tpc_tensor(g,gamma),tpc_tensor(g,output),int(width),int(rows),shared,epsilon);
    });
}
void GraphBuilder::ple_gate(TensorId key,TensorId query,TensorId value,TensorId output) {
    auto& g=impl_->graph;const auto shape=g.slots[key].desc.shape;oracle::require(shape.size()==2,"interpreted PLE gate rank mismatch");
    g.nodes.push_back([&g,key,query,value,output,shape] {
        emulated::offset={};emulated::extent={int(shape[0]),1,1,1,1};
        emulated::ple_gate_kernel(tpc_tensor(g,key),tpc_tensor(g,query),tpc_tensor(g,value),tpc_tensor(g,output),int(shape[1]),int(shape[0]),1/std::sqrt(float(shape[1])));
    });
}
void GraphBuilder::ple_conv(TensorId history,TensorId normalized,TensorId weights,TensorId hidden,TensorId gated,TensorId next,TensorId output,uint32_t dilation) {
    auto& g=impl_->graph;const auto shape=g.slots[weights].desc.shape;
    g.nodes.push_back([&g,history,normalized,weights,hidden,gated,next,output,dilation,shape] {
        emulated::offset={};emulated::extent={int((shape[1]+63)/64),1,1,1,1};
        emulated::ple_conv_kernel(tpc_tensor(g,history),tpc_tensor(g,normalized),tpc_tensor(g,weights),tpc_tensor(g,hidden),tpc_tensor(g,gated),tpc_tensor(g,next),tpc_tensor(g,output),int(shape[1]),int(shape[0]),int(dilation));
    });
}
void GraphBuilder::hc_mix(TensorId normalized,TensorId gate,TensorId output) {
    auto& g=impl_->graph;const auto shape=g.slots[normalized].desc.shape;
    g.nodes.push_back([&g,normalized,gate,output,shape] {emulated::offset={};emulated::extent={int((shape[1]+63)/64),1,1,1,1};emulated::hc_mix_kernel(tpc_tensor(g,normalized),tpc_tensor(g,gate),tpc_tensor(g,output),int(shape[1]),int(shape[0]));});
}
void GraphBuilder::hc_write(TensorId residual,TensorId block,TensorId inject,TensorId output) {
    auto& g=impl_->graph;const auto shape=g.slots[residual].desc.shape;
    g.nodes.push_back([&g,residual,block,inject,output,shape] {emulated::offset={};emulated::extent={int((shape[1]+63)/64),int(shape[0]),1,1,1};emulated::hc_write_kernel(tpc_tensor(g,residual),tpc_tensor(g,block),tpc_tensor(g,inject),tpc_tensor(g,output),int(shape[1]),int(shape[0]));});
}
void GraphBuilder::hc_mix_window(TensorId normalized,TensorId gate,TensorId output) {
    auto& g=impl_->graph;const auto shape=g.slots[normalized].desc.shape;
    g.nodes.push_back([&g,normalized,gate,output,shape] {emulated::offset={};emulated::extent={int((shape[2]+63)/64),int(shape[0]),1,1,1};emulated::hc_mix_window_kernel(tpc_tensor(g,normalized),tpc_tensor(g,gate),tpc_tensor(g,output),int(shape[2]),int(shape[1]),int(shape[0]));});
}
void GraphBuilder::hc_write_window(TensorId residual,TensorId block,TensorId inject,TensorId output) {
    auto& g=impl_->graph;const auto shape=g.slots[residual].desc.shape;
    g.nodes.push_back([&g,residual,block,inject,output,shape] {emulated::offset={};emulated::extent={int((shape[2]+63)/64),int(shape[1]),int(shape[0]),1,1};emulated::hc_write_window_kernel(tpc_tensor(g,residual),tpc_tensor(g,block),tpc_tensor(g,inject),tpc_tensor(g,output),int(shape[2]),int(shape[1]),int(shape[0]));});
}
// Explicit-guid nodes (STRATA_HPU_MERGED_GEMM splits and the like) are not interpreted by these host tests.
void GraphBuilder::op(const char* guid,std::span<const TensorId>,std::span<const TensorId>,const void*,uint32_t) {
    throw std::logic_error(std::string("graph interpreter does not run explicit guid ")+(guid?guid:"(null)"));
}
TensorId GraphBuilder::find(const std::string& name) const {throw std::logic_error("graph interpreter has no tensor lookup: "+name);}
// Nor STRATA_HPU_HC_FUSED's dual-output kernels.
void GraphBuilder::row_norm_dual(TensorId,TensorId,TensorId,TensorId,float) {throw std::logic_error("graph interpreter does not run the dual row norm");}
void GraphBuilder::hc_mix_window_dual(TensorId,TensorId,TensorId,TensorId) {throw std::logic_error("graph interpreter does not run the dual HC mix");}
// STRATA_HPU_FP8_DENSE graphs are not interpreted either.
void GraphBuilder::fp8_linear(TensorId,TensorId,TensorId,TensorId,bool) {
    throw std::logic_error("graph interpreter does not run FP8 linear");
}
}
