#include "strata/hpu/qsa.hpp"
#include "strata/hpu/qsa_window_flow.hpp"
#include "strata/hpu/performance_modes.hpp"
#include "strata/hpu/model_abi.hpp"
#include <perf_lib_layer_params.h>
#include <algorithm>
#include <cmath>
#include <span>

namespace strata::hpu {
TensorId qsa_cache_read(GraphBuilder& g,const std::string& name,uint64_t cells,uint64_t hk,uint64_t d,DType dtype) {
    if(!qsa_kv_fp8_enabled()) {
        const auto cache=g.tensor({name,DType::F32,{cells,hk,d},true});
        if(dtype==DType::F32) return cache;
        const auto out=g.tensor({name+"_cast",dtype,{cells,hk,d},false});g.cast(cache,out);return out;
    }
    if(hk>kQsaScaleSlots) throw std::invalid_argument("FP8 QSA cache: more KV heads than scale slots");
    const auto codes=g.tensor({name,DType::F8,{cells,hk,d},true}),slots=g.tensor({name+"_scale",DType::F32,{cells,kQsaScaleSlots},true});
    const auto values=g.tensor({name+"_values",dtype,{cells,hk,d},false}),out=g.tensor({name+"_dequant",dtype,{cells,hk,d},false});
    g.cast(codes,values);
    const auto scale=g.tensor({name+"_scale_used",DType::F32,{cells,hk},false});
    {synSliceParams q{};q.axes[0]=0;q.starts[0]=0;q.ends[0]=hk;q.steps[0]=1;q.axes[1]=1;q.starts[1]=0;q.ends[1]=cells;q.steps[1]=1;
     const TensorId in[]={slots},o[]={scale};g.op("slice",in,o,&q,sizeof(q));}
    auto scale3=g.tensor({name+"_scale3",DType::F32,{cells,hk,1},false});g.reshape(scale,scale3);
    if(dtype!=DType::F32) {const auto narrow=g.tensor({name+"_scale3_cast",dtype,{cells,hk,1},false});g.cast(scale3,narrow);scale3=narrow;}
    {const TensorId in[]={values,scale3},o[]={out};g.op(dtype==DType::F32?"mult_fwd_f32":"mult_fwd_bf16",in,o);} // broadcast over width
    return out;
}
void qsa_cache_write(GraphBuilder& g,const std::string& name,TensorId rows,uint64_t cells,uint64_t hk,uint64_t d) {
    if(!qsa_kv_fp8_enabled()) {
        const TensorId in[]={rows},out[]={g.tensor({name,DType::F32,{cells,hk,d},true})};g.op("memcpy",in,out);return;
    }
    const auto T=[&](const std::string& stem,DType dtype,std::vector<uint64_t> shape) {return g.tensor({name+"_q_"+stem,dtype,std::move(shape),false});};
    const auto magnitude=T("abs",DType::F32,{cells,hk,d}),peak=T("peak",DType::F32,{cells,hk,1});
    const auto inverse_range=T("inverse_range",DType::F32,{1}),epsilon=T("epsilon",DType::F32,{1});
    const auto scaled_peak=T("scaled_peak",DType::F32,{cells,hk,1}),scale=T("scale",DType::F32,{cells,hk,1}),scaled=T("scaled",DType::F32,{cells,hk,d});
    const auto constant=[&](TensorId out,float value) {
        ns_ConstantKernel::Params params{};params.constant.f=value;const TensorId outs[]={out};g.op("constant_f32",{},outs,&params,sizeof(params));
    };
    const auto binary=[&](const char* guid,TensorId a,TensorId b,TensorId out) {const TensorId ins[]={a,b},outs[]={out};g.op(guid,ins,outs);};
    {const TensorId ins[]={rows},outs[]={magnitude};g.op("abs_fwd_f32",ins,outs);}
    {ns_Reduction::Params params{};params.reductionDimension=0;const TensorId ins[]={magnitude},outs[]={peak};g.op("reduce_max_fwd_f32",ins,outs,&params,sizeof(params));}
    constant(inverse_range,1.0f/240.0f);constant(epsilon,1e-30f);
    binary("mult_fwd_f32",peak,inverse_range,scaled_peak);binary("add_fwd_f32",scaled_peak,epsilon,scale);
    binary("div_fwd_f32",rows,scale,scaled);
    g.cast(scaled,g.tensor({name,DType::F8,{cells,hk,d},true}));
    // The scales into each cell's slot, zero padded.
    if(hk>kQsaScaleSlots) throw std::invalid_argument("FP8 QSA cache: more KV heads than scale slots");
    const auto used=T("scale_used",DType::F32,{cells,hk}),slots=g.tensor({name+"_scale",DType::F32,{cells,kQsaScaleSlots},true});
    g.reshape(scale,used);
    if(hk==kQsaScaleSlots) g.reshape(used,slots);
    else {
        const auto padding=T("scale_padding",DType::F32,{cells,kQsaScaleSlots-hk});
        {ns_ConstantKernel::Params params{};params.constant.f=0.0f;const TensorId outs[]={padding};g.op("constant_f32",{},outs,&params,sizeof(params));}
        const TensorId parts[]={used,padding},o[]={slots};synConcatenateParams q{};q.axis=0;g.op("concat",parts,o,&q,sizeof(q)); // SDK axis 0: the slots
    }
}
uint64_t qsa_cache_row_bytes(const QsaConfig& c) {return uint64_t(c.kv_heads)*c.width*(qsa_kv_fp8_enabled()?1:4);}
uint64_t qsa_cache_scale_bytes(const QsaConfig&) {return kQsaScaleSlots*4;}
namespace {
// F32 -> fp8_143 (e4m3: bias 7, max 240, subnormals in 2^-9 steps), round to nearest even; |x| <= 240 by construction.
uint8_t encode_e4m3(float x) {
    const uint8_t sign=std::signbit(x)?0x80:0;const float a=std::fabs(x);
    if(!(a>0)) return sign;
    if(a>=240.0f) return sign|0x77;
    int exponent=0;std::frexp(a,&exponent);exponent-=1; // a = m * 2^exponent, m in [1, 2)
    if(exponent<-6) {
        // Subnormal steps of 2^-9 (ties to even under the default rounding); 8 steps is the smallest normal code.
        return uint8_t(sign|std::min<int>(int(std::nearbyint(a*512.0f)),8));
    }
    int mantissa=int(std::nearbyint((std::ldexp(a,-exponent)-1.0f)*8.0f));
    if(mantissa==8) {mantissa=0;++exponent;}
    if(exponent>7) return sign|0x77;
    return uint8_t(sign|((exponent+7)<<3)|mantissa);
}
}
void qsa_cache_encode(std::span<const float> rows,uint64_t hk,uint64_t d,uint8_t* codes,float* scales) {
    const uint64_t count=rows.size()/d; // head rows
    std::fill(scales,scales+(count/hk)*kQsaScaleSlots,0.0f);
    for(uint64_t r=0;r<count;++r) {
        float peak=0;for(uint64_t i=0;i<d;++i) peak=std::max(peak,std::fabs(rows[r*d+i]));
        const float scale=peak*(1.0f/240.0f)+1e-30f;scales[(r/hk)*kQsaScaleSlots+r%hk]=scale;
        for(uint64_t i=0;i<d;++i) codes[r*d+i]=encode_e4m3(rows[r*d+i]/scale);
    }
}
void define_qsa_device_gather(GraphBuilder& g,const QsaConfig& c,uint32_t bucket,uint32_t rows,
                              TensorId key,TensorId value,TensorId mask,uint32_t mask_heads) {
    if(!rows || rows>kResidentWindowRows || !bucket || bucket>c.selection_width(c.capacity) || c.width%64 ||
       !c.capacity || c.capacity>262144 || (mask_heads!=1 && mask_heads!=c.kv_heads))
        throw std::invalid_argument("unsupported QSA device gather geometry");
    const auto ids=g.tensor({"kv.ids",DType::F32,{rows,bucket}});
    const abi::QsaGather a{int32_t(c.width),int32_t(c.kv_heads),int32_t(c.capacity),int32_t(bucket),int32_t(rows),int32_t(mask_heads)};
    if(qsa_kv_fp8_enabled()) {
        // STRATA_HPU_QSA_KV_FP8: gather the codes and scales (tpc/qsa_gather_fp8.c), dequantize here.
        if(c.width!=256) throw std::invalid_argument("FP8 QSA cache gather needs 256-wide heads");
        const uint64_t hk=c.kv_heads,cap=c.capacity,b=bucket,rh=uint64_t(rows)*hk,d=c.width;
        const auto k=g.tensor({"kv.cache_k",DType::F8,{cap,hk,d}}),v=g.tensor({"kv.cache_v",DType::F8,{cap,hk,d}});
        const auto ks=g.tensor({"kv.cache_k_scale",DType::F32,{cap,kQsaScaleSlots}}),vs=g.tensor({"kv.cache_v_scale",DType::F32,{cap,kQsaScaleSlots}});
        const auto k8=g.tensor({"kv.key8",DType::F8,{rh,b,d},false}),v8=g.tensor({"kv.value8",DType::F8,{rh,b,d},false});
        const auto kss=g.tensor({"kv.key_scale",DType::F32,{rh,b},false}),vss=g.tensor({"kv.value_scale",DType::F32,{rh,b},false});
        {const TensorId in[]={k,v,ks,vs,ids},out[]={k8,v8,kss,vss,mask};g.op("strata_qsa_gather_fp8",in,out,&a,sizeof(a));}
        const auto dequant=[&](TensorId codes,TensorId scale,TensorId into,const char* stem) {
            const auto values=g.tensor({std::string("kv.")+stem+"_values",DType::F32,{rh,b,d},false}),scale3=g.tensor({std::string("kv.")+stem+"_scale3",DType::F32,{rh,b,1},false});
            g.cast(codes,values);g.reshape(scale,scale3);
            const auto product=g.tensor({std::string("kv.")+stem+"_dequant",DType::F32,{rh,b,d},false});{const TensorId in[]={values,scale3},o[]={product};g.op("mult_fwd_f32",in,o);}g.reshape(product,into);
        };
        dequant(k8,kss,key,"key");dequant(v8,vss,value,"value");
        return;
    }
    const auto k=g.tensor({"kv.cache_k",DType::F32,{c.capacity,c.kv_heads,c.width}});
    const auto v=g.tensor({"kv.cache_v",DType::F32,{c.capacity,c.kv_heads,c.width}});
    const TensorId in[]={k,v,ids},out[]={key,value,mask};
    g.op("strata_qsa_gather_f32",in,out,&a,sizeof(a));
}
void define_qsa_attention(GraphBuilder& graph,const QsaConfig& c,uint32_t bucket,bool capture_half) {
    c.validate();
    // The device-select decode attends to whole selected blocks plus the current cell: a few cells past the width.
    if(!bucket || bucket>c.selection_width(c.capacity)+2*c.block+1) throw std::invalid_argument("QSA recipe bucket out of range");
    const uint64_t hq=c.query_heads,hk=c.kv_heads,d=c.width,b=bucket,group=hq/hk;
    const auto q=graph.tensor({"query",DType::F32,{hq,d}});
    const auto k=graph.tensor({"key",DType::F32,{hk,b,d}});
    const auto v=graph.tensor({"value",DType::F32,{hk,b,d}});
    const auto mask=graph.tensor({"mask",DType::F32,{b}});
    const auto scale=graph.tensor({"scale",DType::F32,{1}});
    const auto gate=graph.tensor({"q_gate",DType::F32,{hq,d}});
    const auto q_group=graph.tensor({"q_group",DType::F32,{hk,group,d},false});
    const auto scores=graph.tensor({"scores",DType::F32,{hk,group,b},false});
    const auto scaled=graph.tensor({"scaled",DType::F32,{hk,group,b},false});
    const auto masked=graph.tensor({"masked",DType::F32,{hk,group,b},false});
    const auto probability=graph.tensor({"probability",DType::F32,{hk,group,b},false});
    const auto context=graph.tensor({"context",DType::F32,{hk,group,d},false});
    const auto context_flat=graph.tensor({"context_flat",DType::F32,{hq,d},capture_half});
    const auto sig_gate=graph.tensor({"sig_gate",DType::F32,{hq,d},false});
    const auto output=graph.tensor({"out",DType::F32,{hq,d}});
    graph.reshape(q,q_group);
    graph.batch_gemm(q_group,k,scores,true);
    graph.multiply(scores,scale,scaled);
    graph.add(scaled,mask,masked);
    graph.softmax_last(masked,probability);
    graph.batch_gemm(probability,v,context);
    graph.reshape(context,context_flat);
    graph.sigmoid(gate,sig_gate);
    graph.multiply(context_flat,sig_gate,output);
}
namespace {
void project(GraphBuilder& graph,TensorId x,uint64_t hidden,uint64_t rows,const char* name,std::vector<uint64_t> shape,uint32_t tokens=1) {
    const std::string stem(name);
    const auto weight=graph.tensor({"w_"+stem,DType::BF16,{rows,hidden}});
    const auto matrix=graph.tensor({stem+"_matrix",DType::F32,{tokens,rows},false});
    const auto output=graph.tensor({stem,DType::F32,std::move(shape)});
    graph.gemm(x,weight,matrix,true);graph.reshape(matrix,output);
}
}
// STRATA_HPU_MERGED_GEMM: one GEMM of x [tokens, hidden] over w_qkvi = [w_q_gate; w_k; w_v; w_index_query;
// w_index_key], split into the five F32 projection matrices and reshaped to the usual outputs.
// internal: only this graph reads q_gate/k/v/index_query, so they are workspace tensors instead of each recipe's own
// persistent copy (402 MB of q_gate at 8192 rows, per recipe); index_key stays persistent for the stage.
static void project_merged(GraphBuilder& graph,TensorId x,uint64_t hidden,const QsaConfig& c,uint32_t tokens,bool window,bool internal=false) {
    const uint64_t hq=c.query_heads,hk=c.kv_heads,d=c.width,hi=c.index_heads,di=c.index_width,t=tokens;
    const uint64_t widths[5]={hq*2*d,hk*d,hk*d,hi*di,di};const char* stems[5]={"q_gate","k","v","index_query","index_key"};
    const std::vector<uint64_t> shapes[5]={window?std::vector<uint64_t>{t,hq,2*d}:std::vector<uint64_t>{hq,2*d},window?std::vector<uint64_t>{t,hk,d}:std::vector<uint64_t>{hk,d},
        window?std::vector<uint64_t>{t,hk,d}:std::vector<uint64_t>{hk,d},window?std::vector<uint64_t>{t,hi,di}:std::vector<uint64_t>{hi,di},
        window?std::vector<uint64_t>{t,di}:std::vector<uint64_t>{di}};
    uint64_t total=0;for(auto w:widths) total+=w;
    const auto all=graph.tensor({"qkvi_matrix",DType::F32,{t,total},false});
    if(fp8_dense_enabled()) {
        // STRATA_HPU_FP8_DENSE: FP8 weights with per-row scales; the BF16 activation widened for fp8_linear.
        const auto x32=graph.tensor({"x_f32",DType::F32,{t,hidden},false});graph.cast(x,x32);
        graph.fp8_linear(x32,graph.tensor({"w_qkvi",DType::F8,{total,hidden}}),graph.tensor({"w_qkvi_scale",DType::F32,{1,total}}),all,true);
    } else graph.gemm(x,graph.tensor({"w_qkvi",DType::BF16,{total,hidden}}),all,true);
    std::vector<TensorId> parts;
    for(int i=0;i<5;++i) parts.push_back(graph.tensor({std::string(stems[i])+"_matrix",DType::F32,{t,widths[i]},false}));
    {synSplitParams p{};p.axis=0;const TensorId in[]={all};graph.op("split",in,parts,&p,sizeof(p));}
    for(int i=0;i<5;++i) {const auto output=graph.tensor({stems[i],DType::F32,shapes[i],!internal || i==4});graph.reshape(parts[size_t(i)],output);}
}
void define_qsa_projections(GraphBuilder& graph,const QsaConfig& c,uint32_t hidden,bool merged) {
    c.validate();if(!hidden) throw std::invalid_argument("QSA hidden width is zero");
    const auto x=graph.tensor({"x",DType::BF16,{1,hidden}});
    const uint64_t hq=c.query_heads,hk=c.kv_heads,d=c.width,hi=c.index_heads,di=c.index_width;
    if(merged) {project_merged(graph,x,hidden,c,1,false);return;}
    project(graph,x,hidden,hq*2*d,"q_gate",{hq,2*d});
    project(graph,x,hidden,hk*d,"k",{hk,d});
    project(graph,x,hidden,hk*d,"v",{hk,d});
    project(graph,x,hidden,hi*di,"index_query",{hi,di});
    project(graph,x,hidden,di,"index_key",{di});
}
void define_qsa_output(GraphBuilder& graph,const QsaConfig& c,uint32_t hidden,bool merged) {
    c.validate();if(!hidden) throw std::invalid_argument("QSA hidden width is zero");
    const uint64_t inner=uint64_t(c.query_heads)*c.width;
    const auto x=graph.tensor({"attention",DType::F32,{c.query_heads,c.width}});
    if(merged && fp8_dense_enabled()) {
        const auto flat=graph.tensor({"flat_f32",DType::F32,{1,inner},false});graph.reshape(x,flat);
        graph.fp8_linear(flat,graph.tensor({"w_out",DType::F8,{hidden,inner}}),graph.tensor({"w_out_scale",DType::F32,{1,hidden}}),
                         graph.tensor({"out",DType::F32,{1,hidden}}),true);
        return;
    }
    const auto rounded=graph.tensor({"rounded",DType::BF16,{c.query_heads,c.width},false});
    const auto flat=graph.tensor({"flat",DType::BF16,{1,inner},false});
    const auto weight=graph.tensor({"w_out",DType::BF16,{hidden,inner}});
    const auto output=graph.tensor({"out",DType::F32,{1,hidden}});
    graph.cast(x,rounded);graph.reshape(rounded,flat);graph.gemm(flat,weight,output,true);
}
std::unique_ptr<Recipe> build_qsa_attention(const Device& d,const QsaConfig& c,uint32_t b,bool capture_half) {
    GraphBuilder graph(d);define_qsa_attention(graph,c,b,capture_half);return graph.compile("strata_qsa_attention_"+std::to_string(b)+(capture_half?"_half":""));
}
std::unique_ptr<Recipe> build_qsa_projections(const Device& d,const QsaConfig& c,uint32_t hidden) {
    GraphBuilder graph(d);define_qsa_projections(graph,c,hidden);return graph.compile("strata_qsa_projections");
}
std::unique_ptr<Recipe> build_qsa_output(const Device& d,const QsaConfig& c,uint32_t hidden) {
    GraphBuilder graph(d);define_qsa_output(graph,c,hidden);return graph.compile("strata_qsa_output");
}
void define_qsa_projections_window(GraphBuilder& graph,const QsaConfig& c,uint32_t hidden,uint32_t rows,bool merged,bool internal) {
    c.validate();if(!hidden || !rows || rows>kResidentWindowRows) throw std::invalid_argument("QSA projection window must have1..32 rows");
    const auto x=graph.tensor({"x",DType::BF16,{rows,hidden}});
    const uint64_t hq=c.query_heads,hk=c.kv_heads,d=c.width,hi=c.index_heads,di=c.index_width;
    if(merged) {project_merged(graph,x,hidden,c,rows,true,internal);return;}
    project(graph,x,hidden,hq*2*d,"q_gate",{rows,hq,2*d},rows);
    project(graph,x,hidden,hk*d,"k",{rows,hk,d},rows);
    project(graph,x,hidden,hk*d,"v",{rows,hk,d},rows);
    project(graph,x,hidden,hi*di,"index_query",{rows,hi,di},rows);
    project(graph,x,hidden,di,"index_key",{rows,di},rows);
}
void define_qsa_output_window(GraphBuilder& graph,const QsaConfig& c,uint32_t hidden,uint32_t rows,bool merged) {
    c.validate();if(!hidden || !rows || rows>kResidentWindowRows) throw std::invalid_argument("QSA output window must have1..32 rows");
    const uint64_t inner=uint64_t(c.query_heads)*c.width;
    const auto x=graph.tensor({"attention",DType::F32,{rows,c.query_heads,c.width}});
    if(merged && fp8_dense_enabled()) {
        const auto flat=graph.tensor({"flat_f32",DType::F32,{rows,inner},false});graph.reshape(x,flat);
        graph.fp8_linear(flat,graph.tensor({"w_out",DType::F8,{hidden,inner}}),graph.tensor({"w_out_scale",DType::F32,{1,hidden}}),
                         graph.tensor({"out",DType::F32,{rows,hidden}}),true);
        return;
    }
    const auto rounded=graph.tensor({"rounded",DType::BF16,{rows,c.query_heads,c.width},false});
    const auto flat=graph.tensor({"flat",DType::BF16,{rows,inner},false});
    const auto weight=graph.tensor({"w_out",DType::BF16,{hidden,inner}});
    const auto output=graph.tensor({"out",DType::F32,{rows,hidden}});
    graph.cast(x,rounded);graph.reshape(rounded,flat);graph.gemm(flat,weight,output,true);
}
void define_qsa_attention_window(GraphBuilder& graph,const QsaConfig& c,uint32_t bucket,uint32_t rows) {
    c.validate();if(!rows || rows>kResidentWindowRows || !bucket || bucket>c.selection_width(c.capacity)) throw std::invalid_argument("QSA attention window geometry mismatch");
    const uint64_t hq=c.query_heads,hk=c.kv_heads,d=c.width,b=bucket,group=hq/hk,batches=uint64_t(rows)*hk;
    const auto query=graph.tensor({"query",DType::F32,{rows,hq,d}});
    const auto key=graph.tensor({"key",DType::F32,{batches,b,d}});
    const auto value=graph.tensor({"value",DType::F32,{batches,b,d}});
    const auto mask=graph.tensor({"mask",DType::F32,{batches,1,b}});
    const auto scale=graph.tensor({"scale",DType::F32,{1}});
    const auto gate=graph.tensor({"q_gate",DType::F32,{rows,hq,d}});
    const auto grouped=graph.tensor({"query_grouped",DType::F32,{batches,group,d},false});
    const auto scores=graph.tensor({"scores",DType::F32,{batches,group,b},false});
    const auto scaled=graph.tensor({"scaled",DType::F32,{batches,group,b},false});
    const auto masked=graph.tensor({"masked",DType::F32,{batches,group,b},false});
    const auto probabilities=graph.tensor({"probabilities",DType::F32,{batches,group,b},false});
    const auto context=graph.tensor({"context",DType::F32,{batches,group,d},false});
    const auto flat=graph.tensor({"context_flat",DType::F32,{rows,hq,d},false});
    const auto sigmoid=graph.tensor({"sig_gate",DType::F32,{rows,hq,d},false});
    const auto output=graph.tensor({"out",DType::F32,{rows,hq,d}});
    graph.reshape(query,grouped);graph.batch_gemm(grouped,key,scores,true);graph.multiply(scores,scale,scaled);
    graph.add(scaled,mask,masked);graph.softmax_last(masked,probabilities);graph.batch_gemm(probabilities,value,context);
    graph.reshape(context,flat);graph.sigmoid(gate,sigmoid);graph.multiply(flat,sigmoid,output);
}
// Gather-free attention for a wide prefill window (STRATA_HPU_QSA_DENSE_WINDOW). The gathered version above batches
// over rows*kv_heads, so its key/value must be one dense copy per row -- rows * bucket * width floats, which is what
// caps the window at ~128 rows. Here the batch axis is the KV head and the rows are the GEMM's M, so every row shares
// one copy of the cache and the per-row selection is expressed by the mask. The dot products are the same ones over
// the same cells in the same order, so with an identity selection this is bit-identical; only the layout differs.
// [a,b,c,d] -> [b,a,c,d]: the same op the window path uses to move the head axis onto the batch axis.
static void transpose4(GraphBuilder& graph,TensorId x,TensorId y) {
    synTransposeParams p{};p.tensorDim=4;p.permutation[0]=TPD_Channel;p.permutation[1]=TPD_Width;
    p.permutation[2]=TPD_Depth;p.permutation[3]=TPD_Height;
    const TensorId in[]={x},out[]={y};graph.op("transpose",in,out,&p,sizeof(p));
}
// [a,b,c] -> [b,a,c]: for the KV cache, whose stored layout is {cells, kv_heads, width}.
static void transpose3(GraphBuilder& graph,TensorId x,TensorId y) {
    synTransposeParams p{};p.tensorDim=3;p.permutation[0]=TPD_Channel;
    p.permutation[1]=TPD_Height;p.permutation[2]=TPD_Width;
    const TensorId in[]={x},out[]={y};graph.op("transpose",in,out,&p,sizeof(p));
}
void define_qsa_attention_dense_window(GraphBuilder& graph,const QsaConfig& c,uint32_t cells,uint32_t rows) {
    c.validate();if(!rows || rows>kResidentWindowRows || !cells || cells>c.capacity) throw std::invalid_argument("QSA dense attention window geometry mismatch");
    const uint64_t hq=c.query_heads,hk=c.kv_heads,d=c.width,group=hq/hk,queries=uint64_t(rows)*group;
    const auto query=graph.tensor({"query",DType::F32,{rows,hq,d}});
    const auto key=graph.tensor({"key",DType::F32,{hk,cells,d}});      // the shared cache, one copy for every row
    const auto value=graph.tensor({"value",DType::F32,{hk,cells,d}});
    const auto mask=graph.tensor({"mask",DType::F32,{rows,1,cells}});  // -inf where the row may not attend
    const auto scale=graph.tensor({"scale",DType::F32,{1}});
    const auto gate=graph.tensor({"q_gate",DType::F32,{rows,hq,d}});
    const auto q4=graph.tensor({"query4",DType::F32,{rows,hk,group,d},false});
    const auto qt=graph.tensor({"query_t",DType::F32,{hk,rows,group,d},false});
    const auto grouped=graph.tensor({"query_grouped",DType::F32,{hk,queries,d},false});
    const auto scores=graph.tensor({"scores",DType::F32,{hk,queries,cells},false});
    const auto scaled=graph.tensor({"scaled",DType::F32,{hk,queries,cells},false});
    const auto scaled4=graph.tensor({"scaled4",DType::F32,{hk,rows,group,cells},false});
    const auto masked4=graph.tensor({"masked4",DType::F32,{hk,rows,group,cells},false});
    const auto masked=graph.tensor({"masked",DType::F32,{hk,queries,cells},false});
    const auto probabilities=graph.tensor({"probabilities",DType::F32,{hk,queries,cells},false});
    const auto context=graph.tensor({"context",DType::F32,{hk,queries,d},false});
    const auto c4=graph.tensor({"context4",DType::F32,{hk,rows,group,d},false});
    const auto ct=graph.tensor({"context_t",DType::F32,{rows,hk,group,d},false});
    const auto flat=graph.tensor({"context_flat",DType::F32,{rows,hq,d},false});
    const auto sigmoid=graph.tensor({"sig_gate",DType::F32,{rows,hq,d},false});
    const auto output=graph.tensor({"out",DType::F32,{rows,hq,d}});
    graph.reshape(query,q4);transpose4(graph,q4,qt);graph.reshape(qt,grouped);
    graph.batch_gemm(grouped,key,scores,true);graph.multiply(scores,scale,scaled);
    graph.reshape(scaled,scaled4);graph.add(scaled4,mask,masked4);
    graph.reshape(masked4,masked);graph.softmax_last(masked,probabilities);
    graph.batch_gemm(probabilities,value,context);
    graph.reshape(context,c4);transpose4(graph,c4,ct);graph.reshape(ct,flat);
    graph.sigmoid(gate,sigmoid);graph.multiply(flat,sigmoid,output);
}
std::unique_ptr<Recipe> build_qsa_projections_window(const Device& d,const QsaConfig& c,uint32_t hidden,uint32_t rows) {
    GraphBuilder graph(d);define_qsa_projections_window(graph,c,hidden,rows);return graph.compile("strata_qsa_projection_window_"+std::to_string(rows));
}
std::unique_ptr<Recipe> build_qsa_attention_window(const Device& d,const QsaConfig& c,uint32_t bucket,uint32_t rows) {
    GraphBuilder graph(d);define_qsa_attention_window(graph,c,bucket,rows);return graph.compile("strata_qsa_attention_window_"+std::to_string(rows)+"_"+std::to_string(bucket));
}
std::unique_ptr<Recipe> build_qsa_attention_dense_window(const Device& d,const QsaConfig& c,uint32_t cells,uint32_t rows) {
    GraphBuilder graph(d);define_qsa_attention_dense_window(graph,c,cells,rows);
    return graph.compile("strata_qsa_attention_dense_"+std::to_string(rows)+"_"+std::to_string(cells));
}
// Same attention, reading the cache in its stored layout {cells, kv_heads, width} instead of the transposed
// {kv_heads, cells, width}. The decode path already binds the cache this way and transposes inside the graph; that
// transpose costs kv_heads*cells*width floats (4 MiB at 2048 cells) instead of one gathered copy per row.
void define_qsa_attention_dense_cache(GraphBuilder& graph,const QsaConfig& c,uint32_t cells,uint32_t rows) {
    c.validate();if(!rows || rows>kResidentWindowRows || !cells || cells>c.capacity) throw std::invalid_argument("QSA dense cache attention geometry mismatch");
    const uint64_t hq=c.query_heads,hk=c.kv_heads,d=c.width,group=hq/hk,queries=uint64_t(rows)*group;
    const auto query=graph.tensor({"query",DType::F32,{rows,hq,d}});
    const auto key_cache=qsa_cache_read(graph,"kv.cache_k",cells,hk,d),value_cache=qsa_cache_read(graph,"kv.cache_v",cells,hk,d);
    const auto mask=graph.tensor({"mask",DType::F32,{rows,1,cells}});
    const auto scale=graph.tensor({"scale",DType::F32,{1}});
    const auto gate=graph.tensor({"q_gate",DType::F32,{rows,hq,d}});
    const auto key=graph.tensor({"key_heads",DType::F32,{hk,cells,d},false});
    const auto value=graph.tensor({"value_heads",DType::F32,{hk,cells,d},false});
    transpose3(graph,key_cache,key);transpose3(graph,value_cache,value);
    const auto q4=graph.tensor({"query4",DType::F32,{rows,hk,group,d},false});
    const auto qt=graph.tensor({"query_t",DType::F32,{hk,rows,group,d},false});
    const auto grouped=graph.tensor({"query_grouped",DType::F32,{hk,queries,d},false});
    const auto scores=graph.tensor({"scores",DType::F32,{hk,queries,cells},false});
    const auto scaled=graph.tensor({"scaled",DType::F32,{hk,queries,cells},false});
    const auto scaled4=graph.tensor({"scaled4",DType::F32,{hk,rows,group,cells},false});
    const auto masked4=graph.tensor({"masked4",DType::F32,{hk,rows,group,cells},false});
    const auto masked=graph.tensor({"masked",DType::F32,{hk,queries,cells},false});
    const auto probabilities=graph.tensor({"probabilities",DType::F32,{hk,queries,cells},false});
    const auto context=graph.tensor({"context",DType::F32,{hk,queries,d},false});
    const auto c4=graph.tensor({"context4",DType::F32,{hk,rows,group,d},false});
    const auto ct=graph.tensor({"context_t",DType::F32,{rows,hk,group,d},false});
    const auto flat=graph.tensor({"context_flat",DType::F32,{rows,hq,d},false});
    const auto sigmoid=graph.tensor({"sig_gate",DType::F32,{rows,hq,d},false});
    const auto output=graph.tensor({"out",DType::F32,{rows,hq,d}});
    graph.reshape(query,q4);transpose4(graph,q4,qt);graph.reshape(qt,grouped);
    graph.batch_gemm(grouped,key,scores,true);graph.multiply(scores,scale,scaled);
    graph.reshape(scaled,scaled4);graph.add(scaled4,mask,masked4);
    graph.reshape(masked4,masked);graph.softmax_last(masked,probabilities);
    graph.batch_gemm(probabilities,value,context);
    graph.reshape(context,c4);transpose4(graph,c4,ct);graph.reshape(ct,flat);
    graph.sigmoid(gate,sigmoid);graph.multiply(flat,sigmoid,output);
}
std::unique_ptr<Recipe> build_qsa_attention_dense_cache(const Device& d,const QsaConfig& c,uint32_t cells,uint32_t rows) {
    GraphBuilder graph(d);define_qsa_attention_dense_cache(graph,c,cells,rows);
    return graph.compile("strata_qsa_attention_dcache_"+std::to_string(rows)+"_"+std::to_string(cells));
}
std::unique_ptr<Recipe> build_qsa_output_window(const Device& d,const QsaConfig& c,uint32_t hidden,uint32_t rows) {
    GraphBuilder graph(d);define_qsa_output_window(graph,c,hidden,rows);return graph.compile("strata_qsa_output_window_"+std::to_string(rows));
}
}
