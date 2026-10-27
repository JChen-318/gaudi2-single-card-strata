#include "strata/hpu/gdn.hpp"
#include "strata/hpu/gdn_abi.hpp"
#include "strata/hpu/performance_modes.hpp"
#include <array>
#include <cmath>
#include <perf_lib_layer_params.h>

namespace strata::hpu {
void define_gdn_step(GraphBuilder& graph, const core::ModelGeometry& geometry) {
    if (geometry.ssm_state_size <= 0 || geometry.ssm_k_heads <= 0 || geometry.ssm_v_heads <= 0)
        throw std::invalid_argument("invalid GDN model geometry");
    const auto s = static_cast<uint64_t>(geometry.ssm_state_size);
    const auto hk = static_cast<uint64_t>(geometry.ssm_k_heads);
    const auto hv = static_cast<uint64_t>(geometry.ssm_v_heads);
    const auto state = graph.tensor({"state", DType::F32, {s, hv, s}});
    const auto q = graph.tensor({"q", DType::F32, {hk, s}});
    const auto k = graph.tensor({"k", DType::F32, {hk, s}});
    const auto v = graph.tensor({"v", DType::F32, {hv, s}});
    const auto gate = graph.tensor({"gate", DType::F32, {hv}});
    const auto beta = graph.tensor({"beta", DType::F32, {hv}});
    const auto next = graph.tensor({"state_out", DType::F32, {s, hv, s}});
    const auto output = graph.tensor({"out", DType::F32, {hv, s}});
    graph.gdn_step(state, q, k, v, gate, beta, next, output);
}
std::unique_ptr<Recipe> build_gdn_step(const Device& device, const core::ModelGeometry& geometry) {
    GraphBuilder graph(device);
    define_gdn_step(graph, geometry);
    return graph.compile("strata_gdn_step");
}
namespace {
// The window gated norm. 128-wide heads run four unrolled rows per member (same bits, about twice as fast).
const char* gdn_norm_rows_bf16_guid(uint64_t s) {
    if(s!=128) return "strata_gdn_norm_rows_bf16_f32";
    return gdn_norm_fast_enabled()?"strata_gdn_norm_fast_bf16_f32":"strata_gdn_norm_rows8_bf16_f32";
}
struct Dimensions { uint64_t s,hk,hv,c,d,h; float eps; };
Dimensions dimensions(const core::ModelGeometry& g) {
    if (g.ssm_state_size <= 0 || g.ssm_state_size > 256 || g.ssm_state_size % 64 ||
        g.ssm_k_heads <= 0 || g.ssm_v_heads <= 0 || g.ssm_v_heads > 1024 ||
        g.ssm_v_heads % g.ssm_k_heads || g.ssm_d_conv < 2 || g.ssm_d_conv > 16 || g.n_embd <= 0 ||
        !std::isfinite(g.rms_eps) || g.rms_eps <= 0 || float(g.rms_eps) <= 0)
        throw std::invalid_argument("unsupported GDN geometry/epsilon");
    const uint64_t s=g.ssm_state_size,hk=g.ssm_k_heads,hv=g.ssm_v_heads,c=s*(2*hk+hv);
    if (g.ssm_conv_channels != int64_t(c) || g.ssm_value_dim != int64_t(s*hv))
        throw std::invalid_argument("GDN derived channel geometry mismatch");
    return {s,hk,hv,c,uint64_t(g.ssm_d_conv),uint64_t(g.n_embd),float(g.rms_eps)};
}
struct BodyInput { TensorId state,conv,qkv,alpha,raw_beta,z,weights,norm,dt,a; };
struct BodyOutput {TensorId y,state,conv;};
BodyOutput append_body(GraphBuilder& graph, const Dimensions& g, const BodyInput& in, bool debug, const char* output_name,
                       const std::string& prefix={},bool final=true,bool keep_state=false) {
    const auto activated = graph.tensor({prefix+"activated",DType::F32,{g.c},debug});
    const auto q = graph.tensor({prefix+"q",DType::F32,{g.hk,g.s},debug});
    const auto k = graph.tensor({prefix+"k",DType::F32,{g.hk,g.s},debug});
    const auto v = graph.tensor({prefix+"v",DType::F32,{g.hv,g.s},debug});
    const auto gate = graph.tensor({prefix+"gate",DType::F32,{g.hv},debug});
    const auto beta = graph.tensor({prefix+"beta",DType::F32,{g.hv},debug});
    const auto recurrent = graph.tensor({prefix+"recurrent",DType::F32,{g.hv,g.s},debug});
    // keep_state: this row's states stay persistent under their row names (every row of a verify window).
    const auto next = graph.tensor({final?"state_out":prefix+"state_out",DType::F32,{g.s,g.hv,g.s},final || keep_state});
    const auto conv_next = graph.tensor({final?"conv_state_out":prefix+"conv_state_out",DType::F32,{g.d-1,g.c},final || keep_state});
    const auto output = graph.tensor({output_name,DType::F32,{g.hv,g.s},debug || std::string(output_name)=="out"});
    graph.gdn_conv(in.conv,in.qkv,in.weights,conv_next,activated);
    graph.gdn_prepare(activated,in.alpha,in.dt,in.a,in.raw_beta,q,k,v,gate,beta,g.eps);
    graph.gdn_step(in.state,q,k,v,gate,beta,next,recurrent);
    graph.gdn_norm(recurrent,in.z,in.norm,output,g.eps);
    return {output,next,conv_next};
}
BodyInput body_inputs(GraphBuilder& graph, const Dimensions& g) {
    return {graph.tensor({"state",DType::F32,{g.s,g.hv,g.s}}),
            graph.tensor({"conv_state",DType::F32,{g.d-1,g.c}}),
            graph.tensor({"qkv",DType::F32,{g.c}}),
            graph.tensor({"alpha",DType::F32,{g.hv}}),
            graph.tensor({"raw_beta",DType::F32,{g.hv}}),
            graph.tensor({"z",DType::F32,{g.hv,g.s}}),
            graph.tensor({"conv_weights",DType::F32,{g.d,g.c}}),
            graph.tensor({"norm",DType::F32,{g.s}}),
            graph.tensor({"dt",DType::F32,{g.hv}}),
            graph.tensor({"ssm_a",DType::F32,{g.hv}})};
}
TensorId project(GraphBuilder& graph, TensorId input, uint64_t h, uint64_t n,
                 const char* weight_name, const char* output_name, std::vector<uint64_t> shape) {
    const std::string stem(output_name);
    const auto weight = graph.tensor({weight_name,DType::BF16,{n,h}});
    const auto matrix = graph.tensor({stem+"_matrix",DType::BF16,{1,n},false});
    const auto reshaped = graph.tensor({stem+"_bf16",DType::BF16,shape,false});
    const auto output = graph.tensor({stem,DType::F32,std::move(shape),false});
    graph.gemm(input,weight,matrix,true);
    graph.reshape(matrix,reshaped);
    graph.cast(reshaped,output);
    return output;
}
}
namespace {
// STRATA_HPU_MERGED_GEMM: one GEMM of x [rows, h] over w_qkvzab = [w_qkv; w_gate; w_alpha; w_beta] (rows of the
// fused buffer in that order), split into the four BF16 projections [rows, width] qkv, z, alpha, beta.
std::array<TensorId,4> project_merged(GraphBuilder& graph,TensorId x,uint64_t rows,const Dimensions& g) {
    const uint64_t widths[4]={g.c,g.hv*g.s,g.hv,g.hv};const char* stems[4]={"qkv","z","alpha","raw_beta"};
    const uint64_t total=widths[0]+widths[1]+widths[2]+widths[3];
    std::array<TensorId,4> parts{};
    for(int i=0;i<4;++i) parts[i]=graph.tensor({std::string(stems[i])+"_matrix",DType::BF16,{rows,widths[i]},false});
    if(fp8_dense_enabled()) {
        // STRATA_HPU_FP8_DENSE: FP8 weights, the BF16 activation widened for fp8_linear, the parts rounded to BF16
        // like the BF16 GEMM's output.
        const auto x32=graph.tensor({"x_f32",DType::F32,{rows,g.h},false});graph.cast(x,x32);
        const auto all=graph.tensor({"qkvzab_f32",DType::F32,{rows,total},false}),rounded=graph.tensor({"qkvzab_matrix",DType::BF16,{rows,total},false});
        graph.fp8_linear(x32,graph.tensor({"w_qkvzab",DType::F8,{total,g.h}}),graph.tensor({"w_qkvzab_scale",DType::F32,{1,total}}),all,true);
        graph.cast(all,rounded);
        synSplitParams p{};p.axis=0;const TensorId in[]={rounded};graph.op("split",in,parts,&p,sizeof(p));
        return parts;
    }
    const auto weight=graph.tensor({"w_qkvzab",DType::BF16,{total,g.h}});
    const auto all=graph.tensor({"qkvzab_matrix",DType::BF16,{rows,total},false});
    graph.gemm(x,weight,all,true);
    synSplitParams p{};p.axis=0;const TensorId in[]={all};graph.op("split",in,parts,&p,sizeof(p));
    return parts;
}
// STRATA_HPU_GDN_FUSED_PARTS: the same merged projection as one F32 GEMM. `all` is [rows, total] row-major, so each
// part is a column block: qkv at 0, z at c, alpha at c+hv*s, raw_beta at c+hv*s+hv. The row kernels read their block.
TensorId project_merged_f32(GraphBuilder& graph,TensorId x,uint64_t rows,const Dimensions& g) {
    const uint64_t total=g.c+g.hv*g.s+2*g.hv;
    const auto all=graph.tensor({"qkvzab_rows_f32",DType::F32,{rows,total},false});
    graph.gemm(x,graph.tensor({"w_qkvzab",DType::BF16,{total,g.h}}),all,true);
    return all;
}
// The output projection "out" BF16 [rows, h] of y [rows, hv*s] BF16; FP8 weights under STRATA_HPU_FP8_DENSE (merged).
TensorId project_out(GraphBuilder& graph,const Dimensions& g,TensorId y_matrix,uint64_t rows,bool merged) {
    const auto output=graph.tensor({"out",DType::BF16,{rows,g.h}});
    if(!(merged && fp8_dense_enabled())) {graph.gemm(y_matrix,graph.tensor({"w_out",DType::BF16,{g.h,g.hv*g.s}}),output,true);return output;}
    const auto y32=graph.tensor({"y_matrix_f32",DType::F32,{rows,g.hv*g.s},false});graph.cast(y_matrix,y32);
    const auto out32=graph.tensor({"out_f32",DType::F32,{rows,g.h},false});
    graph.fp8_linear(y32,graph.tensor({"w_out",DType::F8,{g.h,g.hv*g.s}}),graph.tensor({"w_out_scale",DType::F32,{1,g.h}}),out32,true);
    graph.cast(out32,output);return output;
}
TensorId project_from(GraphBuilder& graph,TensorId matrix,const char* output_name,std::vector<uint64_t> shape) {
    const std::string stem(output_name);
    const auto reshaped=graph.tensor({stem+"_bf16",DType::BF16,shape,false});
    const auto output=graph.tensor({stem,DType::F32,std::move(shape),false});
    graph.reshape(matrix,reshaped);graph.cast(reshaped,output);return output;
}
}
void define_gdn_body(GraphBuilder& graph, const core::ModelGeometry& geometry, bool debug) {
    const auto g = dimensions(geometry);
    const auto inputs = body_inputs(graph,g);
    (void)append_body(graph,g,inputs,debug,"out");
}
std::unique_ptr<Recipe> build_gdn_body(const Device& device, const core::ModelGeometry& geometry, bool debug) {
    GraphBuilder graph(device);
    define_gdn_body(graph,geometry,debug);
    return graph.compile("strata_gdn_body");
}
void define_gdn_block(GraphBuilder& graph, const core::ModelGeometry& geometry, bool merged) {
    const auto g = dimensions(geometry);
    const auto x = graph.tensor({"x",DType::BF16,{1,g.h}});
    const auto state = graph.tensor({"state",DType::F32,{g.s,g.hv,g.s}});
    const auto conv = graph.tensor({"conv_state",DType::F32,{g.d-1,g.c}});
    const auto weights = graph.tensor({"conv_weights",DType::F32,{g.d,g.c}});
    const auto norm = graph.tensor({"norm",DType::F32,{g.s}});
    const auto dt = graph.tensor({"dt",DType::F32,{g.hv}});
    const auto a = graph.tensor({"ssm_a",DType::F32,{g.hv}});
    TensorId qkv,alpha,raw_beta,z;
    if(merged) {
        const auto parts=project_merged(graph,x,1,g);
        qkv=project_from(graph,parts[0],"qkv",{g.c});z=project_from(graph,parts[1],"z",{g.hv,g.s});
        alpha=project_from(graph,parts[2],"alpha",{g.hv});raw_beta=project_from(graph,parts[3],"raw_beta",{g.hv});
    } else {
        qkv = project(graph,x,g.h,g.c,"w_qkv","qkv",{g.c});
        alpha = project(graph,x,g.h,g.hv,"w_alpha","alpha",{g.hv});
        raw_beta = project(graph,x,g.h,g.hv,"w_beta","raw_beta",{g.hv});
        z = project(graph,x,g.h,g.hv*g.s,"w_gate","z",{g.hv,g.s});
    }
    const auto y = append_body(graph,g,{state,conv,qkv,alpha,raw_beta,z,weights,norm,dt,a},false,"gdn_y").y;
    const auto y_bf16 = graph.tensor({"y_bf16",DType::BF16,{g.hv,g.s},false});
    const auto y_matrix = graph.tensor({"y_matrix",DType::BF16,{1,g.hv*g.s},false});
    graph.cast(y,y_bf16);
    graph.reshape(y_bf16,y_matrix);
    (void)project_out(graph,g,y_matrix,1,merged);
}
std::unique_ptr<Recipe> build_gdn_block(const Device& device, const core::ModelGeometry& geometry) {
    GraphBuilder graph(device);
    define_gdn_block(graph,geometry);
    return graph.compile("strata_gdn_bf16_block");
}
namespace {
// A verify window's GDN body with conv, prepare and norm as one launch each over all rows (the recurrent steps stay
// row by row): every row's values are those of the per-row bodies of define_gdn_window. Row states persistent.
void define_gdn_window_rows(GraphBuilder& graph,const Dimensions& g,TensorId x,TensorId state,TensorId conv,TensorId weights,
                            TensorId norm,TensorId dt,TensorId a,uint32_t rows,bool merged) {
    const uint64_t r=rows;
    std::array<TensorId,4> parts{};
    TensorId merged_f32{};
    const bool fused=merged && !fp8_dense_enabled() && gdn_fused_parts_enabled();
    if(fused) merged_f32=project_merged_f32(graph,x,rows,g);
    else if(merged) parts=project_merged(graph,x,rows,g);
    else {
        const char* names[4]={"w_qkv","w_gate","w_alpha","w_beta"};const char* stems[4]={"qkv","z","alpha","raw_beta"};
        const uint64_t widths[4]={g.c,g.hv*g.s,g.hv,g.hv};
        for(int i=0;i<4;++i) {
            const auto weight=graph.tensor({names[i],DType::BF16,{widths[i],g.h}});
            parts[size_t(i)]=graph.tensor({std::string(stems[i])+"_matrix",DType::BF16,{r,widths[i]},false});graph.gemm(x,weight,parts[size_t(i)],true);
        }
    }
    const auto widen=[&](TensorId matrix,const char* stem,std::vector<uint64_t> shape) {
        const auto shaped=graph.tensor({std::string(stem)+"_rows_bf16",DType::BF16,shape,false});graph.reshape(matrix,shaped);
        const auto value=graph.tensor({std::string(stem)+"_rows",DType::F32,std::move(shape),false});graph.cast(shaped,value);return value;
    };
    TensorId qkv,z,alpha,raw_beta;int32_t alpha_col=0,beta_col=0,z_col=0;
    if(fused) {
        qkv=z=alpha=raw_beta=merged_f32;
        z_col=int32_t(g.c);alpha_col=int32_t(g.c+g.hv*g.s);beta_col=int32_t(g.c+g.hv*g.s+g.hv);
    } else {
        qkv=widen(parts[0],"qkv",{r,g.c});
        z=widen(parts[1],"z",{r,g.hv*g.s});
        alpha=widen(parts[2],"alpha",{r,g.hv});
        raw_beta=widen(parts[3],"raw_beta",{r,g.hv});
    }
    // Conv: every row's state (the first `rows` outputs) and activation.
    std::vector<TensorId> conv_outs;
    for(uint32_t row=0;row<(rows>4?8u:4u);++row) {
        const bool real=row<rows;
        conv_outs.push_back(graph.tensor({real?"row_"+std::to_string(row)+"_conv_state_out":"rows_unused_conv_"+std::to_string(row),DType::F32,{g.d-1,g.c},real}));
    }
    const auto activated=graph.tensor({"rows_activated",DType::F32,{r,g.c},false});conv_outs.push_back(activated);
    {abi::GdnConvRows args{int32_t(g.c),int32_t(g.d),int32_t(rows)};const TensorId in[]={conv,qkv,weights};
     graph.op(rows>4?"strata_gdn_conv_rows8_f32":"strata_gdn_conv_rows_f32",in,conv_outs,&args,sizeof(args));}
    const auto q=graph.tensor({"rows_q",DType::F32,{r,g.hk,g.s},false}),k=graph.tensor({"rows_k",DType::F32,{r,g.hk,g.s},false});
    const auto v=graph.tensor({"rows_v",DType::F32,{r,g.hv,g.s},false});
    const auto gate=graph.tensor({"rows_gate",DType::F32,{r,g.hv},false}),beta=graph.tensor({"rows_beta",DType::F32,{r,g.hv},false});
    {abi::GdnPrepareRows args{int32_t(g.s),int32_t(g.hk),int32_t(g.hv),g.eps,int32_t(rows),alpha_col,beta_col};
     const TensorId in[]={activated,alpha,dt,a,raw_beta},out[]={q,k,v,gate,beta};graph.op("strata_gdn_prepare_rows_f32",in,out,&args,sizeof(args));}
    // Row slices for the recurrent steps.
    const auto slices=[&](TensorId all,const char* stem,std::vector<uint64_t> row_shape) {
        std::vector<TensorId> pieces,flat;std::vector<uint64_t> one{1};one.insert(one.end(),row_shape.begin(),row_shape.end());
        for(uint32_t row=0;row<rows;++row) {
            const auto prefix="row_"+std::to_string(row)+"_"+stem;
            pieces.push_back(graph.tensor({prefix+"_slice",DType::F32,one,false}));flat.push_back(graph.tensor({prefix,DType::F32,row_shape,false}));
        }
        synSplitParams sp{};sp.axis=unsigned(row_shape.size());const TensorId in[]={all};graph.op("split",in,pieces,&sp,sizeof(sp));
        for(uint32_t row=0;row<rows;++row) graph.reshape(pieces[row],flat[row]);
        return flat;
    };
    const auto recurrent_all=graph.tensor({"rows_recurrent",DType::F32,{r,g.hv,g.s},false});
    if(gdn_step_rows_enabled() && rows<=4) {
        // STRATA_HPU_GDN_STEP_ROWS: one node for every row's step; rows past the window write nowhere.
        std::vector<TensorId> outs;
        for(uint32_t row=0;row<4;++row)
            outs.push_back(row<rows?graph.tensor({"row_"+std::to_string(row)+"_state_out",DType::F32,{g.s,g.hv,g.s}})
                                   :graph.tensor({"rows_unused_state_"+std::to_string(row),DType::F32,{g.s,g.hv,g.s},false}));
        outs.push_back(recurrent_all);
        abi::GdnStepRows args{int32_t(g.s),int32_t(g.hk),int32_t(rows)};const TensorId in[]={state,q,k,v,gate,beta};
        graph.op("strata_gdn_step_rows4_f32",in,outs,&args,sizeof(args));
    } else {
        const auto qs=slices(q,"q",{g.hk,g.s}),ks=slices(k,"k",{g.hk,g.s}),vs=slices(v,"v",{g.hv,g.s});
        const auto gates=slices(gate,"gate",{g.hv}),betas=slices(beta,"beta",{g.hv});
        std::vector<TensorId> recurrent;
        for(uint32_t row=0;row<rows;++row) {
            const auto prefix="row_"+std::to_string(row)+"_";
            const auto next=graph.tensor({prefix+"state_out",DType::F32,{g.s,g.hv,g.s}});
            const auto out=graph.tensor({prefix+"recurrent",DType::F32,{g.hv,g.s},false});
            graph.gdn_step(state,qs[row],ks[row],vs[row],gates[row],betas[row],next,out);state=next;
            const auto out3=graph.tensor({prefix+"recurrent3",DType::F32,{1,g.hv,g.s},false});graph.reshape(out,out3);recurrent.push_back(out3);
        }
        graph.concat(recurrent,recurrent_all,0);
    }
    const auto y_bf16=graph.tensor({"rows_y_bf16",DType::BF16,{r,g.hv,g.s},false});
    abi::GdnNormRows args{int32_t(g.s),int32_t(g.hv),g.eps,int32_t(rows),z_col};
    if(hc_fused_enabled()) {const TensorId in[]={recurrent_all,z,norm},out[]={y_bf16};graph.op(gdn_norm_rows_bf16_guid(g.s),in,out,&args,sizeof(args));}
    else {
        const auto y=graph.tensor({"rows_y",DType::F32,{r,g.hv,g.s},false});
        const TensorId in[]={recurrent_all,z,norm},out[]={y};graph.op("strata_gdn_norm_rows_f32",in,out,&args,sizeof(args));
        graph.cast(y,y_bf16);
    }
    const auto y_matrix=graph.tensor({"y_matrix",DType::BF16,{r,g.hv*g.s},false});graph.reshape(y_bf16,y_matrix);
    (void)project_out(graph,g,y_matrix,r,merged);
}
}
void define_gdn_window(GraphBuilder& graph,const core::ModelGeometry& geometry,uint32_t rows,bool row_states,bool merged) {
    const auto g=dimensions(geometry);
    if(!rows || rows>(row_states?kResidentWindowRows:kGdnWindowRows))
        throw std::invalid_argument("GDN graph window exceeds its causal/resident row limit");
    const auto x=graph.tensor({"x",DType::BF16,{rows,g.h}});
    auto state=graph.tensor({"state",DType::F32,{g.s,g.hv,g.s}});
    auto conv=graph.tensor({"conv_state",DType::F32,{g.d-1,g.c}});
    const auto weights=graph.tensor({"conv_weights",DType::F32,{g.d,g.c}});
    const auto norm=graph.tensor({"norm",DType::F32,{g.s}});
    const auto dt=graph.tensor({"dt",DType::F32,{g.hv}});
    const auto a=graph.tensor({"ssm_a",DType::F32,{g.hv}});
    if(row_states && rows>=2 && rows<=kGdnWindowRows && (rows<=4 || gdn_wide_rows_enabled()) && g.d>=2 && g.d<=4 && simd_gdn_enabled() && gdn_small_rows_enabled()) {
        define_gdn_window_rows(graph,g,x,state,conv,weights,norm,dt,a,rows,merged);return;
    }
    const auto parts=merged?project_merged(graph,x,rows,g):std::array<TensorId,4>{};
    auto projection=[&](const char* name,const char* stem,uint64_t columns,std::vector<uint64_t> row_shape,int part) {
        TensorId matrix;
        if(merged) matrix=parts[size_t(part)];
        else {
            const auto weight=graph.tensor({name,DType::BF16,{columns,g.h}});
            matrix=graph.tensor({std::string(stem)+"_matrix",DType::BF16,{rows,columns},false});
            graph.gemm(x,weight,matrix,true);
        }
        std::vector<TensorId> pieces,values;
        for(uint32_t row=0;row<rows;++row) {
            const auto prefix=std::string(stem)+"_"+std::to_string(row);
            const auto piece=graph.tensor({prefix+"_piece",DType::BF16,{1,columns},false});pieces.push_back(piece);
            const auto flat=graph.tensor({prefix+"_bf16",DType::BF16,row_shape,false});
            const auto value=graph.tensor({prefix,DType::F32,row_shape,false});values.push_back(value);
            graph.reshape(piece,flat);graph.cast(flat,value);
        }
        graph.split_rows(matrix,pieces);return values;
    };
    const auto qkv=projection("w_qkv","qkv",g.c,{g.c},0);
    const auto alpha=projection("w_alpha","alpha",g.hv,{g.hv},2);
    const auto beta=projection("w_beta","raw_beta",g.hv,{g.hv},3);
    const auto z=projection("w_gate","z",g.hv*g.s,{g.hv,g.s},1);
    std::vector<TensorId> results;
    for(uint32_t row=0;row<rows;++row) {
        const auto prefix="row_"+std::to_string(row)+"_",name=prefix+"y";
        const auto body=append_body(graph,g,{state,conv,qkv[row],alpha[row],beta[row],z[row],weights,norm,dt,a},false,name.c_str(),prefix,
                                    !row_states && row+1==rows,row_states);
        state=body.state;conv=body.conv;
        const auto bf16=graph.tensor({prefix+"y_bf16",DType::BF16,{g.hv,g.s},false});
        const auto matrix=graph.tensor({prefix+"y_matrix",DType::BF16,{1,g.hv*g.s},false});
        graph.cast(body.y,bf16);graph.reshape(bf16,matrix);results.push_back(matrix);
    }
    const auto y=graph.tensor({"y_matrix",DType::BF16,{rows,g.hv*g.s},false});graph.concatenate_rows(results,y);
    (void)project_out(graph,g,y,rows,merged);
}
bool gdn_prefill_chunked(uint32_t rows) {return gdn_chunked_enabled() && rows>=kGdnChunk && rows%kGdnChunk==0;}
void define_gdn_prefill_final(GraphBuilder& graph,const core::ModelGeometry& geometry,uint32_t rows,bool merged) {
    const auto g=dimensions(geometry);
    if(!rows || rows>kResidentWindowRows || g.d>4) throw std::invalid_argument("final-state GDN prefill geometry/rows");
    const uint64_t r=rows;
    const auto T=[&](const std::string& n,std::vector<uint64_t> shape,bool persistent=false) {return graph.tensor({n,DType::F32,std::move(shape),persistent});};
    const auto x=graph.tensor({"x",DType::BF16,{r,g.h}});
    const auto state=T("state",{g.s,g.hv,g.s},true),conv=T("conv_state",{g.d-1,g.c},true);
    const auto weights=T("conv_weights",{g.d,g.c},true),norm=T("norm",{g.s},true),dt=T("dt",{g.hv},true),a=T("ssm_a",{g.hv},true);
    std::array<TensorId,4> parts{};
    if(merged) parts=project_merged(graph,x,rows,g);
    else {
        const char* names[]={"w_qkv","w_gate","w_alpha","w_beta"};const uint64_t widths[]={g.c,g.hv*g.s,g.hv,g.hv};
        for(unsigned i=0;i<4;++i) {
            parts[i]=graph.tensor({"projection_"+std::to_string(i),DType::BF16,{r,widths[i]},false});
            graph.gemm(x,graph.tensor({names[i],DType::BF16,{widths[i],g.h}}),parts[i],true);
        }
    }
    const auto qkv=T("qkv_rows",{r,g.c}),z=T("z_rows",{r,g.hv*g.s}),alpha=T("alpha_rows",{r,g.hv}),raw_beta=T("beta_rows",{r,g.hv});
    graph.cast(parts[0],qkv);graph.cast(parts[1],z);graph.cast(parts[2],alpha);graph.cast(parts[3],raw_beta);
    const auto activated=T("activated_rows",{r,g.c}),conv_next=T("conv_state_out",{g.d-1,g.c},true);
    {abi::GdnConvRows args{int32_t(g.c),int32_t(g.d),int32_t(rows)};const TensorId in[]={conv,qkv,weights},out[]={conv_next,activated};
     graph.op(fast_sigmoid_enabled()?"strata_gdn_conv_final_fast_f32":"strata_gdn_conv_final_f32",in,out,&args,sizeof(args));}
    const auto q=T("q_rows",{r,g.hk,g.s}),k=T("k_rows",{r,g.hk,g.s}),v=T("v_rows",{r,g.hv,g.s});
    const auto gate=T("gate_rows",{r,g.hv}),beta=T("beta_prepared_rows",{r,g.hv});
    {abi::GdnPrepareRows args{int32_t(g.s),int32_t(g.hk),int32_t(g.hv),g.eps,int32_t(rows),0,0};
     const TensorId in[]={activated,alpha,dt,a,raw_beta},out[]={q,k,v,gate,beta};graph.op("strata_gdn_prepare_rows_f32",in,out,&args,sizeof(args));}
    const auto next=T("state_out",{g.s,g.hv,g.s},true),recurrent=T("recurrent_rows",{r,g.hv,g.s});
    if(gdn_prefill_chunked(rows)) {
        define_gdn_chunked_recurrence(graph,g.s,g.hk,g.hv,rows,state,q,k,v,gate,beta,next,recurrent);
    } else if(prefill_local_enabled()) {
        abi::GdnStepRows args{int32_t(g.s),int32_t(g.hk),int32_t(rows)};const TensorId in[]={state,q,k,v,gate,beta},out[]={next,recurrent};
        graph.op("strata_gdn_step_final_f32",in,out,&args,sizeof(args));
    } else if(gdn_step_window_enabled()) {
        // STRATA_HPU_GDN_WINDOW: the same rows in one launch; a member owns its state slice, so the row order and
        // every row's arithmetic are the per-row chain's (bit-identical outputs and final state).
        abi::GdnStepRows args{int32_t(g.s),int32_t(g.hk),int32_t(rows)};const TensorId in[]={state,q,k,v,gate,beta},out[]={next,recurrent};
        graph.op("strata_gdn_step_window_f32",in,out,&args,sizeof(args));
    } else {
        // Full-window prefill needs no intermediate persistent states. Keep the production
        // recurrence kernels and let the compiler reuse the transient state buffers.
        const auto slices=[&](TensorId all,const std::string& stem,std::vector<uint64_t> shape) {
            auto one=shape;one.insert(one.begin(),1);std::vector<TensorId> pieces,values;
            for(uint32_t row=0;row<rows;++row) {
                const auto name=stem+std::to_string(row);
                pieces.push_back(T(name+"slice",one));values.push_back(T(name,shape));
            }
            synSplitParams p{};p.axis=unsigned(shape.size());const TensorId in[]={all};graph.op("split",in,pieces,&p,sizeof(p));
            for(uint32_t row=0;row<rows;++row) graph.reshape(pieces[row],values[row]);
            return values;
        };
        const auto qs=slices(q,"q",{g.hk,g.s}),ks=slices(k,"k",{g.hk,g.s}),vs=slices(v,"v",{g.hv,g.s});
        const auto gates=slices(gate,"gate",{g.hv}),betas=slices(beta,"beta",{g.hv});
        auto previous=state;std::vector<TensorId> outputs;
        for(uint32_t row=0;row<rows;++row) {
            const auto stem="step_"+std::to_string(row);
            const auto current=row+1==rows?next:T(stem+"state",{g.s,g.hv,g.s});
            const auto out=T(stem+"out",{g.hv,g.s}),shaped=T(stem+"row",{1,g.hv,g.s});
            graph.gdn_step(previous,qs[row],ks[row],vs[row],gates[row],betas[row],current,out);
            graph.reshape(out,shaped);outputs.push_back(shaped);previous=current;
        }
        graph.concat(outputs,recurrent,0);
    }
    const auto y=graph.tensor({"y_rows",DType::BF16,{r,g.hv,g.s},false});
    {abi::GdnNormRows args{int32_t(g.s),int32_t(g.hv),g.eps,int32_t(rows),0};const TensorId in[]={recurrent,z,norm},out[]={y};
     graph.op(gdn_norm_rows_bf16_guid(g.s),in,out,&args,sizeof(args));}
    const auto matrix=graph.tensor({"y_matrix",DType::BF16,{r,g.hv*g.s},false});graph.reshape(y,matrix);
    (void)project_out(graph,g,matrix,rows,merged);
}
std::unique_ptr<Recipe> build_gdn_window(const Device& device,const core::ModelGeometry& geometry,uint32_t rows) {
    GraphBuilder graph(device);define_gdn_window(graph,geometry,rows);
    return graph.compile("strata_gdn_window_"+std::to_string(rows));
}
} // namespace strata::hpu
