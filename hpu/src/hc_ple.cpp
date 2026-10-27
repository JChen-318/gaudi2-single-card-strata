#include "strata/hpu/hc_ple.hpp"
#include "strata/hpu/window_limits.hpp"
#include "strata/hpu/performance_modes.hpp"
#include "strata/hpu/model_abi.hpp"
#include "strata/hpu/ple_window_flow.hpp"
#include <cmath>
#include <perf_lib_layer_params.h>

namespace strata::hpu {
void HcConfig::validate() const {
    if(!hidden || hidden>65536 || hidden%64 || !streams || streams>32 || uint64_t(hidden)*streams>1048576 || !rank ||
       rank>65536 || !std::isfinite(epsilon) || epsilon<=0) throw std::invalid_argument("unsupported HC/PLE geometry/epsilon");
}
void define_hc_read(GraphBuilder& g,const HcConfig& c,bool final,bool merged,bool codes) {
    c.validate();const uint64_t h=c.hidden,r=c.streams,n=h*r,lr=c.rank;
    if(merged && final) throw std::invalid_argument("the final HC read has no inject to merge");
    const auto x=g.tensor({"residual",DType::F32,{r,h}}),norm=g.tensor({"norm",DType::F32,{r,h}});
    // STRATA_HPU_FP8_DENSE (merged only): FP8 weights with per-row scales, F32 activations into fp8_linear.
    const bool f8=merged && fp8_dense_enabled();const auto wt=f8?DType::F8:DType::BF16;
    const auto wd=merged?g.tensor({"w_down_inject",wt,{lr+r,n}}):g.tensor({"w_down",DType::BF16,{lr,n}}),wu=g.tensor({"w_up",wt,{n,lr}});
    const auto scale=g.tensor({"inverse_streams",DType::F32,{1}});
    const auto xn=g.tensor({"xn",DType::F32,{r,h},false}),xb=g.tensor({"xn_bf16",DType::BF16,{r,h},false});
    const auto xf=g.tensor({"xn_flat",DType::BF16,{1,n},false});
    if(merged && !f8 && hc_decode_tpc_enabled()) {
        // STRATA_HPU_HC_DECODE_TPC: norm (+BF16) -> TPC down/inject + activation -> up GEMM -> mix (+BF16).
        g.row_norm_dual(x,norm,xn,xb,c.epsilon);g.reshape(xb,xf);
        const auto lob=g.tensor({"lo_bf16",DType::BF16,{1,lr},false}),im=g.tensor({"inject_matrix",DType::F32,{1,r},false});
        {abi::HcDown a{int32_t(n),int32_t(lr),int32_t(r)};const TensorId in[]={xf,wd,scale},out[]={lob,im};
         g.op("strata_hc_down_bf16",in,out,&a,sizeof(a));}
        const auto gate=g.tensor({"gate",DType::F32,{1,n},false}),gates=g.tensor({"gate_streams",DType::F32,{r,h},false});
        g.gemm(lob,wu,gate,true);g.reshape(gate,gates);
        // The mix's BF16 copy and its packed Q8 codes (the resident MoE's input after the FFN read).
        const auto mixed=g.tensor({"mixed",DType::F32,{h}}),rounded=g.tensor({"mixed_bf16",DType::BF16,{1,h},false});
        const bool q8=codes && hc_mix_q8_enabled() && q8_packed_enabled();
        const auto x_codes=g.tensor({"x_codes",DType::U8,{1,q8?h/16:1,256},false}),x_scales=g.tensor({"x_scales",DType::F32,{1,q8?h/32:1,64},false});
        {abi::HcMixDual a{int32_t(h),int32_t(r),q8?1:0};const TensorId in[]={xn,gates},out[]={mixed,rounded,x_codes,x_scales};
         g.op("strata_hc_mix_dual_f32",in,out,&a,sizeof(a));}
        const auto inject=g.tensor({"inject",DType::F32,{r}});g.reshape(im,inject);
        return;
    }
    const auto down=g.tensor({"down",DType::F32,{1,lr},false}),scaled=g.tensor({"scaled",DType::F32,{1,lr},false});
    const auto sig=g.tensor({"sigmoid",DType::F32,{1,lr},false}),lo=g.tensor({"lo",DType::F32,{1,lr},false});
    const auto lob=g.tensor({"lo_bf16",DType::BF16,{1,lr},false}),gate=g.tensor({"gate",DType::F32,{1,n},false});
    const auto gates=g.tensor({"gate_streams",DType::F32,{r,h},false}),mixed=g.tensor({"mixed",DType::F32,{h}});
    g.row_norm(x,norm,xn,c.epsilon);g.cast(xn,xb);g.reshape(xb,xf);
    TensorId im{};
    if(merged) {
        const auto both=g.tensor({"down_inject",DType::F32,{1,lr+r},false});im=g.tensor({"inject_matrix",DType::F32,{1,r},false});
        if(f8) {
            const auto x32=g.tensor({"xn_f32_flat",DType::F32,{1,n},false});g.reshape(xn,x32);
            g.fp8_linear(x32,wd,g.tensor({"w_down_inject_scale",DType::F32,{1,lr+r}}),both,true);
        } else g.gemm(xf,wd,both,true);
        synSplitParams p{};p.axis=0;const TensorId in[]={both},out[]={down,im};g.op("split",in,out,&p,sizeof(p));
    } else g.gemm(xf,wd,down,true);
    g.multiply(down,scale,scaled);g.sigmoid(scaled,sig);g.multiply(scaled,sig,lo);
    g.cast(lo,lob); // (unused by the FP8 GEMM, which takes the F32 activation)
    if(f8) g.fp8_linear(lo,wu,g.tensor({"w_up_scale",DType::F32,{1,n}}),gate,true);
    else g.gemm(lob,wu,gate,true);
    g.reshape(gate,gates);g.hc_mix(xn,gates,mixed);
    if(merged) {const auto inject=g.tensor({"inject",DType::F32,{r}});g.reshape(im,inject);}
    else if(!final) {
        const auto wi=g.tensor({"w_inject",DType::BF16,{r,n}}),im=g.tensor({"inject_matrix",DType::F32,{1,r},false});
        const auto inject=g.tensor({"inject",DType::F32,{r}});g.gemm(xf,wi,im,true);g.reshape(im,inject);
    }
}
void define_hc_write(GraphBuilder& g,const HcConfig& c) {
    c.validate();const uint64_t h=c.hidden,r=c.streams;
    const auto x=g.tensor({"residual",DType::F32,{r,h}}),b=g.tensor({"block",DType::F32,{h}});
    const auto inject=g.tensor({"inject",DType::F32,{r}}),out=g.tensor({"out",DType::F32,{r,h}});g.hc_write(x,b,inject,out);
}
TensorId hc_read_bf16(GraphBuilder& g,const std::string& scope,TensorId row,const std::string& name,uint64_t hidden) {
    if(merged_gemm_enabled() && resident_iq4_enabled() && !fp8_dense_enabled() && hc_decode_tpc_enabled()) return g.find(scope+"mixed_bf16");
    const auto x=g.tensor({name,DType::BF16,{1,hidden},false});g.cast(row,x);return x;
}
TensorId hc_read_window_bf16(GraphBuilder& g,const std::string& scope,const std::string& name,uint64_t rows,uint64_t hidden) {
    if(hc_fused_enabled()) return g.find(scope+"mixed_bf16");
    const auto x=g.tensor({name,DType::BF16,{rows,hidden},false});g.cast(g.find(scope+"mixed"),x);return x;
}
void define_hc_read_window(GraphBuilder& g,const HcConfig& c,uint32_t rows,bool final,bool merged) {
    c.validate();if(!rows || rows>kResidentWindowRows) throw std::invalid_argument("HC read window requires 1.."+std::to_string(kResidentWindowRows)+" rows");
    if(merged && final) throw std::invalid_argument("the final HC read has no inject to merge");
    const uint64_t t=rows,h=c.hidden,r=c.streams,n=h*r,lr=c.rank;
    const auto x=g.tensor({"residual",DType::F32,{t,r,h}}),norm=g.tensor({"norm",DType::F32,{r,h}});
    const auto flat=g.tensor({"norm_input",DType::F32,{t*r,h},false}),xn=g.tensor({"xn_rows",DType::F32,{t*r,h},false});
    const auto normalized=g.tensor({"normalized",DType::F32,{t,r,h},false}),xb=g.tensor({"xn_bf16",DType::BF16,{t*r,h},false});
    const auto xf=g.tensor({"xn_flat",DType::BF16,{t,n},false});
    const bool f8=merged && fp8_dense_enabled();const auto wt=f8?DType::F8:DType::BF16;
    const auto wd=merged?g.tensor({"w_down_inject",wt,{lr+r,n}}):g.tensor({"w_down",DType::BF16,{lr,n}}),wu=g.tensor({"w_up",wt,{n,lr}});
    const auto scale=g.tensor({"inverse_streams",DType::F32,{1}});
    const auto down=g.tensor({"down",DType::F32,{t,lr},false}),scaled=g.tensor({"scaled",DType::F32,{t,lr},false});
    const auto sig=g.tensor({"sigmoid",DType::F32,{t,lr},false}),lo=g.tensor({"lo",DType::F32,{t,lr},false});
    const auto lob=g.tensor({"lo_bf16",DType::BF16,{t,lr},false}),gate=g.tensor({"gate",DType::F32,{t,n},false});
    const auto gates=g.tensor({"gate_streams",DType::F32,{t,r,h},false}),mixed=g.tensor({"mixed",DType::F32,{t,h},final || !window_internal_enabled()});
    g.reshape(x,flat);
    if(hc_fused_enabled()) g.row_norm_dual(flat,norm,xn,xb,c.epsilon);
    else {g.row_norm(flat,norm,xn,c.epsilon);g.cast(xn,xb);}
    g.reshape(xn,normalized);g.reshape(xb,xf);
    if(merged && !f8 && hc_window_tpc_enabled() && rows<=64 && n%256==0 && lr%2==0) {
        // STRATA_HPU_HC_WINDOW_TPC: down/inject and the activation in one TPC node, straight to the up GEMM's input.
        const auto inject=g.tensor({"inject",DType::F32,{t,r}});
        {abi::HcDownRows a{int32_t(n),int32_t(lr),int32_t(r),int32_t(rows)};const TensorId in[]={xf,wd,scale},out[]={lob,inject};
         g.op("strata_hc_down_rows_bf16",in,out,&a,sizeof(a));}
    } else if(merged) {
        const auto both=g.tensor({"down_inject",DType::F32,{t,lr+r},false}),inject=g.tensor({"inject",DType::F32,{t,r}});
        if(f8) {
            const auto x32=g.tensor({"xn_f32_flat",DType::F32,{t,n},false});g.reshape(xn,x32);
            g.fp8_linear(x32,wd,g.tensor({"w_down_inject_scale",DType::F32,{1,lr+r}}),both,true);
        } else g.gemm(xf,wd,both,true);
        synSplitParams p{};p.axis=0;const TensorId in[]={both},out[]={down,inject};g.op("split",in,out,&p,sizeof(p));
    } else g.gemm(xf,wd,down,true);
    if(!(merged && !f8 && hc_window_tpc_enabled() && rows<=64 && n%256==0 && lr%2==0)) {
        g.multiply(down,scale,scaled);g.sigmoid(scaled,sig);g.multiply(scaled,sig,lo);
        g.cast(lo,lob); // (unused by the FP8 GEMM, which takes the F32 activation)
    }
    if(f8) g.fp8_linear(lo,wu,g.tensor({"w_up_scale",DType::F32,{1,n}}),gate,true);
    else g.gemm(lob,wu,gate,true);
    g.reshape(gate,gates);
    // STRATA_HPU_HC_FUSED: "mixed_bf16" [t, h], the mix as the next projection's BF16 input (hc_read_window_bf16()).
    if(hc_fused_enabled()) g.hc_mix_window_dual(normalized,gates,mixed,g.tensor({"mixed_bf16",DType::BF16,{t,h},false}));
    else g.hc_mix_window(normalized,gates,mixed);
    if(!final && !merged) {
        const auto wi=g.tensor({"w_inject",DType::BF16,{r,n}}),inject=g.tensor({"inject",DType::F32,{t,r}});
        g.gemm(xf,wi,inject,true);
    }
}
void define_hc_write_window(GraphBuilder& g,const HcConfig& c,uint32_t rows,bool internal) {
    c.validate();if(!rows || rows>kResidentWindowRows) throw std::invalid_argument("HC write window requires 1.."+std::to_string(kResidentWindowRows)+" rows");
    const uint64_t t=rows,h=c.hidden,r=c.streams;
    const auto x=g.tensor({"residual",DType::F32,{t,r,h}}),b=g.tensor({"block",DType::F32,{t,h}});
    const auto inject=g.tensor({"inject",DType::F32,{t,r}}),out=g.tensor({"out",DType::F32,{t,r,h},!internal});g.hc_write_window(x,b,inject,out);
}
void define_ple_block(GraphBuilder& g,const HcConfig& c,bool debug) {
    c.validate();const uint64_t h=c.hidden,r=c.streams,n=h*r;
    const auto embedding=g.tensor({"embedding",DType::BF16,{1,h}}),hidden=g.tensor({"hidden",DType::F32,{r,h}});
    const auto wk=g.tensor({"w_key",DType::BF16,{n,h}}),wv=g.tensor({"w_value",DType::BF16,{h,h}});
    const auto nk=g.tensor({"norm_key",DType::F32,{r,h}}),nq=g.tensor({"norm_query",DType::F32,{r,h}});
    const auto nc=g.tensor({"norm_conv",DType::F32,{r,h}}),wc=g.tensor({"conv_weights",DType::F32,{4,n}});
    const auto history=g.tensor({"history",DType::F32,{9,n}}),next=g.tensor({"history_out",DType::F32,{9,n}});
    const auto km=g.tensor({"key_matrix",DType::F32,{1,n},false}),kr=g.tensor({"key_raw",DType::F32,{r,h},false});
    const auto vm=g.tensor({"value_matrix",DType::F32,{1,h},false}),value=g.tensor({"value",DType::F32,{h},debug});
    const auto key=g.tensor({"key",DType::F32,{r,h},debug}),query=g.tensor({"query",DType::F32,{r,h},debug});
    const auto gated=g.tensor({"gated",DType::F32,{r,h},debug}),normalized=g.tensor({"normalized",DType::F32,{r,h},debug});
    const auto hf=g.tensor({"hidden_flat",DType::F32,{n},false}),gf=g.tensor({"gated_flat",DType::F32,{n},false});
    const auto nf=g.tensor({"normalized_flat",DType::F32,{n},false}),of=g.tensor({"out_flat",DType::F32,{n},false});
    const auto out=g.tensor({"out",DType::F32,{r,h}});
    g.gemm(embedding,wk,km,true);g.reshape(km,kr);g.gemm(embedding,wv,vm,true);g.reshape(vm,value);
    g.row_norm(kr,nk,key,c.epsilon);g.row_norm(hidden,nq,query,c.epsilon);g.ple_gate(key,query,value,gated);
    g.row_norm(gated,nc,normalized,c.epsilon);g.reshape(hidden,hf);g.reshape(gated,gf);g.reshape(normalized,nf);
    g.ple_conv(history,nf,wc,hf,gf,next,of,3);g.reshape(of,out);
}
std::unique_ptr<Recipe> build_hc_read(const Device& d,const HcConfig& c,bool final) {
    GraphBuilder g(d);define_hc_read(g,c,final);return g.compile(final?"strata_hc_final":"strata_hc_read");
}
std::unique_ptr<Recipe> build_hc_write(const Device& d,const HcConfig& c) {
    GraphBuilder g(d);define_hc_write(g,c);return g.compile("strata_hc_write");
}
std::unique_ptr<Recipe> build_ple_block(const Device& d,const HcConfig& c,bool debug) {
    GraphBuilder g(d);define_ple_block(g,c,debug);return g.compile(debug?"strata_ple_debug":"strata_ple");
}
void define_ple_window(GraphBuilder& g,const HcConfig& c,uint32_t rows,bool row_history,bool resident) {
    c.validate();if(!rows || rows>(row_history || resident?kResidentWindowRows:kPleWindowRows))
        throw std::invalid_argument("PLE native window exceeds its causal/resident row limit");
    const uint64_t h=c.hidden,r=c.streams,n=h*r;
    const auto embedding=g.tensor({"embedding",DType::BF16,{rows,h}}),hidden=g.tensor({"hidden",DType::F32,{rows,r,h}});
    const auto wk=g.tensor({"w_key",DType::BF16,{n,h}}),wv=g.tensor({"w_value",DType::BF16,{h,h}});
    const auto nk=g.tensor({"norm_key",DType::F32,{r,h}}),nq=g.tensor({"norm_query",DType::F32,{r,h}});
    const auto nc=g.tensor({"norm_conv",DType::F32,{r,h}}),wc=g.tensor({"conv_weights",DType::F32,{4,n}});
    auto history=g.tensor({"history",DType::F32,{9,n}});
    const auto keys=g.tensor({"key_matrix",DType::F32,{rows,n},false}),values=g.tensor({"value_matrix",DType::F32,{rows,h},false});
    const auto hidden_flat=g.tensor({"hidden_matrix",DType::F32,{rows,n},false});
    g.gemm(embedding,wk,keys,true);g.gemm(embedding,wv,values,true);g.reshape(hidden,hidden_flat);
    if(resident && !row_history && ple_gate_simd_enabled()) {
        // A final-state resident window: the per-row norms, gate and conv below as one node each over all rows
        // (tpc/ple_gate_rows.c, tpc/ple_conv_rows.c), the same arithmetic per row. A per-row graph is ~6 nodes a row:
        // 50k nodes at 8192 rows, which Synapse takes many minutes to compile.
        const uint64_t t=uint64_t(rows)*r;
        const auto kr=g.tensor({"key_rows",DType::F32,{t,h},false}),input=g.tensor({"hidden_rows",DType::F32,{t,h},false});
        const auto key=g.tensor({"key",DType::F32,{t,h},false}),query=g.tensor({"query",DType::F32,{t,h},false});
        const auto gated=g.tensor({"gated",DType::F32,{t,h},false}),normalized=g.tensor({"normalized",DType::F32,{t,h},false});
        g.reshape(keys,kr);g.reshape(hidden_flat,input);
        g.row_norm(kr,nk,key,c.epsilon);g.row_norm(input,nq,query,c.epsilon);
        {abi::PleGateRows a{int32_t(h),int32_t(t),int32_t(r),1/std::sqrt(float(h))};const TensorId in[]={key,query,values},out[]={gated};
         g.op("strata_ple_gate_rows_f32",in,out,&a,sizeof(a));}
        g.row_norm(gated,nc,normalized,c.epsilon);
        const auto nf=g.tensor({"normalized_matrix",DType::F32,{rows,n},false}),gf=g.tensor({"gated_matrix",DType::F32,{rows,n},false});
        g.reshape(normalized,nf);g.reshape(gated,gf);
        const auto next=g.tensor({"history_out",DType::F32,{9,n}}),matrix=g.tensor({"out_matrix",DType::F32,{rows,n},false});
        {abi::PleConvRows a{int32_t(n),4,3,int32_t(rows)};const TensorId in[]={history,nf,wc,hidden_flat,gf},out[]={next,matrix};
         g.op("strata_ple_conv_rows_f32",in,out,&a,sizeof(a));}
        g.reshape(matrix,g.tensor({"out",DType::F32,{rows,r,h}}));
        return;
    }
    auto split=[&](TensorId matrix,const char* stem,uint64_t width) {
        std::vector<TensorId> pieces;
        for(uint32_t row=0;row<rows;++row) pieces.push_back(g.tensor({std::string(stem)+"_"+std::to_string(row),DType::F32,{1,width},false}));
        g.split_rows(matrix,pieces);return pieces;
    };
    const auto kp=split(keys,"key_piece",n),vp=split(values,"value_piece",h),hp=split(hidden_flat,"hidden_piece",n);
    std::vector<TensorId> outputs;
    for(uint32_t row=0;row<rows;++row) {
        const auto prefix="row_"+std::to_string(row)+"_";
        const auto kr=g.tensor({prefix+"key_raw",DType::F32,{r,h},false}),value=g.tensor({prefix+"value",DType::F32,{h},false});
        const auto input=g.tensor({prefix+"hidden",DType::F32,{r,h},false});
        const auto key=g.tensor({prefix+"key",DType::F32,{r,h},false}),query=g.tensor({prefix+"query",DType::F32,{r,h},false});
        const auto gated=g.tensor({prefix+"gated",DType::F32,{r,h},false}),normalized=g.tensor({prefix+"normalized",DType::F32,{r,h},false});
        const auto hf=g.tensor({prefix+"hidden_flat",DType::F32,{n},false}),gf=g.tensor({prefix+"gated_flat",DType::F32,{n},false});
        const auto nf=g.tensor({prefix+"normalized_flat",DType::F32,{n},false}),out=g.tensor({prefix+"out_flat",DType::F32,{n},false});
        // row_history: every row's history stays persistent under its row name (a verify window's prefixes).
        const bool last=!row_history && row+1==rows;
        const auto next=g.tensor({last?"history_out":prefix+"history_out",DType::F32,{9,n},last || row_history});
        const auto matrix=g.tensor({prefix+"out_matrix",DType::F32,{1,n},false});
        g.reshape(kp[row],kr);g.reshape(vp[row],value);g.reshape(hp[row],input);
        g.row_norm(kr,nk,key,c.epsilon);g.row_norm(input,nq,query,c.epsilon);g.ple_gate(key,query,value,gated);
        g.row_norm(gated,nc,normalized,c.epsilon);g.reshape(input,hf);g.reshape(gated,gf);g.reshape(normalized,nf);
        g.ple_conv(history,nf,wc,hf,gf,next,out,3);g.reshape(out,matrix);history=next;outputs.push_back(matrix);
    }
    const auto matrix=g.tensor({"out_matrix",DType::F32,{rows,n},false}),out=g.tensor({"out",DType::F32,{rows,r,h}});
    g.concatenate_rows(outputs,matrix);g.reshape(matrix,out);
}
std::unique_ptr<Recipe> build_ple_window(const Device& device,const HcConfig& c,uint32_t rows) {
    GraphBuilder graph(device);define_ple_window(graph,c,rows);return graph.compile("strata_ple_window_"+std::to_string(rows));
}
}
