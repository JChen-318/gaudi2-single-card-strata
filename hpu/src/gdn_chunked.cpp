// STRATA_HPU_GDN_CHUNKED: the chunked gated delta rule (strata/hpu/gdn.hpp, define_gdn_chunked_recurrence).
//
// Per value head, with S the [key][value] state entering a chunk of C rows, G[t] the within-chunk cumulative log
// decay (gate is log decay, applied before the contraction as in tpc/gdn_step.c) and Gamma = exp(G):
//   A[t][u]  = beta[t] exp(G[t]-G[u]) (k[t].k[u])            for u < t          (strictly lower, nilpotent)
//   delta    = (I+A)^-1 (beta v) - (I+A)^-1 (beta Gamma k) S = u - w S
//   out      = (Gamma q) S + (exp(G[t]-G[u]) (q[t].k[u]))_{u<=t} delta
//   S'       = Gamma[C-1] S + (exp(G[C-1]-G) k)^T delta
// which unrolls the per-row recurrence exactly (in exact arithmetic). (I+A)^-1 is built hierarchically, because
// correlated keys make A close to an all-ones strictly lower matrix whose powers grow like binomials (C(63,31) ~ 1e18
// at C=64) before they cancel, so a Newton/Neumann series over the whole chunk loses everything in F32: first the
// 8x8 diagonal blocks by Newton's iteration X <- X - X ((I+A8) X masked strictly lower) from X = I - A8 (two steps
// are exact, growth at most C(7,3)), then three doublings X <- X - X (A masked to the lower-left s-blocks of each
// 2s-block) X for s = 8, 16, 32, each combining two exact inverses.
// Every product is a batched MME GEMM over (chunk, value head); the state is read and written once per chunk.
#include "strata/hpu/gdn.hpp"
#include <perf_lib_layer_params.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace strata::hpu {
std::vector<std::pair<std::string,std::vector<float>>> gdn_chunked_constants() {
    const uint32_t c=kGdnChunk;
    std::vector<float> upper(c*c),strict(c*c),inclusive(c*c),lower(c*c),eye(c*c),diagonal8(c*c),ll8(c*c),ll16(c*c),ll32(c*c);
    for(uint32_t t=0;t<c;++t) for(uint32_t u=0;u<c;++u) {
        // [row t][column u]. upper: the cumulative-sum operator (G = gate x upper); strict / inclusive: added to the
        // log-decay difference before exp (a large negative value where the term must vanish, so exp never sees a
        // positive exponent); lower: the Newton residual mask; eye: the identity.
        upper[t*c+u]=t<=u?1.f:0.f;
        strict[t*c+u]=u<t?0.f:-1e30f;
        inclusive[t*c+u]=u<=t?0.f:-1e30f;
        lower[t*c+u]=u<t?1.f:0.f;
        eye[t*c+u]=t==u?1.f:0.f;
        // diagonal8: the 8x8 diagonal blocks; llS: the lower-left SxS block of every 2Sx2S diagonal block.
        diagonal8[t*c+u]=t/8==u/8?1.f:0.f;
        const auto lower_left=[&](uint32_t size) {return t/(2*size)==u/(2*size) && (t/size)%2==1 && (u/size)%2==0?1.f:0.f;};
        ll8[t*c+u]=lower_left(8);ll16[t*c+u]=lower_left(16);ll32[t*c+u]=lower_left(32);
    }
    return {{"chunk.upper",upper},{"chunk.strict",strict},{"chunk.inclusive",inclusive},{"chunk.lower",lower},{"chunk.eye",eye},
            {"chunk.diagonal8",diagonal8},{"chunk.ll8",ll8},{"chunk.ll16",ll16},{"chunk.ll32",ll32}};
}
void define_gdn_chunked_recurrence(GraphBuilder& graph,uint64_t s,uint64_t hk,uint64_t hv,uint32_t rows,TensorId state,TensorId q,
                                   TensorId k,TensorId v,TensorId gate,TensorId beta,TensorId next,TensorId recurrent) {
    const uint64_t c=kGdnChunk,n=rows/c,rep=hv/hk,b=n*hv,r=rows;
    if(!rows || rows%c || !hk || hv%hk) throw std::invalid_argument("chunked GDN needs whole chunks and grouped heads");
    const auto T=[&](const std::string& name,std::vector<uint64_t> shape,bool persistent=false) {
        return graph.tensor({"chunk."+name,DType::F32,std::move(shape),persistent});
    };
    const auto op=[&](const char* guid,std::vector<TensorId> in,TensorId out,const void* params=nullptr,uint32_t bytes=0) {
        const TensorId outputs[]={out};graph.op(guid,in,outputs,params,bytes);
    };
    const auto mul=[&](TensorId a,TensorId x,TensorId out) {op("mult_fwd_f32",{a,x},out);return out;};
    // Logical [a,b,c,d] -> [a,c,b,d]; [a,b,c] -> [a,c,b]; [a,b,c] -> [b,a,c] (the SDK lists dimensions innermost first).
    const auto swap_middle=[&](TensorId x,TensorId y) {
        synTransposeParams p{};p.tensorDim=4;
        p.permutation[0]=TPD_Channel;p.permutation[1]=TPD_Height;p.permutation[2]=TPD_Width;p.permutation[3]=TPD_Depth;
        op("transpose",{x},y,&p,sizeof(p));
    };
    const auto swap_last=[&](TensorId x,TensorId y) {
        synTransposeParams p{};p.tensorDim=3;p.permutation[0]=TPD_Width;p.permutation[1]=TPD_Channel;p.permutation[2]=TPD_Height;
        op("transpose",{x},y,&p,sizeof(p));
    };
    const auto swap_first=[&](TensorId x,TensorId y) {
        synTransposeParams p{};p.tensorDim=3;p.permutation[0]=TPD_Channel;p.permutation[1]=TPD_Height;p.permutation[2]=TPD_Width;
        op("transpose",{x},y,&p,sizeof(p));
    };
    const auto upper=T("upper",{c,c},true),strict=T("strict",{1,c,c},true),inclusive=T("inclusive",{1,c,c},true);
    const auto lower=T("lower",{1,c,c},true),eye=T("eye",{1,c,c},true),diagonal8=T("diagonal8",{1,c,c},true);
    const TensorId lower_left[]={T("ll8",{1,c,c},true),T("ll16",{1,c,c},true),T("ll32",{1,c,c},true)};
    // Per-(chunk, value head) rows: index chunk*hv + head, head = i*hk + key head (tpc/gdn_step.c reads key head
    // head % hk), so [chunk][i][key head] is the same index split three ways.
    const auto per_head=[&](TensorId x,const std::string& name) {
        const auto x3=T(name+"_3",{n,c,hv});graph.reshape(x,x3);
        const auto t3=T(name+"_t",{n,hv,c});swap_last(x3,t3);
        const auto flat=T(name+"_bc",{b,c});graph.reshape(t3,flat);return flat;
    };
    const auto gates=per_head(gate,"gate"),betas=per_head(beta,"beta");
    const auto cum=T("g",{b,c});graph.gemm(gates,upper,cum);
    const auto g_col=T("g_col",{b,c,1}),g_row=T("g_row",{b,1,c}),beta_col=T("beta_col",{b,c,1});
    graph.reshape(cum,g_col);graph.reshape(cum,g_row);graph.reshape(betas,beta_col);
    const auto g_head=T("g_head",{b,c-1}),g_last2=T("g_last2",{b,1});
    {synSplitParams p{};p.axis=0;const TensorId ins[]={cum},outs[]={g_head,g_last2};graph.op("split",ins,outs,&p,sizeof(p));}
    const auto g_last=T("g_last",{b,1,1});graph.reshape(g_last2,g_last);
    const auto decay=[&](TensorId mask,const std::string& name) {
        const auto diff=T(name+"_diff",{b,c,c}),masked=T(name+"_masked",{b,c,c}),out=T(name,{b,c,c});
        op("sub_fwd_f32",{g_col,g_row},diff);op("add_fwd_f32",{diff,mask},masked);op("exp_fwd_f32",{masked},out);return out;
    };
    const auto d_strict=decay(strict,"d_strict"),d_incl=decay(inclusive,"d_incl");
    const auto gamma=T("gamma",{b,c,1});op("exp_fwd_f32",{g_col},gamma);
    const auto tail_log=T("tail_log",{b,c,1}),tail=T("tail",{b,c,1});
    op("sub_fwd_f32",{g_last,g_col},tail_log);op("exp_fwd_f32",{tail_log},tail);
    const auto gamma_last=T("gamma_last",{b,1,1});op("exp_fwd_f32",{g_last},gamma_last);
    // Q/K per (chunk, key head) as [chunk*hk][c][s]; a key head's Gram matrices serve its rep value heads.
    const auto compact=[&](TensorId x,uint64_t heads,const std::string& name) {
        const auto x4=T(name+"_4",{n,c,heads,s});graph.reshape(x,x4);
        const auto t4=T(name+"_t",{n,heads,c,s});swap_middle(x4,t4);
        const auto flat=T(name+"_c",{n*heads,c,s});graph.reshape(t4,flat);return flat;
    };
    const auto kc=compact(k,hk,"k"),qc=compact(q,hk,"q"),vb0=compact(v,hv,"v");
    const auto kk=T("kk",{n*hk,c,c}),qk=T("qk",{n*hk,c,c});
    graph.batch_gemm(kc,kc,kk,true);graph.batch_gemm(qc,kc,qk,true);
    // compact [chunk][1][key head][...] times a per-value-head factor [chunk][rep][key head][...] -> [b][...].
    const auto widen=[&](TensorId compact_x,TensorId factor,uint64_t rows_of,uint64_t cols_of,uint64_t factor_cols,const std::string& name) {
        const auto x5=T(name+"_x5",{n,1,hk,rows_of,cols_of}),f5=T(name+"_f5",{n,rep,hk,rows_of,factor_cols});
        const auto o5=T(name+"_o5",{n,rep,hk,rows_of,cols_of}),out=T(name,{b,rows_of,cols_of});
        graph.reshape(compact_x,x5);graph.reshape(factor,f5);mul(x5,f5,o5);graph.reshape(o5,out);return out;
    };
    const auto a_unscaled=widen(kk,d_strict,c,c,c,"a_unscaled");
    const auto a=T("a",{b,c,c});mul(a_unscaled,beta_col,a);
    const auto p_intra=widen(qk,d_incl,c,c,c,"p_intra");
    const auto w_scale=T("w_scale",{b,c,1});mul(beta_col,gamma,w_scale);
    const auto kw=widen(kc,w_scale,c,s,1,"kw");
    const auto qd=widen(qc,gamma,c,s,1,"qd");
    const auto kd=widen(kc,tail,c,s,1,"kd");
    const auto kd_t=T("kd_t",{b,s,c});swap_last(kd,kd_t);
    const auto vb=T("vb",{b,c,s});mul(vb0,beta_col,vb);
    // (I + A)^-1: the 8x8 diagonal blocks by Newton's iteration, then three exact doublings.
    static_assert(kGdnChunk==64,"the block masks are built for 64-row chunks");
    const auto a8=T("a8",{b,c,c}),l8=T("l8",{b,c,c});mul(a,diagonal8,a8);op("add_fwd_f32",{a8,eye},l8);
    auto x=T("x0",{b,c,c});op("sub_fwd_f32",{eye,a8},x);
    for(int i=1;i<=2;++i) {
        const auto stem="newton"+std::to_string(i)+"_";
        const auto lx=T(stem+"lx",{b,c,c}),residual=T(stem+"res",{b,c,c}),step=T(stem+"step",{b,c,c}),next_x=T(stem+"x",{b,c,c});
        graph.batch_gemm(l8,x,lx);mul(lx,lower,residual);graph.batch_gemm(x,residual,step);op("sub_fwd_f32",{x,step},next_x);x=next_x;
    }
    for(int level=0;level<3;++level) {
        const auto stem="double"+std::to_string(level)+"_";
        const auto q=T(stem+"q",{b,c,c}),qx=T(stem+"qx",{b,c,c}),step=T(stem+"step",{b,c,c}),next_x=T(stem+"x",{b,c,c});
        mul(a,lower_left[level],q);graph.batch_gemm(q,x,qx);graph.batch_gemm(x,qx,step);op("sub_fwd_f32",{x,step},next_x);x=next_x;
    }
    const auto u=T("u",{b,c,s}),w=T("w",{b,c,s});graph.batch_gemm(x,vb,u);graph.batch_gemm(x,kw,w);
    // Chunk by chunk: the state as [head][key][value].
    const auto chunks=[&](TensorId all,uint64_t d1,uint64_t d2,const std::string& name) {
        const auto x4=T(name+"_n4",{n,hv,d1,d2});graph.reshape(all,x4);
        std::vector<TensorId> pieces,out;
        for(uint64_t i=0;i<n;++i) {pieces.push_back(T(name+"_p"+std::to_string(i),{1,hv,d1,d2}));out.push_back(T(name+"_"+std::to_string(i),{hv,d1,d2}));}
        synSplitParams p{};p.axis=3;const TensorId ins[]={x4};graph.op("split",ins,pieces,&p,sizeof(p));
        for(uint64_t i=0;i<n;++i) graph.reshape(pieces[i],out[i]);
        return out;
    };
    const auto us=chunks(u,c,s,"u"),ws=chunks(w,c,s,"w"),qds=chunks(qd,c,s,"qd"),ps=chunks(p_intra,c,c,"p"),kds=chunks(kd_t,s,c,"kd");
    const auto gls=chunks(gamma_last,1,1,"gl");
    auto st=T("s0",{hv,s,s});swap_first(state,st);
    std::vector<TensorId> outs;
    for(uint64_t i=0;i<n;++i) {
        const auto stem="c"+std::to_string(i)+"_";
        const auto ws_=T(stem+"ws",{hv,c,s}),vp=T(stem+"delta",{hv,c,s}),o1=T(stem+"o1",{hv,c,s}),o2=T(stem+"o2",{hv,c,s}),o=T(stem+"o",{hv,c,s});
        graph.batch_gemm(ws[i],st,ws_);op("sub_fwd_f32",{us[i],ws_},vp);
        graph.batch_gemm(qds[i],st,o1);graph.batch_gemm(ps[i],vp,o2);op("add_fwd_f32",{o1,o2},o);outs.push_back(o);
        const auto sd=T(stem+"sd",{hv,s,s}),ds=T(stem+"ds",{hv,s,s}),sn=T(stem+"s",{hv,s,s});
        mul(st,gls[i],sd);graph.batch_gemm(kds[i],vp,ds);op("add_fwd_f32",{sd,ds},sn);st=sn;
    }
    const auto all=T("out_hrs",{hv,r,s});
    if(n==1) graph.reshape(outs[0],all); else graph.concat(outs,all,1);
    swap_first(all,recurrent);
    swap_first(st,next);
}
}
