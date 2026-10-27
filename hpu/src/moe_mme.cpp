#include "strata/hpu/moe_mme.hpp"
#include "strata/hpu/performance_modes.hpp"
#include "strata/hpu/model_abi.hpp"
#include "strata/hpu/resident_experts.hpp"
#include "strata/hpu/window_limits.hpp"
#include <perf_lib_layer_params.h>
#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace strata::hpu {
namespace {
int env_int(const char* name,int fallback) {
    const char* v=std::getenv(name);
    return v && *v?std::atoi(v):fallback;
}
}
uint32_t moe_mme_bucket_rows(uint32_t,const core::ModelGeometry&) {
    static const int rows=env_int("STRATA_HPU_MOE_MME_TILE",64);
    if(rows!=32 && rows!=64 && rows!=128 && rows!=256) throw std::invalid_argument("STRATA_HPU_MOE_MME_TILE must be 32, 64, 128 or 256");
    return uint32_t(rows);
}
uint32_t moe_mme_tiles(uint32_t rows,uint32_t tile_rows,const core::ModelGeometry& c) {
    // Every expert's last tile may be partial: sum(ceil(n_e / T)) <= routed / T + experts. Rounded up to 8 tiles.
    const uint64_t routed=uint64_t(rows)*uint64_t(c.n_expert_used);
    const uint64_t tiles=(routed+tile_rows-1)/tile_rows+uint64_t(c.n_expert);
    return uint32_t((tiles+7)/8*8);
}
std::vector<std::pair<std::string,std::vector<float>>> moe_mme_constants(const core::ModelGeometry& c) {
    // tile_upper[u][e] = u <= e: [1, experts] x tile_upper is the inclusive prefix sum of the experts' tile counts.
    const uint64_t e=c.n_expert;
    std::vector<float> upper(e*e);
    for(uint64_t u=0;u<e;++u) for(uint64_t v=0;v<e;++v) upper[u*e+v]=u<=v?1.f:0.f;
    return {{"tile_upper",std::move(upper)}};
}
TensorId define_resident_mme_moe(GraphBuilder& g,const core::ModelGeometry& c,TensorId x,TensorId logits,
                                 TensorId shared_gate,uint32_t rows,bool unreduced,uint32_t tile_rows) {
    const uint64_t h=c.n_embd,ff=c.n_ff,e=c.n_expert,k=c.n_expert_used,p=k+1,t=rows,slots=p*t,m=tile_rows;
    if(!rows || rows>kResidentWindowRows || slots>kResidentMoeSlots || (m!=32 && m!=64 && m!=128 && m!=256) || h%256 ||
       ff%128 || (2*ff)%1280 || h%1280 || e%64)
        throw std::invalid_argument("grouped MME MoE window or geometry unsupported");
    const uint64_t tiles=moe_mme_tiles(rows,tile_rows,c);
    // STRATA_HPU_MOE_MME_CHUNKS: the tiles run as this many expand -> GEMM -> activation -> expand -> GEMM chains,
    // which bounds the expanded weights alive at once (tiles / chunks experts' worth).
    static const int chunks=env_int("STRATA_HPU_MOE_MME_CHUNKS",4);
    if(chunks<1 || tiles%uint64_t(chunks)) throw std::invalid_argument("STRATA_HPU_MOE_MME_CHUNKS must divide the tile count");
    const auto gu_bytes=iq4_resident_bytes(2*ff,h),down_bytes=iq4_resident_bytes(h,ff);
    const auto T=[&](const std::string& name,DType d,std::vector<uint64_t> shape,bool persistent=false) {
        return g.tensor({name,d,std::move(shape),persistent});
    };
    const auto table=T("table",DType::U8,{256},true),w_gu=T("w_gu",DType::U8,{e,gu_bytes},true),w_down=T("w_down",DType::U8,{e,down_bytes},true);
    const auto s_gu=T("s_gu",DType::U8,{1,gu_bytes},true),s_down=T("s_down",DType::U8,{1,down_bytes},true);
    const auto ids=T("ids",DType::I32,{slots}),route=T("route_weights",DType::F32,{slots}),column=T("route_column",DType::F32,{slots,1});
    {abi::RouteTopk a{int32_t(e),int32_t(k),int32_t(p),int32_t(t)};const TensorId in[]={logits,shared_gate},out[]={ids,route};
     g.op("strata_route_topk_f32",in,out,&a,sizeof(a));}
    g.reshape(route,column);
    // Bucketing: per-(expert, 64-slot block) counts, per-expert running offsets / totals / tile counts, the tile
    // prefix sum (an MME product with the constant upper triangle), the tile -> expert map, every slot's tile row.
    const uint64_t blocks=(slots+63)/64;
    const auto counts=T("route_counts",DType::F32,{blocks,e}),offsets=T("route_offsets",DType::F32,{blocks,e});
    const auto totals=T("expert_totals",DType::I32,{e}),expert_tiles=T("expert_tiles",DType::F32,{e});
    const auto tile_upper=T("tile_upper",DType::F32,{e,e},true);
    const auto tiles_row=T("expert_tiles_row",DType::F32,{1,e}),end_row=T("tile_end_row",DType::F32,{1,e});
    const auto tile_end=T("tile_end",DType::F32,{e}),tile_start=T("tile_start",DType::F32,{e});
    const auto tile_expert=T("tile_expert",DType::I32,{tiles}),tile_fill=T("tile_rows",DType::I32,{tiles});
    // STRATA_HPU_MOE_PLACE: 64 trash entries after the tiles' rows take the shared slots' scatter.
    const auto bucket=T("bucket",DType::I32,{tiles*m+(moe_place_enabled()?64:0)}),place=T("place",DType::I32,{slots});
    {
        abi::MoeMmeRoute a{int32_t(e),int32_t(slots),int32_t(p),int32_t(k),int32_t(m)};
        {const TensorId in[]={ids},out[]={counts};g.op("strata_moe_mme_count_f32",in,out,&a,sizeof(a));}
        {const TensorId in[]={counts},out[]={offsets,totals,expert_tiles};g.op("strata_moe_mme_prefix_f32",in,out,&a,sizeof(a));}
        g.reshape(expert_tiles,tiles_row);g.gemm(tiles_row,tile_upper,end_row);g.reshape(end_row,tile_end);
        {const TensorId in[]={tile_end,expert_tiles},out[]={tile_start};g.op("sub_fwd_f32",in,out);}
        abi::MoeMmeTiles b{int32_t(e),int32_t(tiles),int32_t(m)};
        {const TensorId in[]={tile_end,tile_start,totals},out[]={tile_expert,tile_fill};g.op("strata_moe_mme_tilemap_f32",in,out,&b,sizeof(b));}
        if(!moe_place_enabled()) {const TensorId in[]={ids,offsets,tile_start},out[]={bucket,place};g.op("strata_moe_mme_scatter_f32",in,out,&a,sizeof(a));}
        else {
            abi::MoeMmePlace c{int32_t(e),int32_t(slots),int32_t(p),int32_t(k),int32_t(m),int32_t(tiles*m)};
            const auto target=T("place_target",DType::I32,{slots}),slot_ids=T("slot_ids",DType::I32,{slots});
            // The scatter's base: an output of the place kernel it never writes (its contents are only ever
            // overwritten or unread), instead of a constant fill every layer (0.33 ms).
            const auto blank=T("bucket_blank",DType::I32,{tiles*m+64});
            {const TensorId in[]={ids,offsets,tile_start},out[]={place,target,slot_ids,blank};g.op("strata_moe_mme_place_f32",in,out,&c,sizeof(c));}
            {ns_ScatterKernel::Params q{};q.axis=0;const TensorId in[]={blank,target,slot_ids},out[]={bucket};g.op("scatter_fwd_i32",in,out,&q,sizeof(q));}
        }
    }
    // The tiles' experts expanded to the B operand [count][K=columns][N=out_rows], rows interleaved per 256-row tile
    // (moe_mme_act / moe_mme_join_perm read the GEMM outputs back in natural order).
    const auto expand_tiles=[&](TensorId weights,uint64_t first,uint64_t count,uint64_t out_rows,uint64_t columns,const std::string& name) {
        const auto w=T(name,DType::BF16,{count,columns,out_rows});
        abi::Iq4Expand a{int32_t(out_rows),int32_t(columns),4,int32_t(first)};
        const TensorId in[]={table,weights,tile_expert},out[]={w};g.op("strata_iq4_expand_tiles_bf16",in,out,&a,sizeof(a));
        return w;
    };
    const auto expand_shared=[&](TensorId weights,uint64_t out_rows,uint64_t columns,const std::string& name) {
        const auto w=T(name,DType::BF16,{1,columns,out_rows});
        abi::Iq4Expand a{int32_t(out_rows),int32_t(columns),4,0};
        const TensorId in[]={table,weights},out[]={w};g.op("strata_iq4_expand_bf16",in,out,&a,sizeof(a));
        return w;
    };
    // SiLU(gate) * up of a {batch, rows, 2*ff} BF16 product as the down GEMM's BF16 A operand; with fill, the rows
    // past fill[first+b] are padding and are skipped.
    const auto activate=[&](TensorId gu,TensorId fill,uint64_t first,uint64_t batch,uint64_t count,const std::string& prefix) {
        const auto act16=T(prefix+"activation_bf16",DType::BF16,{batch,count,ff});
        abi::MoeMmeAct a{int32_t(ff),fill==tile_fill?1:0,int32_t(first)};
        const TensorId in[]={gu,fill},out[]={act16};g.op(fast_sigmoid_enabled()?"strata_moe_mme_act_fast_bf16":"strata_moe_mme_act_bf16",in,out,&a,sizeof(a));
        return act16;
    };
    // Routed tiles, chunk by chunk: A [count][m][h] gathered by bucket (padding rows unwritten), B expanded.
    const uint64_t per_chunk=tiles/uint64_t(chunks);
    std::vector<TensorId> downs;
    const bool fp8=moe_fp8_enabled();
    // STRATA_HPU_MOE_FP8: the expansion, the A operands and the GEMMs in FP8 with per-row scales; the scales of the
    // down product (act_scale per tile row, s_down per tile column) are applied by the join.
    std::vector<TensorId> down_scales,act_scales;
    for(int chunk=0;fp8 && chunk<chunks;++chunk) {
        const uint64_t first=uint64_t(chunk)*per_chunk;
        const auto prefix="c"+std::to_string(chunk)+"_";
        const auto a_gu=T(prefix+"a_gu_fp8",DType::F8,{per_chunk,m,h}),a_scale=T(prefix+"a_scale",DType::F32,{per_chunk,m,64});
        {
            abi::MoeMmeGather a{int32_t(h),int32_t(m),int32_t(p),int32_t(first)};
            const TensorId in[]={x,bucket,tile_fill},out[]={a_gu,a_scale};g.op("strata_moe_mme_gather_fp8_f32",in,out,&a,sizeof(a));
        }
        const auto expand=[&](TensorId weights,uint64_t out_rows,uint64_t columns,const std::string& name) {
            const auto w=T(name,DType::F8,{per_chunk,columns,out_rows}),scale=T(name+"_scale",DType::F32,{per_chunk,out_rows});
            abi::Iq4Expand a{int32_t(out_rows),int32_t(columns),4,int32_t(first)};
            const TensorId in[]={table,weights,tile_expert},out[]={w,scale};g.op("strata_iq4_expand_tiles_fp8",in,out,&a,sizeof(a));
            return std::pair{w,scale};
        };
        const auto [w_gu8,s_gu8]=expand(w_gu,2*ff,h,prefix+"w_gu_fp8");
        const auto out_gu=T(prefix+"out_gu",DType::BF16,{per_chunk,m,2*ff});
        g.batch_gemm(a_gu,w_gu8,out_gu,false);
        const auto act8=T(prefix+"activation_fp8",DType::F8,{per_chunk,m,ff}),act_scale=T(prefix+"act_scale",DType::F32,{per_chunk,m,64});
        {
            abi::MoeMmeAct a{int32_t(ff),1,int32_t(first)};
            const TensorId in[]={out_gu,s_gu8,a_scale,tile_fill},out[]={act8,act_scale};g.op("strata_moe_mme_act_fp8_bf16",in,out,&a,sizeof(a));
        }
        const auto [w_down8,s_down8]=expand(w_down,h,ff,prefix+"w_down_fp8");
        const auto out_down=T(prefix+"out_down",DType::BF16,{per_chunk,m,h});
        g.batch_gemm(act8,w_down8,out_down,false);
        downs.push_back(out_down);down_scales.push_back(s_down8);act_scales.push_back(act_scale);
    }
    for(int chunk=0;!fp8 && chunk<chunks;++chunk) {
        const uint64_t first=uint64_t(chunk)*per_chunk;
        const auto prefix="c"+std::to_string(chunk)+"_";
        const auto a_gu=T(prefix+"a_gu",DType::BF16,{per_chunk,m,h});
        {
            abi::MoeMmeGather a{int32_t(h),int32_t(m),int32_t(p),int32_t(first)};
            const TensorId in[]={x,bucket,tile_fill},out[]={a_gu};g.op("strata_moe_mme_gather_f32",in,out,&a,sizeof(a));
        }
        const auto out_gu=T(prefix+"out_gu",DType::BF16,{per_chunk,m,2*ff});
        g.batch_gemm(a_gu,expand_tiles(w_gu,first,per_chunk,2*ff,h,prefix+"w_gu_bf16"),out_gu,false);
        const auto out_down=T(prefix+"out_down",DType::BF16,{per_chunk,m,h});
        g.batch_gemm(activate(out_gu,tile_fill,first,per_chunk,m,prefix),expand_tiles(w_down,first,per_chunk,h,ff,prefix+"w_down_bf16"),out_down,false);
        downs.push_back(out_down);
    }
    const auto out_down=T("out_down",DType::BF16,{tiles,m,h});
    if(chunks==1) g.reshape(downs[0],out_down); else g.concat(downs,out_down,0);
    // Shared expert: every window row in order, so A is x itself.
    const auto xs=T("shared_x",DType::BF16,{t,h}),a_s=T("shared_a",DType::BF16,{1,t,h});
    g.cast(x,xs);g.reshape(xs,a_s);
    const auto s_out_gu=T("shared_out_gu",DType::BF16,{1,t,2*ff});
    g.batch_gemm(a_s,expand_shared(s_gu,2*ff,h,"s_gu_bf16"),s_out_gu,false);
    const auto s_out=T("shared_out",DType::BF16,{1,t,h});
    g.batch_gemm(activate(s_out_gu,ids,0,1,t,"shexp_"),expand_shared(s_down,h,ff,"s_down_bf16"),s_out,false);
    const auto routed_rows=T("routed_rows",DType::BF16,{tiles*m,h}),shared_rows=T("shared_rows",DType::BF16,{t,h});
    g.reshape(out_down,routed_rows);g.reshape(s_out,shared_rows);
    // The tail-fused HC write reads the BF16 expert rows in place (tpc/hc_write_moe_routed.c): no joined F32 copy.
    if(unreduced && !fp8) return routed_rows;
    const auto joined=T("expert_rows",DType::F32,{slots,h},unreduced);
    if(fp8) {
        const auto s_down=T("routed_down_scale",DType::F32,{tiles,h}),act_scale=T("routed_act_scale",DType::F32,{tiles,m,64});
        const auto act_rows=T("routed_act_scale_rows",DType::F32,{tiles*m,64});
        if(chunks==1) {g.reshape(down_scales[0],s_down);g.reshape(act_scales[0],act_scale);}
        else {g.concat(down_scales,s_down,0);g.concat(act_scales,act_scale,0);}
        g.reshape(act_scale,act_rows);
        abi::MoeMmeJoinFp8 a{int32_t(p),int32_t(k),int32_t(tiles*m),int32_t(m)};
        const TensorId in[]={routed_rows,shared_rows,place,s_down,act_rows},out[]={joined};g.op("strata_moe_mme_join_fp8_f32",in,out,&a,sizeof(a));
    } else {
        abi::U4MoeJoin a{int32_t(p),int32_t(k),int32_t(tiles*m)};
        const TensorId in[]={routed_rows,shared_rows,place},out[]={joined};g.op("strata_moe_mme_join_perm_f32",in,out,&a,sizeof(a));
    }
    if(unreduced) return joined;
    const auto weighted=T("weighted_rows",DType::F32,{slots,h});
    {const TensorId in[]={joined,column},out[]={weighted};g.op("mult_fwd_f32",in,out);}
    const auto grouped=T("weighted_grouped",DType::F32,{t,p,h});g.reshape(weighted,grouped);
    const auto sum=T("expert_sum",DType::F32,{t,1,h}),block=T("expert_block",DType::F32,{t,h});
    ns_Reduction::Params rp{};rp.reductionDimension=1;
    {const TensorId in[]={grouped},out[]={sum};g.op("reduce_sum_fwd_f32",in,out,&rp,sizeof(rp));}
    g.reshape(sum,block);return block;
}
}
