#include "strata/hpu/u4_moe.hpp"
#include "strata/hpu/model_abi.hpp"
#include "strata/hpu/performance_modes.hpp"
#include "strata/hpu/window_limits.hpp"
#include <perf_lib_layer_params.h>
#include <algorithm>
#include <cstring>
#include <mutex>
#include <thread>

namespace strata::hpu {
uint32_t u4_bucket_rows(uint32_t rows,uint32_t top_k,uint32_t experts) {
    if(!rows || rows>kResidentWindowRows || !top_k || top_k>experts)
        throw std::invalid_argument("invalid U4 bucket geometry");
    // Capacity per expert, not a guarantee based on the sum over experts.
    // Wider windows require overflow replay before this prototype can be served.
    return 64;
}
TensorId define_ffn_moe(GraphBuilder& g,const core::ModelGeometry& c,TensorId x,TensorId logits,
                        TensorId shared_gate,uint32_t rows,bool unreduced) {
    if(!resident_u4_enabled()) return define_resident_moe(g,c,x,logits,shared_gate,rows,unreduced);
    throw std::invalid_argument("resident U4 serving requires a compatible decode/overflow path; use the isolated U4 probe");
}
TensorId define_resident_u4_moe(GraphBuilder& g,const core::ModelGeometry& c,const U4LayerStore& s,
                                TensorId x,TensorId logits,TensorId shared_gate,uint32_t rows,bool unreduced) {
    const uint64_t h=c.n_embd,ff=c.n_ff,e=c.n_expert,k=c.n_expert_used,p=k+1,t=rows,slots=p*t;
    const auto m=u4_bucket_rows(rows,uint32_t(k),uint32_t(e));
    if(slots>kResidentMoeSlots || s.experts!=e || s.hidden!=h || s.ff!=ff ||
       s.gu_group%64 || s.down_group%64 || ff%64 || h%64)
        throw std::invalid_argument("U4 MoE store or window geometry mismatch");
    const auto T=[&](std::string n,DType d,std::vector<uint64_t> shape,bool persistent=false) {
        return g.tensor({std::move(n),d,std::move(shape),persistent});
    };
    const auto ids=T("ids",DType::I32,{slots});
    const auto route=T("route_weights",DType::F32,{slots},true),column=T("route_column",DType::F32,{slots,1});
    {
        abi::RouteTopk a{int32_t(e),int32_t(k),int32_t(p),int32_t(t)};
        const TensorId in[]={logits,shared_gate},out[]={ids,route};g.op("strata_route_topk_f32",in,out,&a,sizeof(a));
    }
    g.reshape(route,column);
    const auto counts=T("route_blocks",DType::F32,{e,(slots+63)/64});
    const auto bucket=T("bucket",DType::I32,{e*m}),spill=T("overflow",DType::I32,{e});
    const auto place=T("place",DType::I32,{slots});
    {
        abi::RouteGroup a{int32_t(e),int32_t(slots)};
        const TensorId in[]={ids},out[]={counts};g.op("strata_u4_route_blocks_f32",in,out,&a,sizeof(a));
        abi::U4MoeBucket b{int32_t(e),int32_t(m),int32_t(slots),int32_t(p),int32_t(k)};
        const TensorId bi[]={ids,counts},bo[]={bucket,spill,place};g.op("strata_u4_bucket_blocks_f32",bi,bo,&b,sizeof(b));
    }
    g.cast(spill,T("overflow_count",DType::F32,{e},true));
    // Code tensors use an explicit BF16 dequantizer. A mixed U4/BF16 GEMM is
    // rejected by the installed Gaudi2 SDK; the GC may fuse this legal pair.
    const auto multiply=[&](TensorId a,TensorId codes,TensorId out,uint64_t batch,uint64_t n,uint64_t width,const std::string& name) {
        const auto unpacked=T(name+"_unpacked",DType::BF16,{batch,n,width});
        ns_CastKernel::ParamsV3 cast{};
        const TensorId in[]={codes},o[]={unpacked};g.op("cast_u4_to_bf16",in,o,&cast,sizeof(cast));
        g.batch_gemm(a,unpacked,out,true);
    };
    const auto chain=[&](const std::string& prefix,uint64_t experts,uint64_t count,TensorId map) {
        const uint64_t gu_batch=experts*s.gu_groups,down_batch=experts*s.down_groups;
        const auto gu=T(prefix+"gu_codes",DType::U4,{gu_batch,2*ff,s.gu_group},true);
        const auto gs=T(prefix+"gu_scale",DType::F32,{gu_batch,2*ff},true),go=T(prefix+"gu_offset",DType::F32,{gu_batch,2*ff},true);
        const auto down=T(prefix+"down_codes",DType::U4,{down_batch,h,s.down_group},true);
        const auto ds=T(prefix+"down_scale",DType::F32,{down_batch,h},true),dof=T(prefix+"down_offset",DType::F32,{down_batch,h},true);
        const auto a=T(prefix+"a_gu",DType::BF16,{gu_batch,count,s.gu_group});
        const auto sum=T(prefix+"sumx",DType::F32,{gu_batch*count});
        {
            abi::U4MoeGather pms{int32_t(s.gu_groups),int32_t(s.gu_group),int32_t(count),int32_t(p)};
            const TensorId in[]={x,map},out[]={a,sum};g.op("strata_u4_moe_gather_f32",in,out,&pms,sizeof(pms));
        }
        const auto og=T(prefix+"out_gu",DType::F32,{gu_batch,count,2*ff});
        multiply(a,gu,og,gu_batch,2*ff,s.gu_group,prefix+"gu");
        const auto ad=T(prefix+"a_down",DType::BF16,{down_batch,count,s.down_group});
        const auto partial=T(prefix+"partial",DType::F32,{down_batch*count});
        {
            abi::U4MoeGuEpi pms{int32_t(experts),int32_t(s.gu_groups),int32_t(ff),int32_t(count),int32_t(s.down_group),int32_t(s.down_groups)};
            const TensorId in[]={og,gs,go,map,sum},out[]={ad,partial};g.op("strata_u4_moe_gu_epi_f32",in,out,&pms,sizeof(pms));
        }
        const auto od=T(prefix+"out_down",DType::F32,{down_batch,count,h});
        multiply(ad,down,od,down_batch,h,s.down_group,prefix+"down");
        const auto result=T(prefix+"rows",DType::F32,{slots,h});
        {
            abi::U4MoeDownEpi pms{int32_t(experts),int32_t(s.down_groups),int32_t(h),int32_t(count),int32_t(s.down_group),0};
            const TensorId in[]={od,ds,dof,map,partial,column},out[]={result};g.op("strata_u4_moe_down_epi_f32",in,out,&pms,sizeof(pms));
        }
        return result;
    };
    const auto routed=chain("",e,m,bucket);
    const auto shared=chain("shared_",1,t,T("shared_map",DType::I32,{t},true));
    const auto joined=T("expert_rows",DType::F32,{slots,h},true);
    {
        abi::U4MoeJoin a{int32_t(p),int32_t(k),int32_t(e*m)};
        const TensorId in[]={routed,shared,bucket,place},out[]={joined};g.op("strata_u4_moe_join_f32",in,out,&a,sizeof(a));
    }
    if(unreduced) return joined;
    const auto weighted=T("weighted_rows",DType::F32,{slots,h});
    {const TensorId in[]={joined,column},out[]={weighted};g.op("mult_fwd_f32",in,out);}
    const auto grouped=T("weighted_grouped",DType::F32,{t,p,h});g.reshape(weighted,grouped);
    const auto sum=T("expert_sum",DType::F32,{t,1,h}),out=T("expert_block",DType::F32,{t,h},true);
    ns_Reduction::Params rp{};rp.reductionDimension=1;
    const TensorId ins[]={grouped},outs[]={sum};g.op("reduce_sum_fwd_f32",ins,outs,&rp,sizeof(rp));
    g.reshape(sum,out);return out;
}

U4LayerExperts::U4LayerExperts(const Device& device,const WeightTable& table,uint32_t layer,unsigned threads)
    :store_(U4LayerStore::make(uint64_t(table.geometry().n_expert),uint64_t(table.geometry().n_embd),uint64_t(table.geometry().n_ff))),
     layer_(device,store_.bytes()),maps_(device,1024) {
    if(layer>=uint32_t(table.geometry().n_layers)) throw std::invalid_argument("U4 layer index");
    const auto prefix="blk."+std::to_string(layer)+".";
    const auto source=[](const WeightSlice& s) {return U4Source{s.bytes(),s.row_bytes(),s.rows(),s.columns(),s.ggml_type()};};
    HostBuffer staging(device,u4_staging_bytes(store_));Stream copy(device);
    threads=std::max(1u,std::min(threads,unsigned(kU4PackChunk)));
    const uint64_t widths[]={store_.gu_codes_block,store_.gu_affine_block,store_.gu_affine_block,
                             store_.down_codes_block,store_.down_affine_block,store_.down_affine_block};
    const uint64_t starts[]={store_.gu_codes_start,store_.gu_scale_start,store_.gu_offset_start,
                             store_.down_codes_start,store_.down_scale_start,store_.down_offset_start};
    const uint64_t sources[]={store_.staging_gu_codes(),store_.staging_gu_scale(),store_.staging_gu_offset(),
                              store_.staging_down_codes(),store_.staging_down_scale(),store_.staging_down_offset()};
    for(uint64_t first=0;first<store_.blocks();first+=kU4PackChunk) {
        const uint64_t count=std::min(kU4PackChunk,store_.blocks()-first);
        // Workers produce AoS blocks in pageable RAM. Repack the six regions
        // into the pinned chunk before six contiguous uploads; AoS slices are not contiguous.
        std::vector<uint8_t> packed(count*store_.block_bytes());
        std::exception_ptr error;std::mutex lock;std::vector<std::thread> pool;
        for(unsigned worker=0;worker<threads;++worker) pool.emplace_back([&,worker] {
            try {
                for(uint64_t i=worker;i<count;i+=threads) {
                    const uint64_t expert=first+i;const bool shared=expert==store_.experts;
                    const auto get=[&](const char* role) {return shared?table.matrix(prefix+"ffn_"+role+"_shexp.weight"):
                        table.expert(prefix+"ffn_"+role+"_exps.weight",expert);};
                    const auto gate=get("gate"),up=get("up"),down=get("down");
                    pack_u4_block(source(gate),source(up),source(down),
                        std::span<uint8_t>(packed).subspan(i*store_.block_bytes(),store_.block_bytes()),store_);
                }
            } catch(...) {std::lock_guard guard(lock);if(!error) error=std::current_exception();}
        });
        for(auto& worker:pool) worker.join();
        if(error) std::rethrow_exception(error);
        uint64_t host_offset=0;
        for(size_t region=0;region<6;++region) {
            for(uint64_t i=0;i<count;++i) std::memcpy(static_cast<uint8_t*>(staging.data())+host_offset+i*widths[region],
                packed.data()+i*store_.block_bytes()+sources[region],widths[region]);
            upload(copy,layer_,staging,count*widths[region],starts[region]+first*widths[region],host_offset);
            host_offset+=count*widths[region];
        }
        copy.synchronize();
    }
    auto* map=static_cast<int32_t*>(staging.data());
    const auto per_row=table.geometry().n_expert_used+1;
    for(int32_t row=0;row<256;++row) map[row]=row*per_row+table.geometry().n_expert_used;
    upload(copy,maps_,staging,1024);copy.synchronize();
}
std::vector<Binding> U4LayerExperts::bindings() const {
    std::vector<Binding> out;
    const char* names[]={"gu_codes","gu_scale","gu_offset","down_codes","down_scale","down_offset"};
    const uint64_t starts[]={store_.gu_codes_start,store_.gu_scale_start,store_.gu_offset_start,
        store_.down_codes_start,store_.down_scale_start,store_.down_offset_start};
    const uint64_t widths[]={store_.gu_codes_block,store_.gu_affine_block,store_.gu_affine_block,
        store_.down_codes_block,store_.down_affine_block,store_.down_affine_block};
    for(size_t i=0;i<6;++i) {
        out.push_back({names[i],&layer_,starts[i]});
        out.push_back({std::string("shared_")+names[i],&layer_,starts[i]+store_.experts*widths[i]});
    }
    out.push_back({"shared_map",&maps_,0});return out;
}
}
