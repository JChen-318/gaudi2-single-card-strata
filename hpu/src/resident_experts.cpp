#include "strata/hpu/resident_experts.hpp"
#include "strata/hpu/model_abi.hpp"
#include "strata/hpu/performance_modes.hpp"
#include "strata/hpu/window_limits.hpp"
#include "strata/hpu/u4_resident.hpp"
#include "iq4_decode.hpp"
#include <cstdlib>
#include <algorithm>
#include <cstring>
#include <functional>
#include <map>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <perf_lib_layer_params.h>

namespace strata::hpu {
namespace {
constexpr int8_t kvalues[16]={-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113};
void check_layout(uint64_t rows,uint64_t columns) {
    if(!rows || rows%1280 || !columns || columns%32 || rows>65536 || columns>65536) throw std::invalid_argument("resident IQ4 layout needs rows%1280==0 and columns%32==0");
}
}
uint64_t iq4_resident_bytes(uint64_t rows,uint64_t columns) {check_layout(rows,columns);return rows/2*columns+rows*(columns/32)*4;}
std::array<uint8_t,256> iq4_codebook_table() {
    std::array<uint8_t,256> t{};for(int i=0;i<256;++i) t[i]=uint8_t(kvalues[i&15]);return t;
}
void pack_iq4_resident(std::span<const WeightSlice> parts,std::span<uint8_t> out) {
    if(parts.empty()) throw std::invalid_argument("resident IQ4 pack needs a matrix");
    const uint64_t columns=parts[0].columns();uint64_t rows=0;
    for(const auto& part:parts) {
        if(!detail::iq4_type(part.ggml_type()) || part.columns()!=columns || (part.ggml_type()==23 && columns%256)) throw std::invalid_argument("resident IQ4 pack needs IQ4_NL/IQ4_XS parts of one width");
        rows+=part.rows();
    }
    const auto bytes=iq4_resident_bytes(rows,columns);
    if(out.size()<bytes) throw std::invalid_argument("resident IQ4 pack output too small");
    const uint64_t blocks=columns/32,codes_bytes=rows/2*columns;
    std::vector<uint8_t> codes(columns);std::vector<float> scales(blocks);
    auto* scale_out=out.data()+codes_bytes;uint64_t row=0;
    for(const auto& part:parts) {
        const auto source=part.bytes();const uint64_t row_bytes=part.row_bytes();
        for(uint64_t r=0;r<part.rows();++r,++row) {
            detail::decode_iq4_row(source.data()+r*row_bytes,part.ggml_type(),columns,codes.data(),scales.data());
            const uint64_t tile=row/256,i=row%256,group=tile/5,t=tile%5,lane=4*(i%64)+i/64;
            for(uint64_t b=0;b<blocks;++b) {
                const uint64_t base=(group*blocks+b)*16;
                for(uint64_t p=0;p<16;++p)
                    out[((base+p)*5+t)*256+lane]=uint8_t(codes[32*b+2*p]|(codes[32*b+2*p+1]<<4));
                std::memcpy(scale_out+(((group*blocks+b)*5+t)*256+i)*4,&scales[b],4);
            }
        }
    }
}
std::string resident_iq4_unsupported(const WeightTable& table) {
    const auto& g=table.geometry();
    if(g.n_embd<=0 || g.n_ff<=0 || g.n_expert<=0 || g.n_expert_used<=0) return "no MoE geometry";
    if((2*uint64_t(g.n_ff))%1280 || uint64_t(g.n_embd)%1280 || uint64_t(g.n_embd)%256 || uint64_t(g.n_ff)%32) return "hidden/FF do not tile into 1280-row groups";
    if(g.n_expert%64 || g.n_expert>4096 || g.n_expert_used+1>64) return "router geometry unsupported";
    for(uint32_t layer=0;layer<uint32_t(g.n_layers);++layer) {
        const auto prefix="blk."+std::to_string(layer)+".";
        for(const char* role:{"ffn_gate_exps.weight","ffn_up_exps.weight","ffn_down_exps.weight","ffn_gate_shexp.weight","ffn_up_shexp.weight","ffn_down_shexp.weight"}) {
            const auto type=table.info(prefix+role).ggml_type;
            if(!detail::iq4_type(type)) return prefix+role+" is not IQ4_NL/IQ4_XS";
        }
    }
    return {};
}
std::string resident_u4_unsupported(const WeightTable& table) {
    const auto& g=table.geometry();
    if(g.n_embd<=0 || g.n_ff<=0 || g.n_expert<=0 || g.n_expert_used<=0) return "no MoE geometry";
    if(g.n_expert>4096 || g.n_expert_used+1>64) return "router geometry unsupported";
    try {(void)U4LayerStore::make(uint64_t(g.n_expert),uint64_t(g.n_embd),uint64_t(g.n_ff));}
    catch(const std::invalid_argument&) {return "hidden/FF do not tile into 128-byte-aligned U4 blocks";}
    for(uint32_t layer=0;layer<uint32_t(g.n_layers);++layer) {
        const auto prefix="blk."+std::to_string(layer)+".";
        for(const char* role:{"ffn_gate_exps.weight","ffn_up_exps.weight","ffn_down_exps.weight","ffn_gate_shexp.weight","ffn_up_shexp.weight","ffn_down_shexp.weight"}) {
            const auto type=table.info(prefix+role).ggml_type;
            if(!detail::iq4_type(type)) return prefix+role+" is not IQ4_NL/IQ4_XS";
        }
    }
    return {};
}
ResidentLayerExperts::ResidentLayerExperts(const Device& device,const WeightTable& table,uint32_t layer,unsigned threads)
    :gu_bytes_(iq4_resident_bytes(2*uint64_t(table.geometry().n_ff),uint64_t(table.geometry().n_embd))),
     down_bytes_(iq4_resident_bytes(uint64_t(table.geometry().n_embd),uint64_t(table.geometry().n_ff))),
     gu_(device,gu_bytes_*uint64_t(table.geometry().n_expert)),down_(device,down_bytes_*uint64_t(table.geometry().n_expert)),
     shared_gu_(device,gu_bytes_),shared_down_(device,down_bytes_) {
    if(gu_bytes_%128 || down_bytes_%128) throw std::invalid_argument("resident IQ4 expert size must keep 128-byte alignment");
    {
        static std::mutex mutex;static std::map<const void*,std::weak_ptr<const DeviceBuffer>> tables;
        std::lock_guard lock(mutex);auto& entry=tables[device.state().get()];
        if(!(table_=entry.lock())) {
            auto created=std::make_shared<DeviceBuffer>(device,256);HostBuffer host(device,256);
            const auto values=iq4_codebook_table();std::memcpy(host.data(),values.data(),256);
            Stream copy(device);upload(copy,*created,host,256);copy.synchronize();
            table_=created;entry=table_;
        }
    }
    const auto prefix="blk."+std::to_string(layer)+".";const uint32_t experts=uint32_t(table.geometry().n_expert);
    // Experts are packed by `threads` workers into a pinned chunk, uploaded, then the chunk is reused.
    const uint32_t per_chunk=std::max<uint32_t>(1,std::min<uint32_t>(experts,64));
    HostBuffer staging(device,uint64_t(per_chunk)*gu_bytes_);Stream copy(device);
    threads=std::max(1u,std::min(threads,per_chunk));
    const auto run=[&](uint32_t count,const std::function<void(uint32_t)>& one) {
        std::vector<std::thread> pool;std::exception_ptr error;std::mutex error_mutex;
        for(unsigned w=0;w<threads;++w) pool.emplace_back([&,w] {
            try {for(uint32_t i=w;i<count;i+=threads) one(i);}
            catch(...) {std::lock_guard lock(error_mutex);if(!error) error=std::current_exception();}
        });
        for(auto& t:pool) t.join();
        if(error) std::rethrow_exception(error);
    };
    auto* host=static_cast<uint8_t*>(staging.data());
    for(int role=0;role<2;++role) {
        const uint64_t bytes=role?down_bytes_:gu_bytes_;auto& target=role?down_:gu_;
        for(uint32_t first=0;first<experts;first+=per_chunk) {
            const uint32_t count=std::min(per_chunk,experts-first);
            run(count,[&](uint32_t i) {
                const uint32_t e=first+i;std::span<uint8_t> out(host+uint64_t(i)*bytes,bytes);
                if(role) {const WeightSlice part[]={table.expert(prefix+"ffn_down_exps.weight",e)};pack_iq4_resident(part,out);}
                else {const WeightSlice parts[]={table.expert(prefix+"ffn_gate_exps.weight",e),table.expert(prefix+"ffn_up_exps.weight",e)};pack_iq4_resident(parts,out);}
            });
            upload(copy,target,staging,uint64_t(count)*bytes,uint64_t(first)*bytes);copy.synchronize();
        }
    }
    {
        const WeightSlice parts[]={table.matrix(prefix+"ffn_gate_shexp.weight"),table.matrix(prefix+"ffn_up_shexp.weight")};
        pack_iq4_resident(parts,std::span<uint8_t>(host,gu_bytes_));upload(copy,shared_gu_,staging,gu_bytes_);copy.synchronize();
        const WeightSlice down[]={table.matrix(prefix+"ffn_down_shexp.weight")};
        pack_iq4_resident(down,std::span<uint8_t>(host,down_bytes_));upload(copy,shared_down_,staging,down_bytes_);copy.synchronize();
    }
}
std::vector<Binding> ResidentLayerExperts::bindings() const {
    return {{"w_gu",&gu_},{"w_down",&down_},{"s_gu",&shared_gu_},{"s_down",&shared_down_},{"table",table_.get()}};
}
TensorId define_resident_moe(GraphBuilder& g,const core::ModelGeometry& geometry,TensorId x,TensorId logits,TensorId shared_gate,uint32_t rows,bool unreduced,
                             TensorId x_codes_in,TensorId x_scales_in) {
    // Slots: each row's top_k routed experts then its shared expert; row r owns slots [r*per_row, (r+1)*per_row).
    const uint64_t h=geometry.n_embd,ff=geometry.n_ff,experts=geometry.n_expert,top_k=geometry.n_expert_used,per_row=top_k+1,r=rows,slots=per_row*r;
    if(!rows || rows>kResidentWindowRows || slots>kResidentMoeSlots || (rows>8 && !q8_packed_enabled()))
        throw std::invalid_argument("resident MoE needs rows <= the resident window, slots <=4096 and packed Q8 for rows >8");
    const auto gu_bytes=iq4_resident_bytes(2*ff,h),down_bytes=iq4_resident_bytes(h,ff);
    const auto t=[&](const std::string& name,DType dtype,std::vector<uint64_t> shape,bool persistent=false) {
        return g.tensor({name,dtype,std::move(shape),persistent});
    };
    const auto table=t("table",DType::U8,{256},true),w_gu=t("w_gu",DType::U8,{experts,gu_bytes},true),w_down=t("w_down",DType::U8,{experts,down_bytes},true);
    const auto s_gu=t("s_gu",DType::U8,{1,gu_bytes},true),s_down=t("s_down",DType::U8,{1,down_bytes},true);
    const auto ids=t("ids",DType::I32,{slots}),route_flat=t("route_weights",DType::F32,{slots}),route=t("route_column",DType::F32,{slots,1});
    {abi::RouteTopk a{int32_t(experts),int32_t(top_k),int32_t(per_row),int32_t(r)};const TensorId in[]={logits,shared_gate},out[]={ids,route_flat};
     g.op("strata_route_topk_f32",in,out,&a,sizeof(a));}
    g.reshape(route_flat,route);
    const auto quantize=[&](TensorId input,uint64_t count,uint64_t columns,const std::string& name) {
        const bool packed=q8_packed_enabled();
        const auto codes=t(name+"_codes",DType::U8,{count,packed?columns/16:columns,256}),scales=t(name+"_scales",DType::F32,{count,columns/32,64});
        abi::Q8Quant a{int32_t(columns),int32_t(count)};const TensorId in[]={input},out[]={codes,scales};
        g.op(packed?"strata_q8_quant_packed_f32":q8_shuffle_enabled()?"strata_q8_quant_shuffle_f32":"strata_q8_quant_f32",in,out,&a,sizeof(a));return std::pair{codes,scales};
    };
    // A window of several rows (STRATA_HPU_DFLASH verify) reads each expert's weights once for the slots sharing it;
    // one row keeps the original kernel. Both give the same outputs.
    // STRATA_HPU_DFLASH_GEMV_ROWS=1: windows use the shared-read kernel (identical outputs; measured slower so far).
    static const bool rows_kernel=[] {const char* e=std::getenv("STRATA_HPU_DFLASH_GEMV_ROWS");return e && *e=='1';}();
    const bool share_reads=r>1 && slots<=128 && rows_kernel && !q8_packed_enabled();
    TensorId share{};
    if(share_reads) {
        share=t("share",DType::I32,{slots,16});
        abi::RouteShare a{int32_t(slots),int32_t(per_row)};const TensorId in[]={ids},out[]={share};
        g.op("strata_route_share_i32",in,out,&a,sizeof(a));
    }
    // STRATA_HPU_MOE_GROUP: the window's slots are reordered by routed id, so consecutive members of a GEMV read the
    // same expert's weight rows back to back (L2-resident) instead of one full read per slot that routes to it.
    // Every slot's arithmetic and store column are unchanged, so the outputs are bit-identical to the packed kernel.
    const bool pair_reads=r>moe_group_min_rows() && moe_pair_enabled() && q8_packed_enabled();
    const bool group_reads=r>moe_group_min_rows() && (moe_group_enabled() || pair_reads) && q8_packed_enabled();
    TensorId order{};
    if(group_reads) {
        const auto starts=t("group_starts",DType::F32,{experts+1});order=t("group_order",DType::F32,{slots});
        abi::RouteGroup a{int32_t(experts),int32_t(slots)};
        {const TensorId in[]={ids},out[]={starts};g.op("strata_route_group_starts_f32",in,out,&a,sizeof(a));}
        if(pair_reads || moe_group_blocks_enabled()) {
            const auto counts=t("group_counts",DType::F32,{experts,(slots+63)/64});
            {const TensorId in[]={ids},out[]={counts};g.op("strata_u4_route_blocks_f32",in,out,&a,sizeof(a));}
            {const TensorId in[]={ids,starts,counts},out[]={order};g.op("strata_route_group_blocks_f32",in,out,&a,sizeof(a));}
        } else {
            const TensorId in[]={ids,starts},out[]={order};g.op("strata_route_group_order_f32",in,out,&a,sizeof(a));
        }
    }
    const auto gemv=[&](std::pair<TensorId,TensorId> input,TensorId weights,TensorId shared,TensorId out,uint64_t columns,uint64_t out_rows,uint64_t x_group,int parts) {
        abi::Iq4Gemv a{int32_t(columns),int32_t(out_rows),int32_t(x_group),parts,int32_t(slots),int32_t(per_row)};
        if(group_reads) {const TensorId in[]={input.first,input.second,table,ids,weights,shared,order},o[]={out};g.op(pair_reads?"strata_iq4q8_gemv_pair_f32":"strata_iq4q8_gemv_group_f32",in,o,&a,sizeof(a));}
        else if(share_reads) {const TensorId in[]={input.first,input.second,table,ids,weights,shared,share},o[]={out};g.op("strata_iq4q8_gemv_rows_f32",in,o,&a,sizeof(a));}
        else {const TensorId in[]={input.first,input.second,table,ids,weights,shared},o[]={out};
              g.op(q8_packed_enabled()?(gemv_pf_enabled()?"strata_iq4q8_gemv_packed_pf_f32":gemv_tile_enabled()?"strata_iq4q8_gemv_tile_f32":"strata_iq4q8_gemv_packed_f32"):gemv_pipe_enabled()?"strata_iq4q8_gemv_pipe_f32":"strata_iq4q8_gemv_f32",in,o,&a,sizeof(a));}
    };
    // gate/up: each row's slots share its x row; the columns split in two parts so 2*slots members fill the TPCs.
    if((x_codes_in==kNoTensor)!=(x_scales_in==kNoTensor) || (x_codes_in!=kNoTensor && !q8_packed_enabled()))
        throw std::invalid_argument("precomputed resident MoE codes need both tensors and packed Q8");
    const auto x_codes=x_codes_in!=kNoTensor?std::pair{x_codes_in,x_scales_in}:quantize(x,r,h,"x");
    // Column parts: more members for the TPCs (STRATA_HPU_GEMV_PARTS_GU / _DOWN), summed below.
    static const int gu_one=gemv_parts_env("STRATA_HPU_GEMV_PARTS_GU",2),down_count=gemv_parts_env("STRATA_HPU_GEMV_PARTS_DOWN",1);
    // STRATA_HPU_GEMV_PARTS_GU_WINDOW: the gate/up parts of a window of several rows (its slots fill the TPCs alone).
    static const int gu_window=gemv_parts_env("STRATA_HPU_GEMV_PARTS_GU_WINDOW",gu_one);
    const int gu_count=r>1?gu_window:gu_one;
    if((h/32)%uint64_t(gu_count) || (ff/32)%uint64_t(down_count)) throw std::invalid_argument("GEMV parts must divide the 32-column blocks");
    const uint64_t gp=uint64_t(gu_count),dp=uint64_t(down_count);
    const auto parts=t("gu_parts",DType::F32,{gp,slots,2*ff});gemv(x_codes,w_gu,s_gu,parts,h,2*ff,per_row,gu_count);
    std::pair<TensorId,TensorId> a_codes;
    if(moe_act_fused_enabled() && q8_packed_enabled()) {
        a_codes={t("activation_codes",DType::U8,{slots,ff/16,256}),t("activation_scales",DType::F32,{slots,ff/32,64})};
        abi::MoeActQuant a{int32_t(ff),int32_t(slots),int32_t(gp)};const TensorId in[]={parts},out[]={a_codes.first,a_codes.second};
        g.op("strata_moe_act_packed_f32",in,out,&a,sizeof(a));
    } else {
        const auto gu=t("gu_rows",DType::F32,{slots,2*ff});
        if(gp==1) g.reshape(parts,gu);
        else {
            const auto flat=t("gu_parts_flat",DType::F32,{gp,slots*2*ff});g.reshape(parts,flat);
            const auto summed=t("gu_sum",DType::F32,{1,slots*2*ff});
            {ns_Reduction::Params p{};p.reductionDimension=1;const TensorId in[]={flat},out[]={summed};g.op("reduce_sum_fwd_f32",in,out,&p,sizeof(p));}
            g.reshape(summed,gu);
        }
        const auto gate=t("gate",DType::F32,{slots,ff}),up=t("up",DType::F32,{slots,ff});
        {synSplitParams p{};p.axis=0;const TensorId in[]={gu},out[]={gate,up};g.op("split",in,out,&p,sizeof(p));}
        const auto sigmoid=t("gate_sigmoid",DType::F32,{slots,ff}),silu=t("gate_silu",DType::F32,{slots,ff}),activation=t("activation",DType::F32,{slots,ff});
        g.sigmoid(gate,sigmoid);g.multiply(gate,sigmoid,silu);g.multiply(silu,up,activation);
        a_codes=quantize(activation,slots,ff,"activation");
    }
    const auto expert_parts=t("expert_parts",DType::F32,{dp,slots,h});gemv(a_codes,w_down,s_down,expert_parts,ff,h,1,down_count);
    const auto expert_rows=t("expert_rows",DType::F32,{slots,h});
    if(dp==1) g.reshape(expert_parts,expert_rows);
    else {
        const auto down_flat=t("expert_parts_flat",DType::F32,{dp,slots*h}),down_sum=t("expert_parts_sum",DType::F32,{1,slots*h});g.reshape(expert_parts,down_flat);
        {ns_Reduction::Params q{};q.reductionDimension=1;const TensorId in[]={down_flat},out[]={down_sum};g.op("reduce_sum_fwd_f32",in,out,&q,sizeof(q));}
        g.reshape(down_sum,expert_rows);
    }
    if(unreduced) return expert_rows;
    const auto weighted=t("weighted_rows",DType::F32,{slots,h});
    {const TensorId in[]={expert_rows,route},out[]={weighted};g.op("mult_fwd_f32",in,out);}
    if(r==1) {
        const auto sum=t("expert_sum",DType::F32,{1,h}),block=t("expert_block",DType::F32,{h});
        {ns_Reduction::Params p{};p.reductionDimension=1;const TensorId in[]={weighted},out[]={sum};g.op("reduce_sum_fwd_f32",in,out,&p,sizeof(p));}
        g.reshape(sum,block);return block;
    }
    const auto grouped=t("weighted_grouped",DType::F32,{r,per_row,h});g.reshape(weighted,grouped);
    const auto sum=t("expert_sum",DType::F32,{r,1,h}),block=t("expert_block",DType::F32,{r,h});
    {ns_Reduction::Params p{};p.reductionDimension=1;const TensorId in[]={grouped},out[]={sum};g.op("reduce_sum_fwd_f32",in,out,&p,sizeof(p));}
    g.reshape(sum,block);return block;
}
}
