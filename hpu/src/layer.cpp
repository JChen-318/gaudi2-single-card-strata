#include "strata/hpu/layer.hpp"
#include "strata/hpu/layer_flow.hpp"
#include "strata/hpu/expert.hpp"
#include "strata/hpu/gdn_block.hpp"
#include "strata/hpu/gdn.hpp"
#include "strata/hpu/hc_ple.hpp"
#include "strata/hpu/hc_window.hpp"
#include "strata/hpu/host_recipe.hpp"
#include "strata/hpu/moe.hpp"
#include "strata/hpu/parked_pages.hpp"
#include "strata/hpu/qsa_block.hpp"
#include "strata/hpu/qsa_window_flow.hpp"
#include "strata/hpu/state_io.hpp"
#include "strata/hpu/session_flow.hpp"
#include "strata/hpu/ple_window.hpp"
#include "strata/hpu/performance_modes.hpp"
#include "strata/hpu/resident_experts.hpp"
#include "strata/hpu/model_abi.hpp"
#include "strata/hpu/moe_mme.hpp"
#include <perf_lib_layer_params.h>
#include "strata/hpu/qsa_phase_stats.hpp"
#include <array>
#include <chrono>
#include <atomic>
#include <optional>
#include <cmath>
#include <map>
#include <mutex>
#include <thread>

namespace strata::hpu {
// Every recipe of the prefill/verify chain runs on the chain stream, one at a time, so they share one workspace
// (the widest any of them needs) instead of one each: every row count and key bucket compiles its own recipes.
const std::string kChainWorkspace="verify_chain";
namespace {
std::atomic<uint64_t> next_layer_owner{1};
// STRATA_HPU_MERGED_GEMM applies to the resident chain's graphs (and their verify windows) only.
bool merged_layers() {return merged_gemm_enabled() && resident_iq4_enabled();}
// STRATA_HPU_MOE_MME, set by Layer::verify_chain for the window being built: with a bucket, a layer graph ends at the
// post-attention residual ("ffn_write.out" then holds attn_write.out) and the FFN half runs as its own launch of
// one window FFN graph shared by every layer (run_window_ffn), which builds the grouped MME MoE. The split keeps
// one MME workspace per device instead of one per layer graph shape.
thread_local uint32_t window_mme_bucket=0;
thread_local bool building_window_ffn=false;
struct MoeWindowScope {
    uint32_t saved;
    explicit MoeWindowScope(uint32_t bucket):saved(window_mme_bucket) {window_mme_bucket=bucket;}
    ~MoeWindowScope() {window_mme_bucket=saved;}
};
struct WindowFfnBuild {
    bool saved;
    WindowFfnBuild():saved(building_window_ffn) {building_window_ffn=true;}
    ~WindowFfnBuild() {building_window_ffn=saved;}
};
// A layer graph's recipe cache key: its row count, with the FFN split in the top bit.
uint32_t window_key(uint32_t rows) {return rows|(window_mme_bucket?0x80000000u:0u);}
// The bindings among `bindings` that `recipe` declares: after the split a layer graph no longer has the FFN's
// tensors and the window FFN graph has none of the attention's, and both refuse a binding without a tensor.
std::vector<Binding> declared(const Recipe& recipe,std::span<const Binding> bindings) {
    const auto tensors=recipe.tensors();std::vector<Binding> out;
    for(const auto& b:bindings)
        if(std::any_of(tensors.begin(),tensors.end(),[&](const TensorDesc& t){return t.name==b.name;})) out.push_back(b);
    return out;
}
// STRATA_HPU_RESIDENT_IQ4: after the router, the whole FFN half on the device (strata/hpu/resident_experts.hpp)
// and its HC write; the graph's result is "ffn_write.out", the next residual.
void append_resident_ffn(GraphBuilder& g,const core::ModelGeometry& geometry,const HcConfig& hc) {
    g.set_scope("moe.");
    // STRATA_HPU_HC_DECODE_TPC: the FFN read already quantized its mix (tpc/hc_mix_dual.c).
    const bool codes=merged_layers() && !fp8_dense_enabled() && hc_decode_tpc_enabled() && hc_mix_q8_enabled() && q8_packed_enabled();
    const bool fold=moe_tail_decode_enabled();
    const auto block=define_resident_moe(g,geometry,g.find("ffn_row"),g.find("router.logits"),g.find("router.shared_gate"),1,fold,
                                         codes?g.find("ffn.x_codes"):kNoTensor,codes?g.find("ffn.x_scales"):kNoTensor);
    g.set_scope({});
    if(fold) {
        // STRATA_HPU_MOE_TAIL_DECODE: the window kernel of MOE_TAIL_FUSED at one row, on views of the decode tensors.
        const uint64_t h=hc.hidden,r=hc.streams;
        const auto residual=g.tensor({"ffn_write.residual_rows",DType::F32,{1,r,h},false}),inject=g.tensor({"ffn_write.inject_rows",DType::F32,{1,r},false});
        const auto rows=g.tensor({"ffn_write.out_rows",DType::F32,{1,r,h},false});
        g.reshape(g.find("attn_write.out"),residual);g.reshape(g.find("ffn.inject"),inject);
        abi::HcWriteMoe a{int32_t(h),int32_t(r),1,int32_t(geometry.n_expert_used+1)};
        const TensorId in[]={residual,block,g.find("moe.route_weights"),inject},o[]={rows};
        g.op("strata_hc_write_moe_f32",in,o,&a,sizeof(a));
        g.reshape(rows,g.tensor({"ffn_write.out",DType::F32,{r,h}}));
        return;
    }
    g.alias("residual",g.find("attn_write.out"));g.alias("block",block);g.alias("inject",g.find("ffn.inject"));
    g.set_scope("ffn_write.");define_hc_write(g,hc);g.set_scope({});
}
// Fused GDN decode, first launch: HC read (attention) -> GDN -> HC write -> HC read (FFN) -> router. Every
// weight and both state buffers are bound per call, so one recipe per device serves all GDN layers.
// Outputs: attn_write.out (the residual after attention), ffn.mixed, ffn.inject, router.logits and
// router.shared_gate; the BF16 casts are the ones the per-stage path applies on the host.
std::unique_ptr<Recipe> build_fused_gdn_pre(const Device& device,const core::ModelGeometry& geometry,const HcConfig& hc,bool resident=false) {
    GraphBuilder g(device);const uint64_t h=hc.hidden,r=hc.streams;
    const auto residual=g.tensor({"residual",DType::F32,{r,h}});
    g.alias("residual",residual);g.set_scope("attn.");define_hc_read(g,hc,false,merged_layers());g.set_scope({});
    const auto attn_row=g.tensor({"attn_row",DType::F32,{1,h},false});
    g.reshape(g.find("attn.mixed"),attn_row);const auto attn_x=hc_read_bf16(g,"attn.",attn_row,"attn_x",h);
    g.alias("x",attn_x);g.set_scope("gdn.");define_gdn_block(g,geometry,merged_layers());g.set_scope({});
    const auto gdn_row=g.tensor({"gdn_row",DType::F32,{1,h},false}),block=g.tensor({"attn_block",DType::F32,{h},false});
    g.cast(g.find("gdn.out"),gdn_row);g.reshape(gdn_row,block);
    g.alias("residual",residual);g.alias("block",block);g.alias("inject",g.find("attn.inject"));
    g.set_scope("attn_write.");define_hc_write(g,hc);g.set_scope({});
    g.alias("residual",g.find("attn_write.out"));g.set_scope("ffn.");define_hc_read(g,hc,false,merged_layers(),true);g.set_scope({});
    const auto ffn_row=g.tensor({"ffn_row",DType::F32,{1,h},false});
    g.reshape(g.find("ffn.mixed"),ffn_row);const auto ffn_x=hc_read_bf16(g,"ffn.",ffn_row,"ffn_x",h);
    g.alias("x",ffn_x);g.set_scope("router.");define_moe_router(g,geometry,1,router_merge_enabled());g.set_scope({});
    if(resident) {append_resident_ffn(g,geometry,hc);return g.compile("strata_fused_gdn_layer_resident");}
    return g.compile("strata_fused_gdn_pre");
}
struct FusedGdnPre {std::mutex mutex;std::unique_ptr<HostRecipe> recipe,chain;std::map<uint32_t,std::unique_ptr<HostRecipe>> verify,prefill_final;
    std::map<uint32_t,std::unique_ptr<HostRecipe>> chain_layers; // STRATA_HPU_DECODE_RECIPE_PER_LAYER, per layer index
};
std::shared_ptr<FusedGdnPre> shared_fused_gdn_pre(const Device& device,const core::ModelGeometry& geometry,const HcConfig& hc,std::span<const Binding> first) {
    static std::mutex registry_mutex;static std::map<const void*,std::weak_ptr<FusedGdnPre>> registry;
    std::lock_guard lock(registry_mutex);auto& entry=registry[device.state().get()];
    if(auto existing=entry.lock()) return existing;
    auto created=std::make_shared<FusedGdnPre>();
    // The resident mode decodes only through the chain (its own graph, built on first use).
    if(!resident_iq4_enabled()) {
        created->recipe=std::make_unique<HostRecipe>(device,build_fused_gdn_pre(device,geometry,hc),first,
            std::vector<std::string>{"attn_write.out","ffn.mixed","ffn.inject","router.logits","router.shared_gate"});
        const std::array<float,1> scale{1/float(hc.streams)};created->recipe->set("attn.inverse_streams",scale);created->recipe->set("ffn.inverse_streams",scale);
    }
    entry=created;return created;
}
// One current row shape per device, shared by all layers. Wide prefill must
// never fall back to MoeBlock's host routing / FP8 expert-cache executor.
struct ResidentMoeWindow {
    std::mutex mutex;uint32_t rows=0;std::unique_ptr<HostRecipe> recipe;
    std::vector<float> run(const Device& device,const core::ModelGeometry& geometry,
                           std::span<const float> input,std::span<const Binding> weights) {
        std::lock_guard lock(mutex);const auto count=uint32_t(input.size()/geometry.n_embd);
        if(count!=rows) {
            recipe.reset();rows=0;GraphBuilder g(device);const uint64_t h=geometry.n_embd;
            const auto x=g.tensor({"x",DType::F32,{count,h}}),bf=g.tensor({"bf",DType::BF16,{count,h},false});g.cast(x,bf);
            g.alias("x",bf);g.set_scope("router.");define_moe_router(g,geometry,count,router_merge_enabled());g.set_scope({});
            g.set_scope("moe.");const auto block=define_resident_moe(g,geometry,x,g.find("router.logits"),g.find("router.shared_gate"),count);g.set_scope({});
            const auto out=g.tensor({"out",DType::F32,{count,h}});g.reshape(block,out);
            recipe=std::make_unique<HostRecipe>(device,g.compile("strata_resident_moe_window_"+std::to_string(count)),weights,std::vector<std::string>{"out"});rows=count;
        }
        for(const auto& binding:weights) recipe->bind_external(binding);
        recipe->set("x",input);recipe->run();return recipe->values("out");
    }
};
std::shared_ptr<ResidentMoeWindow> shared_resident_moe_window(const Device& device) {
    static std::mutex mutex;static std::map<const void*,std::weak_ptr<ResidentMoeWindow>> registry;
    std::lock_guard lock(mutex);auto& entry=registry[device.state().get()];
    if(auto existing=entry.lock()) return existing;
    auto created=std::make_shared<ResidentMoeWindow>();entry=created;return created;
}
// Fused QSA decode. First launch: HC read (attention) -> projections; outputs qsa.q_gate/k/v/index_query/
// index_key and attn.inject. The host then normalizes, rotates, appends, selects and gathers (QsaBlock::
// fused_attend). Second launch, one recipe per attention bucket: attention -> output projection -> HC write
// -> HC read (FFN) -> router, with the same outputs as the GDN first launch. All layers share them.
std::unique_ptr<Recipe> build_fused_qsa_project(const Device& device,const QsaConfig& c,const HcConfig& hc) {
    GraphBuilder g(device);const uint64_t h=hc.hidden,r=hc.streams;
    const auto residual=g.tensor({"residual",DType::F32,{r,h}});
    g.alias("residual",residual);g.set_scope("attn.");define_hc_read(g,hc,false,merged_layers());g.set_scope({});
    const auto attn_row=g.tensor({"attn_row",DType::F32,{1,h},false});
    g.reshape(g.find("attn.mixed"),attn_row);const auto attn_x=hc_read_bf16(g,"attn.",attn_row,"attn_x",h);
    g.alias("x",attn_x);g.set_scope("qsa.");define_qsa_projections(g,c,uint32_t(h),merged_layers());g.set_scope({});
    return g.compile("strata_fused_qsa_project");
}
std::unique_ptr<Recipe> build_fused_qsa_finish(const Device& device,const core::ModelGeometry& geometry,const QsaConfig& c,const HcConfig& hc,uint32_t bucket,bool resident=false) {
    GraphBuilder g(device);const uint64_t h=hc.hidden,r=hc.streams;
    const auto residual=g.tensor({"residual",DType::F32,{r,h}}),inject=g.tensor({"inject",DType::F32,{r}});
    g.set_scope("att.");define_qsa_attention(g,c,bucket,false);g.set_scope({});
    if(resident && qsa_device_gather_enabled())
        define_qsa_device_gather(g,c,bucket,1,g.find("att.key"),g.find("att.value"),g.find("att.mask"),1);
    g.alias("attention",g.find("att.out"));g.set_scope("o.");define_qsa_output(g,c,uint32_t(h),merged_layers());g.set_scope({});
    const auto block=g.tensor({"attn_block",DType::F32,{h},false});g.reshape(g.find("o.out"),block);
    g.alias("residual",residual);g.alias("block",block);g.alias("inject",inject);
    g.set_scope("attn_write.");define_hc_write(g,hc);g.set_scope({});
    g.alias("residual",g.find("attn_write.out"));g.set_scope("ffn.");define_hc_read(g,hc,false,merged_layers(),true);g.set_scope({});
    const auto ffn_row=g.tensor({"ffn_row",DType::F32,{1,h},false});
    g.reshape(g.find("ffn.mixed"),ffn_row);const auto ffn_x=hc_read_bf16(g,"ffn.",ffn_row,"ffn_x",h);
    g.alias("x",ffn_x);g.set_scope("router.");define_moe_router(g,geometry,1,router_merge_enabled());g.set_scope({});
    if(resident) {append_resident_ffn(g,geometry,hc);return g.compile("strata_fused_qsa_finish_resident_"+std::to_string(bucket));}
    return g.compile("strata_fused_qsa_finish_"+std::to_string(bucket));
}
// STRATA_HPU_QSA_DEVICE: one QSA decode layer as one launch (identity selection): HC read -> projections ->
// strata_qsa_prep_f32 (norm, RoPE, FP16-grid K/V into the cache row and the token's stage) -> attention over
// [current cell; first `bucket` cached cells] -> output -> HC write -> HC read -> router -> resident FFN.
std::unique_ptr<Recipe> build_fused_qsa_device(const Device& device,const core::ModelGeometry& geometry,const QsaConfig& c,const HcConfig& hc,uint32_t bucket) {
    GraphBuilder g(device);const uint64_t h=hc.hidden,r=hc.streams,hq=c.query_heads,hk=c.kv_heads,d=c.width,b=bucket;
    const auto t=[&](const std::string& name,DType dtype,std::vector<uint64_t> shape,bool persistent=false) {return g.tensor({name,dtype,std::move(shape),persistent});};
    const auto residual=g.tensor({"residual",DType::F32,{r,h}});
    g.alias("residual",residual);g.set_scope("attn.");define_hc_read(g,hc,false,merged_layers());g.set_scope({});
    const auto attn_row=t("attn_row",DType::F32,{1,h});
    g.reshape(g.find("attn.mixed"),attn_row);const auto attn_x=hc_read_bf16(g,"attn.",attn_row,"attn_x",h);
    g.alias("x",attn_x);g.set_scope("qsa.");define_qsa_projections(g,c,uint32_t(h),merged_layers());g.set_scope({});
    const auto q_raw=t("dev.q_raw",DType::F32,{hq,d}),gate=t("dev.gate",DType::F32,{hq,d});
    {synSplitParams p{};p.axis=0;const TensorId in[]={g.find("qsa.q_gate")},out[]={q_raw,gate};g.op("split",in,out,&p,sizeof(p));}
    const auto q_gamma=t("dev.q_gamma",DType::F32,{d},true),k_gamma=t("dev.k_gamma",DType::F32,{d},true),rope=t("dev.rope",DType::F32,{c.rotary},true);
    const auto query=t("dev.query",DType::F32,{hq,d});
    // The cache row itself is written by QsaBlock::device_store after this launch: one recipe must not bind two
    // tensors over the same memory (Synapse rejects overlapping persistent sections), so the slots stay internal.
    const auto k_slot=t("dev.k_slot",DType::F32,{hk,d}),v_slot=t("dev.v_slot",DType::F32,{hk,d});
    const auto k_now=t("dev.k_now",DType::F32,{hk,d},true),v_now=t("dev.v_now",DType::F32,{hk,d},true);
    {abi::QsaPrep a{int32_t(d),int32_t(hq),int32_t(hk),int32_t(c.rotary),c.epsilon,1};
     const TensorId in[]={q_raw,g.find("qsa.k"),g.find("qsa.v"),q_gamma,k_gamma,rope},out[]={query,k_slot,v_slot,k_now,v_now};
     g.op("strata_qsa_prep_f32",in,out,&a,sizeof(a));}
    // Keys/values: the current cell first, then the cache's first `bucket` cells (token-major, transposed per head).
    const auto stack=[&](TensorId now,const char* cache_name,const char* stem) {
        const auto cache=qsa_cache_read(g,cache_name,b,hk,d),heads=t(std::string(stem)+"_heads",DType::F32,{hk,b,d});
        synTransposeParams p{};p.tensorDim=3;p.permutation[0]=TPD_Channel;p.permutation[1]=TPD_Height;p.permutation[2]=TPD_Width;
        {const TensorId in[]={cache},out[]={heads};g.op("transpose",in,out,&p,sizeof(p));}
        const auto current=t(std::string(stem)+"_now3",DType::F32,{hk,1,d});g.reshape(now,current);
        const auto all=t(std::string(stem)+"_all",DType::F32,{hk,b+1,d});const TensorId parts[]={current,heads};g.concat(parts,all,1);return all;
    };
    const auto key=stack(k_now,"dev.k_cache","dev.key"),value=stack(v_now,"dev.v_cache","dev.value");
    const auto mask=t("dev.mask",DType::F32,{b+1},true);
    g.alias("query",query);g.alias("key",key);g.alias("value",value);g.alias("mask",mask);g.alias("q_gate",gate);
    g.set_scope("att.");define_qsa_attention(g,c,bucket+1,false);g.set_scope({});
    g.alias("attention",g.find("att.out"));g.set_scope("o.");define_qsa_output(g,c,uint32_t(h),merged_layers());g.set_scope({});
    const auto block=t("attn_block",DType::F32,{h});g.reshape(g.find("o.out"),block);
    g.alias("residual",residual);g.alias("block",block);g.alias("inject",g.find("attn.inject"));
    g.set_scope("attn_write.");define_hc_write(g,hc);g.set_scope({});
    g.alias("residual",g.find("attn_write.out"));g.set_scope("ffn.");define_hc_read(g,hc,false,merged_layers(),true);g.set_scope({});
    const auto ffn_row=t("ffn_row",DType::F32,{1,h});
    g.reshape(g.find("ffn.mixed"),ffn_row);const auto ffn_x=hc_read_bf16(g,"ffn.",ffn_row,"ffn_x",h);
    g.alias("x",ffn_x);g.set_scope("router.");define_moe_router(g,geometry,1,router_merge_enabled());g.set_scope({});
    append_resident_ffn(g,geometry,hc);
    return g.compile("strata_fused_qsa_device_"+std::to_string(bucket));
}
// STRATA_HPU_QSA_DECODE_SELECT: build_fused_qsa_device past the identity selection. The token's indexer query (norm,
// RoPE: tpc/qsa_index_prep.c) scores every complete block's pooled key (dsel.pooled, the host's rows kept on the device),
// tpc/qsa_decode_blocks.c sums the heads, a top-k picks qsa_decode_top_blocks() blocks and tpc/qsa_gather_blocks_fp8.c
// gathers their cells (and the token's own block's earlier cells) from the FP8 cache; the token itself is attended
// first, as in the identity graph. dsel.cell {1} is the token's cell (bound into a table of cell numbers). `blocks`:
// the pooled rows scored (a bucket >= cell / block).
uint32_t qsa_decode_top_blocks(const QsaConfig& c) {return (c.selection_width(c.capacity)-1+c.block-1)/c.block;}
std::unique_ptr<Recipe> build_fused_qsa_decode_select(const Device& device,const core::ModelGeometry& geometry,const QsaConfig& c,const HcConfig& hc,
                                                      uint32_t blocks,uint32_t index_gamma_heads) {
    GraphBuilder g(device);const uint64_t h=hc.hidden,r=hc.streams,hq=c.query_heads,hk=c.kv_heads,d=c.width,hi=c.index_heads,di=c.index_width;
    const uint64_t top=qsa_decode_top_blocks(c),n=(top+1)*c.block,B=blocks;
    if(d!=256 || B%64 || B<top) throw std::invalid_argument("QSA decode select geometry");
    const auto t=[&](const std::string& name,DType dtype,std::vector<uint64_t> shape,bool persistent=false) {return g.tensor({name,dtype,std::move(shape),persistent});};
    const auto residual=g.tensor({"residual",DType::F32,{r,h}});
    g.alias("residual",residual);g.set_scope("attn.");define_hc_read(g,hc,false,merged_layers());g.set_scope({});
    const auto attn_row=t("attn_row",DType::F32,{1,h});
    g.reshape(g.find("attn.mixed"),attn_row);const auto attn_x=hc_read_bf16(g,"attn.",attn_row,"attn_x",h);
    g.alias("x",attn_x);g.set_scope("qsa.");define_qsa_projections(g,c,uint32_t(h),merged_layers());g.set_scope({});
    const auto q_raw=t("dev.q_raw",DType::F32,{hq,d}),gate=t("dev.gate",DType::F32,{hq,d});
    {synSplitParams p{};p.axis=0;const TensorId in[]={g.find("qsa.q_gate")},out[]={q_raw,gate};g.op("split",in,out,&p,sizeof(p));}
    const auto q_gamma=t("dev.q_gamma",DType::F32,{d},true),k_gamma=t("dev.k_gamma",DType::F32,{d},true),rope=t("dev.rope",DType::F32,{c.rotary},true);
    const auto query=t("dev.query",DType::F32,{hq,d});
    const auto k_slot=t("dev.k_slot",DType::F32,{hk,d}),v_slot=t("dev.v_slot",DType::F32,{hk,d});
    const auto k_now=t("dev.k_now",DType::F32,{hk,d},true),v_now=t("dev.v_now",DType::F32,{hk,d},true);
    {abi::QsaPrep a{int32_t(d),int32_t(hq),int32_t(hk),int32_t(c.rotary),c.epsilon,1};
     const TensorId in[]={q_raw,g.find("qsa.k"),g.find("qsa.v"),q_gamma,k_gamma,rope},out[]={query,k_slot,v_slot,k_now,v_now};
     g.op("strata_qsa_prep_f32",in,out,&a,sizeof(a));}
    // The block ranking.
    const auto iq_raw=t("dsel.iq_raw",DType::F32,{1,hi,di}),rope1=t("dsel.rope1",DType::F32,{1,c.rotary});
    g.reshape(g.find("qsa.index_query"),iq_raw);g.reshape(rope,rope1);
    const auto iq_gamma=t("sel.iq_gamma",DType::F32,{di*index_gamma_heads},true),iq=t("dsel.iq",DType::F32,{1,hi,di});
    {abi::QsaIndexPrep a{int32_t(di),int32_t(hi),int32_t(c.rotary),c.epsilon,1,int32_t(index_gamma_heads)};
     const TensorId in[]={iq_raw,iq_gamma,rope1},out[]={iq};g.op("strata_qsa_index_prep_f32",in,out,&a,sizeof(a));}
    const auto iq2=t("dsel.iq2",DType::F32,{hi,di}),pooled=t("dsel.pooled",DType::F32,{B,di},true),scores=t("dsel.scores",DType::F32,{hi,B});
    g.reshape(iq,iq2);g.gemm(iq2,pooled,scores,true);
    const auto cell=t("dsel.cell",DType::F32,{1},true),block_scores=t("dsel.block_scores",DType::F32,{B});
    {abi::QsaDecodeBlocks a{int32_t(B),int32_t(hi),int32_t(c.block)};
     const TensorId in[]={scores,cell},out[]={block_scores};g.op("strata_qsa_decode_blocks_f32",in,out,&a,sizeof(a));}
    const auto top_scores=t("dsel.top_scores",DType::F32,{top}),top_ids=t("dsel.top_ids",DType::I32,{top});
    {ns_TopkNode::Params q{};q.bsw=uint32_t(top);q.axis=0;q.bottomK=0;const TensorId in[]={block_scores},outs[]={top_scores,top_ids};g.op("topk",in,outs,&q,sizeof(q));}
    // The selected cells, dequantized, behind the token's own key/value.
    const uint64_t cap=c.capacity;
    const auto kc=t("dev.k_cache",DType::F8,{cap,hk,d},true),vc=t("dev.v_cache",DType::F8,{cap,hk,d},true);
    const auto ks=t("dev.k_cache_scale",DType::F32,{cap,kQsaScaleSlots},true),vs=t("dev.v_cache_scale",DType::F32,{cap,kQsaScaleSlots},true);
    const auto k8=t("dsel.k8",DType::F8,{hk,n,d}),v8=t("dsel.v8",DType::F8,{hk,n,d}),kss=t("dsel.k_scale",DType::F32,{hk,n}),vss=t("dsel.v_scale",DType::F32,{hk,n});
    const auto gathered_mask=t("dsel.mask_cells",DType::F32,{n});
    {abi::QsaGatherBlocks a{int32_t(hk),int32_t(cap),int32_t(top),int32_t(c.block),int32_t(c.selection_width(c.capacity))};
     const TensorId in[]={kc,vc,ks,vs,top_ids,top_scores,cell},out[]={k8,v8,kss,vss,gathered_mask};
     g.op("strata_qsa_gather_blocks_fp8",in,out,&a,sizeof(a));}
    const auto stack=[&](TensorId now,TensorId codes,TensorId scale,const std::string& stem) {
        const auto values=t(stem+"_values",DType::F32,{hk,n,d}),scale3=t(stem+"_scale3",DType::F32,{hk,n,1}),cells=t(stem+"_cells",DType::F32,{hk,n,d});
        g.cast(codes,values);g.reshape(scale,scale3);
        {const TensorId in[]={values,scale3},o[]={cells};g.op("mult_fwd_f32",in,o);}
        const auto current=t(stem+"_now3",DType::F32,{hk,1,d});g.reshape(now,current);
        const auto all=t(stem+"_all",DType::F32,{hk,n+1,d});const TensorId parts[]={current,cells};g.concat(parts,all,1);return all;
    };
    const auto key=stack(k_now,k8,kss,"dsel.key"),value=stack(v_now,v8,vss,"dsel.value");
    const auto zero=t("dsel.mask_now",DType::F32,{1}),mask=t("dsel.mask",DType::F32,{n+1});
    {ns_ConstantKernel::Params q{};q.constant.f=0.0f;const TensorId out[]={zero};g.op("constant_f32",{},out,&q,sizeof(q));}
    {const TensorId parts[]={zero,gathered_mask};g.concat(parts,mask,0);}
    g.alias("query",query);g.alias("key",key);g.alias("value",value);g.alias("mask",mask);g.alias("q_gate",gate);
    g.set_scope("att.");define_qsa_attention(g,c,uint32_t(n+1),false);g.set_scope({});
    g.alias("attention",g.find("att.out"));g.set_scope("o.");define_qsa_output(g,c,uint32_t(h),merged_layers());g.set_scope({});
    const auto block=t("attn_block",DType::F32,{h});g.reshape(g.find("o.out"),block);
    g.alias("residual",residual);g.alias("block",block);g.alias("inject",g.find("attn.inject"));
    g.set_scope("attn_write.");define_hc_write(g,hc);g.set_scope({});
    g.alias("residual",g.find("attn_write.out"));g.set_scope("ffn.");define_hc_read(g,hc,false,merged_layers(),true);g.set_scope({});
    const auto ffn_row=t("ffn_row",DType::F32,{1,h});
    g.reshape(g.find("ffn.mixed"),ffn_row);const auto ffn_x=hc_read_bf16(g,"ffn.",ffn_row,"ffn_x",h);
    g.alias("x",ffn_x);g.set_scope("router.");define_moe_router(g,geometry,1,router_merge_enabled());g.set_scope({});
    append_resident_ffn(g,geometry,hc);
    return g.compile("strata_fused_qsa_decode_select_"+std::to_string(blocks));
}
// STRATA_HPU_DFLASH verify: the FFN half of a window of `rows` tokens, as append_resident_ffn() for one token.
void append_resident_ffn_rows(GraphBuilder& g,const core::ModelGeometry& geometry,const HcConfig& hc,uint32_t rows) {
    const uint64_t h=hc.hidden,t=rows;
    if(window_mme_bucket && !building_window_ffn) {
        // STRATA_HPU_MOE_MME: hand the post-attention residual to the window FFN launch that follows.
        g.reshape(g.find("attn_write.out"),g.tensor({"ffn_write.out",DType::F32,{t,uint64_t(hc.streams),h}}));
        return;
    }
    g.alias("residual",g.find("attn_write.out"));g.set_scope("ffn.");define_hc_read_window(g,hc,rows,false,merged_layers());g.set_scope({});
    const auto ffn_x=hc_read_window_bf16(g,"ffn.","ffn_x",t,h);
    g.alias("x",ffn_x);g.set_scope("router.");define_moe_router(g,geometry,rows,router_merge_enabled());g.set_scope({});
    g.set_scope("moe.");
    const bool fold=moe_tail_fused_enabled() && rows>1;
    const auto block=window_mme_bucket
        ?define_resident_mme_moe(g,geometry,g.find("ffn.mixed"),g.find("router.logits"),g.find("router.shared_gate"),rows,fold,window_mme_bucket)
        :define_resident_moe(g,geometry,g.find("ffn.mixed"),g.find("router.logits"),g.find("router.shared_gate"),rows,fold);
    g.set_scope({});
    if(fold) {
        // STRATA_HPU_MOE_TAIL_FUSED: the route weighting and expert sum inside the HC write (tpc/hc_write_moe.c).
        const auto out=g.tensor({"ffn_write.out",DType::F32,{t,uint64_t(hc.streams),h}});
        abi::HcWriteMoe a{int32_t(h),int32_t(hc.streams),int32_t(rows),int32_t(geometry.n_expert_used+1)};
        if(window_mme_bucket && !moe_fp8_enabled()) {
            // The grouped MME MoE returns its BF16 routed rows; the shared rows and the slot places are read by name.
            const TensorId in[]={g.find("attn_write.out"),block,g.find("moe.shared_rows"),g.find("moe.place"),
                                 g.find("moe.route_weights"),g.find("ffn.inject")},o[]={out};
            g.op("strata_hc_write_moe_routed_f32",in,o,&a,sizeof(a));
            return;
        }
        const TensorId in[]={g.find("attn_write.out"),block,g.find("moe.route_weights"),g.find("ffn.inject")},o[]={out};
        g.op("strata_hc_write_moe_f32",in,o,&a,sizeof(a));
        return;
    }
    g.alias("residual",g.find("attn_write.out"));g.alias("block",block);g.alias("inject",g.find("ffn.inject"));
    g.set_scope("ffn_write.");define_hc_write_window(g,hc,rows);g.set_scope({});
}
// One GDN layer over a verify window [rows, streams, hidden] as one launch: HC read -> GDN window keeping every
// row's state (gdn.row_<i>_state_out / _conv_state_out) -> HC write -> FFN half. "residual" -> "ffn_write.out".
std::unique_ptr<Recipe> build_verify_gdn(const Device& device,const core::ModelGeometry& geometry,const HcConfig& hc,uint32_t rows,bool final_state=false) {
    GraphBuilder g(device);const uint64_t h=hc.hidden,r=hc.streams,t=rows;
    const auto residual=g.tensor({"residual",DType::F32,{t,r,h}});
    g.alias("residual",residual);g.set_scope("attn.");define_hc_read_window(g,hc,rows,false,merged_layers());g.set_scope({});
    const auto attn_x=hc_read_window_bf16(g,"attn.","attn_x",t,h);
    g.alias("x",attn_x);g.set_scope("gdn.");
    if(final_state) define_gdn_prefill_final(g,geometry,rows,merged_layers());
    else define_gdn_window(g,geometry,rows,true,merged_layers());
    g.set_scope({});
    const auto block=g.tensor({"attn_block",DType::F32,{t,h},false});g.cast(g.find("gdn.out"),block);
    g.alias("residual",residual);g.alias("block",block);g.alias("inject",g.find("attn.inject"));
    g.set_scope("attn_write.");define_hc_write_window(g,hc,rows,window_internal_enabled());g.set_scope({});
    append_resident_ffn_rows(g,geometry,hc,rows);
    return g.compile(std::string(final_state?"strata_prefill_final_gdn_":"strata_verify_gdn_")+std::to_string(rows),kChainWorkspace);
}
// One QSA layer over a verify window as one launch (identity selection): HC read -> projections -> qsa_prep for
// every row -> attention over [the cache's first `bucket` cells; the window's rows] with QsaBlock::verify_mask
// (dev.vmask [rows, 1, bucket+rows]) -> output -> HC write -> FFN half. The rows' K/V and raw index keys are
// persistent outputs (dev.k_now, dev.v_now, qsa.index_key) for the stage; the cache itself is not written here.
namespace {
void select_transpose(GraphBuilder& g,TensorId x,TensorId y,bool four) {
    synTransposeParams p{};p.tensorDim=four?4:3;p.permutation[0]=TPD_Channel;
    if(four) {p.permutation[1]=TPD_Width;p.permutation[2]=TPD_Depth;p.permutation[3]=TPD_Height;} // [a,b,c,d] -> [b,a,c,d]
    else {p.permutation[1]=TPD_Height;p.permutation[2]=TPD_Width;}                               // [a,b,c] -> [b,a,c]
    const TensorId in[]={x},out[]={y};g.op("transpose",in,out,&p,sizeof(p));
}
// [a, b, c] -> [a, c, b].
void inner_transpose(GraphBuilder& g,TensorId x,TensorId y) {
    synTransposeParams p{};p.tensorDim=3;p.permutation[0]=TPD_Width;p.permutation[1]=TPD_Channel;p.permutation[2]=TPD_Height;
    const TensorId in[]={x},out[]={y};g.op("transpose",in,out,&p,sizeof(p));
}
// [a, b, c, d] -> [c, b, a, d].
void outer_transpose(GraphBuilder& g,TensorId x,TensorId y) {
    synTransposeParams p{};p.tensorDim=4;p.permutation[0]=TPD_Channel;p.permutation[1]=TPD_Depth;p.permutation[2]=TPD_Height;p.permutation[3]=TPD_Width;
    const TensorId in[]={x},out[]={y};g.op("transpose",in,out,&p,sizeof(p));
}
// STRATA_HPU_QSA_SDPA: F32 keys or values per KV head {hk, K, d} -> BF16 per query head `out` {1, hq, K, d} (each KV
// head broadcast to its query group).
// The SDPA's KV head count: the query heads, or the KV heads themselves with STRATA_HPU_QSA_SDPA_GQA.
uint64_t sdpa_kv_heads(uint64_t hq,uint64_t hk) {return qsa_sdpa_gqa_enabled() || qsa_sdpa_fold_enabled()?hk:hq;}
void sdpa_repeat_kv(GraphBuilder& g,const std::string& stem,TensorId all,TensorId out,uint64_t hk,uint64_t group,uint64_t keys,uint64_t d) {
    // `all` is F32 (cast here) or already BF16.
    const bool narrow=g.desc(all).dtype==DType::BF16;
    const auto half=narrow?all:g.tensor({stem+"_half",DType::BF16,{hk,keys,d},false});
    if(!narrow) g.cast(all,half);
    if(qsa_sdpa_gqa_enabled() || qsa_sdpa_fold_enabled()) {g.reshape(half,out);return;} // out {1, hk, K, d}: the cast alone
    const auto one=g.tensor({stem+"_one",DType::BF16,{hk,1,keys,d},false}),wide=g.tensor({stem+"_wide",DType::BF16,{hk,group,keys,d},false});
    g.reshape(half,one);
    {const TensorId in[]={one},o[]={wide};g.op("broadcast",in,o);}
    g.reshape(wide,out);
}
// STRATA_HPU_QSA_SDPA: Synapse's fused BF16 SDPA (no materialized scores or probabilities). query {t, hq, d} F32
// already scaled, key/value {1, hq, K, d} BF16, mask {t, 1, K} F32 additive -> cf {t, hq, d} F32.
// folded: `mask` is already the folded SDPA's BF16 mask {t, group, K} (tpc/qsa_select_expand_fold.c).
void sdpa_attend(GraphBuilder& g,const std::string& stem,TensorId query,TensorId key,TensorId value,TensorId mask,TensorId cf,
                 uint64_t t,uint64_t hq,uint64_t keys,uint64_t d,bool folded=false) {
    const auto T=[&](const std::string& name,DType dtype,std::vector<uint64_t> shape) {return g.tensor({stem+name,dtype,std::move(shape),false});};
    ns_Sdpa::ParamsV2 a{};a.scale=1.0f;a.is_causal=false;a.dropout.ratio=0.0f;a.dropout.disableMaskOut=true;a.is_inference=true;
    a.softmax_mode=std::getenv("STRATA_HPU_QSA_SDPA_FAST")?SDPA_FAST_SOFTMAX:SDPA_DEFAULT_SOFTMAX;
    const char* guid=std::getenv("STRATA_HPU_QSA_SDPA_GUID")?std::getenv("STRATA_HPU_QSA_SDPA_GUID"):"sdpa_recomp_fwd_bf16";
    if(qsa_sdpa_fold_enabled() && qsa_sdpa_batch_enabled()) {
        // STRATA_HPU_QSA_SDPA_BATCH: q {t, hk, group, d} -> {group, hk, t, d} (the group is the batch), K/V and the
        // mask {1, 1, t, K} broadcast over it.
        const uint64_t hk=g.desc(key).shape[1],group=hq/hk;
        const auto q4=T("bq4",DType::F32,{t,hk,group,d}),qt=T("bqt",DType::F32,{group,hk,t,d}),q16=T("bq16",DType::BF16,{group,hk,t,d});
        g.reshape(query,q4);outer_transpose(g,q4,qt);g.cast(qt,q16);
        const auto m16=T("m16",DType::BF16,{t,1,keys}),m4=T("m4",DType::BF16,{1,1,t,keys});g.cast(mask,m16);g.reshape(m16,m4);
        const auto o4=T("bo4",DType::BF16,{group,hk,t,d}),of=T("bof",DType::F32,{group,hk,t,d}),ot=T("bot",DType::F32,{t,hk,group,d});
        {const TensorId in[]={q16,key,value,m4},out[]={o4};g.op(guid,in,out,&a,sizeof(a));}
        g.cast(o4,of);outer_transpose(g,of,ot);g.reshape(ot,cf);
        return;
    }
    if(qsa_sdpa_fold_enabled()) {
        // STRATA_HPU_QSA_SDPA_FOLD: per KV head, the query rows are (row, head of the group): q {t, hk, group, d} ->
        // {hk, t, group, d} -> {1, hk, t*group, d}; the mask row of (row, head) is the row's, so {t, 1, K} -> {t, group, K}.
        const uint64_t hk=g.desc(key).shape[1],group=hq/hk;
        const auto q4=T("fq4",DType::F32,{t,hk,group,d}),qt=T("fqt",DType::F32,{hk,t,group,d}),q16=T("fq16",DType::BF16,{hk,t,group,d});
        const auto qf=T("fq",DType::BF16,{1,hk,t*group,d});
        g.reshape(query,q4);select_transpose(g,q4,qt,true);g.cast(qt,q16);g.reshape(q16,qf);
        const auto mf=T("fm4",DType::BF16,{1,1,t*group,keys});
        if(folded) g.reshape(mask,mf);
        else {
            const auto m16=T("m16",DType::BF16,{t,1,keys}),wide=T("fm",DType::BF16,{t,group,keys});
            g.cast(mask,m16);{const TensorId in[]={m16},o[]={wide};g.op("broadcast",in,o);}g.reshape(wide,mf);
        }
        const auto o4=T("fo4",DType::BF16,{1,hk,t*group,d}),o5=T("fo5",DType::BF16,{hk,t,group,d}),of=T("fof",DType::F32,{hk,t,group,d});
        {const TensorId in[]={qf,key,value,mf},out[]={o4};g.op(guid,in,out,&a,sizeof(a));}
        const auto ot=T("fot",DType::F32,{t,hk,group,d});
        g.reshape(o4,o5);g.cast(o5,of);select_transpose(g,of,ot,true);g.reshape(ot,cf);
        return;
    }
    const auto qh=T("qh",DType::F32,{hq,t,d}),q16=T("q16",DType::BF16,{hq,t,d}),q4=T("q4",DType::BF16,{1,hq,t,d});
    select_transpose(g,query,qh,false);g.cast(qh,q16);g.reshape(q16,q4);
    const auto m16=T("m16",DType::BF16,{t,1,keys}),m4=T("m4",DType::BF16,{1,1,t,keys});g.cast(mask,m16);g.reshape(m16,m4);
    const auto o4=T("o4",DType::BF16,{1,hq,t,d}),o16=T("o16",DType::BF16,{hq,t,d}),of=T("of",DType::F32,{hq,t,d});
    {const TensorId in[]={q4,key,value,m4},out[]={o4};g.op(guid,in,out,&a,sizeof(a));}
    g.reshape(o4,o16);g.cast(o16,of);select_transpose(g,of,cf,false);
}
}
std::unique_ptr<Recipe> build_verify_qsa(const Device& device,const core::ModelGeometry& geometry,const QsaConfig& c,const HcConfig& hc,
                                         uint32_t bucket,uint32_t rows) {
    GraphBuilder g(device);
    const uint64_t h=hc.hidden,r=hc.streams,t=rows,hq=c.query_heads,hk=c.kv_heads,d=c.width,b=bucket,group=hq/hk,keys=b+t;
    const auto T=[&](const std::string& name,DType dtype,std::vector<uint64_t> shape,bool persistent=false) {return g.tensor({name,dtype,std::move(shape),persistent});};
    const auto transpose=[&](TensorId x,TensorId y,bool four) {
        synTransposeParams p{};p.tensorDim=four?4:3;p.permutation[0]=TPD_Channel;
        if(four) {p.permutation[1]=TPD_Width;p.permutation[2]=TPD_Depth;p.permutation[3]=TPD_Height;} // [a,b,c,d] -> [b,a,c,d]
        else {p.permutation[1]=TPD_Height;p.permutation[2]=TPD_Width;}                               // [a,b,c] -> [b,a,c]
        const TensorId in[]={x},out[]={y};g.op("transpose",in,out,&p,sizeof(p));
    };
    const auto residual=T("residual",DType::F32,{t,r,h},true);
    g.alias("residual",residual);g.set_scope("attn.");define_hc_read_window(g,hc,rows,false,merged_layers());g.set_scope({});
    const auto attn_x=hc_read_window_bf16(g,"attn.","attn_x",t,h);
    g.alias("x",attn_x);g.set_scope("qsa.");define_qsa_projections_window(g,c,uint32_t(h),rows,merged_layers(),true);g.set_scope({});
    const auto q_raw=T("dev.q_raw",DType::F32,{t,hq,d}),gate=T("dev.gate",DType::F32,{t,hq,d});
    {synSplitParams p{};p.axis=0;const TensorId in[]={g.find("qsa.q_gate")},out[]={q_raw,gate};g.op("split",in,out,&p,sizeof(p));}
    const auto q_gamma=T("dev.q_gamma",DType::F32,{d},true),k_gamma=T("dev.k_gamma",DType::F32,{d},true),rope=T("dev.rope",DType::F32,{t,c.rotary},true);
    const auto query=T("dev.query",DType::F32,{t,hq,d});
    const auto k_slot=T("dev.k_slot",DType::F32,{t,hk,d}),v_slot=T("dev.v_slot",DType::F32,{t,hk,d});
    const auto k_now=T("dev.k_now",DType::F32,{t,hk,d},true),v_now=T("dev.v_now",DType::F32,{t,hk,d},true);
    {abi::QsaPrep a{int32_t(d),int32_t(hq),int32_t(hk),int32_t(c.rotary),c.epsilon,int32_t(rows)};
     const TensorId in[]={q_raw,g.find("qsa.k"),g.find("qsa.v"),q_gamma,k_gamma,rope},out[]={query,k_slot,v_slot,k_now,v_now};
     g.op("strata_qsa_prep_f32",in,out,&a,sizeof(a));}
    // Keys/values per KV head: the cache's first `bucket` cells, then the window's rows.
    const auto stack=[&](TensorId now,const char* cache_name,const std::string& stem) {
        const auto cache=std::string(cache_name).starts_with("dev.")?qsa_cache_read(g,cache_name,b,hk,d):T(cache_name,DType::F32,{b,hk,d},true);
        const auto heads=T(stem+"_heads",DType::F32,{hk,b,d});transpose(cache,heads,false);
        const auto rows_heads=T(stem+"_rows",DType::F32,{hk,t,d});transpose(now,rows_heads,false);
        const auto all=T(stem+"_all",DType::F32,{hk,keys,d});const TensorId parts[]={heads,rows_heads};g.concat(parts,all,1);return all;
    };
    if(qsa_kv_fp8_enabled() && qsa_dense_window_enabled()) throw std::invalid_argument("STRATA_HPU_QSA_KV_FP8 does not cover STRATA_HPU_QSA_DENSE_WINDOW");
    // STRATA_HPU_QSA_DENSE_WINDOW: same shapes, same concatenation, different source for the committed cells. The
    // gathered path copies the selected cells per row (rows x bucket x width floats -- what caps the window width);
    // the dense path reads the shared cache's first `b` cells and transposes them. The window's own `t` rows still
    // come from the stage, because they are not in the cache until the window commits, so `key` remains
    // concat(committed cells, current rows) and everything downstream is untouched.
    TensorId key,value;
    if(qsa_dense_window_enabled()) {
        key=stack(k_now,"kv.cache_slice_k","dk.key");value=stack(v_now,"kv.cache_slice_v","dv.value");
    } else {
        key=stack(k_now,"dev.k_cache","dev.key");value=stack(v_now,"dev.v_cache","dev.value");
    }
    const auto scale=T("att.scale",DType::F32,{1},true);
    const auto mask=T("dev.vmask",DType::F32,{t,1,keys},true),cf=T("dev.cf",DType::F32,{t,hq,d});
    if(qsa_sdpa_enabled()) {
        const auto scaled_query=T("dev.scaled_query",DType::F32,{t,hq,d});g.multiply(query,scale,scaled_query);
        const auto key16=T("dev.key16",DType::BF16,{1,sdpa_kv_heads(hq,hk),keys,d}),value16=T("dev.value16",DType::BF16,{1,sdpa_kv_heads(hq,hk),keys,d});
        sdpa_repeat_kv(g,"dev.k16",key,key16,hk,group,keys,d);sdpa_repeat_kv(g,"dev.v16",value,value16,hk,group,keys,d);
        sdpa_attend(g,"dev.sdpa_",scaled_query,key16,value16,mask,cf,t,hq,keys,d);
    } else {
    const auto q4=T("dev.q4",DType::F32,{t,hk,group,d}),qt=T("dev.qt",DType::F32,{hk,t,group,d}),qg=T("dev.qg",DType::F32,{hk,t*group,d});
    g.reshape(query,q4);transpose(q4,qt,true);g.reshape(qt,qg);
    const auto scores=T("dev.scores",DType::F32,{hk,t*group,keys}),scaled=T("dev.scaled",DType::F32,{hk,t*group,keys});
    g.batch_gemm(qg,key,scores,true);g.multiply(scores,scale,scaled);
    const auto scaled4=T("dev.scaled4",DType::F32,{hk,t,group,keys}),masked4=T("dev.masked4",DType::F32,{hk,t,group,keys});
    g.reshape(scaled,scaled4);g.add(scaled4,mask,masked4);
    const auto masked=T("dev.masked",DType::F32,{hk,t*group,keys}),probability=T("dev.probability",DType::F32,{hk,t*group,keys});
    g.reshape(masked4,masked);g.softmax_last(masked,probability);
    const auto context=T("dev.context",DType::F32,{hk,t*group,d});g.batch_gemm(probability,value,context);
    const auto c4=T("dev.c4",DType::F32,{hk,t,group,d}),ct=T("dev.ct",DType::F32,{t,hk,group,d});
    g.reshape(context,c4);transpose(c4,ct,true);g.reshape(ct,cf);
    }
    const auto sig=T("dev.sig_gate",DType::F32,{t,hq,d}),attention=T("dev.attention",DType::F32,{t,hq,d});
    g.sigmoid(gate,sig);g.multiply(cf,sig,attention);
    g.alias("attention",attention);g.set_scope("o.");define_qsa_output_window(g,c,uint32_t(h),rows,merged_layers());g.set_scope({});
    g.alias("residual",residual);g.alias("block",g.find("o.out"));g.alias("inject",g.find("attn.inject"));
    g.set_scope("attn_write.");define_hc_write_window(g,hc,rows);g.set_scope({});
    append_resident_ffn_rows(g,geometry,hc,rows);
    return g.compile(std::string(qsa_sdpa_enabled()?"strata_verify_qsa_sdpa_":"strata_verify_qsa_")+std::to_string(bucket)+"_"+std::to_string(rows),kChainWorkspace);
}
// STRATA_HPU_QSA_DEVICE_SELECT: build_verify_qsa past the identity selection, as four recipes so no workspace
// grows with the keys (one graph over all row chunks kept every chunk's scores live: 7.5 GB at 32768 keys). The key
// positions are the cache's cells [0, keys): the pre recipe writes the window's rows into the cache first (sel.k_rows /
// sel.v_rows, bound at the window's cell), so blocks of cells stay blocks of positions; cells past a row's own are
// masked. The mask is QsaState::select done on the device at block granularity: the indexer query rows
// (tpc/qsa_index_prep.c), every block's key (tpc/qsa_index_cells.c, sel.pooled holding the host's complete blocks),
// one GEMM for the block scores, tpc/qsa_select_mask.c per row (kept cells per block) and tpc/qsa_select_expand.c.
//   pre (keys, rows): HC read, projections, prep, the window's K/V rows into the cache, indexer, scaled queries;
//   kv (keys): the keys/values per head for the attention (F32 per KV head, or BF16 per query head with SDPA);
//   chunk (keys, chunk rows): one row chunk's mask and attention, launched once per chunk at binding offsets
//     (sel.start {2} = window start cell, the chunk's first row);
//   post (rows): gate, output projection, HC write and the FFN rows.
// Their sel.* tensors are buffers the caller binds (SelectBuffers); all four share one workspace.
uint32_t qsa_select_chunk_rows(uint32_t rows,uint64_t keys,uint32_t group,uint32_t kv_heads) {
    uint32_t chunk=rows;
    // STRATA_HPU_QSA_SDPA materializes no attention scores: only the indexer's (index heads + mask + scratch per
    // key) bound the chunk, so far fewer, wider SDPA launches.
    const bool sdpa=qsa_select_sdpa(keys);
    // (STRATA_HPU_QSA_SDPA_FOLD's mask repeated per query of the group is 2 * group bytes a key more: 200 MB at 256K
    // keys and the 32 rows this bound leaves, so it does not shrink the chunk.)
    const uint64_t per_row=sdpa?keys*4*8:uint64_t(group)*kv_heads*keys*4;
    while(chunk>16 && chunk*per_row>(sdpa?(256ull<<20):(128ull<<20))) chunk/=2;
    return chunk;
}
std::unique_ptr<Recipe> build_select_qsa_pre(const Device& device,const QsaConfig& c,const HcConfig& hc,
                                             uint32_t keys,uint32_t rows,uint32_t index_gamma_heads) {
    GraphBuilder g(device);
    const uint64_t h=hc.hidden,r=hc.streams,t=rows,hq=c.query_heads,hk=c.kv_heads,d=c.width,K=keys,b=K-t;
    const uint64_t hi=c.index_heads,di=c.index_width,blk=c.block;
    if(keys<=rows || keys%64) throw std::invalid_argument("QSA select window geometry");
    const auto T=[&](const std::string& name,DType dtype,std::vector<uint64_t> shape,bool persistent=false) {return g.tensor({name,dtype,std::move(shape),persistent});};
    const auto residual=T("residual",DType::F32,{t,r,h},true);
    g.alias("residual",residual);g.set_scope("attn.");define_hc_read_window(g,hc,rows,false,merged_layers());g.set_scope({});
    const auto attn_x=hc_read_window_bf16(g,"attn.","attn_x",t,h);
    g.alias("x",attn_x);g.set_scope("qsa.");define_qsa_projections_window(g,c,uint32_t(h),rows,merged_layers(),true);g.set_scope({});
    const auto q_raw=T("dev.q_raw",DType::F32,{t,hq,d}),gate=T("dev.gate",DType::F32,{t,hq,d},true);
    {synSplitParams p{};p.axis=0;const TensorId in[]={g.find("qsa.q_gate")},out[]={q_raw,gate};g.op("split",in,out,&p,sizeof(p));}
    const auto q_gamma=T("dev.q_gamma",DType::F32,{d},true),k_gamma=T("dev.k_gamma",DType::F32,{d},true),rope=T("dev.rope",DType::F32,{t,c.rotary},true);
    const auto query=T("dev.query",DType::F32,{t,hq,d});
    const auto k_slot=T("dev.k_slot",DType::F32,{t,hk,d}),v_slot=T("dev.v_slot",DType::F32,{t,hk,d});
    const auto k_now=T("dev.k_now",DType::F32,{t,hk,d},true),v_now=T("dev.v_now",DType::F32,{t,hk,d},true);
    {abi::QsaPrep a{int32_t(d),int32_t(hq),int32_t(hk),int32_t(c.rotary),c.epsilon,int32_t(rows)};
     const TensorId in[]={q_raw,g.find("qsa.k"),g.find("qsa.v"),q_gamma,k_gamma,rope},out[]={query,k_slot,v_slot,k_now,v_now};
     g.op("strata_qsa_prep_f32",in,out,&a,sizeof(a));}
    // The indexer: query rows and every block's key (sel.start {1}: the window's first cell).
    const auto iq_gamma=T("sel.iq_gamma",DType::F32,{di*index_gamma_heads},true),ik_gamma=T("sel.ik_gamma",DType::F32,{di},true);
    const auto iq=T("sel.iq",DType::F32,{t,hi,di},true);
    {abi::QsaIndexPrep a{int32_t(di),int32_t(hi),int32_t(c.rotary),c.epsilon,int32_t(rows),int32_t(index_gamma_heads)};
     const TensorId in[]={g.find("qsa.index_query"),iq_gamma,rope},out[]={iq};g.op("strata_qsa_index_prep_f32",in,out,&a,sizeof(a));}
    const auto start=T("sel.start",DType::F32,{1},true),pooled=T("sel.pooled",DType::F32,{std::max<uint64_t>(b/blk,1),di},true);
    const auto tail=T("sel.tail",DType::F32,{blk,di},true),rope_all=T("sel.rope_all",DType::F32,{K,c.rotary},true);
    const auto cells=T("sel.cells",DType::F32,{K/blk,di},true);
    {abi::QsaIndexCells a{int32_t(di),int32_t(c.rotary),c.epsilon,int32_t(K),int32_t(rows),int32_t(blk)};
     const TensorId in[]={start,pooled,tail,g.find("qsa.index_key"),ik_gamma,rope_all},out[]={cells};
     g.op("strata_qsa_index_cells_f32",in,out,&a,sizeof(a));}
    // The window's K/V rows into the cache at its first cell (the kv recipe then reads cells [0, keys)).
    qsa_cache_write(g,"sel.k_rows",k_now,t,hk,d);qsa_cache_write(g,"sel.v_rows",v_now,t,hk,d);
    // The softmax scale on the queries (t x hq x d) instead of on every score (t x hq x keys).
    const auto scale=T("att.scale",DType::F32,{1},true),scaled_query=T("sel.query",DType::F32,{t,hq,d},true);
    g.multiply(query,scale,scaled_query);
    return g.compile("strata_select_qsa_pre_"+std::to_string(keys)+"_"+std::to_string(rows),kChainWorkspace);
}
std::unique_ptr<Recipe> build_select_qsa_kv(const Device& device,const QsaConfig& c,uint32_t keys) {
    GraphBuilder g(device);
    const uint64_t hq=c.query_heads,hk=c.kv_heads,d=c.width,K=keys;
    const auto T=[&](const std::string& name,DType dtype,std::vector<uint64_t> shape,bool persistent=false) {return g.tensor({name,dtype,std::move(shape),persistent});};
    for(const auto& [cache_name,stem,out_name]:{std::array<const char*,3>{"dev.k_cache","sel.k","sel.key"},std::array<const char*,3>{"dev.v_cache","sel.v","sel.value"}}) {
        if(!qsa_select_sdpa(K)) {select_transpose(g,qsa_cache_read(g,cache_name,K,hk,d),T(out_name,DType::F32,{hk,K,d},true),false);continue;}
        // STRATA_HPU_QSA_SDPA: BF16 per query head {1, hq, K, d} (each KV head broadcast to its group); an FP8 cache
        // dequantizes straight to BF16 (no F32 copy of every key).
        if(qsa_kv_fp8_enabled()) {
            const auto cache=qsa_cache_read(g,cache_name,K,hk,d,DType::BF16),heads=T(std::string(stem)+"_heads16",DType::BF16,{hk,K,d});
            select_transpose(g,cache,heads,false);
            sdpa_repeat_kv(g,stem,heads,T(out_name,DType::BF16,{1,sdpa_kv_heads(hq,hk),K,d},true),hk,hq/hk,K,d);
            continue;
        }
        const auto heads=T(std::string(stem)+"_heads",DType::F32,{hk,K,d});select_transpose(g,qsa_cache_read(g,cache_name,K,hk,d),heads,false);
        sdpa_repeat_kv(g,stem,heads,T(out_name,DType::BF16,{1,sdpa_kv_heads(hq,hk),K,d},true),hk,hq/hk,K,d);
    }
    return g.compile("strata_select_qsa_kv_"+std::to_string(keys),kChainWorkspace);
}
std::unique_ptr<Recipe> build_select_qsa_chunk(const Device& device,const QsaConfig& c,uint32_t keys,uint32_t window,uint32_t chunk) {
    GraphBuilder g(device);
    const uint64_t rc=chunk,hq=c.query_heads,hk=c.kv_heads,d=c.width,group=hq/hk,K=keys,hi=c.index_heads,di=c.index_width;
    const auto T=[&](const std::string& name,DType dtype,std::vector<uint64_t> shape,bool persistent=false) {return g.tensor({name,dtype,std::move(shape),persistent});};
    const uint64_t blk=c.block,KB=K/blk;
    const auto start=T("sel.start",DType::F32,{2},true),iq=T("sel.iq",DType::F32,{rc,hi,di},true),cells=T("sel.cells",DType::F32,{KB,di},true);
    // The rows' selection: block scores, the cells each block keeps, broadcast to the block's positions, the mask.
    const auto i2=T("sel.i2",DType::F32,{rc*hi,di}),s2=T("sel.s2",DType::F32,{rc*hi,KB}),s3=T("sel.s3",DType::F32,{rc,hi,KB});
    g.reshape(iq,i2);g.gemm(i2,cells,s2,true);g.reshape(s2,s3);
    const abi::QsaSelect a{int32_t(K),int32_t(rc),int32_t(window),int32_t(c.selection_width(c.capacity)),int32_t(blk),int32_t(hi)};
    // STRATA_HPU_QSA_SELECT_STATS (diagnostic): the kept counts are a bound output the caller reads back.
    const auto scratch=T("sel.scratch",DType::F32,{rc,KB}),kept=T("sel.kept",DType::F32,{rc,KB},performance_env("STRATA_HPU_QSA_SELECT_STATS"));
    {const TensorId in[]={start,s3},out[]={scratch,kept};g.op("strata_qsa_select_mask_f32",in,out,&a,sizeof(a));}
    const auto kept3=T("sel.kept3",DType::F32,{rc,KB,1}),wide=T("sel.kept_wide",DType::F32,{rc,KB,blk}),cell_kept=T("sel.cell_kept",DType::F32,{rc,K});
    g.reshape(kept,kept3);{const TensorId in[]={kept3},out[]={wide};g.op("broadcast",in,out);}g.reshape(wide,cell_kept);
    if(qsa_select_sdpa(K) && qsa_sdpa_fold_enabled() && K%128==0) {
        // STRATA_HPU_QSA_SDPA_FOLD: the folded SDPA's BF16 mask written directly (no F32 mask, cast and broadcast).
        const auto mask=T("sel.mask_fold",DType::BF16,{rc,group,K});
        {const abi::QsaSelectFold f{a.keys,a.rows,a.window,a.width,a.block,a.heads,int32_t(group)};
         const TensorId in[]={start,cell_kept},out[]={mask};g.op("strata_qsa_select_expand_fold_bf16",in,out,&f,sizeof(f));}
        const auto query=T("sel.query",DType::F32,{rc,hq,d},true),key=T("sel.key",DType::BF16,{1,sdpa_kv_heads(hq,hk),K,d},true),
                   value=T("sel.value",DType::BF16,{1,sdpa_kv_heads(hq,hk),K,d},true);
        sdpa_attend(g,"sel.sdpa_",query,key,value,mask,T("sel.cf",DType::F32,{rc,hq,d},true),rc,hq,K,d,true);
        return g.compile("strata_select_qsa_chunk_sdpa_"+std::to_string(keys)+"_"+std::to_string(window)+"_"+std::to_string(chunk),kChainWorkspace);
    }
    const auto mask=T("sel.mask",DType::F32,{rc,1,K});
    {const TensorId in[]={start,cell_kept},out[]={mask};g.op("strata_qsa_select_expand_f32",in,out,&a,sizeof(a));}
    if(qsa_select_sdpa(K)) {
        // Fused SDPA over every key: BF16 q {1, hq, rc, d} (already scaled), k/v {1, hq, K, d}, mask {1, 1, rc, K}.
        const auto query=T("sel.query",DType::F32,{rc,hq,d},true),key=T("sel.key",DType::BF16,{1,sdpa_kv_heads(hq,hk),K,d},true),
                   value=T("sel.value",DType::BF16,{1,sdpa_kv_heads(hq,hk),K,d},true);
        sdpa_attend(g,"sel.sdpa_",query,key,value,mask,T("sel.cf",DType::F32,{rc,hq,d},true),rc,hq,K,d);
        return g.compile("strata_select_qsa_chunk_sdpa_"+std::to_string(keys)+"_"+std::to_string(window)+"_"+std::to_string(chunk),kChainWorkspace);
    }
    const auto query=T("sel.query",DType::F32,{rc,hq,d},true),key=T("sel.key",DType::F32,{hk,K,d},true),value=T("sel.value",DType::F32,{hk,K,d},true);
    // Their attention over every key.
    const auto q4=T("sel.q4",DType::F32,{rc,hk,group,d}),qt=T("sel.qt",DType::F32,{hk,rc,group,d}),qg=T("sel.qg",DType::F32,{hk,rc*group,d});
    g.reshape(query,q4);select_transpose(g,q4,qt,true);g.reshape(qt,qg);
    const auto scaled=T("sel.scaled",DType::F32,{hk,rc*group,K});
    g.batch_gemm(qg,key,scaled,true);
    const auto scaled4=T("sel.scaled4",DType::F32,{hk,rc,group,K}),masked4=T("sel.masked4",DType::F32,{hk,rc,group,K});
    g.reshape(scaled,scaled4);g.add(scaled4,mask,masked4);
    const auto masked=T("sel.masked",DType::F32,{hk,rc*group,K}),probability=T("sel.probability",DType::F32,{hk,rc*group,K});
    g.reshape(masked4,masked);g.softmax_last(masked,probability);
    const auto context=T("sel.context",DType::F32,{hk,rc*group,d});g.batch_gemm(probability,value,context);
    const auto c4=T("sel.c4",DType::F32,{hk,rc,group,d}),ct=T("sel.ct",DType::F32,{rc,hk,group,d}),cf=T("sel.cf",DType::F32,{rc,hq,d},true);
    g.reshape(context,c4);select_transpose(g,c4,ct,true);g.reshape(ct,cf);
    return g.compile("strata_select_qsa_chunk_"+std::to_string(keys)+"_"+std::to_string(window)+"_"+std::to_string(chunk),kChainWorkspace);
}
// STRATA_HPU_QSA_SPARSE: a chunk's selection as bits for the host (tpc/qsa_select_bits.c) -- the chunk recipe's
// indexer scores and selection law, without the mask and the dense attention.
std::unique_ptr<Recipe> build_sparse_qsa_select(const Device& device,const QsaConfig& c,uint32_t keys,uint32_t window,uint32_t chunk) {
    GraphBuilder g(device);
    const uint64_t rc=chunk,K=keys,hi=c.index_heads,di=c.index_width,blk=c.block,KB=K/blk;
    const auto T=[&](const std::string& name,DType dtype,std::vector<uint64_t> shape,bool persistent=false) {return g.tensor({name,dtype,std::move(shape),persistent});};
    const auto start=T("sel.start",DType::F32,{2},true),iq=T("sel.iq",DType::F32,{rc,hi,di},true),cells=T("sel.cells",DType::F32,{KB,di},true);
    const auto i2=T("sel.i2",DType::F32,{rc*hi,di}),s2=T("sel.s2",DType::F32,{rc*hi,KB}),s3=T("sel.s3",DType::F32,{rc,hi,KB});
    g.reshape(iq,i2);g.gemm(i2,cells,s2,true);g.reshape(s2,s3);
    const abi::QsaSelect a{int32_t(K),int32_t(rc),int32_t(window),int32_t(c.selection_width(c.capacity)),int32_t(blk),int32_t(hi)};
    const auto scratch=T("sel.scratch",DType::F32,{rc,KB}),bits=T("sp.bits",DType::F32,{rc,KB/16},true),info=T("sp.info",DType::F32,{rc,2},true);
    {const TensorId in[]={start,s3},out[]={scratch,bits,info};g.op("strata_qsa_select_bits_f32",in,out,&a,sizeof(a));}
    return g.compile("strata_sparse_qsa_select_"+std::to_string(keys)+"_"+std::to_string(window)+"_"+std::to_string(chunk),kChainWorkspace);
}
// STRATA_HPU_QSA_SPARSE: a chunk's rows attend to their listed blocks only -- gathered from the FP8 cache
// (tpc/qsa_gather_rows_fp8.c), dequantized, then one attention per row and KV head over its `top`*block cells.
// query {rc, hq, d} is already scaled (sel.query); the result is the chunk's sel.cf rows.
std::unique_ptr<Recipe> build_sparse_qsa_attend(const Device& device,const QsaConfig& c,uint32_t chunk,uint32_t top) {
    GraphBuilder g(device);
    const uint64_t rc=chunk,hq=c.query_heads,hk=c.kv_heads,d=c.width,group=hq/hk,cap=c.capacity,n=uint64_t(top)*c.block;
    if(d!=256 || c.block!=4) throw std::invalid_argument("QSA sparse attention geometry");
    const auto T=[&](const std::string& name,DType dtype,std::vector<uint64_t> shape,bool persistent=false) {return g.tensor({name,dtype,std::move(shape),persistent});};
    const auto kc=T("dev.k_cache",DType::F8,{cap,hk,d},true),vc=T("dev.v_cache",DType::F8,{cap,hk,d},true);
    const auto ks=T("dev.k_cache_scale",DType::F32,{cap,kQsaScaleSlots},true),vs=T("dev.v_cache_scale",DType::F32,{cap,kQsaScaleSlots},true);
    const auto ids=T("sp.ids",DType::I32,{rc,top},true),info=T("sp.info",DType::F32,{rc,2},true),start=T("sel.start",DType::F32,{2},true);
    const auto k8=T("sp.k16",DType::BF16,{rc,hk,n,d}),v8=T("sp.v16",DType::BF16,{rc,hk,n,d});
    const auto kss=T("sp.k_scale",DType::F32,{rc,hk,n}),vss=T("sp.v_scale",DType::F32,{rc,hk,n}),mask=T("sp.mask",DType::F32,{rc,n});
    {const abi::QsaGatherRows a{int32_t(hk),int32_t(cap),int32_t(top),int32_t(c.block),int32_t(rc)};
     const TensorId in[]={kc,vc,ks,vs,ids,info,start},out[]={k8,v8,kss,vss,mask};g.op("strata_qsa_gather_rows_fp8",in,out,&a,sizeof(a));}
    // The gathered codes are BF16 in even/odd channel order: the query takes that order too, the context returns
    // from it. The key scale multiplies the scores, the value scale the probabilities (one per gathered cell).
    const auto key=T("sp.key",DType::BF16,{rc*hk,n,d}),value=T("sp.value",DType::BF16,{rc*hk,n,d});g.reshape(k8,key);g.reshape(v8,value);
    const auto query=T("sel.query",DType::F32,{rc,hq,d},true),qp=T("sp.q_pairs",DType::F32,{rc*hq,d/2,2}),qs=T("sp.q_split",DType::F32,{rc*hq,2,d/2});
    const auto q16=T("sp.q16",DType::BF16,{rc*hq,2,d/2}),q2=T("sp.q2",DType::BF16,{rc*hk,group,d});
    g.reshape(query,qp);inner_transpose(g,qp,qs);g.cast(qs,q16);g.reshape(q16,q2);
    const auto scores=T("sp.scores",DType::F32,{rc*hk,group,n});g.batch_gemm(q2,key,scores,true);
    const auto kscale=T("sp.kscale",DType::F32,{rc*hk,1,n}),vscale=T("sp.vscale",DType::F32,{rc*hk,1,n});g.reshape(kss,kscale);g.reshape(vss,vscale);
    const auto scaled=T("sp.scaled",DType::F32,{rc*hk,group,n});{const TensorId in[]={scores,kscale},o[]={scaled};g.op("mult_fwd_f32",in,o);}
    const auto mask4=T("sp.mask4",DType::F32,{rc,1,1,n});g.reshape(mask,mask4);
    const auto scores4=T("sp.scores4",DType::F32,{rc,hk,group,n}),masked4=T("sp.masked4",DType::F32,{rc,hk,group,n});
    g.reshape(scaled,scores4);g.add(scores4,mask4,masked4);
    const auto masked=T("sp.masked",DType::F32,{rc*hk,group,n}),probability=T("sp.probability",DType::F32,{rc*hk,group,n});
    g.reshape(masked4,masked);g.softmax_last(masked,probability);
    const auto weighted=T("sp.weighted",DType::F32,{rc*hk,group,n}),p16=T("sp.p16",DType::BF16,{rc*hk,group,n});
    {const TensorId in[]={probability,vscale},o[]={weighted};g.op("mult_fwd_f32",in,o);}g.cast(weighted,p16);
    const auto context=T("sp.context",DType::F32,{rc*hk,group,d}),cs=T("sp.c_split",DType::F32,{rc*hq,2,d/2}),cp=T("sp.c_pairs",DType::F32,{rc*hq,d/2,2});
    const auto cf=T("sel.cf",DType::F32,{rc,hq,d},true);
    g.batch_gemm(p16,value,context);g.reshape(context,cs);inner_transpose(g,cs,cp);g.reshape(cp,cf);
    return g.compile("strata_sparse_qsa_attend_"+std::to_string(chunk)+"_"+std::to_string(top),kChainWorkspace);
}
// STRATA_HPU_QSA_SPARSE: each row's kept blocks from tpc/qsa_select_bits.c's words (16 blocks a word, exact in F32),
// ascending, padded with -1 to `top`.
void sparse_qsa_ids(const float* bits,uint32_t rows,uint32_t words,uint32_t top,int32_t* ids) {
    for(uint32_t r=0;r<rows;++r) {
        uint32_t n=0;int32_t* out=ids+uint64_t(r)*top;const float* in=bits+uint64_t(r)*words;
        for(uint32_t w=0;w<words;++w) {
            uint32_t v=uint32_t(in[w]);
            while(v) {
                if(n==top) throw std::logic_error("QSA sparse row keeps more blocks than its list holds");
                out[n++]=int32_t(w*16+uint32_t(__builtin_ctz(v)));v&=v-1;
            }
        }
        for(;n<top;++n) out[n]=-1;
    }
}
std::unique_ptr<Recipe> build_select_qsa_post(const Device& device,const core::ModelGeometry& geometry,const QsaConfig& c,const HcConfig& hc,uint32_t rows) {
    GraphBuilder g(device);
    const uint64_t h=hc.hidden,r=hc.streams,t=rows,hq=c.query_heads,d=c.width;
    const auto T=[&](const std::string& name,DType dtype,std::vector<uint64_t> shape,bool persistent=false) {return g.tensor({name,dtype,std::move(shape),persistent});};
    const auto residual=T("residual",DType::F32,{t,r,h},true),inject=T("attn.inject",DType::F32,{t,r},true);
    const auto cf=T("sel.cf",DType::F32,{t,hq,d},true),gate=T("dev.gate",DType::F32,{t,hq,d},true);
    const auto sig=T("dev.sig_gate",DType::F32,{t,hq,d}),attention=T("dev.attention",DType::F32,{t,hq,d});
    g.sigmoid(gate,sig);g.multiply(cf,sig,attention);
    g.alias("attention",attention);g.set_scope("o.");define_qsa_output_window(g,c,uint32_t(h),rows,merged_layers());g.set_scope({});
    g.alias("residual",residual);g.alias("block",g.find("o.out"));g.alias("inject",inject);
    g.set_scope("attn_write.");define_hc_write_window(g,hc,rows);g.set_scope({});
    append_resident_ffn_rows(g,geometry,hc,rows);
    return g.compile("strata_select_qsa_post_"+std::to_string(rows),kChainWorkspace);
}
// STRATA_HPU_QSA_DECODE_SELECT, small verify windows (a DFlash round) past the identity selection: every row ranks
// the complete committed blocks (tpc/qsa_window_blocks.c, a top-k per row) and gathers its cells from the FP8 cache
// (tpc/qsa_gather_window_fp8.c, with the committed part of the window's first block); it attends to those and to the
// window's rows up to its own (dvs.causal). Cost is independent of the context, unlike the dense select window.
std::unique_ptr<Recipe> build_verify_qsa_gather(const Device& device,const core::ModelGeometry& geometry,const QsaConfig& c,const HcConfig& hc,
                                                uint32_t blocks,uint32_t rows,uint32_t index_gamma_heads) {
    GraphBuilder g(device);
    const uint64_t h=hc.hidden,r=hc.streams,t=rows,hq=c.query_heads,hk=c.kv_heads,d=c.width,group=hq/hk,hi=c.index_heads,di=c.index_width;
    const uint64_t top=qsa_decode_top_blocks(c),n=(top+1)*c.block,B=blocks,cap=c.capacity,keys=t+n;
    if(d!=256 || B%64 || B<top || !rows || rows>8) throw std::invalid_argument("QSA verify gather geometry");
    const auto T=[&](const std::string& name,DType dtype,std::vector<uint64_t> shape,bool persistent=false) {return g.tensor({name,dtype,std::move(shape),persistent});};
    const auto residual=T("residual",DType::F32,{t,r,h},true);
    g.alias("residual",residual);g.set_scope("attn.");define_hc_read_window(g,hc,rows,false,merged_layers());g.set_scope({});
    const auto attn_x=hc_read_window_bf16(g,"attn.","attn_x",t,h);
    g.alias("x",attn_x);g.set_scope("qsa.");define_qsa_projections_window(g,c,uint32_t(h),rows,merged_layers(),true);g.set_scope({});
    const auto q_raw=T("dev.q_raw",DType::F32,{t,hq,d}),gate=T("dev.gate",DType::F32,{t,hq,d});
    {synSplitParams p{};p.axis=0;const TensorId in[]={g.find("qsa.q_gate")},out[]={q_raw,gate};g.op("split",in,out,&p,sizeof(p));}
    const auto q_gamma=T("dev.q_gamma",DType::F32,{d},true),k_gamma=T("dev.k_gamma",DType::F32,{d},true),rope=T("dev.rope",DType::F32,{t,c.rotary},true);
    const auto query=T("dev.query",DType::F32,{t,hq,d});
    const auto k_slot=T("dev.k_slot",DType::F32,{t,hk,d}),v_slot=T("dev.v_slot",DType::F32,{t,hk,d});
    const auto k_now=T("dev.k_now",DType::F32,{t,hk,d},true),v_now=T("dev.v_now",DType::F32,{t,hk,d},true);
    {abi::QsaPrep a{int32_t(d),int32_t(hq),int32_t(hk),int32_t(c.rotary),c.epsilon,int32_t(rows)};
     const TensorId in[]={q_raw,g.find("qsa.k"),g.find("qsa.v"),q_gamma,k_gamma,rope},out[]={query,k_slot,v_slot,k_now,v_now};
     g.op("strata_qsa_prep_f32",in,out,&a,sizeof(a));}
    // Each row's block ranking.
    const auto iq_gamma=T("sel.iq_gamma",DType::F32,{di*index_gamma_heads},true),iq=T("dvs.iq",DType::F32,{t,hi,di});
    {abi::QsaIndexPrep a{int32_t(di),int32_t(hi),int32_t(c.rotary),c.epsilon,int32_t(rows),int32_t(index_gamma_heads)};
     const TensorId in[]={g.find("qsa.index_query"),iq_gamma,rope},out[]={iq};g.op("strata_qsa_index_prep_f32",in,out,&a,sizeof(a));}
    const auto iq2=T("dvs.iq2",DType::F32,{t*hi,di}),pooled=T("dsel.pooled",DType::F32,{B,di},true);
    const auto s2=T("dvs.s2",DType::F32,{t*hi,B}),s3=T("dvs.s3",DType::F32,{t,hi,B});
    g.reshape(iq,iq2);g.gemm(iq2,pooled,s2,true);g.reshape(s2,s3);
    const auto cell=T("dsel.cell",DType::F32,{1},true),block_scores=T("dvs.block_scores",DType::F32,{t,B});
    {abi::QsaWindowBlocks a{int32_t(B),int32_t(hi),int32_t(c.block),int32_t(rows)};
     const TensorId in[]={s3,cell},out[]={block_scores};g.op("strata_qsa_window_blocks_f32",in,out,&a,sizeof(a));}
    const auto top_scores=T("dvs.top_scores",DType::F32,{t,top}),top_ids=T("dvs.top_ids",DType::I32,{t,top});
    {ns_TopkNode::Params q{};q.bsw=uint32_t(top);q.axis=0;q.bottomK=0;const TensorId in[]={block_scores},outs[]={top_scores,top_ids};g.op("topk",in,outs,&q,sizeof(q));}
    const auto kc=T("dev.k_cache",DType::F8,{cap,hk,d},true),vc=T("dev.v_cache",DType::F8,{cap,hk,d},true);
    const auto ks=T("dev.k_cache_scale",DType::F32,{cap,kQsaScaleSlots},true),vs=T("dev.v_cache_scale",DType::F32,{cap,kQsaScaleSlots},true);
    const auto kss=T("dvs.k_scale",DType::F32,{t,hk,n}),vss=T("dvs.v_scale",DType::F32,{t,hk,n}),gathered_mask=T("dvs.mask_cells",DType::F32,{t,n});
    const auto scale=T("att.scale",DType::F32,{1},true),scaled=T("dvs.scaled_query",DType::F32,{t,hq,d}),q2=T("dvs.q2",DType::F32,{t*hk,group,d});
    const auto causal=T("dvs.causal",DType::F32,{t,t},true),mask2=T("dvs.mask",DType::F32,{t,keys}),mask4=T("dvs.mask4",DType::F32,{t,1,1,keys});
    const auto cf=T("dvs.cf",DType::F32,{t,hq,d});
    const abi::QsaGatherWindow gather{int32_t(hk),int32_t(cap),int32_t(top),int32_t(c.block),int32_t(c.selection_width(c.capacity)),int32_t(rows)};
    // The window's own rows (every row sees them all; dvs.causal masks the later ones) as {t*hk, t, d} F32.
    const auto window_rows=[&](TensorId now,const std::string& stem) {
        const auto heads=T(stem+"_heads",DType::F32,{hk,t,d}),one=T(stem+"_one",DType::F32,{1,hk,t,d}),wide=T(stem+"_window",DType::F32,{t,hk,t,d});
        select_transpose(g,now,heads,false);g.reshape(heads,one);
        {const TensorId in[]={one},o[]={wide};g.op("broadcast",in,o);}
        return wide;
    };
    g.multiply(query,scale,scaled);g.reshape(scaled,q2);
    // (The mask concat follows the gather in each branch: a tensor read before its producer is added would be taken
    // for an input of the graph.)
    const auto concat_mask=[&] {const TensorId parts[]={causal,gathered_mask};g.concat(parts,mask2,1);};
    if(c.block==4 && qsa_verify_bf16_enabled()) {
        // STRATA_HPU_QSA_VERIFY_BF16: the cells as exact BF16 codes in even/odd channel order (the query takes that
        // order, the context returns from it), the key scale on their scores and the value scale on their
        // probabilities; the window's rows apart, in F32.
        const auto k16=T("dvs.k16",DType::BF16,{t,hk,n,d}),v16=T("dvs.v16",DType::BF16,{t,hk,n,d});
        // STRATA_HPU_QSA_VERIFY_CAST (diagnostic): the FP8 gather and the SDK's casts to BF16 (natural channel order).
        static const bool sdk_cast=performance_env("STRATA_HPU_QSA_VERIFY_CAST");
        if(sdk_cast) {
            const auto k8=T("dvs.k8",DType::F8,{t,hk,n,d}),v8=T("dvs.v8",DType::F8,{t,hk,n,d});
            {const TensorId in[]={kc,vc,ks,vs,top_ids,top_scores,cell},out[]={k8,v8,kss,vss,gathered_mask};
             g.op("strata_qsa_gather_window_fp8",in,out,&gather,sizeof(gather));}
            g.cast(k8,k16);g.cast(v8,v16);
        } else {const TensorId in[]={kc,vc,ks,vs,top_ids,top_scores,cell},out[]={k16,v16,kss,vss,gathered_mask};
         g.op("strata_qsa_gather_window_bf16",in,out,&gather,sizeof(gather));}
        concat_mask();
        const auto key_wide=window_rows(k_now,"dvs.key"),value_wide=window_rows(v_now,"dvs.value");
        const auto key_w=T("dvs.key_wflat",DType::F32,{t*hk,t,d}),value_w=T("dvs.value_wflat",DType::F32,{t*hk,t,d});
        g.reshape(key_wide,key_w);g.reshape(value_wide,value_w);
        const auto scores_w=T("dvs.scores_w",DType::F32,{t*hk,group,t});g.batch_gemm(q2,key_w,scores_w,true);
        const auto qp=T("dvs.q_pairs",DType::F32,{t*hq,d/2,2}),qs=T("dvs.q_split",DType::F32,{t*hq,2,d/2});
        const auto q16=T("dvs.q16",DType::BF16,{t*hq,2,d/2}),q16g=T("dvs.q16g",DType::BF16,{t*hk,group,d});
        if(sdk_cast) {const auto natural=T("dvs.q16n",DType::BF16,{t*hq,2,d/2});g.reshape(scaled,qs);g.cast(qs,q16);(void)qp;(void)natural;}
        else {g.reshape(scaled,qp);inner_transpose(g,qp,qs);g.cast(qs,q16);}
        g.reshape(q16,q16g);
        const auto key_c=T("dvs.key_c",DType::BF16,{t*hk,n,d}),value_c=T("dvs.value_c",DType::BF16,{t*hk,n,d});g.reshape(k16,key_c);g.reshape(v16,value_c);
        const auto scores_c=T("dvs.scores_c",DType::F32,{t*hk,group,n});
        static const bool split_query=performance_env("STRATA_HPU_QSA_VERIFY_QLO");
        if(split_query) {
            // STRATA_HPU_QSA_VERIFY_QLO: the query as BF16 high + low parts (two products), near its F32 precision.
            const auto back=T("dvs.q16_f32",DType::F32,{t*hq,2,d/2}),rest=T("dvs.q_rest",DType::F32,{t*hq,2,d/2});
            const auto lo=T("dvs.q16lo",DType::BF16,{t*hq,2,d/2}),log=T("dvs.q16log",DType::BF16,{t*hk,group,d});
            g.cast(q16,back);{const TensorId in[]={qs,back},o[]={rest};g.op("sub_fwd_f32",in,o);}g.cast(rest,lo);g.reshape(lo,log);
            const auto hi_scores=T("dvs.scores_hi",DType::F32,{t*hk,group,n}),lo_scores=T("dvs.scores_lo",DType::F32,{t*hk,group,n});
            g.batch_gemm(q16g,key_c,hi_scores,true);g.batch_gemm(log,key_c,lo_scores,true);g.add(hi_scores,lo_scores,scores_c);
        } else g.batch_gemm(q16g,key_c,scores_c,true);
        // Scales, mask and softmax in one node (tpc/qsa_verify_softmax.c).
        const auto kscale=T("dvs.kscale",DType::F32,{t*hk,n}),vscale=T("dvs.vscale",DType::F32,{t*hk,n});g.reshape(kss,kscale);g.reshape(vss,vscale);
        const auto p_w=T("dvs.p_w",DType::F32,{t*hk,group,t}),p16=T("dvs.p16",DType::BF16,{t*hk,group,n}),pf=T("dvs.p_f32",DType::F32,{t*hk,group,n});
        {const abi::QsaVerifySoftmax a{int32_t(t),int32_t(hk),int32_t(group),int32_t(n)};
         const auto parts=T("dvs.softmax_parts",DType::F32,{t*hk,group,2,(n+255)/256});
         {const TensorId in[]={scores_w,scores_c,kscale,mask2},out[]={parts};g.op("strata_qsa_verify_partials_f32",in,out,&a,sizeof(a));}
         const TensorId in[]={scores_w,scores_c,kscale,vscale,mask2,parts},out[]={p_w,p16,pf};
         g.op("strata_qsa_verify_softmax_f32",in,out,&a,sizeof(a));}
        const auto ctx_w=T("dvs.ctx_w",DType::F32,{t*hk,group,d});g.batch_gemm(p_w,value_w,ctx_w);
        const auto ctx_c=T("dvs.ctx_c",DType::F32,{t*hk,group,d}),cs=T("dvs.c_split",DType::F32,{t*hq,2,d/2}),cp=T("dvs.c_pairs",DType::F32,{t*hq,d/2,2});
        static const bool split_probability=performance_env("STRATA_HPU_QSA_VERIFY_PLO");
        if(split_probability) {
            // STRATA_HPU_QSA_VERIFY_PLO: the probabilities as BF16 high + low parts (two products).
            const auto hi=T("dvs.ctx_hi",DType::F32,{t*hk,group,d}),lo=T("dvs.ctx_lo",DType::F32,{t*hk,group,d});
            const auto back=T("dvs.p16_f32",DType::F32,{t*hk,group,n}),rest=T("dvs.p_rest",DType::F32,{t*hk,group,n}),p16lo=T("dvs.p16lo",DType::BF16,{t*hk,group,n});
            g.cast(p16,back);{const TensorId in[]={pf,back},o[]={rest};g.op("sub_fwd_f32",in,o);}g.cast(rest,p16lo);
            g.batch_gemm(p16,value_c,hi);g.batch_gemm(p16lo,value_c,lo);g.add(hi,lo,ctx_c);
        } else g.batch_gemm(p16,value_c,ctx_c);
        if(sdk_cast) g.reshape(ctx_c,cp); else {g.reshape(ctx_c,cs);inner_transpose(g,cs,cp);}
        const auto ctx_cn=T("dvs.ctx_cn",DType::F32,{t*hk,group,d}),context=T("dvs.context",DType::F32,{t*hk,group,d});
        g.reshape(cp,ctx_cn);g.add(ctx_w,ctx_cn,context);g.reshape(context,cf);
    } else {
        // The gathered cells per row, dequantized.
        const auto k8=T("dvs.k8",DType::F8,{t,hk,n,d}),v8=T("dvs.v8",DType::F8,{t,hk,n,d});
        {const TensorId in[]={kc,vc,ks,vs,top_ids,top_scores,cell},out[]={k8,v8,kss,vss,gathered_mask};
         g.op("strata_qsa_gather_window_fp8",in,out,&gather,sizeof(gather));}
        concat_mask();g.reshape(mask2,mask4);
        // Keys per row: the window's rows, then its cells.
        const auto stack=[&](TensorId now,TensorId codes,TensorId cell_scale,const std::string& stem) {
            const auto values=T(stem+"_values",DType::F32,{t,hk,n,d}),scale4=T(stem+"_scale4",DType::F32,{t,hk,n,1}),cells=T(stem+"_cells",DType::F32,{t,hk,n,d});
            g.cast(codes,values);g.reshape(cell_scale,scale4);
            {const TensorId in[]={values,scale4},o[]={cells};g.op("mult_fwd_f32",in,o);}
            const auto wide=window_rows(now,stem);
            const auto all=T(stem+"_all",DType::F32,{t,hk,keys,d}),flat=T(stem+"_flat",DType::F32,{t*hk,keys,d});
            const TensorId parts[]={wide,cells};g.concat(parts,all,2);g.reshape(all,flat);return flat;
        };
        const auto key=stack(k_now,k8,kss,"dvs.key"),value=stack(v_now,v8,vss,"dvs.value");
        const auto scores=T("dvs.scores",DType::F32,{t*hk,group,keys});g.batch_gemm(q2,key,scores,true);
        const auto scores4=T("dvs.scores4",DType::F32,{t,hk,group,keys}),masked4=T("dvs.masked4",DType::F32,{t,hk,group,keys});
        g.reshape(scores,scores4);g.add(scores4,mask4,masked4);
        const auto masked=T("dvs.masked",DType::F32,{t*hk,group,keys}),probability=T("dvs.probability",DType::F32,{t*hk,group,keys});
        g.reshape(masked4,masked);g.softmax_last(masked,probability);
        const auto context=T("dvs.context",DType::F32,{t*hk,group,d});
        g.batch_gemm(probability,value,context);g.reshape(context,cf);
    }
    const auto sig=T("dev.sig_gate",DType::F32,{t,hq,d}),attention=T("dev.attention",DType::F32,{t,hq,d});
    g.sigmoid(gate,sig);g.multiply(cf,sig,attention);
    g.alias("attention",attention);g.set_scope("o.");define_qsa_output_window(g,c,uint32_t(h),rows,merged_layers());g.set_scope({});
    g.alias("residual",residual);g.alias("block",g.find("o.out"));g.alias("inject",g.find("attn.inject"));
    g.set_scope("attn_write.");define_hc_write_window(g,hc,rows,window_internal_enabled());g.set_scope({});
    append_resident_ffn_rows(g,geometry,hc,rows);
    return g.compile("strata_verify_qsa_gather_"+std::to_string(blocks)+"_"+std::to_string(rows),kChainWorkspace);
}
struct QsaChunk {uint32_t first,rows,bucket;};
std::vector<QsaChunk> canonical_qsa_chunks(const QsaConfig& c,uint32_t cell,uint32_t rows) {
    if(rows<=32 || rows>kResidentWindowRows || rows%32) throw std::invalid_argument("canonical QSA window rows");
    std::vector<QsaChunk> chunks;
    for(uint32_t first=0;first<rows;first+=32) chunks.push_back({first,32,qsa_device_bucket(c,cell+first,32)});
    return chunks;
}
std::unique_ptr<Recipe> build_verify_qsa_canonical(const Device& device,const core::ModelGeometry& geometry,const QsaConfig& c,const HcConfig& hc,
                                         uint32_t bucket,uint32_t rows,std::span<const QsaChunk> chunks) {
    GraphBuilder g(device);
    const uint64_t h=hc.hidden,r=hc.streams,t=rows,hq=c.query_heads,hk=c.kv_heads,d=c.width,b=bucket,group=hq/hk;
    const auto T=[&](const std::string& name,DType dtype,std::vector<uint64_t> shape,bool persistent=false) {return g.tensor({name,dtype,std::move(shape),persistent});};
    const auto transpose=[&](TensorId x,TensorId y,bool four) {
        synTransposeParams p{};p.tensorDim=four?4:3;p.permutation[0]=TPD_Channel;
        if(four) {p.permutation[1]=TPD_Width;p.permutation[2]=TPD_Depth;p.permutation[3]=TPD_Height;} // [a,b,c,d] -> [b,a,c,d]
        else {p.permutation[1]=TPD_Height;p.permutation[2]=TPD_Width;}                               // [a,b,c] -> [b,a,c]
        const TensorId in[]={x},out[]={y};g.op("transpose",in,out,&p,sizeof(p));
    };
    const auto residual=T("residual",DType::F32,{t,r,h},true);
    g.alias("residual",residual);g.set_scope("attn.");define_hc_read_window(g,hc,rows,false,merged_layers());g.set_scope({});
    const auto attn_x=hc_read_window_bf16(g,"attn.","attn_x",t,h);
    g.alias("x",attn_x);g.set_scope("qsa.");define_qsa_projections_window(g,c,uint32_t(h),rows,merged_layers(),true);g.set_scope({});
    const auto q_raw=T("dev.q_raw",DType::F32,{t,hq,d}),gate=T("dev.gate",DType::F32,{t,hq,d});
    {synSplitParams p{};p.axis=0;const TensorId in[]={g.find("qsa.q_gate")},out[]={q_raw,gate};g.op("split",in,out,&p,sizeof(p));}
    const auto q_gamma=T("dev.q_gamma",DType::F32,{d},true),k_gamma=T("dev.k_gamma",DType::F32,{d},true),rope=T("dev.rope",DType::F32,{t,c.rotary},true);
    const auto query=T("dev.query",DType::F32,{t,hq,d});
    const auto k_slot=T("dev.k_slot",DType::F32,{t,hk,d}),v_slot=T("dev.v_slot",DType::F32,{t,hk,d});
    const auto k_now=T("dev.k_now",DType::F32,{t,hk,d},true),v_now=T("dev.v_now",DType::F32,{t,hk,d},true);
    {abi::QsaPrep a{int32_t(d),int32_t(hq),int32_t(hk),int32_t(c.rotary),c.epsilon,int32_t(rows)};
     const TensorId in[]={q_raw,g.find("qsa.k"),g.find("qsa.v"),q_gamma,k_gamma,rope},out[]={query,k_slot,v_slot,k_now,v_now};
     g.op("strata_qsa_prep_f32",in,out,&a,sizeof(a));}
    if(qsa_kv_fp8_enabled()) throw std::invalid_argument("STRATA_HPU_QSA_KV_FP8 does not cover the canonical QSA32 window");
    const auto cache_k=T("dev.k_cache",DType::F32,{b,hk,d},true),cache_v=T("dev.v_cache",DType::F32,{b,hk,d},true);
    const auto cell=T("dev.cell",DType::F32,{1},true);
    const auto q4=T("dev.q4",DType::F32,{t,hk,group,d}),qt=T("dev.qt",DType::F32,{hk,t,group,d}),qg=T("dev.qg",DType::F32,{hk,t*group,d});
    g.reshape(query,q4);transpose(q4,qt,true);g.reshape(qt,qg);
    std::vector<TensorId> queries,contexts;
    for(uint32_t i=0;i<chunks.size();++i) queries.push_back(T("chunk_q_"+std::to_string(i),DType::F32,{hk,uint64_t(chunks[i].rows)*group,d}));
    {synSplitParams sp{};sp.axis=1;const TensorId in[]={qg};g.op("split",in,queries,&sp,sizeof(sp));}
    const auto scale=T("att.scale",DType::F32,{1},true);
    for(uint32_t i=0;i<chunks.size();++i) {
        const auto chunk=chunks[i];const uint64_t cr=chunk.rows,keys=chunk.bucket+cr;const auto prefix="chunk_"+std::to_string(i)+"_";
        const auto key=T(prefix+"key",DType::F32,{hk,keys,d}),value=T(prefix+"value",DType::F32,{hk,keys,d});
        abi::QsaVirtual args{int32_t(d),int32_t(hk),int32_t(b),int32_t(rows),int32_t(chunk.first),int32_t(cr),int32_t(chunk.bucket)};
        {const TensorId in[]={cache_k,cache_v,k_now,v_now,cell},out[]={key,value};g.op("strata_qsa_virtual_f32",in,out,&args,sizeof(args));}
        const auto scores=T(prefix+"scores",DType::F32,{hk,cr*group,keys}),scaled=T(prefix+"scaled",DType::F32,{hk,cr*group,keys});
        g.batch_gemm(queries[i],key,scores,true);g.multiply(scores,scale,scaled);
        const auto scaled4=T(prefix+"scaled4",DType::F32,{hk,cr,group,keys}),masked4=T(prefix+"masked4",DType::F32,{hk,cr,group,keys});
        const auto mask=T("dev.vmask_"+std::to_string(i),DType::F32,{cr,1,keys},true);
        g.reshape(scaled,scaled4);g.add(scaled4,mask,masked4);
        const auto masked=T(prefix+"masked",DType::F32,{hk,cr*group,keys}),probability=T(prefix+"probability",DType::F32,{hk,cr*group,keys});
        g.reshape(masked4,masked);g.softmax_last(masked,probability);
        const auto context=T(prefix+"context",DType::F32,{hk,cr*group,d});g.batch_gemm(probability,value,context);contexts.push_back(context);
    }
    const auto context=T("dev.context",DType::F32,{hk,t*group,d});g.concat(contexts,context,1);
    const auto c4=T("dev.c4",DType::F32,{hk,t,group,d}),ct=T("dev.ct",DType::F32,{t,hk,group,d}),cf=T("dev.cf",DType::F32,{t,hq,d});
    g.reshape(context,c4);transpose(c4,ct,true);g.reshape(ct,cf);
    const auto sig=T("dev.sig_gate",DType::F32,{t,hq,d}),attention=T("dev.attention",DType::F32,{t,hq,d});
    g.sigmoid(gate,sig);g.multiply(cf,sig,attention);
    g.alias("attention",attention);g.set_scope("o.");define_qsa_output_window(g,c,uint32_t(h),rows,merged_layers());g.set_scope({});
    g.alias("residual",residual);g.alias("block",g.find("o.out"));g.alias("inject",g.find("attn.inject"));
    g.set_scope("attn_write.");define_hc_write_window(g,hc,rows);g.set_scope({});
    append_resident_ffn_rows(g,geometry,hc,rows);
    std::string name="strata_verify_qsa_canonical_"+std::to_string(rows);for(const auto& chunk:chunks) name+="_"+std::to_string(chunk.bucket);
    return g.compile(name);
}
// Long-context verify (the device cache only holds the identity selection): the window's projections and HC
// read, then - after QsaBlock::stage_rows() has done norms, RoPE, append and selection on the host - the
// attention window over the gathered rows, output, HC write and FFN half, all still `rows` at a time.
std::unique_ptr<Recipe> build_staged_qsa_project(const Device& device,const QsaConfig& c,const HcConfig& hc,uint32_t rows) {
    GraphBuilder g(device);const uint64_t h=hc.hidden,r=hc.streams,t=rows;
    const auto residual=g.tensor({"residual",DType::F32,{t,r,h}});
    g.alias("residual",residual);g.set_scope("attn.");define_hc_read_window(g,hc,rows,false,merged_layers());g.set_scope({});
    const auto attn_x=hc_read_window_bf16(g,"attn.","attn_x",t,h);
    g.alias("x",attn_x);g.set_scope("qsa.");define_qsa_projections_window(g,c,uint32_t(h),rows,merged_layers());g.set_scope({});
    return g.compile("strata_staged_qsa_project_"+std::to_string(rows),kChainWorkspace);
}
std::unique_ptr<Recipe> build_staged_qsa_finish(const Device& device,const core::ModelGeometry& geometry,const QsaConfig& c,
                                                const HcConfig& hc,uint32_t bucket,uint32_t rows,bool attention_only=false) {
    GraphBuilder g(device);const uint64_t h=hc.hidden,r=hc.streams,t=rows;
    g.set_scope("att.");define_qsa_attention_window(g,c,bucket,rows);g.set_scope({});
    if(qsa_device_gather_enabled())
        define_qsa_device_gather(g,c,bucket,rows,g.find("att.key"),g.find("att.value"),g.find("att.mask"),c.kv_heads);
    g.alias("attention",g.find("att.out"));g.set_scope("o.");define_qsa_output_window(g,c,uint32_t(h),rows,merged_layers());g.set_scope({});
    const auto residual=g.tensor({"residual",DType::F32,{t,r,h}}),inject=g.tensor({"inject",DType::F32,{t,r}});
    g.alias("residual",residual);g.alias("block",g.find("o.out"));g.alias("inject",inject);
    g.set_scope("attn_write.");define_hc_write_window(g,hc,rows);g.set_scope({});
    if(!attention_only) append_resident_ffn_rows(g,geometry,hc,rows);
    return g.compile(std::string(attention_only?"strata_staged_qsa_attention_":"strata_staged_qsa_finish_")+
                     std::to_string(bucket)+"_"+std::to_string(rows),kChainWorkspace);
}
std::unique_ptr<Recipe> build_staged_qsa_ffn(const Device& device,const core::ModelGeometry& geometry,
                                           const HcConfig& hc,uint32_t rows) {
    WindowFfnBuild ffn_graph;GraphBuilder g(device);
    g.tensor({"attn_write.out",DType::F32,{uint64_t(rows),uint64_t(hc.streams),uint64_t(hc.hidden)}});
    append_resident_ffn_rows(g,geometry,hc,rows);
    return g.compile((window_mme_bucket?"strata_window_ffn_mme_"+std::to_string(window_mme_bucket)+"_":std::string("strata_staged_qsa_ffn_"))+
                     std::to_string(rows),kChainWorkspace);
}
struct FusedQsa {
    std::mutex mutex;Device device;core::ModelGeometry geometry;QsaConfig config;HcConfig hc;
    std::unique_ptr<HostRecipe> project;std::map<uint32_t,std::unique_ptr<HostRecipe>> finish;
    // STRATA_HPU_CHAIN variants: the residual (and the finish graph's attention-half outputs) are device buffers.
    std::unique_ptr<HostRecipe> project_chain;std::map<uint32_t,std::unique_ptr<HostRecipe>> finish_chain;
    FusedQsa(const Device& d,core::ModelGeometry g,QsaConfig c,HcConfig h):device(d),geometry(g),config(std::move(c)),hc(h) {}
    HostRecipe& finisher(uint32_t bucket,std::span<const Binding> first) {
        auto& recipe=finish[bucket];
        if(!recipe) {
            recipe=std::make_unique<HostRecipe>(device,build_fused_qsa_finish(device,geometry,config,hc,bucket),first,
                std::vector<std::string>{"attn_write.out","ffn.mixed","ffn.inject","router.logits","router.shared_gate"});
            const std::array<float,1> scale{1/float(hc.streams)};recipe->set("ffn.inverse_streams",scale);
        }
        return *recipe;
    }
    std::map<uint32_t,std::unique_ptr<HostRecipe>> device_layer; // STRATA_HPU_QSA_DEVICE, one per bucket
    std::map<std::pair<uint32_t,uint32_t>,std::unique_ptr<HostRecipe>> verify; // per (bucket, rows)
    // STRATA_HPU_QSA_DEVICE_SELECT: the pre and chunk recipes per (keys, rows), kv per keys, post per rows, and the
    // buffers between them (every layer's select window uses them in turn on the chain stream).
    std::map<std::pair<uint32_t,uint32_t>,std::unique_ptr<HostRecipe>> select_pre,select_chunk;
    std::map<uint32_t,std::unique_ptr<HostRecipe>> select_post,select_kv;
    // STRATA_HPU_QSA_SPARSE: per (keys, window rows) the chunk selection to bits, per chunk the gathered attention.
    std::map<std::pair<uint32_t,uint32_t>,std::unique_ptr<HostRecipe>> sparse_select;std::unique_ptr<HostRecipe> sparse_attend;
    struct SparseSlot {std::unique_ptr<DeviceBuffer> bits,info,ids;std::unique_ptr<HostBuffer> bits_host,info_host,ids_host;std::unique_ptr<Event> read;};
    std::array<SparseSlot,2> sparse_slots;std::unique_ptr<DeviceBuffer> sparse_start;std::unique_ptr<HostBuffer> sparse_start_host;
    struct SelectBuffers {
        std::unique_ptr<DeviceBuffer> query,gate,iq,inject,cf,cells,key,value,start;std::unique_ptr<HostBuffer> start_host;
        std::unique_ptr<Event> start_uploaded;
        std::unique_ptr<DeviceBuffer> kept_debug;std::unique_ptr<HostBuffer> kept_host; // STRATA_HPU_QSA_SELECT_STATS
        uint32_t cell=~0u,chunk=0,rows=0;
    } select_buffers;
    std::map<std::vector<uint32_t>,std::unique_ptr<HostRecipe>> verify_canonical; // per canonical bucket pattern
    // Long-context verify: the projection half per row count and the finish half per (bucket, rows). The finish
    // recipe's attention window is its pinned/HBM payload, so its budget is priced from that geometry.
    std::map<uint32_t,std::unique_ptr<HostRecipe>> staged_project;
    std::map<std::pair<uint32_t,uint32_t>,std::unique_ptr<HostRecipe>> staged_finish;
    std::map<std::pair<uint32_t,uint32_t>,std::unique_ptr<HostRecipe>> staged_attention;
    std::map<uint32_t,std::unique_ptr<HostRecipe>> staged_ffn;
    HostRecipe& staged_attention_for(uint32_t bucket,uint32_t rows,std::span<const Binding> first) {
        auto& recipe=staged_attention[{bucket,rows}];
        if(!recipe) {
            const auto pinned=qsa_window_gather_pinned_bytes(config,rows,bucket)+uint64_t(rows)*hc.streams*4;
            recipe=std::make_unique<HostRecipe>(device,build_staged_qsa_finish(device,geometry,config,hc,bucket,rows,true),
                                               first,std::vector<std::string>{},pinned+(16ull<<20));
            const std::array<float,1> scale{1/std::sqrt(float(config.width))};recipe->set("att.scale",scale);
        }
        return *recipe;
    }
    HostRecipe& staged_ffn_for(uint32_t rows,std::span<const Binding> first) {
        auto& recipe=staged_ffn[window_key(rows)];
        if(!recipe) {
            auto graph=build_staged_qsa_ffn(device,geometry,hc,rows);
            const auto bound=window_mme_bucket?declared(*graph,first):std::vector<Binding>(first.begin(),first.end());
            recipe=std::make_unique<HostRecipe>(device,std::move(graph),bound,std::vector<std::string>{});
            const std::array<float,1> streams{1/float(hc.streams)};recipe->set("ffn.inverse_streams",streams);
        }
        return *recipe;
    }
    HostRecipe& staged_projection(uint32_t rows,std::span<const Binding> first) {
        auto& recipe=staged_project[rows];
        if(!recipe) {
            recipe=std::make_unique<HostRecipe>(device,build_staged_qsa_project(device,config,hc,rows),first,
                std::vector<std::string>{"qsa.q_gate","qsa.k","qsa.v","qsa.index_query","qsa.index_key","attn.inject"});
            const std::array<float,1> scale{1/float(hc.streams)};recipe->set("attn.inverse_streams",scale);
        }
        return *recipe;
    }
    HostRecipe& staged_finisher(uint32_t bucket,uint32_t rows,std::span<const Binding> first) {
        auto& recipe=staged_finish[{bucket,window_key(rows)}];
        if(!recipe) {
            const uint64_t pinned=qsa_host_budget_enabled() && qsa_device_gather_enabled()
                ?qsa_window_gather_pinned_bytes(config,rows,bucket)+uint64_t(rows)*hc.streams*4+4
                :qsa_window_attention_pinned_bytes(config,rows,bucket);
            auto graph=build_staged_qsa_finish(device,geometry,config,hc,bucket,rows);
            const auto bound=window_mme_bucket?declared(*graph,first):std::vector<Binding>(first.begin(),first.end());
            recipe=std::make_unique<HostRecipe>(device,std::move(graph),bound,std::vector<std::string>{},pinned+(16ull<<20));
            const std::array<float,1> streams{1/float(hc.streams)},scale{1/std::sqrt(float(config.width))};
            if(!window_mme_bucket) {recipe->set("ffn.inverse_streams",streams);}
            recipe->set("att.scale",scale);
        }
        return *recipe;
    }
    // STRATA_HPU_QSA_DECODE_SELECT: one fused layer recipe per pooled-row bucket, and a table of cell numbers (one
    // 128-byte row per cell holding the cell as F32) that dsel.cell binds into, so no per-token upload is needed.
    std::map<uint32_t,std::unique_ptr<HostRecipe>> decode_select;
    std::map<std::pair<uint32_t,uint32_t>,std::unique_ptr<HostRecipe>> window_gather; // per (blocks, rows)
    std::unique_ptr<DeviceBuffer> cell_table;
    const DeviceBuffer& cells_for(const Stream& stream) {
        if(!cell_table) {
            const uint64_t cells=config.capacity;
            cell_table=std::make_unique<DeviceBuffer>(device,cells*128);HostBuffer host(device,cells*128);
            auto* values=static_cast<float*>(host.data());
            for(uint64_t i=0;i<cells;++i) {std::fill(values+i*32,values+i*32+32,0.0f);values[i*32]=float(i);}
            upload(stream,*cell_table,host,cells*128);stream.synchronize();
        }
        return *cell_table;
    }
    HostRecipe& decode_select_for(uint32_t blocks,uint32_t gamma_heads,std::span<const Binding> first,uint32_t layer=0) {
        auto& recipe=decode_select[blocks|(layer<<24)];
        if(!recipe) {
            auto graph=build_fused_qsa_decode_select(device,geometry,config,hc,blocks,gamma_heads);
            recipe=std::make_unique<HostRecipe>(device,std::move(graph),declared(*graph,first),std::vector<std::string>{});
            const std::array<float,1> streams{1/float(hc.streams)},scale{1/std::sqrt(float(config.width))};
            recipe->set("attn.inverse_streams",streams);recipe->set("ffn.inverse_streams",streams);recipe->set("att.scale",scale);
        }
        return *recipe;
    }
    // `key`: the bucket, with STRATA_HPU_DECODE_RECIPE_PER_LAYER the layer index in its top byte.
    HostRecipe& device_layer_for(uint32_t key,std::span<const Binding> first) {
        auto& recipe=device_layer[key];
        if(!recipe) {
            recipe=std::make_unique<HostRecipe>(device,build_fused_qsa_device(device,geometry,config,hc,key&0xFFFFFFu),first,std::vector<std::string>{});
            const std::array<float,1> streams{1/float(hc.streams)},scale{1/std::sqrt(float(config.width))};
            recipe->set("attn.inverse_streams",streams);recipe->set("ffn.inverse_streams",streams);recipe->set("att.scale",scale);
        }
        return *recipe;
    }
    HostRecipe& finisher_chain(uint32_t bucket,std::span<const Binding> first) {
        auto& recipe=finish_chain[bucket];
        if(!recipe) {
            const bool resident=resident_iq4_enabled();
            recipe=std::make_unique<HostRecipe>(device,build_fused_qsa_finish(device,geometry,config,hc,bucket,resident),first,
                resident?std::vector<std::string>{}:std::vector<std::string>{"router.logits","router.shared_gate"});
            const std::array<float,1> scale{1/float(hc.streams)};recipe->set("ffn.inverse_streams",scale);
        }
        return *recipe;
    }
};
std::shared_ptr<FusedQsa> shared_fused_qsa(const Device& device,const core::ModelGeometry& geometry,const QsaConfig& c,const HcConfig& hc,
                                           std::span<const Binding> first_project) {
    static std::mutex registry_mutex;static std::map<const void*,std::weak_ptr<FusedQsa>> registry;
    std::lock_guard lock(registry_mutex);auto& entry=registry[device.state().get()];
    if(auto existing=entry.lock()) return existing;
    auto created=std::make_shared<FusedQsa>(device,geometry,c,hc);
    created->project=std::make_unique<HostRecipe>(device,build_fused_qsa_project(device,c,hc),first_project,
        std::vector<std::string>{"qsa.q_gate","qsa.k","qsa.v","qsa.index_query","qsa.index_key","attn.inject"});
    const std::array<float,1> scale{1/float(hc.streams)};created->project->set("attn.inverse_streams",scale);
    entry=created;return created;
}
// STRATA_HPU_MOE_MME: the window FFN graphs (HC read -> router -> grouped MME MoE -> HC write) per (rows, bucket),
// shared by every layer of a device; each launch binds that layer's weights.
struct WindowFfn {std::mutex mutex;std::map<std::pair<uint32_t,uint32_t>,std::unique_ptr<HostRecipe>> recipes;};
std::shared_ptr<WindowFfn> shared_window_ffn(const Device& device) {
    static std::mutex mutex;static std::map<const void*,std::weak_ptr<WindowFfn>> registry;
    std::lock_guard lock(mutex);auto& entry=registry[device.state().get()];
    if(auto existing=entry.lock()) return existing;
    auto created=std::make_shared<WindowFfn>();entry=created;return created;
}
// The FFN half of the window just written to chain.in() by a split layer graph: chain.in() -> chain.out().
// `holder` keeps the device's shared WindowFfn (and its compiled graphs) alive for the layer's lifetime.
void run_window_ffn(const Device& device,std::shared_ptr<WindowFfn>& holder,const core::ModelGeometry& geometry,const HcConfig& hc,
                    VerifyChain& chain,std::span<const Binding> weights) {
    if(!holder) holder=shared_window_ffn(device);
    auto& ffn=holder;std::lock_guard lock(ffn->mutex);
    const uint32_t rows=chain.rows;
    const Binding io[]={{"attn_write.out",&chain.in()},{"ffn_write.out",&chain.out()}};
    auto& recipe=ffn->recipes[{rows,window_mme_bucket}];
    if(!recipe) {
        auto graph=build_staged_qsa_ffn(device,geometry,hc,rows);
        auto first=declared(*graph,weights);first.insert(first.end(),std::begin(io),std::end(io));
        recipe=std::make_unique<HostRecipe>(device,std::move(graph),first,std::vector<std::string>{});
        const std::array<float,1> streams{1/float(hc.streams)};recipe->set("ffn.inverse_streams",streams);
        for(const auto& [name,values]:moe_mme_constants(geometry)) recipe->set("moe."+name,values);
    }
    // STRATA_HPU_MOE_MME_SYNC=1 (diagnostic): wait before and after the FFN launch and print the three spans.
    static const bool sync=performance_env("STRATA_HPU_MOE_MME_SYNC");
    using Clock=std::chrono::steady_clock;
    const auto t0=Clock::now();
    if(sync) chain.stream.synchronize();
    const auto t1=Clock::now();
    for(const auto& b:weights) if(recipe->declares(b.name)) recipe->bind_external(b);
    for(const auto& b:io) recipe->bind_external(b);
    recipe->launch(chain.stream);chain.advance();
    if(sync) {
        const auto t2=Clock::now();chain.stream.synchronize();const auto t3=Clock::now();
        const auto ms=[](Clock::time_point a,Clock::time_point b) {return std::chrono::duration<double,std::milli>(b-a).count();};
        std::fprintf(stderr,"strata-hpu: window ffn rows=%u attention-half %.3f ms, ffn host %.3f ms, ffn device %.3f ms\n",
                     rows,ms(t0,t1),ms(t1,t2),ms(t2,t3));
    }
}
}
struct Layer::Impl {
    core::ModelGeometry geometry;uint32_t index;LayerOptions options;uint64_t position=0;
    // Declare weights before users, so recipes are destroyed before bindings.
    std::unique_ptr<DenseWeights> weights;
    std::unique_ptr<HostRecipe> read_attention,read_ffn,write,ple;
    std::unique_ptr<PleWindow> ple_window;
    std::unique_ptr<GdnBlock> gdn;
    std::unique_ptr<QsaBlock> qsa;
    std::unique_ptr<MoeBlock> moe;
    std::vector<float> ple_history;
    // --fp8-experts fused GDN decode: the shared first-launch recipe and this layer's bindings for it.
    std::shared_ptr<FusedGdnPre> fused;std::vector<Binding> fused_bindings;HcConfig hc;
    std::shared_ptr<FusedQsa> fused_qsa;std::vector<Binding> qsa_project_bindings,qsa_finish_bindings;
    std::shared_ptr<WindowFfn> window_ffn; // STRATA_HPU_MOE_MME's window FFN graphs, shared by the device's layers
    std::optional<Device> device;std::unique_ptr<HostRecipe> ple_chain; // STRATA_HPU_CHAIN layer-1 PLE on device
    std::unique_ptr<ResidentLayerExperts> resident; // STRATA_HPU_RESIDENT_IQ4: this layer's experts in HBM
    std::shared_ptr<ResidentMoeWindow> resident_window;
    // STRATA_HPU_DFLASH verify: every window row's GDN states, and the layer-1 PLE window per row count.
    std::vector<std::unique_ptr<DeviceBuffer>> verify_states,verify_convs;std::map<uint32_t,std::unique_ptr<HostRecipe>> ple_verify;
    bool poisoned=false;
    uint64_t owner=next_layer_owner++;
    mutable std::mutex mutex;
    Impl(core::ModelGeometry g,uint32_t l,LayerOptions o):geometry(g),index(l),options(std::move(o)) {}
    std::vector<Binding> select(std::initializer_list<const char*> names,const std::string& prefix={}) const {
        std::vector<Binding> result;
        for(const auto* name:names) {auto b=weights->binding(prefix+name);b.name=name;result.push_back(std::move(b));}
        return result;
    }
    // select() for the merged graphs: a matrix with an FP8 copy (STRATA_HPU_FP8_DENSE) binds that copy under its
    // own name, plus its row scales as name+"_scale".
    std::vector<Binding> select_merged(std::initializer_list<const char*> names,const std::string& prefix={}) const {
        std::vector<Binding> result;
        for(const auto* name:names) {
            const std::string full=prefix+name;
            if(!weights->contains(full+"_f8")) {auto b=weights->binding(full);b.name=name;result.push_back(std::move(b));continue;}
            auto codes=weights->binding(full+"_f8");codes.name=name;result.push_back(std::move(codes));
            auto scales=weights->binding(full+"_f8_scale");scales.name=std::string(name)+"_scale";result.push_back(std::move(scales));
        }
        return result;
    }
};
Layer::Layer(const Device& device,WeightTable table,uint32_t layer,ExpertCache& cache,LayerOptions options) {
    const auto g=table.geometry();
    if(layer>=uint64_t(g.n_layers) || !options.context || options.prefill_rows>kResidentWindowRows || !options.staging_budget || !options.dense_payload_budget)
        throw std::invalid_argument("invalid layer index/options");
    HcConfig hc{uint32_t(g.n_embd),uint32_t(g.hc),uint32_t(g.hc_lr),float(g.rms_eps)};hc.validate();
    const uint64_t h=g.n_embd,r=g.hc,n=h*r,lr=g.hc_lr;
    const auto prefix="blk."+std::to_string(layer)+".";
    auto p=std::make_unique<Impl>(g,layer,options);p->device=device;
    p->weights=std::make_unique<DenseWeights>(device,options.dense_payload_budget,options.staging_budget);
    auto matrix=[&](const std::string& binding,const std::string& suffix,uint64_t columns,uint64_t rows) {
        p->weights->matrix(binding,table,prefix+suffix,columns,rows);
    };
    auto constant=[&](const std::string& binding,const std::string& suffix,uint64_t count) {
        p->weights->constant(binding,table,prefix+suffix,count);
    };
    for(const auto* half:{"hc_attn_","hc_ffn_"}) {
        const std::string stem(half);
        matrix(stem+"w_down",stem+"down.weight",n,lr);matrix(stem+"w_up",stem+"up.weight",lr,n);
        matrix(stem+"w_inject",stem+"inject.weight",n,r);constant(stem+"norm",stem+"norm.weight",n);
    }
    // STRATA_HPU_MERGED_GEMM: the matrices one input feeds become one buffer before any binding is taken.
    const bool merged=merged_layers() && options.fused_decode;
    if(merged) for(const std::string stem:{"hc_attn_","hc_ffn_"}) p->weights->fuse(stem+"w_down_inject",{stem+"w_down",stem+"w_inject"});
    // STRATA_HPU_FP8_DENSE: FP8 copies of the merged graphs' matrices (the BF16 ones stay for every other graph).
    const bool fp8=merged && fp8_dense_enabled();
    if(fp8) for(const std::string stem:{"hc_attn_","hc_ffn_"}) {p->weights->to_fp8(stem+"w_down_inject");p->weights->to_fp8(stem+"w_up");}
    p->read_attention=std::make_unique<HostRecipe>(device,build_hc_read(device,hc),p->select({"w_down","w_up","w_inject","norm"},"hc_attn_"),std::vector<std::string>{"mixed","inject"});
    p->read_ffn=std::make_unique<HostRecipe>(device,build_hc_read(device,hc),p->select({"w_down","w_up","w_inject","norm"},"hc_ffn_"),std::vector<std::string>{"mixed","inject"});
    const std::array<float,1> scale{1/float(r)};p->read_attention->set("inverse_streams",scale);p->read_ffn->set("inverse_streams",scale);
    p->write=std::make_unique<HostRecipe>(device,build_hc_write(device,hc),std::span<const Binding>{},std::vector<std::string>{"out"});
    if(core::is_qsa_layer(g,layer)) {
        auto c=qsa_config(g,options.context,options.page_size);c.kv=options.kv;c.position_base=options.position_base;c.rope=options.rope;c.positions=options.positions;c.validate();
        matrix("w_q_gate","attn_q.weight",h,uint64_t(g.n_head)*g.head_dim*2);
        matrix("w_k","attn_k.weight",h,uint64_t(g.n_head_kv)*g.head_dim);matrix("w_v","attn_v.weight",h,uint64_t(g.n_head_kv)*g.head_dim);
        matrix("w_index_query","indexer.q_proj.weight",h,uint64_t(g.idx_q_heads)*g.idx_key_dim);matrix("w_index_key","indexer.k_proj.weight",h,g.idx_key_dim);
        if(merged) p->weights->fuse("w_qkvi",{"w_q_gate","w_k","w_v","w_index_query","w_index_key"});
        matrix("w_out","attn_output.weight",uint64_t(g.n_head)*g.head_dim,h);
        if(fp8) {p->weights->to_fp8("w_qkvi");p->weights->to_fp8("w_out");}
        QsaWeights w;w.projections=p->select({"w_q_gate","w_k","w_v","w_index_query","w_index_key"});w.output=p->weights->binding("w_out");
        w.query_gamma=table.constant(prefix+"attn_q_norm.weight",g.head_dim);w.key_gamma=table.constant(prefix+"attn_k_norm.weight",g.head_dim);
        w.index_query_gamma=table.constant(prefix+"indexer.q_norm.weight",g.idx_key_dim);w.index_key_gamma=table.constant(prefix+"indexer.k_norm.weight",g.idx_key_dim);
        QsaHalfSink half;if(options.qsa_half_sink) half=[sink=options.qsa_half_sink,layer](const QsaHalfView& view){sink(layer,view);};
        p->qsa=std::make_unique<QsaBlock>(device,c,uint32_t(h),std::move(w),options.qsa_window,std::move(half));
    } else {
        matrix("w_qkv","attn_qkv.weight",h,g.ssm_conv_channels);matrix("w_gate","attn_gate.weight",h,g.ssm_value_dim);
        matrix("w_alpha","ssm_alpha.weight",h,g.ssm_v_heads);matrix("w_beta","ssm_beta.weight",h,g.ssm_v_heads);matrix("w_out","ssm_out.weight",g.ssm_value_dim,h);
        if(merged) p->weights->fuse("w_qkvzab",{"w_qkv","w_gate","w_alpha","w_beta"});
        if(fp8) {p->weights->to_fp8("w_qkvzab");p->weights->to_fp8("w_out");}
        constant("norm","ssm_norm.weight",g.ssm_state_size);constant("dt","ssm_dt.bias",g.ssm_v_heads);constant("ssm_a","ssm_a",g.ssm_v_heads);
        p->weights->transposed("conv_weights",table,prefix+"ssm_conv1d.weight",g.ssm_d_conv,g.ssm_conv_channels);
        p->gdn=std::make_unique<GdnBlock>(device,g,p->select({"w_qkv","w_gate","w_alpha","w_beta","w_out","conv_weights","norm","dt","ssm_a"}),options.gdn_window);
    }
    if(layer==1 && options.ple_enabled) {
        matrix("ple_w_key","ple_key.weight",h,n);matrix("ple_w_value","ple_value.weight",h,h);
        for(const auto* name:{"norm_key","norm_query","norm_conv"}) constant(std::string("ple_")+name,std::string("ple_")+name+".weight",n);
        p->weights->transposed("ple_conv_weights",table,prefix+"ple_conv1d.weight",4,n);
        p->ple=std::make_unique<HostRecipe>(device,build_ple_block(device,hc),p->select({"w_key","w_value","norm_key","norm_query","norm_conv","conv_weights"},"ple_"),std::vector<std::string>{"out","history_out"});
        p->ple_history.resize(9*n);
        if(options.prefill_rows && options.hc_window) p->ple_window=std::make_unique<PleWindow>(device,hc,p->select({"w_key","w_value","norm_key","norm_query","norm_conv","conv_weights"},"ple_"));
    }
    p->moe=std::make_unique<MoeBlock>(device,table,layer,cache,MoeOptions{options.cpu,options.hpu_load_budget,options.routing_sink});
    p->hc=hc;
    std::vector<Binding> resident_bindings;
    if(resident_iq4_enabled() && options.fused_decode) {
        p->resident=std::make_unique<ResidentLayerExperts>(device,table,layer,std::max(1u,std::thread::hardware_concurrency()));
        if(options.prefill_rows) p->resident_window=shared_resident_moe_window(device);
        for(auto binding:p->resident->bindings()) {binding.name="moe."+binding.name;resident_bindings.push_back(std::move(binding));}
    }
    if(options.fused_decode && p->gdn && !options.half_sink && p->moe->fused_supported()) {
        const auto add=[&](std::vector<Binding> bindings,const std::string& scope) {
            for(auto& binding:bindings) {binding.name=scope+binding.name;p->fused_bindings.push_back(std::move(binding));}
        };
        add(merged?p->select_merged({"w_down_inject","w_up","norm"},"hc_attn_"):p->select({"w_down","w_up","w_inject","norm"},"hc_attn_"),"attn.");
        add(merged?p->select_merged({"w_down_inject","w_up","norm"},"hc_ffn_"):p->select({"w_down","w_up","w_inject","norm"},"hc_ffn_"),"ffn.");
        add(merged?p->select_merged({"w_qkvzab","w_out","conv_weights","norm","dt","ssm_a"}):
                   p->select({"w_qkv","w_gate","w_alpha","w_beta","w_out","conv_weights","norm","dt","ssm_a"}),"gdn.");
        const auto router=p->moe->router_bindings();add(router_merge_enabled()?std::vector<Binding>{router[0]}:std::vector<Binding>{router.begin(),router.end()},"router.");
        p->fused_bindings.insert(p->fused_bindings.end(),resident_bindings.begin(),resident_bindings.end());
        // The recipe's first bindings are this layer's, with the GDN state read and written in place: no
        // launch happens here, and every call rebinds the state from GdnBlock::external_step().
        auto first=p->fused_bindings;
        for(auto state:p->gdn->external_step()) {state.name="gdn."+state.name;first.push_back(std::move(state));}
        p->fused=shared_fused_gdn_pre(device,g,hc,first);
    }
    if(options.fused_decode && p->qsa && !options.half_sink && !options.qsa_half_sink && p->moe->fused_supported()) {
        const auto add=[](std::vector<Binding>& into,std::vector<Binding> bindings,const std::string& scope) {
            for(auto& binding:bindings) {binding.name=scope+binding.name;into.push_back(std::move(binding));}
        };
        add(p->qsa_project_bindings,merged?p->select_merged({"w_down_inject","w_up","norm"},"hc_attn_"):p->select({"w_down","w_up","w_inject","norm"},"hc_attn_"),"attn.");
        add(p->qsa_project_bindings,merged?p->select_merged({"w_qkvi"}):p->qsa->projection_bindings(),"qsa.");
        add(p->qsa_finish_bindings,merged?p->select_merged({"w_out"}):std::vector<Binding>{p->qsa->output_binding()},"o.");
        add(p->qsa_finish_bindings,merged?p->select_merged({"w_down_inject","w_up","norm"},"hc_ffn_"):p->select({"w_down","w_up","w_inject","norm"},"hc_ffn_"),"ffn.");
        const auto router=p->moe->router_bindings();
        add(p->qsa_finish_bindings,router_merge_enabled()?std::vector<Binding>{router[0]}:std::vector<Binding>{router.begin(),router.end()},"router.");
        p->qsa_finish_bindings.insert(p->qsa_finish_bindings.end(),resident_bindings.begin(),resident_bindings.end());
        p->fused_qsa=shared_fused_qsa(device,g,p->qsa->config(),hc,p->qsa_project_bindings);
    }
    impl_=std::move(p);
}
Layer::~Layer()=default;
std::vector<float> Layer::decode(std::span<const float> residual,std::span<const float> embedding) {
    auto& p=*impl_;std::lock_guard lock(p.mutex);const auto& g=p.geometry;
    if(p.poisoned || p.position>=p.options.context || (p.ple?embedding.size()!=uint64_t(g.n_embd):!embedding.empty()))
        throw std::invalid_argument("layer state/capacity/PLE input mismatch");
    const auto gdn_point=p.gdn?p.gdn->checkpoint():GdnBlock::Checkpoint{};
    const auto qsa_point=p.qsa?p.qsa->checkpoint():QsaState::Checkpoint{};
    if(p.resident) throw std::logic_error("STRATA_HPU_RESIDENT_IQ4 decodes only through the layer chain");
    if(p.fused) {
        const auto finite=[](std::span<const float> values,const char* stage) {
            if(!all_finite(values)) throw NonFiniteOutput(std::string("non-finite layer values: ")+stage);
        };
        std::vector<float> current(residual.begin(),residual.end()),history;finite(current,"input");
        if(p.ple) {
            p.ple->set("hidden",current);p.ple->set("embedding",embedding);p.ple->set("history",p.ple_history);p.ple->run();
            current=p.ple->values("out");history=p.ple->values("history_out");finite(current,"PLE");finite(history,"PLE history");
        }
        std::vector<float> mid,x,inject,logits,gate;
        {
            auto& f=*p.fused;std::lock_guard fused_lock(f.mutex);
            for(const auto& binding:p.fused_bindings) f.recipe->bind_external(binding);
            for(auto state:p.gdn->external_step()) {state.name="gdn."+state.name;f.recipe->bind_external(state);}
            f.recipe->set("residual",current);f.recipe->run();
            mid=f.recipe->values("attn_write.out");x=f.recipe->values("ffn.mixed");inject=f.recipe->values("ffn.inject");
            logits=f.recipe->values("router.logits");gate=f.recipe->values("router.shared_gate");
        }
        finite(mid,"HC attention write");finite(x,"HC FFN mixed");finite(inject,"HC FFN inject");finite(logits,"router");
        p.gdn->commit_external();
        std::vector<float> result;
        try {result=p.moe->fused_ffn(logits,gate.at(0),x,mid,inject,p.hc);finite(result,"HC FFN write");}
        catch(...) {
            try {p.gdn->restore(gdn_point);}
            catch(const std::exception& error) {p.poisoned=true;throw SessionFatal(std::string("layer mixer rollback failed: ")+error.what());}
            throw;
        }
        if(p.ple) p.ple_history=std::move(history);
        ++p.position;return result;
    }
    if(p.fused_qsa) {
        const auto finite=[](std::span<const float> values,const char* stage) {
            if(!all_finite(values)) throw NonFiniteOutput(std::string("non-finite layer values: ")+stage);
        };
        std::vector<float> current(residual.begin(),residual.end()),history;finite(current,"input");
        if(p.ple) {
            p.ple->set("hidden",current);p.ple->set("embedding",embedding);p.ple->set("history",p.ple_history);p.ple->run();
            current=p.ple->values("out");history=p.ple->values("history_out");finite(current,"PLE");finite(history,"PLE history");
        }
        auto& f=*p.fused_qsa;std::lock_guard fused_lock(f.mutex);
        for(const auto& binding:p.qsa_project_bindings) f.project->bind_external(binding);
        f.project->set("residual",current);f.project->run();
        const auto q_gate=f.project->values("qsa.q_gate"),k=f.project->values("qsa.k"),v=f.project->values("qsa.v");
        const auto index_query=f.project->values("qsa.index_query"),index_key=f.project->values("qsa.index_key"),attn_inject=f.project->values("attn.inject");
        for(const auto* values:{&q_gate,&k,&v,&index_query,&index_key,&attn_inject}) finite(*values,"QSA projections");
        std::vector<float> result;
        try {
            auto a=p.qsa->fused_attend(q_gate,k,v,index_query,index_key);
            auto& finish=f.finisher(a.bucket,p.qsa_finish_bindings);
            for(const auto& binding:p.qsa_finish_bindings) finish.bind_external(binding);
            const std::array<float,1> scale{a.scale};
            finish.set("att.query",a.query);finish.set("att.q_gate",a.gate);finish.set("att.scale",scale);
            finish.set("residual",current);finish.set("inject",attn_inject);
            // The selected rows land in the attention recipe's pinned inputs: no multi-megabyte temporary
            // per token and no second copy.
            p.qsa->gather_selected(a.ids,a.bucket,finish.input("att.key"),finish.input("att.value"),finish.input("att.mask"));
            finish.run();
            const auto mid=finish.values("attn_write.out"),x=finish.values("ffn.mixed"),inject=finish.values("ffn.inject");
            const auto logits=finish.values("router.logits"),gate=finish.values("router.shared_gate");
            finite(mid,"HC attention write");finite(x,"HC FFN mixed");finite(inject,"HC FFN inject");finite(logits,"router");
            result=p.moe->fused_ffn(logits,gate.at(0),x,mid,inject,p.hc);finite(result,"HC FFN write");
        } catch(...) {
            try {p.qsa->restore(qsa_point);}
            catch(const std::exception& error) {p.poisoned=true;throw SessionFatal(std::string("layer mixer rollback failed: ")+error.what());}
            throw;
        }
        if(p.ple) p.ple_history=std::move(history);
        ++p.position;return result;
    }
    std::vector<float> pending_history;LayerOperations op;
    if(p.ple) op.ple=[&](std::span<const float> r) {
        p.ple->set("hidden",r);p.ple->set("embedding",embedding);p.ple->set("history",p.ple_history);p.ple->run();
        auto result=p.ple->values("out");pending_history=p.ple->values("history_out");
        if(!all_finite(pending_history)) throw std::runtime_error("non-finite PLE history");
        return result;
    };
    op.read=[&](std::span<const float> r,bool ffn) {
        auto& read=ffn?*p.read_ffn:*p.read_attention;read.set("residual",r);read.run();return HcReadResult{read.values("mixed"),read.values("inject")};
    };
    op.mixer=[&](std::span<const float> x) {return p.gdn?p.gdn->decode(x):p.qsa->decode(x);};
    op.moe=[&](std::span<const float> x) {return p.moe->run(x);};
    op.write=[&](std::span<const float> r,std::span<const float> b,std::span<const float> inject,bool ffn) {
        if(p.options.half_sink) {
            p.options.half_sink(p.index,ffn?HalfField::FfnBlock:HalfField::AttentionBlock,b);
            p.options.half_sink(p.index,ffn?HalfField::FfnInject:HalfField::AttentionInject,inject);
        }
        p.write->set("residual",r);p.write->set("block",b);p.write->set("inject",inject);p.write->run();return p.write->values("out");
    };
    op.rollback_mixer=[&] {
        try {if(p.gdn) p.gdn->restore(gdn_point);else p.qsa->restore(qsa_point);}
        catch(...) {p.poisoned=true;throw;}
    };
    auto result=execute_layer(residual,uint32_t(g.n_embd),uint32_t(g.hc),op);
    if(p.ple) p.ple_history=std::move(pending_history);
    ++p.position;return result;
}
bool Layer::chainable() const {const auto& p=*impl_;return bool(p.fused) || bool(p.fused_qsa);}
void Layer::decode_chain(DecodeChain& chain,std::span<const float> embedding) {
    auto& p=*impl_;std::lock_guard lock(p.mutex);const auto& g=p.geometry;
    if(p.poisoned || !(p.fused || p.fused_qsa) || p.position>=p.options.context || (p.ple?embedding.size()!=uint64_t(g.n_embd):!embedding.empty()))
        throw std::invalid_argument("layer chain state/capacity/PLE input mismatch");
    const auto finite=[](std::span<const float> values,const char* stage) {
        if(!all_finite(values)) throw NonFiniteOutput(std::string("non-finite layer values: ")+stage);
    };
    const auto gdn_point=p.gdn?p.gdn->checkpoint():GdnBlock::Checkpoint{};
    const auto qsa_point=p.qsa?p.qsa->checkpoint():QsaState::Checkpoint{};
    std::vector<float> history;bool mixer_changed=false;
    // The chain's device buffers that carry the attention half's outputs into the expert group.
    const std::array<Binding,3> half{{{"attn_write.out",&chain.mid},{"ffn.mixed",&chain.x},{"ffn.inject",&chain.inject}}};
    try {
        if(p.ple) {
            if(!p.ple_chain) {
                auto bound=p.select({"w_key","w_value","norm_key","norm_query","norm_conv","conv_weights"},"ple_");
                bound.push_back({"hidden",&chain.in()});bound.push_back({"out",&chain.out()});
                p.ple_chain=std::make_unique<HostRecipe>(*p.device,build_ple_block(*p.device,p.hc),bound,std::vector<std::string>{"history_out"});
            }
            p.ple_chain->bind_external({"hidden",&chain.in()});p.ple_chain->bind_external({"out",&chain.out()});
            p.ple_chain->set("embedding",embedding);p.ple_chain->set("history",p.ple_history);
            p.ple_chain->launch(chain.stream);p.ple_chain->wait();chain.leases.clear();
            history=p.ple_chain->values("history_out");finite(history,"PLE history");chain.advance();
        }
        std::vector<float> logits,gate;
        if(p.fused) {
            auto& f=*p.fused;std::lock_guard fused_lock(f.mutex);
            auto& decode=decode_recipe_per_layer()?f.chain_layers[p.index]:f.chain;
            if(!decode) {
                auto first=p.fused_bindings;
                for(auto state:p.gdn->external_step()) {state.name="gdn."+state.name;first.push_back(std::move(state));}
                first.push_back({"residual",&chain.in()});first.insert(first.end(),half.begin(),half.end());
                if(p.resident) first.push_back({"ffn_write.out",&chain.out()});
                decode=std::make_unique<HostRecipe>(*p.device,build_fused_gdn_pre(*p.device,g,p.hc,bool(p.resident)),first,
                    p.resident?std::vector<std::string>{}:std::vector<std::string>{"router.logits","router.shared_gate"});
                const std::array<float,1> scale{1/float(p.hc.streams)};decode->set("attn.inverse_streams",scale);decode->set("ffn.inverse_streams",scale);
            }
            for(const auto& binding:p.fused_bindings) decode->bind_external(binding);
            for(auto state:p.gdn->external_step()) {state.name="gdn."+state.name;decode->bind_external(state);}
            decode->bind_external({"residual",&chain.in()});for(const auto& binding:half) decode->bind_external(binding);
            if(p.resident) {
                // The whole layer is this one launch: nothing to wait for, the next layer queues behind it.
                decode->bind_external({"ffn_write.out",&chain.out()});decode->launch(chain.stream);
                p.gdn->commit_external();mixer_changed=true;
            } else {
            decode->launch(chain.stream);decode->wait();chain.leases.clear(); // the previous expert group has run
            logits=decode->values("router.logits");gate=decode->values("router.shared_gate");
            p.gdn->commit_external();mixer_changed=true;
            }
        } else if(p.resident && qsa_device_enabled() && p.qsa->device_ready()) {
            // The whole layer is one launch; the host state takes this token's rows after the token (chain.commits).
            auto& f=*p.fused_qsa;std::lock_guard fused_lock(f.mutex);
            auto step=p.qsa->device_step(chain.stream);
            const auto& c=f.config;const uint64_t row=uint64_t(c.kv_heads)*c.width*4,stride=(2*row+uint64_t(c.index_width)*4+127)/128*128;
            const uint64_t offset=uint64_t(p.index)*stride;
            if(offset+stride>chain.layer_stage.size()) throw std::logic_error("QSA device stage too small");
            auto io=step.bindings;
            io.push_back({"residual",&chain.in()});io.insert(io.end(),half.begin(),half.end());io.push_back({"ffn_write.out",&chain.out()});
            io.push_back({"dev.k_now",&chain.layer_stage,offset});io.push_back({"dev.v_now",&chain.layer_stage,offset+row});
            io.push_back({"qsa.index_key",&chain.layer_stage,offset+2*row});
            auto first=p.qsa_project_bindings;first.insert(first.end(),p.qsa_finish_bindings.begin(),p.qsa_finish_bindings.end());
            first.insert(first.end(),io.begin(),io.end());
            auto& layer=f.device_layer_for(step.bucket|(decode_recipe_per_layer()?uint32_t(p.index)<<24:0u),first);
            for(const auto& binding:first) layer.bind_external(binding);
            layer.launch(chain.stream);
            p.qsa->device_store(chain.stream,step.cell,chain.layer_stage,offset);
            const uint64_t values=uint64_t(c.kv_heads)*c.width,index=c.index_width;
            chain.commits.push_back([qsa=p.qsa.get(),offset,row,values,index](const uint8_t* stage) {
                const auto* k=reinterpret_cast<const float*>(stage+offset);const auto* v=reinterpret_cast<const float*>(stage+offset+row);
                const auto* key=reinterpret_cast<const float*>(stage+offset+2*row);
                qsa->device_commit({k,values},{v,values},{key,index});
            });
        } else if(p.resident && qsa_device_enabled() && p.qsa->device_decode_select_ready()) {
            // STRATA_HPU_QSA_DECODE_SELECT: past the identity selection the token still runs as one launch: the device
            // ranks the blocks and gathers the cells (no projection wait, no host indexer); the host state takes the
            // token's rows after it, as above.
            auto& f=*p.fused_qsa;std::lock_guard fused_lock(f.mutex);
            auto step=p.qsa->device_decode_select_step(chain.stream);
            const auto& c=f.config;const uint64_t row=uint64_t(c.kv_heads)*c.width*4,stride=(2*row+uint64_t(c.index_width)*4+127)/128*128;
            const uint64_t offset=uint64_t(p.index)*stride;
            if(offset+stride>chain.layer_stage.size()) throw std::logic_error("QSA device stage too small");
            auto io=step.bindings;
            io.push_back({"residual",&chain.in()});io.insert(io.end(),half.begin(),half.end());io.push_back({"ffn_write.out",&chain.out()});
            io.push_back({"dev.k_now",&chain.layer_stage,offset});io.push_back({"dev.v_now",&chain.layer_stage,offset+row});
            io.push_back({"qsa.index_key",&chain.layer_stage,offset+2*row});
            io.push_back({"dsel.cell",&f.cells_for(chain.stream),uint64_t(step.cell)*128});
            auto first=p.qsa_project_bindings;first.insert(first.end(),p.qsa_finish_bindings.begin(),p.qsa_finish_bindings.end());
            first.insert(first.end(),io.begin(),io.end());
            auto& layer=f.decode_select_for(step.bucket,p.qsa->index_query_gamma_heads(),first,decode_recipe_per_layer()?p.index:0);
            for(const auto& binding:first) if(layer.declares(binding.name)) layer.bind_external(binding);
            layer.launch(chain.stream);
            p.qsa->device_store(chain.stream,step.cell,chain.layer_stage,offset);
            const uint64_t values=uint64_t(c.kv_heads)*c.width,index=c.index_width;
            chain.commits.push_back([qsa=p.qsa.get(),offset,row,values,index](const uint8_t* stage) {
                const auto* k=reinterpret_cast<const float*>(stage+offset);const auto* v=reinterpret_cast<const float*>(stage+offset+row);
                const auto* key=reinterpret_cast<const float*>(stage+offset+2*row);
                qsa->device_commit({k,values},{v,values},{key,index});
            });
        } else {
            auto& f=*p.fused_qsa;std::lock_guard fused_lock(f.mutex);
            if(!f.project_chain) {
                auto first=p.qsa_project_bindings;first.push_back({"residual",&chain.in()});
                f.project_chain=std::make_unique<HostRecipe>(*p.device,build_fused_qsa_project(*p.device,f.config,f.hc),first,
                    std::vector<std::string>{"qsa.q_gate","qsa.k","qsa.v","qsa.index_query","qsa.index_key","attn.inject"});
                const std::array<float,1> scale{1/float(f.hc.streams)};f.project_chain->set("attn.inverse_streams",scale);
            }
            for(const auto& binding:p.qsa_project_bindings) f.project_chain->bind_external(binding);
            f.project_chain->bind_external({"residual",&chain.in()});
            f.project_chain->launch(chain.stream);f.project_chain->wait();chain.leases.clear();
            QsaPhaseTimer timer;
            const auto q_gate=f.project_chain->values("qsa.q_gate"),k=f.project_chain->values("qsa.k"),v=f.project_chain->values("qsa.v");
            const auto index_query=f.project_chain->values("qsa.index_query"),index_key=f.project_chain->values("qsa.index_key");
            const auto attn_inject=f.project_chain->values("attn.inject");
            for(const auto* values:{&q_gate,&k,&v,&index_query,&index_key,&attn_inject}) finite(*values,"QSA projections");
            timer.mark("chain.projection-copy-check");
            auto a=p.qsa->fused_attend(q_gate,k,v,index_query,index_key);mixer_changed=true;
            timer.mark("chain.attend-total");
            auto first=p.qsa_finish_bindings;first.push_back({"residual",&chain.in()});first.insert(first.end(),half.begin(),half.end());
            if(p.resident) first.push_back({"ffn_write.out",&chain.out()});
            if(p.resident && qsa_device_gather_enabled()) {
                auto kv=p.qsa->device_gather_bindings(chain.stream);first.insert(first.end(),kv.begin(),kv.end());
            }
            auto& finish=f.finisher_chain(a.bucket,first);
            if(p.resident && qsa_device_gather_enabled())
                for(const auto& binding:first) finish.bind_external(binding);
            for(const auto& binding:p.qsa_finish_bindings) finish.bind_external(binding);
            finish.bind_external({"residual",&chain.in()});for(const auto& binding:half) finish.bind_external(binding);
            if(p.resident) finish.bind_external({"ffn_write.out",&chain.out()});
            const std::array<float,1> scale{a.scale};
            finish.set("att.query",a.query);finish.set("att.q_gate",a.gate);finish.set("att.scale",scale);
            finish.set("inject",attn_inject);
            if(p.resident && qsa_device_gather_enabled()) {
                auto ids=finish.input("kv.ids");std::fill(ids.begin(),ids.end(),-1.0f);
                for(size_t i=0;i<a.ids.size();++i) ids[i]=float(a.ids[i]);
            } else p.qsa->gather_selected(a.ids,a.bucket,finish.input("att.key"),finish.input("att.value"),finish.input("att.mask"));
            timer.mark("chain.finish-bind-set");
            finish.launch(chain.stream);
            timer.mark("chain.finish-launch");
            if(!p.resident) {finish.wait();logits=finish.values("router.logits");gate=finish.values("router.shared_gate");}
        }
        if(p.resident) chain.advance();
        else {
            finite(logits,"router");
            p.moe->fused_ffn_chain(logits,gate.at(0),chain,p.hc);chain.advance();
        }
    } catch(...) {
        // Nothing of this layer may still be running when its mixer state is put back.
        try {chain.stream.synchronize();} catch(...) {}
        if(mixer_changed) {
            try {if(p.gdn) p.gdn->restore(gdn_point);else p.qsa->restore(qsa_point);}
            catch(const std::exception& error) {p.poisoned=true;throw SessionFatal(std::string("layer mixer rollback failed: ")+error.what());}
        }
        throw;
    }
    if(p.ple) p.ple_history=std::move(history);
    ++p.position;
}
uint64_t qsa_select_window_bytes(const QsaConfig& c,uint32_t rows,uint32_t context) {
    // The widest key bucket a context admits sizes the shared buffers and workspace (narrower buckets reuse them).
    const uint64_t keys=QsaBlock::device_select_keys(context>rows?context-rows:0,rows,context);
    const uint64_t t=rows,hq=c.query_heads,hk=c.kv_heads,group=hq/hk,hi=c.index_heads,di=c.index_width,d=c.width,blk=c.block;
    const bool sdpa=qsa_select_sdpa(keys);
    const uint64_t rc=qsa_select_chunk_rows(rows,keys,uint32_t(group),uint32_t(hk));
    // SelectBuffers: query, gate and context rows, indexer rows, block keys, and the attention's keys/values (BF16 per
    // query head with the SDPA, F32 per KV head without).
    const uint64_t buffers=checked_product(t,4*(3*hq*d+hi*di))+checked_product(keys/blk,di*4)+
                           checked_product(keys,d*(sdpa?2*sdpa_kv_heads(hq,hk)*2:2*hk*4));
    // One workspace for every select recipe: the widest of the kv recipe (the cache rows dequantized and transposed)
    // and a row chunk (its indexer scores and mask, plus the F32 scores without the SDPA).
    const uint64_t kv=checked_product(keys,hk*d*(sdpa?2*2:4*2));
    const uint64_t chunk=sdpa?checked_product(checked_product(rc,keys),4*(2*hi+4)):checked_product(checked_product(rc,keys),4*(2*hi+2+5*hk*group));
    // The four key buckets kept (layer.cpp trims the rest) each hold their pre recipe's own tensors (~0.2 GB).
    return buffers+std::max<uint64_t>({kv,chunk,256ull<<20})+checked_product(4,256ull<<20);
}
uint64_t verify_stage_stride(const core::ModelGeometry& g,uint32_t max_rows) {
    const uint64_t row=uint64_t(g.n_head_kv)*g.head_dim*4,index=uint64_t(g.idx_key_dim)*4;
    return (uint64_t(max_rows)*(2*row+index)+127)/128*128;
}
uint64_t verify_stage_bytes(const core::ModelGeometry& g,uint32_t max_rows) {
    uint64_t layers=0;for(int64_t i=0;i<g.n_layers;++i) if(core::is_qsa_layer(g,uint32_t(i))) ++layers;
    return layers*verify_stage_stride(g,max_rows);
}
uint64_t verify_stage_offset(const core::ModelGeometry& g,uint32_t layer,uint32_t max_rows) {
    uint64_t before=0;for(uint32_t i=0;i<layer;++i) if(core::is_qsa_layer(g,i)) ++before;
    return before*verify_stage_stride(g,max_rows);
}
bool Layer::verify_ready(uint32_t rows) const {
    const auto& p=*impl_;std::lock_guard lock(p.mutex);
    if(p.poisoned || !p.resident || !rows || rows>kResidentWindowRows || p.position+rows>p.options.context) return false;
    if(p.gdn) return bool(p.fused);
    // Beyond the device cache's identity selection the same window still runs: the host keeps the
    // norm/RoPE/append/select law and the device graphs run the projections and the attention half.
    return p.fused_qsa && qsa_device_enabled() && (p.qsa->device_verify_ready(rows) || p.qsa->staged_verify_ready(rows));
}
void Layer::verify_chain(VerifyChain& chain,std::span<const float> ple_rows) {
    auto& p=*impl_;std::lock_guard lock(p.mutex);const auto& g=p.geometry;const uint32_t rows=chain.rows;
    // Window recipes built here share the chain's buffers for their internal persistent tensors (one set for every
    // row count instead of one per recipe: ~1 GB per row count at 8192 rows otherwise).
    // STRATA_HPU_NO_SCRATCH_SCOPE=1 (diagnostic): every recipe keeps its own internal tensors.
    static const bool no_scope=performance_env("STRATA_HPU_NO_SCRATCH_SCOPE");
    RecipeScratchScope scratch(no_scope?std::span<const Binding>{}:std::span<const Binding>(chain.scratch));
    MoeWindowScope moe_window(chain.moe_mme_bucket);const bool split=chain.moe_mme_bucket!=0;
    const uint64_t h=g.n_embd,n=h*g.hc;
    if(p.poisoned || !p.resident || !rows || rows>chain.max_rows || p.position+rows>p.options.context || (p.ple?ple_rows.size()!=rows*h:!ple_rows.empty()))
        throw std::invalid_argument("layer verify state/capacity/PLE input mismatch");
    if(p.ple) {
        // Every row's PLE history goes to chain.ple_history (row r at r * 9 * n floats); the layer's own history
        // changes only at verify_commit(). A final-state prefill window commits all its rows, so it keeps only the
        // last row's history (at 0): 9 * n floats instead of a window's worth (3 GB at 8192 rows).
        const uint64_t history=uint64_t(9)*n*4;const bool final=chain.prefill_final;
        std::vector<Binding> io{{"hidden",&chain.in()},{"out",&chain.out()}};
        if(final) io.push_back({"history_out",&chain.ple_history,0});
        else for(uint32_t r=0;r<rows;++r) io.push_back({"row_"+std::to_string(r)+"_history_out",&chain.ple_history,uint64_t(r)*history});
        auto& recipe=p.ple_verify[rows|(final?0x80000000u:0u)];
        if(!recipe) {
            GraphBuilder gb(*p.device);define_ple_window(gb,p.hc,rows,!final,true);
            auto bound=p.select({"w_key","w_value","norm_key","norm_query","norm_conv","conv_weights"},"ple_");bound.insert(bound.end(),io.begin(),io.end());
            recipe=std::make_unique<HostRecipe>(*p.device,gb.compile("strata_ple_verify_"+std::to_string(rows)+(final?"_final":""),kChainWorkspace),bound,std::vector<std::string>{});
        }
        for(const auto& binding:io) recipe->bind_external(binding);
        recipe->set("embedding",ple_rows);recipe->set("history",p.ple_history);
        recipe->launch(chain.stream);chain.advance();
    }
    if(p.gdn) {
        if(!p.fused) throw std::logic_error("GDN verify needs the fused resident layer");
        auto& f=*p.fused;std::lock_guard fused_lock(f.mutex);
        const uint64_t sb=p.gdn->state_bytes(),cb=p.gdn->conv_bytes();
        // One buffer per row, so the accepted row's pair can become the block's state without a copy.
        while(p.verify_states.size()<(chain.prefill_final?1u:rows)) {
            p.verify_states.push_back(std::make_unique<DeviceBuffer>(*p.device,sb));p.verify_convs.push_back(std::make_unique<DeviceBuffer>(*p.device,cb));
        }
        std::vector<Binding> io;
        for(auto state:p.gdn->current_state()) {state.name="gdn."+state.name;io.push_back(std::move(state));}
        for(uint32_t r=0;r<(chain.prefill_final?1u:rows);++r) {
            const auto prefix=chain.prefill_final?std::string("gdn."):"gdn.row_"+std::to_string(r)+"_";
            io.push_back({prefix+"state_out",p.verify_states[r].get()});io.push_back({prefix+"conv_state_out",p.verify_convs[r].get()});
        }
        io.push_back({"residual",&chain.in()});io.push_back({"ffn_write.out",&chain.out()});
        // STRATA_HPU_MOE_MERGED: the window FFN built into this graph (WindowFfnBuild) instead of run_window_ffn.
        const bool merged=split && moe_merged_enabled();
        auto& recipe=(chain.prefill_final?f.prefill_final:f.verify)[window_key(rows)];
        if(!recipe) {
            auto first=p.fused_bindings;first.insert(first.end(),io.begin(),io.end());
            auto graph=[&] {
                if(!merged) return build_verify_gdn(*p.device,g,p.hc,rows,chain.prefill_final);
                WindowFfnBuild inline_ffn;return build_verify_gdn(*p.device,g,p.hc,rows,chain.prefill_final);
            }();
            if(split) first=declared(*graph,first);
            recipe=std::make_unique<HostRecipe>(*p.device,std::move(graph),first,std::vector<std::string>{});
            const std::array<float,1> scale{1/float(p.hc.streams)};recipe->set("attn.inverse_streams",scale);
            if(!split || merged) recipe->set("ffn.inverse_streams",scale);
            if(merged) for(const auto& [name,values]:moe_mme_constants(g)) recipe->set("moe."+name,values);
            if(chain.prefill_final && gdn_prefill_chunked(rows))
                for(const auto& [name,values]:gdn_chunked_constants()) recipe->set("gdn."+name,values);
        }
        static const bool sync=performance_env("STRATA_HPU_MOE_MME_SYNC");
        const auto b0=std::chrono::steady_clock::now();
        for(const auto& binding:p.fused_bindings) if(!split || recipe->declares(binding.name)) recipe->bind_external(binding);
        for(const auto& binding:io) recipe->bind_external(binding);
        const auto b1=std::chrono::steady_clock::now();
        recipe->launch(chain.stream);chain.advance();
        if(sync) {
            const auto b2=std::chrono::steady_clock::now();chain.stream.synchronize();const auto b3=std::chrono::steady_clock::now();
            const auto ms=[](auto a,auto b) {return std::chrono::duration<double,std::milli>(b-a).count();};
            std::fprintf(stderr,"strata-hpu: gdn layer graph binds %.3f ms, launch %.3f ms, device %.3f ms\n",ms(b0,b1),ms(b1,b2),ms(b2,b3));
        }
        if(split && !merged) run_window_ffn(*p.device,p.window_ffn,g,p.hc,chain,p.fused_bindings);
        return;
    }
    if(!p.fused_qsa || !qsa_device_enabled()) throw std::logic_error("QSA verify needs STRATA_HPU_QSA_DEVICE");
    auto& f=*p.fused_qsa;std::lock_guard fused_lock(f.mutex);
    if(chain.prefill_final && prefill_qsa32_enabled() && rows>32 && p.qsa->device_verify_ready(32) && !p.qsa->device_verify_ready(rows))
        throw std::logic_error("canonical QSA32 needs a window split at the device/host boundary");
    if(!p.qsa->device_verify_ready(rows) && p.qsa->device_decode_select_ready(rows) && rows>1) {
        // STRATA_HPU_QSA_DECODE_SELECT: a small window (a DFlash round) gathers each row's selected cells
        // (build_verify_qsa_gather) instead of attending densely over the whole cache.
        auto step=p.qsa->device_decode_select_step(chain.stream,rows);
        const auto& c=f.config;
        const uint64_t row=uint64_t(c.kv_heads)*c.width*4,stride=verify_stage_stride(g,chain.max_rows),offset=verify_stage_offset(g,p.index,chain.max_rows);
        if(offset+stride>chain.stage.size()) throw std::logic_error("QSA verify stage too small");
        auto io=step.bindings;
        io.push_back({"residual",&chain.in()});io.push_back({"ffn_write.out",&chain.out()});
        io.push_back({"dev.k_now",&chain.stage,offset});io.push_back({"dev.v_now",&chain.stage,offset+rows*row});
        io.push_back({"qsa.index_key",&chain.stage,offset+2*rows*row});
        io.push_back({"dsel.cell",&f.cells_for(chain.stream),uint64_t(step.cell)*128});
        std::vector<Binding> unique;
        {
            auto first=p.qsa_project_bindings;first.insert(first.end(),p.qsa_finish_bindings.begin(),p.qsa_finish_bindings.end());
            first.insert(first.end(),io.begin(),io.end());
            std::vector<std::string> seen;
            for(const auto& b:first) {
                if(b.name=="kv.ids" || std::find(seen.begin(),seen.end(),b.name)!=seen.end()) continue;
                seen.push_back(b.name);unique.push_back(b);
            }
        }
        auto& recipe=f.window_gather[{step.bucket,window_key(rows)}];
        if(!recipe) {
            auto graph=build_verify_qsa_gather(*p.device,g,c,p.hc,step.bucket,rows,p.qsa->index_query_gamma_heads());
            recipe=std::make_unique<HostRecipe>(*p.device,std::move(graph),declared(*graph,unique),std::vector<std::string>{});
            const std::array<float,1> streams{1/float(p.hc.streams)},scale{1/std::sqrt(float(c.width))};
            recipe->set("attn.inverse_streams",streams);recipe->set("att.scale",scale);
            if(!split) recipe->set("ffn.inverse_streams",streams);
            std::vector<float> causal(uint64_t(rows)*rows,0.0f);
            for(uint32_t i=0;i<rows;++i) for(uint32_t j=i+1;j<rows;++j) causal[uint64_t(i)*rows+j]=-3.0e38f;
            recipe->set("dvs.causal",causal);
        }
        for(const auto& binding:unique) if(recipe->declares(binding.name)) recipe->bind_external(binding);
        recipe->launch(chain.stream);chain.advance();
        if(split) run_window_ffn(*p.device,p.window_ffn,g,p.hc,chain,p.qsa_finish_bindings);
        p.qsa->device_store_rows(chain.stream,step.cell,rows,chain.stage,offset);
        return;
    }
    if(!p.qsa->device_verify_ready(rows) && p.qsa->device_select_ready(rows)) {
        // STRATA_HPU_QSA_DEVICE_SELECT: past the identity selection, the window selects and attends on the device
        // (build_select_qsa_pre/_chunk/_post); its rows go to the stage and the cache exactly as the identity window's do.
        auto step=p.qsa->device_select_step(chain.stream,rows);
        const auto& c=f.config;
        const uint64_t row=uint64_t(c.kv_heads)*c.width*4,stride=verify_stage_stride(g,chain.max_rows),offset=verify_stage_offset(g,p.index,chain.max_rows);
        if(offset+stride>chain.stage.size()) throw std::logic_error("QSA verify stage too small");
        auto io=step.bindings;
        io.push_back({"residual",&chain.in()});io.push_back({"ffn_write.out",&chain.out()});
        io.push_back({"dev.k_now",&chain.stage,offset});io.push_back({"dev.v_now",&chain.stage,offset+rows*row});
        io.push_back({"qsa.index_key",&chain.stage,offset+2*rows*row});
        // The buffers between the three select recipes (grown with the keys; the device drains before they move).
        const uint64_t keys=step.bucket,hq=c.query_heads,hk=c.kv_heads,d=c.width,hi=c.index_heads,di=c.index_width;
        const uint32_t chunk=qsa_select_chunk_rows(rows,keys,uint32_t(hq/hk),uint32_t(hk)),chunks=rows/chunk;
        // STRATA_HPU_QSA_SPARSE: past the threshold the chunks attend to their rows' listed blocks (64-row chunks).
        const bool sparse=qsa_sparse_enabled() && qsa_kv_fp8_enabled() && keys>=qsa_sparse_min_keys() && c.block==4 && rows%64==0;
        auto& sb=f.select_buffers;
        {
            const auto fit=[&](std::unique_ptr<DeviceBuffer>& buffer,uint64_t bytes) {
                if(buffer && buffer->size()>=bytes) return;
                p.device->synchronize();buffer.reset();buffer=std::make_unique<DeviceBuffer>(*p.device,bytes);
            };
            const uint64_t window=uint64_t(rows)*hq*d*4;
            fit(sb.query,window);fit(sb.gate,window);fit(sb.cf,window);fit(sb.iq,uint64_t(rows)*hi*di*4);
            fit(sb.inject,uint64_t(rows)*f.hc.streams*4);fit(sb.cells,keys*di*4);
            if(!sparse) {const uint64_t kv=keys*d*std::max<uint64_t>(hk*4,qsa_select_sdpa(keys)?sdpa_kv_heads(hq,hk)*2:0);fit(sb.key,kv);fit(sb.value,kv);}
            // sel.start per chunk, 128 B apart: {window start cell, the chunk's first row}.
            if(!sb.start || sb.start->size()<uint64_t(chunks)*128) {
                p.device->synchronize();sb.start=std::make_unique<DeviceBuffer>(*p.device,uint64_t(chunks)*128);
                sb.start_host=std::make_unique<HostBuffer>(*p.device,uint64_t(chunks)*128);sb.cell=~0u;
            }
            if(sb.cell!=step.cell || sb.chunk!=chunk || sb.rows!=rows) {
                if(sb.start_uploaded && sb.start_uploaded->recorded()) sb.start_uploaded->synchronize();
                auto* host=static_cast<float*>(sb.start_host->data());
                for(uint32_t i=0;i<chunks;++i) {host[i*32]=float(step.cell);host[i*32+1]=float(i*chunk);}
                upload(chain.stream,*sb.start,*sb.start_host,uint64_t(chunks)*128);sb.cell=step.cell;sb.chunk=chunk;sb.rows=rows;
                if(!sb.start_uploaded) sb.start_uploaded=std::make_unique<Event>(*p.device);
                sb.start_uploaded->record(chain.stream);
            }
        }
        const Binding shared[]={{"sel.query",sb.query.get()},{"dev.gate",sb.gate.get()},{"sel.iq",sb.iq.get()},{"attn.inject",sb.inject.get()},
                                {"sel.cf",sb.cf.get()},{"sel.cells",sb.cells.get()},{"sel.key",sb.key.get()},{"sel.value",sb.value.get()},
                                {"sel.start",sb.start.get()}};
        io.insert(io.end(),std::begin(shared),std::end(shared));
        // The pre recipe writes the window's K/V rows into the cache at the window's cell (the key positions are cells).
        for(const auto& b:step.bindings) {
            const uint64_t at=uint64_t(step.cell)*qsa_cache_row_bytes(c),scale_at=uint64_t(step.cell)*qsa_cache_scale_bytes(c);
            if(b.name=="dev.k_cache") io.push_back({"sel.k_rows",b.buffer,at});
            if(b.name=="dev.v_cache") io.push_back({"sel.v_rows",b.buffer,at});
            if(b.name=="dev.k_cache_scale") io.push_back({"sel.k_rows_scale",b.buffer,scale_at});
            if(b.name=="dev.v_cache_scale") io.push_back({"sel.v_rows_scale",b.buffer,scale_at});
        }
        std::vector<Binding> unique;
        {
            auto first=p.qsa_project_bindings;first.insert(first.end(),p.qsa_finish_bindings.begin(),p.qsa_finish_bindings.end());
            first.insert(first.end(),io.begin(),io.end());
            std::vector<std::string> seen;
            for(const auto& b:first) {
                if(b.name=="kv.ids" || std::find(seen.begin(),seen.end(),b.name)!=seen.end()) continue;
                seen.push_back(b.name);unique.push_back(b);
            }
        }
        const std::array<float,1> streams{1/float(p.hc.streams)},scale{1/std::sqrt(float(c.width))};
        auto& pre=f.select_pre[{uint32_t(keys),window_key(rows)}];
        if(!pre) {
            auto graph=build_select_qsa_pre(*p.device,c,p.hc,uint32_t(keys),rows,p.qsa->index_query_gamma_heads());
            pre=std::make_unique<HostRecipe>(*p.device,std::move(graph),declared(*graph,unique),std::vector<std::string>{});
            pre->set("attn.inverse_streams",streams);pre->set("att.scale",scale);
        }
        auto& kv=f.select_kv[uint32_t(keys)];
        if(!kv && !sparse) {
            auto graph=build_select_qsa_kv(*p.device,c,uint32_t(keys));
            kv=std::make_unique<HostRecipe>(*p.device,std::move(graph),declared(*graph,unique),std::vector<std::string>{});
        }
        auto& part=f.select_chunk[{uint32_t(keys),rows}];
        static const bool stats=performance_env("STRATA_HPU_QSA_SELECT_STATS");
        if(stats) {
            const uint64_t bytes=uint64_t(chunk)*(keys/c.block)*4;
            if(!sb.kept_debug || sb.kept_debug->size()<bytes) {
                p.device->synchronize();
                sb.kept_debug=std::make_unique<DeviceBuffer>(*p.device,bytes);sb.kept_host=std::make_unique<HostBuffer>(*p.device,bytes);
            }
            unique.push_back({"sel.kept",sb.kept_debug.get()});
        }
        if(!part && !sparse) {
            auto graph=build_select_qsa_chunk(*p.device,c,uint32_t(keys),rows,chunk);
            part=std::make_unique<HostRecipe>(*p.device,std::move(graph),declared(*graph,unique),std::vector<std::string>{});
        }
        // Each key bucket's recipes hold their own tensors: only the four widest stay (prefill windows only grow the
        // key count; a shorter request recompiles its first windows' buckets) -- or, with STRATA_HPU_QSA_SELECT_KEEP_MIB,
        // the widest whose own HBM fits that budget (the narrowest are dropped first).
        const uint64_t keep=qsa_select_keep_bytes();
        const auto owned=[&] {
            uint64_t total=0;
            for(const auto& [k,r]:f.select_pre) total+=r?r->device_bytes():0;
            for(const auto& [k,r]:f.select_chunk) total+=r?r->device_bytes():0;
            for(const auto& [k,r]:f.select_kv) total+=r?r->device_bytes():0;
            for(const auto& [k,r]:f.sparse_select) total+=r?r->device_bytes():0;
            return total;
        };
        const auto over=[&](size_t count) {return keep?owned()>keep:count>4;};
        const auto trim=[&](auto& recipes) {
            while(recipes.size()>1 && over(recipes.size())) {
                auto smallest=recipes.begin();
                if(smallest->first.first==uint32_t(keys)) break;
                recipes.erase(smallest);
            }
        };
        trim(f.select_pre);trim(f.select_chunk);trim(f.sparse_select);
        // STRATA_HPU_QSA_KEEP_TRACE=1 (diagnostic): the select recipes' own HBM after the trim, per window.
        static const bool keep_trace=performance_env("STRATA_HPU_QSA_KEEP_TRACE");
        if(keep_trace && p.index==3) std::fprintf(stderr,"strata-hpu: qsa select keep keys %llu rows %u: pre %zu chunk %zu kv %zu sparse %zu, own %.1f MiB\n",
            (unsigned long long)keys,rows,f.select_pre.size(),f.select_chunk.size(),f.select_kv.size(),f.sparse_select.size(),double(owned())/1048576.0);
        while(f.select_kv.size()>1 && over(f.select_kv.size()) && f.select_kv.begin()->first!=uint32_t(keys)) f.select_kv.erase(f.select_kv.begin());
        auto& recipe=f.select_post[window_key(rows)];
        if(!recipe) {
            auto graph=build_select_qsa_post(*p.device,g,c,p.hc,rows);
            recipe=std::make_unique<HostRecipe>(*p.device,std::move(graph),declared(*graph,unique),std::vector<std::string>{});
            if(!split) recipe->set("ffn.inverse_streams",streams);
        }
        for(auto* r:{pre.get(),kv.get(),recipe.get()})
            if(r) for(const auto& binding:unique) if(r->declares(binding.name)) r->bind_external(binding);
        if(sparse) {
            // Chunk i+1 selects on the device while the host lists chunk i's blocks; chunk i then gathers and attends.
            const uint32_t sc=64,schunks=rows/sc,top=528,words=uint32_t(keys/c.block/16);
            const uint64_t bits_bytes=uint64_t(sc)*words*4,info_bytes=uint64_t(sc)*2*4,ids_bytes=uint64_t(sc)*top*4;
            const auto fit2=[&](std::unique_ptr<DeviceBuffer>& buffer,std::unique_ptr<HostBuffer>& host,uint64_t bytes) {
                if(buffer && buffer->size()>=bytes) return;
                p.device->synchronize();buffer=std::make_unique<DeviceBuffer>(*p.device,bytes);host=std::make_unique<HostBuffer>(*p.device,bytes);
            };
            for(auto& s:f.sparse_slots) {
                fit2(s.bits,s.bits_host,bits_bytes);fit2(s.info,s.info_host,(info_bytes+127)/128*128);fit2(s.ids,s.ids_host,ids_bytes);
                if(!s.read) s.read=std::make_unique<Event>(*p.device);
            }
            if(!f.sparse_start || f.sparse_start->size()<uint64_t(schunks)*128) {
                p.device->synchronize();f.sparse_start=std::make_unique<DeviceBuffer>(*p.device,uint64_t(schunks)*128);
                f.sparse_start_host=std::make_unique<HostBuffer>(*p.device,uint64_t(schunks)*128);
            }
            // The previous window's start upload has run (the host rewrites its staging).
            if(sb.start_uploaded && sb.start_uploaded->recorded()) sb.start_uploaded->synchronize();
            {auto* host=static_cast<float*>(f.sparse_start_host->data());
             for(uint32_t i=0;i<schunks;++i) {host[i*32]=float(step.cell);host[i*32+1]=float(i*sc);}}
            upload(chain.stream,*f.sparse_start,*f.sparse_start_host,uint64_t(schunks)*128);
            if(!sb.start_uploaded) sb.start_uploaded=std::make_unique<Event>(*p.device);
            sb.start_uploaded->record(chain.stream);
            std::vector<Binding> sparse_io=unique;
            sparse_io.push_back({"sp.bits",f.sparse_slots[0].bits.get()});sparse_io.push_back({"sp.info",f.sparse_slots[0].info.get()});
            sparse_io.push_back({"sp.ids",f.sparse_slots[0].ids.get()});
            for(auto& b:sparse_io) if(b.name=="sel.start") b={"sel.start",f.sparse_start.get()};
            auto& select=f.sparse_select[{uint32_t(keys),window_key(rows)}];
            if(!select) {
                auto graph=build_sparse_qsa_select(*p.device,c,uint32_t(keys),rows,sc);
                select=std::make_unique<HostRecipe>(*p.device,std::move(graph),declared(*graph,sparse_io),std::vector<std::string>{});
            }
            if(!f.sparse_attend) {
                auto graph=build_sparse_qsa_attend(*p.device,c,sc,top);
                f.sparse_attend=std::make_unique<HostRecipe>(*p.device,std::move(graph),declared(*graph,sparse_io),std::vector<std::string>{});
            }
            auto& attend=*f.sparse_attend;
            for(const auto& binding:sparse_io) {if(select->declares(binding.name)) select->bind_external(binding);if(attend.declares(binding.name)) attend.bind_external(binding);}
            pre->launch(chain.stream);
            const uint64_t query_chunk=uint64_t(sc)*hq*d*4,index_chunk=uint64_t(sc)*hi*di*4;
            const auto run_select=[&](uint32_t i) {
                auto& s=f.sparse_slots[i%2];
                select->bind_external({"sel.start",f.sparse_start.get(),uint64_t(i)*128});select->bind_external({"sel.iq",sb.iq.get(),i*index_chunk});
                select->bind_external({"sp.bits",s.bits.get()});select->bind_external({"sp.info",s.info.get()});
                select->launch(chain.stream);
                download(chain.stream,*s.bits_host,*s.bits,bits_bytes);s.read->record(chain.stream);
            };
            run_select(0);
            for(uint32_t i=0;i<schunks;++i) {
                if(i+1<schunks) run_select(i+1);
                auto& s=f.sparse_slots[i%2];s.read->synchronize();
                sparse_qsa_ids(static_cast<const float*>(s.bits_host->data()),sc,words,top,static_cast<int32_t*>(s.ids_host->data()));
                upload(chain.stream,*s.ids,*s.ids_host,ids_bytes);
                attend.bind_external({"sel.start",f.sparse_start.get(),uint64_t(i)*128});
                attend.bind_external({"sp.ids",s.ids.get()});attend.bind_external({"sp.info",s.info.get()});
                attend.bind_external({"sel.query",sb.query.get(),i*query_chunk});attend.bind_external({"sel.cf",sb.cf.get(),i*query_chunk});
                attend.launch(chain.stream);
            }
            recipe->launch(chain.stream);chain.advance();
            if(split) run_window_ffn(*p.device,p.window_ffn,g,p.hc,chain,p.qsa_finish_bindings);
            p.qsa->device_store_rows(chain.stream,step.cell,rows,chain.stage,offset);
            return;
        }
        pre->launch(chain.stream);kv->launch(chain.stream);
        for(const auto& binding:unique) if(part->declares(binding.name)) part->bind_external(binding);
        const uint64_t query_chunk=uint64_t(chunk)*hq*d*4,index_chunk=uint64_t(chunk)*hi*di*4;
        for(uint32_t i=0;i<chunks;++i) {
            part->bind_external({"sel.start",sb.start.get(),uint64_t(i)*128});
            part->bind_external({"sel.query",sb.query.get(),i*query_chunk});part->bind_external({"sel.iq",sb.iq.get(),i*index_chunk});
            part->bind_external({"sel.cf",sb.cf.get(),i*query_chunk});
            if(stats) part->bind_external({"sel.kept",sb.kept_debug.get()});
            part->launch(chain.stream);
            if(stats) {
                // How many blocks the chunk's rows select together (a compacted attention would attend over these),
                // for the whole chunk and for its 64- and 128-row parts.
                const uint64_t kb=keys/c.block;
                download(chain.stream,*sb.kept_host,*sb.kept_debug,uint64_t(chunk)*kb*4);chain.stream.synchronize();
                const auto* kept=static_cast<const float*>(sb.kept_host->data());
                const auto unite=[&](uint32_t first,uint32_t count) {
                    uint64_t u=0;
                    for(uint64_t b=0;b<kb;++b) {bool any=false;for(uint32_t r=first;r<first+count && !any;++r) any=kept[r*kb+b]>0;u+=any;}
                    return u;
                };
                uint64_t per_row=0;for(uint64_t x=0;x<uint64_t(chunk)*kb;++x) per_row+=kept[x]>0;
                const auto part_mean=[&](uint32_t size) {
                    if(size>chunk) return 0.0;
                    uint64_t s=0;for(uint32_t f=0;f<chunk;f+=size) s+=unite(f,size);
                    return double(s)/double(chunk/size);
                };
                const uint64_t used=(uint64_t(step.cell)+uint64_t(i+1)*chunk+c.block-1)/c.block;
                std::fprintf(stderr,"strata-hpu: qsa select stats layer %u cells %u keys %llu chunk %u/%u rows %u: row blocks %.1f, union 64 %.1f, 128 %.1f, chunk %llu of %llu used blocks\n",
                             p.index,step.cell,(unsigned long long)keys,i,chunks,chunk,double(per_row)/chunk,part_mean(64),part_mean(128),
                             (unsigned long long)unite(0,chunk),(unsigned long long)used);
            }
        }
        recipe->launch(chain.stream);chain.advance();
        if(split) run_window_ffn(*p.device,p.window_ffn,g,p.hc,chain,p.qsa_finish_bindings);
        p.qsa->device_store_rows(chain.stream,step.cell,rows,chain.stage,offset);
        return;
    }
    if(!p.qsa->device_verify_ready(rows)) {
        // Long context: the device cache's identity selection no longer covers the window, so the same rows run
        // through the grouped host law between two device graphs. The host state goes back to the committed
        // prefix as soon as the finish graph is enqueued; verify_commit() re-appends the accepted rows.
        if(!p.qsa->staged_verify_ready(rows)) throw std::logic_error("QSA staged verify window is not ready");
        const auto& c=f.config;
        try {
            auto first=p.qsa_project_bindings;first.push_back({"residual",&chain.in()});
            auto& project=f.staged_projection(rows,first);
            for(const auto& binding:p.qsa_project_bindings) project.bind_external(binding);
            project.bind_external({"residual",&chain.in()});
            project.launch(chain.stream);project.wait();
            // Diagnostic split of the staged window's host work (STRATA_HPU_QSA_PHASE_STATS). The projection's six
            // host downloads were outside every mark before, and the three uploads below were inside "gather".
            QsaPhaseTimer timer;
            const auto q_gate=project.values("qsa.q_gate"),k=project.values("qsa.k"),v=project.values("qsa.v");
            const auto index_query=project.values("qsa.index_query"),index_key=project.values("qsa.index_key");
            const auto attn_inject=project.values("attn.inject");
            timer.mark("verify.staged-download");
            const auto a=p.qsa->stage_rows(q_gate,k,v,index_query,index_key,rows);
            timer.mark("verify.staged-stage");
            if(chain.prefill_final && qsa_chunk_rows() && rows>qsa_chunk_rows()) {
                // Attention uses a bounded reusable KV slab. Assemble every HC output
                // in the other residual buffer, then keep one full-window MoE graph.
                // Reusing a HostRecipe waits before its pinned inputs are rewritten;
                // all device writes and the FFN remain ordered on chain.stream.
                auto attention_weights=std::vector<Binding>{},ffn_bindings=std::vector<Binding>{};
                for(const auto& b:p.qsa_finish_bindings)
                    (b.name.starts_with("o.")?attention_weights:ffn_bindings).push_back(b);
                auto kv=p.qsa->device_gather_bindings(chain.stream);
                attention_weights.insert(attention_weights.end(),kv.begin(),kv.end());
                const uint64_t residual_row=uint64_t(f.hc.streams)*f.hc.hidden*4;
                const uint64_t query_row=uint64_t(c.query_heads)*c.width;
                for(uint32_t first_row=0;first_row<rows;first_row+=qsa_chunk_rows()) {
                    const uint32_t count=std::min<uint32_t>(qsa_chunk_rows(),rows-first_row);
                    auto bindings=attention_weights;
                    bindings.push_back({"residual",&chain.in(),uint64_t(first_row)*residual_row});
                    bindings.push_back({"attn_write.out",&chain.out(),uint64_t(first_row)*residual_row});
                    auto& attention=f.staged_attention_for(a.bucket,count,bindings);
                    for(const auto& b:bindings) attention.bind_external(b);
                    attention.set("att.query",std::span<const float>(a.query).subspan(uint64_t(first_row)*query_row,uint64_t(count)*query_row));
                    attention.set("att.q_gate",std::span<const float>(a.gate).subspan(uint64_t(first_row)*query_row,uint64_t(count)*query_row));
                    attention.set("inject",std::span<const float>(attn_inject).subspan(uint64_t(first_row)*f.hc.streams,uint64_t(count)*f.hc.streams));
                    p.qsa->staged_gather_ids(attention.input("kv.ids"),first_row,count);
                    attention.launch(chain.stream);
                }
                chain.advance();
                if(split) run_window_ffn(*p.device,p.window_ffn,g,p.hc,chain,ffn_bindings);
                else {
                    ffn_bindings.push_back({"attn_write.out",&chain.in()});ffn_bindings.push_back({"ffn_write.out",&chain.out()});
                    auto& ffn=f.staged_ffn_for(rows,ffn_bindings);
                    for(const auto& b:ffn_bindings) ffn.bind_external(b);
                    ffn.launch(chain.stream);chain.advance();
                }
                timer.mark("verify.staged-chunk-launch");
                p.qsa->rewind_staged();return;
            }
            auto finish_bindings=p.qsa_finish_bindings;finish_bindings.push_back({"residual",&chain.in()});
            finish_bindings.push_back({"ffn_write.out",&chain.out()});
            if(qsa_device_gather_enabled()) {
                auto kv=p.qsa->device_gather_bindings(chain.stream);finish_bindings.insert(finish_bindings.end(),kv.begin(),kv.end());
            }
            auto& finish=f.staged_finisher(a.bucket,rows,finish_bindings);
            for(const auto& binding:finish_bindings) if(!split || finish.declares(binding.name)) finish.bind_external(binding);
            finish.set("att.query",a.query);finish.set("att.q_gate",a.gate);finish.set("inject",attn_inject);
            timer.mark("verify.staged-upload");
            const uint64_t kv=uint64_t(c.kv_heads)*a.bucket*c.width,masks=uint64_t(c.kv_heads)*a.bucket;
            if(qsa_device_gather_enabled()) p.qsa->staged_gather_ids(finish.input("kv.ids"));
            else {
            auto key=finish.input("att.key"),value=finish.input("att.value"),mask=finish.input("att.mask");
            if(key.size()!=kv*rows || value.size()!=key.size() || mask.size()!=masks*rows)
                throw std::logic_error("QSA staged finish geometry mismatch");
            for(uint32_t row=0;row<rows;++row)
                p.qsa->gather_row(row,key.subspan(uint64_t(row)*kv,kv),value.subspan(uint64_t(row)*kv,kv),
                                  mask.subspan(uint64_t(row)*masks,masks));
            }
            timer.mark("verify.staged-gather");
            finish.launch(chain.stream);
            timer.mark("verify.staged-launch");
            chain.advance();
            if(split) run_window_ffn(*p.device,p.window_ffn,g,p.hc,chain,p.qsa_finish_bindings);
            p.qsa->rewind_staged();
        } catch(...) {p.qsa->drop_staged();throw;}
        return;
    }
    auto step=p.qsa->device_verify_step(chain.stream,rows);
    const auto& c=f.config;const uint64_t row=uint64_t(c.kv_heads)*c.width*4,stride=verify_stage_stride(g,chain.max_rows),offset=verify_stage_offset(g,p.index,chain.max_rows);
    if(offset+stride>chain.stage.size()) throw std::logic_error("QSA verify stage too small");
    const bool canonical=chain.prefill_final && prefill_qsa32_enabled() && rows>32;
    // STRATA_HPU_QSA_DENSE_WINDOW reads the shared cache instead of the gathered copy, which is the same cell set
    // only when the selection covers every cell (bucket == cell). Otherwise the gather holds a subset in id order and
    // the dense read would silently attend different cells, so refuse rather than return different logits.
    if(qsa_dense_window_enabled() && step.bucket!=step.cell)
        throw std::logic_error("STRATA_HPU_QSA_DENSE_WINDOW needs an identity selection: bucket="+std::to_string(step.bucket)+
                               " cell="+std::to_string(step.cell));
    const auto chunks=canonical?canonical_qsa_chunks(c,step.cell,rows):std::vector<QsaChunk>{};
    const uint64_t key=(uint64_t(step.cell)<<24)^(uint64_t(step.bucket)<<8)^rows;
    std::vector<Binding> masks;
    if(canonical) {
        uint64_t at=128;masks.push_back({"dev.cell",&chain.mask,0});
        if(chain.mask_key!=key) {const float cell=float(step.cell);std::memcpy(chain.mask_host.data(),&cell,4);}
        for(uint32_t i=0;i<chunks.size();++i) {
            const auto& chunk=chunks[i];const uint64_t bytes=uint64_t(chunk.rows)*(chunk.bucket+chunk.rows)*4;
            if(at+bytes>chain.mask.size()) throw std::logic_error("canonical QSA mask buffer too small");
            masks.push_back({"dev.vmask_"+std::to_string(i),&chain.mask,at});
            if(chain.mask_key!=key) {
                const auto mask=QsaBlock::verify_mask(step.cell+chunk.first,chunk.bucket,chunk.rows);
                std::memcpy(static_cast<uint8_t*>(chain.mask_host.data())+at,mask.data(),bytes);
            }
            at+=bytes;
        }
        if(chain.mask_key!=key) {upload(chain.stream,chain.mask,chain.mask_host,at);chain.mask_key=key;}
    } else {
        masks.push_back({"dev.vmask",&chain.mask});
        if(chain.mask_key!=key) {
            const auto mask=QsaBlock::verify_mask(step.cell,step.bucket,rows);const uint64_t bytes=mask.size()*4;
            if(bytes>chain.mask.size()) throw std::logic_error("QSA verify mask buffer too small");
            std::memcpy(chain.mask_host.data(),mask.data(),bytes);upload(chain.stream,chain.mask,chain.mask_host,bytes);chain.mask_key=key;
        }
    }
    auto io=step.bindings;io.insert(io.end(),masks.begin(),masks.end());
    io.push_back({"residual",&chain.in()});io.push_back({"ffn_write.out",&chain.out()});
    io.push_back({"dev.k_now",&chain.stage,offset});io.push_back({"dev.v_now",&chain.stage,offset+rows*row});
    io.push_back({"qsa.index_key",&chain.stage,offset+2*rows*row});
    if(qsa_dense_window_enabled()) {
        // The dense window reads the shared cache's committed cells instead of a per-row gathered copy, so the
        // verify graph binds the cache under the slice names at offset 0 (b*hk*d floats of a capacity-sized buffer).
        // With QSA_DEVICE_GATHER the step bindings may already carry the cache, and a duplicate name is rejected.
        auto kv=p.qsa->device_gather_bindings(chain.stream);
        for(auto candidate:kv) {
            candidate.name=candidate.name=="kv.cache_k"?"kv.cache_slice_k":"kv.cache_slice_v";
            const bool present=std::any_of(io.begin(),io.end(),[&](const Binding& b){return b.name==candidate.name;});
            if(!present) io.push_back(candidate);
        }
    }
    std::vector<uint32_t> canonical_key{step.bucket,window_key(rows)};for(const auto& chunk:chunks) canonical_key.push_back(chunk.bucket);
    auto& recipe=canonical?f.verify_canonical[canonical_key]:f.verify[{step.bucket,window_key(rows)}];
    if(!recipe) {
        auto first=p.qsa_project_bindings;first.insert(first.end(),p.qsa_finish_bindings.begin(),p.qsa_finish_bindings.end());first.insert(first.end(),io.begin(),io.end());
        // A name may arrive from more than one of those lists (the KV cache in particular), and HostRecipe rejects
        // duplicates: keep the first occurrence of each.
        {
            std::vector<Binding> unique;std::vector<std::string> seen;
            for(const auto& b:first) {
                // The dense window does not declare the gathered tensors, so any binding for them (or for the gather
                // ids) has no tensor to attach to and must be dropped as well as deduplicated.
                if(qsa_dense_window_enabled() && (b.name=="dev.k_cache" || b.name=="dev.v_cache" || b.name=="kv.ids")) continue;
                if(std::find(seen.begin(),seen.end(),b.name)!=seen.end()) continue;
                seen.push_back(b.name);unique.push_back(b);
            }
            first.swap(unique);
        }
        auto graph=canonical?build_verify_qsa_canonical(*p.device,g,c,p.hc,step.bucket,rows,chunks):build_verify_qsa(*p.device,g,c,p.hc,step.bucket,rows);
        if(split) first=declared(*graph,first);
        recipe=std::make_unique<HostRecipe>(*p.device,std::move(graph),first,std::vector<std::string>{});
        const std::array<float,1> streams{1/float(p.hc.streams)},scale{1/std::sqrt(float(c.width))};
        recipe->set("attn.inverse_streams",streams);recipe->set("att.scale",scale);
        if(!split) recipe->set("ffn.inverse_streams",streams);
    }
    // The dense window's graph does not declare the gathered tensors (nor the gather ids), so bindings for them are
    // dropped here exactly as they are when the recipe is first built.
    const auto keep_binding=[&](const Binding& b) {
        if(split && !recipe->declares(b.name)) return false;
        return !(qsa_dense_window_enabled() && (b.name=="dev.k_cache" || b.name=="dev.v_cache" || b.name=="kv.ids"));
    };
    for(const auto& binding:p.qsa_project_bindings) if(keep_binding(binding)) recipe->bind_external(binding);
    for(const auto& binding:p.qsa_finish_bindings) if(keep_binding(binding)) recipe->bind_external(binding);
    for(const auto& binding:io) if(keep_binding(binding)) recipe->bind_external(binding);
    recipe->launch(chain.stream);chain.advance();
    if(split) run_window_ffn(*p.device,p.window_ffn,g,p.hc,chain,p.qsa_finish_bindings);
    p.qsa->device_store_rows(chain.stream,step.cell,rows,chain.stage,offset);
}
void Layer::verify_commit(VerifyChain& chain,uint32_t accepted) {
    auto& p=*impl_;std::lock_guard lock(p.mutex);const auto& g=p.geometry;const uint32_t rows=chain.rows;
    if(p.poisoned || !accepted || accepted>rows || p.position+accepted>p.options.context) throw std::invalid_argument("layer verify commit rows");
    if(p.ple) {
        // chain.ple_history_host already holds every row's history (downloaded with the window's results).
        const uint64_t history=uint64_t(9)*g.n_embd*g.hc*4;
        // A final-state prefill window kept only its last row's history, at 0.
        const uint64_t at=chain.prefill_final?0:uint64_t(accepted-1)*history;
        const auto* values=reinterpret_cast<const float*>(static_cast<const uint8_t*>(chain.ple_history_host.data())+at);
        std::vector<float> next(values,values+history/4);
        if(!all_finite(next)) throw NonFiniteOutput("non-finite verify PLE history");
        p.ple_history=std::move(next);
    }
    if(p.gdn) {
        // The window has finished (the caller downloaded its results), so its row buffers are no longer written.
        if(chain.prefill_final && accepted!=rows) throw std::logic_error("final-state prefill cannot partially commit");
        const uint32_t at=chain.prefill_final?0:accepted-1;
        p.gdn->adopt(p.verify_states[at],p.verify_convs[at],accepted);
    } else if(p.qsa->staged_pending()) {
        // Long-context window: the host holds the exact rows it appended, so the accepted prefix is re-appended
        // from the staged copies (the device stage above never received them).
        p.qsa->commit_staged(accepted);
    } else {
        const auto& c=p.qsa->config();const uint64_t row=uint64_t(c.kv_heads)*c.width,index=c.index_width;
        const auto* base=static_cast<const uint8_t*>(chain.stage_host.data())+verify_stage_offset(g,p.index,chain.max_rows);
        const auto* k=reinterpret_cast<const float*>(base);const auto* v=reinterpret_cast<const float*>(base+rows*row*4);
        const auto* key=reinterpret_cast<const float*>(base+2*rows*row*4);
        p.qsa->device_commit_rows({k,rows*row},{v,rows*row},{key,rows*index},accepted);
    }
    p.position+=accepted;
}
std::vector<float> Layer::prefill(std::span<const float> residual,std::span<const float> embedding,const std::function<bool()>& cancelled) {
    auto& p=*impl_;std::lock_guard lock(p.mutex);const auto& g=p.geometry;
    const uint64_t h=g.n_embd,n=h*g.hc,rows=n?residual.size()/n:0;
    if(p.poisoned || !rows || rows>p.options.prefill_rows || residual.size()%n || rows>p.options.context-p.position ||
       (p.ple?embedding.size()!=rows*h:!embedding.empty())) throw std::invalid_argument("layer prefill window state/shape mismatch");
    const auto gdn_point=p.gdn?p.gdn->checkpoint():GdnBlock::Checkpoint{};
    const auto qsa_point=p.qsa?p.qsa->checkpoint():QsaState::Checkpoint{};
    auto history=p.ple_history;size_t ple_row=0;LayerOperations op;
    if(p.ple) op.ple=[&](std::span<const float> r) {
        p.ple->set("hidden",r);p.ple->set("embedding",embedding.subspan(ple_row++*h,h));p.ple->set("history",history);p.ple->run();
        auto result=p.ple->values("out");history=p.ple->values("history_out");
        if(!all_finite(history)) throw std::runtime_error("non-finite window PLE history");
        return result;
    };
    op.read=[&](std::span<const float> r,bool ffn) {
        auto& read=ffn?*p.read_ffn:*p.read_attention;read.set("residual",r);read.run();return HcReadResult{read.values("mixed"),read.values("inject")};
    };
    op.mixer=[&](std::span<const float> x) {return p.gdn?p.gdn->decode(x):p.qsa->decode(x);};
    op.moe=[&](std::span<const float> x) {
        if(!p.resident) return p.moe->run(x);
        if(!p.resident_window) throw std::logic_error("resident prefill executor missing");
        std::vector<Binding> weights;
        const auto router=p.moe->router_bindings();
        const size_t count=router_merge_enabled()?1:router.size();
        for(size_t i=0;i<count;++i) {auto binding=router[i];binding.name="router."+binding.name;weights.push_back(std::move(binding));}
        for(auto binding:p.resident->bindings()) {binding.name="moe."+binding.name;weights.push_back(std::move(binding));}
        return p.resident_window->run(*p.device,g,x,weights);
    };
    op.write=[&](std::span<const float> r,std::span<const float> b,std::span<const float> inject,bool) {
        p.write->set("residual",r);p.write->set("block",b);p.write->set("inject",inject);p.write->run();return p.write->values("out");
    };
    op.rollback_mixer=[&] {
        try {if(p.gdn) p.gdn->restore(gdn_point);else p.qsa->restore(qsa_point);}
        catch(...) {p.poisoned=true;throw SessionFatal("layer window mixer rollback failed");}
    };
    if(p.options.hc_window) {
        if(p.ple_window) op.window_ple=[&](std::span<const float> r) {
            auto next=p.ple_window->run(r,embedding,history,cancelled);history=std::move(next.history);return std::move(next.output);
        };
        if(p.gdn && p.options.gdn_window) op.window_mixer=[&](std::span<const float> x) {return p.gdn->prefill(x,cancelled);};
        if(p.qsa && p.options.qsa_window) op.window_mixer=[&](std::span<const float> x) {return p.qsa->prefill(x,cancelled);};
        op.window_read=[&](std::span<const float> r,bool ffn) {
            const auto bindings=p.select({"w_down","w_up","w_inject","norm"},ffn?"hc_ffn_":"hc_attn_");
            return p.options.hc_window->read(r,bindings);
        };
        op.window_write=[&](std::span<const float> r,std::span<const float> b,std::span<const float> inject,bool) {
            return p.options.hc_window->write(r,b,inject);
        };
    }
    auto result=execute_layer_window(residual,uint32_t(h),uint32_t(g.hc),op,cancelled);
    if(p.ple) p.ple_history=std::move(history);
    p.position+=rows;return result;
}
void Layer::reset() {
    auto& p=*impl_;std::lock_guard lock(p.mutex);if(p.gdn) p.gdn->reset();else p.qsa->reset();
    std::fill(p.ple_history.begin(),p.ple_history.end(),0);p.position=0;p.poisoned=false;
}
uint64_t Layer::size() const {const auto& p=*impl_;std::lock_guard lock(p.mutex);return p.position;}
bool Layer::healthy() const {const auto& p=*impl_;std::lock_guard lock(p.mutex);return !p.poisoned;}
uint64_t Layer::dense_payload_bytes() const {
    const auto& p=*impl_;return p.weights->payload_bytes()+checked_product(uint64_t(p.geometry.n_embd),checked_product(uint64_t(p.geometry.n_expert)+1,2));
}
Layer::Checkpoint Layer::checkpoint() const {
    const auto& p=*impl_;std::lock_guard lock(p.mutex);
    if(p.poisoned) throw std::logic_error("cannot checkpoint failed layer");
    Checkpoint result;result.owner=p.owner;result.position=p.position;result.ple_history=p.ple_history;
    if(p.gdn) result.gdn=p.gdn->checkpoint();else result.qsa=p.qsa->checkpoint();
    return result;
}
void Layer::restore(const Checkpoint& point) {
    auto& p=*impl_;std::lock_guard lock(p.mutex);
    if(p.poisoned || point.owner!=p.owner || (!point.persistent && point.position>p.position) || point.position>p.options.context || point.ple_history.size()!=p.ple_history.size())
        throw std::invalid_argument("stale/foreign layer checkpoint");
    // Mixer validates its private owner/epoch before any state is mutated.
    if(p.gdn) p.gdn->restore(point.gdn);else p.qsa->restore(point.qsa);
    std::copy(point.ple_history.begin(),point.ple_history.end(),p.ple_history.begin());p.position=point.position;
}
Layer::Checkpoint Layer::snapshot() const {
    const auto& p=*impl_;std::lock_guard lock(p.mutex);
    if(p.poisoned) throw std::logic_error("cannot snapshot failed layer");
    Checkpoint result;result.owner=p.owner;result.position=p.position;result.ple_history=p.ple_history;result.persistent=true;
    if(p.gdn) result.gdn=p.gdn->snapshot();else result.qsa=p.qsa->snapshot();
    return result;
}
uint64_t Layer::parked_bytes(uint64_t cells) const {
    const auto& p=*impl_;std::lock_guard lock(p.mutex);
    uint64_t total=checked_product(p.ple_history.size(),sizeof(float));
    if(!p.qsa) return total+p.gdn->parked_bytes();
    const auto& c=p.qsa->config();ParkedKvLayout layout;uint64_t image=0;std::string error;
    if(!parked_kv_layout(c.kv==KvMode::Q4,c.kv==KvMode::Int8,c.kv==KvMode::K8V4,true,false,cells,true,c.width,c.kv_heads,c.index_width,layout,error))
        throw std::invalid_argument("parked K/V layer cost: "+error);
    if(cells && !parked_kv_image_bytes(layout,cells,image,error)) throw std::invalid_argument("parked K/V layer cost: "+error);
    // The raw rows of the block that is not yet complete, plus the rotating spare row: live state the image
    // carries on every park, so it is part of the cost even when the reuse keeps the completed rows.
    return total+image+checked_product(uint64_t(c.block)+1,checked_product(c.index_width,sizeof(float)));
}
uint64_t Layer::snapshot_reused(Checkpoint& point,uint64_t unchanged_tokens) const {
    const auto& p=*impl_;std::lock_guard lock(p.mutex);
    if(p.poisoned) throw std::logic_error("cannot snapshot failed layer");
    if(point.owner!=p.owner || !point.persistent) throw std::invalid_argument("foreign/unowned parked layer image");
    if(!p.qsa) {
        // A GDN layer holds no parked K/V image, so upstream's reuse decision does not cover it: re-capture the
        // block's own state and report nothing reused.
        Checkpoint fresh;fresh.owner=p.owner;fresh.position=p.position;fresh.ple_history=p.ple_history;fresh.persistent=true;
        fresh.gdn=p.gdn->snapshot();
        point=std::move(fresh);
        return 0;
    }
    if(!p.qsa->size()) {
        if(unchanged_tokens) throw std::invalid_argument("parked K/V layer: invalid unchanged prefix");
        point.qsa=p.qsa->snapshot_reused(std::move(point.qsa),0,0);
        point.position=p.position;point.ple_history=p.ple_history;return 0;
    }
    uint64_t sizes[5]={};p.qsa->snapshot_sizes(point.qsa,sizes);
    const auto& c=p.qsa->config();
    ParkedKvLayout layout;ParkedKvReuse reuse;std::string error;
    // This backend parks the host pool image in the live --kv format (F16, INT8 codes+scales, rotated Q4_0
    // blocks or the K8V4 INT8-K/Q4_0-V pair), the indexer part is always present for a QSA layer, and `ring` is
    // upstream's kv_mode == 2, which this residency model has not.
    if(!parked_kv_layout(c.kv==KvMode::Q4,c.kv==KvMode::Int8,c.kv==KvMode::K8V4,true,false,p.qsa->size(),true,c.width,c.kv_heads,c.index_width,layout,error))
        throw std::invalid_argument("parked K/V layer: "+error);
    if(!parked_kv_reuse(layout,sizes,unchanged_tokens,reuse,error)) throw std::invalid_argument("parked K/V layer: "+error);
    // The boundaries come out of the law's own byte counts, so nothing here recomputes a page or a block.
    const uint64_t per_cell=layout.sizes[0]/layout.cells;
    const auto whole_cells=uint32_t(reuse.keep[0]/per_cell);
    const auto pooled_rows_kept=uint32_t(reuse.keep[4]/(uint64_t(sizeof(float))*c.index_width));
    point.qsa=p.qsa->snapshot_reused(std::move(point.qsa),whole_cells,pooled_rows_kept);
    point.position=p.position;point.ple_history=p.ple_history;
    return reuse.reused_bytes;
}
void Layer::write_snapshot(StateWriter& writer,const Checkpoint& point) const {
    const auto& p=*impl_;std::lock_guard lock(p.mutex);
    if(point.owner!=p.owner || !point.persistent) throw std::invalid_argument("layer serialization requires an owned full snapshot");
    writer.integer(p.index);writer.integer(point.position);writer.vector<float>(point.ple_history);
    if(p.gdn) p.gdn->write_snapshot(writer,point.gdn);else p.qsa->write_snapshot(writer,point.qsa);
}
Layer::Checkpoint Layer::read_snapshot(StateReader& reader,uint64_t position) const {
    const auto& p=*impl_;std::lock_guard lock(p.mutex);
    if(position>p.options.context || reader.integer()!=p.index || reader.integer()!=position) throw std::invalid_argument("serialized layer index/position mismatch");
    Checkpoint point;point.owner=p.owner;point.position=position;point.persistent=true;point.ple_history=reader.vector<float>(p.ple_history.size());
    if(!all_finite(point.ple_history)) throw std::invalid_argument("non-finite serialized PLE history");
    if(p.gdn) point.gdn=p.gdn->read_snapshot(reader,position);else point.qsa=p.qsa->read_snapshot(reader,uint32_t(position));
    return point;
}
}
