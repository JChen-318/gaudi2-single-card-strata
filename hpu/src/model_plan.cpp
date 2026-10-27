#include "strata/hpu/model_plan.hpp"
#include "strata/hpu/performance_modes.hpp"
#include "strata/hpu/kv_cell.hpp"
#include "strata/hpu/ple_input.hpp"
#include "strata/hpu/qsa_state.hpp"
#include "strata/hpu/rope_config.hpp"
#include "strata/hpu/mtp_weights.hpp"
#include "strata/hpu/packed_quant.hpp"
#include "strata/hpu/expert_format.hpp"
#include "strata/hpu/expert_scales.hpp"
#include "strata/hpu/prefill_plan.hpp"
#include "strata/hpu/mtp_window_plan.hpp"
#include "strata/hpu/slot_session.hpp"
#include "strata/hpu/cache_metadata.hpp"
#include "strata/hpu/u4_resident.hpp"
#include <algorithm>
#include <numeric>

namespace strata::hpu {
namespace {
void add(uint64_t& destination,uint64_t value) {
    if(value>UINT64_MAX-destination) throw std::overflow_error("model memory sum overflow");
    destination+=value;
}
}
ModelPlan plan_model(const WeightTable& table,const ModelMemoryOptions& o,const std::string& external_ple,const strata::kernels::PleIoOptions& io,const bool ple_enabled) {
    const auto& g=table.geometry();const uint64_t h=g.n_embd,r=g.hc,n=h*r,lr=g.hc_lr,ff=g.n_ff;
    if(!o.context || o.context>INT32_MAX || !o.page_size || !o.hbm_limit || !o.pinned_limit || !o.host_limit ||
       o.verify_rows>7 || o.staging/4<std::max(h,ff) || o.expert_cache<checked_product(checked_product(h,ff),6)) throw std::invalid_argument("invalid model memory policy");
    // One token of K+V in the selected format: FP16 is two bytes per element, INT8 one code byte plus one FP16
    // scale per 64 values, Q4_0 one 18-byte block per 32 rotated values. The arena and residency prices below
    // use this same cell, so --kv cannot undercount it.
    const uint64_t kv_cell_bytes=kv_token_bytes(uint32_t(g.head_dim),uint32_t(g.n_head_kv),o.kv_mode);
    (void)resolve_rope(table);
    ModelPlan plan;plan.layer_dense_bytes.resize(g.n_layers);
    // --no-ple is upstream's explicit ablation (:3434): there is no table, no reader and no PLE run, so
    // none of that host/reader storage is priced. The layer's PLE weights are not uploaded either (see the
    // layer-1 dense block below).
    if(ple_enabled) {
        const auto ple=inspect_ple_source(external_ple.empty()?table.tensor_path("per_layer_token_embd.weight"):external_ple);
        (void)ple_hash_config(table,ple.rows); // Main model constants, external or internal storage bounds.
        const auto ram=ple_ram_memory(ple,io);
        plan.ple_ram_table_bytes=ram.table_bytes;plan.ple_ram_host_bytes=ram.host_bytes;plan.ple_ram_worker_bytes=ram.worker_bytes;
        const auto reader=ple_reader_memory(ple,io);
        plan.ple_reader_host_bytes=std::max(kPleReaderBaselineHostBytes,reader.host_bytes);
        plan.ple_reader_extra_host_bytes=plan.ple_reader_host_bytes-kPleReaderBaselineHostBytes;
        plan.ple_reader_cache_rows=reader.cache_rows;plan.ple_reader_worker_bytes=reader.worker_bytes;
    }
    // FP16 K+V bytes of one page slot summed over every QSA layer: the arena --kv caps and --kv-resident pins.
    uint64_t kv_per_page_slot=0;
    const auto candidates=plan_candidate_window(g,o.verify_rows,o.context);
    const auto prefill=plan_prefill_window(g,std::max(o.prefill_rows,candidates.rows),o.context);
    plan.prefill_rows=std::min(o.prefill_rows,o.context);plan.window_rows=prefill.rows;plan.verify_rows=candidates.rows;
    plan.prefill_host_bytes=prefill.host_bytes;plan.prefill_hbm_bytes=prefill.hbm_bytes;
    plan.verify_logits_bytes=candidates.logits_bytes;plan.verify_host_bytes=candidates.host_bytes;
    plan.verify_hbm_bytes=candidates.hbm_bytes;plan.verify_head_pinned_bytes=candidates.head_pinned_bytes;
    if(plan.verify_head_pinned_bytes>o.pinned_limit) throw std::invalid_argument("candidate head exceeds pinned memory policy");
    add(plan.runtime_reserve,prefill.hbm_bytes);
    add(plan.runtime_reserve,plan.verify_hbm_bytes);
    plan.packed_expert_execution_bytes=packed_expert_mutable_bytes(h,ff,g.n_layers);
    add(plan.runtime_reserve,plan.packed_expert_execution_bytes);
    auto require=[&](const std::string& name,std::vector<uint64_t> shape,bool constant=false) {
        const auto& info=table.info(name);
        if(info.ggml_type==40 && name.find("_exps.weight")==std::string::npos && name.find("_shexp.weight")==std::string::npos)
            throw std::invalid_argument("NVFP4 global scales currently require an expert projection: "+name);
        if(!info.bf16_conversion_supported) throw std::invalid_argument("unsupported model tensor conversion: "+name);
        const auto elements=std::accumulate(shape.begin(),shape.end(),uint64_t(1),[](uint64_t a,uint64_t b){return checked_product(a,b);});
        bool same=info.gguf_shape==shape;
        if(constant && !info.gguf_shape.empty() && info.gguf_shape.size()<=2) {
            const auto actual=std::accumulate(info.gguf_shape.begin(),info.gguf_shape.end(),uint64_t(1),[](uint64_t a,uint64_t b){return checked_product(a,b);});
            same=actual==elements;
        }
        if(!same) throw std::invalid_argument("model tensor shape mismatch: "+name);
        return checked_product(elements,constant?4:2);
    };
    (void)require("token_embd.weight",{h,uint64_t(g.n_vocab)});
    if(embedding_ram_enabled()) plan.embedding_ram_host_bytes=checked_product(checked_product(h,uint64_t(g.n_vocab)),4)+65536;
    const auto bf16_head=require("output.weight",{h,uint64_t(g.n_vocab)});
    const auto head_type=table.info("output.weight").ggml_type;
    add(plan.head_bytes,packed_supported(head_type) && !o.bf16_head?checked_product(packed_row_bytes(head_type,h),g.n_vocab):bf16_head);
    add(plan.head_bytes,require("output_hc_norm.weight",{n},true));
    add(plan.head_bytes,require("output_hc_down.weight",{n,lr}));add(plan.head_bytes,require("output_hc_up.weight",{lr,n}));
    for(uint32_t layer=0;layer<uint64_t(g.n_layers);++layer) {
        const auto prefix="blk."+std::to_string(layer)+".";auto& bytes=plan.layer_dense_bytes[layer];
        auto matrix=[&](const std::string& name,uint64_t columns,uint64_t rows) {add(bytes,require(prefix+name,{columns,rows}));};
        auto constant=[&](const std::string& name,uint64_t elements) {add(bytes,require(prefix+name,{elements},true));};
        for(const auto* half:{"hc_attn_","hc_ffn_"}) {
            const std::string stem(half);matrix(stem+"down.weight",n,lr);matrix(stem+"up.weight",lr,n);matrix(stem+"inject.weight",n,r);constant(stem+"norm.weight",n);
        }
        matrix("ffn_gate_inp.weight",h,g.n_expert);
        // Upstream stores shared gate as [H] or [H,1]. Its runtime is BF16.
        add(bytes,require(prefix+"ffn_gate_inp_shexp.weight",{h},true)/2);
        for(const auto* name:{"ffn_gate_exps.weight","ffn_up_exps.weight"}) (void)require(prefix+name,{h,ff,uint64_t(g.n_expert)});
        (void)require(prefix+"ffn_down_exps.weight",{ff,h,uint64_t(g.n_expert)});
        for(const auto* name:{"ffn_gate_shexp.weight","ffn_up_shexp.weight"}) (void)require(prefix+name,{h,ff});
        (void)require(prefix+"ffn_down_shexp.weight",{ff,h});
        bool scaled=false;
        for(const auto* name:{"ffn_gate_exps.weight","ffn_up_exps.weight","ffn_down_exps.weight","ffn_gate_shexp.weight","ffn_up_shexp.weight","ffn_down_shexp.weight"}) scaled=scaled || table.info(prefix+name).ggml_type==40;
        if(scaled) {
            validate_expert_global_scales(table,layer);const auto reserve=scaled_expert_mutable_bytes(h,ff);
            add(plan.scaled_expert_execution_bytes,reserve);add(plan.runtime_reserve,reserve);
        }
        if(core::is_qsa_layer(g,layer)) {
            auto config=qsa_config(g,o.context,o.page_size);
            matrix("attn_q.weight",h,uint64_t(g.n_head)*g.head_dim*2);matrix("attn_k.weight",h,uint64_t(g.n_head_kv)*g.head_dim);
            matrix("attn_v.weight",h,uint64_t(g.n_head_kv)*g.head_dim);matrix("attn_output.weight",uint64_t(g.n_head)*g.head_dim,h);
            matrix("indexer.q_proj.weight",h,uint64_t(g.idx_q_heads)*g.idx_key_dim);matrix("indexer.k_proj.weight",h,g.idx_key_dim);
            // These four gamma vectors stay on the host; do not count as HBM.
            for(const auto* name:{"attn_q_norm.weight","attn_k_norm.weight"}) (void)require(prefix+name,{uint64_t(g.head_dim)},true);
            for(const auto* name:{"indexer.q_norm.weight","indexer.k_norm.weight"}) (void)require(prefix+name,{uint64_t(g.idx_key_dim)},true);
            const auto cells=checked_product((uint64_t(o.context)+o.page_size-1)/o.page_size,o.page_size);
            add(plan.host_state_bytes,checked_product(cells,kv_cell_bytes));
            // The K/V pools only: what the residency controls (hpu/include/strata/hpu/kv_residency.hpp) cap,
            // pin and would grow. The indexer/pool/page-table rows below are state, not the arena.
            add(kv_per_page_slot,checked_product(uint64_t(o.page_size),kv_cell_bytes));
            add(plan.host_state_bytes,checked_product((uint64_t(o.context)/config.block+1+config.block)*config.index_width,4));
            add(plan.host_state_bytes,checked_product((uint64_t(o.context)+o.page_size-1)/o.page_size,4));
            add(plan.host_state_bytes,checked_product(uint64_t(o.context)+1,8)); // nested QSA rewind ancestry
            add(plan.transaction_bytes,checked_product(uint64_t(config.block)*config.index_width,4));
        } else {
            matrix("attn_qkv.weight",h,g.ssm_conv_channels);matrix("attn_gate.weight",h,g.ssm_value_dim);
            matrix("ssm_out.weight",g.ssm_value_dim,h);matrix("ssm_alpha.weight",h,g.ssm_v_heads);matrix("ssm_beta.weight",h,g.ssm_v_heads);
            constant("ssm_norm.weight",g.ssm_state_size);constant("ssm_dt.bias",g.ssm_v_heads);constant("ssm_a",g.ssm_v_heads);
            add(bytes,require(prefix+"ssm_conv1d.weight",{uint64_t(g.ssm_d_conv),uint64_t(g.ssm_conv_channels)}));
            // Conv is kept FP32 rather than BF16.
            add(bytes,checked_product(uint64_t(g.ssm_d_conv)*g.ssm_conv_channels,2));
            const auto state=checked_product(uint64_t(g.ssm_state_size)*g.ssm_state_size*g.ssm_v_heads+uint64_t(g.ssm_d_conv-1)*g.ssm_conv_channels,4);
            add(plan.host_state_bytes,state);add(plan.transaction_bytes,state);
        }
        if(layer==1 && ple_enabled) {
            matrix("ple_key.weight",h,n);matrix("ple_value.weight",h,h);
            for(const auto* name:{"ple_norm_key.weight","ple_norm_query.weight","ple_norm_conv.weight"}) constant(name,n);
            add(bytes,require(prefix+"ple_conv1d.weight",{4,n}));add(bytes,checked_product(4*n,2));
            add(plan.host_state_bytes,checked_product(9*n,4));add(plan.transaction_bytes,checked_product(9*n,4));
        }
        add(plan.dense_bytes,bytes);
    }
    // Upstream's KV residency controls: --kv caps the arena (src/program/generate.cpp:1950, validated at
    // 2547/2559-2563), --kv-resident pins a floor of cells (src/core/layer.cpp:633-651,664-666) and
    // --kv-grow/--no-kv-grow decide whether the arena may grow (generate.cpp:4394-4408,6788-6889).  This
    // backend allocates the arena whole at plan time - there is no VMM range and no lendable expert-cache
    // slot - so a cap the arena cannot fit is refused HERE, where upstream discovers it mid-run when the
    // driver has no VRAM left (src/core/layer.cpp:558-562, generate.cpp:12649), and --kv-grow has nothing
    // to grow into.  The gaps are declared in hpu/include/strata/hpu/kv_residency.hpp, not faked.
    if(kv_per_page_slot) {
        KvArenaShape arena;
        arena.page_size=o.page_size;arena.capacity_cells=o.context;arena.arrays=1;arena.per_page[0]=kv_per_page_slot;
        arena.granularity_bytes=o.kv.granularity_bytes;
        validate_kv_residency(arena,o.kv);   // --kv-resident >= 0, and the inclusive --kv cap
        plan.kv_arena_bytes=kv_arena_bytes(arena,kv_arena_commit_cells(arena));
        plan.kv_resident_cells=kv_resident_pinned_cells(arena,o.kv);
    }
    add(plan.dense_bytes,plan.head_bytes);
    plan.hbm_required=plan.dense_bytes;add(plan.hbm_required,o.expert_cache);add(plan.hbm_required,plan.runtime_reserve);
    plan.host_required=plan.host_state_bytes;add(plan.host_required,plan.transaction_bytes*2);
    add(plan.host_required,plan.embedding_ram_host_bytes);
    add(plan.host_required,o.pinned_limit);add(plan.host_required,plan.ple_reader_host_bytes);add(plan.host_required,plan.host_compiler_reserve);
    add(plan.host_required,o.snapshot_budget);
    // Original suffix index uses a power-of-two table of 32-byte slots, at
    // most 4*context slots. Include history, request/model token copies and the
    // vocabulary-sized sampling/logit staging, even when lookup is disabled.
    plan.control_bytes=checked_product(o.context,160);add(plan.control_bytes,checked_product(uint64_t(g.n_vocab),32));
    add(plan.control_bytes,checked_product(n,8)); // committed/staged final residual
    add(plan.control_bytes,checked_product(uint64_t(g.n_embd),8)); // committed/staged mixed head input
    plan.slot_host_bytes=slot_session_host_bytes(o.batch_slots,o.context);
    add(plan.control_bytes,plan.slot_host_bytes);
    plan.cache_metadata_bytes=generation_cache_metadata_bytes(o.context,uint64_t(g.n_vocab));
    add(plan.control_bytes,plan.cache_metadata_bytes);
    plan.scalar_window_bytes=generation_scalar_window_bytes(o.context,uint64_t(g.n_vocab));
    add(plan.control_bytes,plan.scalar_window_bytes);
    add(plan.host_required,plan.control_bytes);
    add(plan.host_required,plan.prefill_host_bytes);
    add(plan.host_required,plan.verify_host_bytes);
    // CPU pool buffers/threads reserve. ExpertPool gives every worker its own ExpertScratch, which is
    // 17,408 bytes at the compiled H=2560/FF=640/MAXT=8 geometry, so this fixed 128 MiB covers ~7,700
    // workers - more than any Gaudi2 host's physical core count - without pricing the count, which
    // --pool-workers 0 only resolves to at pool construction.
    add(plan.host_required,o.cpu_cache);add(plan.host_required,128ull<<20);
    if(plan.ple_ram_host_bytes && (plan.ple_ram_host_bytes>o.host_limit || plan.host_required>o.host_limit-plan.ple_ram_host_bytes))
        throw std::invalid_argument("PLE full-table RAM reservation exceeds host memory policy");
    add(plan.host_required,plan.ple_ram_host_bytes);
    if(plan.hbm_required>o.hbm_limit || plan.host_required>o.host_limit) throw std::invalid_argument("model plan exceeds HBM/host memory policy");
    return plan;
}
void extend_resident_iq4_plan(ModelPlan& plan,const core::ModelGeometry& g,const ModelMemoryOptions& o) {
    if(plan.resident_iq4_bytes || plan.resident_u4_bytes || g.n_embd<=0 || g.n_ff<=0 || g.n_layers<=0 || g.n_expert<=0)
        throw std::invalid_argument("invalid resident IQ4 memory policy");
    const uint64_t h=g.n_embd,ff=g.n_ff,experts=g.n_expert,gu_rows=checked_product(ff,2);
    if(h%1280 || gu_rows%1280 || h%256 || ff%32)
        throw std::invalid_argument("resident IQ4 layout does not fit memory policy");
    // Exactly ResidentLayerExperts' layout: half a byte per weight, F32 scale per row/32 columns.
    const auto projection=[](uint64_t rows,uint64_t columns) {
        uint64_t bytes=checked_product(rows,columns)/2;add(bytes,checked_product(checked_product(rows,columns/32),4));return bytes;
    };
    const auto gu=projection(gu_rows,h),down=projection(h,ff);
    uint64_t layer_bytes=gu;add(layer_bytes,down);
    auto payload=checked_product(checked_product(layer_bytes,experts+1),uint64_t(g.n_layers));
    add(payload,256); // One shared codebook for the device, not one per layer.
    const auto staging=checked_product(std::min<uint64_t>(experts,64),gu);
    if(staging>o.pinned_limit) throw std::invalid_argument("resident IQ4 upload staging exceeds pinned memory policy");
    if(plan.hbm_required>o.hbm_limit || payload>o.hbm_limit-plan.hbm_required)
        throw std::invalid_argument("resident IQ4 model plan exceeds HBM memory policy");
    plan.resident_iq4_bytes=payload;plan.resident_iq4_staging_bytes=staging;add(plan.hbm_required,payload);
    // plan_model already reserves o.pinned_limit in host_required; the transient pack buffer lives there.
}
// Exactly the U4LayerStore layout the packer builds (strata/hpu/u4_resident.hpp): half a byte per weight plus one
// F32 scale and one F32 offset per output channel and K group, six consecutive [blocks] regions per layer with the
// shared expert as the last block. Gate/up uses a 512-column group (K=512, 5 groups) and down 320 (K=320, 2), so
// the MME batch weights are [experts*5][1280][512] and [experts*2][2560][320]; those groups are part of the store
// format, and the payload is the store's own bytes(), so the plan cannot drift from the packer.
void extend_resident_u4_plan(ModelPlan& plan,const core::ModelGeometry& g,const ModelMemoryOptions& o) {
    if(plan.resident_u4_bytes || plan.resident_iq4_bytes || g.n_embd<=0 || g.n_ff<=0 || g.n_layers<=0 || g.n_expert<=0)
        throw std::invalid_argument("invalid resident U4 memory policy");
    U4LayerStore store;
    try {store=U4LayerStore::make(uint64_t(g.n_expert),uint64_t(g.n_embd),uint64_t(g.n_ff));}
    catch(const std::invalid_argument&) {throw std::invalid_argument("resident U4 layout does not fit memory policy");}
    const auto payload=checked_product(store.bytes(),uint64_t(g.n_layers));
    // One pack round is the staging buffer: eight experts packed on the host, uploaded as six region spans.
    const auto staging=u4_staging_bytes(store);
    if(staging>o.pinned_limit) throw std::invalid_argument("resident U4 upload staging exceeds pinned memory policy");
    if(plan.hbm_required>o.hbm_limit || payload>o.hbm_limit-plan.hbm_required)
        throw std::invalid_argument("resident U4 model plan exceeds HBM memory policy");
    plan.resident_u4_bytes=payload;plan.resident_u4_staging_bytes=staging;add(plan.hbm_required,payload);
}
void extend_mtp_plan(ModelPlan& plan,const MtpWeights& weights,const ModelMemoryOptions& o,uint64_t cache,uint32_t window,uint32_t subset,uint32_t head_type,bool q4_projections,bool q4_head) {
    const auto& g=core::kCompiledGeometry;
    if(plan.mtp_dense_bytes || cache<checked_product(uint64_t(g.n_embd)*g.n_ff,6) || window>o.context || subset>uint32_t(g.n_vocab)) throw std::invalid_argument("invalid MTP memory policy");
    // The drafter follows the same --kv format as the trunk (upstream's globals decide both) except under k8v4,
    // where its ring is plain INT8 (src/core/mtp.cpp:344-349); kv_drafter_mode is that rule, so its cell is priced
    // by the format DenseKvState actually builds. Its K/V is part of the one arena --kv caps.
    const uint64_t dense_kv_cell=kv_token_bytes(uint32_t(g.head_dim),uint32_t(g.n_head_kv),kv_drafter_mode(o.kv_mode));
    plan.mtp_q4_projections=q4_projections;plan.mtp_q4_head=q4_head;
    plan.mtp_dense_bytes=weights.packed_projection_payload_bytes(q4_projections);plan.mtp_cache_bytes=cache;
    const uint64_t head_rows=q4_head?(subset?subset:uint32_t(g.n_vocab)):subset;
    add(plan.mtp_dense_bytes,checked_product(head_rows,q4_head?packed_row_bytes(2,g.n_embd):packed_supported(head_type)?packed_row_bytes(head_type,g.n_embd):uint64_t(g.n_embd)*2));
    plan.mtp_state_bytes=checked_product(uint64_t(o.context),dense_kv_cell);
    add(plan.mtp_state_bytes,(uint64_t(o.context)+1)*8);
    add(plan.dense_bytes,plan.mtp_dense_bytes);add(plan.host_state_bytes,plan.mtp_state_bytes);
    // The drafter's dense K/V (the line above's first term) is part of the same capped arena, so --kv is
    // re-checked against it and --kv-resident's floor recomputed over the grown arena.  Page-rounded, so
    // this counts at most one page of slack per KV head - a conservative cap, never an undercount.
    if(plan.kv_arena_bytes && o.page_size && o.context) {
        const uint64_t slots=(uint64_t(o.context)+o.page_size-1)/o.page_size;
        KvArenaShape arena;
        arena.page_size=o.page_size;arena.capacity_cells=o.context;arena.arrays=2;
        arena.per_page[0]=(plan.kv_arena_bytes>64?(plan.kv_arena_bytes-64)/slots:1);   // exact inverse above
        arena.per_page[1]=checked_product(uint64_t(o.page_size),dense_kv_cell);
        arena.granularity_bytes=o.kv.granularity_bytes;
        validate_kv_residency(arena,o.kv);
        plan.kv_arena_bytes=kv_arena_bytes(arena,kv_arena_commit_cells(arena));
        plan.kv_resident_cells=kv_resident_pinned_cells(arena,o.kv);
    }
    const auto control=uint64_t(g.n_vocab)*8+uint64_t(g.n_embd)*g.hc*8+uint64_t(g.n_embd)*4+(3ull<<20);
    add(plan.control_bytes,control);add(plan.host_required,plan.mtp_state_bytes);add(plan.host_required,control);
    add(plan.host_required,checked_product(uint64_t(window?window:o.context)*g.n_head_kv*g.head_dim,8)); // gathered F32 K/V
    add(plan.runtime_reserve,256ull<<20);add(plan.hbm_required,256ull<<20);
    const auto packed_execution=packed_expert_mutable_bytes(g.n_embd,g.n_ff,1);
    add(plan.packed_expert_execution_bytes,packed_execution);add(plan.runtime_reserve,packed_execution);add(plan.hbm_required,packed_execution);
    add(plan.hbm_required,plan.mtp_dense_bytes);add(plan.hbm_required,cache);
    const auto batch=plan_mtp_window(g,plan.window_rows,o.context,window);
    plan.mtp_window_rows=batch.rows;plan.mtp_window_host_bytes=batch.host_bytes;plan.mtp_window_hbm_bytes=batch.hbm_bytes;plan.mtp_window_pinned_bytes=batch.pinned_bytes;
    if(batch.pinned_bytes>o.pinned_limit) throw std::invalid_argument("MTP window exceeds pinned memory policy");
    add(plan.host_required,batch.host_bytes);add(plan.hbm_required,batch.hbm_bytes);add(plan.runtime_reserve,batch.hbm_bytes);
    if(plan.hbm_required>o.hbm_limit || plan.host_required>o.host_limit) throw std::invalid_argument("MTP model plan exceeds HBM/host memory policy");
}
}
