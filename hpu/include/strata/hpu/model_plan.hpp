#pragma once
#include "strata/hpu/weights.hpp"
// The upstream KV residency knobs (--kv / --kv-grow / --no-kv-grow / --kv-resident); a pure policy unit
// that pulls in no Synapse header, the same way prefill_schedule.hpp stays standalone.
#include "strata/hpu/kv_residency.hpp"
#include "strata/hpu/kv_mode.hpp"
#include "strata/kernels/ngram.hpp"

namespace strata::hpu {
class MtpWeights;
struct ModelMemoryOptions {
    uint32_t context=32768,page_size=4;
    uint32_t prefill_rows=0;
    uint32_t verify_rows=0; // explicit native candidate-window opt-in, <=7
    uint32_t batch_slots=0; // single-card slot token histories, additional host metadata
    uint64_t hbm_limit=88ull<<30,pinned_limit=1ull<<30,host_limit=8ull<<30;
    uint64_t expert_cache=64ull<<30,staging=1ull<<20,driver_margin=4ull<<30;
    uint64_t snapshot_budget=1ull<<30;
    uint64_t cpu_cache=256ull<<20;
    // Set by --fp8-experts: the output head is decoded to BF16 and runs on the MME instead of the packed
    // TPC dot, so it is priced at its BF16 size.
    bool bf16_head=false;
    KvResidencyOptions kv; // --kv cap / --kv-grow / --no-kv-grow / --kv-resident, upstream defaults
    // --kv's storage format. It changes the cell's bytes (and therefore the priced pool), never the residency
    // controls above: FP16 cells stay two bytes per value, INT8 cells are codes plus one FP16 scale per 64.
    KvMode kv_mode=KvMode::Fp16;
};
struct ModelPlan {
    uint64_t dense_bytes=0,head_bytes=0,host_state_bytes=0,transaction_bytes=0,control_bytes=0;
    uint64_t slot_host_bytes=0;
    uint64_t cache_metadata_bytes=0; // bank/interrupted-read overhead, outside image pool
    uint64_t scalar_window_bytes=0; // root token/logit metadata; image uses snapshot pool
    uint64_t runtime_reserve=1ull<<30,host_compiler_reserve=2ull<<30;
    uint64_t hbm_required=0,host_required=0;
    uint64_t mtp_dense_bytes=0,mtp_state_bytes=0,mtp_cache_bytes=0;
    uint32_t mtp_window_rows=0;
    uint64_t mtp_window_host_bytes=0,mtp_window_hbm_bytes=0,mtp_window_pinned_bytes=0;
    uint64_t control_vector_host_bytes=0,control_vector_hbm_bytes=0;
    uint64_t packed_expert_execution_bytes=0;
    uint64_t scaled_expert_execution_bytes=0;
    // STRATA_HPU_RESIDENT_IQ4 adds these on top of the cache/router shell, never inside its cache budget.
    uint64_t resident_iq4_bytes=0,resident_iq4_staging_bytes=0;
    // STRATA_HPU_RESIDENT_U4 (with RESIDENT_IQ4) swaps the codebook store for the MME-readable one: uint4 codes
    // plus one F32 (scale, offset) per output channel and K group, so one batch GEMM reads every expert once.
    uint64_t resident_u4_bytes=0,resident_u4_staging_bytes=0;
    uint64_t native_expert_pack_host_bytes=0;
    // --embd-gguf: the materialised row block (rows x n_embd FP32) plus its metadata reserve. It is host
    // state only - upstream keeps the GGUF blocks and dequantizes per token, this port checks the rows once
    // at load, so the charge is the decoded block, not a second mapped copy of the file.
    uint64_t embedding_host_bytes=0;
    uint64_t embedding_ram_host_bytes=0; // optional canonical FP32 table plus allocation slack
    // Full mapped PLE payload pages and upstream's bounded pretouch workers.
    // OS lock success is a runtime fact, never implied by a successful plan.
    uint64_t ple_ram_table_bytes=0,ple_ram_host_bytes=0,ple_ram_worker_bytes=0;
    uint64_t ple_reader_host_bytes=0,ple_reader_extra_host_bytes=0,ple_reader_cache_rows=0,ple_reader_worker_bytes=0;
    // --expert-profile/--adapt-*: the routing counters, heat, residency table and the profile's prior rank,
    // all host-side (strata/hpu/expert_profile.hpp). Zero when the knobs are off.
    uint64_t expert_profile_host_bytes=0;
    // --dump-layers staging ladder + bounded --dump-routing physical-window
    // capture and one transient record export; charged by Model preflight.
    uint64_t dump_host_bytes=0;
    uint64_t routing_capture_host_bytes=0; // included in dump_host_bytes
    uint64_t half_capture_host_bytes=0,half_capture_hbm_bytes=0,half_capture_pinned_bytes=0;
    // --resident-experts / --resident-cpu-experts / --resident-budget-gib / --expert-cache-per-layer /
    // --mmap-experts (strata/hpu/expert_residency.hpp). The policy charges NO new bytes: it redistributes the
    // expert_cache budget already counted in hbm_required, and the RAM copy upstream page-locks is this
    // backend's mmap'd source. Reported so the split is visible in the plan, never inferred from a flag.
    uint64_t residency_per_layer_experts=0,residency_admitted=0;
    bool residency_per_layer_honored=false,resident_cpu_experts=false,resident_pin=false,resident_soft=false,
        mmap_experts=false;
    uint32_t prefill_rows=0;
    uint32_t window_rows=0,verify_rows=0;
    uint64_t prefill_host_bytes=0,prefill_hbm_bytes=0;
    // The K/V arena itself (FP16 cells, every QSA layer plus the MTP dense KV) and the cells --kv-resident
    // pins in it. Both are reported so the residency policy is visible in the plan, not only enforced.
    uint64_t kv_arena_bytes=0,kv_resident_cells=0;
    uint64_t verify_logits_bytes=0,verify_host_bytes=0,verify_hbm_bytes=0,verify_head_pinned_bytes=0;
    bool mtp_q4_projections=false,mtp_q4_head=false;
    std::vector<uint64_t> layer_dense_bytes;
};
// Full trunk and PLE schema/memory preflight, without initializing Synapse or
// acquiring a device. Runtime allocation caps still guard actual buffer sizes.
ModelPlan plan_model(const WeightTable&,const ModelMemoryOptions&,const std::string& external_ple={},const strata::kernels::PleIoOptions& io={},bool ple_enabled=true);
void extend_resident_iq4_plan(ModelPlan&,const core::ModelGeometry&,const ModelMemoryOptions&);
void extend_resident_u4_plan(ModelPlan&,const core::ModelGeometry&,const ModelMemoryOptions&);
void extend_mtp_plan(ModelPlan&,const MtpWeights&,const ModelMemoryOptions&,uint64_t expert_cache,uint32_t window=0,uint32_t subset=0,uint32_t head_type=30,bool q4_projections=false,bool q4_head=false);
}
