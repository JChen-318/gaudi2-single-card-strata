#pragma once
#include "strata/hpu/model_plan.hpp"
#include <ostream>

namespace strata::hpu {
// Only call after a no-device preflight, never as a runtime health claim.
inline void write_plan_json(std::ostream& out,const ModelPlan& plan) {
    out<<"{\"dense_bytes\":"<<plan.dense_bytes<<",\"head_bytes\":"<<plan.head_bytes
       <<",\"host_state_bytes\":"<<plan.host_state_bytes<<",\"transaction_bytes\":"<<plan.transaction_bytes
       <<",\"control_bytes\":"<<plan.control_bytes<<",\"slot_host_bytes\":"<<plan.slot_host_bytes<<",\"cache_metadata_bytes\":"<<plan.cache_metadata_bytes<<",\"hbm_required\":"<<plan.hbm_required
       <<",\"scalar_window_bytes\":"<<plan.scalar_window_bytes<<",\"host_required\":"<<plan.host_required<<",\"mtp_dense_bytes\":"<<plan.mtp_dense_bytes
       <<",\"mtp_state_bytes\":"<<plan.mtp_state_bytes<<",\"mtp_cache_bytes\":"<<plan.mtp_cache_bytes
       <<",\"mtp_window_rows\":"<<plan.mtp_window_rows<<",\"mtp_window_host_bytes\":"<<plan.mtp_window_host_bytes<<",\"mtp_window_hbm_bytes\":"<<plan.mtp_window_hbm_bytes<<",\"mtp_window_pinned_bytes\":"<<plan.mtp_window_pinned_bytes
       <<",\"control_vector_host_bytes\":"<<plan.control_vector_host_bytes<<",\"control_vector_hbm_bytes\":"<<plan.control_vector_hbm_bytes
       <<",\"packed_expert_execution_bytes\":"<<plan.packed_expert_execution_bytes
       <<",\"scaled_expert_execution_bytes\":"<<plan.scaled_expert_execution_bytes
       <<",\"native_expert_pack_host_bytes\":"<<plan.native_expert_pack_host_bytes
       <<",\"embedding_host_bytes\":"<<plan.embedding_host_bytes
       <<",\"ple_ram_table_bytes\":"<<plan.ple_ram_table_bytes
       <<",\"ple_ram_host_bytes\":"<<plan.ple_ram_host_bytes
       <<",\"ple_ram_worker_bytes\":"<<plan.ple_ram_worker_bytes
       <<",\"ple_reader_host_bytes\":"<<plan.ple_reader_host_bytes
       <<",\"ple_reader_extra_host_bytes\":"<<plan.ple_reader_extra_host_bytes
       <<",\"ple_reader_cache_rows\":"<<plan.ple_reader_cache_rows
       <<",\"ple_reader_worker_bytes\":"<<plan.ple_reader_worker_bytes
       <<",\"expert_profile_host_bytes\":"<<plan.expert_profile_host_bytes
       <<",\"dump_host_bytes\":"<<plan.dump_host_bytes
       <<",\"routing_capture_host_bytes\":"<<plan.routing_capture_host_bytes
       <<",\"half_capture_host_bytes\":"<<plan.half_capture_host_bytes
       <<",\"half_capture_hbm_bytes\":"<<plan.half_capture_hbm_bytes
       <<",\"half_capture_pinned_bytes\":"<<plan.half_capture_pinned_bytes
       <<",\"residency_per_layer_experts\":"<<plan.residency_per_layer_experts
       <<",\"residency_per_layer_honored\":"<<(plan.residency_per_layer_honored?"true":"false")
       <<",\"residency_admitted\":"<<plan.residency_admitted
       <<",\"resident_cpu_experts\":"<<(plan.resident_cpu_experts?"true":"false")
       <<",\"resident_pin\":"<<(plan.resident_pin?"true":"false")
       <<",\"resident_soft\":"<<(plan.resident_soft?"true":"false")
       <<",\"mmap_experts\":"<<(plan.mmap_experts?"true":"false")
       <<",\"prefill_rows\":"<<plan.prefill_rows<<",\"prefill_host_bytes\":"<<plan.prefill_host_bytes<<",\"prefill_hbm_bytes\":"<<plan.prefill_hbm_bytes
       <<",\"window_rows\":"<<plan.window_rows<<",\"verify_rows\":"<<plan.verify_rows<<",\"verify_logits_bytes\":"<<plan.verify_logits_bytes<<",\"verify_host_bytes\":"<<plan.verify_host_bytes
       <<",\"verify_hbm_bytes\":"<<plan.verify_hbm_bytes<<",\"verify_head_pinned_bytes\":"<<plan.verify_head_pinned_bytes
       <<",\"mtp_q4_projections\":"<<(plan.mtp_q4_projections?"true":"false")<<",\"mtp_q4_head\":"<<(plan.mtp_q4_head?"true":"false");
    if(plan.resident_iq4_bytes) out<<",\"resident_iq4_bytes\":"<<plan.resident_iq4_bytes<<",\"resident_iq4_staging_bytes\":"<<plan.resident_iq4_staging_bytes;
    if(plan.resident_u4_bytes) out<<",\"resident_u4_bytes\":"<<plan.resident_u4_bytes<<",\"resident_u4_staging_bytes\":"<<plan.resident_u4_staging_bytes;
    if(plan.embedding_ram_host_bytes) out<<",\"embedding_ram_host_bytes\":"<<plan.embedding_ram_host_bytes;
    out<<",\"layers\":"<<plan.layer_dense_bytes.size()<<",\"no_device_acquired\":true}\n";
}
}
