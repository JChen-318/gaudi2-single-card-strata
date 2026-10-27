#include "strata/hpu/model.hpp"
#include "strata/hpu/simd_norm.hpp"
#include "strata/hpu/performance_modes.hpp"
#include "strata/hpu/moe_mme.hpp"
#include "strata/hpu/resident_experts.hpp"
#include "strata/hpu/dflash.hpp"
#include <perf_lib_layer_params.h>
#include <chrono>
#include <bit>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <exception>
#include <future>
#include "strata/hpu/stop_tokens.hpp"
#include "strata/hpu/ple_io_mode.hpp"
#include "strata/hpu/mtp_window_plan.hpp"
#include "strata/hpu/hc_ple.hpp"
#include "strata/hpu/hc_window.hpp"
#include "strata/hpu/gdn_window.hpp"
#include "strata/hpu/qsa_window.hpp"
#include "strata/hpu/qsa_window_flow.hpp"
#include "strata/hpu/host_recipe.hpp"
#include "strata/hpu/qsa.hpp"
#include "strata/hpu/model_head.hpp"
#include "strata/hpu/ple_input.hpp"
#include "strata/hpu/ple_mode.hpp"
#include "strata/hpu/embedding_route.hpp"
#include "strata/hpu/cpu_experts.hpp"
#include "strata/hpu/state_io.hpp"
#include "strata/hpu/image_state.hpp"
#include "strata/hpu/control_graph.hpp"
#include "strata/hpu/packed_quant.hpp"
#include "strata/hpu/native_pack.hpp"
#include "strata/hpu/prefill_plan.hpp"
#include "strata/hpu/expert_residency.hpp"
#include "strata/hpu/expert_format.hpp"
#include "strata/hpu/expert_profile.hpp"
#include "strata/hpu/snapshot_memory.hpp"
#include "strata/hpu/head_mixed.hpp"
#include "strata/hpu/routing_capture.hpp"
#include "strata/hpu/half_capture.hpp"
#include <array>
#include <cstring>
#include <algorithm>
#include <atomic>
#include <mutex>
#include <cmath>
#include <cstdlib>
#include <filesystem>

namespace strata::hpu {
namespace {std::atomic<uint64_t> next_model_owner{1};}
struct Model::Snapshot::Data {
    uint64_t owner=0;
    SnapshotReservation allocation; // destroyed AFTER every owned buffer below
    std::vector<Layer::Checkpoint> layers;
    std::vector<int32_t> tokens;
    std::vector<float> logits,residual,mixed;
    bool has_mtp=false;
    bool control_enabled=true;
    DenseKvState::Checkpoint mtp;
    std::vector<float> mtp_logits,mtp_residual;
    Data(uint64_t id,std::shared_ptr<AllocationBudget> b,uint64_t bytes):owner(id),allocation(std::move(b),bytes) {}
    Data(const Data&)=delete;
    Data& operator=(const Data&)=delete;
    void validate_capacity(uint64_t envelope) const {
        uint64_t actual=sizeof(*this);
        for(auto n:{snapshot_vector_bytes(tokens),snapshot_vector_bytes(logits),snapshot_vector_bytes(residual),snapshot_vector_bytes(mixed),snapshot_vector_bytes(mtp_logits),snapshot_vector_bytes(mtp_residual),snapshot_vector_bytes(layers)}) actual=snapshot_add_bytes(actual,n);
        for(const auto& point:layers) actual=snapshot_add_bytes(actual,Layer::snapshot_capacity_bytes(point));
        actual=snapshot_add_bytes(actual,DenseKvState::snapshot_capacity_bytes(mtp));
        if(actual>envelope) throw AllocationBudgetExceeded();
    }
};
// STRATA_HPU_DFLASH_LOGITS=<path>: the diagnostic for the one question argmaxes cannot answer. The verify window and
// sequential decode pick different tokens somewhere; whether that is a hair-thin tie (two arithmetic routes rounding
// differently) or a real state error is only visible in the logits. Off by default, and off costs one pointer test.
// Every COMMITTED window row is appended as `vocabulary` float32 values, and `<path>.idx` carries
// "<position> <depth> <byte_offset>" per row: a window publishes several rows at once, so a bare row stream cannot say
// which row is which. This is a new file, not upstream's --dump-logits layout, and makes no compatibility claim.
class DflashLogitSink {
public:
    explicit DflashLogitSink(const std::string& path) {
        rows_ = std::fopen(path.c_str(), "wb");
        if(!rows_) throw std::invalid_argument("cannot write " + path);
        const auto index = path + ".idx";
        index_ = std::fopen(index.c_str(), "wb");
        if(!index_) { std::fclose(rows_); rows_ = nullptr; throw std::invalid_argument("cannot write " + index); }
    }
    ~DflashLogitSink() { if(rows_) std::fclose(rows_); if(index_) std::fclose(index_); }
    void record(uint64_t position, uint32_t depth, uint32_t vocabulary, const float* values) {
        if(std::fwrite(values, sizeof(float) * vocabulary, 1, rows_) != 1) throw std::runtime_error("DFlash logit sink write failed");
        std::fflush(rows_);
        std::fprintf(index_, "%llu %u %llu\n", static_cast<unsigned long long>(position), depth,
                     static_cast<unsigned long long>(bytes_));
        std::fflush(index_);
        bytes_ += static_cast<uint64_t>(vocabulary) * sizeof(float);
    }
private:
    std::FILE* rows_ = nullptr;
    std::FILE* index_ = nullptr;
    uint64_t bytes_ = 0;
};

struct Model::Impl {
    WeightTable table;ModelOptions options;PleMode ple_mode;ModelPlan memory_plan;
    strata::kernels::RopeScaling rope;
    // --vision: upstream's mrope_host (:3197-3215), cells = max_context + 64, the identity until a GENI
    // request. The table lives for the whole Model so every QSA/MTP config borrows one address.
    RopePositions positions;
    std::shared_ptr<const VisionEmbeddings> vision_embeddings;
    std::vector<int64_t> vision_rows; // prompt-cell image rows, -1 for text
    std::string weight_identity;
    std::vector<float> last_logits,last_residual,pending_residual;
    // STRATA_HPU_CHAIN_HEAD: the decode chain already ran the head; op.head returns these logits.
    bool chain_head_ready=false;
    HeadMixed mixed_output;
    // Upstream's retained-reuse slot (include/strata/core/conversation_cache.hpp:217-233): the parked image of
    // the conversation this model serves, held so the next park of the SAME conversation is grown in place
    // instead of re-copying the whole K/V (src/core/conversation_snapshot.cpp:147-181). Exclusivity is the
    // contract: a cache entry that still references the image owns it, and PrefixCache drops that copy as
    // superseded before the next capture. The pointee is a Model::Snapshot::Data, held as void because that type
    // is private to Snapshot and Impl is not its friend.
    std::shared_ptr<void> parked;
    std::vector<int32_t> parked_tokens;
    uint64_t parked_reused=0,parked_reused_total=0;
    // Diagnostic capture (ModelOptions::capture_layer_ladder / capture_routing); both free when off.
    std::vector<float> ladder;
    RoutingCapture routing_capture;
    HalfCapture half_capture;
    std::optional<ControlVectors> control_vectors;
    std::shared_ptr<const NativeExpertPack> expert_pack;
    bool control_enabled=true;
    std::optional<MtpWeights> mtp_source;
    std::optional<DraftVocabulary> mtp_vocabulary;
    std::vector<float> last_mtp_logits,pending_mtp_logits,last_mtp_residual,pending_mtp_residual,pending_embedding;
    // Declared in lifetime order: recipes/layers die before shared weights,
    // expert cache and device. Flow callbacks may borrow this Impl.
    std::unique_ptr<Device> device;
#if STRATA_HPU_CPU_EXPERTS
    std::unique_ptr<CpuExperts> cpu;
#endif
    std::unique_ptr<ExpertCache> cache;
    std::unique_ptr<DenseWeights> head_weights;
    std::unique_ptr<HostRecipe> final_mix;
    std::unique_ptr<ModelHead> head;
    std::unique_ptr<HostRecipe> control_recipe;
    std::unique_ptr<PleInput> ple;
    std::vector<std::unique_ptr<Layer>> layers;
    // Destroy shared recipes before the layer weights they last borrowed.
    std::unique_ptr<HcWindow> hc_window;
    std::unique_ptr<GdnWindow> gdn_window;
    std::unique_ptr<QsaWindow> qsa_window;
    std::unique_ptr<MtpBlock> mtp;
    WeightSlice embedding;
    std::vector<float> embedding_ram;
    // --embd-gguf: the loaded row file, the one route every seam reads, and the positional cursor the
    // windowed seams open. Inactive by default, so the default path is the canonical table gather.
    std::optional<EmbeddingInput> embedding_input;
    EmbeddingRows embedding_rows;
    EmbeddingWindow embedding_window;
    std::vector<int32_t> stops;
    uint64_t owner=next_model_owner++;
    bool poisoned=false;
    // Resident prefill windows: per layer, the events of its early QSA stage download (prefill_resident_window).
    std::vector<std::unique_ptr<Event>> layer_ready,layer_copied;
    std::shared_ptr<AllocationBudget> snapshot_budget=std::make_shared<AllocationBudget>();
    std::unique_ptr<SessionFlow> flow;
    mutable std::mutex mutex;
    Impl(const std::string& path,ModelOptions o):table(path),options(std::move(o)),ple_mode(resolve_ple_mode(options.no_ple,options.ple_gguf)),
        memory_plan(plan_model(table,options.memory,ple_mode.external,options.ple_io,ple_mode.enabled)),
        positions(options.vision?uint64_t(options.memory.context)+64:0) {
        validate_ple_io_policy(options.ple_io);
        parse_layer_windows(std::getenv("STRATA_HPU_PREFILL_LAYERS"));
        if(tpc_colpair_enabled() && (!tpc_experts_enabled() || !direct_experts_enabled() || !options.fp8_experts ||
           tpc_hilo_enabled() || options.memory.prefill_rows || options.prefill_auto || !options.mtp_runtime.empty() ||
           options.hpu_load_budget!=UINT32_MAX || table.geometry().n_embd%128 || table.geometry().n_ff%128 ||
           table.geometry().n_embd>2560 || table.geometry().n_ff>2560 || table.geometry().n_expert_used!=10))
            throw std::invalid_argument("STRATA_HPU_TPC_COLPAIR requires TPC_EXPERTS, DIRECT_EXPERTS, --fp8-experts, dimensions divisible by128; no HILO, prefill window, MTP or CPU expert fallback");
        if(resident_iq4_enabled()) {
            // Small windows reuse DFlash verification. Wider explicit windows use
            // the layer-major path: native causal chunks, whole-window resident MoE.
            const bool wide=memory_plan.prefill_rows>kResidentWindowRows;
            if(memory_plan.prefill_rows>kGdnWindowRows && !q8_packed_enabled())
                throw std::invalid_argument("resident IQ4 prefill >8 requires STRATA_HPU_Q8_PACKED");
            if(wide && checked_product(uint64_t(memory_plan.prefill_rows),uint64_t(table.geometry().n_expert_used)+1)>kResidentMoeSlots)
                throw std::invalid_argument("resident IQ4 prefill exceeds4096 expert slots");
            if(options.memory.prefill_rows>=2 && !qsa_device_enabled())
                throw std::invalid_argument("STRATA_HPU_RESIDENT_IQ4 prefill window needs STRATA_HPU_QSA_DEVICE");
            if(!chain_decode_enabled() || !options.fp8_experts || options.prefill_auto || !options.mtp_runtime.empty() ||
               options.hpu_load_budget!=UINT32_MAX || options.capture_routing)
                throw std::invalid_argument("STRATA_HPU_RESIDENT_IQ4 requires STRATA_HPU_CHAIN and --fp8-experts (its router/cache shell); no automatic prefill rows, MTP, CPU expert fallback or routing capture");
            // U4 shares the IQ4 shell (router, Q8 activations, chain) but not its 1280-row/256-column tiling, so its
            // weight check is the store's own geometry.
            const bool u4=resident_u4_enabled();
            const auto unsupported=u4?resident_u4_unsupported(table):resident_iq4_unsupported(table);
            if(!unsupported.empty()) throw std::invalid_argument(std::string(u4?"STRATA_HPU_RESIDENT_U4: ":"STRATA_HPU_RESIDENT_IQ4: ")+unsupported);
            if(prefill_final_enabled() && table.geometry().ssm_d_conv>4)
                throw std::invalid_argument("STRATA_HPU_PREFILL_FINAL requires GDN taps <=4");
            if(prefill_local_enabled() && (!prefill_final_enabled() || table.geometry().ssm_state_size>128))
                throw std::invalid_argument("STRATA_HPU_PREFILL_LOCAL requires PREFILL_FINAL and GDN width <=128");
            if(prefill_qsa32_enabled() && (!prefill_final_enabled() ||
               (memory_plan.prefill_rows!=64 && memory_plan.prefill_rows!=128 && memory_plan.prefill_rows!=256)))
                throw std::invalid_argument("STRATA_HPU_PREFILL_QSA32 requires PREFILL_FINAL and --prefill 64,128 or256");
            if(resident_u4_enabled()) extend_resident_u4_plan(memory_plan,table.geometry(),options.memory);
            else extend_resident_iq4_plan(memory_plan,table.geometry(),options.memory);
            // Small resident windows use one verify chain shared with DFlash;
            // replace the unused generic reservation only for that path. Wide
            // windows retain the pools and reservation plan_model admitted.
            if(memory_plan.window_rows && !wide) {
                if(memory_plan.prefill_host_bytes>memory_plan.host_required || memory_plan.prefill_hbm_bytes>memory_plan.hbm_required ||
                   memory_plan.prefill_hbm_bytes>memory_plan.runtime_reserve)
                    throw std::logic_error("resident IQ4 prefill reservation mismatch");
                memory_plan.host_required-=memory_plan.prefill_host_bytes;
                memory_plan.hbm_required-=memory_plan.prefill_hbm_bytes;
                memory_plan.runtime_reserve-=memory_plan.prefill_hbm_bytes;
                memory_plan.prefill_host_bytes=memory_plan.prefill_hbm_bytes=0;
                memory_plan.window_rows=0;
            }
            if(const uint32_t rows=std::min<uint32_t>(memory_plan.prefill_rows,kResidentWindowRows);rows>=2 && !wide) {
                // Exactly what VerifyChain allocates below for the window, plus one recipe allowance for the head
                // and the per-layer verify graphs (the DFlash round shares that same allowance with runtime_reserve).
                const auto& g=table.geometry();const uint64_t n=checked_product(uint64_t(g.n_embd),uint64_t(g.hc));
                const auto qc=qsa_config(g,options.memory.context,options.memory.page_size);
                const uint64_t stage=std::max<uint64_t>(verify_stage_bytes(g,rows),128);
                // Final-state prefill windows keep one PLE history row; a DFlash round its block (at most kGdnWindowRows).
                const uint64_t history_rows=prefill_final_enabled()?std::min<uint64_t>(rows,kGdnWindowRows):rows;
                const uint64_t history=std::max<uint64_t>(checked_product(checked_product(history_rows,n),uint64_t(9*sizeof(float))),128);
                const uint64_t mask=std::max<uint64_t>(checked_product(uint64_t(rows),checked_product(uint64_t(qc.selection_width(qc.capacity))+rows,uint64_t(sizeof(float)))),128)+(prefill_qsa32_enabled()?128:0);
                // One row of logits when only the last-row prefill head writes them (VerifyChain's logit_rows).
                const char* dflash_verify=std::getenv("STRATA_HPU_DFLASH_VERIFY");
                const bool full_logits=!prefill_last_head_enabled() || (std::getenv("STRATA_HPU_DFLASH") && *std::getenv("STRATA_HPU_DFLASH") &&
                                                                         dflash_verify && *dflash_verify && *dflash_verify!='0');
                // The last-row head writes one row; DFlash verification a draft block (at most kGdnWindowRows rows).
                const uint64_t logit_rows=!prefill_last_head_enabled()?rows:full_logits?std::min<uint64_t>(rows,kGdnWindowRows):1;
                const uint64_t logits=checked_product(logit_rows,checked_product(uint64_t(g.n_vocab),uint64_t(sizeof(float))));
                const uint64_t mixed=checked_product(uint64_t(rows),checked_product(uint64_t(g.n_embd),uint64_t(sizeof(float))));
                // Long-context QSA verify beyond the device cache's identity selection: a projection recipe per
                // row count and a finish recipe per (bucket, rows) whose attention window is its pinned and
                // resident payload. The window runs at the widest bucket once the cache is full, and a DFlash
                // drafter block can use a second row count, so two widest-bucket copies are priced.
                // The staged (gathered) window only runs once the context outgrows the device cache's identity
                // selection, which covers cells + rows <= top_k + block - 1. A context that fits inside that never
                // stages, so pricing two widest-bucket copies of it would refuse windows the identity path alone
                // serves -- which is exactly what blocked the 512/1024-row windows of round 33.
                const bool staged_reachable=uint64_t(options.memory.context)+1>
                    uint64_t(qc.top_k)+uint64_t(qc.block>0?qc.block:1)-1;
                const uint32_t staged_rows=staged_reachable?(qsa_chunk_rows()?std::max(std::min(rows,qsa_chunk_rows()),kQsaWindowRows):std::max(rows,kQsaWindowRows)):0;
                // STRATA_HPU_QSA_DEVICE_SELECT runs those windows as select recipes (HBM only, nothing pinned): the
                // shared workspace and every key bucket's own tensors (qsa_select_window_bytes).
                const bool select=staged_reachable && qsa_device_select_enabled() && qsa_device_gather_enabled();
                const uint64_t staged=select?qsa_select_window_bytes(qc,rows,options.memory.context):
                    staged_reachable?checked_product(2,qsa_window_attention_pinned_bytes(qc,staged_rows,
                                                                                         qc.selection_width(qc.capacity))+(16ull<<20)):0;
                const uint64_t staged_host=select?0:staged_reachable && (qsa_host_budget_enabled() || qsa_chunk_rows()) && qsa_device_gather_enabled()
                    ?checked_product(2,qsa_window_gather_pinned_bytes(qc,staged_rows,qc.selection_width(qc.capacity))+
                        checked_product(checked_product(staged_rows,uint64_t(g.hc)),4)+4+(16ull<<20))
                    :staged;
                uint64_t gdn_layers=0;for(int64_t i=0;i<g.n_layers;++i) if(!core::is_qsa_layer(g,uint32_t(i))) ++gdn_layers;
                const uint64_t cells=checked_product(uint64_t(g.ssm_state_size)*g.ssm_state_size,uint64_t(g.ssm_v_heads))+
                                     checked_product(uint64_t(g.ssm_d_conv-1),uint64_t(g.ssm_conv_channels));
                // The baseline runtime reserve already covers8 row-state pairs.
                const uint64_t extra_states=rows>8 && !prefill_final_enabled()?checked_product(checked_product(gdn_layers,rows-8),checked_product(cells,4)):0;
                // The canonical 32-row chunk recipes grow with the window, so their allowance is one MiB per row
                // (64 MiB at 64 rows, 256 MiB at 256).
                const uint64_t recipe_allowance=(64ull<<20)+(prefill_qsa32_enabled()?uint64_t(rows)<<20:0);
                // VerifyChain's shared scratch for the window graphs' internal tensors (decode_chain.hpp).
                const uint64_t scratch=checked_product(uint64_t(rows),checked_product(uint64_t(g.n_embd),uint64_t(4*(uint64_t(g.hc)+4)))+4096);
                const uint64_t hbm=checked_product(checked_product(n,uint64_t(rows)),uint64_t(3*sizeof(float)))+stage+history+mask+logits+mixed+staged+extra_states+scratch;
                if(memory_plan.hbm_required>options.memory.hbm_limit || hbm+recipe_allowance>options.memory.hbm_limit-memory_plan.hbm_required ||
                   memory_plan.host_required>options.memory.host_limit || stage+history+mask+logits+mixed+staged_host+recipe_allowance>options.memory.host_limit-memory_plan.host_required)
                    throw std::invalid_argument("resident IQ4 prefill window exceeds the memory policy: rows="+std::to_string(rows)+
                        " window_hbm_mib="+std::to_string((hbm+recipe_allowance)>>20)+
                        " plan_hbm_mib="+std::to_string(memory_plan.hbm_required>>20)+
                        " hbm_limit_mib="+std::to_string(options.memory.hbm_limit>>20)+
                        " window_host_mib="+std::to_string((stage+history+mask+logits+mixed+staged_host+recipe_allowance)>>20)+
                        " plan_host_mib="+std::to_string(memory_plan.host_required>>20)+
                        " host_limit_mib="+std::to_string(options.memory.host_limit>>20)+
                        " staged_mib="+std::to_string(staged>>20)+" extra_gdn_states_mib="+std::to_string(extra_states>>20));
                memory_plan.prefill_hbm_bytes=hbm+recipe_allowance;memory_plan.prefill_host_bytes=stage+history+mask+logits+mixed+staged_host+recipe_allowance;
                memory_plan.runtime_reserve+=memory_plan.prefill_hbm_bytes;memory_plan.hbm_required+=memory_plan.prefill_hbm_bytes;
                memory_plan.host_required+=memory_plan.prefill_host_bytes;
            }
        }
        if(memory_plan.window_rows>128 && !resident_iq4_enabled())
            throw std::invalid_argument("a prefill window wider than128 rows needs STRATA_HPU_RESIDENT_IQ4");
        if(qsa_device_enabled() && !resident_iq4_enabled()) throw std::invalid_argument("STRATA_HPU_QSA_DEVICE requires STRATA_HPU_RESIDENT_IQ4");
        if(qsa_chunk_rows() && (!resident_iq4_enabled() || !qsa_device_gather_enabled() || !prefill_final_enabled() ||
                               memory_plan.prefill_rows<=qsa_chunk_rows() || memory_plan.prefill_rows>kResidentWindowRows))
            throw std::invalid_argument("STRATA_HPU_QSA_CHUNK_ROWS requires resident IQ4, QSA_DEVICE_GATHER, PREFILL_FINAL and a larger prefill window <=256");
        if(qsa_device_gather_enabled()) {
            if(!resident_iq4_enabled() || !qsa_device_enabled() || options.memory.kv_mode!=KvMode::Fp16 ||
               options.memory.context>262144 || table.geometry().head_dim%64)
                throw std::invalid_argument("STRATA_HPU_QSA_DEVICE_GATHER requires RESIDENT_IQ4, QSA_DEVICE, FP16 KV, aligned width and context <=262144");
            // Full F32 mirrors of the already FP16-rounded K/V (STRATA_HPU_QSA_KV_FP8: FP8 codes and per-head
            // scales), with bounded upload chunks even when restoring an entire prefix.
            const auto& g=table.geometry();uint64_t qsa_layers=0;
            for(int64_t i=0;i<g.n_layers;++i) if(core::is_qsa_layer(g,uint32_t(i))) ++qsa_layers;
            const auto qc=qsa_config(g,options.memory.context,options.memory.page_size);
            const uint64_t row=qsa_cache_row_bytes(qc)+(qsa_kv_fp8_enabled()?qsa_cache_scale_bytes(qc):0);
            const uint64_t bytes=checked_product(qsa_layers,checked_product(uint64_t(options.memory.context),checked_product(row,2)));
            const uint64_t staging=checked_product(qsa_layers,checked_product(uint64_t(std::min(options.memory.context,128u)),
                                               checked_product(uint64_t(g.n_head_kv)*g.head_dim,8)));
            if(memory_plan.hbm_required>options.memory.hbm_limit || bytes>options.memory.hbm_limit-memory_plan.hbm_required ||
               memory_plan.host_required>options.memory.host_limit || staging>options.memory.host_limit-memory_plan.host_required)
                throw std::invalid_argument("QSA full device gather cache exceeds memory policy");
            memory_plan.hbm_required+=bytes;memory_plan.runtime_reserve+=bytes;memory_plan.host_required+=staging;
        }
        if(merged_gemm_enabled() && !resident_iq4_enabled()) throw std::invalid_argument("STRATA_HPU_MERGED_GEMM requires STRATA_HPU_RESIDENT_IQ4");
        if(fp8_dense_enabled() && !merged_gemm_enabled()) throw std::invalid_argument("STRATA_HPU_FP8_DENSE requires STRATA_HPU_MERGED_GEMM");
        if(fp8_static_enabled() && !fp8_dense_enabled()) throw std::invalid_argument("STRATA_HPU_FP8_STATIC requires STRATA_HPU_FP8_DENSE");
        if(fp8_dense_enabled()) {
            // The FP8 copies sit beside the BF16 matrices: at most half their bytes, plus 4 bytes of scale per row of
            // at least 64 columns (1/32 more).
            uint64_t extra=0;
            for(auto& bytes:memory_plan.layer_dense_bytes) {const uint64_t more=bytes/2+bytes/32+4096;bytes+=more;extra+=more;}
            if(memory_plan.hbm_required>options.memory.hbm_limit || extra>options.memory.hbm_limit-memory_plan.hbm_required)
                throw std::invalid_argument("STRATA_HPU_FP8_DENSE copies exceed the HBM memory policy");
            memory_plan.hbm_required+=extra;
        }
        if(tpc_hilo_enabled() && (!tpc_experts_enabled() || !direct_experts_enabled() || !options.fp8_experts))
            throw std::invalid_argument("STRATA_HPU_TPC_HILO requires TPC_EXPERTS, DIRECT_EXPERTS and --fp8-experts");
        if(direct_experts_enabled() && (!options.fp8_experts || table.geometry().n_ff%32 || table.geometry().n_embd%32))
            throw std::invalid_argument("STRATA_HPU_DIRECT_EXPERTS requires --fp8-experts and dimensions divisible by 32");
        (void)resolve_ple_delay_us(options.ple_delay_us);
        if(options.prefill_auto && options.memory.prefill_rows) throw std::invalid_argument("automatic and explicit model prefill rows are mutually exclusive");
        if(!ple_mode.external.empty()) ple_mode.external=std::filesystem::canonical(ple_mode.external).string();
        options.ple_gguf=ple_mode.external;
        if(!options.embedding_gguf.empty()) {
            // Upstream loads and installs the file before the session exists (:3032-3037) and binds it into
            // the checkpoint fingerprint (:8155 / conversation_file.cpp:935). Loading here keeps the same
            // order: a bad row file is refused by --plan as well, before any device exists.
            options.embedding_gguf=std::filesystem::canonical(options.embedding_gguf).string();
            embedding_input.emplace(EmbeddingInput::load(options.embedding_gguf,uint64_t(table.geometry().n_embd),uint64_t(table.geometry().n_vocab)));
            const auto bytes=checked_product(checked_product(embedding_input->rows(),uint64_t(table.geometry().n_embd)),uint64_t(sizeof(float)))+uint64_t(256)*1024;
            if(bytes>options.memory.host_limit-memory_plan.host_required) throw std::invalid_argument("embedding row file exceeds host memory policy");
            memory_plan.embedding_host_bytes=bytes;memory_plan.host_required+=bytes;memory_plan.control_bytes+=bytes;
        }
        // ---- upstream's expert-cache profiling/adaptation counters, charged before the device exists.
        if(options.expert_profile.monitoring()) {
            const auto entries=checked_product(uint64_t(table.geometry().n_layers),uint64_t(table.geometry().n_expert));
            const auto per_entry=uint64_t(sizeof(float)+2*sizeof(uint64_t)+sizeof(double)+sizeof(int32_t));
            const auto bytes=checked_product(entries,per_entry)+(uint64_t(1)<<20);
            if(bytes>options.memory.host_limit-memory_plan.host_required) throw std::invalid_argument("expert profile counters exceed host memory policy");
            memory_plan.expert_profile_host_bytes=bytes;memory_plan.host_required+=bytes;memory_plan.control_bytes+=bytes;
        }
        // ---- upstream's expert-cache residency knobs (strata/hpu/expert_residency.hpp), resolved and refused
        // before a device exists. Two of upstream's conditions come from other options and are made single
        // truths here: the static profile is this port's --expert-profile, and one Gaudi2 has no remote expert
        // cache to exclude. --mmap-experts is inherent (expert rows are read from the mmap'd GGUF and never
        // copied into host RAM), so the flag is parsed, validated as a combination and reported instead of
        // switching a read path that has no second form.
        options.residency.has_expert_profile=!options.expert_profile.profile.empty();
        options.residency.remote_caches=false;
        if(const auto refusal=validate_expert_residency(options.residency);!refusal.empty()) throw std::invalid_argument(refusal);
        memory_plan.resident_cpu_experts=options.residency.resident_cpu_experts;
        memory_plan.resident_pin=options.residency.resident_pin;
        memory_plan.resident_soft=options.residency.resident_soft;
        memory_plan.mmap_experts=options.residency.mmap_experts;
        {
            // The per-layer split (--expert-cache-per-layer) and the profile fill, over the geometry the plan
            // already validated. Both read metadata only: no expert byte is touched and no device is opened.
            const auto& rg=table.geometry();const int64_t layers=int64_t(rg.n_layers);
            std::vector<uint64_t> blob;blob.reserve(size_t(layers));
            for(int64_t l=0;l<layers;++l) {
                const auto prefix="blk."+std::to_string(l)+".";
                const std::array<uint32_t,3> types{table.info(prefix+"ffn_gate_exps.weight").ggml_type,
                    table.info(prefix+"ffn_up_exps.weight").ggml_type,table.info(prefix+"ffn_down_exps.weight").ggml_type};
                blob.push_back(expert_format(uint64_t(rg.n_embd),uint64_t(rg.n_ff),0,types).bytes);
            }
            const auto split=per_layer_byte_split(options.memory.expert_cache,blob,int64_t(rg.n_expert));
            memory_plan.residency_per_layer_experts=options.residency.expert_cache_per_layer?split.q:0;
            memory_plan.residency_per_layer_honored=options.residency.expert_cache_per_layer && split.honored;
            if(options.residency.resident_cpu_experts && options.residency.has_expert_profile) {
                std::vector<std::pair<int32_t,int32_t>> ranked;int64_t profile_slots=0;std::string error;
                if(!read_expert_profile(options.expert_profile.profile,layers,int64_t(rg.n_expert),ranked,profile_slots,error))
                    throw std::invalid_argument("expert profile: "+error);
                uint64_t slot_bytes=0;for(const uint64_t b:blob) slot_bytes=std::max(slot_bytes,round256(b));
                const ExpertCacheShape shape{slot_bytes?int64_t(options.memory.expert_cache/slot_bytes):0,
                    layers,int64_t(rg.n_expert)};
                memory_plan.residency_admitted=uint64_t(admitted_before_routing(options.residency,shape,ranked).size());
            }
        }
        if(!options.expert_pack.empty()) {
            if(NativeExpertPack::host_reserve>options.memory.host_limit-memory_plan.host_required) throw std::invalid_argument("native expert pack exceeds host memory policy");
            memory_plan.native_expert_pack_host_bytes=NativeExpertPack::host_reserve;
            memory_plan.host_required+=NativeExpertPack::host_reserve;memory_plan.control_bytes+=NativeExpertPack::host_reserve;
            expert_pack=std::make_shared<NativeExpertPack>(options.expert_pack,table);
        }
        rope=resolve_rope(table,options.rope);
        if(!options.control_vector.files.empty()) {
            const auto& g=table.geometry();control_vectors.emplace(options.control_vector,uint32_t(g.n_layers),uint32_t(g.n_embd));
            memory_plan.control_vector_host_bytes=control_vectors->host_bytes()*2+(1ull<<20);
            memory_plan.control_vector_hbm_bytes=(uint64_t(g.n_embd)*(g.hc*3+1)+g.hc*2+1)*4+(1ull<<20);
            if(memory_plan.control_vector_host_bytes>options.memory.host_limit-memory_plan.host_required ||
               memory_plan.control_vector_hbm_bytes>options.memory.hbm_limit-memory_plan.hbm_required) throw std::invalid_argument("control vector model plan exceeds HBM/host memory policy");
            memory_plan.host_required+=memory_plan.control_vector_host_bytes;memory_plan.hbm_required+=memory_plan.control_vector_hbm_bytes;
            memory_plan.control_bytes+=memory_plan.control_vector_host_bytes;memory_plan.runtime_reserve+=memory_plan.control_vector_hbm_bytes;
        }
        auto qconfig=qsa_config(table.geometry(),options.memory.context,options.memory.page_size);
        qconfig.kv=options.memory.kv_mode;
        qconfig.position_base=options.position_base;qconfig.rope=rope;qconfig.validate();
        if(options.mtp_runtime.empty() && (!options.mtp_draft_vocab.empty() || options.mtp.q4_projections || options.mtp.q4_head)) throw std::invalid_argument("MTP draft vocabulary/Q4 options require an MTP runtime");
        if(options.hpu_load_budget!=UINT32_MAX) {
#if STRATA_HPU_CPU_EXPERTS
            CpuExperts::validate(table,options.memory.cpu_cache,options.cpu_workers,options.cpu_tasks);
#else
            throw std::invalid_argument("this build has STRATA_HPU_CPU_EXPERTS=OFF");
#endif
        }
        if(!options.mtp_runtime.empty()) {
            if(!std::isfinite(options.mtp_min_p) || options.mtp_min_p<0 || options.mtp_min_p>1) throw std::invalid_argument("invalid MTP extension probability");
            const auto& sampling=options.mtp_sampling;
            if(!std::isfinite(sampling.temperature_scale) || sampling.temperature_scale<=0 || !std::isfinite(sampling.min_temperature) || sampling.min_temperature<0)
                throw std::invalid_argument("invalid MTP draft sampling configuration");
            options.mtp.context=options.memory.context;options.mtp.position_base=options.position_base;options.mtp.rope=rope;options.mtp.positions=&positions;options.mtp.staging=options.memory.staging;options.mtp.kv=options.memory.kv_mode;
            auto config=qsa_config(table.geometry(),options.mtp.context);config.top_k=config.capacity;config.position_base=options.position_base;config.rope=rope;config.validate();
            if(!options.mtp.draft_vocab.empty()) throw std::invalid_argument("Model requires --mtp-draft-vocab file identity for a draft subset");
            auto vocab_path=options.mtp_draft_vocab;
            if(vocab_path.empty()) {const auto path=std::filesystem::path(options.mtp_runtime)/"draft_vocab.bin";if(std::filesystem::exists(path)) vocab_path=path.string();}
            if(!vocab_path.empty()) {mtp_vocabulary.emplace(vocab_path,uint32_t(table.geometry().n_vocab));options.mtp.draft_vocab=mtp_vocabulary->ids();}
            mtp_source.emplace(options.mtp_runtime);extend_mtp_plan(memory_plan,*mtp_source,options.memory,options.mtp.expert_cache,options.mtp.window,uint32_t(options.mtp.draft_vocab.size()),table.info("output.weight").ggml_type,options.mtp.q4_projections,options.mtp.q4_head);
            const auto n=uint64_t(table.geometry().n_embd)*table.geometry().hc;
            last_mtp_logits.resize(table.geometry().n_vocab);pending_mtp_logits.resize(last_mtp_logits.size());
            last_mtp_residual.resize(n);pending_mtp_residual.resize(n);pending_embedding.resize(table.geometry().n_embd);
        }
        const auto& geometry=table.geometry();
        const auto ladder_elements=options.capture_layer_ladder?checked_product(checked_product(uint64_t(geometry.n_layers)+1,uint64_t(geometry.hc)),uint64_t(geometry.n_embd)):0;
        const auto ladder_bytes=checked_product(ladder_elements,uint64_t(sizeof(float)));
        const auto half=options.capture_halves?plan_half_producer(geometry):HalfProducerMemory{};
        uint64_t window_pinned=options.memory.pinned_limit;
        if(options.capture_halves) {
            if(half.pinned_bytes>window_pinned || memory_plan.verify_head_pinned_bytes>window_pinned-half.pinned_bytes)
                throw std::invalid_argument("half capture and verify head exceed pinned memory policy");
            window_pinned-=half.pinned_bytes+memory_plan.verify_head_pinned_bytes;
            if(memory_plan.mtp_window_pinned_bytes>window_pinned) throw std::invalid_argument("half capture and MTP window exceed pinned memory policy");
            if(half.hbm_bytes>options.memory.hbm_limit-memory_plan.hbm_required) throw std::invalid_argument("half capture exceeds HBM memory policy");
            memory_plan.half_capture_hbm_bytes=half.hbm_bytes;memory_plan.half_capture_pinned_bytes=half.pinned_bytes;
            memory_plan.hbm_required+=half.hbm_bytes;memory_plan.runtime_reserve+=half.hbm_bytes;
        }
        if(half.host_bytes>UINT64_MAX-ladder_bytes) throw std::overflow_error("model diagnostic memory overflow");
        const auto fixed_diagnostics=ladder_bytes+half.host_bytes;
        const auto routing_cost=[&](uint32_t rows)->uint64_t {
            return options.capture_routing?RoutingCapture::host_bytes(uint32_t(geometry.n_layers),std::max(7u,rows),uint32_t(geometry.n_expert),uint32_t(geometry.n_expert_used)):0;
        };
        if(options.prefill_auto) {
            const auto host=options.memory.host_limit-memory_plan.host_required+memory_plan.prefill_host_bytes+memory_plan.mtp_window_host_bytes;
            const auto hbm=options.memory.hbm_limit-memory_plan.hbm_required+memory_plan.prefill_hbm_bytes+memory_plan.mtp_window_hbm_bytes;
            // Price diagnostics per candidate before choosing the maximum;
            // charge them exactly once below, using these same calculators.
            const auto diagnostics=[&](uint32_t rows) {
                const auto routing=routing_cost(rows);
                if(routing>UINT64_MAX-fixed_diagnostics) throw std::overflow_error("model diagnostic memory overflow");
                return fixed_diagnostics+routing;
            };
            const auto admitted=select_model_prefill_window(table.geometry(),options.memory.context,host,hbm,window_pinned,bool(mtp_source),options.mtp.window,diagnostics,memory_plan.window_rows);
            const auto& window=admitted.target;const auto& mtp_window=admitted.mtp;
            memory_plan.host_required=memory_plan.host_required-memory_plan.prefill_host_bytes-memory_plan.mtp_window_host_bytes+window.host_bytes+mtp_window.host_bytes;
            memory_plan.hbm_required=memory_plan.hbm_required-memory_plan.prefill_hbm_bytes-memory_plan.mtp_window_hbm_bytes+window.hbm_bytes+mtp_window.hbm_bytes;
            memory_plan.runtime_reserve=memory_plan.runtime_reserve-memory_plan.prefill_hbm_bytes-memory_plan.mtp_window_hbm_bytes+window.hbm_bytes+mtp_window.hbm_bytes;
            memory_plan.mtp_window_rows=mtp_window.rows;memory_plan.mtp_window_host_bytes=mtp_window.host_bytes;memory_plan.mtp_window_hbm_bytes=mtp_window.hbm_bytes;memory_plan.mtp_window_pinned_bytes=mtp_window.pinned_bytes;
            memory_plan.prefill_rows=window.rows;memory_plan.window_rows=window.rows;
            memory_plan.prefill_host_bytes=window.host_bytes;memory_plan.prefill_hbm_bytes=window.hbm_bytes;
        }
        weight_identity=file_identity();if(weight_identity.size()>65536) throw std::invalid_argument("combined model/MTP identity too large");last_logits.resize(table.geometry().n_vocab);
        last_residual.resize(uint64_t(table.geometry().n_embd)*table.geometry().hc);pending_residual.resize(last_residual.size());
        mixed_output=HeadMixed(table.geometry().n_embd);
        if(options.capture_halves) {
            if(half.host_bytes>options.memory.host_limit-memory_plan.host_required) throw std::invalid_argument("half capture exceeds host memory policy");
            half_capture.configure(geometry,half.host_bytes);memory_plan.half_capture_host_bytes=half.host_bytes;
            memory_plan.dump_host_bytes+=half.host_bytes;memory_plan.host_required+=half.host_bytes;memory_plan.control_bytes+=half.host_bytes;
        }
        if(options.capture_layer_ladder) {
            // One full (n_layers+1)*hc*n_embd staging buffer, charged before any device exists.
            const auto elements=ladder_elements,bytes=ladder_bytes;
            if(bytes>options.memory.host_limit-memory_plan.host_required) throw std::invalid_argument("the --dump-layers ladder exceeds host memory policy");
            ladder.assign(size_t(elements),0.0f);
            memory_plan.dump_host_bytes+=bytes;memory_plan.host_required+=bytes;memory_plan.control_bytes+=bytes;
        }
        if(options.capture_routing) {
            // Also price scalar candidate routing assembly (<=7 rows). Its
            // IDs slab + one scalar export fit the same single-export envelope.
            const auto& g=table.geometry();const auto rows=std::max(7u,memory_plan.window_rows);
            const auto bytes=routing_cost(rows);
            if(bytes>options.memory.host_limit-memory_plan.host_required) throw std::invalid_argument("the --dump-routing capture exceeds host memory policy");
            routing_capture.configure(uint32_t(g.n_layers),rows,uint32_t(g.n_expert),uint32_t(g.n_expert_used));
            memory_plan.routing_capture_host_bytes=bytes;
            memory_plan.dump_host_bytes+=bytes;memory_plan.host_required+=bytes;memory_plan.control_bytes+=bytes;
        }
        // A zero policy explicitly disables snapshots.
        snapshot_budget->configure(options.memory.snapshot_budget?options.memory.snapshot_budget:1);
        for(const auto* key:{"tokenizer.ggml.eos_token_id","tokenizer.ggml.eot_token_id"}) if(const auto ids=table.integers(key)) {
            if(ids->size()!=1 || ids->front()>=uint64_t(table.geometry().n_vocab)) throw std::invalid_argument("invalid generation stop token metadata");
            const auto id=int32_t(ids->front());if(std::find(stops.begin(),stops.end(),id)==stops.end()) stops.push_back(id);
        }
        // --eos-ids wins over the file's metadata, because upstream's list is the operator's (generate.cpp:2187-2191).
        // An id this vocabulary cannot emit is refused here rather than kept as a token that could never match:
        // Generation's stop set doubles as the verify contract, which validates every id it is given
        // (hpu/src/generation.cpp:76).
        if(!options.eos_ids.empty()) {
            stops.clear();
            for(const auto id:options.eos_ids) {
                if(id<0 || uint64_t(id)>=uint64_t(table.geometry().n_vocab))
                    throw std::invalid_argument("--eos-ids names a token outside the model vocabulary");
                if(std::find(stops.begin(),stops.end(),id)==stops.end()) stops.push_back(id);
            }
        }
        // A GGUF without metadata is not a refusal: upstream's own pair still applies (:612), and a pair above this
        // vocabulary simply stops nothing, exactly as an unmatchable id does upstream.
        else if(stops.empty()) stops=implicit_stop_tokens(table.geometry().n_vocab);
    }
    // --adapt-every's round, executed under the model lock (the public entry re-takes it).
    uint64_t adapt_locked() {
        std::vector<ExpertSwap> applied;
        if(!cache->adapt_experts(applied)) return 0;
        for(const auto& swap:applied) {
            if(swap.in<0 || swap.in>=int32_t(table.geometry().n_expert) || swap.layer<0 || swap.layer>=int32_t(table.geometry().n_layers))
                throw std::logic_error("expert adaptation proposed an out-of-range residency change");
            auto lease=cache->acquire(uint32_t(swap.layer),swap.in);
            (void)lease;
        }
        return applied.size();
    }
    void after_round() { // one --adapt-every tick per decode round/window (generate.cpp:10031)
        if(const auto* profiler=cache->expert_profiler();profiler && profiler->options().tier_enabled()) (void)adapt_locked();
    }
    std::string file_identity() const {return table.file_identity()+(options.ple_gguf.empty()?std::string{}:ple_source_identity(options.ple_gguf))+(embedding_input?embedding_input->file_identity():std::string{})+(expert_pack?expert_pack->file_identity():std::string{})+(mtp_source?mtp_source->file_identity():std::string{})+(mtp_vocabulary?mtp_vocabulary->file_identity():std::string{})+(control_vectors?control_vectors->file_identity():std::string{});}
    // The one row gather every seam uses: the --embd-gguf route when the flag is present, else the
    // canonical token_embd.weight table lookup this backend always had (byte-identical default path).
    std::vector<float> table_embedding(const int32_t token) {
        std::vector<float> row(table.geometry().n_embd);
        if(embedding_ram.empty()) embedding.convert_rows_f32(uint32_t(token),row);
        else {
            if(token<0 || uint64_t(token)>=uint64_t(table.geometry().n_vocab)) throw std::out_of_range("embedding token row");
            std::copy_n(embedding_ram.data()+uint64_t(token)*row.size(),row.size(),row.data());
        }
        return row;
    }
    std::vector<float> embedding_row(const int32_t token,const uint64_t position) {
        // --vision: upstream's sp.embd_rows replaces the embedding table for the prompt's image cells
        // (:10827). Only cells inside the current prompt carry a row; decoded tokens are text.
        if(position<vision_rows.size() && vision_rows[std::size_t(position)]>=0) {
            if(!vision_embeddings) throw std::logic_error("vision row installed without an embeddings file");
            const auto row=vision_embeddings->row(uint64_t(vision_rows[std::size_t(position)]));
            if(row.size()!=uint64_t(table.geometry().n_embd)) throw std::runtime_error("vision embedding row width changed");
            return {row.begin(),row.end()};
        }
        if(embedding_rows.active()) {
            auto row=embedding_rows.row(token,position);
            if(row.size()!=uint64_t(table.geometry().n_embd)) throw std::runtime_error("embedding route row width changed");
            return row;
        }
        return table_embedding(token);
    }
    // A park's own bytes: every layer's parked image at this extent, the MTP pool, and the session-side parts
    // the Snapshot carries (ids, the last logits row, the committed residual, the MTP draft row) plus the same
    // 128 KiB slack the arena-wide limit keeps. Summed with overflow refusal, like upstream's add().
    uint64_t parked_bytes(uint64_t cells) const {
        if(cells>options.memory.context) throw std::out_of_range("parked image exceeds context");
        uint64_t total=0;
        auto add=[&](uint64_t n){if(n>UINT64_MAX-total) throw std::overflow_error("parked image byte count overflow");total+=n;};
        for(const auto& layer:layers) add(layer->parked_bytes(cells));
        if(mtp) add(mtp->parked_bytes(cells?cells-1:0));
        add(checked_product(cells,4));
        if(cells) {add(checked_product(table.geometry().n_vocab,4));add(checked_product(last_residual.size(),4));add(checked_product(mixed_output.width(),4));}
        if(cells>1) add(checked_product(last_mtp_logits.size()+last_mtp_residual.size(),4));
        add(131072);
        return total;
    }
    uint64_t snapshot_overlap_bytes(uint64_t old_cells) const {
        // snapshot_resize replaces one vector at a time, with exact capacity.
        // GDN replaces a complete layer checkpoint at once. At any point the
        // partially updated image fits the FINAL resident envelope, plus at
        // most one old layer (or one old session-side component) still alive.
        uint64_t largest=std::max(checked_product(old_cells,4),checked_product(table.geometry().n_vocab,4));
        for(const auto& layer:layers) largest=std::max(largest,layer->parked_bytes(old_cells));
        if(mtp) largest=std::max(largest,mtp->parked_bytes(old_cells?old_cells-1:0));
        largest=std::max(largest,checked_product(last_residual.size(),4));
        largest=std::max(largest,checked_product(mixed_output.width(),4));
        largest=std::max(largest,checked_product(last_mtp_logits.size()+last_mtp_residual.size(),4));
        return largest;
    }
    uint64_t snapshot_bytes() const {return memory_plan.host_state_bytes+checked_product(options.memory.context,4)+checked_product(table.geometry().n_vocab,4)+checked_product(last_residual.size(),4)+checked_product(mixed_output.width(),4)+checked_product(last_mtp_logits.size()+last_mtp_residual.size(),4)+131072;}
    void commit_output(std::span<const float> logits) {
        std::copy(logits.begin(),logits.end(),last_logits.begin());std::copy(pending_residual.begin(),pending_residual.end(),last_residual.begin());
        mixed_output.commit();
        if(mtp && flow->tokens().size()>1) {
            std::copy(pending_mtp_logits.begin(),pending_mtp_logits.end(),last_mtp_logits.begin());
            std::copy(pending_mtp_residual.begin(),pending_mtp_residual.end(),last_mtp_residual.begin());
        }
    }
    std::vector<uint64_t> state_configuration() const {
        const auto& g=table.geometry();std::vector<uint64_t> config{uint64_t(g.n_layers),uint64_t(g.n_embd),uint64_t(g.n_vocab),uint64_t(g.hc),uint64_t(g.n_ff),uint64_t(g.n_expert),options.memory.context,options.memory.page_size,options.position_base,options.hpu_load_budget,uint64_t(rope.type),uint64_t(options.vision)};
        for(const double value:{rope.freq_base,rope.factor,rope.freq_scale_in,rope.orig_ctx,rope.ext_factor,rope.attn_factor,rope.beta_fast,rope.beta_slow}) config.push_back(std::bit_cast<uint64_t>(value));
        config.push_back(bool(mtp_source));if(mtp_source) {config.push_back(options.mtp.window);config.push_back(options.mtp.stream_norm);config.push_back(options.mtp.draft_vocab.size());config.push_back(options.mtp.q4_projections);config.push_back(options.mtp.q4_head);}
        config.push_back(control_vectors?control_vectors->signature():0);
        config.push_back(packed_supported(table.info("output.weight").ggml_type)?table.info("output.weight").ggml_type:30);
        config.push_back(29); // RoutedQ8 host staging uses Q8_1 rather than S2 metadata
        config.push_back(memory_plan.prefill_rows); // reject snapshots from different window arithmetic
        config.push_back(memory_plan.verify_rows);
        // --embd-gguf changes every embedding row, so a state saved under one row file is never reused
        // under another: encode the consumption mode and the claim (rows) the run was built with.
        config.push_back(embedding_input?uint64_t(embedding_input->mode())+1:0);
        config.push_back(embedding_input?embedding_input->rows():0);
        // --no-ple changes every layer-1 input; file_identity() cannot see it because the ablation leaves
        // ple_gguf empty just like the in-model table does.
        config.push_back(uint64_t(!ple_mode.enabled));
        // --kv changes the parked image's part sizes (and, for int8/q4_0/k8v4, its arithmetic): only an image of the
        // same format may be reused, so the mode is part of the configuration. q4_0 and k8v4 payloads are state-v38
        // with the mode bit already distinguishing them: a v38 writer could never have emitted mode 3.
        config.push_back(uint64_t(options.memory.kv_mode));
        // Opt-in tree reductions must never restore a scalar-reduction disk image (or vice versa).
        // No extra word on the default path, preserving its existing configuration identity.
        if(simd_norm_enabled()) config.push_back(0x53494d444e4f524dull); // SIMDNorm v1
        if(fused_fp8_enabled()) config.push_back(0x465038454e430001ull); // fused FP8 encode v1
        if(direct_experts_enabled()) config.push_back(0x4449524558500001ull); // direct per-expert MME v1
        if(tpc_experts_enabled()) config.push_back(0x5450434558500001ull); // TPC FP8 GEMV experts v1
        if(tpc_colpair_enabled()) config.push_back(0x434f4c5041495201ull); // tiled column-pair FP8 GEMV v1
        if(tpc_hilo_enabled()) config.push_back(0x54504348494c4f01ull); // TPC scaled FP8 hi/lo v1
        if(simd_gdn_enabled()) config.push_back(0x53494d4447444e01ull); // SIMD GDN prepare/norm reductions v1
        if(resident_iq4_enabled()) config.push_back(0x5245534951340001ull); // resident IQ4 x Q8 experts, device routing v1
        if(qsa_device_enabled()) config.push_back(0x5153414445560001ull); // device QSA norm/RoPE/attention v1
        if(merged_gemm_enabled()) config.push_back(0x4d52474753454d01ull); // merged projection GEMMs v1
        if(fp8_dense_enabled()) config.push_back(0x4650384453450001ull); // FP8 dense merged projections v1
        if(gdn_wide_rows_enabled()) config.push_back(0x47444e5749440001ull); // fused GDN windows up to8 rows
        if(prefill_qsa32_enabled()) config.push_back(0x5052515341333201ull); // canonical QSA32 inside larger prefill
        if(prefill_final_enabled()) config.push_back(0x505246494e414c01ull); // final-state prefill v1
        if(prefill_last_head_enabled()) config.push_back(0x50524c4845414401ull); // single-row prefill head v1
        if(qsa_chunk_rows()) {config.push_back(0x51534143484e4b01ull);config.push_back(qsa_chunk_rows());}
        if(tpc_access_enabled()) config.push_back(0x5450434143430001ull); // affine norm/HC access patterns v1
        if(fp8_static_enabled()) config.push_back(0x4650385354410001ull); // native fixed-scale FP8 dense v1
        if(q8_packed_enabled()) config.push_back(0x51385041434b0001ull); // packed Q8 expert activations v1
        if(moe_act_fused_enabled() && q8_packed_enabled()) config.push_back(0x4d4f454143540001ull); // fused MoE activation v1
        if(hc_fused_enabled()) config.push_back(0x4843465553450001ull); // HC reads / GDN norm with fused BF16 casts v1
        if(moe_tail_fused_enabled()) config.push_back(0x4d4f455441494c01ull); // MoE route sum inside the HC write v1
        // Bit-identical device variants still identify the graph that produced an image, so an A/B cannot
        // restore an image built by the other kernel.
        if(gdn_step_window_enabled()) config.push_back(0x47444e57494e4401ull); // whole-window GDN steps v1
        if(moe_group_enabled() && q8_packed_enabled()) config.push_back(0x4d4f454752500001ull); // expert-grouped MoE order v1
        if(moe_pair_enabled() && q8_packed_enabled()) config.push_back(0x4d4f455041495201ull); // paired grouped IQ4 prefill v1
        if(moe_group_blocks_enabled() && moe_group_enabled() && q8_packed_enabled()) config.push_back(0x4d4f4547424c4b01ull);
        if(gemv_pf_enabled() && q8_packed_enabled()) config.push_back(0x47454d5650460001ull); // packed GEMV paired prefetch v1
        return config;
    }
    // STRATA_HPU_CHAIN: the token's device residual and stream. Last member, so it (and the expert leases it
    // may hold) goes before the layers, the cache and the device.
    std::unique_ptr<DecodeChain> decode_chain;
    // STRATA_HPU_DFLASH=<directory> probe: the drafter (borrowing the BF16 head, so declared after it) drafts after
    // every decoded token from the chain's captured features; later tokens score the drafts (greedy acceptance).
    std::unique_ptr<DflashDrafter> drafter;std::vector<float> mask_embedding;
    struct DraftRecord {uint64_t start=0;std::vector<int32_t> tokens;uint32_t matched=0;};
    std::deque<DraftRecord> drafts;std::array<uint64_t,16> accepted{};uint64_t drafted=0;
    // STRATA_HPU_DFLASH_VERIFY: the speculative window (Model::dflash_window) instead of the probe's scoring.
    bool dflash_verify=false;std::map<uint32_t,std::unique_ptr<HostRecipe>> verify_head;std::unique_ptr<VerifyChain> verify_chain;
    uint64_t dflash_rounds=0,dflash_committed=0,dflash_proposed=0;
    std::array<double,6> dflash_ms{}; // draft, inputs, window + head, select, commit, of which layer commits (host wall time)
    bool dflash_host_profile=performance_env("STRATA_HPU_DFLASH_HOST_PROFILE");
    std::array<double,5> dflash_host_ms{}; // embedding, PLE, input scatter/history, GDN commit, QSA commit
    // STRATA_HPU_DFLASH_SYNC_TIMING=1: the window waits after every launch; GDN layers, QSA layers, capture, head.
    bool dflash_sync_timing=false,dflash_mark=false;std::array<double,4> dflash_kind_ms{};
    std::unique_ptr<DflashLogitSink> dflash_logits; // STRATA_HPU_DFLASH_LOGITS: committed rows' full-vocabulary logits.
    // STRATA_HPU_PREFILL_PROFILE=1: the resident prefill window's wall time, every 32 windows (every window with PHASES).
    // STRATA_HPU_PREFILL_PHASES=1: additionally waits at three boundaries of every window and reports where the
    // device time went (layers; head+readback; last row+commit). A diagnostic: the waits keep the window serial.
    bool prefill_profile=performance_env("STRATA_HPU_PREFILL_PROFILE");
    bool prefill_phases=performance_env("STRATA_HPU_PREFILL_PHASES");
    // STRATA_HPU_PREFILL_LAYERS=1: the first two windows wait after every layer and print its class and ms.
    // A value of `<first>:<count>` profiles those windows instead (a late window shows the long-context law).
    bool prefill_layers=false;
    uint64_t prefill_layers_first=0,prefill_layers_count=2;
    void parse_layer_windows(const char* value) {
        if(!value || !*value || (value[0]=='0' && !value[1])) return;
        prefill_layers=true;prefill_layers_first=0;prefill_layers_count=2;
        if(!std::strchr(value,':')) return;
        char* end=nullptr;const auto first=std::strtoull(value,&end,10);
        if(!end || *end!=':') return;
        prefill_layers_first=first;prefill_layers_count=std::strtoull(end+1,nullptr,10);
        if(!prefill_layers_count) prefill_layers_count=2;
    }
    std::array<double,6> prefill_phase_ms{}; // host rows, layers, head+readback, tail, commit, (row build in host rows)
    // The next resident window's host rows (embedding and layer-1 PLE rows), built on a host thread while the device
    // runs the current window. prefill_window() names the next window; a window whose tokens or positions differ
    // from the prefetched ones builds its own rows as before (the same rows either way).
    struct PrefetchedRows {uint64_t position=0;std::vector<int32_t> tokens,history;std::vector<float> embedding,ple;};
    std::vector<int32_t> next_window,hinted_next;
    std::future<PrefetchedRows> prefetched_rows;
    // with_ple=false: the embedding only (the caller gathers the PLE rows itself).
    PrefetchedRows build_rows(std::span<const int32_t> tokens,uint64_t first,const std::vector<int32_t>& previous,bool with_ple=true) {
        const uint64_t h=table.geometry().n_embd;
        PrefetchedRows out;out.position=first;out.tokens.assign(tokens.begin(),tokens.end());out.history=previous;out.embedding.resize(tokens.size()*h);
        const auto embed=[&](size_t a,size_t b) {
            for(size_t r=a;r<b;++r) {
                const auto row=embedding_row(tokens[r],first+r);
                if(row.size()!=h) throw std::runtime_error("embedding row width mismatch");
                std::copy(row.begin(),row.end(),out.embedding.begin()+r*h);
            }
        };
        // Plain table rows (no image cells, no --embd-gguf route) are independent const decodes: sixteen threads.
        if(!embedding_rows.active() && (vision_rows.empty() || first>=vision_rows.size()) && tokens.size()>=256) {
            std::vector<std::future<void>> parts;
            for(size_t part=0;part<16;++part) parts.push_back(std::async(std::launch::async,embed,tokens.size()*part/16,tokens.size()*(part+1)/16));
            for(auto& part:parts) part.get();
        } else embed(0,tokens.size());
        // The window's n-gram rows in one reader request, like native prefill's gather_batch: a per-row
        // gather is one SSD request per token, serialized in front of layer 1 (0.66 ms a row measured).
        if(ple && with_ple) {out.ple.resize(tokens.size()*h);ple->gather_rows(tokens,previous,out.ple.data(),h);}
        return out;
    }
    uint64_t prefill_windows=0,prefill_committed=0;double prefill_ms=0;
    // Adaptive use (off by default: measured, every prompt was fastest with windows throughout): a window costs about
    // as much as `dflash_break_even` decoded tokens, so while the recent mean of
    // committed tokens per window (dflash_recent) stays below it, the next dflash_backoff tokens are decoded instead
    // (STRATA_HPU_DFLASH_MIN overrides the threshold; 0 always drafts).
    double dflash_break_even=0.0,dflash_recent=4.0;uint32_t dflash_skip=0,dflash_backoff=16;uint64_t dflash_skipped=0;
    // The tokens of the drafter's context cells: a restored or rewound sequence of the same length must not
    // pass for the one the drafter saw.
    std::vector<int32_t> drafter_tokens;
    // The next window's draft, begun at the last commit for the pending token the target's argmax predicted.
    struct PreparedDraft {uint64_t position=0;int32_t pending=-1;bool active=false;} dflash_prepared;
    void settle_draft() {if(dflash_prepared.active) {dflash_prepared.active=false;(void)drafter->end_draft();}}
    void note_drafter() {
        if(!drafter) return;
        const auto tokens=flow->tokens();
        if(drafter->cells()==tokens.size()) drafter_tokens.assign(tokens.begin(),tokens.end());else drafter_tokens.clear();
    }
    void dflash_observe(uint64_t position,int32_t token,std::span<const float> logits) {
        // Score the open drafts: tokens[0] is the anchor (the greedy next token), then the drafted ones.
        for(auto it=drafts.begin();it!=drafts.end();) {
            if(position<it->start || position>=it->start+it->tokens.size()) {++it;continue;}
            const auto at=position-it->start;
            bool finished=true;
            if(at==it->matched && it->tokens[at]==token) {++it->matched;finished=it->matched==it->tokens.size();}
            if(!finished) {++it;continue;}
            // matched==0: the anchor was not the next token (not a greedy step), so the draft says nothing.
            if(it->matched>0) {
                // accepted[k]: drafts whose first k drafted tokens matched (k = 0 .. block-1).
                ++accepted[std::min<size_t>(it->matched-1,accepted.size()-1)];++drafted;
                if(drafted%32==0) {
                    const size_t buckets=std::min<size_t>(drafter->config().block,accepted.size());
                    double mean=0;std::string hist;
                    for(size_t k=0;k<buckets;++k) {mean+=double(k)*double(accepted[k]);hist+=(k?",":"")+std::to_string(accepted[k]);}
                    mean/=double(drafted);
                    std::fprintf(stderr,"strata-hpu: dflash drafts=%llu accepted_mean=%.3f tokens_per_step=%.3f hist=%s\n",
                        (unsigned long long)drafted,mean,1+mean,hist.c_str());
                }
            }
            it=drafts.erase(it);
        }
        const auto next=position+1;
        if(next!=drafter->cells()) return;
        int32_t anchor=0;for(size_t i=1;i<logits.size();++i) if(logits[i]>logits[size_t(anchor)]) anchor=int32_t(i);
        auto row=embedding_row(anchor,next);
        auto tokens=drafter->draft(row,mask_embedding,anchor,uint32_t(next));
        DraftRecord record;record.start=next;record.tokens.push_back(anchor);record.tokens.insert(record.tokens.end(),tokens.begin(),tokens.end());
        drafts.push_back(std::move(record));
    }
    // The window head shared by the DFlash round and the resident prefill window: HC read, the head's logits, and
    // the [rows, hidden] mixed input of every row (a prefill window stages the committed row's value).
    // The resident prefill window's input: the embedding rows [rows, 1, hidden] (uploaded once) broadcast to every
    // HC stream [rows, streams, hidden] on the device, instead of four host copies and a 4x larger upload.
    std::map<uint32_t,std::unique_ptr<HostRecipe>> window_replicate;
    HostRecipe& window_replicate_for(uint32_t rows,const DeviceBuffer& embedding,const DeviceBuffer& residual) {
        auto& recipe=window_replicate[rows];
        if(!recipe) {
            const auto& g=table.geometry();const uint64_t h=uint64_t(g.n_embd),r=uint64_t(g.hc);
            GraphBuilder graph(*device);
            const auto x=graph.tensor({"embedding",DType::F32,{rows,1,h}}),y=graph.tensor({"residual",DType::F32,{rows,r,h}});
            {const TensorId in[]={x},out[]={y};graph.op("broadcast",in,out);}
            const Binding io[]={{"embedding",&embedding},{"residual",&residual}};
            recipe=std::make_unique<HostRecipe>(*device,graph.compile("strata_prefill_replicate_"+std::to_string(rows),"verify_chain"),io,std::vector<std::string>{});
        }
        recipe->bind_external({"embedding",&embedding});recipe->bind_external({"residual",&residual});
        return *recipe;
    }
    HostRecipe& verify_head_for(uint32_t rows,bool last_only=false) {
        // Key zero is a separate single-row prefill head. Its bindings select
        // the last row, so all admitted prefill widths share this recipe.
        auto& head=verify_head[last_only?0:rows];
        if(head) return *head;
        if(last_only) rows=1;
        const auto& g=table.geometry();const uint64_t h=uint64_t(g.n_embd),vocabulary=uint64_t(g.n_vocab);
        GraphBuilder graph(*device);const HcConfig hc{uint32_t(h),uint32_t(g.hc),uint32_t(g.hc_lr),float(g.rms_eps)};
        graph.set_scope("final.");define_hc_read_window(graph,hc,rows,true);graph.set_scope({});
        const auto x=graph.tensor({"head_x",DType::BF16,{rows,h},false});graph.cast(graph.find("final.mixed"),x);
        graph.alias("x",x);graph.set_scope("head.");define_model_head(graph,uint32_t(h),uint32_t(vocabulary),rows);graph.set_scope({});
        // The logits stay in VerifyChain::logits; only each row's argmax (top-1, as F32 ids) comes back.
        if(!last_only) {
        const auto best_value=graph.tensor({"head.best_value",DType::F32,{rows,1},false}),best_id=graph.tensor({"head.best_id",DType::I32,{rows,1},false});
        {ns_TopkNode::Params q{};q.bsw=1;q.axis=0;q.bottomK=0;const TensorId in[]={graph.find("head.logits")},outs[]={best_value,best_id};graph.op("topk",in,outs,&q,sizeof(q));}
        const auto best=graph.tensor({"head.best",DType::F32,{rows,1}});graph.cast(best_id,best);
        }
        auto& v=*verify_chain;
        std::vector<Binding> bound;
        for(const char* name:{"w_down","w_up","norm"}) {auto binding=head_weights->binding(name);binding.name=std::string("final.")+name;bound.push_back(binding);}
        {auto binding=head_weights->binding("weight");binding.name="head.weight";bound.push_back(binding);}
        bound.push_back({"final.residual",&v.in()});bound.push_back({"head.logits",&v.logits});bound.push_back({"final.mixed",&v.mixed});
        head=std::make_unique<HostRecipe>(*device,graph.compile(last_only?"strata_prefill_last_head":"strata_verify_head_"+std::to_string(rows),"verify_chain"),bound,
            last_only?std::vector<std::string>{}:std::vector<std::string>{"head.best"});
        const std::array<float,1> scale{1/float(g.hc)};head->set("final.inverse_streams",scale);
        return *head;
    }
};
ModelPlan Model::preflight(const std::string& path,ModelOptions options) {const Impl p(path,std::move(options));return p.memory_plan;}
Model::Model(const std::string& path,ModelOptions options) {
    // Keep preflight available for the experimental store, but never price U4
    // then construct IQ4 weights while decode/overflow support is incomplete.
    if(resident_u4_enabled())
        throw std::invalid_argument("STRATA_HPU_RESIDENT_U4 is an isolated prototype: serving needs U4 decode and overflow replay");
    // Complete schema and payload plan are checked before device acquisition.
    auto p=std::make_unique<Impl>(path,std::move(options));const auto& g=p->table.geometry();const auto& m=p->options.memory;
    auto qconfig=qsa_config(g,m.context,m.page_size);qconfig.kv=m.kv_mode;qconfig.position_base=p->options.position_base;qconfig.rope=p->rope;qconfig.validate();
    // --no-ple keeps the model free of any PLE table: there is no gather, no reader reservation and the
    // layer-1 PLE weights were not priced or loaded (model_plan.cpp).
    if(p->ple_mode.enabled)
        p->ple=std::make_unique<PleInput>(p->table,p->options.ple_io,p->ple_mode.external,p->memory_plan.ple_ram_host_bytes,p->memory_plan.ple_reader_host_bytes,p->options.ple_delay_us);
    p->embedding=p->table.matrix("token_embd.weight");
    if(p->memory_plan.embedding_ram_host_bytes) {
        const auto elements=checked_product(uint64_t(g.n_vocab),uint64_t(g.n_embd));
        if(p->memory_plan.embedding_ram_host_bytes!=checked_product(elements,4)+65536) throw std::logic_error("embedding RAM budget mismatch");
        p->embedding_ram.resize(elements);p->embedding.convert_rows_f32(0,p->embedding_ram);
        std::fprintf(stderr,"strata-hpu: embedding RAM rows=%llu bytes=%llu\n",(unsigned long long)g.n_vocab,(unsigned long long)(elements*4));
    }
    if(p->embedding_input) {
        // The route's table side is the same WeightSlice gather the closure used before this option existed,
        // so a file that only claims positions keeps the table for everything after them.
        auto* impl=p.get();
        p->embedding_rows=EmbeddingRows::for_input(*p->embedding_input,uint64_t(g.n_vocab),uint64_t(g.n_embd),
            [impl](int32_t token){return impl->table_embedding(token);});
    }
    if(p->file_identity()!=p->weight_identity) throw std::runtime_error("model/PLE files changed during load");
    if(p->options.hpu_load_budget!=UINT32_MAX) {
#if STRATA_HPU_CPU_EXPERTS
        // Validate every CPU layer format before acquiring the HPU.
        p->cpu=std::make_unique<CpuExperts>(p->table,m.cpu_cache,p->options.cpu_workers,p->expert_pack,p->options.cpu_affinity,p->options.cpu_tasks);
#else
        throw std::invalid_argument("this build has STRATA_HPU_CPU_EXPERTS=OFF");
#endif
    }
    p->device=std::make_unique<Device>(p->options.module_id);
    const auto available=p->device->info().free_bytes;
    if(m.driver_margin>available || m.hbm_limit>available-m.driver_margin) throw std::runtime_error("requested HBM cap plus driver margin exceeds available memory");
    p->device->allocation_limits(m.hbm_limit,m.pinned_limit);
    p->cache=std::make_unique<ExpertCache>(*p->device,p->table,m.expert_cache,m.staging,p->expert_pack);
    // Before the residency fill below, which already loads experts in the chosen storage.
    if(p->options.fp8_experts) p->cache->set_fp8(true);
    // --expert-profile pre-loads the tier's ranking before any routing is seen (generate.cpp:2429,
    // 3657-3698); a malformed profile is refused here, before the run, as upstream refuses it.
    p->cache->set_expert_profile_options(p->options.expert_profile);
    // --resident-experts / --resident-cpu-experts / --expert-cache-per-layer: install the per-layer byte
    // partition and, when the RAM tier is on, the profile's admitted set AFTER the tier's ranking exists, so
    // the fill reads the same profile the profiler preloaded.
    p->cache->set_expert_residency(p->options.residency);
    p->head_weights=std::make_unique<DenseWeights>(*p->device,p->memory_plan.head_bytes,m.staging);
    const uint64_t h=g.n_embd,n=h*g.hc,lr=g.hc_lr;
    p->head_weights->matrix("w_down",p->table,"output_hc_down.weight",n,lr);p->head_weights->matrix("w_up",p->table,"output_hc_up.weight",lr,n);
    p->head_weights->constant("norm",p->table,"output_hc_norm.weight",n);
    // --fp8-experts also moves the head to a BF16 MME GEMM (ModelHead takes the BF16 path for type 30). The
    // MTP drafter shares the packed head binding, so it keeps the packed head.
    const bool bf16_head=m.bf16_head && !p->mtp_source;
    const auto head_type=bf16_head?uint32_t(30):p->table.info("output.weight").ggml_type;
    if(packed_supported(head_type)) p->head_weights->packed("weight",p->table.matrix("output.weight"));
    else p->head_weights->matrix("weight",p->table,"output.weight",h,g.n_vocab);
    const std::array<Binding,3> mix_weights{p->head_weights->binding("w_down"),p->head_weights->binding("w_up"),p->head_weights->binding("norm")};
    const HcConfig hc{uint32_t(h),uint32_t(g.hc),uint32_t(lr),float(g.rms_eps)};
    if(p->memory_plan.window_rows) p->hc_window=std::make_unique<HcWindow>(*p->device,hc,p->memory_plan.window_rows);
    if(p->memory_plan.window_rows) p->gdn_window=std::make_unique<GdnWindow>(*p->device,g);
    if(p->memory_plan.window_rows) p->qsa_window=std::make_unique<QsaWindow>(*p->device,qsa_config(g,m.context,m.page_size),uint32_t(g.n_embd));
    p->final_mix=std::make_unique<HostRecipe>(*p->device,build_hc_read(*p->device,hc,true),mix_weights,std::vector<std::string>{"mixed"});
    const std::array<float,1> scale{1/float(g.hc)};p->final_mix->set("inverse_streams",scale);
    p->head=std::make_unique<ModelHead>(*p->device,p->head_weights->binding("weight"),uint32_t(h),uint32_t(g.n_vocab),head_type);
    if(p->control_vectors) {
        GraphBuilder graph(*p->device);define_control_vector(graph,uint32_t(h),uint32_t(g.hc),p->control_vectors->mode());
        p->control_recipe=std::make_unique<HostRecipe>(*p->device,graph.compile("strata_control_vector"),std::span<const Binding>{},std::vector<std::string>{"out"});
    }
    if(p->mtp_source) p->mtp=std::make_unique<MtpBlock>(*p->device,*p->mtp_source,p->head_weights->binding("weight"),p->options.mtp,p->table.matrix("output.weight"));
    for(uint32_t layer=0;layer<uint64_t(g.n_layers);++layer) {
        LayerOptions o;o.context=m.context;o.page_size=m.page_size;o.kv=m.kv_mode;o.staging_budget=m.staging;o.dense_payload_budget=p->memory_plan.layer_dense_bytes[layer];
        o.position_base=p->options.position_base;o.rope=p->rope;o.positions=&p->positions;o.fused_decode=p->options.fp8_experts && !std::getenv("STRATA_HPU_UNFUSED"); // env: A/B against the per-stage path
#if STRATA_HPU_CPU_EXPERTS
        o.cpu=p->cpu.get();
#endif
        o.hpu_load_budget=p->options.hpu_load_budget;
        o.prefill_rows=p->memory_plan.window_rows;
        o.ple_enabled=p->ple_mode.enabled;
        o.hc_window=p->hc_window.get();
        o.gdn_window=p->gdn_window.get();
        o.qsa_window=p->qsa_window.get();
        if(p->options.capture_halves) {
            auto* impl=p.get();
            o.half_sink=[impl](uint32_t layer,HalfField field,std::span<const float> values){impl->half_capture.half(layer,field,values);};
            o.qsa_half_sink=[impl](uint32_t layer,const QsaHalfView& view){impl->half_capture.qsa(layer,view);};
        }
        if(p->options.capture_routing) {
            // Bounded layer-major rows of the initial physical pass. Accepted
            // prefix replay is deliberately excluded by RoutingCapture.
            auto* impl=p.get();
            o.routing_sink=[impl](uint32_t layer,const Routes& routes) {
                impl->routing_capture.capture(layer,routes);
            };
        }
        p->layers.push_back(std::make_unique<Layer>(*p->device,p->table,layer,*p->cache,o));
    }
    auto* state=p.get();SessionOperations op;
    op.embedding=[state,h](int32_t token) {
        // The position a positional row file claims comes from the window the owning call opened
        // (prefill_window / verify_window); the token path's fallback is the committed length, which IS
        // the position of the token about to be embedded.
        const auto position=state->embedding_window.next(token,uint64_t(state->flow->tokens().size()));
        auto row=state->embedding_row(token,position);
        if(row.size()!=uint64_t(h)) throw std::runtime_error("embedding row width mismatch");
        if(state->mtp) std::copy(row.begin(),row.end(),state->pending_embedding.begin());
        return row;
    };
    // An empty callback is the ablation: SessionFlow gathers nothing and layer 1 receives an empty span.
    if(state->ple) op.ple=[state](int32_t token,std::span<const int32_t> previous) {return state->ple->gather(token,previous);};
    op.checkpoint=[state](uint32_t layer) {
        if(layer==state->layers.size()) {
            auto point=state->mtp->checkpoint();return std::function<void()>([state,point=std::move(point)]{state->mtp->restore(point);});
        }
        auto point=state->layers.at(layer)->checkpoint();
        return std::function<void()>([state,layer,point=std::move(point)] {state->layers.at(layer)->restore(point);});
    };
    op.layer=[state](uint32_t layer,std::span<const float> residual,std::span<const float> ple) {
        if(layer<state->layers.size()) {
            // --dump-layers' in-model capture: slot 0 is the INPUT residual (this layer's input at layer 0),
            // slot layer+1 the value after this layer (upstream's `layer_stage`, src/core/session.cpp:660-672).
            const auto width=uint64_t(state->table.geometry().hc)*uint64_t(state->table.geometry().n_embd);
            if(!state->ladder.empty() && layer==0 && residual.size()==width)
                std::copy(residual.begin(),residual.end(),state->ladder.begin());
            auto output=state->layers.at(layer)->decode(residual,ple);
            if(!state->ladder.empty() && output.size()==width)
                std::copy(output.begin(),output.end(),state->ladder.begin()+size_t(layer+1)*width);
            return output;
        }
        // This auxiliary transaction stage runs after the 48 target layers.
        // Its rewind joins the same SessionFlow transaction, including a failed
        // target head or post-head cancellation. Allocate passthrough first.
        std::vector<float> passthrough(residual.begin(),residual.end());
        if(!state->flow->tokens().empty()) {
            auto output=state->mtp->step(state->last_residual,state->pending_embedding);
            std::copy(output.logits.begin(),output.logits.end(),state->pending_mtp_logits.begin());
            std::copy(output.residual.begin(),output.residual.end(),state->pending_mtp_residual.begin());
        }
        return passthrough;
    };
    op.head=[state](std::span<const float> residual) {
        std::copy(residual.begin(),residual.end(),state->pending_residual.begin());
        if(state->chain_head_ready) {
            state->chain_head_ready=false;auto& v=*state->verify_chain;const auto& g=state->table.geometry();
            state->mixed_output.stage(std::span<const float>(static_cast<const float*>(v.mixed_host.data()),g.n_embd),1);
            const auto* logits=static_cast<const float*>(v.logits_host.data());
            return std::vector<float>(logits,logits+g.n_vocab);
        }
        state->final_mix->set("residual",residual);state->final_mix->run();
        const auto mixed=state->final_mix->output("mixed");
        state->mixed_output.stage(mixed,1);
        return state->head->run(mixed);
    };
    if(p->memory_plan.verify_rows) op.window_head=[state,mix_weights,n](std::span<const float> residual,const std::function<bool()>& cancelled) {
        if(cancelled && cancelled()) throw Cancelled();
        auto mixed=state->hc_window->read(residual,mix_weights,true);
        if(cancelled && cancelled()) throw Cancelled();
        auto output=state->head->run_rows(mixed.mixed);
        if(cancelled && cancelled()) throw Cancelled();
        std::copy(residual.end()-n,residual.end(),state->pending_residual.begin());
        state->mixed_output.stage(mixed.mixed,residual.size()/n);
        return output;
    };
    if(p->memory_plan.window_rows) op.window_layer=[state,h,n](uint32_t layer,std::span<const float> residual,std::span<const float> embeddings,
            std::span<const float> ple,const std::function<bool()>& cancelled) {
        if(layer<state->layers.size()) return state->layers.at(layer)->prefill(residual,ple,cancelled);
        // Explicit known target pairs; private MTP transaction remains nested
        // inside the outer target transaction, including rejected prefixes.
        std::vector<float> passthrough(residual.begin(),residual.end());
        const auto point=state->mtp->checkpoint();
        try {
            if(cancelled && cancelled()) throw Cancelled();
            const auto pairs=pair_mtp_prefix(residual,embeddings,state->flow->tokens().empty()?std::span<const float>{}:std::span<const float>(state->last_residual),uint32_t(h),uint32_t(n/h));
            if(!pairs.embedding.empty()) {
                auto out=state->mtp->prefill(pairs.residual,pairs.embedding,cancelled);
                std::copy(out.logits.begin(),out.logits.end(),state->pending_mtp_logits.begin());
                std::copy(out.residual.begin(),out.residual.end(),state->pending_mtp_residual.begin());
            }
            return passthrough;
        } catch(...) {
            try {state->mtp->restore(point);} catch(...) {state->poisoned=true;throw SessionFatal("MTP prefill window rollback failed");}
            throw;
        }
    };
    if(p->control_vectors) op.after_layer=[state](uint32_t layer,std::span<const float> residual) {
        if(!state->control_enabled || !state->control_vectors->covers(layer)) return std::vector<float>(residual.begin(),residual.end());
        auto& graph=*state->control_recipe;graph.set("residual",residual);graph.set("direction",state->control_vectors->direction(layer));
        if(state->control_vectors->mode()==ControlVectorMode::Project) {const std::array<float,1> negative{-state->control_vectors->scale(layer)};graph.set("negative_scale",negative);}
        graph.run();return graph.values("out");
    };
    op.reset_layer=[state](uint32_t layer) {if(layer==state->layers.size()) state->mtp->reset();else state->layers.at(layer)->reset();};
    // STRATA_HPU_CHAIN: every layer fused, nothing that needs a host residual between layers (MTP stage,
    // control vectors, the --dump-layers ladder). Same graphs as the per-layer path, ordered on one stream.
    bool chain=chain_decode_enabled() && !p->mtp && !p->control_vectors && p->ladder.empty();
    for(const auto& layer:p->layers) chain=chain && layer->chainable();
    if(resident_iq4_enabled() && !chain)
        throw std::invalid_argument("STRATA_HPU_RESIDENT_IQ4 needs the layer chain: every layer fused, no MTP, control vectors or --dump-layers");
    if(chain) {
        // STRATA_HPU_QSA_DEVICE stage: per layer the token's K and V rows and raw index key, 128-byte aligned.
        const uint64_t stage=qsa_device_enabled()?uint64_t(g.n_layers)*((2*uint64_t(g.n_head_kv)*g.head_dim*4+uint64_t(g.idx_key_dim)*4+127)/128*128):0;
        p->decode_chain=std::make_unique<DecodeChain>(*p->device,h,uint32_t(g.hc),stage);
        // The verify chain serves both the DFlash round and the resident prefill window: one set of per-layer
        // verify graphs, sized for the largest window either runs.
        uint32_t dflash_rows=0;uint32_t verify_rows=resident_iq4_enabled() && !p->memory_plan.window_rows?
            std::min<uint32_t>(p->memory_plan.prefill_rows,kResidentWindowRows):0;
        if(verify_rows<2) verify_rows=0;
        bool dflash_verify=false;
        if(const char* directory=std::getenv("STRATA_HPU_DFLASH");directory && *directory) {
            // The drafter's candidate GEMM reads the head as BF16 [vocabulary, hidden] (--fp8-experts selects it).
            if(packed_supported(head_type)) throw std::invalid_argument("STRATA_HPU_DFLASH needs the BF16 output head (--fp8-experts, no MTP)");
            p->drafter=std::make_unique<DflashDrafter>(*p->device,directory,p->head_weights->binding("weight"),uint32_t(g.n_vocab),
                                                       m.context,std::max<uint32_t>(p->memory_plan.prefill_rows,1));
            p->mask_embedding=p->embedding_row(p->drafter->config().mask_token,0);
            if(const char* verify=std::getenv("STRATA_HPU_DFLASH_VERIFY");verify && *verify && *verify!='0') {
                if(!resident_iq4_enabled() || !qsa_device_enabled())
                    throw std::invalid_argument("STRATA_HPU_DFLASH_VERIFY needs STRATA_HPU_RESIDENT_IQ4 and STRATA_HPU_QSA_DEVICE");
                const uint32_t rows=std::min<uint32_t>(p->drafter->config().block,kGdnWindowRows);dflash_rows=rows;
                // A prefill window wider than a draft block is captured in one pass (the drafter holds prefill_rows
                // feature rows) and committed to its context block by block; the drafter is only fed whole windows.
                verify_rows=std::max(verify_rows,rows);dflash_verify=true;
                if(const char* timing=std::getenv("STRATA_HPU_DFLASH_SYNC_TIMING");timing && *timing=='1') p->dflash_sync_timing=true;
                if(const char* minimum=std::getenv("STRATA_HPU_DFLASH_MIN");minimum && *minimum) p->dflash_break_even=std::strtod(minimum,nullptr);
                // STRATA_HPU_DFLASH_MARK=1: each window's enqueue range, for a profiler range of one window.
                if(const char* mark=std::getenv("STRATA_HPU_DFLASH_MARK");mark && *mark=='1') p->dflash_mark=true;
                if(const char* sink=std::getenv("STRATA_HPU_DFLASH_LOGITS");sink && *sink)
                    p->dflash_logits=std::make_unique<DflashLogitSink>(sink);
            }
        }
        if(verify_rows) {
            const auto qc=qsa_config(g,m.context,m.page_size);
            const uint64_t stage_bytes=verify_stage_bytes(g,verify_rows),history=uint64_t(9)*n*4;
            const uint64_t mask=uint64_t(verify_rows)*(uint64_t(qc.selection_width(qc.capacity))+verify_rows)*4;
            // With the last-row prefill head a window writes one row of logits (2 GB at 2048 rows otherwise); a DFlash
            // verification round needs its draft block's rows.
            const uint32_t logit_rows=!prefill_last_head_enabled()?verify_rows:dflash_verify?std::max<uint32_t>(dflash_rows,1):1;
            // Final-state prefill windows keep one PLE history; a DFlash round keeps its block's.
            const uint32_t history_rows=prefill_final_enabled()?std::max<uint32_t>(dflash_rows,1):verify_rows;
            p->verify_chain=std::make_unique<VerifyChain>(*p->device,h,uint32_t(g.hc),verify_rows,stage_bytes,history,mask,uint64_t(g.n_vocab),prefill_qsa32_enabled()?128:0,logit_rows,history_rows);
            p->dflash_verify=dflash_verify;
        }
        op.chain=[state,n,streams=uint32_t(g.hc)](std::span<const float> residual,std::span<const float> ple,const SessionOperations::ChainHooks& hooks) {
            auto& c=*state->decode_chain;const auto bytes=checked_product(n,sizeof(float));
            if(residual.size()!=n) throw std::invalid_argument("chain residual shape mismatch");
            try {
                static const bool timing=[] {const char* v=std::getenv("STRATA_HPU_DECODE_TIMING");return v && v[0]=='1';}();
                using Clock=std::chrono::steady_clock;static double enqueue_ms=0,wait_ms=0,commit_ms=0;static uint64_t tokens=0;
                const auto t0=Clock::now();
                std::memcpy(c.staging.data(),residual.data(),bytes);upload(c.stream,c.in(),c.staging,bytes);
                for(uint32_t layer=0;layer<state->layers.size();++layer) {
                    hooks.begin(layer);
                    state->layers[layer]->decode_chain(c,layer==1?ple:std::span<const float>{});
                    if(state->drafter) if(const int slot=state->drafter->feature_slot(layer);slot>=0) state->drafter->capture(c.stream,c.in(),streams,slot);
                    hooks.done(layer);
                }
                // STRATA_HPU_CHAIN_HEAD: the final HC read and the head behind the layers, their results downloaded on
                // the same stream (the verify chain's head buffers are idle while a token decodes).
                const bool head=chain_head_enabled() && state->verify_chain;
                if(head) {
                    auto& v=*state->verify_chain;auto& recipe=state->verify_head_for(1,true);
                    const uint64_t vocabulary=uint64_t(state->table.geometry().n_vocab),h=n/streams;
                    recipe.bind_external({"final.residual",&c.in()});recipe.bind_external({"head.logits",&v.logits});
                    recipe.bind_external({"final.mixed",&v.mixed});
                    recipe.launch(c.stream);
                    download(c.stream,v.logits_host,v.logits,vocabulary*4);download(c.stream,v.mixed_host,v.mixed,h*4);
                }
                download(c.stream,c.staging,c.in(),bytes);
                if(!c.commits.empty()) download(c.stream,c.layer_stage_host,c.layer_stage,c.layer_stage.size());
                const auto t1=Clock::now();
                c.stream.synchronize();c.leases.clear();
                state->chain_head_ready=head;
                const auto t2=Clock::now();
                // STRATA_HPU_QSA_DEVICE layers append their rows to the host state now, in layer order.
                auto commits=std::move(c.commits);c.commits.clear();
                for(auto& commit:commits) commit(static_cast<const uint8_t*>(c.layer_stage_host.data()));
                if(timing) {
                    const auto t3=Clock::now();const auto ms=[](auto a,auto b) {return std::chrono::duration<double,std::milli>(b-a).count();};
                    enqueue_ms+=ms(t0,t1);wait_ms+=ms(t1,t2);commit_ms+=ms(t2,t3);
                    if(++tokens%64==0) {
                        std::fprintf(stderr,"strata-hpu: decode chain ms/token enqueue=%.3f wait=%.3f commit=%.3f\n",enqueue_ms/64,wait_ms/64,commit_ms/64);
                        enqueue_ms=wait_ms=commit_ms=0;
                    }
                }
                if(state->drafter) {
                    // This token's features become drafter context at its position (a new sequence starts over).
                    const auto position=state->flow->tokens().size();
                    // A new sequence, or one rewound below the drafter's context, starts the drafter over.
                    if(!position || position<state->drafter->cells()) {state->drafter->reset();state->drafts.clear();}
                    if(state->drafter->cells()==position) state->drafter->append(c.stream,uint32_t(position));
                }
                const auto* values=static_cast<const float*>(c.staging.data());
                return std::vector<float>(values,values+n);
            } catch(...) {
                try {c.stream.synchronize();} catch(...) {}
                c.leases.clear();c.commits.clear();throw;
            }
        };
    }
    p->flow=std::make_unique<SessionFlow>(uint32_t(h),uint32_t(g.hc),uint32_t(g.n_layers)+bool(p->mtp),uint32_t(g.n_vocab),m.context,std::move(op));impl_=std::move(p);
}
Model::~Model()=default;
std::vector<float> Model::decode(int32_t token,const std::function<bool()>& cancelled,const RoutingWindowSink& routing) {
    auto& p=*impl_;std::lock_guard lock(p.mutex);if(p.poisoned) throw std::logic_error("model restore failed; reset required");
    if(routing && !p.routing_capture.enabled()) throw std::invalid_argument("scalar routing callback requires capture");
    // STRATA_HPU_DECODE_TIMING=1 (diagnostic): the model-level phases around the session's token.
    static const bool timing=[] {const char* v=std::getenv("STRATA_HPU_DECODE_TIMING");return v && v[0]=='1';}();
    using Clock=std::chrono::steady_clock;static std::array<double,4> spent{};static uint64_t tokens=0;
    auto mark=Clock::now();
    const auto lap=[&](size_t i) {if(!timing) return;const auto now=Clock::now();spent[i]+=std::chrono::duration<double,std::milli>(now-mark).count();mark=now;};
    auto trace=p.routing_capture.round(1);
    auto halves=p.half_capture.round();
    if(p.drafter) p.settle_draft();
    lap(0);
    auto result=p.flow->decode(token,cancelled);lap(1);
    p.commit_output(result);p.after_round();trace.finish();halves.finish();p.note_drafter();lap(2);
    if(routing) routing(p.routing_capture.physical_view(),false);
    if(p.drafter && !p.dflash_verify) p.dflash_observe(p.flow->tokens().size()-1,token,result);
    lap(3);
    if(timing && ++tokens%64==0) {
        std::fprintf(stderr,"strata-hpu: decode model ms/token before=%.3f flow=%.3f after=%.3f observe=%.3f\n",
                     spent[0]/64,spent[1]/64,spent[2]/64,spent[3]/64);
        spent={};
    }
    return result;
}
std::vector<float> Model::prefill(std::span<const int32_t> tokens,const std::function<bool()>& cancelled) {
    auto& p=*impl_;std::lock_guard lock(p.mutex);
    if(p.poisoned || tokens.empty() || tokens.size()>p.options.memory.context-p.flow->tokens().size()) throw std::invalid_argument("model state or prefill length invalid");
    for(auto token:tokens) if(token<0 || uint64_t(token)>=uint64_t(p.table.geometry().n_vocab)) throw std::invalid_argument("prefill token out of vocabulary");
    if(p.drafter) p.settle_draft();
    std::vector<float> result;
    for(auto token:tokens) {
        auto trace=p.routing_capture.round(1);
        auto halves=p.half_capture.round();
        result=p.flow->decode(token,cancelled);p.commit_output(result);trace.finish();halves.finish();p.note_drafter();
    }
    return result;
}
// The resident multi-row prefill window: the same per-layer verify graphs a DFlash round runs, every row of the
// window committed. Nothing is accepted before every layer's verify_commit(), so a failure or a cancellation
// leaves the committed prefix untouched (the layers only wrote scratch, staged rows and cache cells past it).
std::vector<float> Model::prefill_resident_window(std::span<const int32_t> tokens,const std::function<bool()>& cancelled) {
    auto& p=*impl_;
    if(!p.verify_chain || tokens.size()<2 || tokens.size()>p.verify_chain->max_rows) throw std::logic_error("resident prefill window is not configured");
    if(p.poisoned || p.flow->poisoned()) throw SessionFatal("resident prefill target state failed");
    auto& v=*p.verify_chain;
    struct FinalMode {VerifyChain& chain;bool previous;~FinalMode(){chain.prefill_final=previous;}} mode{v,v.prefill_final};
    v.prefill_final=prefill_final_enabled();
    // STRATA_HPU_MOE_MME: a wide final-state window runs its MoE as grouped MME GEMMs over row tiles (moe_mme.hpp).
    struct MoeMode {VerifyChain& chain;uint32_t previous;~MoeMode(){chain.moe_mme_bucket=previous;}} moe_mode{v,v.moe_mme_bucket};
    v.moe_mme_bucket=moe_mme_enabled() && v.prefill_final && tokens.size()>=moe_mme_min_rows()
        ?moe_mme_bucket_rows(uint32_t(tokens.size()),p.table.geometry()):0;
    const auto& g=p.table.geometry();
    const uint64_t h=g.n_embd,n=h*g.hc,vocabulary=g.n_vocab;const uint32_t rows=uint32_t(tokens.size());
    const auto position=p.flow->tokens().size();
    for(const auto token:tokens) if(token<0 || uint64_t(token)>=vocabulary) throw std::invalid_argument("resident prefill token out of vocabulary");
    if(cancelled && cancelled()) throw Cancelled();
    using Clock=std::chrono::steady_clock;
    const auto ms=[](Clock::time_point a,Clock::time_point b) {return std::chrono::duration<double,std::milli>(b-a).count();};
    // The window's host input rows (the embedding in every stream) and the layer-1 PLE rows, causal like run_window().
    v.rows=rows;v.current=0;
    const auto started=p.prefill_profile?Clock::now():Clock::time_point{};
    auto* input=static_cast<float*>(v.staging.data());
    std::vector<int32_t> previous(p.flow->tokens().begin(),p.flow->tokens().end());
    // The same position source as the token seam, so a positional --embd-gguf file and the tail's scalar
    // rows cannot diverge.
    std::vector<uint64_t> at(rows);
    for(uint32_t r=0;r<rows;++r) at[r]=p.embedding_window.next(tokens[r],position+r);
    Impl::PrefetchedRows built;
    if(p.prefetched_rows.valid()) {
        try {built=p.prefetched_rows.get();} catch(...) {built={};}
        bool same=built.tokens.size()==rows && std::equal(tokens.begin(),tokens.end(),built.tokens.begin()) && built.history==previous;
        for(uint32_t r=0;same && r<rows;++r) same=at[r]==built.position+r;
        if(!same) built={};
    }
    const auto ple_started=p.prefill_phases?Clock::now():Clock::time_point{};
    bool contiguous=true;for(uint32_t r=1;r<rows;++r) contiguous=contiguous && at[r]==at[0]+r;
    // Not prefetched: the PLE rows (first read by layer 1) are gathered on a thread while layer 0 runs; the next
    // window's prefetch (which gathers PLE rows too) starts once they are in.
    std::future<void> ple_pending;
    if(built.tokens.empty() && contiguous) {
        built=p.build_rows(tokens,at[0],previous,false);
        if(p.ple) {
            built.ple.resize(uint64_t(rows)*h);
            ple_pending=std::async(std::launch::async,[&p,&built,tokens,&previous,h] {p.ple->gather_rows(tokens,previous,built.ple.data(),h);});
        }
    }
    if(built.tokens.empty()) {
        // No prefetch (the first window, or a different one): the rows at the cursor's positions, as always.
        built.embedding.resize(uint64_t(rows)*h);
        for(uint32_t r=0;r<rows;++r) {
            const auto row=p.embedding_row(tokens[r],at[r]);
            if(row.size()!=h) throw std::runtime_error("embedding row width mismatch");
            std::copy(row.begin(),row.end(),built.embedding.begin()+uint64_t(r)*h);
        }
        if(p.ple) {built.ple.resize(uint64_t(rows)*h);p.ple->gather_rows(tokens,previous,built.ple.data(),h);}
    }
    if(p.prefill_phases) p.prefill_phase_ms[5]+=ms(ple_started,Clock::now());
    // The window input: the embedding rows once (the device broadcasts them to every HC stream, window_replicate_for),
    // copied into the staging buffer on eight threads.
    {
        const auto copy=[&](uint32_t a,uint32_t b) {std::copy_n(built.embedding.begin()+uint64_t(a)*h,uint64_t(b-a)*h,input+uint64_t(a)*h);};
        if(rows>=256) {
            std::vector<std::future<void>> parts;
            for(uint32_t part=0;part<8;++part) parts.push_back(std::async(std::launch::async,copy,rows*part/8,rows*(part+1)/8));
            for(auto& part:parts) part.get();
        } else copy(0,rows);
    }
    const std::vector<float>& ple_rows=built.ple;
    // The drafter's context follows the chain's token rule one window at a time: whole windows while its cells are
    // exactly this prefix (its ring keeps the last sliding-window positions of any context).
    auto& drafter=p.drafter;
    if(drafter && uint64_t(position)<drafter->cells()) {drafter->reset();p.drafts.clear();}
    const bool capture=drafter && drafter->cells()==position;
    const uint64_t last=uint64_t(rows-1);
    if(v.logit_rows<rows && !prefill_last_head_enabled()) throw std::logic_error("verify chain logits buffer too small");
    const uint64_t logit_row=v.logit_rows<rows?0:last; // where the head writes the committed row's logits
    // Every phase below ends at a stream wait (except the host input rows), so the five sum to the window total.
    std::array<double,5> phase{};auto mark=started;
    // A QSA layer's rows (its stage slice) are downloaded and committed to its host state as soon as its graph ran,
    // overlapping the later layers (no later layer of the window reads another layer's QSA state); layer 1 also
    // commits the PLE history, which only the window's end downloads, so it commits there.
    // STRATA_HPU_PREFILL_EARLY_COMMIT=0: every commit at the window's end, as before.
    static const bool early_commit=[] {const char* v=std::getenv("STRATA_HPU_PREFILL_EARLY_COMMIT");return !v || v[0]!='0';}();
    const auto early_layer=[&](uint32_t layer) {return early_commit && core::is_qsa_layer(g,layer) && layer!=1;};
    std::vector<std::future<void>> early;
    while(p.layer_ready.size()<p.layers.size()) {
        p.layer_ready.push_back(std::make_unique<Event>(*p.device));p.layer_copied.push_back(std::make_unique<Event>(*p.device));
    }
    const auto join_early=[&] {
        std::exception_ptr failed;
        for(auto& f:early) try {f.get();} catch(...) {if(!failed) failed=std::current_exception();}
        early.clear();return failed;
    };
    if(p.prefill_phases) {phase[0]=ms(mark,Clock::now());mark=Clock::now();}
    try {
        // The embedding rows into the other residual buffer (free until layer 0 writes it), broadcast into in().
        upload(v.stream,v.out(),v.staging,uint64_t(rows)*h*4);
        p.window_replicate_for(rows,v.out(),v.in()).launch(v.stream);
        const auto prefetch_next=[&] {
        if(!p.next_window.empty()) {
            // The next window starts right after this one's last position, with this window in its n-gram history.
            auto next=std::move(p.next_window);p.next_window.clear();
            auto history=previous;history.insert(history.end(),tokens.begin(),tokens.end());
            const uint64_t first=at[rows-1]+1;
            p.prefetched_rows=std::async(std::launch::async,[&p,next=std::move(next),history=std::move(history),first] {
                return p.build_rows(next,first,history);
            });
        }
        };
        if(!ple_pending.valid()) prefetch_next();
        for(uint32_t layer=0;layer<p.layers.size();++layer) {
            if(layer==1 && ple_pending.valid()) {ple_pending.get();prefetch_next();}
            p.layers[layer]->verify_chain(v,layer==1?std::span<const float>(ple_rows):std::span<const float>{});
            if(early_layer(layer)) {
                const uint64_t stride=verify_stage_stride(g,v.max_rows);
                p.layer_ready[layer]->record(v.stream);v.copy.wait(*p.layer_ready[layer]);
                const uint64_t at=verify_stage_offset(g,layer,v.max_rows);
                download(v.copy,v.stage_host,v.stage,stride,at,at);
                p.layer_copied[layer]->record(v.copy);
                early.push_back(std::async(std::launch::async,[&p,&v,layer,rows] {p.layer_copied[layer]->synchronize();p.layers[layer]->verify_commit(v,rows);}));
            }
            if(capture) if(const int slot=drafter->feature_slot(layer);slot>=0) drafter->capture(v.stream,v.in(),uint32_t(g.hc),slot,rows);
            if(p.prefill_layers && p.prefill_windows>=p.prefill_layers_first && p.prefill_windows<p.prefill_layers_first+p.prefill_layers_count) {
                v.stream.synchronize();const auto now=Clock::now();
                std::fprintf(stderr,"strata-hpu: prefill layer %u %s %.3f ms\n",layer,core::is_qsa_layer(g,layer)?"qsa":"gdn",ms(mark,now));mark=now;
            }
        }
        const bool last_head=prefill_last_head_enabled();
        auto& head=p.verify_head_for(rows,last_head);
        head.bind_external({"final.residual",&v.in(),last_head?last*n*4:0});
        if(last_head) {
            head.bind_external({"head.logits",&v.logits,logit_row*vocabulary*4});
            head.bind_external({"final.mixed",&v.mixed,last*h*4});
        }
        // The stage and residual rows are read only at/after the commit, so their readback rides the copy stream
        // alongside the head instead of holding the head back by the transfer time.
        if(p.prefill_phases) {v.stream.synchronize();phase[1]=ms(mark,Clock::now());mark=Clock::now();}
        // STRATA_HPU_MEM_TRACE=1 (diagnostic): per window, the HBM ledger and what compiled recipes hold of it.
        static const bool mem_trace=performance_env("STRATA_HPU_MEM_TRACE");
        if(mem_trace) {
            const auto s=p.device->hbm_allocations();
            std::fprintf(stderr,"strata-hpu: mem rows %u used %.1f MiB peak %.1f MiB recipes %llu own workspace %.1f MiB host-recipe slots %.1f MiB\n",
                         rows,double(s.used)/1048576.0,double(s.peak)/1048576.0,(unsigned long long)recipe_live_count(),
                         double(recipe_own_workspace_bytes())/1048576.0,double(HostRecipe::total_device_bytes())/1048576.0);
        }
        v.copy_ready.record(v.stream);v.copy.wait(v.copy_ready);
        // Only what the commit reads, at the same offsets: the QSA layers' stage slices (the GDN layers' slices are
        // never written), and the last row's residual and PLE history (the window commits all its rows, so those
        // are the session's next state). The whole buffers were ~650 MB a 1024-row window.
        {
            const uint64_t stride=verify_stage_stride(g,v.max_rows);
            for(uint32_t layer=0;layer<p.layers.size();++layer)
                if(core::is_qsa_layer(g,layer) && !early_layer(layer)) {
                    const uint64_t at=verify_stage_offset(g,layer,v.max_rows);download(v.copy,v.stage_host,v.stage,stride,at,at);
                }
        }
        download(v.copy,v.staging,v.in(),n*4,last*n*4,last*n*4);
        if(p.ple) {const uint64_t at=v.prefill_final?0:last*9*n*4;download(v.copy,v.ple_history_host,v.ple_history,9*n*4,at,at);}
        v.copy_done.record(v.copy);
        head.launch(v.stream);head.wait();
        v.copy_done.synchronize();
        if(p.prefill_phases) {phase[2]=ms(mark,Clock::now());mark=Clock::now();}
        download(v.stream,v.logits_host,v.logits,vocabulary*4,logit_row*vocabulary*4,logit_row*vocabulary*4);
        download(v.stream,v.mixed_host,v.mixed,h*4,last*h*4,last*h*4);
        v.stream.synchronize();
        if(p.prefill_phases) {phase[3]=ms(mark,Clock::now());mark=Clock::now();}
    } catch(...) {
        try {v.stream.synchronize();} catch(...) {}
        try {v.copy.synchronize();} catch(...) {}
        // Layers that already committed this window's rows cannot be rolled back.
        if(!early.empty()) {join_early();p.poisoned=true;}
        throw;
    }
    // Once QSA layers committed early the window is finished (commits are not rolled back), so a cancellation here
    // takes effect after it.
    if(cancelled && cancelled() && early.empty()) throw Cancelled();
    // Both rows were downloaded at their row offset above: the committed row is the window's last.
    const auto* logits=static_cast<const float*>(v.logits_host.data())+logit_row*vocabulary;
    const auto* mixed=static_cast<const float*>(v.mixed_host.data())+last*h;
    // Checked before the commit, so a bad head row is refused without poisoning the session state -- unless QSA
    // layers already committed early.
    const bool bad_logits=!all_finite(std::span<const float>(logits,vocabulary)),bad_mixed=!bad_logits && !all_finite(std::span<const float>(mixed,h));
    if(bad_logits || bad_mixed) {
        if(!early.empty()) {join_early();p.poisoned=true;}
        throw NonFiniteOutput(bad_logits?"non-finite resident prefill logits":"non-finite resident prefill mixed input");
    }
    try {
        // The QSA layers' commits append every row to their own host state (independent layers, each under its own
        // lock, reading the downloaded stage only), so they run concurrently; the GDN commits stay on this thread.
        {
            std::vector<std::future<void>> qsa;
            for(size_t index=0;index<p.layers.size();++index)
                if(core::is_qsa_layer(g,uint32_t(index)) && !early_layer(uint32_t(index))) qsa.push_back(std::async(std::launch::async,[&,index] {p.layers[index]->verify_commit(v,rows);}));
            std::exception_ptr failed=join_early();
            for(size_t index=0;index<p.layers.size();++index)
                if(!core::is_qsa_layer(g,uint32_t(index))) try {p.layers[index]->verify_commit(v,rows);} catch(...) {if(!failed) failed=std::current_exception();}
            for(auto& f:qsa) try {f.get();} catch(...) {if(!failed) failed=std::current_exception();}
            if(failed) std::rethrow_exception(failed);
        }
        p.flow->history_.insert(p.flow->history_.end(),tokens.begin(),tokens.end());
        // Waits: the token path may capture into the drafter's staging on the decode stream right after this.
        if(capture) {
            // The whole window was captured in one pass and goes into the drafter's ring in one append.
            drafter->append(v.stream,uint32_t(position),0,rows,true);
        }
        const auto* residual=static_cast<const float*>(v.staging.data())+uint64_t(rows-1)*n;
        std::copy(residual,residual+n,p.pending_residual.begin());
        p.mixed_output.stage(std::span<const float>(mixed,h),1);
    } catch(const std::exception& error) {p.poisoned=true;throw SessionFatal(std::string("resident prefill window commit failed: ")+error.what());}
    if(p.prefill_phases) {
        v.stream.synchronize();phase[4]=ms(mark,Clock::now());
        for(size_t i=0;i<phase.size();++i) p.prefill_phase_ms[i]+=phase[i];
    }
    // The window index orders the diagnostic layer profile too, so it advances whenever any diagnostic runs.
    if(p.prefill_profile || p.prefill_layers) ++p.prefill_windows;
    if(p.prefill_profile) {
        p.prefill_committed+=rows;p.prefill_ms+=std::chrono::duration<double,std::milli>(Clock::now()-started).count();
        if(p.prefill_windows%32==0 || p.prefill_phases) {
            const double windows=double(p.prefill_windows);
            std::fprintf(stderr,"strata-hpu: prefill resident windows=%llu rows=%llu ms/window=%.2f ms/row=%.3f\n",
                         (unsigned long long)p.prefill_windows,(unsigned long long)p.prefill_committed,p.prefill_ms/windows,p.prefill_ms/double(p.prefill_committed));
            if(p.prefill_phases)
                std::fprintf(stderr,"strata-hpu: prefill phases ms/window host_rows=%.2f (rows_build=%.2f) layers=%.2f head_readback=%.2f tail=%.2f commit_append=%.2f (sum=%.2f total=%.2f)\n",
                             p.prefill_phase_ms[0]/windows,p.prefill_phase_ms[5]/windows,p.prefill_phase_ms[1]/windows,p.prefill_phase_ms[2]/windows,p.prefill_phase_ms[3]/windows,
                             p.prefill_phase_ms[4]/windows,
                             (p.prefill_phase_ms[0]+p.prefill_phase_ms[1]+p.prefill_phase_ms[2]+p.prefill_phase_ms[3]+p.prefill_phase_ms[4])/windows,
                             p.prefill_ms/windows);
        }
    }
    return std::vector<float>(logits,logits+vocabulary);
}
void Model::prefill_window(std::span<const int32_t> tokens,const PromptCommit& commit,const std::function<bool()>& cancelled,
                           const RoutingWindowSink& routing,bool scalar) {
    auto& p=*impl_;std::lock_guard lock(p.mutex);
    if(!commit || p.poisoned || tokens.empty() || tokens.size()>p.options.memory.context-p.flow->tokens().size() || (routing && !p.routing_capture.enabled())) throw std::invalid_argument("model state or prefill callback/window invalid");
    for(auto token:tokens) if(token<0 || uint64_t(token)>=uint64_t(p.table.geometry().n_vocab)) throw std::invalid_argument("prefill token out of vocabulary");
    if(p.drafter) p.settle_draft();
    // The positional row file needs the positions this window claims; the guard below also closes it when a
    // stage throws, so a failed window never leaves a stale cursor for the next request.
    struct WindowGuard {EmbeddingWindow& window;~WindowGuard(){window.close();}} guard{p.embedding_window};
    p.embedding_window.open(p.flow->tokens().size(),std::vector<int32_t>(tokens.begin(),tokens.end()));
    // A prefetch left behind by a failed or cancelled window is waited for and dropped, so no host thread reads the
    // embedding or PLE sources once this call returns.
    // The prefetch of the next window (named by prefill_next()) finishes before this call returns, so no host thread
    // reads the embedding or PLE sources outside a prefill call; its rows wait for the next call.
    struct PrefetchGuard {Impl& p;~PrefetchGuard(){
        if(p.prefetched_rows.valid()) try {p.prefetched_rows.wait();} catch(...) {}
        p.next_window.clear();p.hinted_next.clear();
    }} prefetch_guard{p};
    uint32_t accepted=0;
    while(accepted<tokens.size()) {
        const size_t remaining=tokens.size()-accepted;
        // The resident window takes full rows only (every row count is its own set of compiled graphs) and only
        // while every layer can still take that many rows; anything else falls back to the token path.
        bool resident=false;uint32_t rows=1;
        if(p.verify_chain && !scalar && p.memory_plan.prefill_rows>=2) {
            uint32_t count=p.memory_plan.prefill_rows;
            if(prefill_qsa32_enabled() && remaining>=32) {
                count=std::bit_floor(uint32_t(std::min<size_t>(remaining,count)));
                // Do not mix the device norm law and the staged host law inside a canonical window.
                const auto& g=p.table.geometry();const uint64_t limit=uint64_t(g.idx_top_k)+g.idx_block-1,position=p.flow->tokens().size();
                if(count>32 && position<limit) {
                    const uint32_t fit=uint32_t(std::min<uint64_t>(count,(limit-position)/32*32));
                    if(fit && fit<count) count=std::bit_floor(fit);
                }
            }
            // A tail shorter than a window runs as power-of-two windows down to 2 rows (each size compiles its
            // graphs once) instead of one decode step per token: a 1,800-token tail at 100k context was ~4 minutes.
            if(remaining<count && !prefill_qsa32_enabled()) count=remaining>=2?std::bit_floor(uint32_t(remaining)):0;
            if(count>=2 && remaining>=count) {
                bool ready=true;for(const auto& layer:p.layers) ready=ready && layer->verify_ready(count);
                if(ready) {resident=true;rows=count;}
            }
        }
        const bool window=!resident && p.memory_plan.window_rows && !scalar;
        if(!resident) rows=window?uint32_t(std::min<size_t>(p.memory_plan.window_rows,remaining)):1;
        auto trace=p.routing_capture.round(rows);
        auto halves=p.half_capture.round(!resident && !window);
        // A following full window of the same size is prefetched while this one runs on the device.
        if(resident && tokens.size()-accepted-rows>=rows) p.next_window.assign(tokens.begin()+accepted+rows,tokens.begin()+accepted+2*rows);
        else if(resident && tokens.size()-accepted==rows && p.hinted_next.size()==rows) p.next_window=p.hinted_next;
        else p.next_window.clear();
        const auto logits=resident?prefill_resident_window(tokens.subspan(accepted,rows),cancelled)
                          :window?p.flow->prefill_window(tokens.subspan(accepted,rows),cancelled):p.flow->decode(tokens[accepted],cancelled);
        p.commit_output(logits);trace.finish();halves.finish();accepted+=rows;p.note_drafter();commit(accepted,logits);
        if(routing) routing(p.routing_capture.physical_view(),window || resident);
        if(!scalar) p.after_round(); // forced scalar matches Model::prefill's adaptation cadence
    }
}
void Model::prefill_next(std::span<const int32_t> tokens) {
    auto& p=*impl_;std::lock_guard lock(p.mutex);p.hinted_next.assign(tokens.begin(),tokens.end());
}
void Model::reset() {
    auto& p=*impl_;std::lock_guard lock(p.mutex);p.routing_capture.clear();p.half_capture.clear();p.poisoned=true;
    const auto reset_started=std::chrono::steady_clock::now();
    p.flow->reset();p.poisoned=false;
    if(p.prefill_phases) std::fprintf(stderr,"strata-hpu: model reset %.2f ms\n",
                                      std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-reset_started).count());
    // The conversation the retained image described is gone: upstream discards the reuse slot the same
    // way when a session is replaced (generate.cpp:10599).
    p.parked.reset();p.parked_tokens.clear();p.parked_reused=0;
}
void Model::set_control_vector(bool enabled) {
    auto& p=*impl_;std::lock_guard lock(p.mutex);if(!p.control_vectors || enabled==p.control_enabled) return;
    p.routing_capture.clear();p.half_capture.clear();p.poisoned=true;p.flow->reset();p.control_enabled=enabled;p.poisoned=false;
}
bool Model::control_vector_enabled() const {auto& p=*impl_;std::lock_guard lock(p.mutex);return p.control_enabled;}
void Model::set_vision(VisionSelection selection) {
    auto& p=*impl_;std::lock_guard lock(p.mutex);
    if(!p.options.vision) {
        if(selection.embeddings || !selection.rows.empty() || !selection.positions.empty())
            throw std::invalid_argument("this engine was started without --vision");
        return;
    }
    const bool identity=selection.positions.empty();
    if(identity) {
        if(!selection.rows.empty()) throw std::invalid_argument("image rows require an image position table");
        p.positions.reset_identity();p.vision_embeddings.reset();p.vision_rows.clear();return;
    }
    // Validate everything before the first write, so a refused request leaves the engine's table as it was.
    if(selection.positions.size()!=p.positions.cells()*3) throw std::invalid_argument("image position table shape mismatch");
    if(!selection.embeddings) throw std::invalid_argument("image rows need their embeddings file");
    if(selection.rows.size()>p.positions.cells()) throw std::invalid_argument("image row list exceeds the position table");
    for(const auto row:selection.rows) {
        if(row>=int64_t(selection.embeddings->rows())) throw std::invalid_argument("image row index out of range");
        if(row>=0 && selection.embeddings->hidden()!=uint64_t(p.table.geometry().n_embd))
            throw std::invalid_argument("bad embeddings file (expected strata-vision records of width "+
                                        std::to_string(int64_t(p.table.geometry().n_embd))+")");
    }
    for(const auto value:selection.positions) if(value<0) throw std::invalid_argument("image position table contains a negative position");
    p.positions.install(selection.positions);
    p.vision_embeddings=std::move(selection.embeddings);
    p.vision_rows=std::move(selection.rows);
}
std::string Model::control_vector_summary() const {auto& p=*impl_;std::lock_guard lock(p.mutex);return p.control_vectors?p.control_vectors->summary():"0";}
std::vector<int32_t> Model::tokens() const {const auto& p=*impl_;std::lock_guard lock(p.mutex);const auto ids=p.flow->tokens();return {ids.begin(),ids.end()};}
uint64_t Model::size() const {const auto& p=*impl_;std::lock_guard lock(p.mutex);return p.flow->tokens().size();}
std::vector<float> Model::logits() const {const auto& p=*impl_;std::lock_guard lock(p.mutex);if(p.poisoned || p.flow->poisoned()) throw std::logic_error("model state is failed");return p.flow->tokens().empty()?std::vector<float>{}:p.last_logits;}
std::vector<float> Model::residual() const {const auto& p=*impl_;std::lock_guard lock(p.mutex);if(p.poisoned || p.flow->poisoned()) throw std::logic_error("model state is failed");return p.flow->tokens().empty()?std::vector<float>{}:p.last_residual;}
std::vector<float> Model::mixed() const {const auto& p=*impl_;std::lock_guard lock(p.mutex);if(p.poisoned || p.flow->poisoned()) throw std::logic_error("model state is failed");const auto row=p.mixed_output.current();return p.flow->tokens().empty()?std::vector<float>{}:std::vector<float>(row.begin(),row.end());}
std::vector<float> Model::layer_ladder() const {const auto& p=*impl_;std::lock_guard lock(p.mutex);return p.ladder;}
std::vector<RoutingRecord> Model::routing_records() const {const auto& p=*impl_;std::lock_guard lock(p.mutex);return p.routing_capture.records();}
std::vector<float> Model::half_records() const {const auto& p=*impl_;std::lock_guard lock(p.mutex);return p.half_capture.records();}
std::vector<DraftRow> Model::draft(const DraftContext& context,uint32_t maximum,const std::function<bool()>& cancelled) {
    auto& p=*impl_;std::lock_guard lock(p.mutex);if(!p.mtp || !maximum) return {};
    const auto tokens=p.flow->tokens();
    if(p.poisoned || p.flow->poisoned() || !p.mtp->healthy()) throw SessionFatal("MTP/target state failed");
    if(maximum>7 || tokens.size()>p.options.memory.context || maximum>p.options.memory.context-tokens.size() ||
       !std::equal(tokens.begin(),tokens.end(),context.history.begin(),context.history.end()) || context.logits.size()!=p.last_logits.size() ||
       !std::equal(p.last_logits.begin(),p.last_logits.end(),context.logits.begin()) || (!context.bias.empty() && context.bias.size()!=p.last_logits.size()) ||
       maximum-1>UINT64_MAX-context.sampling.counter) throw std::invalid_argument("MTP proposal context/window mismatch");
    if(tokens.size()<2) return {}; // first real residual/token pair does not yet exist
    if(!context.confidence.empty() && context.confidence.size()<maximum) throw std::invalid_argument("adaptive confidence window is too short");
    if(p.mtp->size()!=tokens.size()-1) throw SessionFatal("MTP is not aligned with target prefix");
    std::vector<DraftRow> rows;rows.reserve(maximum);
    std::vector<int32_t> history(tokens.begin(),tokens.end());history.reserve(history.size()+maximum);
    auto logits=p.last_mtp_logits,residual=p.last_mtp_residual;std::vector<float> embedding(p.table.geometry().n_embd);
    const auto point=p.mtp->checkpoint();
    auto rewind=[&] {
        try {p.mtp->restore(point);}
        catch(const std::exception& error) {p.poisoned=true;throw SessionFatal(std::string("MTP draft rollback failed: ")+error.what());}
    };
    try {
        for(uint32_t j=0;j<maximum;++j) {
            if(cancelled && cancelled()) throw Cancelled();
            auto parameters=context.sampling;parameters.counter+=j;
            auto pick=pick_mtp_draft(logits,history,parameters,context.bias,context.mode,p.options.mtp_sampling);
            if(pick.row.token<0) break;
            if(!context.confidence.empty()) context.confidence[j]=pick.confidence;
            const auto id=pick.row.token;rows.push_back(std::move(pick.row));
            if(j+1==maximum || (p.options.mtp_min_p>0 && pick.confidence<p.options.mtp_min_p) ||
               std::find(p.stops.begin(),p.stops.end(),id)!=p.stops.end()) break;
            history.push_back(id);embedding=p.embedding_row(id,uint64_t(history.size())-1);
            auto next=p.mtp->step(residual,embedding);residual=std::move(next.residual);logits=std::move(next.logits);
        }
    } catch(...) {rewind();throw;}
    rewind();return rows;
}
bool Model::dflash_ready() const {const auto& p=*impl_;std::lock_guard lock(p.mutex);return p.dflash_verify && p.verify_chain;}
DflashRound Model::dflash_window(int32_t pending,uint32_t maximum,const DflashSelection& select,const std::function<bool()>& cancelled) {
    auto& p=*impl_;std::lock_guard lock(p.mutex);
    if(!p.dflash_verify || !p.verify_chain || !p.drafter || !select) throw std::logic_error("STRATA_HPU_DFLASH_VERIFY is not configured");
    if(p.poisoned || p.flow->poisoned()) throw SessionFatal("DFlash target state failed");
    const auto& g=p.table.geometry();const uint64_t h=g.n_embd,n=h*g.hc,vocabulary=g.n_vocab;
    const auto history=p.flow->tokens();const uint64_t position=history.size();const auto context=p.options.memory.context;
    if(pending<0 || uint64_t(pending)>=vocabulary || !maximum) throw std::invalid_argument("DFlash window pending token/limit");
    if(!position || position>=context) return {};
    if(p.dflash_skip) {--p.dflash_skip;++p.dflash_skipped;return {};}
    auto& v=*p.verify_chain;auto& drafter=*p.drafter;
    // The window keeps its full row count even when fewer tokens may be committed (`maximum`): every row count is
    // its own set of compiled graphs, so a short tail would compile them all again.
    const uint32_t limit=uint32_t(std::min<uint64_t>(v.max_rows,context-position));
    // The drafter's context must be exactly this prefix; otherwise the window verifies the pending token alone.
    const bool synced=drafter.cells()==position &&
                      std::equal(p.drafter_tokens.begin(),p.drafter_tokens.end(),history.begin(),history.end());
    using Clock=std::chrono::steady_clock;const auto ms=[](Clock::time_point a,Clock::time_point b) {return std::chrono::duration<double,std::milli>(b-a).count();};
    const auto t0=Clock::now();
    std::vector<int32_t> ids{pending};
    const bool prepared=p.dflash_prepared.active && p.dflash_prepared.position==position && p.dflash_prepared.pending==pending;
    if(!prepared) p.settle_draft();
    if(limit>1 && synced) {
        std::vector<int32_t> drafted;
        if(prepared) {p.dflash_prepared.active=false;drafted=drafter.end_draft();}
        else drafted=drafter.draft(p.embedding_row(pending,position),p.mask_embedding,pending,uint32_t(position));
        for(const auto token:drafted) {if(ids.size()>=limit || token<0 || uint64_t(token)>=vocabulary) break;ids.push_back(token);}
    } else p.settle_draft();
    uint32_t rows=uint32_t(ids.size());
    // A layer that cannot take the window (QSA beyond the identity selection, capacity) shrinks it; without a
    // draft to check, decode() is the cheaper way to feed the pending token.
    while(rows>1) {
        bool ready=true;for(const auto& layer:p.layers) ready=ready && layer->verify_ready(rows);
        if(ready) break;
        --rows;
    }
    if(rows<2) return {};
    ids.resize(rows);
    if(cancelled && cancelled()) throw Cancelled();
    // The window's input rows (the embedding in every stream) and the layer-1 PLE rows, causal like run_window().
    const auto t1=Clock::now();
    v.rows=rows;v.current=0;
    auto* input=static_cast<float*>(v.staging.data());
    std::vector<float> ple_rows(p.ple?uint64_t(rows)*h:0);std::vector<int32_t> previous(history.begin(),history.end());
    for(uint32_t r=0;r<rows;++r) {
        auto host_mark=p.dflash_host_profile?Clock::now():Clock::time_point{};
        const auto row=p.embedding_row(ids[r],position+r);
        if(p.dflash_host_profile) {const auto now=Clock::now();p.dflash_host_ms[0]+=ms(host_mark,now);host_mark=now;}
        if(row.size()!=h) throw std::runtime_error("embedding row width mismatch");
        for(uint64_t stream=0;stream<uint64_t(g.hc);++stream) std::copy(row.begin(),row.end(),input+r*n+stream*h);
        if(p.dflash_host_profile) {const auto now=Clock::now();p.dflash_host_ms[2]+=ms(host_mark,now);}
    }
    // One reader request for the window's rows instead of one per row (the prefill window's law).
    if(p.ple) {
        auto host_mark=p.dflash_host_profile?Clock::now():Clock::time_point{};
        p.ple->gather_rows(ids,previous,ple_rows.data(),h);
        if(p.dflash_host_profile) p.dflash_host_ms[1]+=ms(host_mark,Clock::now());
    }
    std::vector<int32_t> argmax(rows);
    const auto t2=Clock::now();const uint64_t first_launch=recipe_launch_count();
    try {
        upload(v.stream,v.in(),v.staging,uint64_t(rows)*n*4);
        auto mark=Clock::now();
        const auto lap=[&](size_t kind) {
            if(!p.dflash_sync_timing) return;
            v.stream.synchronize();const auto now=Clock::now();p.dflash_kind_ms[kind]+=ms(mark,now);mark=now;
        };
        for(uint32_t layer=0;layer<p.layers.size();++layer) {
            p.layers[layer]->verify_chain(v,layer==1?std::span<const float>(ple_rows):std::span<const float>{});
            lap(core::is_qsa_layer(g,layer)?1:0);
            if(const int slot=drafter.feature_slot(layer);slot>=0) {drafter.capture(v.stream,v.in(),uint32_t(g.hc),slot,rows);lap(2);}
        }
        auto& head=p.verify_head_for(rows);
        head.bind_external({"final.residual",&v.in()});
        // The stage and residual rows are read only after the head (selection and commit), so their readback
        // rides the copy stream alongside it instead of holding the head back by the transfer time.
        v.copy_ready.record(v.stream);v.copy.wait(v.copy_ready);
        // Only what the commit reads: each QSA layer's rows of K, V and index keys at the head of its stage slice
        // (the whole stage is sized for a prefill window: ~450 MB a round at 2048 rows).
        {
            const uint64_t kv=uint64_t(g.n_head_kv)*g.head_dim*4,index=uint64_t(g.idx_key_dim)*4;
            const uint64_t used=uint64_t(rows)*(2*kv+index);
            for(uint32_t layer=0;layer<p.layers.size();++layer)
                if(core::is_qsa_layer(g,layer)) {const uint64_t at=verify_stage_offset(g,layer,v.max_rows);download(v.copy,v.stage_host,v.stage,used,at,at);}
        }
        download(v.copy,v.staging,v.in(),uint64_t(rows)*n*4);
        v.copy_done.record(v.copy);
        head.launch(v.stream);head.wait();lap(3);
        v.copy_done.synchronize();
        const auto best=head.values("head.best");
        for(uint32_t r=0;r<rows;++r) argmax[r]=int32_t(best.at(r));
    } catch(...) {
        // Nothing was accepted: the layers only wrote scratch, staged rows and cache cells past the committed prefix.
        try {v.stream.synchronize();} catch(...) {}
        try {v.copy.synchronize();} catch(...) {}
        throw;
    }
    for(const auto id:argmax) if(id<0 || uint64_t(id)>=vocabulary) throw NonFiniteOutput("DFlash window argmax outside the vocabulary");
    // Rows of logits only when the selection (or the commit) needs them.
    std::vector<uint8_t> fetched(rows,0);
    const auto* host_logits=static_cast<const float*>(v.logits_host.data());
    const auto row=[&](uint32_t r) {
        if(r>=rows) throw std::out_of_range("DFlash window row");
        if(!fetched[r]) {
            download(v.stream,v.logits_host,v.logits,vocabulary*4,uint64_t(r)*vocabulary*4,uint64_t(r)*vocabulary*4);v.stream.synchronize();
            if(!all_finite(std::span<const float>(host_logits+uint64_t(r)*vocabulary,vocabulary))) throw NonFiniteOutput("non-finite DFlash window logits");
            fetched[r]=1;
        }
        return std::span<const float>(host_logits+uint64_t(r)*vocabulary,vocabulary);
    };
    // Everything the commit reads comes down now: then the drafter's work for the next round can be queued at once
    // and run while the host selects, commits and the caller emits tokens (the commit itself never touches the stream).
    // The plain-greedy prediction of the kept rows: the argmax after each row matches the next drafted token.
    uint32_t predicted=1;
    while(predicted<rows && argmax[predicted-1]==ids[predicted] &&
          std::find(p.stops.begin(),p.stops.end(),ids[predicted-1])==p.stops.end()) ++predicted;
    predicted=std::min(predicted,maximum);
    if(p.ple) download(v.stream,v.ple_history_host,v.ple_history,uint64_t(rows)*9*n*4);
    (void)row(predicted-1); // downloads and synchronizes: the stage, histories and residual rows are on the host too
    const uint64_t next=position+predicted;
    const bool early=synced && next<context;
    if(early) {
        drafter.append(v.stream,uint32_t(position),0,predicted,false);
        const int32_t guess=argmax[predicted-1];
        drafter.begin_draft(v.stream,p.embedding_row(guess,next),p.mask_embedding,guess,uint32_t(next));p.dflash_prepared={next,guess,true};
    }
    const auto t3=Clock::now();
    if(p.dflash_mark) {
        std::string id_list,argmax_list;
        for(const auto id:ids) {id_list+=std::to_string(id);id_list+=',';}
        for(const auto id:argmax) {argmax_list+=std::to_string(id);argmax_list+=',';}
        std::fprintf(stderr,"strata-hpu: dflash window %llu at %llu enqueues %llu-%llu rows %u predicted=%u ids=%s argmax=%s cells=%u\n",
                     (unsigned long long)p.dflash_rounds,(unsigned long long)position,(unsigned long long)first_launch,
                     (unsigned long long)recipe_launch_count()-1,rows,predicted,id_list.c_str(),argmax_list.c_str(),drafter.cells());
    }
    const auto selected=select(DflashRows{ids,argmax,row});
    const auto t4=Clock::now();
    if(!selected || selected>rows) throw std::invalid_argument("DFlash selection outside the window");
    const uint32_t accepted=std::min(selected,maximum);
    if(p.dflash_mark) std::fprintf(stderr,"strata-hpu: dflash window %llu selected=%u accepted=%u synced=%d early=%d\n",
                                    (unsigned long long)p.dflash_rounds,selected,accepted,int(synced),int(early));
    try {
        if(early && accepted!=predicted) {
            // The selection kept another prefix (bias, penalties or a tie): the queued draft is for the wrong token.
            p.settle_draft();drafter.truncate(uint32_t(position+std::min(accepted,predicted)));
        }
        const auto last_row=row(accepted-1);
        for(size_t index=0;index<p.layers.size();++index) {
            const auto before=p.dflash_host_profile?Clock::now():Clock::time_point{};
            p.layers[index]->verify_commit(v,accepted);
            if(p.dflash_host_profile) p.dflash_host_ms[core::is_qsa_layer(g,uint32_t(index))?4:3]+=ms(before,Clock::now());
        }
        p.dflash_ms[5]+=ms(t4,Clock::now());
        // The diagnostic records what the layers were actually left holding: one row per committed token, in commit
        // order. row(k) downloads into the chain's one shared host buffer, so each row is written before the next
        // download overwrites it.
        p.flow->history_.insert(p.flow->history_.end(),ids.begin(),ids.begin()+accepted);
        if(synced) {
            if(!early) drafter.append(v.stream,uint32_t(position),0,accepted);
            else if(accepted>predicted) drafter.append(v.stream,uint32_t(position+predicted),predicted,accepted-predicted);
            p.drafter_tokens.insert(p.drafter_tokens.end(),ids.begin(),ids.begin()+accepted);
        } else {
            if(early) {p.settle_draft();drafter.truncate(uint32_t(position));}
            p.drafter_tokens.clear();
        }
        std::copy(last_row.begin(),last_row.end(),p.last_logits.begin());
        // Last, not earlier: row(k) downloads into the chain's ONE host logits buffer, so writing the sink before
        // this point would overwrite the `last_row` span the commit still reads and poison p.last_logits.
        if(p.dflash_logits) for(uint32_t k=0;k<accepted;++k) p.dflash_logits->record(position+k,k,uint32_t(vocabulary),row(k).data());
    } catch(const std::exception& error) {p.poisoned=true;throw SessionFatal(std::string("DFlash window commit failed: ")+error.what());}
    const auto* last=p.last_logits.data();
    const auto* residual=static_cast<const float*>(v.staging.data())+uint64_t(accepted-1)*n;
    std::copy(residual,residual+n,p.last_residual.begin());
    ++p.dflash_rounds;p.dflash_committed+=accepted;p.dflash_proposed+=rows-1;
    p.dflash_recent=0.75*p.dflash_recent+0.25*double(accepted);
    if(p.dflash_recent<p.dflash_break_even) {p.dflash_skip=p.dflash_backoff;p.dflash_recent=p.dflash_break_even+0.5;}
    const auto t5=Clock::now();
    const double spans[]={ms(t0,t1),ms(t1,t2),ms(t2,t3),ms(t3,t4),ms(t4,t5)};
    for(size_t k=0;k<5;++k) p.dflash_ms[k]+=spans[k];
    if(p.dflash_rounds%32==0) {
        const double r=double(p.dflash_rounds);
        std::fprintf(stderr,"strata-hpu: dflash verify rounds=%llu skipped=%llu tokens_per_round=%.3f drafted=%llu accepted=%llu ms/round draft=%.2f inputs=%.2f window=%.2f select=%.2f commit=%.2f (layers %.2f)\n",
                     (unsigned long long)p.dflash_rounds,(unsigned long long)p.dflash_skipped,double(p.dflash_committed)/r,(unsigned long long)p.dflash_proposed,(unsigned long long)(p.dflash_committed-p.dflash_rounds),
                     p.dflash_ms[0]/r,p.dflash_ms[1]/r,p.dflash_ms[2]/r,p.dflash_ms[3]/r,p.dflash_ms[4]/r,p.dflash_ms[5]/r);
        if(p.dflash_sync_timing) std::fprintf(stderr,"strata-hpu: dflash window ms/round gdn=%.2f qsa=%.2f capture=%.2f head=%.2f\n",
                                              p.dflash_kind_ms[0]/r,p.dflash_kind_ms[1]/r,p.dflash_kind_ms[2]/r,p.dflash_kind_ms[3]/r);
        if(p.dflash_host_profile) std::fprintf(stderr,"strata-hpu: dflash host ms/round embedding=%.3f ple=%.3f scatter=%.3f commit_gdn=%.3f commit_qsa=%.3f\n",
                                              p.dflash_host_ms[0]/r,p.dflash_host_ms[1]/r,p.dflash_host_ms[2]/r,p.dflash_host_ms[3]/r,p.dflash_host_ms[4]/r);
    }
    p.after_round();
    DflashRound round;round.tokens.assign(ids.begin(),ids.begin()+accepted);round.logits.assign(last,last+vocabulary);round.proposed=rows-1;
    return round;
}
VerifiedCandidates Model::verify_candidates(const DraftContext& context,std::span<const DraftRow> drafts,const std::function<bool()>& cancelled) {
    auto& p=*impl_;std::lock_guard lock(p.mutex);const auto tokens=p.flow->tokens();
    if(p.poisoned || p.flow->poisoned() || (p.mtp && !p.mtp->healthy())) throw SessionFatal("candidate target/MTP state failed");
    if(!p.memory_plan.verify_rows || drafts.empty() || drafts.size()>p.memory_plan.verify_rows || tokens.empty() ||
       drafts.size()>p.options.memory.context-tokens.size() ||
       context.follow.size()>drafts.size()+1 || drafts.size()>UINT64_MAX-context.sampling.counter ||
       (context.mode==VerifyMode::Probabilistic && context.sampling.counter+drafts.size()>=(1ull<<61)) ||
       (context.mode!=VerifyMode::Exact && context.mode!=VerifyMode::Probabilistic) ||
       (!context.bias.empty() && context.bias.size()!=p.last_logits.size()) ||
       (context.routing && !p.routing_capture.enabled()) ||
       !std::equal(tokens.begin(),tokens.end(),context.history.begin(),context.history.end()) ||
       !std::equal(p.last_logits.begin(),p.last_logits.end(),context.logits.begin(),context.logits.end()))
        throw std::invalid_argument("native candidate verification context/window mismatch");
    validate_drafts(drafts,uint32_t(p.last_logits.size()),p.memory_plan.verify_rows);
    for(auto token:context.follow) if(token<0 || size_t(token)>=p.last_logits.size()) throw std::invalid_argument("native candidate follow outside vocabulary");
    const std::array<float,1> dummy{0};(void)sampling_distribution(dummy,{},context.sampling);
    for(float value:context.bias) if(std::isnan(value) || value==INFINITY) throw std::invalid_argument("invalid candidate bias");
    if(!context.bias.empty() && std::all_of(context.bias.begin(),context.bias.end(),[](float value){return value==-INFINITY;}))
        throw std::invalid_argument("all candidate tokens are banned");
    std::vector<int32_t> ids;ids.reserve(drafts.size());for(const auto& row:drafts) ids.push_back(row.token);
    VerifiedCandidates selected;
    struct WindowGuard {EmbeddingWindow& window;~WindowGuard(){window.close();}} guard{p.embedding_window};
    p.embedding_window.open(tokens.size(),ids);
    auto trace=p.routing_capture.round(uint32_t(ids.size()));
    auto halves=p.half_capture.round(false);
    auto committed=p.flow->verify_window(ids,p.memory_plan.verify_logits_bytes,[&](std::span<const float> after) {
        selected=select_verified_candidates(p.last_logits,after,drafts,tokens,context.sampling,context.bias,context.mode,p.stops,cancelled,context.follow);
        if(context.routing) context.routing(p.routing_capture.physical_view(),true);
        return selected.retained;
    },cancelled);
    if(committed.accepted) p.commit_output(committed.logits);
    trace.finish();halves.finish();
    selected.logits=committed.accepted?std::move(committed.logits):p.last_logits;
    return selected;
}
std::optional<SlotCandidates> Model::slot_window(int32_t pending,std::span<const int32_t> history,
                                               const strata::kernels::SamplerParams& sampling,const RoutingWindowSink& routing) {
    auto& p=*impl_;std::lock_guard lock(p.mutex);const auto tokens=p.flow->tokens();
    if(p.poisoned || p.flow->poisoned() || (p.mtp && !p.mtp->healthy())) throw SessionFatal("slot target/MTP state failed");
    if(!p.mtp || p.memory_plan.verify_rows<2) throw std::logic_error("slot MTP requires drafter and two native rows");
    if(routing && !p.routing_capture.enabled()) throw std::invalid_argument("slot routing callback requires capture");
    if(tokens.empty() || tokens.size()+3>p.options.memory.context || pending<0 || size_t(pending)>=p.last_logits.size() ||
       std::find(p.stops.begin(),p.stops.end(),pending)!=p.stops.end() || sampling.counter==UINT64_MAX ||
       !std::equal(tokens.begin(),tokens.end(),history.begin(),history.end()))
        throw std::invalid_argument("slot MTP context/window mismatch");
    if(p.mtp->size()!=tokens.size()-1) throw SessionFatal("slot MTP prefix is not aligned");
    const std::array<float,1> dummy{0};(void)sampling_distribution(dummy,{},sampling);
    // Pair the KNOWN pending token with the previous target residual. The
    // ordinary draft() predicts pending itself and would be one position late
    // if used here. This temporary step must rewind before target verification.
    const auto point=p.mtp->checkpoint();SlotCandidates result;
    auto rewind=[&] {
        try {p.mtp->restore(point);}
        catch(const std::exception& error) {p.poisoned=true;throw SessionFatal(std::string("slot MTP proposal rollback failed: ")+error.what());}
    };
    try {
        auto next=p.mtp->step(p.last_residual,p.embedding_row(pending,tokens.size()));
        std::vector<int32_t> fed(tokens.begin(),tokens.end());fed.push_back(pending);
        // Upstream batch-MTP accepts exact sampled-token matches even when
        // solo --spec-prob is enabled. Do not change that target token stream.
        auto proposed=pick_mtp_draft(next.logits,fed,sampling,{},VerifyMode::Exact,p.options.mtp_sampling);
        result.draft=std::move(proposed.row);
    } catch(...) {rewind();throw;}
    rewind();
    if(result.draft.token<0) return std::nullopt;
    validate_drafts(std::span(&result.draft,1),uint32_t(p.last_logits.size()),1);
    const std::vector<int32_t> ids{pending,result.draft.token};
    struct WindowGuard {EmbeddingWindow& window;~WindowGuard(){window.close();}} guard{p.embedding_window};
    p.embedding_window.open(tokens.size(),ids);
    auto trace=p.routing_capture.round(uint32_t(ids.size()));
    auto halves=p.half_capture.round(false);
    auto committed=p.flow->verify_window(ids,p.memory_plan.verify_logits_bytes,[&](std::span<const float> after) {
        result.selected=select_slot_candidates(after,result.draft,pending,tokens,sampling,p.stops);
        if(routing) routing(p.routing_capture.physical_view(),true);
        return 1+result.selected.retained; // pending was already emitted
    });
    p.commit_output(committed.logits);
    trace.finish();halves.finish();
    result.selected.logits=std::move(committed.logits);
    return result;
}
bool Model::healthy() const {
    const auto& p=*impl_;std::lock_guard lock(p.mutex);if(p.poisoned || p.flow->poisoned()) return false;
    for(const auto& layer:p.layers) if(!layer->healthy()) return false;
    if(p.mtp && !p.mtp->healthy()) return false;
    return true;
}
const ModelPlan& Model::plan() const {return impl_->memory_plan;}
uint32_t Model::vocabulary() const {return uint32_t(impl_->table.geometry().n_vocab);}
std::vector<int32_t> Model::stop_tokens() const {return impl_->stops;}
Model::Snapshot Model::snapshot() const {
    auto& p=*impl_;std::lock_guard lock(p.mutex);
    if(p.poisoned || p.flow->poisoned() || !p.options.memory.snapshot_budget) throw std::logic_error("snapshot disabled or model state is failed");
    const auto tokens=p.flow->tokens();
    const auto charge=p.parked_bytes(tokens.size());
    p.parked_reused=0;
    std::shared_ptr<Snapshot::Data> data;
    // How much of the retained image is still byte-identical: upstream caps the same quantity with
    // limit_reuse(first_dirty) (conversation_cache.hpp:227-231), where `first_dirty` is the earliest position
    // the outgoing conversation no longer holds. Comparing the parked ids with the live ones measures exactly
    // that, and the per-layer ancestry check in QsaState rejects a prefix whose rows were rewritten anyway.
    uint64_t unchanged=0;
    if(p.parked && p.parked.use_count()==1) {
        const auto shared=std::min(p.parked_tokens.size(),tokens.size());
        while(unchanged<shared && uint64_t(p.parked_tokens[size_t(unchanged)])==uint64_t(tokens[unchanged])) ++unchanged;
    }
    if(unchanged) {
        // Upstream's reuse branch: the image moves out of the slot (std::move(reuse) into the fresh
        // SavedConversation), every layer grows it in place past the last whole page, and the counter collects
        // the bytes that were NOT re-read from the pools. A refusal here propagates with the image dropped, the
        // way upstream skips parking rather than parking a half-trusted copy.
        auto retained=std::static_pointer_cast<Snapshot::Data>(std::move(p.parked));
        p.parked_tokens.clear();
        if(retained->owner!=p.owner || retained->layers.size()!=p.layers.size() ||
           retained->logits.size()!=(retained->tokens.empty()?0:p.last_logits.size()))
            throw std::logic_error("retained parked image is foreign or misshapen");
        retained->allocation.prepare(charge,p.snapshot_overlap_bytes(retained->tokens.size()));
        uint64_t reused=0;
        for(size_t i=0;i<p.layers.size();++i) reused+=p.layers[i]->snapshot_reused(retained->layers[i],unchanged);
        snapshot_assign(retained->tokens,std::span<const int32_t>(tokens));
        retained->control_enabled=p.control_enabled;
        p.mixed_output.capture(retained->mixed,!tokens.empty());
        if(!tokens.empty()) {snapshot_assign(retained->logits,std::span<const float>(p.last_logits));snapshot_assign(retained->residual,std::span<const float>(p.last_residual));}
        if(p.mtp) {
            if(p.mtp->size()!=(tokens.empty()?0:tokens.size()-1)) throw std::logic_error("MTP/target sequence position mismatch");
            // The draft pool joins the reuse - with upstream's cap applied inside the call, so the page it is
            // still writing is always re-read (src/core/conversation_state.cpp:291-294).
            const uint64_t draft_reused=p.mtp->snapshot_reused(std::move(retained->mtp),unchanged,retained->mtp);
            reused+=draft_reused;
            retained->has_mtp=true;
            if(tokens.size()>1) {snapshot_assign(retained->mtp_logits,std::span<const float>(p.last_mtp_logits));snapshot_assign(retained->mtp_residual,std::span<const float>(p.last_mtp_residual));}
        }
        retained->validate_capacity(charge);retained->allocation.commit(charge);
        p.parked_reused=reused;p.parked_reused_total+=reused;data=std::move(retained);
    }
    if(!data) {
        data=std::make_shared<Snapshot::Data>(p.owner,p.snapshot_budget,charge);
        data->control_enabled=p.control_enabled;
        p.mixed_output.capture(data->mixed,!tokens.empty());
        data->tokens.assign(tokens.begin(),tokens.end());data->layers.reserve(p.layers.size());
        if(!tokens.empty()) {data->logits=p.last_logits;data->residual=p.last_residual;}
        for(const auto& layer:p.layers) {
            if(layer->size()!=tokens.size()) throw std::logic_error("layer/session position mismatch");
            data->layers.push_back(layer->snapshot());
        }
        if(p.mtp) {
            if(p.mtp->size()!=(tokens.empty()?0:tokens.size()-1)) throw std::logic_error("MTP/target sequence position mismatch");
            data->has_mtp=true;data->mtp=p.mtp->snapshot();
            if(tokens.size()>1) {data->mtp_logits=p.last_mtp_logits;data->mtp_residual=p.last_mtp_residual;}
        }
    }
    // Upstream feeds the reuse slot at RESTORE (retain(), conversation_cache.hpp:217-226) and the park CONSUMES
    // it (take_reuse(), :232, moved into the fresh image at :8350). So a park does not re-arm the slot: the next
    // park in the same request copies wholesale, exactly like upstream's second park_current() call.
    data->validate_capacity(charge);
    Snapshot result;result.data_=std::move(data);return result;
}
uint64_t Model::parked_reused_bytes() const {return impl_->parked_reused;}
uint64_t Model::parked_reused_total() const {return impl_->parked_reused_total;}
uint64_t Model::parked_retained_bytes() const {
    auto& p=*impl_;std::lock_guard lock(p.mutex);
    // Cache entries already include their shared payload. Only an exclusively
    // retained image adds a separate amount after take() vacates that entry.
    return exclusive_snapshot_bytes(p.parked,p.parked?static_cast<Snapshot::Data*>(p.parked.get())->allocation.bytes():0);
}
void Model::discard_parked() {
    auto& p=*impl_;std::lock_guard lock(p.mutex);
    p.parked.reset();p.parked_tokens.clear();p.parked_reused=0;
}
uint64_t Model::snapshot_size_limit() const {return impl_->snapshot_bytes();}
uint64_t Model::snapshot_memory_cost() const {auto& p=*impl_;std::lock_guard lock(p.mutex);return p.parked_bytes(p.flow->tokens().size());}
uint64_t Model::parked_bytes(uint64_t cells) const {auto& p=*impl_;std::lock_guard lock(p.mutex);return p.parked_bytes(cells);}
void Model::restore(const Snapshot& snapshot) {
    auto& p=*impl_;std::lock_guard lock(p.mutex);const auto& data=snapshot.data_;
    if(p.poisoned || !data || data->owner!=p.owner || data->layers.size()!=p.layers.size() || data->tokens.size()>p.options.memory.context ||
       data->logits.size()!=(data->tokens.empty()?0:p.last_logits.size()) || data->residual.size()!=(data->tokens.empty()?0:p.last_residual.size()))
        throw std::invalid_argument("foreign/empty snapshot or model state is failed");
    if(data->has_mtp!=bool(p.mtp) || data->mtp_logits.size()!=(p.mtp && data->tokens.size()>1?p.last_mtp_logits.size():0) ||
       data->mtp_residual.size()!=(p.mtp && data->tokens.size()>1?p.last_mtp_residual.size():0)) throw std::invalid_argument("snapshot MTP configuration/shape mismatch");
    if(!p.control_vectors && !data->control_enabled) throw std::invalid_argument("snapshot control vector configuration mismatch");
    HeadMixed::validate(data->mixed,p.mixed_output.width(),!data->tokens.empty());
    p.routing_capture.clear();p.half_capture.clear();
    p.poisoned=true;
    for(size_t layer=0;layer<p.layers.size();++layer) p.layers[layer]->restore(data->layers[layer]);
    if(p.mtp) p.mtp->restore(data->mtp);
    p.flow->restore_history(data->tokens);
    p.control_enabled=data->control_enabled;
    p.mixed_output.restore(data->mixed,!data->tokens.empty());
    if(!data->tokens.empty()) {
        std::copy(data->logits.begin(),data->logits.end(),p.last_logits.begin());
        std::copy(data->residual.begin(),data->residual.end(),p.last_residual.begin());
    }
    if(p.mtp && data->tokens.size()>1) {
        std::copy(data->mtp_logits.begin(),data->mtp_logits.end(),p.last_mtp_logits.begin());
        std::copy(data->mtp_residual.begin(),data->mtp_residual.end(),p.last_mtp_residual.begin());
    }
    p.poisoned=false;
    // Upstream's retain(): the K/V buffers this restore just wrote into the live pools are exactly the image
    // the next park of THIS conversation grows from (conversation_cache.hpp:217-226). The parked entry keeps its
    // own reference, so the grow-in-place path waits until the cache drops that copy as superseded.
    p.parked=std::const_pointer_cast<Snapshot::Data>(snapshot.data_);p.parked_tokens=data->tokens;p.parked_reused=0;
}
void Model::save(const std::string& path,std::span<const RequestImage> images) const {
    const auto& p=*impl_;
    check_session_memory(snapshot_memory_cost(),p.options.session_min_free,p.options.session_memory_probe);
    const auto point=snapshot();
    check_session_memory(0,p.options.session_min_free,p.options.session_memory_probe);
    std::lock_guard lock(p.mutex);
    if(p.file_identity()!=p.weight_identity) throw std::runtime_error("model files changed since load");
    StateWriter writer(path,p.snapshot_bytes()+state_image_bytes(p.options.memory.context));
    writer.text("strata-hpu-296b59d-state-v38");writer.text(p.weight_identity);
    const auto configuration=p.state_configuration();writer.integer(configuration.size());for(auto value:configuration) writer.integer(value);
    writer.integer(point.data_->control_enabled);
    writer.integer(point.data_->tokens.size());writer.vector<int32_t>(point.data_->tokens);
    write_image_identity(writer,images,point.data_->tokens.size());
    writer.vector<float>(point.data_->logits);
    writer.vector<float>(point.data_->residual);
    HeadMixed::write(writer,point.data_->mixed,p.mixed_output.width(),!point.data_->tokens.empty());
    for(size_t layer=0;layer<p.layers.size();++layer) p.layers[layer]->write_snapshot(writer,point.data_->layers[layer]);
    writer.integer(bool(p.mtp));
    if(p.mtp) {writer.vector<float>(point.data_->mtp_logits);writer.vector<float>(point.data_->mtp_residual);p.mtp->write_snapshot(writer,point.data_->mtp);}
    writer.finish();
}
std::vector<RequestImage> Model::load(const std::string& path) {
    Snapshot point;std::vector<RequestImage> images;
    {
        auto& p=*impl_;std::lock_guard lock(p.mutex);
        if(p.poisoned || !p.options.memory.snapshot_budget) throw std::logic_error("model state is failed or snapshots are disabled");
        if(p.file_identity()!=p.weight_identity) throw std::runtime_error("model files changed since load");
        StateReader reader(path,p.snapshot_bytes()+state_image_bytes(p.options.memory.context));
        if(reader.text()!="strata-hpu-296b59d-state-v38" || reader.text()!=p.weight_identity) throw std::invalid_argument("snapshot belongs to different model files/backend");
        const auto configuration=p.state_configuration();if(reader.integer()!=configuration.size()) throw std::invalid_argument("snapshot configuration shape mismatch");
        for(auto value:configuration) if(reader.integer()!=value) throw std::invalid_argument("snapshot geometry/RoPE/expert policy mismatch");
        const auto control_enabled=reader.integer();if(control_enabled>1 || (!p.control_vectors && !control_enabled)) throw std::invalid_argument("invalid serialized control vector mode");
        const auto position=reader.integer();if(position>p.options.memory.context) throw std::invalid_argument("snapshot position exceeds context");
        const auto charge=p.parked_bytes(position);
        check_session_memory(charge,p.options.session_min_free,p.options.session_memory_probe);
        auto data=std::make_shared<Snapshot::Data>(p.owner,p.snapshot_budget,charge);
        data->control_enabled=control_enabled!=0;
        data->tokens=reader.vector<int32_t>(position);images=read_image_identity(reader,position);
        data->logits=reader.vector<float>(position?p.last_logits.size():0);
        data->residual=reader.vector<float>(position?p.last_residual.size():0);
        data->mixed=HeadMixed::read(reader,p.mixed_output.width(),position!=0);
        for(auto token:data->tokens) if(token<0 || uint64_t(token)>=uint64_t(p.table.geometry().n_vocab)) throw std::invalid_argument("invalid snapshot token ID");
        for(float value:data->logits) if(!std::isfinite(value)) throw std::invalid_argument("non-finite snapshot logits");
        for(float value:data->residual) if(!std::isfinite(value)) throw std::invalid_argument("non-finite snapshot residual");
        data->layers.reserve(p.layers.size());for(const auto& layer:p.layers) data->layers.push_back(layer->read_snapshot(reader,position));
        if(reader.integer()!=bool(p.mtp)) throw std::invalid_argument("serialized MTP configuration mismatch");
        data->has_mtp=bool(p.mtp);
        if(p.mtp) {
            data->mtp_logits=reader.vector<float>(position>1?p.last_mtp_logits.size():0);data->mtp_residual=reader.vector<float>(position>1?p.last_mtp_residual.size():0);
            for(float x:data->mtp_logits) if(std::isnan(x) || x==INFINITY) throw std::invalid_argument("invalid serialized MTP logits");
            for(float x:data->mtp_residual) if(!std::isfinite(x)) throw std::invalid_argument("non-finite serialized MTP residual");
            if(position>1 && std::none_of(data->mtp_logits.begin(),data->mtp_logits.end(),[](float x){return std::isfinite(x);})) throw std::invalid_argument("empty serialized MTP head support");
            data->mtp=p.mtp->read_snapshot(reader,uint32_t(position?position-1:0));
        }
        reader.finish();data->validate_capacity(charge);point.data_=std::move(data);
        check_session_memory(0,p.options.session_min_free,p.options.session_memory_probe);
    }
    // No device state changes occur until the complete file, its picture list and checksum pass.
    restore(point);
    return images;
}
AllocationStats Model::hbm_allocations() const {return impl_->device->hbm_allocations();}
AllocationStats Model::pinned_allocations() const {return impl_->device->pinned_allocations();}
AllocationStats Model::snapshot_allocations() const {return impl_->snapshot_budget->stats();}
CacheStats Model::expert_cache_stats() const {return impl_->cache->stats();}
bool Model::ple_ram_locked() const {std::lock_guard lock(impl_->mutex);return impl_->ple && impl_->ple->ram_locked();}
// ---- upstream's expert-cache profiling/adaptation surface (strata/hpu/expert_profile.hpp) -------------
uint64_t Model::adapt_experts() {
    auto& p=*impl_;std::lock_guard lock(p.mutex);
    return p.adapt_locked();
}
bool Model::expert_profile_save_due() const {
    const auto& p=*impl_;const auto* profiler=p.cache->expert_profiler();
    if(!profiler) return false;
    std::lock_guard lock(p.mutex);
    return profiler->save_due(ExpertProfiler::Clock::now());
}
bool Model::save_expert_profile(std::string& error) {
    auto& p=*impl_;std::lock_guard lock(p.mutex);
    auto* profiler=p.cache->expert_profiler();
    if(!profiler) {error="the expert profile tier is off: nothing was counted";return false;}
    return profiler->save_profile(ExpertProfiler::Clock::now(),error);
}
const ExpertProfileOptions& Model::expert_profile_options() const {return impl_->cache->expert_profile_options();}
const ExpertResidencyOptions& Model::expert_residency_options() const {return impl_->cache->expert_residency_options();}
ExpertCache::ResidencyReport Model::residency_report() const {return impl_->cache->residency_report();}
std::string Model::reserve_vram(std::optional<uint64_t> reserve) {
    auto& p=*impl_;std::lock_guard lock(p.mutex);const auto required=reserve.value_or(p.options.memory.driver_margin);
    const auto before=p.device->info();const auto allocated=p.device->hbm_allocations();
    const auto current=p.cache->capacity(),resident=p.cache->stats().resident_bytes;
    uint64_t target=resident;
    if(before.free_bytes>=required) {
        const auto free=std::min(before.free_bytes-required,allocated.limit-allocated.used);
        target=free>UINT64_MAX-resident?UINT64_MAX:resident+free;
    } else {
        const auto missing=required-before.free_bytes;
        if(missing>resident) throw std::invalid_argument("requested reserve exceeds reclaimable expert memory");
        target=resident-missing;
    }
    const auto bytes=checked_product(checked_product(uint64_t(p.table.geometry().n_embd),p.table.geometry().n_ff),6);
    target=std::min(target,p.options.memory.expert_cache);
    if(target<bytes) throw std::invalid_argument("requested reserve leaves no complete expert slot");
    p.cache->resize(target);const auto after=p.device->info();
    if(after.free_bytes<required) {p.cache->resize(current);throw std::runtime_error("driver free memory did not meet requested reserve after eviction");}
    return "expert_slots="+std::to_string(target/bytes)+" expert_cache_mib="+std::to_string(target/(1ull<<20))+" vram_free_mib="+std::to_string(after.free_bytes/(1ull<<20))+" reserve_mib="+std::to_string(required/(1ull<<20));
}
}
