#include "strata/hpu/expert_cache.hpp"
#include "strata/hpu/expert_scales.hpp"
#include "strata/hpu/native_pack.hpp"
#include "strata/hpu/fp8.hpp"
#include "strata/hpu/fp8_colpair.hpp"
#include "strata/hpu/performance_modes.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <exception>
#include <thread>

namespace strata::hpu {
struct ExpertCache::Impl {
    Device device;core::ModelGeometry geometry;ExpertResolver resolve;
    Stream copy;ResidentCache<ExpertWeights> cache;uint64_t pinned,expert_bytes;
    std::mutex loading; // resolve at most one bounded expert slice on a cold miss
    // Upstream's --expert-profile / --expert-profile-save / --adapt-* knobs (strata/hpu/expert_profile.hpp).
    // `profiler` stays null unless the options ask for something, and every hook below is one null check.
    ExpertProfileOptions expert_profile;std::unique_ptr<ExpertProfiler> profiler;
    // Upstream's residency knobs (strata/hpu/expert_residency.hpp): the parsed options, this layer's own stored
    // expert blob size (upstream lay.blob_bytes(l), which is what the per-layer split sizes against), the
    // report of what was installed, and whether the per-layer partition is actually in force.
    ExpertResidencyOptions residency;std::vector<uint64_t> layer_blob_bytes;ResidencyReport report;bool partitioned=false;
    bool fp8=false; // --fp8-experts
    // One role as FP8 codes followed by its F32 row scales, multiplied by the role's global output scale.
    // The dequantize + encode runs on row ranges in parallel, straight into the pinned staging buffer
    // (measured single-threaded, it was the whole cost of a cold FP8 miss).
    void encode_fp8(const WeightSlice& source,float global,uint8_t* codes,float* scales) {
        const uint64_t rows=source.rows(),columns=source.columns();
        const uint64_t workers=std::clamp<uint64_t>(std::thread::hardware_concurrency()/2,1,16),step=(rows+workers-1)/workers;
        std::vector<std::thread> threads;std::vector<std::exception_ptr> errors(workers);
        for(uint64_t w=0;w<workers;++w) threads.emplace_back([&,w] {
            try {
                const uint64_t first=w*step,last=std::min(rows,first+step);if(first>=last) return;
                std::vector<float> values((last-first)*columns);source.convert_rows_f32(first,values);
                if(global!=1) for(auto& value:values) value*=global;
                for(uint64_t r=first;r<last;++r)
                    scales[r]=quantize_fp8_row(std::span<const float>(values).subspan((r-first)*columns,columns),codes+r*columns);
            } catch(...) {errors[w]=std::current_exception();}
        });
        for(auto& thread:threads) thread.join();
        for(auto& error:errors) if(error) std::rethrow_exception(error);
    }
    DeviceBuffer upload_fp8(const WeightSlice& source,float global) {
        const uint64_t rows=source.rows(),columns=source.columns(),bytes=fp8_role_bytes(rows,columns);
        HostBuffer staging(device,bytes);auto* codes=static_cast<uint8_t*>(staging.data());
        auto* scales=reinterpret_cast<float*>(codes+rows*columns);
        encode_fp8(source,global,codes,scales);
        if(tpc_colpair_enabled()) pack_fp8_colpair({codes,size_t(rows*columns)},rows,columns);
        DeviceBuffer output(device,bytes);
        try {upload(copy,output,staging,bytes);copy.synchronize();} catch(...) {device.synchronize();throw;}
        return output;
    }
    DeviceBuffer upload_fp8_pair(const ExpertSlices& slices) {
        const uint64_t rows=slices.gate.rows(),columns=slices.gate.columns(),gu=rows*columns;
        const auto bytes=2*fp8_role_bytes(rows,columns);
        HostBuffer staging(device,bytes);auto* codes=static_cast<uint8_t*>(staging.data());
        auto* scales=reinterpret_cast<float*>(codes+2*gu);
        encode_fp8(slices.gate,slices.output_scales[0],codes,scales);
        encode_fp8(slices.up,slices.output_scales[1],codes+gu,scales+rows);
        if(tpc_colpair_enabled()) pack_fp8_colpair({codes,size_t(2*gu)},2*rows,columns);
        DeviceBuffer output(device,bytes);
        try {upload(copy,output,staging,bytes);copy.synchronize();} catch(...) {device.synchronize();throw;}
        return output;
    }
    Impl(const Device& d,core::ModelGeometry g,ExpertResolver r,uint64_t budget,uint64_t staging)
        :device(d),geometry(g),resolve(std::move(r)),copy(d),cache(budget),pinned(staging),expert_bytes(checked_product(checked_product(g.n_embd,g.n_ff),6)) {
        // Geometry-only construction (the MTP cache) has no table to read formats from, so the split sizes
        // against the BF16 blob this cache already charges; the table constructor overwrites it below.
        layer_blob_bytes.assign((size_t) g.n_layers,expert_bytes);
    }
    // The slot size the admission arithmetic counts: upstream's lay.max_blob (generate.cpp:5254's `asked*max_blob`).
    uint64_t slot_bytes() const {
        uint64_t biggest=0;for(const uint64_t b:layer_blob_bytes) biggest=std::max(biggest,round256(b));
        return biggest?biggest:expert_bytes;
    }
    int64_t slots() const {const auto slot=slot_bytes();return slot?int64_t(cache.capacity()/slot):0;}
    ExpertCacheShape shape() const {return ExpertCacheShape{slots(),geometry.n_layers,geometry.n_expert};}
    // A layer may only take room from its own partition when --expert-cache-per-layer installed one.
    std::string partition(uint32_t layer) const {return partitioned?std::to_string(layer):std::string();}
    // (Re)install the per-layer partition from the cache's CURRENT budget. A zero quota means the split cannot
    // give one expert per layer, so the uniform cache stands - upstream's `if (q > 0)` fall-through.
    void install_split() {
        cache.clear_partitions();partitioned=false;report.per_layer_experts=0;report.per_layer_honored=false;
        if(!residency.expert_cache_per_layer) return;
        const auto split=per_layer_byte_split(cache.capacity(),layer_blob_bytes,geometry.n_expert);
        report.per_layer_experts=split.q;report.per_layer_honored=split.honored;
        if(!split.honored) return;
        for(size_t l=0;l<split.capacity.size();++l) cache.set_partition_capacity(std::to_string(l),split.capacity[l]);
        partitioned=true;
    }
};
namespace {
ExpertResolver resolve_table(WeightTable table,std::shared_ptr<const NativeExpertPack> pack) {
    return [table=std::move(table),pack=std::move(pack)](uint32_t layer,int32_t expert) {
        if(pack && expert>=0) {
            const auto source=pack->slices(layer,uint32_t(expert));
            return ExpertSlices{source[0],source[1],source[2],pack->scales(layer,uint32_t(expert)),
                source[0].ggml_type()==40 || source[1].ggml_type()==40 || source[2].ggml_type()==40};
        }
        const auto& g=table.geometry();const auto prefix="blk."+std::to_string(layer)+".";
        auto slice=[&](const char* routed,const char* shared,uint64_t columns,uint64_t rows) {
            const auto name=prefix+(expert==-1?shared:routed);const auto& info=table.info(name);
            const std::vector<uint64_t> expected=expert==-1?std::vector<uint64_t>{columns,rows}:std::vector<uint64_t>{columns,rows,uint64_t(g.n_expert)};
            if(info.gguf_shape!=expected) throw std::invalid_argument("expert GGUF shape mismatch: "+name);
            return expert==-1?table.matrix(name):table.expert(name,uint32_t(expert));
        };
        ExpertSlices result{slice("ffn_gate_exps.weight","ffn_gate_shexp.weight",g.n_embd,g.n_ff),
            slice("ffn_up_exps.weight","ffn_up_shexp.weight",g.n_embd,g.n_ff),
            slice("ffn_down_exps.weight","ffn_down_shexp.weight",g.n_ff,g.n_embd)};
        result.scaled_outputs=result.gate.ggml_type()==40 || result.up.ggml_type()==40 || result.down.ggml_type()==40;
        if(result.scaled_outputs) result.output_scales=expert_global_scales(table,layer,expert);
        return result;
    };
}
}
ExpertCache::ExpertCache(const Device& d,WeightTable table,uint64_t budget,uint64_t staging,std::shared_ptr<const NativeExpertPack> pack)
    :ExpertCache(d,table.geometry(),resolve_table(table,std::move(pack)),budget,staging) {
    // --expert-cache-per-layer sizes each layer's partition from THAT layer's stored expert blob (upstream
    // lay.blob_bytes(l)), so the role formats have to come from the table rather than the BF16 default the
    // geometry constructor charges. Three metadata reads per layer, no bytes touched.
    auto& g=impl_->geometry;std::vector<uint64_t> bytes;bytes.reserve((size_t) g.n_layers);
    for(int64_t l=0;l<g.n_layers;++l) {
        const auto prefix="blk."+std::to_string(l)+".";
        const std::array<uint32_t,3> types{table.info(prefix+"ffn_gate_exps.weight").ggml_type,
            table.info(prefix+"ffn_up_exps.weight").ggml_type,table.info(prefix+"ffn_down_exps.weight").ggml_type};
        bytes.push_back(expert_format(uint64_t(g.n_embd),uint64_t(g.n_ff),0,types).bytes);
    }
    impl_->layer_blob_bytes=std::move(bytes);
}
ExpertCache::ExpertCache(const Device& d,core::ModelGeometry g,ExpertResolver r,uint64_t budget,uint64_t staging) {
    if(!r || g.n_embd<=0 || g.n_ff<=0 || g.n_embd>65536 || g.n_ff>65536 || g.n_layers<=0 ||
       g.n_expert<=0 || g.n_expert>4096 || staging/4<uint64_t(std::max(g.n_embd,g.n_ff)))
        throw std::invalid_argument("invalid expert cache geometry/resolver/staging budget");
    const auto bytes=checked_product(checked_product(g.n_embd,g.n_ff),6);
    if(budget<bytes) throw std::invalid_argument("expert cache budget must hold one complete BF16 expert");
    impl_=std::make_unique<Impl>(d,g,std::move(r),budget,staging);
}
ExpertCache::~ExpertCache()=default;
ExpertLease ExpertCache::acquire(uint32_t layer,int32_t expert) {
    auto& p=*impl_;const auto& g=p.geometry;
    if(layer>=uint64_t(g.n_layers) || expert< -1 || expert>=g.n_expert) throw std::out_of_range("expert cache layer/ID");
    const auto key=expert_cache_key(layer,expert);
    std::lock_guard lock(p.loading);
    if(auto hit=p.cache.find(key)) { // no raw I/O or MTP planar relayout on a hit
        if(p.profiler) p.profiler->note_hit(layer,expert);
        return hit;
    }
    if(p.profiler) p.profiler->note_miss(layer,expert);
    const auto slices=p.resolve(layer,expert);
    if(slices.gate.columns()!=uint64_t(g.n_embd) || slices.gate.rows()!=uint64_t(g.n_ff) ||
       slices.up.columns()!=uint64_t(g.n_embd) || slices.up.rows()!=uint64_t(g.n_ff) ||
       slices.down.columns()!=uint64_t(g.n_ff) || slices.down.rows()!=uint64_t(g.n_embd)) throw std::invalid_argument("resolved expert shape mismatch");
    const auto format=expert_format(g.n_embd,g.n_ff,expert,{slices.gate.ggml_type(),slices.up.ggml_type(),slices.down.ggml_type()});
    const std::array<uint32_t,3> types{slices.gate.ggml_type(),slices.up.ggml_type(),slices.down.ggml_type()};
    // NVFP4 keeps its per-expert global scales alongside packed weights: the scale applies to
// each projection's F32 result (see packed_expert_flow.hpp), not to its storage format.
if(slices.scaled_outputs!=(types[0]==40 || types[1]==40 || types[2]==40)) throw std::invalid_argument("expert global scale contract missing/mismatched");
    for(size_t role=0;role<3;++role) if(!std::isfinite(slices.output_scales[role]) || slices.output_scales[role]<=0 || (types[role]!=40 && slices.output_scales[role]!=1)) throw std::invalid_argument("invalid expert global scale value");
    if(p.fp8) {
        auto fp8_format=fp8_expert_format(g.n_embd,g.n_ff);
        fp8_format.joined_gate_up=direct_experts_enabled();
        fp8_format.fp8_column_pairs=tpc_colpair_enabled();
        return p.cache.acquire_in(p.partition(layer),key,fp8_format.bytes,[&]()->std::unique_ptr<const ExpertWeights> {
            if(fp8_format.joined_gate_up) {
                auto pair=p.upload_fp8_pair(slices),dd=p.upload_fp8(slices.down,slices.output_scales[2]);
                return std::make_unique<ExpertWeights>(std::move(pair),DeviceBuffer(p.device,0),std::move(dd),fp8_format);
            }
            auto dg=p.upload_fp8(slices.gate,slices.output_scales[0]),du=p.upload_fp8(slices.up,slices.output_scales[1]);
            auto dd=p.upload_fp8(slices.down,slices.output_scales[2]);
            return std::make_unique<ExpertWeights>(std::move(dg),std::move(du),std::move(dd),fp8_format);
        });
    }
    return p.cache.acquire_in(p.partition(layer),key,format.bytes,[&]()->std::unique_ptr<const ExpertWeights> {
        // Publish only after all three uploads completed. finish() frees staging
        // immediately, preventing pinned memory from growing with cache entries.
        auto load=[&](const WeightSlice& source) {
            if(format.packed) return upload_packed_weights(p.device,source,p.copy,p.pinned);
            Bf16WeightUpload transfer(p.device,source,p.copy,p.pinned);return transfer.finish();
        };
        auto dg=load(slices.gate),du=load(slices.up),dd=load(slices.down);
        return std::make_unique<ExpertWeights>(std::move(dg),std::move(du),std::move(dd),format,slices.output_scales,slices.scaled_outputs);
    });
}
void ExpertCache::set_fp8(bool enabled) {
    auto& p=*impl_;
    if(enabled && direct_experts_enabled() && (p.geometry.n_embd%32 || p.geometry.n_ff%32))
        throw std::invalid_argument("direct FP8 expert dimensions must be divisible by 32");
    if(p.cache.stats().resident_bytes) throw std::logic_error("--fp8-experts must be chosen before any expert is loaded");
    p.fp8=enabled;
    // The admission arithmetic (slots, the per-layer split, the profile fill) counts the stored blob.
    if(enabled) p.layer_blob_bytes.assign((size_t) p.geometry.n_layers,fp8_expert_format(p.geometry.n_embd,p.geometry.n_ff).bytes);
    p.install_split();
}
bool ExpertCache::fp8() const {return impl_->fp8;}
CacheStats ExpertCache::stats() const {return impl_->cache.stats();}
ExpertLease ExpertCache::resident(uint32_t layer,int32_t expert) {
    const auto& g=impl_->geometry;
    if(layer>=uint64_t(g.n_layers) || expert< -1 || expert>=g.n_expert) throw std::out_of_range("expert resident layer/ID");
    return impl_->cache.find(expert_cache_key(layer,expert));
} void ExpertCache::resize(uint64_t budget) {
    if(budget<impl_->expert_bytes) throw std::invalid_argument("expert cache must retain room for one complete expert");
    impl_->cache.resize(budget);
    // resize() drops the installed partition map, so re-derive it from the new budget before anything can
    // acquire: otherwise a per-layer run would quietly fall back to one shared pool.
    impl_->install_split();
    if(impl_->profiler) impl_->profiler->sync_residency(impl_->cache.keys());
}
uint64_t ExpertCache::capacity() const {return impl_->cache.capacity();}
const core::ModelGeometry& ExpertCache::geometry() const {return impl_->geometry;}
// ---- the upstream profiling and adaptation knobs, off unless asked for (strata/hpu/expert_profile.hpp) ----
void ExpertCache::set_expert_profile_options(ExpertProfileOptions options) {
    auto& p=*impl_;p.expert_profile=std::move(options);
    if(!p.expert_profile.monitoring()) {p.profiler.reset();return;} // back to the cache exactly as it was
    if(!p.profiler) p.profiler=std::make_unique<ExpertProfiler>(p.geometry.n_layers,p.geometry.n_expert,p.expert_profile);
    else p.profiler->set_options(p.expert_profile); // the law moved: what was counted survives it
    if(!p.expert_profile.profile.empty()) {
        std::string err;
        if(!p.profiler->preload_profile(err)) throw std::invalid_argument("expert profile: "+err);
    }
}
const ExpertProfileOptions& ExpertCache::expert_profile_options() const {return impl_->expert_profile;}
ExpertProfiler* ExpertCache::expert_profiler() {return impl_->profiler.get();}
void ExpertCache::note_routing(uint32_t layer,const int32_t* ids,int64_t n) {
    if(impl_->profiler) impl_->profiler->note_routes(layer,ids,n);
}
bool ExpertCache::adapt_experts(std::vector<ExpertSwap>& applied) {
    applied.clear();
    auto& p=*impl_;
    if(!p.profiler || !p.profiler->counting()) return false;
    p.profiler->sync_residency(p.cache.keys()); // the cache is the truth about what is resident
    return p.profiler->tick(applied);
}
// ---- upstream's residency knobs (strata/hpu/expert_residency.hpp) -------------------------------------
void ExpertCache::set_expert_residency(ExpertResidencyOptions options) {
    auto& p=*impl_;
    p.residency=std::move(options);
    p.report=ResidencyReport{};
    p.install_split();   // --expert-cache-per-layer: one byte partition per layer, from this layer's own blob
    if(!p.residency.resident_cpu_experts) return;   // no RAM tier: nothing is admitted before routing
    if(!p.residency.has_expert_profile) return;     // refused by validate_expert_residency(); never reached
    std::vector<std::pair<int32_t,int32_t>> ranked;int64_t profile_slots=0;std::string error;
    if(!read_expert_profile(p.expert_profile.profile,p.geometry.n_layers,p.geometry.n_expert,ranked,profile_slots,error))
        throw std::invalid_argument("expert profile: "+error);
    // Which experts the static profile admits before any routing evidence (generate.cpp:5460-5478).
    const auto admitted=admitted_before_routing(p.residency,p.shape(),ranked);
    const bool pin=p.residency.pins_resident_copy();
    // Upstream's fill: in order, stop at the first expert that does not fit, KEEP what already landed and report
    // how many are resident. The ordering law is the pure unit's (strata/hpu/expert_residency.hpp), so a host test
    // can pin it without a Device.
    const auto fill=fill_admitted_residency(admitted,pin,[this,&p,pin](const int32_t layer,const int32_t expert) {
        try {(void)acquire(uint32_t(layer),expert);}
        catch(const std::exception&) {return false;}
        if(pin) p.cache.pin(expert_cache_key(uint32_t(layer),expert));
        return true;
    });
    p.report.admitted=fill.admitted;p.report.pinned=fill.pinned;
    if(p.profiler) p.profiler->sync_residency(p.cache.keys());
}
const ExpertResidencyOptions& ExpertCache::expert_residency_options() const {return impl_->residency;}
ExpertCache::ResidencyReport ExpertCache::residency_report() const {return impl_->report;}
}
