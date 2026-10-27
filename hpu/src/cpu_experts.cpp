#include "strata/hpu/cpu_experts.hpp"
#include "strata/hpu/expert_scales.hpp"
#include "strata/hpu/native_pack.hpp"
#include "strata/kernels/cpu/pool.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <mutex>

namespace strata::hpu {
namespace cpu=strata::kernels::cpu;
namespace {
cpu::PoolAffinity upstream_affinity(PoolAffinity affinity) {
    switch (affinity) {
        case PoolAffinity::Auto: return cpu::PoolAffinity::Auto;
        case PoolAffinity::PCores: return cpu::PoolAffinity::PCores;
        case PoolAffinity::All: return cpu::PoolAffinity::All;
    }
    return cpu::PoolAffinity::All;
}
struct CpuBlob {std::vector<uint8_t> bytes;};
cpu::NativeFmt layer_format(const WeightTable& table,uint32_t layer) {
    const auto& g=table.geometry();const auto prefix="blk."+std::to_string(layer)+".";
    const auto& gate=table.info(prefix+"ffn_gate_exps.weight");const auto& up=table.info(prefix+"ffn_up_exps.weight");const auto& down=table.info(prefix+"ffn_down_exps.weight");
    if(gate.gguf_shape!=std::vector<uint64_t>{uint64_t(g.n_embd),uint64_t(g.n_ff),uint64_t(g.n_expert)} || up.gguf_shape!=gate.gguf_shape ||
       down.gguf_shape!=std::vector<uint64_t>{uint64_t(g.n_ff),uint64_t(g.n_embd),uint64_t(g.n_expert)} || gate.ggml_type!=up.ggml_type)
        throw std::invalid_argument("CPU expert gate/up/down schema mismatch");
    cpu::NativeFmt format;std::string error;
    if(!cpu::native_fmt(int(gate.ggml_type),int(down.ggml_type),g.n_embd,g.n_ff,format,error)) throw std::invalid_argument(error);
    if(format.tail_off) (void)expert_global_scales(table,layer,0);
    return format;
}
std::vector<cpu::NativeFmt> validate_cpu(const WeightTable& table,uint64_t budget,uint32_t workers,uint32_t tasks) {
    const auto& g=table.geometry();
    // --pool-workers is upstream's non-negative int (generate.cpp:1980,2699); 0 means every physical
    // core and is resolved inside ExpertPool (pool.cpp:480-486). Values above INT_MAX would become a
    // negative int there and silently select that automatic count, so the cast is checked here.
    if(g.n_embd!=cpu::H || g.n_ff!=cpu::FF || workers>uint32_t(std::numeric_limits<int>::max()) ||
       !cpu::cpu_avx2_ok())
        throw std::invalid_argument("CPU experts require target H/FF, AVX2/FMA/F16C and --pool-workers in 0..2147483647");
    if(tasks>uint32_t(cpu::ExpertPool::kMaxTasks)) throw std::invalid_argument("expert pool tasks must be in 0..4096");
    std::vector<cpu::NativeFmt> formats;formats.reserve(g.n_layers);
    for(uint32_t layer=0;layer<uint64_t(g.n_layers);++layer) {
        auto format=layer_format(table,layer);if(format.bytes>budget) throw std::invalid_argument("CPU raw cache cannot hold one complete expert");
        formats.push_back(format);
    }
    return formats;
}
}
struct CpuExperts::Impl {
    WeightTable table;core::ModelGeometry geometry;std::vector<cpu::NativeFmt> formats;
    std::shared_ptr<const NativeExpertPack> pack;
    ResidentCache<CpuBlob> cache;std::unique_ptr<cpu::ExpertPool> pool;mutable std::mutex mutex;
    Impl(WeightTable t,uint64_t budget):table(std::move(t)),geometry(table.geometry()),cache(budget) {}
};
void CpuExperts::validate(const WeightTable& table,uint64_t budget,uint32_t workers,uint32_t tasks) {(void)validate_cpu(table,budget,workers,tasks);}
CpuPlacement CpuExperts::placement(uint32_t workers, CpuAffinity affinity) {
    const auto topology=cpu::detect_cpu_topology(true,upstream_affinity(affinity.affinity));
    CpuPlacement out;
    out.host_core=topology.host_core;out.worker_cores=topology.worker_cores;out.hybrid=topology.is_hybrid;
    out.p_cores=topology.p_cores;out.p_threads=topology.p_threads;out.e_cores=topology.e_cores;
    // ExpertPool takes the first n_ planned cores and leaves later workers unpinned (pool.cpp:500-505);
    // report exactly those when a count is given, the whole plan when it is 0 (upstream's automatic count).
    if(workers && out.worker_cores.size()>workers) out.worker_cores.resize(workers);
    return out;
}
void CpuExperts::apply_host_core(CpuAffinity affinity) {
    cpu::set_host_core(affinity.host_core == HostCore::Last ? cpu::HostCore::Last : cpu::HostCore::First);
}
CpuExperts::CpuExperts(WeightTable table,uint64_t budget,uint32_t workers,std::shared_ptr<const NativeExpertPack> pack,CpuAffinity affinity,uint32_t tasks) {
    auto formats=validate_cpu(table,budget,workers,tasks);
    // Upstream applies --host-core before the pool exists (:2293-2299); the swap is in detect_cpu_topology,
    // which the pool calls when it pins its workers. The calling (driver) thread is still never pinned.
    apply_host_core(affinity);
    auto p=std::make_unique<Impl>(std::move(table),budget);
    p->formats=std::move(formats);
    p->pack=std::move(pack);
    if(p->pack) for(uint32_t l=0;l<p->formats.size();++l) {
        const auto slices=p->pack->slices(l,0);const auto& format=p->formats[l];
        if(p->pack->blob_bytes(l)!=format.bytes || slices[0].ggml_type()!=uint32_t(format.gu_type) ||
           slices[1].ggml_type()!=uint32_t(format.gu_type) || slices[2].ggml_type()!=uint32_t(format.d_type) ||
           slices[0].columns()!=uint64_t(format.n_embd) || slices[0].rows()!=uint64_t(format.n_ff)) throw std::invalid_argument("CPU native expert pack blob geometry/type mismatch");
    }
    // Upstream constructs the pool with pin=true (src/program/generate.cpp:5051): the worker threads
    // are pinned to the topology's physical cores (src/kernels/cpu/pool.cpp:500-505). --host-core is
    // applied above, before the pool exists (:2293-2299); it selects which planned core the pool's host
    // worker keeps. The calling (driver) thread is never pinned by either backend, because the Synapse
    // runtime and its callback threads share the process.
    p->pool=std::make_unique<cpu::ExpertPool>(int(workers),affinity.pin,true,upstream_affinity(affinity.affinity),int(tasks));
    // generate.cpp:5052-5055: say which adaptation the pool took (automatic tasks, explicit cap, and
    // the worker count it actually resolved after topology detection).
    const int participating=p->pool->workers()+(p->pool->host_works()?1:0);
    std::cerr<<"strata generate: CPU pool tasks/phase: "<<(tasks?int(tasks):3*participating)
             <<(tasks?" (capped by rows)":" (automatic)")
             <<", participating threads: "<<participating<<'\n';
    impl_=std::move(p);
}
CpuExperts::~CpuExperts()=default;
std::vector<float> CpuExperts::run(uint32_t layer,int32_t expert,std::span<const float> rows) {
    auto& p=*impl_;std::lock_guard lock(p.mutex);const auto& g=p.geometry;const auto h=uint64_t(g.n_embd);
    if(layer>=p.formats.size() || expert<0 || expert>=g.n_expert || rows.empty() || rows.size()%h || rows.size()/h>cpu::MAXT)
        throw std::invalid_argument("CPU expert layer/ID/rows out of range");
    for(float x:rows) if(!std::isfinite(x)) throw std::invalid_argument("CPU expert requires finite activations");
    const auto& format=p.formats[layer];const auto prefix="blk."+std::to_string(layer)+".";
    auto lease=p.cache.acquire(std::to_string(layer)+":"+std::to_string(expert),format.bytes,[&]()->std::unique_ptr<const CpuBlob> {
        if(p.pack) {auto blob=std::make_unique<CpuBlob>();blob->bytes=p.pack->native_blob(layer,uint32_t(expert));return blob;}
        const auto gate=p.table.expert(prefix+"ffn_gate_exps.weight",expert),up=p.table.expert(prefix+"ffn_up_exps.weight",expert),down=p.table.expert(prefix+"ffn_down_exps.weight",expert);
        const auto scales=format.tail_off?expert_global_scales(p.table,layer,expert):std::array<float,3>{1,1,1};
        auto blob=std::make_unique<CpuBlob>();blob->bytes=native_expert_blob({gate,up,down},format.up_off,format.down_off,format.bytes,format.tail_off,scales);return blob;
    });
    const auto tokens=rows.size()/h;std::vector<float> output(rows.size());cpu::ExpertJobMulti job;job.blob=lease->bytes.data();job.nt=int(tokens);
    std::array<cpu::ActQ,cpu::MAXT> quantized{};alignas(64) std::array<std::array<uint8_t,cpu::kNativeActBytes>,cpu::MAXT> native{};
    for(size_t token=0;token<tokens;++token) {
        const auto* x=rows.data()+token*h;job.out[token]=output.data()+token*h;
        if(cpu::q2_native_kernels(format.gu_type)) {cpu::act_quant_any(x,int(h),quantized[token]);job.act[token]=&quantized[token];}
        else {cpu::native_quant_act(format,x,native[token].data());job.nact[token]=native[token].data();}
    }
    p.pool->run_split_multi_native(format,&job,1);
    for(float x:output) if(!std::isfinite(x)) throw std::runtime_error("non-finite CPU expert output");
    return output;
}
CacheStats CpuExperts::stats() const {return impl_->cache.stats();}
}
