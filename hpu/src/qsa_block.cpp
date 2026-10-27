#include "strata/hpu/qsa_block.hpp"
#include <memory>
#include <tuple>
#include "strata/hpu/qsa.hpp"
#include "strata/hpu/qsa_window.hpp"
#include "strata/hpu/qsa_window_flow.hpp"
#include "strata/hpu/qsa_phase_stats.hpp"
#include "strata/hpu/performance_modes.hpp"
#include "strata/hpu/session_flow.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <mutex>
#include <map>
#include <set>

namespace strata::hpu {
namespace {
// STRATA_HPU_QSA_KV_FP8: the scales of cache bindings `k`/`v` (cell offset `cell`), as "<name>_scale".
void add_cache_scales(std::vector<Binding>& out,const std::unique_ptr<DeviceBuffer>& keys,const std::unique_ptr<DeviceBuffer>& values,
                      const QsaConfig& c,const std::string& k,const std::string& v,uint64_t cell=0) {
    if(!qsa_kv_fp8_enabled()) return;
    out.push_back({k+"_scale",keys.get(),cell*qsa_cache_scale_bytes(c)});out.push_back({v+"_scale",values.get(),cell*qsa_cache_scale_bytes(c)});
}
}
namespace {
// A K/V store graph shared by the device's QSA layers: the layers launch it one after another on one stream and bind
// their own cache rows before each launch, so one compiled recipe serves all of them.
struct SharedStore {std::mutex mutex;std::unique_ptr<HostRecipe> recipe;};
std::shared_ptr<SharedStore> shared_store(const Device& device,uint64_t heads,uint64_t width,uint32_t rows) {
    static std::mutex mutex;static std::map<std::tuple<const void*,uint64_t,uint64_t,uint32_t>,std::weak_ptr<SharedStore>> registry;
    std::lock_guard lock(mutex);auto& entry=registry[{device.state().get(),heads,width,rows}];
    if(auto existing=entry.lock()) return existing;
    auto created=std::make_shared<SharedStore>();entry=created;return created;
}
}
struct QsaBlock::Impl {
    Device device;QsaConfig config;uint32_t hidden;QsaWeights weights;QsaState state;
    std::unique_ptr<HostRecipe> projection,attention,output;
    QsaWindow* window=nullptr;
    QsaHalfSink half_sink;std::vector<uint16_t> selected_value;
    uint32_t attention_bucket=0;
    std::map<uint32_t,std::shared_ptr<const DeviceBuffer>> mask_tables; // kept alive while bound
    // STRATA_HPU_QSA_DEVICE cache; `valid` cells equal the host pool.
    struct DeviceCache {DeviceBuffer keys,values,rope,q_gamma,k_gamma;uint32_t cells=0,valid=0;std::unique_ptr<HostBuffer> sync_host;
        // STRATA_HPU_QSA_DEVICE_SELECT: indexer gammas, the host's pooled blocks and the open block's raw keys.
        std::unique_ptr<DeviceBuffer> iq_gamma,ik_gamma,pooled,tail;std::unique_ptr<HostBuffer> select_host;
        // The window rows' RoPE in a buffer of its own: two bindings of one buffer's overlapping ranges fail
        // Synapse's section validation.
        std::unique_ptr<DeviceBuffer> window_rope;std::unique_ptr<HostBuffer> window_rope_host;
        // Recorded after a select window's uploads: the next window waits on it before rewriting the host buffers
        // (instead of draining the stream every layer, which kept the host from running ahead).
        std::unique_ptr<Event> select_uploaded;
        // STRATA_HPU_QSA_KV_FP8: keys/values hold FP8 codes and these one F32 scale per (cell, KV head).
        std::unique_ptr<DeviceBuffer> key_scale,value_scale;
        // Pooled rows already on the device (pooled holds the host's complete blocks; sync_pooled tops it up).
        uint64_t pooled_valid;};
    std::unique_ptr<DeviceCache> cache;
    void sync_pooled(const Stream&,DeviceCache&);
    // Copies the token's K/V rows from the chain stage into the cache; shared by every layer of the device (one
    // compile, not one per QSA layer: the graph depends only on the row geometry).
    std::shared_ptr<SharedStore> store;
    std::map<uint32_t,std::shared_ptr<SharedStore>> store_rows; // STRATA_HPU_DFLASH verify windows, per row count
    // Long-context verify window: the appended rows in the exact form the host law re-appends them at commit.
    struct Staged {
        QsaState::Checkpoint point;bool active=false,rewound=false;
        uint32_t start=0,rows=0,bucket=0;
        std::vector<float> keys,values,index_keys;
        std::vector<std::vector<int32_t>> ids;
    };
    std::unique_ptr<Staged> staged;
    mutable std::mutex mutex;
    Impl(const Device& d,QsaConfig c,uint32_t h,QsaWeights w):device(d),config(std::move(c)),hidden(h),weights(std::move(w)),state(config) {}
    uint32_t bucket(uint32_t count) const {
        return qsa_attention_bucket(config,count);
    }
    // Creates the device cache on first use and brings it up to the committed prefix (caller holds mutex).
    DeviceCache& ensure_cache(const Stream& stream);
};
QsaBlock::QsaBlock(const Device& d,QsaConfig c,uint32_t h,QsaWeights w,QsaWindow* window,QsaHalfSink half_sink) {
    c.validate();if(!h) throw std::invalid_argument("zero QSA hidden width");
    if(window && !window->compatible(c,h)) throw std::invalid_argument("QSA shared executor geometry mismatch");
    std::set<std::string> names;
    for(const auto& b:w.projections) names.insert(b.name);
    if(w.projections.size()!=5 || names!=std::set<std::string>{"w_q_gate","w_k","w_v","w_index_query","w_index_key"} || w.output.name!="w_out")
        throw std::invalid_argument("QSA requires five projection weight bindings and w_out");
    auto gamma=[](const std::vector<float>& v,uint64_t width,uint64_t heads) {
        if(v.size()!=width && v.size()!=width*heads) throw std::invalid_argument("QSA norm gamma shape mismatch");
        if(!all_finite(v)) throw std::invalid_argument("QSA gamma must be finite");
    };
    gamma(w.query_gamma,c.width,c.query_heads);gamma(w.key_gamma,c.width,c.kv_heads);
    gamma(w.index_query_gamma,c.index_width,c.index_heads);gamma(w.index_key_gamma,c.index_width,1);
    auto p=std::make_unique<Impl>(d,c,h,std::move(w));
    p->window=window;
    p->half_sink=std::move(half_sink);if(p->half_sink) {
        p->selected_value.resize(uint64_t(c.kv_heads)*c.width);
        if(p->selected_value.capacity()!=p->selected_value.size()) throw std::length_error("QSA half scratch exceeds admitted capacity");
    }
    p->projection=std::make_unique<HostRecipe>(d,build_qsa_projections(d,c,h),p->weights.projections,
        std::vector<std::string>{"q_gate","k","v","index_query","index_key"});
    const std::array<Binding,1> out_weights{p->weights.output};
    p->output=std::make_unique<HostRecipe>(d,build_qsa_output(d,c,h),out_weights,std::vector<std::string>{"out"});
    impl_=std::move(p);
}
QsaBlock::~QsaBlock()=default;
std::vector<float> QsaBlock::decode(std::span<const float> x,std::span<const float> bias) {
    auto& p=*impl_;std::lock_guard lock(p.mutex);const auto& c=p.config;
    if(x.size()!=p.hidden || p.state.size()>=c.capacity) throw std::invalid_argument("QSA decode shape/capacity mismatch");
    // Validate bias before the native projections or cache append.
    if(!bias.empty() && bias.size()!=uint64_t((p.state.size()+1)/c.block)+1) throw std::invalid_argument("QSA block bias shape mismatch");
    if(!all_finite(bias)) throw std::invalid_argument("QSA block bias must be finite");
    p.projection->set("x",x);p.projection->run();
    const auto qg=p.projection->output("q_gate");
    std::vector<float> query(uint64_t(c.query_heads)*c.width),gate(query.size());
    for(uint32_t head=0;head<c.query_heads;++head) {
        std::copy_n(qg.begin()+uint64_t(head)*2*c.width,c.width,query.begin()+uint64_t(head)*c.width);
        std::copy_n(qg.begin()+(uint64_t(head)*2+1)*c.width,c.width,gate.begin()+uint64_t(head)*c.width);
    }
    const auto projected_k=p.projection->output("k"),projected_idx=p.projection->output("index_query");
    std::vector<float> key(projected_k.begin(),projected_k.end()),index(projected_idx.begin(),projected_idx.end());
    rms_norm_rows(query,c.width,p.weights.query_gamma,c.epsilon);rms_norm_rows(key,c.width,p.weights.key_gamma,c.epsilon);
    rms_norm_rows(index,c.index_width,p.weights.index_query_gamma,c.epsilon);
    const auto rotation=rope_row_at(c.positions,c.rotary,c.position_base+p.state.size(),c.rope);
    rotate_neox(query,c.width,rotation);rotate_neox(key,c.width,rotation);rotate_neox(index,c.index_width,rotation);
    const auto point=p.state.checkpoint();
    bool appended=false;
    try {
        p.state.append(key,p.projection->output("v"),p.projection->output("index_key"),p.weights.index_key_gamma);
        appended=true;
        const auto ids=p.state.select(index,bias);
        const auto bucket=p.bucket(uint32_t(ids.size()));
        auto kv=p.state.gather(ids,bucket);
        if(bucket!=p.attention_bucket) {
            std::vector<std::string> outputs{"out"};if(p.half_sink) outputs.push_back("context_flat");
            auto executable=std::make_unique<HostRecipe>(p.device,build_qsa_attention(p.device,c,bucket,bool(p.half_sink)),std::span<const Binding>{},std::move(outputs));
            const std::array<float,1> scale{1/std::sqrt(float(c.width))};executable->set("scale",scale);
            p.attention=std::move(executable);p.attention_bucket=bucket;
        }
        p.attention->set("query",query);p.attention->set("key",kv.key);p.attention->set("value",kv.value);
        p.attention->set("mask",kv.mask);p.attention->set("q_gate",gate);p.attention->run();
        p.output->set("attention",p.attention->output("out"));p.output->run();
        const auto out=p.output->output("out");
        if(!all_finite(out)) throw std::runtime_error("non-finite QSA block output");
        if(p.half_sink) {
            p.state.selected_value_bits(ids.front(),p.selected_value);
            const auto value=p.projection->output("v");
            QsaHalfView view{p.attention->output("context_flat"),value,value,key,p.selected_value,p.state.value_pool_prefix()};
            std::copy_n(ids.begin(),std::min<size_t>(4,ids.size()),view.ids.begin());
            const auto cells=p.state.size();view.step={int32_t(cells-1),int32_t(cells),int32_t(cells/c.block),int32_t(c.selection_width(cells))};
            p.half_sink(view);
        }
        return {out.begin(),out.end()};
    } catch(...) {if(appended) p.state.restore(point);throw;}
}
const std::vector<Binding>& QsaBlock::projection_bindings() const {return impl_->weights.projections;}
const Binding& QsaBlock::output_binding() const {return impl_->weights.output;}
QsaBlock::FusedAttention QsaBlock::fused_attend(std::span<const float> qg,std::span<const float> projected_k,std::span<const float> v,
                                                std::span<const float> projected_idx,std::span<const float> index_key) {
    auto& p=*impl_;std::lock_guard lock(p.mutex);const auto& c=p.config;
    if(p.state.size()>=c.capacity || qg.size()!=uint64_t(c.query_heads)*2*c.width) throw std::invalid_argument("QSA fused decode shape/capacity mismatch");
    QsaPhaseTimer timer;
    // The same host steps as decode(), on the projections the layer's graph produced.
    FusedAttention out;out.query.resize(uint64_t(c.query_heads)*c.width);out.gate.resize(out.query.size());
    for(uint32_t head=0;head<c.query_heads;++head) {
        std::copy_n(qg.begin()+uint64_t(head)*2*c.width,c.width,out.query.begin()+uint64_t(head)*c.width);
        std::copy_n(qg.begin()+(uint64_t(head)*2+1)*c.width,c.width,out.gate.begin()+uint64_t(head)*c.width);
    }
    std::vector<float> key(projected_k.begin(),projected_k.end()),index(projected_idx.begin(),projected_idx.end());
    rms_norm_rows(out.query,c.width,p.weights.query_gamma,c.epsilon);rms_norm_rows(key,c.width,p.weights.key_gamma,c.epsilon);
    rms_norm_rows(index,c.index_width,p.weights.index_query_gamma,c.epsilon);
    timer.mark("attend.norm");
    const auto rotation=rope_row_at(c.positions,c.rotary,c.position_base+p.state.size(),c.rope);
    rotate_neox(out.query,c.width,rotation);rotate_neox(key,c.width,rotation);rotate_neox(index,c.index_width,rotation);
    timer.mark("attend.rope");
    p.state.append(key,v,index_key,p.weights.index_key_gamma);
    timer.mark("attend.append");
    auto ids=p.state.select(index,{});
    timer.mark("attend.select");
    out.ids=std::move(ids);out.bucket=p.bucket(uint32_t(out.ids.size()));out.scale=1/std::sqrt(float(c.width));
    return out;
}
void QsaBlock::gather_selected(std::span<const int32_t> ids,uint32_t bucket,std::span<float> key,std::span<float> value,
                               std::span<float> mask) {
    auto& p=*impl_;std::lock_guard lock(p.mutex);p.state.gather_into(ids,bucket,key,value,mask);
}
uint32_t QsaBlock::size() const {auto& p=*impl_;std::lock_guard lock(p.mutex);return p.state.size();}
std::vector<float> QsaBlock::prefill(std::span<const float> input,const std::function<bool()>& cancelled) {
    auto& p=*impl_;std::lock_guard lock(p.mutex);
    if(!p.window) throw std::invalid_argument("QSA prefill requires a shared native executor");
    QsaWindowOperations operations;
    operations.project=[&](auto x) {return p.window->project(x,p.weights.projections);};
    operations.attention=[&](const QsaWindowBatch& batch) {return p.window->attention(batch);};
    operations.output=[&](auto attention) {return p.window->output(attention,p.weights.output);};
    std::unique_ptr<Stream> copy;
    if(qsa_device_gather_enabled()) {
        copy=std::make_unique<Stream>(p.device);operations.selected_ids_only=true;
        operations.attention=[&](const QsaWindowBatch& batch) {
            auto& cache=p.ensure_cache(*copy);
            std::vector<Binding> bindings{{"kv.cache_k",&cache.keys},{"kv.cache_v",&cache.values}};
            add_cache_scales(bindings,cache.key_scale,cache.value_scale,p.config,"kv.cache_k","kv.cache_v");
            return p.window->attention(batch,bindings);
        };
    }
    try {
        return execute_qsa_window(p.state,input,p.hidden,{p.weights.query_gamma,p.weights.key_gamma,p.weights.index_query_gamma,p.weights.index_key_gamma},operations,cancelled);
    } catch(...) {
        // execute_qsa_window rewinds host state. A subsequent different branch
        // can append to the same extent before ensure_cache sees that rewind.
        if(p.cache) p.cache->valid=0;
        throw;
    }
}
QsaState::Checkpoint QsaBlock::checkpoint() const {auto& p=*impl_;std::lock_guard lock(p.mutex);return p.state.checkpoint();}
QsaState::Checkpoint QsaBlock::snapshot() const {auto& p=*impl_;std::lock_guard lock(p.mutex);return p.state.snapshot();}
void QsaBlock::snapshot_sizes(const QsaState::Checkpoint& point,uint64_t (&sizes)[5]) const {auto& p=*impl_;std::lock_guard lock(p.mutex);p.state.snapshot_sizes(point,sizes);}
QsaState::Checkpoint QsaBlock::snapshot_reused(QsaState::Checkpoint&& prior,uint32_t whole_cells,uint32_t pooled_rows_kept) const {
    auto& p=*impl_;std::lock_guard lock(p.mutex);return p.state.snapshot_reused(std::move(prior),whole_cells,pooled_rows_kept);
}
const QsaConfig& QsaBlock::config() const {auto& p=*impl_;std::lock_guard lock(p.mutex);return p.state.config();}
void QsaBlock::write_snapshot(StateWriter& writer,const QsaState::Checkpoint& point) const {auto& p=*impl_;std::lock_guard lock(p.mutex);p.state.write_snapshot(writer,point);}
QsaState::Checkpoint QsaBlock::read_snapshot(StateReader& reader,uint32_t cells) const {auto& p=*impl_;std::lock_guard lock(p.mutex);return p.state.read_snapshot(reader,cells);}
void QsaBlock::restore(const QsaState::Checkpoint& point) {
    auto& p=*impl_;std::lock_guard lock(p.mutex);p.state.restore(point);if(p.cache) p.cache->valid=0;
    // The whole state was replaced, so a window staged before the restore no longer describes it.
    if(p.staged) p.staged->active=false;
}
void QsaBlock::reset() {
    auto& p=*impl_;std::lock_guard lock(p.mutex);p.state.reset();if(p.cache) p.cache->valid=0;
    if(p.staged) p.staged->active=false;
}
namespace {
// One 128-byte-aligned row per cached-cell count p in [0, bucket]: [bucket+1] F32, 0 for the current cell and the p
// cached cells, -inf for the unused tail. Bindings must be 128-byte aligned, so rows are padded.
uint64_t device_mask_stride(uint32_t bucket) {return (uint64_t(bucket+1)*4+127)/128*128;}
std::shared_ptr<const DeviceBuffer> device_mask(const Device& device,uint32_t bucket) {
    static std::mutex mutex;static std::map<std::pair<const void*,uint32_t>,std::weak_ptr<const DeviceBuffer>> tables;
    std::lock_guard lock(mutex);auto& entry=tables[{device.state().get(),bucket}];
    if(auto existing=entry.lock()) return existing;
    const uint64_t stride=device_mask_stride(bucket),bytes=stride*(uint64_t(bucket)+1);
    auto created=std::make_shared<DeviceBuffer>(device,bytes);HostBuffer host(device,bytes);
    std::memset(host.data(),0,bytes);
    for(uint64_t cells=0;cells<=bucket;++cells) {
        auto* row=reinterpret_cast<float*>(static_cast<uint8_t*>(host.data())+cells*stride);
        for(uint64_t i=0;i<=bucket;++i) row[i]=i<=cells?0.0f:-INFINITY;
    }
    Stream copy(device);upload(copy,*created,host,bytes);copy.synchronize();
    entry=created;return created;
}
}
bool QsaBlock::device_ready() const {
    const auto& p=*impl_;std::lock_guard lock(p.mutex);const auto& c=p.config;
    const auto cells=p.state.size();
    return c.kv==KvMode::Fp16 && (!c.positions || !c.positions->installed()) && p.weights.query_gamma.size()==c.width &&
           p.weights.key_gamma.size()==c.width && c.rotary<=64 && c.rotary%2==0 && c.width>=128 && c.width%64==0 &&
           cells<c.capacity && uint64_t(cells)+1<=uint64_t(c.top_k)+c.block-1 && !p.half_sink;
}
// The select buffers (first use) and the host's complete pooled blocks on the device: only the rows not there yet,
// plus the last eight again (a rewind of a few tokens, then other tokens, can rewrite them between two calls).
// Waits for the previous upload from the staging rows first.
void QsaBlock::Impl::sync_pooled(const Stream& stream,DeviceCache& cache) {
    auto& p=*this;const auto& c=p.config;
    if(cache.select_uploaded && cache.select_uploaded->recorded()) cache.select_uploaded->synchronize();
    const uint64_t width=c.index_width,blocks=uint64_t(c.capacity)/c.block+2;
    if(!cache.pooled) {
        const auto upload_vector=[&](std::unique_ptr<DeviceBuffer>& target,const std::vector<float>& values) {
            target=std::make_unique<DeviceBuffer>(p.device,values.size()*4);HostBuffer host(p.device,values.size()*4);
            std::memcpy(host.data(),values.data(),values.size()*4);upload(stream,*target,host,values.size()*4);stream.synchronize();
        };
        upload_vector(cache.iq_gamma,p.weights.index_query_gamma);upload_vector(cache.ik_gamma,p.weights.index_key_gamma);
        cache.pooled=std::make_unique<DeviceBuffer>(p.device,blocks*width*4);
        cache.tail=std::make_unique<DeviceBuffer>(p.device,uint64_t(c.block)*width*4);
        cache.select_host=std::make_unique<HostBuffer>(p.device,(blocks+c.block)*width*4);
        cache.pooled_valid=0;
    }
    const uint64_t complete=p.state.size()/c.block;
    const uint64_t from=std::min(cache.pooled_valid,complete)>8?std::min(cache.pooled_valid,complete)-8:0;
    if(complete>from) {
        const auto pooled=p.state.pooled_rows();auto* host=static_cast<float*>(cache.select_host->data());
        std::copy_n(pooled.begin()+from*width,(complete-from)*width,host+from*width);
        upload(stream,*cache.pooled,*cache.select_host,(complete-from)*width*4,from*width*4,from*width*4);
    }
    cache.pooled_valid=complete;
}
QsaBlock::Impl::DeviceCache& QsaBlock::Impl::ensure_cache(const Stream& stream) {
    auto& p=*this;const auto& c=p.config;
    const uint64_t row=uint64_t(c.kv_heads)*c.width;const uint32_t cell=p.state.size();
    if(!p.cache) {
        // Identity attention only needs top_k+block-1 cells. Sparse device gather
        // opts into the admitted full-context mirror; RoPE stays identity-sized.
        const uint32_t identity_cells=std::min(c.capacity,c.selection_width(c.capacity));
        const bool select=qsa_device_select_enabled() && qsa_device_gather_enabled();
        // The device selection reads RoPE rows at any cell (window rows and block starts), so its table covers the context.
        const uint32_t rope_cells=select?c.capacity:identity_cells;
        const uint32_t cells=qsa_device_gather_enabled()?c.capacity:identity_cells;
        const uint64_t row_bytes=qsa_cache_row_bytes(c),fp8=qsa_kv_fp8_enabled();
        if(fp8 && !qsa_device_gather_enabled()) throw std::logic_error("STRATA_HPU_QSA_KV_FP8 needs STRATA_HPU_QSA_DEVICE_GATHER");
        auto cache=std::make_unique<Impl::DeviceCache>(Impl::DeviceCache{DeviceBuffer(p.device,cells*row_bytes),DeviceBuffer(p.device,cells*row_bytes),
            DeviceBuffer(p.device,uint64_t(rope_cells)*c.rotary*4),DeviceBuffer(p.device,uint64_t(c.width)*4),DeviceBuffer(p.device,uint64_t(c.width)*4),cells,0,{},{},{},{},{},{},{},{},{},{},{},0});
        // Only the short identity path can read masked padding. Sparse gather never
        // reads an uncommitted cell, so zeroing the full context is unnecessary.
        HostBuffer host(p.device,std::max<uint64_t>(uint64_t(identity_cells)*row*4,uint64_t(rope_cells)*c.rotary*4));
        std::memset(host.data(),0,uint64_t(identity_cells)*row*4);
        if(fp8) {
            cache->key_scale=std::make_unique<DeviceBuffer>(p.device,uint64_t(cells)*qsa_cache_scale_bytes(c));
            cache->value_scale=std::make_unique<DeviceBuffer>(p.device,uint64_t(cells)*qsa_cache_scale_bytes(c));
        }
        if(select || fp8) {
            // Dense select attention reads the masked gap of the cache: every cell starts finite (zero codes, zero scales).
            zero(stream,cache->keys,uint64_t(cells)*row_bytes);zero(stream,cache->values,uint64_t(cells)*row_bytes);
            if(fp8) {zero(stream,*cache->key_scale,uint64_t(cells)*qsa_cache_scale_bytes(c));zero(stream,*cache->value_scale,uint64_t(cells)*qsa_cache_scale_bytes(c));}
            stream.synchronize();
        } else {
            upload(stream,cache->keys,host,uint64_t(identity_cells)*row*4);upload(stream,cache->values,host,uint64_t(identity_cells)*row*4);stream.synchronize();
        }
        auto* rope=static_cast<float*>(host.data());const uint32_t half=c.rotary/2;
        for(uint32_t at=0;at<rope_cells;++at) {
            const auto r=rope_row(c.rotary,c.position_base+at,c.rope);
            std::copy(r.cosine.begin(),r.cosine.end(),rope+uint64_t(at)*c.rotary);std::copy(r.sine.begin(),r.sine.end(),rope+uint64_t(at)*c.rotary+half);
        }
        upload(stream,cache->rope,host,uint64_t(rope_cells)*c.rotary*4);stream.synchronize();
        std::memcpy(host.data(),p.weights.query_gamma.data(),uint64_t(c.width)*4);upload(stream,cache->q_gamma,host,uint64_t(c.width)*4);stream.synchronize();
        std::memcpy(host.data(),p.weights.key_gamma.data(),uint64_t(c.width)*4);upload(stream,cache->k_gamma,host,uint64_t(c.width)*4);stream.synchronize();
        p.cache=std::move(cache);
    }
    auto& cache=*p.cache;
    if(cache.valid>cell) cache.valid=0;
    if(cache.valid<cell) {
        // Bounded pinned staging even when restoring an entire long context.
        const uint32_t chunk=std::min<uint32_t>(cache.cells,128);
        const bool fp8=qsa_kv_fp8_enabled();const uint64_t hk=c.kv_heads,row_bytes=qsa_cache_row_bytes(c),scale_bytes=qsa_cache_scale_bytes(c);
        // F32 rows, then (FP8) their codes and scales behind them: keys [0, 2 chunk rows), codes, scales.
        if(!cache.sync_host) cache.sync_host=std::make_unique<HostBuffer>(p.device,2*chunk*row*4+(fp8?2*chunk*(row+scale_bytes):0));
        auto& host=*cache.sync_host;auto* keys=static_cast<float*>(host.data());
        auto* codes=static_cast<uint8_t*>(host.data())+2*chunk*row*4;auto* scales=reinterpret_cast<float*>(codes+2*chunk*row);
        while(cache.valid<cell) {
            const uint32_t count=std::min(chunk,cell-cache.valid);
            p.state.fp16_cells(cache.valid,count,std::span<float>(keys,count*row),std::span<float>(keys+count*row,count*row));
            if(!fp8) {
                upload(stream,cache.keys,host,count*row*4,cache.valid*row*4,0);
                upload(stream,cache.values,host,count*row*4,cache.valid*row*4,count*row*4);
            } else {
                qsa_cache_encode(std::span<const float>(keys,count*row),hk,c.width,codes,scales);
                qsa_cache_encode(std::span<const float>(keys+count*row,count*row),hk,c.width,codes+count*row,scales+count*kQsaScaleSlots);
                const uint64_t base=2*chunk*row*4,scale_base=base+2*chunk*row;
                upload(stream,cache.keys,host,count*row_bytes,cache.valid*row_bytes,base);
                upload(stream,cache.values,host,count*row_bytes,cache.valid*row_bytes,base+count*row);
                upload(stream,*cache.key_scale,host,count*scale_bytes,cache.valid*scale_bytes,scale_base);
                upload(stream,*cache.value_scale,host,count*scale_bytes,cache.valid*scale_bytes,scale_base+count*scale_bytes);
            }
            stream.synchronize();cache.valid+=count;
        }
    }
    return cache;
}
std::vector<Binding> QsaBlock::device_gather_bindings(const Stream& stream) {
    auto& p=*impl_;std::lock_guard lock(p.mutex);
    if(!qsa_device_gather_enabled() || p.config.kv!=KvMode::Fp16 || p.config.capacity>262144 || p.config.width%64)
        throw std::logic_error("QSA device gather not admitted");
    auto& cache=p.ensure_cache(stream);
    std::vector<Binding> out{{"kv.cache_k",&cache.keys},{"kv.cache_v",&cache.values}};
    add_cache_scales(out,cache.key_scale,cache.value_scale,p.config,"kv.cache_k","kv.cache_v");
    return out;
}
void QsaBlock::staged_gather_ids(std::span<float> destination,uint32_t first,uint32_t count) {
    auto& p=*impl_;std::lock_guard lock(p.mutex);
    if(!p.staged || !p.staged->active || first>=p.staged->rows)
        throw std::invalid_argument("QSA staged device ids shape/state mismatch");
    if(!count) count=p.staged->rows-first;
    if(count>p.staged->rows-first || destination.size()!=uint64_t(count)*p.staged->bucket)
        throw std::invalid_argument("QSA staged device ids shape/state mismatch");
    std::fill(destination.begin(),destination.end(),-1.0f);
    for(uint32_t row=0;row<count;++row)
        for(size_t i=0;i<p.staged->ids[first+row].size();++i)
            destination[uint64_t(row)*p.staged->bucket+i]=float(p.staged->ids[first+row][i]);
}
QsaBlock::DeviceStep QsaBlock::device_step(const Stream& stream) {
    auto& p=*impl_;std::lock_guard lock(p.mutex);const auto& c=p.config;
    const uint32_t cell=p.state.size();auto& cache=p.ensure_cache(stream);
    DeviceStep step;step.cell=cell;step.bucket=qsa_device_bucket(c,cell,1);
    const auto mask=device_mask(p.device,step.bucket);
    if(!p.mask_tables.count(step.bucket)) p.mask_tables[step.bucket]=mask;
    step.bindings={{"dev.k_cache",&cache.keys,0},{"dev.v_cache",&cache.values,0},
                   {"dev.rope",&cache.rope,uint64_t(cell)*c.rotary*4},{"dev.mask",mask.get(),uint64_t(cell)*device_mask_stride(step.bucket)},
                   {"dev.q_gamma",&cache.q_gamma,0},{"dev.k_gamma",&cache.k_gamma,0}};
    add_cache_scales(step.bindings,cache.key_scale,cache.value_scale,c,"dev.k_cache","dev.v_cache");
    return step;
}
void QsaBlock::device_store(const Stream& stream,uint32_t cell,const DeviceBuffer& stage,uint64_t offset) {
    auto& p=*impl_;std::lock_guard lock(p.mutex);const auto& c=p.config;const uint64_t row=uint64_t(c.kv_heads)*c.width;
    if(!p.cache || cell>=p.cache->cells) throw std::logic_error("QSA device store before its step");
    const uint64_t row_bytes=qsa_cache_row_bytes(c);
    std::vector<Binding> io{{"k_now",&stage,offset},{"v_now",&stage,offset+row*4},
                            {"k_row",&p.cache->keys,uint64_t(cell)*row_bytes},{"v_row",&p.cache->values,uint64_t(cell)*row_bytes}};
    add_cache_scales(io,p.cache->key_scale,p.cache->value_scale,c,"k_row","v_row",cell);
    if(!p.store) p.store=shared_store(p.device,c.kv_heads,c.width,0);
    std::lock_guard store_lock(p.store->mutex);auto& recipe=p.store->recipe;
    if(!recipe) {
        GraphBuilder g(p.device);
        const auto ks=g.tensor({"k_now",DType::F32,{1,c.kv_heads,c.width}}),vs=g.tensor({"v_now",DType::F32,{1,c.kv_heads,c.width}});
        qsa_cache_write(g,"k_row",ks,1,c.kv_heads,c.width);qsa_cache_write(g,"v_row",vs,1,c.kv_heads,c.width);
        recipe=std::make_unique<HostRecipe>(p.device,g.compile("strata_qsa_device_store"),io,std::vector<std::string>{});
    }
    for(const auto& binding:io) recipe->bind_external(binding);
    recipe->launch(stream);
}
void QsaBlock::device_commit(std::span<const float> k,std::span<const float> v,std::span<const float> raw_index_key) {
    auto& p=*impl_;std::lock_guard lock(p.mutex);
    const uint32_t before=p.state.size();
    p.state.append(k,v,raw_index_key,p.weights.index_key_gamma);
    if(p.cache && p.cache->valid==before) p.cache->valid=before+1;
}
uint32_t QsaBlock::device_select_keys(uint32_t cells,uint32_t rows,uint64_t capacity) {
    const uint64_t need=uint64_t(cells)+rows;
    if(performance_env("STRATA_HPU_QSA_SELECT_POW2")) {uint32_t keys=4096;while(keys<need) keys*=2;return keys;}
    // Rounded up to a step in (need/16, need/8] (at least 4096): at most ~12% padded keys instead of up to 2x, with
    // log-many key buckets (recipes). Clamped to the capacity when that still fits. STRATA_HPU_QSA_SELECT_POW2=1:
    // the former powers of two.
    uint64_t step=4096;while(step*8<=need) step*=2;
    uint64_t keys=std::max<uint64_t>(4096,(need+step-1)/step*step);
    // The select graphs take at least one cached key (keys > rows): a window wider than the identity selection that
    // starts the context (cells 0, an 8192-row window) needs one more step.
    if(keys<=rows) keys+=step;
    const uint64_t cap=capacity/1024*1024; // tpc/qsa_select_mask.c: keys / block % 256 == 0
    if(keys>cap && cap>=need && cap>rows) keys=cap;
    return uint32_t(keys);
}
bool QsaBlock::device_select_ready(uint32_t rows) const {
    if(!rows || !qsa_device_select_enabled() || !qsa_device_gather_enabled()) return false;
    const auto& p=*impl_;std::lock_guard lock(p.mutex);const auto& c=p.config;const uint64_t cells=p.state.size();
    const uint64_t keys=device_select_keys(uint32_t(cells),rows,c.capacity);
    return c.kv==KvMode::Fp16 && (!c.positions || !c.positions->installed()) && p.weights.query_gamma.size()==c.width &&
           p.weights.key_gamma.size()==c.width && c.rotary<=64 && c.rotary%2==0 && c.width>=128 && c.width%64==0 &&
           c.index_width>=128 && c.index_width%64==0 && c.rotary<=c.index_width && !p.half_sink &&
           cells+rows>uint64_t(c.top_k)+c.block-1 && cells+rows<=c.capacity && keys<=c.capacity && keys-rows>=cells;
}
uint32_t QsaBlock::index_query_gamma_heads() const {
    const auto& p=*impl_;return p.weights.index_query_gamma.size()==p.config.index_width?1u:p.config.index_heads;
}
QsaBlock::DeviceStep QsaBlock::device_select_step(const Stream& stream,uint32_t rows) {
    auto& p=*impl_;std::lock_guard lock(p.mutex);const auto& c=p.config;
    const uint32_t cell=p.state.size(),keys=device_select_keys(cell,rows,c.capacity);
    if(!rows || uint64_t(keys)>c.capacity) throw std::invalid_argument("QSA select window exceeds capacity");
    auto& cache=p.ensure_cache(stream);
    const uint64_t width=c.index_width,blocks=uint64_t(c.capacity)/c.block+2;
    p.sync_pooled(stream,cache);
    const auto tail=p.state.tail_rows();
    auto* host=static_cast<float*>(cache.select_host->data());
    if(!cache.window_rope || cache.window_rope->size()<uint64_t(rows)*c.rotary*4) {
        cache.window_rope=std::make_unique<DeviceBuffer>(p.device,uint64_t(rows)*c.rotary*4);
        cache.window_rope_host=std::make_unique<HostBuffer>(p.device,uint64_t(rows)*c.rotary*4);
    }
    {
        auto* rope=static_cast<float*>(cache.window_rope_host->data());const uint32_t half=c.rotary/2;
        for(uint32_t at=0;at<rows;++at) {
            const auto r=rope_row(c.rotary,c.position_base+cell+at,c.rope);
            std::copy(r.cosine.begin(),r.cosine.end(),rope+uint64_t(at)*c.rotary);std::copy(r.sine.begin(),r.sine.end(),rope+uint64_t(at)*c.rotary+half);
        }
        upload(stream,*cache.window_rope,*cache.window_rope_host,uint64_t(rows)*c.rotary*4);
    }
    std::fill(host+blocks*width,host+(blocks+c.block)*width,0.0f);
    std::copy(tail.begin(),tail.end(),host+blocks*width);
    upload(stream,*cache.tail,*cache.select_host,uint64_t(c.block)*width*4,0,blocks*width*4);
    if(!cache.select_uploaded) cache.select_uploaded=std::make_unique<Event>(p.device);
    cache.select_uploaded->record(stream);
    DeviceStep step;step.cell=cell;step.bucket=keys;
    step.bindings={{"dev.k_cache",&cache.keys,0},{"dev.v_cache",&cache.values,0},{"dev.rope",cache.window_rope.get(),0},
                   {"sel.rope_all",&cache.rope,0},{"dev.q_gamma",&cache.q_gamma,0},{"dev.k_gamma",&cache.k_gamma,0},
                   {"sel.iq_gamma",cache.iq_gamma.get(),0},{"sel.ik_gamma",cache.ik_gamma.get(),0},
                   {"sel.pooled",cache.pooled.get(),0},{"sel.tail",cache.tail.get(),0}};
    add_cache_scales(step.bindings,cache.key_scale,cache.value_scale,c,"dev.k_cache","dev.v_cache");
    return step;
}
bool QsaBlock::device_decode_select_ready(uint32_t rows) const {
    if(!rows || rows>8 || !qsa_decode_select_enabled() || !qsa_kv_fp8_enabled() || !device_select_ready(std::max(rows,2u))) return false;
    const auto& p=*impl_;std::lock_guard lock(p.mutex);const auto& c=p.config;const uint64_t cells=p.state.size();
    // Past the identity selection (device_ready() covers the rest) and with room for the rows.
    return c.width==256 && cells+1>uint64_t(c.top_k)+c.block-1 && cells+rows<=c.capacity &&
           device_select_keys(uint32_t(cells),rows,c.capacity)/c.block>=(c.selection_width(c.capacity)+c.block-2)/c.block;
}
QsaBlock::DeviceStep QsaBlock::device_decode_select_step(const Stream& stream,uint32_t rows) {
    auto& p=*impl_;std::lock_guard lock(p.mutex);const auto& c=p.config;
    const uint32_t cell=p.state.size();
    auto& cache=p.ensure_cache(stream);
    p.sync_pooled(stream,cache);
    if(!cache.select_uploaded) cache.select_uploaded=std::make_unique<Event>(p.device);
    cache.select_uploaded->record(stream);
    DeviceStep step;step.cell=cell;step.bucket=device_select_keys(cell,rows,c.capacity)/c.block; // pooled rows scored
    step.bindings={{"dev.k_cache",&cache.keys,0},{"dev.v_cache",&cache.values,0},{"dev.rope",&cache.rope,uint64_t(cell)*c.rotary*4},
                   {"dev.q_gamma",&cache.q_gamma,0},{"dev.k_gamma",&cache.k_gamma,0},{"sel.iq_gamma",cache.iq_gamma.get(),0},
                   {"dsel.pooled",cache.pooled.get(),0}};
    add_cache_scales(step.bindings,cache.key_scale,cache.value_scale,c,"dev.k_cache","dev.v_cache");
    return step;
}
bool QsaBlock::device_verify_ready(uint32_t rows) const {
    if(!rows || !device_ready()) return false;
    const auto& p=*impl_;std::lock_guard lock(p.mutex);const auto& c=p.config;const uint64_t cells=p.state.size();
    return cells+rows<=c.capacity && cells+rows<=uint64_t(c.top_k)+c.block-1;
}
QsaBlock::DeviceStep QsaBlock::device_verify_step(const Stream& stream,uint32_t rows) {
    auto& p=*impl_;std::lock_guard lock(p.mutex);const auto& c=p.config;
    const uint32_t cell=p.state.size();
    if(!rows || uint64_t(cell)+rows>c.capacity) throw std::invalid_argument("QSA verify window exceeds capacity");
    auto& cache=p.ensure_cache(stream);
    DeviceStep step;step.cell=cell;step.bucket=qsa_device_bucket(c,cell,rows);
    if(step.bucket<cell || step.bucket>cache.cells || uint64_t(cell)+rows>cache.cells) throw std::logic_error("QSA verify bucket does not cover the cache");
    step.bindings={{"dev.k_cache",&cache.keys,0},{"dev.v_cache",&cache.values,0},{"dev.rope",&cache.rope,uint64_t(cell)*c.rotary*4},
                   {"dev.q_gamma",&cache.q_gamma,0},{"dev.k_gamma",&cache.k_gamma,0}};
    add_cache_scales(step.bindings,cache.key_scale,cache.value_scale,c,"dev.k_cache","dev.v_cache");
    return step;
}
void QsaBlock::device_store_rows(const Stream& stream,uint32_t cell,uint32_t rows,const DeviceBuffer& stage,uint64_t offset) {
    auto& p=*impl_;std::lock_guard lock(p.mutex);const auto& c=p.config;const uint64_t row=uint64_t(c.kv_heads)*c.width,count=row*rows;
    if(!p.cache || !rows || uint64_t(cell)+rows>p.cache->cells) throw std::logic_error("QSA device verify store before its step");
    auto& shared=p.store_rows[rows];
    if(!shared) shared=shared_store(p.device,c.kv_heads,c.width,rows);
    std::lock_guard store_lock(shared->mutex);auto& recipe=shared->recipe;
    const uint64_t row_bytes=qsa_cache_row_bytes(c);
    std::vector<Binding> io{{"k_now",&stage,offset},{"v_now",&stage,offset+count*4},
                            {"k_row",&p.cache->keys,uint64_t(cell)*row_bytes},{"v_row",&p.cache->values,uint64_t(cell)*row_bytes}};
    add_cache_scales(io,p.cache->key_scale,p.cache->value_scale,c,"k_row","v_row",cell);
    if(!recipe) {
        GraphBuilder g(p.device);
        const auto ks=g.tensor({"k_now",DType::F32,{rows,c.kv_heads,c.width}}),vs=g.tensor({"v_now",DType::F32,{rows,c.kv_heads,c.width}});
        qsa_cache_write(g,"k_row",ks,rows,c.kv_heads,c.width);qsa_cache_write(g,"v_row",vs,rows,c.kv_heads,c.width);
        recipe=std::make_unique<HostRecipe>(p.device,g.compile("strata_qsa_verify_store_"+std::to_string(rows)),io,std::vector<std::string>{});
    }
    for(const auto& binding:io) recipe->bind_external(binding);
    recipe->launch(stream);
}
void QsaBlock::device_commit_rows(std::span<const float> k,std::span<const float> v,std::span<const float> raw_index_key,uint32_t count) {
    auto& p=*impl_;std::lock_guard lock(p.mutex);const auto& c=p.config;
    const uint64_t row=uint64_t(c.kv_heads)*c.width,index=c.index_width;
    if(!count || k.size()<row*count || v.size()<row*count || raw_index_key.size()<index*count) throw std::invalid_argument("QSA verify commit rows");
    const uint32_t before=p.state.size();
    for(uint32_t r=0;r<count;++r) p.state.append(k.subspan(r*row,row),v.subspan(r*row,row),raw_index_key.subspan(r*index,index),p.weights.index_key_gamma);
    // The window stored every row from `before`; the accepted ones now equal the host pool.
    if(p.cache && p.cache->valid==before) p.cache->valid=before+count;
}
bool QsaBlock::staged_verify_ready(uint32_t rows) const {
    auto& p=*impl_;std::lock_guard lock(p.mutex);const auto& c=p.config;
    return rows && rows<=kResidentWindowRows && rows<=c.capacity-p.state.size();
}
QsaBlock::StagedAttention QsaBlock::stage_rows(std::span<const float> q_gate,std::span<const float> projected_k,
                                               std::span<const float> v,std::span<const float> projected_index,
                                               std::span<const float> index_key,uint32_t rows) {
    auto& p=*impl_;std::lock_guard lock(p.mutex);const auto& c=p.config;
    const uint64_t q=uint64_t(c.query_heads)*c.width,k=uint64_t(c.kv_heads)*c.width,iq=uint64_t(c.index_heads)*c.index_width;
    if(!rows || rows>kResidentWindowRows || rows>c.capacity-p.state.size()) throw std::invalid_argument("QSA staged window row/capacity mismatch");
    const auto check=[](std::span<const float> values,uint64_t count,const char* what) {
        if(values.size()!=count) throw std::invalid_argument(std::string("QSA staged window shape mismatch: ")+what);
        if(!all_finite(values)) throw NonFiniteOutput(std::string("non-finite QSA staged window tensor: ")+what);
    };
    check(q_gate,uint64_t(rows)*q*2,"q_gate");check(projected_k,uint64_t(rows)*k,"key");check(v,uint64_t(rows)*k,"value");
    check(projected_index,uint64_t(rows)*iq,"index_query");check(index_key,uint64_t(rows)*c.index_width,"index_key");
    // An abandoned window (a round that failed before its commit) is dropped first: its rows were already
    // rewound, so only the staged copies remain and the committed prefix is intact.
    if(p.staged && p.staged->active) {
        if(!p.staged->rewound && p.state.size()!=p.staged->start) p.state.restore(p.staged->point);
        p.staged->active=false;
    }
    const auto point=p.state.checkpoint();const auto start=p.state.size();
    auto staged=std::make_unique<Impl::Staged>();
    staged->point=point;staged->start=start;staged->rows=rows;
    try {
        StagedAttention out;out.query.resize(uint64_t(rows)*q);out.gate.resize(out.query.size());
        staged->keys.resize(uint64_t(rows)*k);staged->values.resize(staged->keys.size());
        staged->index_keys.resize(uint64_t(rows)*c.index_width);staged->ids.reserve(rows);
        std::vector<float> key(k),index(iq);uint32_t maximum=0;
        QsaPhaseTimer timer;
        for(uint32_t row=0;row<rows;++row) {
            // The same per-row law as stage_qsa_window(): split q_gate, normalize, rotate at this row's
            // position, append the raw V/index key, then select against the causal prefix.
            auto query=std::span<float>(out.query).subspan(uint64_t(row)*q,q),gate=std::span<float>(out.gate).subspan(uint64_t(row)*q,q);
            for(uint32_t head=0;head<c.query_heads;++head) {
                const auto begin=q_gate.begin()+uint64_t(row)*q*2+uint64_t(head)*2*c.width;
                std::copy_n(begin,c.width,query.begin()+uint64_t(head)*c.width);
                std::copy_n(begin+c.width,c.width,gate.begin()+uint64_t(head)*c.width);
            }
            std::copy_n(projected_k.begin()+uint64_t(row)*k,k,key.begin());
            std::copy_n(projected_index.begin()+uint64_t(row)*iq,iq,index.begin());
            rms_norm_rows(query,c.width,p.weights.query_gamma,c.epsilon);rms_norm_rows(key,c.width,p.weights.key_gamma,c.epsilon);
            rms_norm_rows(index,c.index_width,p.weights.index_query_gamma,c.epsilon);
            const auto rotation=rope_row_at(c.positions,c.rotary,c.position_base+p.state.size(),c.rope);
            rotate_neox(query,c.width,rotation);rotate_neox(key,c.width,rotation);rotate_neox(index,c.index_width,rotation);
            timer.mark("staged.norm-rope");
            const auto value=std::span<const float>(v).subspan(uint64_t(row)*k,k);
            const auto key_row=std::span<const float>(index_key).subspan(uint64_t(row)*c.index_width,c.index_width);
            p.state.append(key,value,key_row,p.weights.index_key_gamma);
            timer.mark("staged.append");
            auto ids=p.state.select(index);
            timer.mark("staged.select");
            for(auto id:ids) if(id<0 || uint32_t(id)>=start+row+1) throw std::logic_error("QSA staged selection escaped causal prefix");
            maximum=std::max(maximum,uint32_t(ids.size()));
            std::copy(key.begin(),key.end(),staged->keys.begin()+uint64_t(row)*k);
            std::copy(value.begin(),value.end(),staged->values.begin()+uint64_t(row)*k);
            std::copy(key_row.begin(),key_row.end(),staged->index_keys.begin()+uint64_t(row)*c.index_width);
            staged->ids.push_back(std::move(ids));
        }
        out.bucket=qsa_attention_bucket(c,maximum);out.scale=1/std::sqrt(float(c.width));
        staged->bucket=out.bucket;staged->active=true;p.staged=std::move(staged);
        return out;
    } catch(...) {if(p.state.size()!=start) p.state.restore(point);throw;}
}
void QsaBlock::gather_row(uint32_t row,std::span<float> key,std::span<float> value,std::span<float> mask) {
    auto& p=*impl_;std::lock_guard lock(p.mutex);const auto& c=p.config;
    if(!p.staged || !p.staged->active || row>=p.staged->rows) throw std::logic_error("QSA gather without a staged window");
    const auto bucket=p.staged->bucket;
    if(mask.size()!=uint64_t(c.kv_heads)*bucket) throw std::invalid_argument("QSA staged mask shape mismatch");
    p.state.gather_into(p.staged->ids[row],bucket,key,value,mask.first(bucket));
    for(uint32_t head=1;head<c.kv_heads;++head) std::copy_n(mask.begin(),bucket,mask.begin()+uint64_t(head)*bucket);
}
void QsaBlock::rewind_staged() {
    auto& p=*impl_;std::lock_guard lock(p.mutex);
    if(!p.staged || !p.staged->active) throw std::logic_error("QSA rewind without a staged window");
    if(!p.staged->rewound) {
        if(p.state.size()!=p.staged->start) p.state.restore(p.staged->point);
        p.staged->rewound=true;
    }
    // The window never wrote the device cache; cells past the committed prefix are not valid there.
    if(p.cache && p.cache->valid>p.staged->start) p.cache->valid=p.staged->start;
}
void QsaBlock::commit_staged(uint32_t count) {
    auto& p=*impl_;std::lock_guard lock(p.mutex);const auto& c=p.config;
    if(!p.staged || !p.staged->active || !count || count>p.staged->rows) throw std::invalid_argument("QSA staged commit rows");
    auto& staged=*p.staged;const uint64_t row=uint64_t(c.kv_heads)*c.width,index=c.index_width;
    if(!staged.rewound && p.state.size()!=staged.start) p.state.restore(staged.point);
    staged.rewound=true;
    for(uint32_t r=0;r<count;++r)
        p.state.append(std::span<const float>(staged.keys).subspan(uint64_t(r)*row,row),
                       std::span<const float>(staged.values).subspan(uint64_t(r)*row,row),
                       std::span<const float>(staged.index_keys).subspan(uint64_t(r)*index,index),p.weights.index_key_gamma);
    if(p.cache && p.cache->valid==staged.start) p.cache->valid=staged.start+count;
    staged.active=false;
}
void QsaBlock::drop_staged() {
    auto& p=*impl_;std::lock_guard lock(p.mutex);
    if(!p.staged) return;
    if(p.staged->active && !p.staged->rewound && p.state.size()!=p.staged->start) p.state.restore(p.staged->point);
    p.staged->active=false;
}
bool QsaBlock::staged_pending() const {auto& p=*impl_;std::lock_guard lock(p.mutex);return p.staged && p.staged->active;}
std::vector<float> QsaBlock::verify_mask(uint32_t cell,uint32_t bucket,uint32_t rows) {
    const uint64_t keys=uint64_t(bucket)+rows;std::vector<float> mask(keys*rows,-INFINITY);
    for(uint32_t j=0;j<rows;++j) {
        for(uint32_t i=0;i<cell && i<bucket;++i) mask[j*keys+i]=0.0f;
        for(uint32_t i=0;i<=j;++i) mask[j*keys+bucket+i]=0.0f;
    }
    return mask;
}
}
