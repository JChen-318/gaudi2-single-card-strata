#include "strata/hpu/recipe.hpp"
#include "strata/hpu/simd_norm.hpp"
#include "strata/hpu/performance_modes.hpp"
#include "strata/hpu/window_limits.hpp"
#include "../tpc/packed_codebook_layout.inc"
#include "strata/hpu/gdn_abi.hpp"
#include "strata/hpu/model_abi.hpp"
#include "strata/hpu/packed_abi.hpp"
#include "strata/hpu/packed_quant.hpp"
#include <perf_lib_layer_params.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <map>
#include <cstdio>
#include <mutex>
#include <cmath>
#include <set>
#include <utility>
#include <unordered_map>
#include <string_view>

namespace strata::hpu {
namespace {std::atomic<uint64_t> launch_total{0};}
uint64_t recipe_launch_count() {return launch_total.load(std::memory_order_relaxed);}
namespace {std::atomic<uint64_t> live_recipes{0},own_workspace{0};}
uint64_t recipe_live_count() {return live_recipes.load(std::memory_order_relaxed);}
uint64_t recipe_own_workspace_bytes() {return own_workspace.load(std::memory_order_relaxed);}
namespace {
synDataType sdk_type(DType type) {
    switch(type) {case DType::F32:return syn_type_float;case DType::BF16:return syn_type_bf16;case DType::U8:return syn_type_uint8;case DType::F8:return syn_type_fp8_143;case DType::I32:return syn_type_int32;case DType::U4:return syn_type_uint4;}
    throw std::invalid_argument("invalid SDK tensor dtype");
}
struct TensorSlot {
    TensorDesc desc;
    synTensor tensor = nullptr;
    synSectionHandle section = nullptr;
};
void release_status(synStatus status, const char* operation) noexcept {
    if (status != synSuccess)
        std::fprintf(stderr, "strata-hpu: %s failed during cleanup (status %d)\n", operation, status);
}
}

// One workspace shared by the recipes compiled under one key (GraphBuilder::compile with shared_workspace).
struct SharedWorkspace {
    std::mutex mutex;std::unique_ptr<DeviceBuffer> buffer;
    static std::shared_ptr<SharedWorkspace> get(const std::string& key) {
        static std::mutex registry_mutex;static std::map<std::string,std::weak_ptr<SharedWorkspace>> registry;
        std::lock_guard lock(registry_mutex);auto& entry=registry[key];
        auto existing=entry.lock();if(!existing) {existing=std::make_shared<SharedWorkspace>();entry=existing;}
        return existing;
    }
    void reserve(const Device& device,uint64_t bytes) {
        std::lock_guard lock(mutex);
        if(buffer && buffer->size()>=bytes) return;
        device.synchronize(); // no launch may still use the old buffer
        buffer.reset();buffer=std::make_unique<DeviceBuffer>(device,std::max<uint64_t>(bytes,128));
    }
};
struct Recipe::Impl {
    Device device;
    synRecipeHandle handle = nullptr;
    std::vector<TensorDesc> tensors;
    std::vector<uint64_t> tensor_ids;
    std::vector<std::string> written;
    std::string name; // the compile name, for STRATA_HPU_RECIPE_STATS
    DeviceBuffer workspace;
    std::shared_ptr<SharedWorkspace> shared;uint64_t shared_bytes=0; // when compiled under a shared workspace key
    uint64_t workspace_address() const {return shared?shared->buffer->address():workspace.address();}
    Event complete;
    std::atomic<uint64_t> launch_count{0};
    std::mutex submit;
    // Opt-in host submit plan. Keys borrow immutable tensor descriptors; no buffer pointer is cached.
    std::unordered_map<std::string_view,size_t> tensor_index;
    std::vector<synLaunchTensorInfo> launch_info;
    std::vector<uint64_t> binding_seen;
    uint64_t binding_epoch=0;
    std::vector<std::string> binding_names;
    std::vector<size_t> binding_order;
    Impl(const Device& d, synRecipeHandle h, std::vector<TensorDesc> desc,
         std::vector<uint64_t> ids, uint64_t bytes)
        : device(d), handle(h), tensors(std::move(desc)), tensor_ids(std::move(ids)),
          workspace(d, bytes), complete(d) {live_recipes+=1;own_workspace+=workspace.size();}
    ~Impl() {
        live_recipes-=1;own_workspace-=workspace.size();
        // Destroying a recipe/workspace cannot race a previously submitted launch.
        try { device.synchronize(); }
        catch (const std::exception& e) { std::fprintf(stderr, "strata-hpu: %s\n", e.what()); }
        if (handle) release_status(synRecipeDestroy(handle), "synRecipeDestroy");
    }
};
Recipe::Recipe(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
Recipe::~Recipe() = default;
const Event& Recipe::completion() const { return impl_->complete; }
uint64_t Recipe::workspace_bytes() const { return impl_->shared?impl_->shared_bytes:impl_->workspace.size(); }
uint64_t Recipe::launches() const { return impl_->launch_count.load(); }
std::span<const TensorDesc> Recipe::tensors() const { return impl_->tensors; }
bool Recipe::produced_by_graph(const std::string& name) const {
    return std::find(impl_->written.begin(), impl_->written.end(), name) != impl_->written.end();
}
const std::string& Recipe::name() const { return impl_->name; }

void Recipe::enqueue(const Stream& stream, std::span<const Binding> bindings) {
    auto& p = *impl_;
    std::lock_guard lock(p.submit);
    if (stream.state() != p.device.state()) throw std::invalid_argument("recipe stream device mismatch");
    if (bindings.size() != p.tensors.size()) throw std::invalid_argument("recipe binding count mismatch");
    if(fast_bindings_enabled()) {
        if(p.launch_info.empty()) {
            p.tensor_index.reserve(p.tensors.size());p.launch_info.resize(p.tensors.size());
            p.binding_seen.resize(p.tensors.size());
            for(size_t i=0;i<p.tensors.size();++i) {
                const auto& desc=p.tensors[i];p.tensor_index.emplace(desc.name,i);
                auto& info=p.launch_info[i];info.tensorName=desc.name.c_str();
                info.tensorType=DATA_TENSOR;info.tensorId=p.tensor_ids[i];
                for(size_t dim=0;dim<desc.shape.size();++dim) info.tensorSize[dim]=desc.shape[desc.shape.size()-1-dim];
            }
        }
        bool same=bindings.size()==p.binding_names.size();
        for(size_t i=0;same && i<bindings.size();++i) same=bindings[i].name==p.binding_names[i];
        if(!same) {
            if(++p.binding_epoch==0) {std::fill(p.binding_seen.begin(),p.binding_seen.end(),0);++p.binding_epoch;}
            std::vector<size_t> order;order.reserve(bindings.size());
            for(const auto& binding:bindings) {
                const auto it=p.tensor_index.find(binding.name);
                if(it==p.tensor_index.end()) throw std::invalid_argument("unexpected recipe binding: "+binding.name);
                if(p.binding_seen[it->second]==p.binding_epoch) throw std::invalid_argument("duplicate recipe binding: "+binding.name);
                p.binding_seen[it->second]=p.binding_epoch;order.push_back(it->second);
            }
            std::vector<std::string> names;names.reserve(bindings.size());
            for(const auto& binding:bindings) names.push_back(binding.name);
            p.binding_order=std::move(order);p.binding_names=std::move(names);
        }
        for(size_t i=0;i<bindings.size();++i) {
            const auto& binding=bindings[i];const auto index=p.binding_order[i];const auto& desc=p.tensors[index];
            if(!binding.buffer) throw std::invalid_argument("missing recipe binding: "+desc.name);
            if(binding.buffer->state()!=p.device.state()) throw std::invalid_argument("binding device mismatch");
            check_range(binding.buffer->size(),binding.offset,desc.bytes());
            const auto address=binding.buffer->address()+binding.offset;
            if(address%128) throw std::invalid_argument("recipe tensor address must be 128-byte aligned");
            p.launch_info[index].pTensorAddress=address;
        }
        if(p.complete.recorded()) stream.wait(p.complete);
        check(synLaunch(stream.handle(),p.launch_info.data(),static_cast<uint32_t>(p.launch_info.size()),
                        p.workspace_address(),p.handle,0),"synLaunch");
        launch_total.fetch_add(1,std::memory_order_relaxed);
        p.complete.record(stream);++p.launch_count;return;
    }
    std::set<std::string> seen;
    std::vector<synLaunchTensorInfo> info(p.tensors.size());
    for (size_t i = 0; i < p.tensors.size(); ++i) {
        const auto& desc = p.tensors[i];
        const Binding* found = nullptr;
        for (const auto& binding : bindings) {
            if (binding.name == desc.name) {
                if (found) throw std::invalid_argument("duplicate recipe binding: " + desc.name);
                found = &binding;
            }
        }
        if (!found || !found->buffer) throw std::invalid_argument("missing recipe binding: " + desc.name);
        if (found->buffer->state() != p.device.state()) throw std::invalid_argument("binding device mismatch");
        check_range(found->buffer->size(), found->offset, desc.bytes());
        auto address = found->buffer->address() + found->offset;
        if (address % 128) throw std::invalid_argument("recipe tensor address must be 128-byte aligned");
        seen.insert(found->name);
        info[i].tensorName = desc.name.c_str();
        info[i].pTensorAddress = address;
        info[i].tensorType = DATA_TENSOR;
        info[i].tensorId = p.tensor_ids[i];
        for (size_t dim = 0; dim < desc.shape.size(); ++dim)
            info[i].tensorSize[dim] = desc.shape[desc.shape.size() - 1 - dim];
    }
    if (seen.size() != bindings.size()) throw std::invalid_argument("unexpected recipe binding");
    // Serialize launches across streams because the recipe owns one workspace.
    if (p.complete.recorded()) stream.wait(p.complete);
    check(synLaunch(stream.handle(), info.data(), static_cast<uint32_t>(info.size()),
                    p.workspace_address(), p.handle, 0), "synLaunch");
    launch_total.fetch_add(1, std::memory_order_relaxed);
    p.complete.record(stream);
    ++p.launch_count;
}

struct GraphBuilder::Impl {
    std::optional<Device> device;
    synGraphHandle graph = nullptr;
    std::vector<TensorSlot> tensors;
    bool compiled = false;
    uint64_t nodes = 0;
    // Every tensor some node writes, so HostRecipe can tell a graph-internal tensor apart
    // from one the caller has to fill.
    std::vector<std::string> written;
    std::string scope;                        // set_scope prefix for new tensor names
    std::map<std::string, TensorId> aliases;  // unprefixed name -> existing tensor, consumed once
    // Per-channel U4 scale/zero-point arrays. synQuantMetadata stores a pointer into these, so the
    // graph must own them until synGraphCompile has read the metadata.
    std::vector<std::vector<synQuantZPScale>> quant;
    explicit Impl(const Device& d) : device(d) { check(synGraphCreate(&graph, synDeviceGaudi2), "synGraphCreate"); }
    Impl() { check(synGraphCreate(&graph, synDeviceGaudi2), "offline synGraphCreate"); }
    ~Impl() {
        for (auto& item : tensors) {
            if (item.tensor) release_status(synTensorDestroy(item.tensor), "synTensorDestroy");
            if (item.section) release_status(synSectionDestroy(item.section), "synSectionDestroy");
        }
        if (graph) release_status(synGraphDestroy(graph), "synGraphDestroy");
    }
    void building() const { if (compiled) throw std::logic_error("graph has already been compiled"); }
    TensorSlot& get(TensorId id) {
        building();
        if (id >= tensors.size()) throw std::out_of_range("unknown graph tensor");
        return tensors[id];
    }
    void node(std::span<const TensorId> input_ids, std::span<const TensorId> output_ids,
              const char* guid, void* parameters, uint32_t bytes, bool require_f32 = false, bool any_bytes = false) {
        std::vector<synTensor> inputs, outputs;
        for (auto id : input_ids) {
            if (id == kNoTensor) {inputs.push_back(nullptr);continue;}
            const auto& slot = get(id);
            if (require_f32 && slot.desc.dtype != DType::F32) throw std::invalid_argument("TPC input must be FP32");
            if(slot.desc.dtype==DType::U8 && !any_bytes && std::string(guid)!="strata_packed_matmul") throw std::invalid_argument("byte tensor requires packed operation");
            inputs.push_back(slot.tensor);
        }
        for (auto id : output_ids) {
            const auto& slot = get(id);
            if (require_f32 && slot.desc.dtype != DType::F32) throw std::invalid_argument("TPC output must be FP32");
            // The packed dot and the device activation quantizer are the operations that legitimately
            // produce bytes; any other U8 output is a mistake in the graph.
            if(slot.desc.dtype==DType::U8 && !any_bytes && std::string(guid)!="reshape" && std::string(guid)!="strata_activation_quant_f32") throw std::invalid_argument("byte output requires explicit supported operation");
            written.push_back(slot.desc.name);
            outputs.push_back(slot.tensor);
        }
        const auto name = std::string(guid) + "_" + std::to_string(nodes);
        check(synNodeCreate(graph, inputs.data(), outputs.data(), inputs.size(), outputs.size(),
                            parameters, bytes, guid, name.c_str(), nullptr, nullptr), guid);
        ++nodes;
    }
};
GraphBuilder::GraphBuilder(const Device& device) : impl_(std::make_unique<Impl>(device)) {}
GraphBuilder::GraphBuilder(OfflineCompilation) : impl_(std::make_unique<Impl>()) {}
GraphBuilder::~GraphBuilder() = default;
void GraphBuilder::set_scope(std::string prefix) { impl_->building(); impl_->scope = std::move(prefix); }
const TensorDesc& GraphBuilder::desc(TensorId id) const {return impl_->get(id).desc;}
TensorId GraphBuilder::find(const std::string& name) const {
    const auto& tensors = impl_->tensors;
    for (size_t i = 0; i < tensors.size(); ++i) if (tensors[i].desc.name == name) return static_cast<TensorId>(i);
    throw std::invalid_argument("unknown graph tensor: " + name);
}
void GraphBuilder::alias(const std::string& name, TensorId existing) {
    auto& p = *impl_;
    (void)p.get(existing);
    if (!p.aliases.emplace(name, existing).second) throw std::invalid_argument("duplicate graph alias: " + name);
}
TensorId GraphBuilder::tensor(TensorDesc desc) {
    auto& p = *impl_;
    p.building();
    if (const auto found = p.aliases.find(desc.name); found != p.aliases.end()) {
        const auto id = found->second;
        const auto& existing = p.get(id).desc;
        if (existing.dtype != desc.dtype || existing.shape != desc.shape)
            throw std::invalid_argument("graph alias dtype/shape mismatch: " + desc.name + " -> " + existing.name);
        p.aliases.erase(found);
        return id;
    }
    desc.name = p.scope + desc.name;
    (void)desc.bytes();
    for (const auto& slot : p.tensors)
        if (slot.desc.name == desc.name) throw std::invalid_argument("duplicate tensor name");
    p.tensors.push_back({std::move(desc), nullptr, nullptr});
    auto& slot = p.tensors.back();
    check(synTensorHandleCreate(&slot.tensor, p.graph, DATA_TENSOR, slot.desc.name.c_str()), "synTensorHandleCreate");
    synTensorGeometry geometry{};
    geometry.dims = static_cast<uint32_t>(slot.desc.shape.size());
    for (size_t dim = 0; dim < slot.desc.shape.size(); ++dim)
        geometry.sizes[dim] = slot.desc.shape[slot.desc.shape.size() - 1 - dim];
    check(synTensorSetGeometry(slot.tensor, &geometry, synGeometrySizes), "synTensorSetGeometry");
    check(synTensorSetDeviceDataType(slot.tensor, sdk_type(slot.desc.dtype)), "synTensorSetDeviceDataType");
    if (slot.desc.persistent) {
        check(synSectionCreate(&slot.section, 0, p.graph), "synSectionCreate");
        check(synSectionSetPersistent(slot.section, true), "synSectionSetPersistent");
        check(synTensorAssignToSection(slot.tensor, slot.section, 0), "synTensorAssignToSection");
    }
    return static_cast<TensorId>(p.tensors.size() - 1);
}
void GraphBuilder::gemm(TensorId a_id, TensorId b_id, TensorId out_id, bool transpose_b) {
    auto& p = *impl_;
    const auto& a = p.get(a_id);
    const auto& b = p.get(b_id);
    const auto& out = p.get(out_id);
    // A U4 weight is decompressed by the MME, so it is the one case where the two inputs differ in dtype.
    const bool u4_weight = b.desc.dtype == DType::U4;
    const bool input_ok = u4_weight ? (a.desc.dtype == DType::BF16 || a.desc.dtype == DType::F32)
                                    : a.desc.dtype == b.desc.dtype;
    const bool output_ok = u4_weight ? (out.desc.dtype == DType::F32 || out.desc.dtype == DType::BF16)
                                     : out.desc.dtype == a.desc.dtype ||
        (a.desc.dtype == DType::BF16 && out.desc.dtype == DType::F32) ||
        (a.desc.dtype == DType::F8 && (out.desc.dtype == DType::F32 || out.desc.dtype == DType::BF16));
    if (a.desc.dtype==DType::U8 || out.desc.dtype==DType::F8 || out.desc.dtype==DType::U4 ||
        !input_ok || !output_ok)
        throw std::invalid_argument("GEMM requires matching inputs and same/F32 accumulation output");
    if (u4_weight && !transpose_b)
        throw std::invalid_argument("U4 weight GEMM requires a [N,K] weight (transpose_b)");
    if (a.desc.shape.size() != 2 || b.desc.shape.size() != 2 || out.desc.shape.size() != 2)
        throw std::invalid_argument("GEMM requires rank-2 tensors");
    const uint64_t n = b.desc.shape[transpose_b ? 0 : 1];
    if (a.desc.shape[1] != b.desc.shape[transpose_b ? 1 : 0] ||
        out.desc.shape != std::vector<uint64_t>{a.desc.shape[0], n})
        throw std::invalid_argument("GEMM shape mismatch");
    // STRATA_HPU_GEMV_TPC: one BF16 row against a [N, K] weight streams faster on the TPC (tpc/gemv_bf16.c).
    const uint64_t k = a.desc.shape[1];
    if (gemv_tpc_enabled() && transpose_b && a.desc.shape[0] == 1 && a.desc.dtype == DType::BF16 &&
        b.desc.dtype == DType::BF16 && (out.desc.dtype == DType::BF16 || out.desc.dtype == DType::F32) &&
        k % 128 == 0 && k <= 8192 && n <= 65536) {
        const abi::GemvBf16 args{int32_t(k), int32_t(n), out.desc.dtype == DType::BF16 ? 1 : 0};
        const TensorId in[] = {a_id, b_id}, o[] = {out_id};
        op("strata_gemv_bf16", in, o, &args, sizeof(args));
        return;
    }
    const synTensor inputs[] = {a.tensor, b.tensor};
    const synTensor outputs[] = {out.tensor};
    synGEMMParams params(false, transpose_b);
    const auto node_name = "strata_gemm_" + std::to_string(p.nodes);
    check(synNodeCreate(p.graph, inputs, outputs, 2, 1, &params, sizeof(params),
                        "gemm", node_name.c_str(), nullptr, nullptr), "synNodeCreate gemm");
    p.written.push_back(out.desc.name);
    ++p.nodes;
}
void GraphBuilder::quantize_per_channel(TensorId weight, std::span<const float> scales, std::span<const float> zero_points) {
    auto& p = *impl_;
    const auto& w = p.get(weight).desc;
    if (w.dtype != DType::U4 || w.shape.size() != 2 || w.shape[0] != scales.size())
        throw std::invalid_argument("per-channel quantization needs a U4 [N,K] weight and N scales");
    if (!zero_points.empty() && zero_points.size() != scales.size())
        throw std::invalid_argument("per-channel zero points must match the scale count");
    std::vector<synQuantZPScale> values(scales.size());
    for (size_t channel = 0; channel < scales.size(); ++channel) {
        const double zp = zero_points.empty() ? 0.0 : double(zero_points[channel]);
        if (!std::isfinite(scales[channel]) || scales[channel] <= 0 || !std::isfinite(zp))
            throw std::invalid_argument("invalid per-channel scale/zero point");
        values[channel] = synQuantZPScale{zp, double(scales[channel])};
    }
    p.quant.push_back(std::move(values));
    auto& kept = p.quant.back();
    synQuantMetadata metadata{syn_type_uint4, kept.data(), static_cast<unsigned>(kept.size())};
    check(synTensorSetQuantizationData(p.get(weight).tensor, SYN_QUANT_METADATA, &metadata, sizeof(metadata)),
          "synTensorSetQuantizationData");
}
void GraphBuilder::quantize_uniform(TensorId weight, float scale, float zero_point) {
    auto& p = *impl_;
    const auto& w = p.get(weight).desc;
    if (w.dtype != DType::U4)
        throw std::invalid_argument("uniform quantization needs a U4 tensor");
    if (!std::isfinite(scale) || scale <= 0 || !std::isfinite(zero_point))
        throw std::invalid_argument("invalid uniform scale/zero point");
    p.quant.emplace_back(1, synQuantZPScale{double(zero_point), double(scale)});
    auto& kept = p.quant.back();
    synQuantMetadata metadata{syn_type_uint4, kept.data(), 1};
    check(synTensorSetQuantizationData(p.get(weight).tensor, SYN_QUANT_METADATA, &metadata, sizeof(metadata)),
          "synTensorSetQuantizationData");
}
void GraphBuilder::quantize_table(TensorId weight, std::span<const float> zero_points, std::span<const float> scales) {
    auto& p = *impl_;
    const auto& w = p.get(weight).desc;
    if (w.dtype != DType::U4 || w.shape.size() < 2) throw std::invalid_argument("table quantization needs a U4 weight");
    if (zero_points.size() != scales.size() || scales.empty())
        throw std::invalid_argument("table quantization needs one zero point per scale");
    std::vector<synQuantZPScale> values(scales.size());
    for (size_t channel = 0; channel < scales.size(); ++channel) {
        if (!std::isfinite(scales[channel]) || scales[channel] <= 0 || !std::isfinite(zero_points[channel]))
            throw std::invalid_argument("invalid table scale/zero point");
        values[channel] = synQuantZPScale{double(zero_points[channel]), double(scales[channel])};
    }
    p.quant.push_back(std::move(values));
    auto& kept = p.quant.back();
    synQuantMetadata metadata{syn_type_uint4, kept.data(), static_cast<unsigned>(kept.size())};
    check(synTensorSetQuantizationData(p.get(weight).tensor, SYN_QUANT_METADATA, &metadata, sizeof(metadata)),
          "synTensorSetQuantizationData");
}
void GraphBuilder::batch_gemm(TensorId aid,TensorId bid,TensorId oid,bool trans_b) {
    auto& p=*impl_;
    const auto& a=p.get(aid).desc; const auto& b=p.get(bid).desc; const auto& o=p.get(oid).desc;
    // A U4 weight is decompressed by the MME; as with gemm it is the one case where A and B differ in dtype.
    const bool u4_weight=b.dtype==DType::U4;
    const bool input_ok=u4_weight?(a.dtype==DType::BF16 || a.dtype==DType::F32):a.dtype==b.dtype;
    const bool output_ok=u4_weight?(o.dtype==DType::F32 || o.dtype==DType::BF16):
        o.dtype==a.dtype || (a.dtype==DType::BF16 && o.dtype==DType::F32) ||
        (a.dtype==DType::F8 && (o.dtype==DType::F32 || o.dtype==DType::BF16));
    if(a.shape.size()!=3 || b.shape.size()!=3 || o.shape.size()!=3 || a.shape[0]!=b.shape[0] ||
       !input_ok || !output_ok || o.dtype==DType::F8)
        throw std::invalid_argument("batch GEMM rank/dtype/batch mismatch");
    if(u4_weight && !trans_b)
        throw std::invalid_argument("U4 weight batch GEMM requires a [batch,N,K] weight (transpose_b)");
    const auto n=b.shape[trans_b?1:2],k=b.shape[trans_b?2:1];
    if(a.shape[2]!=k || o.shape!=std::vector<uint64_t>{a.shape[0],a.shape[1],n})
        throw std::invalid_argument("batch GEMM matrix geometry mismatch");
    synGEMMParams parameters(false,trans_b);
    const TensorId inputs[]={aid,bid},outputs[]={oid};
    p.node(inputs,outputs,"batch_gemm",&parameters,sizeof(parameters));
}
void GraphBuilder::packed_matmul(TensorId wid,TensorId aid,TensorId mid,TensorId cid,TensorId oid,uint32_t columns,uint32_t type,PackedActivation mode) {
    auto& p=*impl_;const auto& w=p.get(wid).desc;const auto& a=p.get(aid).desc;const auto& m=p.get(mid).desc;const auto& c=p.get(cid).desc;const auto& o=p.get(oid).desc;
    const auto row_bytes=packed_row_bytes(type,columns);
    if(!packed_activation_supported(type,mode)) throw std::invalid_argument("invalid packed activation contract/type");
    if(a.dtype!=DType::U8 || w.dtype!=DType::U8 || m.dtype!=DType::F32 || o.dtype!=DType::F32 ||
       c.dtype!=DType::U8 || c.shape!=std::vector<uint64_t>{1,STRATA_PACKED_CODEBOOK_BYTES} ||
       a.shape.size()!=2 || w.shape.size()!=2 || !a.shape[0] || a.shape[0]>128 || !w.shape[0] || w.shape[0]>1048576 ||
       a.shape[1]!=columns || w.shape[1]!=row_bytes || m.shape!=std::vector<uint64_t>{a.shape[0],columns/32,2} ||
       o.shape!=std::vector<uint64_t>{a.shape[0],w.shape[0]}) throw std::invalid_argument("invalid packed matmul tensor geometry/dtype");
    abi::PackedMatmul parameters{int32_t(columns),int32_t(w.shape[0]),int32_t(a.shape[0]),int32_t(type),int32_t(mode)};
    const TensorId inputs[]={wid,aid,mid,cid},outputs[]={oid};p.node(inputs,outputs,"strata_packed_matmul",&parameters,sizeof(parameters));
}
void GraphBuilder::activation_quant(TensorId input,TensorId codes,TensorId metadata,uint32_t batch,uint32_t columns) {
    auto& p=*impl_;const auto& x=p.get(input).desc;const auto& c=p.get(codes).desc;const auto& m=p.get(metadata).desc;
    if(x.dtype!=DType::F32||c.dtype!=DType::U8||m.dtype!=DType::F32||columns%32||
       x.shape!=std::vector<uint64_t>{batch,columns}||c.shape!=std::vector<uint64_t>{batch,columns}||
       m.shape!=std::vector<uint64_t>{batch,columns/32,2})
        throw std::invalid_argument("invalid activation quantizer tensor geometry/dtype");
    abi::ActivationQuant parameters{int32_t(columns),int32_t(batch)};
    const TensorId inputs[]={input},outputs[]={codes,metadata};
    // require_f32 stays off: the codes output is U8 and the dtypes are checked above.
    p.node(inputs,outputs,"strata_activation_quant_f32",&parameters,sizeof(parameters));
}
void GraphBuilder::add(TensorId aid,TensorId bid,TensorId oid) {
    auto& p=*impl_;
    const auto& a=p.get(aid).desc; const auto& b=p.get(bid).desc; const auto& o=p.get(oid).desc;
    if(a.dtype!=b.dtype || a.dtype!=o.dtype || a.shape!=o.shape || b.shape.size()>a.shape.size())
        throw std::invalid_argument("add dtype/rank/output mismatch");
    for(size_t i=0;i<b.shape.size();++i)
        if(b.shape[b.shape.size()-1-i]!=1 && b.shape[b.shape.size()-1-i]!=a.shape[a.shape.size()-1-i])
            throw std::invalid_argument("add broadcast mismatch");
    const TensorId inputs[]={aid,bid},outputs[]={oid};
    p.node(inputs,outputs,a.dtype==DType::F32?"add_fwd_f32":"add_fwd_bf16",nullptr,0);
}
void GraphBuilder::softmax_last(TensorId input,TensorId output) {
    auto& p=*impl_;
    const auto& a=p.get(input).desc; const auto& b=p.get(output).desc;
    if(a.dtype!=b.dtype || a.shape!=b.shape) throw std::invalid_argument("softmax geometry mismatch");
    ns_Softmax::Params parameters{};
    parameters.dim=0; // Logical last axis is physical SDK axis zero.
    const TensorId inputs[]={input},outputs[]={output};
    p.node(inputs,outputs,a.dtype==DType::F32?"softmax_fwd_f32":"softmax_fwd_bf16",&parameters,sizeof(parameters));
}
void GraphBuilder::multiply(TensorId a_id, TensorId b_id, TensorId out_id) {
    auto& p = *impl_;
    const auto& a = p.get(a_id);
    const auto& b = p.get(b_id);
    const auto& out = p.get(out_id);
    if (a.desc.dtype != b.desc.dtype || a.desc.dtype != out.desc.dtype || out.desc.shape != a.desc.shape ||
        (b.desc.shape != a.desc.shape && b.desc.shape != std::vector<uint64_t>{1}))
        throw std::invalid_argument("multiply requires matching tensors or a scalar multiplier");
    const synTensor inputs[] = {a.tensor, b.tensor};
    const synTensor outputs[] = {out.tensor};
    const char* guid = a.desc.dtype == DType::F32 ? "mult_fwd_f32" : "mult_fwd_bf16";
    const auto node_name = "strata_multiply_" + std::to_string(p.nodes);
    check(synNodeCreate(p.graph, inputs, outputs, 2, 1, nullptr, 0, guid,
                        node_name.c_str(), nullptr, nullptr), "synNodeCreate multiply");
    p.written.push_back(out.desc.name);
    ++p.nodes;
}
void GraphBuilder::sigmoid(TensorId input_id, TensorId output_id) {
    auto& p = *impl_;
    const auto& input = p.get(input_id);
    const auto& output = p.get(output_id);
    if (input.desc.dtype==DType::U8 || input.desc.dtype != output.desc.dtype || input.desc.shape != output.desc.shape)
        throw std::invalid_argument("sigmoid shape/dtype mismatch");
    const auto node_name = "strata_sigmoid_" + std::to_string(p.nodes);
    const char* guid = input.desc.dtype == DType::F32 ? "sigmoid_fwd_f32" : "sigmoid_fwd_bf16";
    check(synNodeCreate(p.graph, &input.tensor, &output.tensor, 1, 1, nullptr, 0, guid,
                        node_name.c_str(), nullptr, nullptr), "synNodeCreate sigmoid");
    p.written.push_back(output.desc.name);
    ++p.nodes;
}
void GraphBuilder::cast(TensorId input, TensorId output) {
    auto& p = *impl_;
    const auto& a = p.get(input).desc;
    const auto& b = p.get(output).desc;
    if (a.dtype==DType::U8 || b.dtype==DType::U8 || a.dtype==DType::U4 || b.dtype==DType::U4 ||
        a.shape != b.shape || a.dtype == b.dtype)
        throw std::invalid_argument("cast requires float tensors with matching shape and distinct dtypes");
    const TensorId inputs[] = {input}, outputs[] = {output};
    const char* guid = a.dtype == DType::F8 ? (b.dtype == DType::F32 ? "cast_hf8_to_f32" : "cast_hf8_to_bf16")
                     : b.dtype == DType::F8 ? (a.dtype == DType::F32 ? "cast_f32_to_hf8" : "cast_bf16_to_hf8")
                     : a.dtype == DType::I32 ? "cast_i32_to_f32"
                     : a.dtype == DType::BF16 ? "cast_bf16_to_f32" : "cast_f32_to_bf16";
    p.node(inputs, outputs, guid, nullptr, 0);
}
void GraphBuilder::op(const char* guid, std::span<const TensorId> inputs, std::span<const TensorId> outputs,
                      const void* parameters, uint32_t bytes) {
    if (!guid || !*guid || outputs.empty()) throw std::invalid_argument("graph op needs a guid and an output");
    // An explicit guid owns its dtypes, bytes included (the packed-only byte guard is for the typed helpers).
    impl_->node(inputs, outputs, guid, const_cast<void*>(parameters), bytes, false, true);
}
void GraphBuilder::concat(std::span<const TensorId> inputs, TensorId output, uint32_t axis) {
    auto& p = *impl_;
    const auto out = p.get(output).desc;
    if (inputs.empty() || axis >= out.shape.size()) throw std::invalid_argument("concat needs inputs and an axis inside the output rank");
    uint64_t total = 0;
    for (const auto id : inputs) {
        const auto& in = p.get(id).desc;
        if (in.dtype != out.dtype || in.shape.size() != out.shape.size()) throw std::invalid_argument("concat dtype/rank mismatch");
        for (size_t d = 0; d < in.shape.size(); ++d)
            if (d != axis && in.shape[d] != out.shape[d]) throw std::invalid_argument("concat shape mismatch off the axis");
        total += in.shape[axis];
    }
    if (total != out.shape[axis]) throw std::invalid_argument("concat axis length mismatch");
    synConcatenateParams params{};
    params.axis = static_cast<unsigned>(out.shape.size() - 1 - axis); // logical axis -> SDK dimension
    const TensorId outputs[] = {output};
    p.node(inputs, outputs, "concat", &params, sizeof(params));
}
void GraphBuilder::fp8_linear(TensorId input, TensorId weight, TensorId row_scale, TensorId output, bool one_pass) {
    auto& p = *impl_;
    const auto x = p.get(input).desc, w = p.get(weight).desc, s = p.get(row_scale).desc, o = p.get(output).desc;
    if (x.dtype != DType::F32 || w.dtype != DType::F8 || s.dtype != DType::F32 || o.dtype != DType::F32 ||
        x.shape.size() != 2 || w.shape.size() != 2 || w.shape[1] != x.shape[1] ||
        s.shape != std::vector<uint64_t>{1, w.shape[0]} || o.shape != std::vector<uint64_t>{x.shape[0], w.shape[0]})
        throw std::invalid_argument("FP8 linear requires F32 x[b,K], F8 w[N,K], F32 scale[1,N], F32 out[b,N]");
    const uint64_t rows = x.shape[0], columns = x.shape[1];
    if(one_pass && fp8_static_enabled()) {
        if(columns%64 || columns>65536 || rows>64) throw std::invalid_argument("static FP8 linear geometry");
        const auto codes=tensor({o.name+"_fp8_fixed",DType::F8,{rows,columns},false});
        // Gaudi2 exponent bias11 represents each raw FP8 code times1/16. Set before node insertion.
        synFpQuantParam parameter{1.0,11};synFpQuantMetadata metadata{syn_type_fp8_143,&parameter,1};
        check(synTensorSetQuantizationData(p.get(codes).tensor,SYN_FP_QUANT_METADATA,&metadata,sizeof(metadata)),"static FP8 activation bias");
        check(synTensorSetQuantizationData(p.get(weight).tensor,SYN_FP_QUANT_METADATA,&metadata,sizeof(metadata)),"static FP8 weight bias");
        abi::Fp8Encode encode{int32_t(columns),int32_t(rows)};
        const TensorId in[]={input},out[]={codes};p.node(in,out,"strata_fp8_encode_fixed_f32",&encode,sizeof(encode));
        gemm(codes,weight,output,true);
        return;
    }
    if (one_pass) {
        // Three nodes: the activation's hi and lo codes stacked by one TPC kernel (tpc/fp8_encode_stacked.c), one FP8
        // GEMM reading the weight once for both, and the combination and both scales in one TPC kernel
        // (tpc/fp8_linear_finish.c). The SDK version below costs about fifteen small nodes per projection.
        if (columns % 64 || columns > 65536 || rows > 64) throw std::invalid_argument("one-pass FP8 linear needs columns % 64 == 0 and <= 64 rows");
        const auto stacked = tensor({o.name + "_fp8_stacked", DType::F8, {2 * rows, columns}, false});
        const auto scale = tensor({o.name + "_fp8_scale", DType::F32, {rows, 1}, false});
        const auto both = tensor({o.name + "_fp8_both", DType::F32, {2 * rows, w.shape[0]}, false});
        abi::Fp8Encode encode{int32_t(columns), int32_t(rows)}, finish{int32_t(w.shape[0]), int32_t(rows)};
        { const TensorId in[] = {input}, out[] = {stacked, scale}; p.node(in, out, "strata_fp8_encode_stacked_f32", &encode, sizeof(encode)); }
        gemm(stacked, weight, both, true);
        { const TensorId in[] = {both, scale, row_scale}, out[] = {output}; p.node(in, out, "strata_fp8_linear_finish_f32", &finish, sizeof(finish)); }
        return;
    }
    // Per-row dynamic activation scale: x is divided by max|x|/240 so it fits the FP8 range. The scaled row
    // is carried as two FP8 terms, hi = fp8(x) and lo = fp8(16 (x - hi)), multiplied on the MME against the
    // per-output-row FP8 weights as hi.w + (lo.w)/16; the two scales are applied to the F32 result. The
    // second term removes the activation's own FP8 rounding: on real IQ3/IQ4 experts the relative error
    // against the dequantized GGUF weights drops from 3.4-3.9% to the 2.2-2.7% the FP8 weights alone
    // cause. The epsilon keeps an all-zero row finite.
    const auto name = [&](const char* part) { return o.name + "_fp8_" + part; };
    const auto internal = [&](const char* part, DType dtype, std::vector<uint64_t> shape) {
        return tensor({name(part), dtype, std::move(shape), false});
    };
    const auto magnitude = internal("abs", DType::F32, {rows, columns}), peak = internal("peak", DType::F32, {rows, 1});
    const auto inverse_range = internal("inverse_range", DType::F32, {1}), epsilon = internal("epsilon", DType::F32, {1});
    const auto scaled_peak = internal("scaled_peak", DType::F32, {rows, 1}), scale = internal("scale", DType::F32, {rows, 1});
    // The hi codes are persistent on purpose: as a graph-internal tensor the compiler folds the
    // F32 -> FP8 -> F32 round trip below into an identity (measured: the residual came back zero), which
    // would silently drop the second term. A bound tensor keeps the rounding.
    const auto scaled = internal("scaled", DType::F32, {rows, columns});
    const auto codes = tensor({name("codes"), DType::F8, {rows, columns}});
    const auto high = internal("high", DType::F32, {rows, columns}), residual = internal("residual", DType::F32, {rows, columns});
    const auto sixteen = internal("sixteen", DType::F32, {1}), sixteenth = internal("sixteenth", DType::F32, {1});
    const auto lifted = internal("lifted", DType::F32, {rows, columns}), low_codes = internal("low_codes", DType::F8, {rows, columns});
    const auto main_product = internal("main_product", DType::F32, {rows, w.shape[0]});
    const auto low_product = internal("low_product", DType::F32, {rows, w.shape[0]});
    const auto low_scaled = internal("low_scaled", DType::F32, {rows, w.shape[0]});
    const auto product = internal("product", DType::F32, {rows, w.shape[0]}), rescaled = internal("rescaled", DType::F32, {rows, w.shape[0]});
    { const TensorId in[] = {input}, out[] = {magnitude}; p.node(in, out, "abs_fwd_f32", nullptr, 0); }
    { ns_Reduction::Params params{}; params.reductionDimension = 0;
      const TensorId in[] = {magnitude}, out[] = {peak}; p.node(in, out, "reduce_max_fwd_f32", &params, sizeof(params)); }
    { ns_ConstantKernel::Params params{}; params.constant.f = 1.0f / 240.0f;
      const TensorId out[] = {inverse_range}; p.node({}, out, "constant_f32", &params, sizeof(params)); }
    { ns_ConstantKernel::Params params{}; params.constant.f = 1e-30f;
      const TensorId out[] = {epsilon}; p.node({}, out, "constant_f32", &params, sizeof(params)); }
    { const TensorId in[] = {peak, inverse_range}, out[] = {scaled_peak}; p.node(in, out, "mult_fwd_f32", nullptr, 0); }
    { const TensorId in[] = {scaled_peak, epsilon}, out[] = {scale}; p.node(in, out, "add_fwd_f32", nullptr, 0); }
    { const TensorId in[] = {input, scale}, out[] = {scaled}; p.node(in, out, "div_fwd_f32", nullptr, 0); }
    { ns_ConstantKernel::Params params{}; params.constant.f = 16.0f;
      const TensorId out[] = {sixteen}; p.node({}, out, "constant_f32", &params, sizeof(params)); }
    { ns_ConstantKernel::Params params{}; params.constant.f = 1.0f / 16.0f;
      const TensorId out[] = {sixteenth}; p.node({}, out, "constant_f32", &params, sizeof(params)); }
    cast(scaled, codes);
    cast(codes, high);
    { const TensorId in[] = {scaled, high}, out[] = {residual}; p.node(in, out, "sub_fwd_f32", nullptr, 0); }
    { const TensorId in[] = {residual, sixteen}, out[] = {lifted}; p.node(in, out, "mult_fwd_f32", nullptr, 0); }
    cast(lifted, low_codes);
    gemm(codes, weight, main_product, true);
    gemm(low_codes, weight, low_product, true);
    { const TensorId in[] = {low_product, sixteenth}, out[] = {low_scaled}; p.node(in, out, "mult_fwd_f32", nullptr, 0); }
    { const TensorId in[] = {main_product, low_scaled}, out[] = {product}; p.node(in, out, "add_fwd_f32", nullptr, 0); }
    { const TensorId in[] = {product, scale}, out[] = {rescaled}; p.node(in, out, "mult_fwd_f32", nullptr, 0); }
    { const TensorId in[] = {rescaled, row_scale}, out[] = {output}; p.node(in, out, "mult_fwd_f32", nullptr, 0); }
}
void GraphBuilder::split_rows(TensorId input,std::span<const TensorId> outputs) {
    auto& p=*impl_;const auto& in=p.get(input).desc;
    if(in.dtype==DType::U8 || in.shape.size()!=2 || outputs.empty() || outputs.size()!=in.shape[0])
        throw std::invalid_argument("split rows requires one output per matrix row");
    std::set<TensorId> seen;
    for(auto id:outputs) {
        const auto& out=p.get(id).desc;
        if(id==input || !seen.insert(id).second || out.dtype!=in.dtype || out.shape!=std::vector<uint64_t>{1,in.shape[1]})
            throw std::invalid_argument("split row output dtype/shape/alias mismatch");
    }
    if(outputs.size()==1) {reshape(input,outputs.front());return;}
    synSplitParams params{};params.axis=1; // logical row axis is SDK dimension1
    const TensorId inputs[]={input};p.node(inputs,outputs,"split",&params,sizeof(params));
}
void GraphBuilder::concatenate_rows(std::span<const TensorId> inputs,TensorId output) {
    auto& p=*impl_;const auto& out=p.get(output).desc;
    if(out.dtype==DType::U8 || out.shape.size()!=2 || inputs.empty() || inputs.size()!=out.shape[0])
        throw std::invalid_argument("concatenate rows requires one input per matrix row");
    std::set<TensorId> seen;
    for(auto id:inputs) {
        const auto& in=p.get(id).desc;
        if(id==output || !seen.insert(id).second || in.dtype!=out.dtype || in.shape!=std::vector<uint64_t>{1,out.shape[1]})
            throw std::invalid_argument("concatenate row input dtype/shape/alias mismatch");
    }
    if(inputs.size()==1) {reshape(inputs.front(),output);return;}
    synConcatenateParams params{};params.axis=1;
    const TensorId outputs[]={output};p.node(inputs,outputs,"concat",&params,sizeof(params));
}
void GraphBuilder::reshape(TensorId input, TensorId output) {
    auto& p = *impl_;
    const auto& a = p.get(input).desc;
    const auto& b = p.get(output).desc;
    if (a.dtype != b.dtype || a.bytes() != b.bytes()) throw std::invalid_argument("reshape size/dtype mismatch");
    const TensorId inputs[] = {input}, outputs[] = {output};
    p.node(inputs, outputs, "reshape", nullptr, 0);
}
void GraphBuilder::row_norm(TensorId input,TensorId gamma,TensorId output,float epsilon) {
    auto& p=*impl_;const auto& s=p.get(input).desc.shape;const auto& g=p.get(gamma).desc.shape;
    const bool repeated=s.size()==2 && g.size()==2 && g[1]==s[1] && g[0] && s[0]%g[0]==0;
    if(s.size()!=2 || s[0]>65536 || s[1]>65536 || s[1]%64 || p.get(output).desc.shape!=s ||
       (g!=s && g!=std::vector<uint64_t>{s[1]} && !repeated) || !std::isfinite(epsilon) || epsilon<=0)
        throw std::invalid_argument("row norm geometry/epsilon mismatch");
    abi::RowNorm args{int32_t(s[1]),int32_t(s[0]),g.size()==1?1:g==s?0:-int32_t(g[0]),epsilon};
    const TensorId in[]={input,gamma},out[]={output};
    p.node(in,out,simd_norm_enabled()?"strata_row_norm_simd_f32":"strata_row_norm_f32",&args,sizeof(args),true);
}
void GraphBuilder::row_norm_dual(TensorId input,TensorId gamma,TensorId output,TensorId rounded,float epsilon) {
    auto& p=*impl_;const auto& s=p.get(input).desc.shape;const auto& g=p.get(gamma).desc.shape;const auto& b=p.get(rounded).desc;
    const bool repeated=s.size()==2 && g.size()==2 && g[1]==s[1] && g[0] && s[0]%g[0]==0;
    if(s.size()!=2 || s[0]>65536 || s[1]>65536 || s[1]%64 || p.get(output).desc.shape!=s || b.shape!=s || b.dtype!=DType::BF16 ||
       (g!=s && g!=std::vector<uint64_t>{s[1]} && !repeated) || !std::isfinite(epsilon) || epsilon<=0)
        throw std::invalid_argument("dual row norm geometry/epsilon mismatch");
    abi::RowNorm args{int32_t(s[1]),int32_t(s[0]),g.size()==1?1:g==s?0:-int32_t(g[0]),epsilon};
    const TensorId in[]={input,gamma},out[]={output,rounded};
    p.node(in,out,row_norm_fast_enabled()?"strata_row_norm_dual_fast_f32":"strata_row_norm_dual_f32",&args,sizeof(args),false); // the BF16 output is checked above
}
void GraphBuilder::hc_mix_window_dual(TensorId normalized,TensorId gate,TensorId output,TensorId rounded) {
    auto& p=*impl_;const auto& s=p.get(normalized).desc.shape;const auto& b=p.get(rounded).desc;
    if(s.size()!=3 || s[0]>kResidentWindowRows || s[1]>32 || s[2]>65536 || s[2]%64 || p.get(gate).desc.shape!=s ||
       p.get(output).desc.shape!=std::vector<uint64_t>{s[0],s[2]} || b.shape!=std::vector<uint64_t>{s[0],s[2]} || b.dtype!=DType::BF16)
        throw std::invalid_argument("dual HC window mix geometry mismatch");
    abi::HcWindow args{int32_t(s[2]),int32_t(s[1]),int32_t(s[0])};const TensorId in[]={normalized,gate},out[]={output,rounded};
    p.node(in,out,fast_sigmoid_enabled()?"strata_hc_mix_window_dual_fast_f32":"strata_hc_mix_window_dual_f32",&args,sizeof(args),false); // the BF16 output is checked above
}
void GraphBuilder::hc_mix(TensorId normalized,TensorId gate,TensorId output) {
    auto& p=*impl_;const auto& s=p.get(normalized).desc.shape;
    if(s.size()!=2 || s[0]>32 || s[1]>65536 || s[1]%64 || p.get(gate).desc.shape!=s || p.get(output).desc.shape!=std::vector<uint64_t>{s[1]})
        throw std::invalid_argument("HC mix geometry mismatch");
    abi::Hc args{int32_t(s[1]),int32_t(s[0])};const TensorId in[]={normalized,gate},out[]={output};
    p.node(in,out,"strata_hc_mix_f32",&args,sizeof(args),true);
}
void GraphBuilder::hc_write(TensorId residual,TensorId block,TensorId inject,TensorId output) {
    auto& p=*impl_;const auto& s=p.get(residual).desc.shape;
    if(s.size()!=2 || s[0]>32 || s[1]>65536 || s[1]%64 || p.get(block).desc.shape!=std::vector<uint64_t>{s[1]} ||
       p.get(inject).desc.shape!=std::vector<uint64_t>{s[0]} || p.get(output).desc.shape!=s || residual==output)
        throw std::invalid_argument("HC write geometry/alias mismatch");
    abi::Hc args{int32_t(s[1]),int32_t(s[0])};const TensorId in[]={residual,block,inject},out[]={output};
    p.node(in,out,"strata_hc_write_f32",&args,sizeof(args),true);
}
void GraphBuilder::hc_mix_window(TensorId normalized,TensorId gate,TensorId output) {
    auto& p=*impl_;const auto& s=p.get(normalized).desc.shape;
    if(s.size()!=3 || s[0]>kResidentWindowRows || s[1]>32 || s[2]>65536 || s[2]%64 || p.get(gate).desc.shape!=s ||
       p.get(output).desc.shape!=std::vector<uint64_t>{s[0],s[2]}) throw std::invalid_argument("HC window mix geometry mismatch");
    abi::HcWindow args{int32_t(s[2]),int32_t(s[1]),int32_t(s[0])};const TensorId in[]={normalized,gate},out[]={output};
    p.node(in,out,"strata_hc_mix_window_f32",&args,sizeof(args),true);
}
void GraphBuilder::hc_write_window(TensorId residual,TensorId block,TensorId inject,TensorId output) {
    auto& p=*impl_;const auto& s=p.get(residual).desc.shape;
    if(s.size()!=3 || s[0]>kResidentWindowRows || s[1]>32 || s[2]>65536 || s[2]%64 || p.get(block).desc.shape!=std::vector<uint64_t>{s[0],s[2]} ||
       p.get(inject).desc.shape!=std::vector<uint64_t>{s[0],s[1]} || p.get(output).desc.shape!=s || residual==output)
        throw std::invalid_argument("HC window write geometry/alias mismatch");
    abi::HcWindow args{int32_t(s[2]),int32_t(s[1]),int32_t(s[0])};const TensorId in[]={residual,block,inject},out[]={output};
    p.node(in,out,"strata_hc_write_window_f32",&args,sizeof(args),true);
}
void GraphBuilder::ple_gate(TensorId key,TensorId query,TensorId value,TensorId output) {
    auto& p=*impl_;const auto& s=p.get(key).desc.shape;
    if(s.size()!=2 || s[0]>32 || s[1]>65536 || s[1]%64 || p.get(query).desc.shape!=s || p.get(output).desc.shape!=s ||
       p.get(value).desc.shape!=std::vector<uint64_t>{s[1]}) throw std::invalid_argument("PLE gate geometry mismatch");
    abi::PleGate args{int32_t(s[1]),int32_t(s[0]),1/std::sqrt(float(s[1]))};const TensorId in[]={key,query,value},out[]={output};
    p.node(in,out,ple_gate_simd_enabled()?"strata_ple_gate_simd_f32":"strata_ple_gate_f32",&args,sizeof(args),true);
}
void GraphBuilder::ple_conv(TensorId history,TensorId normalized,TensorId weights,TensorId hidden,TensorId gated,
                            TensorId next,TensorId output,uint32_t dilation) {
    auto& p=*impl_;const auto& w=p.get(weights).desc.shape;
    if(w.size()!=2 || w[0]<2 || w[0]>16 || w[1]>1048576 || w[1]%64 || !dilation || dilation>16 ||
       p.get(history).desc.shape!=std::vector<uint64_t>{(w[0]-1)*dilation,w[1]} || p.get(next).desc.shape!=p.get(history).desc.shape || history==next)
        throw std::invalid_argument("PLE conv history/weight geometry mismatch");
    for(auto id:{normalized,hidden,gated,output}) if(p.get(id).desc.shape!=std::vector<uint64_t>{w[1]})
        throw std::invalid_argument("PLE conv flat vector mismatch");
    abi::PleConv args{int32_t(w[1]),int32_t(w[0]),int32_t(dilation)};
    const TensorId in[]={history,normalized,weights,hidden,gated},out[]={next,output};
    p.node(in,out,"strata_ple_conv_f32",&args,sizeof(args),true);
}
void GraphBuilder::gdn_conv(TensorId state, TensorId input, TensorId weights, TensorId next_state, TensorId activated) {
    auto& p = *impl_;
    const auto& w = p.get(weights).desc.shape;
    if (w.size() != 2 || w[0] < 2 || w[0] > 16 || w[1] > 1048576 ||
        p.get(state).desc.shape != std::vector<uint64_t>{w[0]-1,w[1]} ||
        p.get(next_state).desc.shape != p.get(state).desc.shape || state == next_state ||
        p.get(input).desc.shape != std::vector<uint64_t>{w[1]} || p.get(activated).desc.shape != p.get(input).desc.shape)
        throw std::invalid_argument("invalid GDN time-major conv geometry");
    abi::GdnConv args{int32_t(w[1]),int32_t(w[0])};
    const TensorId inputs[] = {state,input,weights}, outputs[] = {next_state,activated};
    p.node(inputs, outputs, "strata_gdn_conv_f32", &args, sizeof(args), true);
}
void GraphBuilder::gdn_prepare(TensorId activated, TensorId alpha, TensorId dt, TensorId ssm_a, TensorId raw_beta,
                               TensorId q, TensorId k, TensorId v, TensorId gate, TensorId beta, float epsilon) {
    auto& p = *impl_;
    const auto& qs = p.get(q).desc.shape;
    const auto& vs = p.get(v).desc.shape;
    if (qs.size() != 2 || vs.size() != 2 || !std::isfinite(epsilon) || epsilon <= 0)
        throw std::invalid_argument("invalid GDN prepare rank/epsilon");
    const auto hk = qs[0], s = qs[1], hv = vs[0];
    const auto c = checked_product(s, checked_product(hk, 2) + hv);
    if (s > 256 || s % 64 || hv > 1024 || hv % hk || vs[1] != s || p.get(k).desc.shape != qs ||
        p.get(activated).desc.shape != std::vector<uint64_t>{c}) throw std::invalid_argument("invalid GDN prepare geometry");
    for (auto id : {alpha,dt,ssm_a,raw_beta,gate,beta})
        if (p.get(id).desc.shape != std::vector<uint64_t>{hv}) throw std::invalid_argument("GDN gate vector shape");
    abi::GdnPrepare args{int32_t(s),int32_t(hk),int32_t(hv),epsilon};
    const TensorId inputs[] = {activated,alpha,dt,ssm_a,raw_beta}, outputs[] = {q,k,v,gate,beta};
    p.node(inputs, outputs, simd_gdn_enabled() ? "strata_gdn_prepare_simd_f32" : "strata_gdn_prepare_f32", &args, sizeof(args), true);
}
void GraphBuilder::gdn_norm(TensorId input, TensorId z, TensorId norm, TensorId output, float epsilon) {
    auto& p = *impl_;
    const auto& dims = p.get(input).desc.shape;
    if (dims.size() != 2 || dims[1] > 256 || dims[1] % 64 || dims[0] > 1024 ||
        p.get(z).desc.shape != dims || p.get(output).desc.shape != dims ||
        p.get(norm).desc.shape != std::vector<uint64_t>{dims[1]} || !std::isfinite(epsilon) || epsilon <= 0)
        throw std::invalid_argument("invalid GDN output norm geometry/epsilon");
    abi::GdnNorm args{int32_t(dims[1]),int32_t(dims[0]),epsilon};
    const TensorId inputs[] = {input,z,norm}, outputs[] = {output};
    p.node(inputs, outputs, simd_gdn_enabled() ? "strata_gdn_norm_simd_f32" : "strata_gdn_norm_f32", &args, sizeof(args), true);
}
void GraphBuilder::gdn_step(TensorId state_id, TensorId q_id, TensorId k_id, TensorId v_id,
                             TensorId gate_id, TensorId beta_id, TensorId state_output_id, TensorId output_id) {
    auto& p = *impl_;
    // STRATA_HPU_SIMD_GDN: q and k broadcast to [key heads, s, 64] first, so the step reads each q[i]/k[i] as a
    // vector. Created before any tensor reference below is taken (new tensors may move the slot storage).
    TensorId q_read = q_id, k_read = k_id;
    if (simd_gdn_enabled() && !cached_gdn_step_enabled()) {
        const auto qs = p.get(q_id).desc.shape;
        if (qs.size() != 2) throw std::invalid_argument("invalid GDN tensor ranks");
        const auto tag = "gdn_step_" + std::to_string(p.nodes);
        for (auto* id : {&q_read, &k_read}) {
            const auto stem = tag + (id == &q_read ? "_q" : "_k");
            const auto column = tensor({stem + "_column", DType::F32, {qs[0], qs[1], 1}, false});
            const auto wide = tensor({stem + "_broadcast", DType::F32, {qs[0], qs[1], 64}, false});
            reshape(*id, column);
            const TensorId in[] = {column}, out[] = {wide};
            op("broadcast", in, out);
            *id = wide;
        }
    }
    const auto& state = p.get(state_id);
    const auto& q = p.get(q_id);
    const auto& k = p.get(k_id);
    const auto& q_input = p.get(q_read);
    const auto& k_input = p.get(k_read);
    const auto& v = p.get(v_id);
    const auto& gate = p.get(gate_id);
    const auto& beta = p.get(beta_id);
    const auto& state_out = p.get(state_output_id);
    const auto& output = p.get(output_id);
    if (state.desc.shape.size() != 3 || q.desc.shape.size() != 2)
        throw std::invalid_argument("invalid GDN tensor ranks");
    const auto s = state.desc.shape[0], hv = state.desc.shape[1], hk = q.desc.shape[0];
    if (s > 256 || s % 64 || hv % hk || state.desc.shape != std::vector<uint64_t>{s, hv, s} ||
        q.desc.shape != std::vector<uint64_t>{hk, s} || k.desc.shape != q.desc.shape ||
        v.desc.shape != std::vector<uint64_t>{hv, s} || gate.desc.shape != std::vector<uint64_t>{hv} ||
        beta.desc.shape != gate.desc.shape || state_out.desc.shape != state.desc.shape || output.desc.shape != v.desc.shape)
        throw std::invalid_argument("invalid GDN geometry");
    const TensorSlot* tensors[] = {&state, &q, &k, &v, &gate, &beta, &state_out, &output};
    for (auto* tensor : tensors) if (tensor->desc.dtype != DType::F32)
        throw std::invalid_argument("GDN recurrence requires FP32 tensors");
    const synTensor inputs[] = {state.tensor, q_input.tensor, k_input.tensor, v.tensor, gate.tensor, beta.tensor};
    const synTensor outputs[] = {state_out.tensor, output.tensor};
    abi::GdnStep params{static_cast<int>(s), static_cast<int>(hk)};
    const auto name = "strata_gdn_step_" + std::to_string(p.nodes);
    check(synNodeCreate(p.graph, inputs, outputs, 6, 2, &params, sizeof(params),
                        cached_gdn_step_enabled() ? (gdn_step_pipe_enabled() ? "strata_gdn_step_pipe_f32" : "strata_gdn_step_cached_f32")
                                                  : (q_read == q_id ? "strata_gdn_step_f32" : "strata_gdn_step_bcast_f32"), name.c_str(), nullptr, nullptr), "synNodeCreate GDN");
    p.written.push_back(state_out.desc.name);p.written.push_back(output.desc.name);
    ++p.nodes;
}
std::unique_ptr<Recipe> GraphBuilder::compile(const std::string& name) {return compile(name,std::string{});}
std::unique_ptr<Recipe> GraphBuilder::compile(const std::string& name,const std::string& shared_workspace) {
    auto& p = *impl_;
    p.building();
    if (!p.device) throw std::logic_error("offline graph cannot create an executable Device recipe");
    if (!p.nodes || name.empty()) throw std::invalid_argument("cannot compile an empty or unnamed graph");
    if (!p.aliases.empty()) throw std::invalid_argument("unused graph alias: " + p.aliases.begin()->first);
    p.compiled = true;
    synRecipeHandle handle = nullptr;
    // STRATA_HPU_COMPILE_TRACE=1 (diagnostic): every graph compile's name, host time and workspace on stderr.
    static const bool trace = performance_env("STRATA_HPU_COMPILE_TRACE");
    const auto compile_start = std::chrono::steady_clock::now();
    check(synGraphCompile(&handle, p.graph, name.c_str(), nullptr), "synGraphCompile");
    try {
        uint64_t workspace = 0;
        check(synWorkspaceGetSize(&workspace, handle), "synWorkspaceGetSize");
        if (trace) std::fprintf(stderr, "strata-hpu: compile %s %.0f ms workspace %.1f MiB\n", name.c_str(),
                                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - compile_start).count(),
                                double(workspace) / 1048576.0);
        std::vector<TensorDesc> desc;
        for (const auto& slot : p.tensors) if (slot.desc.persistent) desc.push_back(slot.desc);
        std::vector<const char*> names;
        for (const auto& item : desc) names.push_back(item.name.c_str());
        std::vector<uint64_t> ids(names.size());
        check(synTensorRetrieveIds(handle, names.data(), ids.data(), static_cast<uint32_t>(names.size())),
              "synTensorRetrieveIds");
        auto impl = std::make_unique<Recipe::Impl>(*p.device, handle, std::move(desc), std::move(ids), shared_workspace.empty()?workspace:0);
        if(!shared_workspace.empty()) {
            impl->shared=SharedWorkspace::get(shared_workspace);impl->shared->reserve(*p.device,workspace);impl->shared_bytes=workspace;
        }
        impl->written = p.written;
        impl->name = name;
        handle = nullptr;
        return std::unique_ptr<Recipe>(new Recipe(std::move(impl)));
    } catch (...) {
        if (handle) release_status(synRecipeDestroy(handle), "synRecipeDestroy failed compile");
        throw;
    }
}
uint64_t GraphBuilder::compile_offline(const std::string& name) {
    auto& p = *impl_;
    p.building();
    if (p.device) throw std::logic_error("use OfflineCompilation to validate without acquiring a device");
    if (!p.nodes || name.empty()) throw std::invalid_argument("cannot compile an empty or unnamed graph");
    if (!p.aliases.empty()) throw std::invalid_argument("unused graph alias: " + p.aliases.begin()->first);
    p.compiled = true;
    synRecipeHandle handle = nullptr;
    check(synGraphCompile(&handle, p.graph, name.c_str(), nullptr), "offline synGraphCompile");
    try {
        uint64_t workspace = 0;
        check(synWorkspaceGetSize(&workspace, handle), "offline synWorkspaceGetSize");
        const auto destroyed=synRecipeDestroy(handle);
        handle=nullptr;
        check(destroyed, "offline synRecipeDestroy");
        return workspace;
    } catch (...) {
        if (handle) release_status(synRecipeDestroy(handle), "offline recipe cleanup");
        throw;
    }
}
} // namespace strata::hpu
