#include "strata/hpu/host_recipe.hpp"
#include "strata/hpu/performance_modes.hpp"
#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <future>
#include <map>
#include <mutex>
#include <set>
#include <unordered_map>
#include <string_view>

namespace strata::hpu {
namespace {std::atomic<uint64_t> host_recipe_bytes{0};} // HostRecipe::total_device_bytes()
namespace {thread_local std::span<const Binding> scratch_buffers;}
RecipeScratchScope::RecipeScratchScope(std::span<const Binding> scratch):previous_(scratch_buffers) {scratch_buffers=scratch;}
RecipeScratchScope::~RecipeScratchScope() {scratch_buffers=previous_;}
struct HostRecipe::Impl {
    struct Slot {
        TensorDesc desc;bool output=false,internal=false,initialized=false,dirty=false;
        std::unique_ptr<HostBuffer> host;
        std::unique_ptr<DeviceBuffer> device;
        // STRATA_HPU_PACKED_IO: the slot lives at `offset` in the shared input or output block instead.
        bool packed=false;uint64_t offset=0;void* data=nullptr;
    };
    Device device;
    std::unique_ptr<Recipe> recipe;
    Stream input,compute,readback;
    Event ready,done;
    std::vector<Slot> slots;
    std::vector<Binding> bindings;
    uint64_t pinned=0,device_owned=0;bool completed=false,launched=false;
    std::unique_ptr<HostBuffer> input_host,output_host;std::unique_ptr<DeviceBuffer> input_device,output_device;
    struct External {const TensorDesc* desc;size_t index;};
    std::unordered_map<std::string_view,size_t> slot_index;
    std::unordered_map<std::string_view,External> external_index;
    std::set<std::string> external_names;
    Impl(const Device& d,std::unique_ptr<Recipe> r):device(d),recipe(std::move(r)),input(d),compute(d),readback(d),ready(d),done(d) {}
    // A launch not yet waited for: its uploads may still read the pinned inputs, so wait before reuse.
    void settle() {if(launched) {launched=false;done.synchronize();completed=true;}}
    Slot& get(const std::string& name) {
        if(fast_bindings_enabled()) {
            const auto it=slot_index.find(name);
            if(it==slot_index.end()) throw std::invalid_argument("unknown/externally bound host slot: "+name);
            return slots[it->second];
        }
        auto found=std::find_if(slots.begin(),slots.end(),[&](const Slot& s){return s.desc.name==name;});
        if(found==slots.end()) throw std::invalid_argument("unknown/externally bound host slot: "+name);
        return *found;
    }
    const Slot& get(const std::string& name) const {
        if(fast_bindings_enabled()) {
            const auto it=slot_index.find(name);
            if(it==slot_index.end()) throw std::invalid_argument("unknown/externally bound host slot: "+name);
            return slots[it->second];
        }
        auto found=std::find_if(slots.begin(),slots.end(),[&](const Slot& s){return s.desc.name==name;});
        if(found==slots.end()) throw std::invalid_argument("unknown/externally bound host slot: "+name);
        return *found;
    }
};
HostRecipe::HostRecipe(const Device& device,std::unique_ptr<Recipe> recipe,std::span<const Binding> external,
                       std::vector<std::string> outputs,uint64_t budget) {
    if(!recipe) throw std::invalid_argument("null host recipe");
    std::set<std::string> output_names(outputs.begin(),outputs.end()),external_names;
    if(output_names.size()!=outputs.size()) throw std::invalid_argument("duplicate output names");
    uint64_t bytes=0;
    const auto tensors=recipe->tensors();
    // RecipeScratchScope: graph-internal persistent tensors borrow the scope's buffer of their name.
    std::vector<Binding> borrowed(external.begin(),external.end());
    for(const auto& t:tensors) {
        if(output_names.contains(t.name) || !recipe->produced_by_graph(t.name) ||
           std::any_of(external.begin(),external.end(),[&](const Binding& b){return b.name==t.name;})) continue;
        const auto s=std::find_if(scratch_buffers.begin(),scratch_buffers.end(),[&](const Binding& b){return b.name==t.name;});
        if(s!=scratch_buffers.end() && s->buffer && s->buffer->size()>=s->offset+t.bytes()) borrowed.push_back(*s);
    }
    external=borrowed;
    for(const auto& binding:external) {
        auto t=std::find_if(tensors.begin(),tensors.end(),[&](const auto& d){return d.name==binding.name;});
        if(t==tensors.end() || !binding.buffer || !external_names.insert(binding.name).second || output_names.contains(binding.name) ||
           binding.buffer->state()!=device.state() || (binding.buffer->address()+binding.offset)%128)
            throw std::invalid_argument("invalid external host recipe binding: "+binding.name);
        check_range(binding.buffer->size(),binding.offset,t->bytes());
    }
    for(const auto& t:tensors) if(!external_names.contains(t.name)) {
        // A graph-produced non-output tensor is written and read on the device only: no pinned host copy exists
        // for it, and it is not part of the transfer budget (U4 grouped-MoE intermediates are hundreds of MiB).
        if(!output_names.contains(t.name) && recipe->produced_by_graph(t.name)) continue;
        if(t.bytes()>budget-bytes) throw std::invalid_argument("host recipe pinned budget exceeded");
        bytes+=t.bytes();
    }
    for(const auto& name:output_names)
        if(std::none_of(tensors.begin(),tensors.end(),[&](const auto& t){return t.name==name;})) throw std::invalid_argument("unknown recipe output");
    auto p=std::make_unique<Impl>(device,std::move(recipe));p->pinned=bytes;p->bindings.assign(external.begin(),external.end());
    p->external_names=external_names;
    const bool packed=packed_io_enabled();uint64_t input_bytes=0,output_bytes=0;
    for(const auto& t:p->recipe->tensors()) if(!external_names.contains(t.name)) {
        Impl::Slot s;s.desc=t;s.output=output_names.contains(t.name);
        // A tensor some node writes is allocated and bound, but the host neither fills it nor reads
        // it back; requiring it to be set would make two chained kernels impossible.
        s.internal=!s.output&&p->recipe->produced_by_graph(t.name);
        if(packed && !s.internal) {
            // 128-byte aligned places, the alignment every recipe binding requires.
            auto& end=s.output?output_bytes:input_bytes;s.packed=true;s.offset=end;end+=(t.bytes()+127)&~uint64_t(127);
        } else {
            // Internal tensors never touch the host: only their HBM storage is allocated.
            if(!s.internal) s.host=std::make_unique<HostBuffer>(device,t.bytes());
            s.device=std::make_unique<DeviceBuffer>(device,t.bytes());s.data=s.host?s.host->data():nullptr;p->device_owned+=t.bytes();
            // STRATA_HPU_MEM_TRACE=1 (diagnostic): the large tensors a recipe owns instead of borrowing.
            static const bool trace=performance_env("STRATA_HPU_MEM_TRACE");
            if(trace && t.bytes()>=(16ull<<20)) std::fprintf(stderr,"strata-hpu: mem own %s %.1f MiB%s\n",t.name.c_str(),double(t.bytes())/1048576.0,s.internal?" (internal)":"");
        }
        p->slots.push_back(std::move(s));
    }
    if(input_bytes) {p->input_host=std::make_unique<HostBuffer>(device,input_bytes);p->input_device=std::make_unique<DeviceBuffer>(device,input_bytes);}
    if(output_bytes) {p->output_host=std::make_unique<HostBuffer>(device,output_bytes);p->output_device=std::make_unique<DeviceBuffer>(device,output_bytes);}
    p->device_owned+=input_bytes+output_bytes;host_recipe_bytes+=p->device_owned;
    for(auto& s:p->slots) {
        if(!s.packed) {p->bindings.push_back({s.desc.name,s.device.get()});continue;}
        auto& host=s.output?*p->output_host:*p->input_host;
        s.data=static_cast<uint8_t*>(host.data())+s.offset;
        p->bindings.push_back({s.desc.name,s.output?p->output_device.get():p->input_device.get(),s.offset});
    }
    if(fast_bindings_enabled()) {
        for(size_t i=0;i<p->slots.size();++i) p->slot_index.emplace(p->slots[i].desc.name,i);
        const auto descs=p->recipe->tensors();
        for(size_t i=0;i<p->bindings.size();++i) if(external_names.contains(p->bindings[i].name)) {
            const auto d=std::find_if(descs.begin(),descs.end(),[&](const auto& desc){return desc.name==p->bindings[i].name;});
            p->external_index.emplace(d->name,Impl::External{&*d,i});
        }
    }
    if(performance_env("STRATA_HPU_RECIPE_MEMORY")) {
        uint64_t pinned_allocated=0,local_hbm=0,internal_hbm=0,borrowed=0;
        for(const auto& s:p->slots) {
            if(s.host) pinned_allocated+=s.host->size();
            if(s.device) {local_hbm+=s.device->size();if(s.internal) internal_hbm+=s.device->size();}
        }
        if(p->input_host) pinned_allocated+=p->input_host->size();
        if(p->output_host) pinned_allocated+=p->output_host->size();
        if(p->input_device) local_hbm+=p->input_device->size();
        if(p->output_device) local_hbm+=p->output_device->size();
        for(const auto& t:tensors) if(external_names.contains(t.name)) borrowed+=t.bytes();
        // Borrowed tensor views are not additional allocations and may alias
        // other recipes. Workspace and owned persistent buffers are separate.
        std::fprintf(stderr,"strata-hpu recipe-memory %s pinned_payload=%llu pinned_allocated=%llu local_hbm=%llu internal_hbm=%llu workspace=%llu borrowed_tensor_bytes=%llu\n",
            p->recipe->name().c_str(),(unsigned long long)p->pinned,(unsigned long long)pinned_allocated,
            (unsigned long long)local_hbm,(unsigned long long)internal_hbm,(unsigned long long)p->recipe->workspace_bytes(),
            (unsigned long long)borrowed);
    }
    impl_=std::move(p);
}
HostRecipe::~HostRecipe() {if(impl_) host_recipe_bytes-=impl_->device_owned;}
void HostRecipe::set(const std::string& name,std::span<const float> values) {
    auto& p=*impl_;p.settle();auto& s=p.get(name);
    if(s.internal) throw std::invalid_argument("graph-produced tensor cannot be set from the host: "+name);
    if(s.output || s.desc.dtype==DType::U8 || values.size()!=s.desc.bytes()/element_bytes(s.desc.dtype)) throw std::invalid_argument("host recipe input shape/direction mismatch");
    // Large inputs (a window's PLE embedding rows: 4M values, ~15 ms on one thread while the device waited) are
    // checked and converted in eight parts; the same checks and the same bits.
    const size_t parts=values.size()>=(size_t(1)<<20)?8:1;
    const auto each=[&](auto&& work) {
        if(parts==1) {work(size_t(0),values.size());return;}
        std::vector<std::future<void>> running;
        for(size_t i=0;i<parts;++i) running.push_back(std::async(std::launch::async,work,values.size()*i/parts,values.size()*(i+1)/parts));
        for(auto& part:running) part.get();
    };
    if(s.desc.dtype==DType::F32) {
        each([&](size_t a,size_t b) {
            if(any_nan(values.subspan(a,b-a))) throw std::invalid_argument("invalid host recipe input value");
            std::memcpy(static_cast<float*>(s.data)+a,values.data()+a,(b-a)*sizeof(float));
        });
    } else {
        // -inf is valid for the attention mask (F32). BF16 projections require finite input.
        auto* dest=static_cast<uint16_t*>(s.data);
        each([&](size_t a,size_t b) {
            if(!all_finite(values.subspan(a,b-a))) throw std::invalid_argument("invalid host recipe input value");
            for(size_t i=a;i<b;++i) {
                const auto v=to_bf16(values[i]);if(!std::isfinite(from_bf16(v))) throw std::overflow_error("BF16 host recipe input overflow");
                dest[i]=v;
            }
        });
    }
    s.initialized=true;s.dirty=true;p.completed=false;
}
void HostRecipe::set_bytes(const std::string& name,std::span<const uint8_t> values) {
    auto& p=*impl_;p.settle();auto& s=p.get(name);
    if(s.internal) throw std::invalid_argument("graph-produced tensor cannot be set from the host: "+name);
    // U4 is a packed byte payload too: desc.bytes() is already the nibble count halved.
    if(s.output || (s.desc.dtype!=DType::U8 && s.desc.dtype!=DType::U4) || values.size()!=s.desc.bytes())
        throw std::invalid_argument("host recipe byte input shape/dtype/direction mismatch");
    std::memcpy(s.data,values.data(),values.size());s.initialized=true;s.dirty=true;p.completed=false;
}
std::span<float> HostRecipe::input(const std::string& name) {
    auto& p=*impl_;p.settle();auto& s=p.get(name);
    if(s.internal) throw std::invalid_argument("graph-produced tensor cannot be set from the host: "+name);
    if(s.output || s.desc.dtype!=DType::F32) throw std::invalid_argument("direct host recipe input requires an F32 input: "+name);
    s.initialized=true;s.dirty=true;p.completed=false;
    return {static_cast<float*>(s.data),size_t(s.desc.bytes()/sizeof(float))};
}
bool HostRecipe::declares(const std::string& name) const {return impl_->external_names.contains(name);}
void HostRecipe::bind_external(Binding binding) {
    auto& p=*impl_;
    if(fast_bindings_enabled()) {
        const auto it=p.external_index.find(binding.name);
        if(it==p.external_index.end() || !binding.buffer || binding.buffer->state()!=p.device.state() ||
           (binding.buffer->address()+binding.offset)%128) throw std::invalid_argument("invalid external host recipe rebind");
        check_range(binding.buffer->size(),binding.offset,it->second.desc->bytes());
        auto& target=p.bindings[it->second.index];target.buffer=binding.buffer;target.offset=binding.offset;
        p.completed=false;return;
    }
    const auto tensors=p.recipe->tensors();
    const auto desc=std::find_if(tensors.begin(),tensors.end(),[&](const auto& t){return t.name==binding.name;});
    auto target=std::find_if(p.bindings.begin(),p.bindings.end(),[&](const auto& b){return b.name==binding.name;});
    if(desc==tensors.end() || target==p.bindings.end() || !binding.buffer || binding.buffer->state()!=p.device.state() ||
       std::any_of(p.slots.begin(),p.slots.end(),[&](const auto& s){return s.desc.name==binding.name;}) ||
       (binding.buffer->address()+binding.offset)%128) throw std::invalid_argument("invalid external host recipe rebind");
    check_range(binding.buffer->size(),binding.offset,desc->bytes());*target=std::move(binding);p.completed=false;
}
namespace {
// STRATA_HPU_RECIPE_STATS=1 (port diagnostic, off by default): wall time of every HostRecipe::run, summed
// per recipe name with the number of runs, printed to stderr at exit. The name has its trailing bucket or
// size digits folded so one table row covers every layer and bucket of a graph.
struct RecipeStats {
    std::mutex mutex;std::map<std::string,std::pair<uint64_t,double>> rows;
    static bool enabled() {static const bool on=[]{const char* v=std::getenv("STRATA_HPU_RECIPE_STATS");return v && *v && *v!='0';}();return on;}
    void add(const std::string& name,double ms) {
        auto key=name;while(!key.empty() && (std::isdigit(static_cast<unsigned char>(key.back())) || key.back()=='_')) key.pop_back();
        std::lock_guard lock(mutex);auto& row=rows[key.empty()?name:key];++row.first;row.second+=ms;
    }
    ~RecipeStats() {
        if(rows.empty()) return;
        std::vector<std::pair<std::string,std::pair<uint64_t,double>>> sorted(rows.begin(),rows.end());
        std::sort(sorted.begin(),sorted.end(),[](const auto& a,const auto& b){return a.second.second>b.second.second;});
        std::fprintf(stderr,"strata-hpu recipe stats (HostRecipe::run wall time, all runs):\n");
        for(const auto& [name,row]:sorted)
            std::fprintf(stderr,"  %-44s %10llu runs %12.1f ms %8.3f ms/run\n",name.c_str(),(unsigned long long)row.first,row.second,row.second/double(row.first));
    }
};
RecipeStats& recipe_stats() {static RecipeStats stats;return stats;}
}
void HostRecipe::run() {
    const bool timed=RecipeStats::enabled();const auto started=timed?std::chrono::steady_clock::now():std::chrono::steady_clock::time_point{};
    struct Report {
        bool timed;std::chrono::steady_clock::time_point started;const std::string& name;
        ~Report() {if(timed) recipe_stats().add(name,std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count());}
    } report{timed,started,impl_->recipe->name()};
    auto& p=*impl_;p.settle();p.completed=false;
    for(const auto& s:p.slots) if(!s.output && !s.internal && !s.initialized) throw std::logic_error("uninitialized host recipe input: "+s.desc.name);
    try {
        // Packed inputs go up as one copy spanning every dirty one; the clean ones inside that span hold
        // exactly what was uploaded last, so copying them again changes nothing.
        uint64_t first=UINT64_MAX,last=0;
        for(auto& s:p.slots) if(!s.output && !s.internal && s.dirty) {
            if(s.packed) {first=std::min(first,s.offset);last=std::max(last,s.offset+s.desc.bytes());}
            else upload(p.input,*s.device,*s.host,s.desc.bytes());
        }
        if(first<last) upload(p.input,*p.input_device,*p.input_host,last-first,first,first);
        p.ready.record(p.input);p.compute.wait(p.ready);p.recipe->enqueue(p.compute,p.bindings);p.readback.wait(p.recipe->completion());
        for(auto& s:p.slots) if(s.output && !s.packed) download(p.readback,*s.host,*s.device,s.desc.bytes());
        if(p.output_host) download(p.readback,*p.output_host,*p.output_device,p.output_host->size());
        p.done.record(p.readback);p.done.synchronize();
        for(auto& s:p.slots) s.dirty=false;
        p.completed=true;
    } catch(...) {
        // A failed submit may leave earlier DMA in flight; drain before callers
        // can replace pinned data or retry with a different input.
        p.device.synchronize();throw;
    }
}
void HostRecipe::launch(const Stream& stream) {
    auto& p=*impl_;
    // A launch that neither uploads a dirty input nor downloads an output touches no pinned buffer, so it may
    // queue behind an unfinished one (STRATA_HPU_RESIDENT_IQ4 layers share one recipe and never wait in between).
    bool host_traffic=bool(p.output_host);
    for(const auto& s:p.slots) host_traffic=host_traffic || s.output || (!s.internal && s.dirty);
    if(host_traffic) p.settle();
    p.completed=false;
    for(const auto& s:p.slots) if(!s.output && !s.internal && !s.initialized) throw std::logic_error("uninitialized host recipe input: "+s.desc.name);
    try {
        uint64_t first=UINT64_MAX,last=0;
        for(auto& s:p.slots) if(!s.output && !s.internal && s.dirty) {
            if(s.packed) {first=std::min(first,s.offset);last=std::max(last,s.offset+s.desc.bytes());}
            else upload(stream,*s.device,*s.host,s.desc.bytes());
        }
        if(first<last) upload(stream,*p.input_device,*p.input_host,last-first,first,first);
        p.recipe->enqueue(stream,p.bindings);
        for(auto& s:p.slots) if(s.output && !s.packed) download(stream,*s.host,*s.device,s.desc.bytes());
        if(p.output_host) download(stream,*p.output_host,*p.output_device,p.output_host->size());
        p.done.record(stream);
        for(auto& s:p.slots) s.dirty=false;
        p.launched=true;
    } catch(...) {p.device.synchronize();throw;}
}
void HostRecipe::wait() {
    auto& p=*impl_;
    if(!p.launched) throw std::logic_error("host recipe wait without launch");
    p.launched=false;
    try {p.done.synchronize();} catch(...) {p.device.synchronize();throw;}
    p.completed=true;
}
std::span<const float> HostRecipe::output(const std::string& name) const {
    const auto& p=*impl_;const auto& s=p.get(name);
    if(!p.completed || !s.output || s.desc.dtype!=DType::F32) throw std::logic_error("host output unavailable/wrong dtype");
    return {static_cast<const float*>(s.data),size_t(s.desc.bytes()/4)};
}
uint64_t HostRecipe::pinned_bytes() const {return impl_->pinned;}
uint64_t HostRecipe::device_bytes() const {return impl_->device_owned;}
uint64_t HostRecipe::total_device_bytes() {return host_recipe_bytes.load(std::memory_order_relaxed);}
std::vector<float> HostRecipe::values(const std::string& name) const {
    const auto& p=*impl_;const auto& s=p.get(name);
    if(!p.completed || !s.output || s.desc.dtype==DType::U8) throw std::logic_error("host output unavailable/wrong direction/dtype");
    std::vector<float> out(s.desc.bytes()/element_bytes(s.desc.dtype));
    if(s.desc.dtype==DType::F32) std::memcpy(out.data(),s.data,s.desc.bytes());
    else {
        const auto* source=static_cast<const uint16_t*>(s.data);
        for(size_t i=0;i<out.size();++i) out[i]=from_bf16(source[i]);
    }
    return out;
}
}
