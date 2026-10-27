#include "strata/hpu/gdn_block.hpp"
#include "strata/hpu/gdn_window.hpp"
#include "strata/hpu/host_recipe.hpp"
#include "strata/hpu/state_io.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstring>
#include <mutex>
#include <optional>
#include <set>
#include <functional>
#include <map>
#include <tuple>

namespace strata::hpu {
namespace {std::atomic<uint64_t> next_gdn_owner{1};}
using StateVector=std::shared_ptr<const std::vector<float>>;
struct GdnBlock::Impl {
    core::ModelGeometry geometry;std::unique_ptr<HostRecipe> execute;
    GdnWindow* window=nullptr;std::vector<Binding> weights;
    // Device state: the recipe reads buffer `current` and writes buffer 1-current; a successful decode flips
    // `current`. contents[b] is the generation buffer b holds; live[b] is the checkpoint reference of that
    // generation while anyone still holds one. `state`/`conv` are the host copy of buffer `current` when it
    // is known (after construction, prefill, restore, reset), empty after a decode.
    std::array<std::unique_ptr<DeviceBuffer>,2> state_device,conv_device;int current=0;
    std::array<uint64_t,2> contents{0,0};std::array<std::weak_ptr<StateRef>,2> live;uint64_t next_generation=1;
    std::unique_ptr<HostBuffer> state_readback,conv_readback;std::unique_ptr<Stream> copy;
    uint64_t state_count=0,conv_count=0;
    StateVector state,conv;uint64_t owner=next_gdn_owner++,epoch=0,position=0;
    std::optional<Device> device;
    mutable std::mutex mutex;
    void bind() {
        execute->bind_external({"state",state_device[current].get()});execute->bind_external({"conv_state",conv_device[current].get()});
        execute->bind_external({"state_out",state_device[1-current].get()});execute->bind_external({"conv_state_out",conv_device[1-current].get()});
    }
    std::pair<StateVector,StateVector> read(int buffer) const {
        const auto one=[&](HostBuffer& staging,const DeviceBuffer& device,uint64_t count) {
            download(*copy,staging,device,count*sizeof(float));copy->synchronize();
            const auto* values=static_cast<const float*>(staging.data());
            return std::make_shared<const std::vector<float>>(values,values+count);
        };
        return {one(*state_readback,*state_device[buffer],state_count),one(*conv_readback,*conv_device[buffer],conv_count)};
    }
    // Before buffer b is overwritten: a checkpoint still holding its generation gets the host copy now.
    void materialize(int buffer) {
        const auto ref=live[buffer].lock();
        if(ref && ref->generation==contents[buffer] && !ref->state) std::tie(ref->state,ref->conv)=read(buffer);
        live[buffer].reset();
    }
    // The host copy of buffer `current` (read back when a decode left it unknown).
    void known() {if(!state) std::tie(state,conv)=read(current);}
    // The reset state: both buffers of the other slot zeroed on the device (no staging copy or upload) and one
    // shared all-zero host copy per size, the same state install() of zero vectors leaves.
    void install_zero() {
        static std::mutex cache_mutex;static std::map<uint64_t,StateVector> zeros;
        const auto shared=[&](uint64_t count) {
            std::lock_guard lock(cache_mutex);auto& v=zeros[count];
            if(!v) v=std::make_shared<const std::vector<float>>(count);
            return v;
        };
        const int next=1-current;materialize(next);
        zero(*copy,*state_device[next],state_count*sizeof(float));zero(*copy,*conv_device[next],conv_count*sizeof(float));
        copy->synchronize();
        current=next;contents[current]=next_generation++;state=shared(state_count);conv=shared(conv_count);bind();
    }
    // Make `values` the accepted state: written into the other buffer, which then becomes current.
    void install(StateVector new_state,StateVector new_conv) {
        const int next=1-current;materialize(next);
        for(const auto& [values,device,staging]:{std::tuple{new_state.get(),state_device[next].get(),state_readback.get()},
                                                 std::tuple{new_conv.get(),conv_device[next].get(),conv_readback.get()}}) {
            const auto bytes=checked_product(values->size(),sizeof(float));
            std::memcpy(staging->data(),values->data(),bytes);upload(*copy,*device,*staging,bytes);
        }
        copy->synchronize();
        current=next;contents[current]=next_generation++;state=std::move(new_state);conv=std::move(new_conv);bind();
    }
    std::shared_ptr<StateRef> reference() {
        auto ref=live[current].lock();
        if(!ref || ref->generation!=contents[current]) {ref=std::make_shared<StateRef>();ref->generation=contents[current];live[current]=ref;}
        if(!ref->state && state) {ref->state=state;ref->conv=conv;}
        return ref;
    }
};
GdnBlock::GdnBlock(const Device& d,core::ModelGeometry g,std::span<const Binding> weights,GdnWindow* window) {
    std::set<std::string> names;for(const auto& w:weights) names.insert(w.name);
    if(weights.size()!=9 || names!=std::set<std::string>{"w_qkv","w_gate","w_alpha","w_beta","w_out","conv_weights","norm","dt","ssm_a"})
        throw std::invalid_argument("GDN requires five projection and four constant bindings");
    auto recipe=build_gdn_block(d,g); // Validates derived geometry before sizes.
    auto p=std::make_unique<Impl>();p->geometry=g;p->window=window;p->weights.assign(weights.begin(),weights.end());p->device=d;
    p->state_count=checked_product(checked_product(g.ssm_state_size,g.ssm_v_heads),g.ssm_state_size);
    p->conv_count=checked_product(g.ssm_d_conv-1,g.ssm_conv_channels);
    const auto state_bytes=checked_product(p->state_count,sizeof(float)),conv_bytes=checked_product(p->conv_count,sizeof(float));
    for(int i=0;i<2;++i) {p->state_device[i]=std::make_unique<DeviceBuffer>(d,state_bytes);p->conv_device[i]=std::make_unique<DeviceBuffer>(d,conv_bytes);}
    p->state_readback=std::make_unique<HostBuffer>(d,state_bytes);p->conv_readback=std::make_unique<HostBuffer>(d,conv_bytes);
    p->copy=std::make_unique<Stream>(d);
    std::vector<Binding> bound(weights.begin(),weights.end());
    bound.push_back({"state",p->state_device[0].get()});bound.push_back({"conv_state",p->conv_device[0].get()});
    bound.push_back({"state_out",p->state_device[1].get()});bound.push_back({"conv_state_out",p->conv_device[1].get()});
    p->execute=std::make_unique<HostRecipe>(d,std::move(recipe),bound,std::vector<std::string>{"out"});
    p->install(std::make_shared<const std::vector<float>>(p->state_count),std::make_shared<const std::vector<float>>(p->conv_count));
    impl_=std::move(p);
}
GdnBlock::~GdnBlock()=default;
std::vector<float> GdnBlock::decode(std::span<const float> x) {
    auto& p=*impl_;std::lock_guard lock(p.mutex);
    if(x.size()!=uint64_t(p.geometry.n_embd) || p.position==UINT64_MAX) throw std::invalid_argument("GDN input/position mismatch");
    const int next=1-p.current;p.materialize(next);
    p.execute->set("x",x);p.execute->run();
    auto out=p.execute->values("out");
    if(!all_finite(out)) throw std::runtime_error("GDN produced non-finite output");
    // The recipe wrote the other buffer, so a failed computation never touched the accepted state.
    p.current=next;p.contents[p.current]=p.next_generation++;p.state.reset();p.conv.reset();p.bind();++p.position;
    return out;
}
std::array<Binding,4> GdnBlock::external_step() {
    auto& p=*impl_;std::lock_guard lock(p.mutex);
    if(p.position==UINT64_MAX) throw std::invalid_argument("GDN position overflow");
    const int next=1-p.current;p.materialize(next);
    return {{{"state",p.state_device[p.current].get()},{"conv_state",p.conv_device[p.current].get()},
             {"state_out",p.state_device[next].get()},{"conv_state_out",p.conv_device[next].get()}}};
}
void GdnBlock::commit_external() {
    auto& p=*impl_;std::lock_guard lock(p.mutex);
    p.current=1-p.current;p.contents[p.current]=p.next_generation++;p.state.reset();p.conv.reset();p.bind();++p.position;
}
std::array<Binding,2> GdnBlock::current_state() {
    auto& p=*impl_;std::lock_guard lock(p.mutex);
    return {{{"state",p.state_device[p.current].get()},{"conv_state",p.conv_device[p.current].get()}}};
}
uint64_t GdnBlock::state_bytes() const {const auto& p=*impl_;return p.state_count*sizeof(float);}
uint64_t GdnBlock::conv_bytes() const {const auto& p=*impl_;return p.conv_count*sizeof(float);}
void GdnBlock::adopt(std::unique_ptr<DeviceBuffer>& state,std::unique_ptr<DeviceBuffer>& conv,uint64_t tokens) {
    auto& p=*impl_;std::lock_guard lock(p.mutex);
    if(!tokens || tokens>UINT64_MAX-p.position || !state || !conv || state->size()!=p.state_count*sizeof(float) || conv->size()!=p.conv_count*sizeof(float))
        throw std::invalid_argument("GDN adopt buffers/position");
    // The buffer about to be written holds an older generation; a checkpoint still referring to it is read back first.
    const int next=1-p.current;p.materialize(next);
    std::swap(p.state_device[next],state);std::swap(p.conv_device[next],conv);
    p.current=next;p.contents[p.current]=p.next_generation++;p.state.reset();p.conv.reset();p.bind();p.position+=tokens;
}
std::vector<float> GdnBlock::prefill(std::span<const float> input,const std::function<bool()>& cancelled) {
    auto& p=*impl_;std::lock_guard lock(p.mutex);const auto h=uint64_t(p.geometry.n_embd);
    if(!p.window || input.empty() || input.size()%h || input.size()/h>128 || input.size()/h>UINT64_MAX-p.position)
        throw std::invalid_argument("GDN prefill requires shared executor and1..128 complete rows");
    p.known();
    auto next=p.window->run(input,*p.state,*p.conv,p.weights,cancelled);
    // No intermediate subwindow is accepted; the layer can still rewind a
    // successful whole call if a later HC/MoE/head stage fails.
    p.install(std::make_shared<const std::vector<float>>(std::move(next.state)),std::make_shared<const std::vector<float>>(std::move(next.conv)));
    p.position+=input.size()/h;return std::move(next.output);
}
uint64_t GdnBlock::size() const {const auto& p=*impl_;std::lock_guard lock(p.mutex);return p.position;}
GdnBlock::Checkpoint GdnBlock::checkpoint() const {
    auto& p=*impl_;std::lock_guard lock(p.mutex);
    Checkpoint c;c.owner=p.owner;c.epoch=p.epoch;c.position=p.position;c.ref=p.reference();return c;
}
GdnBlock::Checkpoint GdnBlock::snapshot() const {
    auto point=checkpoint();
    auto& p=*impl_;std::lock_guard lock(p.mutex);
    if(!point.ref->state) {p.known();point.ref->state=p.state;point.ref->conv=p.conv;}
    point.persistent=true;return point;
}
uint64_t GdnBlock::parked_bytes() const {auto& p=*impl_;std::lock_guard lock(p.mutex);return checked_product(p.state_count+p.conv_count,sizeof(float));}
void GdnBlock::write_snapshot(StateWriter& writer,const Checkpoint& point) const {
    const auto& p=*impl_;std::lock_guard lock(p.mutex);
    if(point.owner!=p.owner || !point.persistent || !point.ref || !point.ref->state || !point.ref->conv)
        throw std::invalid_argument("GDN serialization requires an owned full snapshot");
    writer.integer(point.position);writer.vector<float>(*point.ref->state);writer.vector<float>(*point.ref->conv);
}
GdnBlock::Checkpoint GdnBlock::read_snapshot(StateReader& reader,uint64_t position) const {
    const auto& p=*impl_;std::lock_guard lock(p.mutex);
    if(reader.integer()!=position) throw std::invalid_argument("GDN serialized position mismatch");
    Checkpoint point;point.owner=p.owner;point.position=position;point.persistent=true;
    auto state=reader.vector<float>(p.state_count);auto conv=reader.vector<float>(p.conv_count);
    for(const auto& values:{std::cref(state),std::cref(conv)}) for(float value:values.get())
        if(!std::isfinite(value)) throw std::invalid_argument("non-finite serialized GDN state");
    point.ref=std::make_shared<StateRef>();
    point.ref->state=std::make_shared<const std::vector<float>>(std::move(state));point.ref->conv=std::make_shared<const std::vector<float>>(std::move(conv));
    return point;
}
void GdnBlock::restore(const Checkpoint& c) {
    auto& p=*impl_;std::lock_guard lock(p.mutex);
    if(c.owner!=p.owner || (!c.persistent && (c.epoch!=p.epoch || c.position>p.position)) || !c.ref)
        throw std::invalid_argument("stale/foreign GDN checkpoint");
    if(c.ref->state) {
        if(c.ref->state->size()!=p.state_count || !c.ref->conv || c.ref->conv->size()!=p.conv_count) throw std::invalid_argument("stale/foreign GDN checkpoint");
        p.install(c.ref->state,c.ref->conv);
    } else {
        // Not read back, so its generation is still in one of the buffers (materialize() runs before any
        // buffer holding a referenced generation is overwritten).
        int found=-1;for(int b=0;b<2;++b) if(c.ref->generation && p.contents[b]==c.ref->generation) found=b;
        if(found<0) throw std::logic_error("GDN checkpoint generation is no longer on the device");
        if(found!=p.current) {p.current=found;p.state.reset();p.conv.reset();p.bind();}
    }
    p.position=c.position;++p.epoch;
}
void GdnBlock::reset() {
    auto& p=*impl_;std::lock_guard lock(p.mutex);
    p.install_zero();
    p.position=0;++p.epoch;
}
}
