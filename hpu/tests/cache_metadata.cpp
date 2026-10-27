#include "strata/hpu/generation.hpp"
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <new>
namespace {
thread_local size_t fail_size=0;
thread_local uint32_t failures=0;
}
// Fail precisely the optional marker's token copy after cancellation was
// raised. Other model/metadata buffers are already reserved before arming.
[[gnu::noinline]] void* operator new(std::size_t size) {
    if(fail_size && size==fail_size) {fail_size=0;++failures;throw std::bad_alloc();}
    if(auto* p=std::malloc(size?size:1)) return p;
    throw std::bad_alloc();
}
[[gnu::noinline]] void operator delete(void* p) noexcept {std::free(p);}
[[gnu::noinline]] void operator delete(void* p,std::size_t) noexcept {std::free(p);}
using namespace strata::hpu;
namespace {
void need(bool ok,const char* message) {if(!ok) throw std::runtime_error(message);}
template<class F> void rejects(F f) {bool failed=false;try {f();} catch(const std::exception&) {failed=true;}need(failed,"invalid metadata geometry admitted");}
struct Fixture {
    std::vector<int32_t> state;
    bool telemetry=true;
    Fixture() {state.reserve(256);}
    GenerationOperations operations() {
        GenerationOperations op;op.reset=[&]{state.clear();};
        op.decode=[&](int32_t id,const auto& cancelled){if(cancelled && cancelled()) throw Cancelled();state.push_back(id);std::vector<float> values(128,0);values[0]=1;return values;};
        op.checkpoint_bytes=2048;op.checkpoint=[&]{auto saved=std::make_shared<std::vector<int32_t>>(state);return std::function<void()>([&,saved]{state=*saved;});};
        op.available_host_bytes=[&]{return telemetry?std::optional<uint64_t>(UINT64_MAX):std::nullopt;};return op;
    }
};
GenerationOptions options(uint32_t slots) {GenerationOptions o;o.max_draft=0;o.cache_slots=slots;o.cache_bytes=16u<<20;o.cache_every=0;o.boundaries.turn_token=-1;return o;}
GenerationRequest request(int32_t first) {GenerationRequest r;r.prompt.assign(191,126);r.prompt[0]=first;r.sampling.greedy=true;r.max_new=1;return r;}
void full_capacity() {
    Fixture f;Generation gen(128,256,{},f.operations(),options(64));GenerationEvents events;events.token=[](int32_t id){need(id==0,"metadata fill changed sampler output");};
    for(int32_t i=0;i<64;++i) gen.run(request(i),events);
    need(gen.cache_stats().entries==64,"fixture did not fill the supported checkpoint bank");
    auto metadata=gen.metadata_stats();need(metadata.vector_bytes>64*(191*4+128*4) && metadata.vector_bytes<metadata.envelope,"resident bank vectors escaped the planned metadata envelope");
    for(int32_t i=64;i<128;++i) {
        bool armed=false;events.reused=[&](uint32_t){armed=true;};
        auto result=gen.run(request(i),events,[&]{return armed && f.state.size()>=128;});
        need(result.generated==0 && result.prompt_read==128,"partial metadata fixture did not cancel at its committed prefix");
    }
    metadata=gen.metadata_stats();need(metadata.interrupted_entries==64 && metadata.vector_bytes>64*(128*4+128*4+128*4) && metadata.vector_bytes<metadata.envelope,"bank plus all interrupted-read token vectors were not independently budgeted");
    bool armed=false;events.reused=[&](uint32_t){armed=true;};gen.run(request(64),events,[&]{return armed && f.state.size()>=129;});
    need(gen.metadata_stats().interrupted_entries==64,"interrupted-read table grew beyond admitted entries");
}
void optional_allocation_failure() {
    Fixture f;f.telemetry=false;Generation gen(128,256,{},f.operations(),options(1));
    GenerationEvents events;bool armed=false;events.reused=[&](uint32_t){armed=true;};events.token=[](int32_t){};
    const auto result=gen.run(request(1),events,[&]{if(armed && f.state.size()>=128) {fail_size=128*sizeof(int32_t);return true;}return false;});
    need(result.generated==0 && result.finish=="cancelled" && result.prompt_read==128 && failures==1 && !fail_size,"optional marker allocation did not fail at the intended boundary");
    need(gen.metadata_stats().interrupted_entries==0 && f.state.size()==128,"failed optional marker changed committed model state");
    auto resumed=gen.run(request(1),events);need(resumed.reused==128 && resumed.generated==1,"marker allocation failure prevented valid prefix continuation");
}
void geometry() {
    need(generation_cache_metadata_bytes(256,128)>generation_cache_metadata_bytes(256,128,1),"maximum planning was silently limited to configured entries");
    rejects([]{generation_cache_metadata_bytes(0,128);});rejects([]{generation_cache_metadata_bytes(UINT64_MAX,128);});rejects([]{generation_cache_metadata_bytes(256,UINT64_MAX);});rejects([]{generation_cache_metadata_bytes(256,128,65);});
    PrefixCache bounded(1,10000,{},0,4,8);const std::vector<float> values(8,0),wrong(7,0);bool produced=false;
    rejects([&]{bounded.store(std::vector<int32_t>(5,1),values,100,[&]{produced=true;return std::function<void()>([]{});});});
    rejects([&]{bounded.store(std::vector<int32_t>(2,1),wrong,100,[&]{produced=true;return std::function<void()>([]{});});});
    need(!produced && !bounded.stats().entries,"oversized metadata reached the Model image producer");
}
}
int main() {
    try {full_capacity();geometry();optional_allocation_failure();std::cout<<"PASS full 64-entry bank plus interrupted-read capacities inside independent metadata envelope, shape refusal before image capture, and non-terminating optional marker allocation failure\n";}
    catch(const std::exception& e) {fail_size=0;std::cerr<<e.what()<<'\n';return 1;}
}
