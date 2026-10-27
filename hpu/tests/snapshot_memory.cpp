#include "strata/hpu/snapshot_memory.hpp"
#include "strata/hpu/prefix_cache.hpp"
#include <array>
#include <iostream>
#include <numeric>
using namespace strata::hpu;
namespace {
void require(bool value,const char* why){if(!value) throw std::runtime_error(why);}
template<class Error,class F> void rejects(F f){bool bad=false;try{f();}catch(const Error&){bad=true;}require(bad,"snapshot operation accepted an invalid reservation");}
struct Audit {
    std::shared_ptr<AllocationBudget> budget=std::make_shared<AllocationBudget>();
    uint64_t physical=0,peak=0;bool escaped=false,fail=false;
    explicit Audit(uint64_t limit){budget->configure(limit);}
};
template<class T> struct ObservedAllocator {
    using value_type=T;
    std::shared_ptr<Audit> audit;
    ObservedAllocator(std::shared_ptr<Audit> a):audit(std::move(a)){}
    template<class U> ObservedAllocator(const ObservedAllocator<U>& other):audit(other.audit){}
    T* allocate(size_t count) {
        const auto bytes=checked_product(count,sizeof(T));
        if(audit->fail) throw std::bad_alloc();
        if(audit->budget->stats().used<audit->physical+bytes) audit->escaped=true;
        auto* out=std::allocator<T>{}.allocate(count);audit->physical+=bytes;audit->peak=std::max(audit->peak,audit->physical);return out;
    }
    void deallocate(T* data,size_t count) noexcept {
        if(audit->budget->stats().used<audit->physical) audit->escaped=true;
        audit->physical-=count*sizeof(T);std::allocator<T>{}.deallocate(data,count);
    }
    template<class U> bool operator==(const ObservedAllocator<U>& other) const {return audit==other.audit;}
};
struct Image {
    SnapshotReservation reservation; // MUST outlive data, tested by deallocate's observer
    std::vector<int32_t,ObservedAllocator<int32_t>> data;
    static uint64_t resident(size_t cells){return checked_product(cells,4)+128;}
    Image(std::shared_ptr<Audit> audit,size_t cells):reservation(audit->budget,resident(cells)),data(cells,0,ObservedAllocator<int32_t>(audit)) {
        std::iota(data.begin(),data.end(),1);
    }
    void grow(size_t cells) {
        reservation.prepare(resident(cells),data.capacity()*sizeof(int32_t));
        snapshot_resize(data,cells);reservation.commit(resident(cells));
    }
};
uint64_t retained_bytes(const std::shared_ptr<void>& held) {
    return exclusive_snapshot_bytes(held,held?static_cast<Image*>(held.get())->reservation.bytes():0);
}
void short_images_and_growth() {
    auto audit=std::make_shared<Audit>(4096);
    rejects<AllocationBudgetExceeded>([&]{audit->budget->charge(1u<<20);});
    std::vector<std::shared_ptr<Image>> images;
    for(int i=0;i<6;++i) images.push_back(std::make_shared<Image>(audit,16));
    require(audit->budget->stats().used==6*Image::resident(16),"short captures charged an arena instead of owned extents");
    auto& image=*images[0];
    for(size_t n:{17u,33u,65u,129u}) {
        image.grow(n);require(image.data.capacity()==n && image.data.size()==n,"snapshot retained geometric excess capacity");
        for(int i=0;i<16;++i) require(image.data[i]==i+1,"growth lost the unchanged prefix");
        require(audit->budget->stats().used==5*Image::resident(16)+Image::resident(n),"settled image charge differs from cache envelope");
    }
    require(audit->peak<=audit->budget->stats().peak && !audit->escaped,"allocation or temporary overlap escaped reservation");
    images.clear();require(!audit->physical && !audit->budget->stats().used && !audit->escaped,"payload freed after reservation or ledger leak");
}
void failures_and_release_order() {
    auto audit=std::make_shared<Audit>(1024);
    auto image=std::make_shared<Image>(audit,64);const auto before=audit->budget->stats().used;
    rejects<AllocationBudgetExceeded>([&]{image->grow(256);});
    require(image->data.size()==64 && audit->budget->stats().used==before,"failed admission modified image/charge");
    rejects<std::overflow_error>([&]{image->reservation.prepare(UINT64_MAX,1);});
    rejects<std::invalid_argument>([&]{image->reservation.prepare(1,0);});
    rejects<std::logic_error>([&]{image->reservation.commit(1);});
    rejects<std::logic_error>([&]{image->reservation.commit(1025);});
    require(audit->budget->stats().used==before,"invalid envelope changed ledger");
    audit->fail=true;rejects<std::bad_alloc>([&]{image->grow(80);});
    require(image->data.size()==64,"failed component allocation damaged old image");
    // Model drops an exclusively consumed reuse slot on failure. Its peak
    // charge must survive until the last partial/old buffer is destroyed.
    image.reset();require(!audit->budget->stats().used && !audit->physical && !audit->escaped,"failure unwinding credited a live allocation");
}
void alias_take_drop_and_reclaim() {
    auto audit=std::make_shared<Audit>(4096);std::shared_ptr<void> held;
    auto available=std::make_shared<uint64_t>(1u<<20);
    PrefixCache cache(4,4096,[available]{return std::optional<uint64_t>(*available);},0);
    cache.set_retained([&]{return retained_bytes(held);},[&]{held.reset();});
    const std::array<int32_t,2> ids{0,1};const std::array<float,4> logits{1,2,3,4};
    auto producer=[&]{auto image=std::make_shared<Image>(audit,16);return std::function<void()>([image,&held]{held=image;});};
    require(cache.store(ids,logits,Image::resident(16),producer),"short park rejected with resident budget available");
    cache.entry(0).restore();
    require(cache.stats().retained==0 && audit->budget->stats().used==Image::resident(16),"aliased restored payload was charged twice");
    {auto taken=cache.take(0);require(!cache.stats().entries && cache.stats().retained==0,"take credited a still aliased payload");taken.restore={};}
    require(cache.stats().retained==Image::resident(16),"exclusive retained payload disappeared from cache accounting");
    auto* image=static_cast<Image*>(held.get());image->grow(33);
    require(cache.stats().bytes==Image::resident(33),"retained growth reported an arena/old extent");
    // SAVE pressure drops the reconstructible retained pool; it never drops
    // live device state. Telemetry changes when ownership actually releases.
    *available=1;auto reclaim=cache.reclaim_for(10,{});
    require(reclaim.released==1 && !held && !audit->budget->stats().used,"reclaim left retained allocation charged");
    require(!audit->physical && !audit->escaped,"alias/reclaim released charge before buffers");
    *available=1u<<20;require(cache.store(ids,logits,Image::resident(16),producer),"cache failed after reclaim");
    cache.clear();require(!audit->physical && !audit->budget->stats().used,"cache clear leaked snapshot reservation");
}
void eviction_can_make_an_alias_exclusive() {
    auto audit=std::make_shared<Audit>(1500);std::shared_ptr<void> held;
    PrefixCache cache(4,1500);
    cache.set_retained([&]{return retained_bytes(held);},[&]{held.reset();});
    const std::array<float,4> logits{1,2,3,4};
    auto producer=[&]{auto image=std::make_shared<Image>(audit,143);return std::function<void()>([image,&held]{held=image;});};
    require(cache.store(std::array<int32_t,2>{0,1},logits,700,producer),"first alias-transition fixture rejected");
    cache.entry(0).restore();require(!cache.stats().retained,"fixture is not sharing its cached payload");
    require(cache.store(std::array<int32_t,2>{2,3},logits,700,producer),"second alias-transition fixture rejected");
    require(cache.stats().bytes<=1500,"eviction made a retained alias exclusive AFTER admission and escaped the cache cap");
    cache.clear();require(!audit->physical && !audit->budget->stats().used && !audit->escaped,"alias transition leaked live payload");
}
}
int main(){try{short_images_and_growth();failures_and_release_order();alias_take_drop_and_reclaim();eviction_can_make_an_alias_exclusive();std::cout<<"PASS real allocation observer: short-image ledger, exact capacities, pre-allocation overlap, budget and allocator failures, buffer-before-credit destruction, alias/take/drop/reclaim and eviction alias transitions; no device\n";}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
