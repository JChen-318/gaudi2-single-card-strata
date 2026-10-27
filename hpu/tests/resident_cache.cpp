#include "strata/hpu/resident_cache.hpp"
#include <atomic>
#include <iostream>
#include <thread>
using namespace strata::hpu;
namespace {
void require(bool ok,const char* message) {if(!ok) throw std::runtime_error(message);}
template<class F> void rejects(F&& f) {bool rejected=false;try {f();} catch(const std::exception&) {rejected=true;}require(rejected,"invalid cache operation accepted");}
struct Resource {
    static inline std::atomic<int> alive=0;
    int id;
    explicit Resource(int value):id(value) {++alive;}
    ~Resource() {--alive;}
};
}
int main() {
    try {
        ResidentCache<Resource> cache(12);int calls=0;
        auto load=[&](int id) {return [&,id]()->std::unique_ptr<const Resource> {++calls;return std::make_unique<Resource>(id);};};
        auto a=cache.acquire("a",4,load(1)),b=cache.acquire("b",4,load(2));const auto a_generation=a.generation();
        auto again=cache.acquire("a",4,load(9));require(calls==2 && again->id==1 && again.generation()==a_generation,"hit reloaded/changed generation");
        again={};a={};
        auto c=cache.acquire("c",8,load(3));
        require(calls==3 && cache.stats().resident_bytes==12 && cache.stats().evictions==1,"LRU must skip leased b and evict a");
        rejects([&]{(void)cache.acquire("d",1,load(4));});require(calls==3 && cache.stats().resident_bytes==12,"busy cache must not allocate/evict");
        rejects([&]{cache.clear();});rejects([&]{(void)cache.acquire("b",5,load(2));});
        b={};c={};auto reloaded=cache.acquire("a",4,load(7));
        require(reloaded.generation()>a_generation && reloaded->id==7,"ABA generation/fresh weight content");
        reloaded={};
        rejects([&]{(void)cache.acquire("failure",12,[]()->std::unique_ptr<const Resource> {throw std::runtime_error("synthetic load failure");});});
        require(cache.stats().resident_bytes==0 && cache.stats().failed_loads==1 && Resource::alive==0,"failed load leaked reservation/resource");
        rejects([&]{(void)cache.acquire("null",4,[]()->std::unique_ptr<const Resource> {return {};});});
        rejects([&]{(void)cache.acquire("huge",13,load(1));});
        // Concurrent hits have one published immutable generation/one loader.
        std::atomic<int> loads=0,errors=0;
        auto threaded=[&] {
            try {for(int i=0;i<100;++i) {auto lease=cache.acquire("shared",4,[&]()->std::unique_ptr<const Resource> {++loads;return std::make_unique<Resource>(42);});if(lease->id!=42) ++errors;}}
            catch(...) {++errors;}
        };
        std::thread t1(threaded),t2(threaded),t3(threaded);t1.join();t2.join();t3.join();
        require(loads==1 && errors==0 && cache.stats().peak_bytes<=12,"concurrent duplicate publication/budget");cache.clear();require(Resource::alive==0,"cache clear leaked resource");
        ResidentCache<Resource>::Lease survivor;
        {ResidentCache<Resource> temporary(4);survivor=temporary.acquire("x",4,load(5));}
        require(Resource::alive==1 && survivor->id==5,"lease must own resource beyond cache lifetime");survivor={};
        require(Resource::alive==0,"survivor release leaked resource");
        {
            ResidentCache<Resource> partial(12);
            {auto cold=partial.acquire("cold",4,load(1));}
            auto pinned=partial.acquire("pinned",8,load(2));
            rejects([&]{(void)partial.acquire("cannot_fit",8,load(3));});
            require(partial.stats().resident_bytes==12 && partial.stats().evictions==0 && partial.stats().loads==2,
                    "capacity failure evicted a partial victim set");
        }
        require(Resource::alive==0,"partial victim test leaked resources");
        {
            ResidentCache<Resource> lookup(4);require(!lookup.find("missing"),"resident lookup invoked a load");
            {auto entry=lookup.acquire("hit",4,load(7));}
            auto hit=lookup.find("hit");require(hit && hit->id==7 && lookup.stats().loads==1,"resident lookup changed immutable payload");
            rejects([&]{(void)lookup.acquire("other",4,load(8));});hit={};
            (void)lookup.acquire("other",4,load(8));require(!lookup.find("hit"),"evicted entry found");
        }
        {
            ResidentCache<Resource> elastic(12);{auto a=elastic.acquire("a",4,load(1));}
            auto b=elastic.acquire("b",8,load(2));rejects([&]{elastic.resize(4);});
            require(elastic.capacity()==12 && elastic.stats().resident_bytes==12 && elastic.stats().evictions==0,"failed resize mutated cache");
            elastic.resize(20);require(elastic.capacity()==20 && b->id==2,"growing cache changed active lease");
            b={};elastic.resize(4);require(elastic.capacity()==4 && elastic.stats().resident_bytes<=4,"cache shrink did not release payload");
            {auto a=elastic.acquire("again",4,load(3));require(a->id==3,"reallocation after shrink");}
            rejects([&]{elastic.resize(0);});
        }
        std::cout<<"PASS bounded LRU, active consumer pinning, immutable generations, ABA reload, atomic capacity failure\n"
                   "PASS loader failures, concurrent single publication, lease lifetime and exact resource accounting\n";
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
