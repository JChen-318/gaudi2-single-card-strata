// The parked-page reuse law, executed by the port's SAVE path instead of only pinned as arithmetic:
//  * QsaState grows a retained image in place - the K/V cells before the last whole page and the completed
//    indexer rows stay, everything past those boundaries is re-read from the live pools - and the grown image
//    restores to exactly the state a full park of the same extent would have parked
//    (src/core/conversation_snapshot.cpp:147-181, layout :29-73);
//  * a retained image whose prefix was rewound and re-appended is refused, the way upstream's ancestry check
//    rejects a checkpoint of the old tail (hpu/src/qsa_state.cpp restore());
//  * PrefixCache hands the conversation it is about to RUN back to the caller with take(), so the model's
//    restored image is exclusively owned when the park at the end of the request grows it -
//    include/strata/core/conversation_cache.hpp:257-262 (upstream's take(), the `held` accounting).
// Host only: no device, no model weights, no state file.
#include "strata/hpu/parked_pages.hpp"
#include "strata/hpu/prefix_cache.hpp"
#include "strata/hpu/dense_kv.hpp"
#include "strata/hpu/qsa_state.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

using namespace strata::hpu;
namespace {
void require(bool ok,std::string message) {if(!ok) throw std::runtime_error(message);}
template<class F> void rejects_with(F&& f,const char* fragment) {
    bool caught=false;std::string what;
    try {f();} catch(const std::exception& error) {caught=true;what=error.what();}
    const std::string note=caught?("got: "+what):std::string("no exception at all");
    require(caught && what.find(fragment)!=std::string::npos,"expected a refusal naming the checked condition, "+note);
}
QsaConfig config() {
    QsaConfig c;c.query_heads=6;c.kv_heads=2;c.width=4;c.rotary=2;
    c.index_heads=2;c.index_width=4;c.block=4;c.top_k=64;c.capacity=64;c.page_size=4;return c;
}
void append(QsaState& s,int n) {
    const auto& c=s.config();std::vector<float> k(c.kv_heads*c.width),v(k.size()),idx(c.index_width),gamma(idx.size(),1);
    for(size_t i=0;i<k.size();++i) {k[i]=float(n+1)*0.0137f+float(i)*0.021f;v[i]=float(n-int(i))*0.137f;}
    for(size_t i=0;i<idx.size();++i) idx[i]=float((n+3)*(int(i)+2)%11)-4;
    s.append(k,v,idx,gamma);
}
// The parked layout the SAVE path builds for this state: the host F16 pool image with the indexer part present,
// exactly the switch set Layer::snapshot_reused passes.
ParkedKvLayout layout_of(const QsaState& state,uint64_t cells) {
    const auto& c=state.config();ParkedKvLayout out;std::string error;
    if(!parked_kv_layout(false,false,false,false,false,cells,true,c.width,c.kv_heads,c.index_width,out,error))
        throw std::runtime_error("parked layout refused: "+error);
    return out;
}
// The park the SAVE path performs: the retained image's sizes go into the law, the law's boundaries come out,
// and the image is grown in place. `reused` reports the bytes that were NOT re-read from the pools.
QsaState::Checkpoint park_reused(const QsaState& state,QsaState::Checkpoint image,uint64_t unchanged,uint64_t& reused) {
    uint64_t sizes[5]={};state.snapshot_sizes(image,sizes);
    const auto layout=layout_of(state,state.size());
    ParkedKvReuse decision;std::string error;
    if(!parked_kv_reuse(layout,sizes,unchanged,decision,error)) throw std::runtime_error("parked reuse refused: "+error);
    for(unsigned part=0;part<5;++part)
        if(decision.keep[part]+decision.copied[part]!=layout.sizes[part]) throw std::runtime_error("a parked part is neither kept nor copied");
    const uint64_t per_cell=layout.sizes[0]/layout.cells;
    const auto whole_cells=uint32_t(decision.keep[0]/per_cell);
    const auto kept_rows=uint32_t(decision.keep[4]/(uint64_t(sizeof(float))*layout.idx_dim));
    if(whole_cells!=uint32_t((unchanged/layout.page_size)*layout.page_size)) throw std::runtime_error("the K/V boundary is not whole pages");
    const auto grown=state.snapshot_reused(std::move(image),whole_cells,kept_rows);
    reused=decision.reused_bytes;   // only a park that actually grew may report anything
    return grown;
}
struct ParkedView {
    std::vector<float> key,value,mask,pooled;
};
// What a parked image *is*, seen from outside the private Checkpoint: the state it restores into. Two images are
// equal when restoring either leaves the same gathered K/V and the same pooled indexer rows.
ParkedView view_of(QsaState& state,const QsaState::Checkpoint& image,uint32_t cells) {
    state.restore(image);
    std::vector<int32_t> ids(cells);std::iota(ids.begin(),ids.end(),0);
    const auto gathered=state.gather(ids,cells);
    const auto pooled=state.pooled_rows();
    return ParkedView{gathered.key,gathered.value,gathered.mask,std::vector<float>(pooled.begin(),pooled.end())};
}
bool equal(const ParkedView& a,const ParkedView& b) {return a.key==b.key && a.value==b.value && a.mask==b.mask && a.pooled==b.pooled;}
void grow_in_place_matches_a_full_park() {
    const auto c=config();QsaState state(c);
    for(int i=0;i<21;++i) append(state,i);
    const auto parked_at_21=state.snapshot();                 // the image the next park grows
    const uint64_t parked_cells=state.size();
    for(int i=21;i<31;++i) append(state,i);                   // the same conversation, ten tokens further
    const auto reference=state.snapshot();                     // what a full park of this extent would write
    uint64_t reused=0;
    const auto grown=park_reused(state,parked_at_21,parked_cells,reused);
    uint64_t capacity_parts[5]={};state.snapshot_sizes(grown,capacity_parts);
    uint64_t capacity_expected=uint64_t(c.block)*c.index_width*sizeof(float);
    for(auto bytes:capacity_parts) capacity_expected+=bytes;
    require(QsaState::snapshot_capacity_bytes(grown)==capacity_expected,"grown QSA image retained excess capacity beyond its priced extent");
    require(reused>0,"the reuse carried nothing over");
    const auto first=view_of(state,grown,31),second=view_of(state,reference,31);
    require(equal(first,second),"the grown image parks a different state than a full park of the same extent");
    // The page cut really is a page cut: 21 unchanged tokens keep 20 cells, not 21.
    const auto layout=layout_of(state,31);
    uint64_t sizes[5]={};state.snapshot_sizes(parked_at_21,sizes);
    ParkedKvReuse decision;std::string error;
    require(parked_kv_reuse(layout,sizes,21,decision,error),"the law refused its own reuse case: "+error);
    require(decision.keep[0]/(layout.sizes[0]/layout.cells)==20,"the K/V boundary dropped the wrong number of cells");
    require(decision.keep[4]/(sizeof(float)*layout.idx_dim)==5,"the indexer boundary dropped the wrong number of blocks");
    require(decision.copied[0]>0,"the park re-read nothing past the boundary");
    // An unchanged re-park keeps everything it already had and still describes the same state.
    uint64_t again_reused=0;
    const auto again=park_reused(state,reference,31,again_reused);
    require(equal(view_of(state,again,31),second),"an unchanged re-park changed the parked state");
    require(again_reused>reused,"a re-park of the whole extent reused less than a partial one");
}
void rewritten_prefix_is_refused() {
    const auto c=config();QsaState state(c);
    for(int i=0;i<8;++i) append(state,i);
    const auto rewind=state.checkpoint();                     // strictly inside the image parked below
    for(int i=8;i<12;++i) append(state,i);
    const auto image=state.snapshot();                        // parked at 12 cells
    for(int i=12;i<20;++i) append(state,i);
    state.restore(rewind);                                     // rewind back to 8...
    for(int i=50;i<59;++i) append(state,i);                    // ...and re-append a different tail (17 cells)
    uint64_t reused=0;
    // The ancestry at the parked extent no longer holds, so nothing may be trusted from that image.
    rejects_with([&]{ park_reused(state,image,12,reused); },"stale/foreign");
    // And an unchanged count past the live extent is refused before any image is even consulted.
    rejects_with([&]{ park_reused(state,image,20,reused); },"invalid unchanged prefix");
    require(reused==0,"a refused park reported reused bytes");
    // A park from the current ancestry still works: re-park at 17 and grow to 20.
    const auto current=state.snapshot();
    for(int i=17;i<20;++i) append(state,i);
    const auto grown=park_reused(state,current,17,reused);
    require(reused>0,"the rewritten-prefix case lost the ability to reuse a fresh image");
    require(equal(view_of(state,grown,20),view_of(state,state.snapshot(),20)),"a park after a rewritten tail is wrong");
}
void foreign_image_is_refused() {
    const auto c=config();QsaState a(c),b(c);
    for(int i=0;i<8;++i) {append(a,i);append(b,i);}
    const auto image=a.snapshot();
    for(int i=8;i<16;++i) append(b,i);
    uint64_t reused=0;
    rejects_with([&]{ b.snapshot_reused(QsaState::Checkpoint(image),8,2); },"stale/foreign");
    require(reused==0,"a foreign image reported reused bytes");
}
// Upstream charges the retained K/V inside the cache's own byte total (conversation_cache.hpp:207) and releases
// it BEFORE evicting a parked conversation, because it is pure reconstruction (make_room, :266-268; and
// core::session_save_reclaim starts by dropping exactly this pool).
struct Owner {
    uint64_t held=0;
    uint32_t drops=0;
    void set(PrefixCache& cache) {
        cache.set_retained([this]{return held;},[this]{++drops;held=0;});
    }
};
void retained_image_is_charged_and_dropped_first() {
    std::vector<int32_t> a{1,2},bc{3,4};
    std::vector<float> logits(8,0.5f);
    auto producer=[] {return std::function<void()>([]{});};
    Owner owner;
    const uint64_t cost=4000,limit=10000;
    PrefixCache cache(4,limit);
    owner.set(cache);
    owner.held=5000;
    require(cache.store(a,logits,cost,producer),"the first park was refused");
    // bytes() is the parked entries plus what the owner is holding for the next park.
    require(cache.stats().bytes==cache.entry(0).bytes+5000,"the retained image is not charged against the cache");
    require(cache.stats().retained==5000,"the retained bytes are not reported");
    // An incoming park that fits the conversations but not the conversations PLUS the retained image costs the
    // retained image, not an entry: upstream's order.
    auto before=cache.stats();
    owner.held=5000;
    require(cache.store(bc,logits,cost,producer),"the park that could pay by releasing the reuse pool was refused");
    auto after=cache.stats();
    require(owner.drops==1,"the retained image survived a park that could not pay for itself with it dropped");
    require(after.evicted-before.evicted==0,"an entry was evicted to pay for a park the reuse pool could cover");
    require(after.reuse_dropped-before.reuse_dropped==1,"the reuse drop is not reported");
    require(after.entries==2,"the park should have landed");
    // clear() drops the slot too (conversation_cache.hpp:294-300).
    cache.clear();
    require(owner.drops==2,"clear() kept the retained image alive");
    // SAVE admission releases the pool first, and only then touches parked caches: the probe says "plenty" while
    // the park happens, "too little" for the save's first reading, and "enough again" right after the release -
    // so a reclaim that evicted an entry instead would be caught by the counters below.
    Owner save_owner;save_owner.held=6000;
    uint32_t readings=0;
    const uint64_t need=1ull<<30;
    auto probe=[&]() -> std::optional<uint64_t> {
        ++readings;
        return std::optional<uint64_t>(readings==3?(need>>4):(need<<6));
    };
    // A roomy byte limit: this fixture is about the RAM admission, and the byte rule must not fire here - it
    // is the previous fixture's subject.
    PrefixCache tight(4,100000,probe);
    tight.set_retained([&]{return save_owner.held;},[&]{++save_owner.drops;save_owner.held=0;});
    require(tight.store(a,logits,cost,producer),"tight cache first park refused");
    const auto saved=tight.stats();
    require(saved.retained==6000,"the save fixture's retained image is not charged");
    auto reclaim=tight.reclaim_for(need,a);
    require(reclaim.telemetry && reclaim.enough,"the save should admit once the pool is released");
    require(reclaim.released==1 && reclaim.bytes==6000,
            "the SAVE admission did not release the reuse pool alone: released "+std::to_string(reclaim.released)+
            " bytes "+std::to_string(reclaim.bytes));
    require(save_owner.drops==1,"the SAVE admission did not ask the owner for its pool");
    const auto released=tight.stats();
    require(released.evicted==saved.evicted,"the SAVE admission evicted a parked conversation instead of the pool");
    require(released.entries==saved.entries && released.bytes==saved.bytes-6000,
            "the release is not reflected in the charged total");
    require(released.reuse_dropped==saved.reuse_dropped+1,"the reuse drop is not reported by the reclaim path");
}
// The draft pool parks the same way, but with upstream's cap: the layer that is still writing its last page
// re-reads it (src/core/conversation_state.cpp:291-294), so one token of an exact-page boundary costs a whole page
// of reuse. `MtpBlock::snapshot_reused` applies that cap itself; here it is reproduced the way the SAVE path
// reaches it, so the arithmetic and the grown image are both pinned.
void draft_pool_reuses_with_the_cap() {
    const uint32_t capacity=64,heads=2,width=4,page=4;
    auto row=[&](int n) {
        std::vector<float> values(uint64_t(heads)*width);
        for(size_t i=0;i<values.size();++i) values[i]=float(n+1)*0.031f+float(i)*0.017f;
        return values;
    };
    DenseKvState pool(capacity,heads,width);
    require(pool.parked_bytes(0)==0 && pool.parked_bytes(8)==8*heads*width*4 && !pool.size(),"draft extent pricing depends on the current live position");
    const auto empty=pool.snapshot_reused(pool.snapshot(),0);
    require(!DenseKvState::snapshot_capacity_bytes(empty),"empty draft reuse allocated a pool");
    pool.restore(empty);require(!pool.size(),"empty draft reuse advanced its sequence");
    rejects_with([&]{(void)pool.parked_bytes(capacity+1);},"exceeds capacity");
    for(int i=0;i<8;++i){const auto k=row(i),v=row(-i);pool.append(k,v);}
    const auto parked=pool.snapshot();                       // 8 cells
    for(int i=8;i<12;++i){const auto k=row(i),v=row(-i);pool.append(k,v);}
    const auto reference=pool.snapshot();                    // full park at 12
    uint64_t sizes[5]={};pool.snapshot_sizes(parked,sizes);
    const auto layout=[&](uint64_t upto) {
        ParkedKvLayout l;std::string e;
        if(!parked_kv_layout(false,false,false,true,false,upto,false,width,heads,1,l,e))
            throw std::runtime_error("draft layout: "+e);
        return l;
    };
    const auto current=layout(12);
    ParkedKvReuse uncapped, capped;std::string error;
    require(parked_kv_reuse(current,sizes,8,uncapped,error),"draft reuse refused its own case: "+error);
    require(parked_kv_reuse(current,sizes,7,capped,error),"draft reuse refused the capped case: "+error);
    const uint64_t per_cell=current.sizes[0]/current.cells;
    require(uncapped.keep[0]/per_cell==8,"the uncapped draft reuse did not reach the page boundary");
    require(capped.keep[0]/per_cell==4,"the cap did not drop exactly one page: kept "
            +std::to_string(capped.keep[0]/per_cell)+" cells");
    require(uncapped.keep[0]-capped.keep[0]==per_cell*page,"the cap did not cost exactly one page");
    require(capped.copied[0]+capped.copied[1]>uncapped.copied[0]+uncapped.copied[1],
            "the capped park reads no more than the uncapped one");
    // The image the SAVE path grows under the cap still restores to the live draft state.
    const auto grown=pool.snapshot_reused(DenseKvState::Checkpoint(parked),uint32_t(capped.keep[0]/per_cell));
    require(DenseKvState::snapshot_capacity_bytes(grown)==pool.parked_bytes(),"grown draft image retained unpriced capacity");
    pool.restore(grown);
    require(pool.size()==12,"the grown draft image did not reach the live extent");
    const auto after_cap=pool.gather_prefix(12,12);
    pool.restore(reference);
    const auto after_full=pool.gather_prefix(12,12);
    require(after_cap.key==after_full.key && after_cap.value==after_full.value,
            "the capped draft park restores a different state than a full park");
    // An image from another pool, or one whose ancestry was rewritten, is refused.
    DenseKvState other(capacity,heads,width);
    for(int i=0;i<6;++i){const auto k=row(i),v=row(-i);other.append(k,v);}
    rejects_with([&]{ other.snapshot_reused(DenseKvState::Checkpoint(parked),4); },"is foreign");
    // A rewind strictly inside the parked image plus a re-append that passes its extent rewrites the ancestry.
    DenseKvState branch(capacity,heads,width);
    for(int i=0;i<5;++i){const auto k=row(i),v=row(-i);branch.append(k,v);}
    const auto rewind=branch.checkpoint();                   // five cells, inside the image parked next
    for(int i=5;i<8;++i){const auto k=row(i),v=row(-i);branch.append(k,v);}
    const auto branch_image=branch.snapshot();               // parked at 8
    for(int i=8;i<12;++i){const auto k=row(i),v=row(-i);branch.append(k,v);}
    branch.restore(rewind);                                  // back to five...
    for(int i=90;i<95;++i){const auto k=row(i),v=row(-i);branch.append(k,v);}   // ...and re-appended past eight
    rejects_with([&]{ branch.snapshot_reused(DenseKvState::Checkpoint(branch_image),4); },"rewritten prefix");
    // Shrinking below the image's own extent is the third, separately named refusal.
    branch.reset();
    for(int i=0;i<4;++i){const auto k=row(i),v=row(-i);branch.append(k,v);}
    rejects_with([&]{ branch.snapshot_reused(DenseKvState::Checkpoint(branch_image),4); },"exceeds the live extent");
}
// Upstream serves a parked conversation OUT of the cache: take() moves it (conversation_cache.hpp:257-262) and
// the park at the end of the request puts it back. The image the model restored from is then its own to grow, so
// the reuse branch of the SAVE path has something exclusively owned to grow.
struct Fixture {
    std::vector<std::string> order;
    bool released=false;
    uint32_t captures=0;
    struct Guard {
        bool* released;
        ~Guard() {*released=true;}
    };
    std::function<void()> produce(const char* name) {
        ++captures;order.push_back(name);
        return [guard=std::make_shared<Guard>(Guard{&released})]{};
    }
};
void take_hands_the_parked_image_back_to_the_model() {
    Fixture fixture;std::vector<int32_t> a{1,2},ab{1,2,3};
    std::vector<float> logits(8,0.5f);
    PrefixCache cache(4,1u<<24);
    require(cache.store(a,logits,1000,[&]{return fixture.produce("first");}),"the first park was refused");
    const auto hit=cache.find(ab);
    require(hit.has_value(),"the parked prefix is not a hit for the longer prompt");
    const auto parked_bytes=cache.stats().bytes;
    require(parked_bytes>0,"the cache charges nothing for a parked conversation");
    auto taken=cache.take(*hit);
    auto stats=cache.stats();
    require(stats.entries==0 && stats.bytes==0,"take() left the conversation charged as parked");
    require(stats.taken==1 && stats.evicted==0,"serving a parked conversation is not an eviction");
    require(taken.tokens==a,"take() did not move the parked prefix out");
    // The model restores from the image, then every caller-side reference dies: that is the moment the image
    // becomes the model's alone, and the park that follows may grow it in place.
    auto restore=taken.restore;
    taken=PrefixCache::Entry{};
    restore();
    restore=nullptr;
    require(fixture.released,"the taken image is still referenced after its last owner went away");
    require(cache.store(ab,logits,1000,[&]{
        fixture.order.push_back(fixture.released?"second:exclusive":"second:aliased");
        return fixture.produce("second");
    }),"the re-park was refused");
    require(fixture.order==std::vector<std::string>{"first","second:exclusive","second"},
            "the re-park did not run with the taken image released");
    stats=cache.stats();
    require(stats.entries==1 && stats.saved==2,"the re-park did not put the conversation back");
    require(stats.taken==1,"the take is not reported");
    // A conversation that is hit but not taken stays parked, so a second request can come back to it.
    PrefixCache kept(4,1u<<24);
    require(kept.store(a,logits,1000,[&]{return fixture.produce("keep:first");}),"keep park refused");
    require(kept.find(ab).has_value() && kept.find(a).has_value(),"a parked conversation stopped being hittable");
    require(kept.stats().entries==1 && kept.stats().taken==0,"finding a conversation took it out");
    // The chain law still runs over what is left: taking the only entry leaves the slot free, not a hole.
    auto vacated=cache;
    require(vacated.stats().entries==1,"the copy of the cache is not the parked one");
    (void)vacated.take(*vacated.find(ab));
    require(vacated.stats().entries==0 && vacated.stats().bytes==0,"the second take left stale accounting");
    require(cache.store(std::vector<int32_t>{7,6},logits,1000,[&]{return fixture.produce("other");}),"a fresh park was refused");
    require(cache.stats().entries==2,"a different conversation should occupy the freed slot");
}
}
// A valid branch's prefix ID is not its length. Persistent restore must carry
// the tip identity into the live pool, while reset must never recycle old IDs.
void restored_branches_remain_reusable() {
    QsaState qsa(config());
    for(int i=0;i<4;++i) append(qsa,i);
    const auto root=qsa.checkpoint();
    for(int i=4;i<12;++i) append(qsa,i);
    qsa.restore(root);
    for(int i=40;i<44;++i) append(qsa,i);
    const auto branch=qsa.snapshot();
    qsa.restore(branch);
    for(int i=8;i<12;++i) append(qsa,i);
    const auto full=qsa.snapshot();uint64_t reused=0;
    const auto grown=park_reused(qsa,branch,8,reused);
    require(reused>0,"restored QSA branch reused no pages");
    require(equal(view_of(qsa,grown,12),view_of(qsa,full,12)),"restored QSA branch park differs from full image");
    qsa.reset();for(int i=50;i<62;++i) append(qsa,i);
    rejects_with([&]{qsa.snapshot_reused(QsaState::Checkpoint(branch),8,2);},"stale/foreign");

    DenseKvState draft(64,2,4);std::array<float,8> row{};
    for(int i=0;i<4;++i) draft.append(row,row);
    const auto draft_root=draft.checkpoint();
    for(int i=4;i<12;++i) draft.append(row,row);
    draft.restore(draft_root);
    for(int i=4;i<8;++i) {row.fill(float(i));draft.append(row,row);}
    const auto image=draft.snapshot();draft.restore(image);
    for(int i=8;i<12;++i) {row.fill(float(i));draft.append(row,row);}
    const auto expected=draft.gather_prefix(12,12);
    const auto grown_draft=draft.snapshot_reused(DenseKvState::Checkpoint(image),8);
    draft.restore(grown_draft);const auto actual=draft.gather_prefix(12,12);
    require(actual.key==expected.key && actual.value==expected.value,"restored draft branch park differs from full image");
    draft.reset();for(int i=0;i<12;++i) draft.append(row,row);
    rejects_with([&]{draft.snapshot_reused(DenseKvState::Checkpoint(image),8);},"rewritten prefix");
}
int main() {
    try {
        grow_in_place_matches_a_full_park();
        restored_branches_remain_reusable();
        rewritten_prefix_is_refused();
        foreign_image_is_refused();
        retained_image_is_charged_and_dropped_first();
        draft_pool_reuses_with_the_cap();
        take_hands_the_parked_image_back_to_the_model();
        std::cout<<"PASS parked park: grow-in-place reuse restores the same state a full park parks, rewritten and "\
"foreign images are refused, and a parked image is taken out before the re-park grows it\n";
        return 0;
    } catch(const std::exception& error) {
        std::cout<<"FAIL parked park: "<<error.what()<<"\n";
        return 1;
    }
}
