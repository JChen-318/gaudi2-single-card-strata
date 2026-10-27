#include "strata/hpu/generation.hpp"
#include "strata/hpu/snapshot_memory.hpp"
#include <cmath>
#include <iostream>
#include <memory>
using namespace strata::hpu;
namespace {
void need(bool ok,const char* why) {if(!ok) throw std::runtime_error(why);}
template<class F> void rejects(F f) {bool failed=false;try {f();} catch(const std::exception&) {failed=true;}need(failed,"invalid scalar transaction admitted");}
struct Model {
    struct Image {SnapshotReservation charge;std::vector<int32_t> ids;Image(std::shared_ptr<AllocationBudget> p):charge(std::move(p),100) {}};
    std::shared_ptr<AllocationBudget> pool=std::make_shared<AllocationBudget>();std::vector<int32_t> ids;
    uint32_t calls=0,captures=0,restores=0,bad_at=0,cancel_at=0,fail_at=0,fatal_at=0;
    bool active=false,stop=false,fail_restore=false,wrong_first=false;
    explicit Model(uint64_t limit=100) {pool->configure(limit);ids.reserve(32);}
    std::vector<float> logits() const {std::vector<float> x(16,-10);x[(ids.back()+1)%16]=10;return x;}
    GenerationOperations operations() {
        GenerationOperations op;op.reset=[&]{ids.clear();active=false;};
        op.decode=[&](int32_t id,const auto& cancel) {
            if(cancel && cancel()) throw Cancelled();
            ids.push_back(id);
            if(active) {++calls;if(calls==fatal_at) throw SessionFatal("failed physical candidate state");if(calls==fail_at) throw std::runtime_error("injected candidate/replay failure");if(calls==cancel_at) stop=true;}
            auto out=logits();if(active && calls==bad_at) out[0]=NAN;return out;
        };
        op.checkpoint=[&]()->std::function<void()> {
            auto saved=std::make_shared<Image>(pool);saved->ids=ids;++captures;calls=0;active=true;
            return [&,saved] {++restores;if(fail_restore) throw std::runtime_error("injected owned root restore failure");ids=saved->ids;};
        };
        op.checkpoint_bytes=100;op.available_host_bytes=[] {return std::optional<uint64_t>(UINT64_MAX);};
        op.draft=[&](const DraftContext& context,uint32_t maximum,const auto&) {
            std::vector<DraftRow> out;for(uint32_t i=0;i<maximum;++i) out.push_back({int32_t((context.history.back()+1+i+(wrong_first && i==0?5:0))%16),{},{}});return out;
        };return op;
    }
};
GenerationOptions options(bool cache=false) {GenerationOptions o;o.max_draft=2;o.cache_slots=cache?1:0;o.cache_bytes=10000;o.cache_every=0;o.boundaries.turn_token=-1;return o;}
GenerationRequest request() {GenerationRequest r;r.prompt={0,1};r.max_new=6;r.sampling.greedy=true;return r;}
void nonfinite_all_rows() {
    for(bool mismatch:{false,true}) for(int32_t eos:{-1,2}) for(uint32_t at:{1u,2u}) {
        Model model;model.wrong_first=mismatch;model.bad_at=at;
        Generation gen(16,32,eos<0?std::vector<int32_t>{}:std::vector<int32_t>{eos},model.operations(),options());
        std::vector<int32_t> out;GenerationEvents events;events.token=[&](int32_t id){model.active=false;out.push_back(id);};
        const auto result=gen.run(request(),events);
        need(out.empty() && !result.generated && !result.drafts_checked && !result.drafts_accepted && result.finish=="nonfinite" && gen.nonfinite_pending(),"later physical nonfinite row leaked an earlier mismatch/accepted/EOS token");
        need(model.ids==request().prompt && model.calls==at && model.restores==1,"failed scalar candidate window did not restore its exact root once");
        need(!model.pool->stats().used && model.pool->stats().peak==100,"scalar rollback leaked or exceeded the hard image pool");
        model.bad_at=0;model.active=false;auto next=request();next.max_new=1;next.prompt.push_back(5);
        need(gen.run(next,events).reused==0,"nonfinite window preserved a reusable pre-failure prefix");
    }
}
void cancellation() {
    for(uint32_t at:{1u,2u,3u,4u}) {
        Model model;model.cancel_at=at;Generation gen(16,32,{},model.operations(),options());
        GenerationEvents events;bool emitted=false;events.token=[&](int32_t){emitted=true;};
        const auto result=gen.run(request(),events,[&]{return model.stop;});
        need(!emitted && !result.generated && result.finish=="cancelled" && model.ids==request().prompt,"cancel before commit leaked or retained speculative state");
        need(!model.pool->stats().used,"cancelled scalar transaction kept its root image");
        model.stop=false;model.active=false;model.cancel_at=0;auto resumed=request();resumed.max_new=1;resumed.prompt.push_back(5);
        need(gen.run(resumed,events).reused==2,"cancelled uncommitted window lost the valid root prefix");
    }
    Model model;Generation gen(16,32,{},model.operations(),options());std::vector<int32_t> out;
    GenerationEvents events;events.token=[&](int32_t id){need(model.ids==std::vector<int32_t>({0,1,2,3}),"output began before the selected prefix committed");need(!model.pool->stats().used,"transaction image survived into publication");model.active=false;out.push_back(id);model.stop=true;};
    const auto committed=gen.run(request(),events,[&]{return model.stop;});
    need(out==std::vector<int32_t>({2,3,4}) && committed.generated==3 && committed.finish=="cancelled" && model.ids==std::vector<int32_t>({0,1,2,3}),"cancel during committed publication truncated the window or consumed bonus");
}
void failures_and_pool() {
    for(uint32_t at:{1u,3u}) {
        Model model;model.fail_at=at;Generation gen(16,32,{},model.operations(),options());bool emitted=false;GenerationEvents events;events.token=[&](int32_t){emitted=true;};
        rejects([&]{gen.run(request(),events);});need(!emitted && model.ids==request().prompt && !model.pool->stats().used,"candidate/replay exception did not restore owned root before propagating");
    }
    for(bool restore_failure:{false,true}) {
        Model model;model.fail_restore=restore_failure;model.fatal_at=restore_failure?0:1;Generation gen(16,32,{},model.operations(),options());
        GenerationEvents events;bool emitted=false;events.token=[&](int32_t){emitted=true;};bool fatal=false;
        try {gen.run(request(),events);} catch(const SessionFatal&) {fatal=true;}
        need(fatal && !emitted && model.restores==uint32_t(restore_failure) && !model.pool->stats().used,"physical/restore fatal was hidden or root restore was retried");
        rejects([&]{gen.attach_live();});
    }
    Model tight(1);Generation refused(16,32,{},tight.operations(),options());bool emitted=false;GenerationEvents events;events.token=[&](int32_t){emitted=true;};
    bool budget=false;try {refused.run(request(),events);} catch(const AllocationBudgetExceeded&) {budget=true;}
    need(budget && !emitted && tight.ids==request().prompt && !tight.captures && !tight.pool->stats().peak,"budget refusal evaluated a speculative row or changed the hard cap");
    Model missing;auto op=missing.operations();op.checkpoint={};Generation absent(16,32,{},op,options());rejects([&]{absent.run(request(),events);});need(!emitted && missing.ids==request().prompt,"missing producer was silently replaced by unbounded replay");
    Model alias;Generation cached(16,32,{},alias.operations(),options(true));auto once=request();once.max_new=3;
    events.token=[&](int32_t){alias.active=false;need(alias.captures==1 && alias.pool->stats().used==100,"scalar transaction duplicated a cached exact root image");};
    need(cached.run(once,events).generated==3 && alias.pool->stats().peak==100,"exact-root alias did not keep scalar verification within one-image budget");
    need(generation_scalar_window_bytes(32,16)==(32+16)*4+4096,"scalar metadata plan lost token/logit root copies");rejects([]{generation_scalar_window_bytes(0,16);});
}
}
int main() {
    try {nonfinite_all_rows();cancellation();failures_and_pool();std::cout<<"PASS scalar whole-window publication: rejected/EOS later-row numerical failures, candidate/replay/final-gap cancel, native-style committed publication, exact owned rollback, fatal propagation and hard-pool/root-alias admission; no device\n";}
    catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
