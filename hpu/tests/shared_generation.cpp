#include "strata/hpu/slot_session.hpp"
#include "strata/hpu/snapshot_memory.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
using namespace strata::hpu;
namespace {
void need(bool ok,const char* message) {if(!ok) throw std::runtime_error(message);}
template<class F> void rejects(F f) {bool failed=false;try {f();} catch(const std::exception&) {failed=true;}need(failed,"invalid shared session accepted");}
std::vector<float> target(std::span<const int32_t> ids,bool mode) {
    uint32_t hash=mode?17:3;for(auto id:ids) hash=(hash*3+uint32_t(id))%31;
    std::vector<float> result(32);for(uint32_t i=0;i<32;++i) result[i]=float((hash*13+i*i*3+i)%47)/9;return result;
}
struct Model {
    struct Image {
        SnapshotReservation reservation;std::vector<int32_t> ids;bool mode;
        Image(std::shared_ptr<AllocationBudget> pool):reservation(std::move(pool),100),mode(true) {}
    };
    std::shared_ptr<AllocationBudget> budget=std::make_shared<AllocationBudget>();std::shared_ptr<Image> retained;
    std::vector<int32_t> ids;bool mode=true,in_callback=false,fail_restore=false,reject_draft=false,telemetry=true;
    uint32_t captures=0,resets=0;bool bad_decode=false,bad_prefill=false,bad_verify=false;
    explicit Model(uint64_t limit=10000) {budget->configure(limit);}
    std::vector<float> logits() const {return target(ids,mode);}
    void reset() {ids.clear();retained={};++resets;}
    std::vector<float> decode(int32_t id) {ids.push_back(id);auto values=logits();if(bad_decode) values[0]=NAN;return values;}
    std::function<void()> checkpoint() {
        need(!in_callback,"cooperate reentered Model from a locked prefill callback");++captures;
        auto image=retained && retained.use_count()==1?std::move(retained):std::make_shared<Image>(budget);
        image->ids=ids;image->mode=mode;
        return [this,image] {need(!in_callback,"restore reentered a locked Model callback");if(fail_restore) {fail_restore=false;throw std::runtime_error("injected restore failure");}ids=image->ids;mode=image->mode;retained=image;};
    }
    GenerationOperations gen_ops(SlotSession*& slots) {
        GenerationOperations op;op.reset=[&]{reset();};
        op.decode=[&](int32_t id,const auto& cancelled){if(cancelled && cancelled()) throw Cancelled();return decode(id);};
        op.prefill=[&](std::span<const int32_t> input,const PromptCommit& commit,const auto& cancelled) {
            struct Guard {bool& flag;~Guard(){flag=false;}} guard{in_callback};in_callback=true;
            for(size_t i=0;i<input.size();++i) {if(cancelled && cancelled()) throw Cancelled();auto values=decode(input[i]);if(bad_prefill) values[0]=INFINITY;commit(uint32_t(i+1),values);}
        };
        op.control_vector=[&](bool value){mode=value;};op.checkpoint=[&]{return checkpoint();};op.checkpoint_bytes=100;
        op.checkpoint_park_bytes=[](uint64_t){return 100;};op.available_host_bytes=[&] {return telemetry?std::optional<uint64_t>(UINT64_MAX):std::nullopt;};
        op.parked_retained_bytes=[&]{return retained && retained.use_count()==1?100:0;};op.discard_parked=[&]{retained={};};
        op.select_prefix=[&slots](std::span<const int32_t> prompt,uint64_t maximum,bool control,std::span<const RequestImage> images)->std::optional<GenerationPrefix> {return slots?slots->select_prefix(prompt,maximum,control,images):std::nullopt;};
        op.draft=[&](const DraftContext& context,uint32_t maximum,const auto&) {
            std::vector<DraftRow> rows;if(maximum) {auto token=sample(context.logits,context.history,context.sampling,{});if(reject_draft) token=(token+1)%32;rows.push_back({token,{},{}});}return rows;
        };
        op.verify=[&](const DraftContext& context,std::span<const DraftRow> drafts,const auto& cancelled) {
            if(bad_verify) throw NonFiniteOutput("injected native candidate row");
            need(std::equal(ids.begin(),ids.end(),context.history.begin(),context.history.end()),"native window saw another session's prefix");
            auto future=ids;std::vector<float> after;
            for(const auto& row:drafts) {future.push_back(row.token);auto values=target(future,mode);after.insert(after.end(),values.begin(),values.end());}
            auto selected=select_verified_candidates(context.logits,after,drafts,context.history,context.sampling,{},context.mode,{},cancelled);
            for(uint32_t i=0;i<selected.retained;++i) ids.push_back(drafts[i].token);
            selected.logits=logits();return selected;
        };
        return op;
    }
    SlotSessionOperations slot_ops(Generation& gen) {
        SlotSessionOperations op;op.before_foreign=[&gen]{gen.suspend_live();};
        op.select_prefix=[&gen](std::span<const int32_t> prompt,uint64_t maximum,bool control,std::span<const RequestImage> images){return gen.select_prefix(prompt,maximum,control,images);};
        op.import_prefixes=[&gen](const GenerationRead& read){gen.import_prefixes(read);};
        op.logits=[&]{return logits();};op.reset=[&]{reset();};op.decode=[&](int32_t id){return decode(id);};
        op.prefill=[&](std::span<const int32_t> input){ids.insert(ids.end(),input.begin(),input.end());return logits();};op.prefill_window=op.prefill;
        op.control_vector=[&](bool value){mode=value;};op.checkpoint=[&]{return checkpoint();};op.discard_retained=[&]{retained={};};return op;
    }
};
GenerationOptions options(uint32_t drafts=0) {GenerationOptions o;o.max_draft=drafts;o.cache_slots=4;o.cache_bytes=50000;o.prefill_chunk=2;o.short_read=0;o.cache_every=0;o.boundaries.turn_token=-1;return o;}
GenerationRequest request(std::initializer_list<int32_t> ids,bool mode=true,uint32_t count=4) {
    GenerationRequest r;r.prompt=ids;r.control_vector=mode;r.max_new=count;r.sampling.seed=41;r.sampling.counter=17;r.sampling.top_k=19;r.sampling.top_p=.86f;r.sampling.min_p=.02f;
    r.sampling.penalty_repeat=1.17f;r.sampling.penalty_freq=.09f;r.sampling.penalty_present=.1f;r.sampling.penalty_last_n=9;return r;
}
std::vector<int32_t> oracle(const GenerationRequest& r,uint32_t count) {
    auto history=r.prompt;auto p=r.sampling;std::vector<int32_t> out;
    for(uint32_t i=0;i<count;++i) {auto next=sample(target(history,r.control_vector),history,p,{});out.push_back(next);history.push_back(next);++p.counter;}return out;
}
std::pair<GenerationResult,std::vector<int32_t>> run(Generation& gen,const GenerationRequest& r,GenerationEvents e={}) {
    std::vector<int32_t> out;e.token=[&](int32_t id){out.push_back(id);};auto result=gen.run(r,e);const auto expected=oracle(r,result.generated);
    if(out!=expected) {
        std::cerr<<"mode="<<r.control_vector<<" reused="<<result.reused<<" prompt:";for(auto id:r.prompt) std::cerr<<' '<<id;
        std::cerr<<" output:";for(auto id:out) std::cerr<<' '<<id;std::cerr<<" expected:";for(auto id:expected) std::cerr<<' '<<id;std::cerr<<'\n';
        throw std::runtime_error("shared Generation differs from independent target sampler");
    }
    return {result,out};
}
int32_t admit(SlotSession& slots,int slot,const GenerationRequest& r,uint32_t reused=0) {
    int32_t first=-1;const auto result=slots.admit(slot,r,8,[&](int32_t id){first=id;});need(first==oracle(r,1)[0] && result.reused_tokens==reused,"cross-bank slot source/sampling mismatch");return first;
}
void cross_bank() {
    Model model;SlotSession* pointer=nullptr;Generation gen(32,128,{},model.gen_ops(pointer),options());
    SlotSession slots(3,32,128,{},model.slot_ops(gen),true);pointer=&slots;
    auto a=request({1,2});auto [first_result,out]=run(gen,a);(void)first_result;
    const auto main_fed=model.ids;const auto captures=model.captures;
    auto b=request({8,9,10},false);const auto first=admit(slots,0,b);
    need(model.captures==captures+1,"suspend copied an already parked end instead of sharing its owner"); // only the slot end capture
    auto follow=a;follow.prompt=main_fed;follow.prompt.push_back(out.back());follow.prompt.push_back(7);
    const auto resumed=run(gen,follow).first;need(resumed.reused==main_fed.size() && model.mode,"suspended main live was not restored");
    need(slots.step(0).token==oracle(b,2)[1] && first==oracle(b,1)[0] && !model.mode,"GEN lost an active slot's mode or pending token");slots.finish(0);
    auto from_slot=b;from_slot.prompt.push_back(first);from_slot.prompt.push_back(6);from_slot.max_new=2;
    auto result=run(gen,from_slot).first;need(result.reused==4,"GEN did not select a longer idle slot over its own main/cache");
    const auto parked=model.ids;
    auto other=request({21,22},true,2);run(gen,other);
    auto from_main=b;from_main.prompt=parked;from_main.prompt.push_back(5);
    admit(slots,1,from_main,uint32_t(parked.size())); // parked main, not current unrelated live or shorter idle
    auto active=from_main;active.prompt.push_back(4);
    const auto no_active=run(gen,active).first;need(no_active.reused<from_main.prompt.size(),"GEN borrowed an active slot's mutable image");
}
void cooperative_paths() {
    for(bool gumbel:{false,true}) for(bool reject:{false,true}) for(uint32_t drafts:{0u,1u}) {
        Model model;model.reject_draft=reject;SlotSession* pointer=nullptr;Generation gen(32,128,{},model.gen_ops(pointer),options(drafts));
        SlotSession slots(2,32,128,{},model.slot_ops(gen),true);pointer=&slots;
        auto a=request({1,2},false),b=request({5,6,7,8,9,10,11,12,13,14},true,7);a.sampling.gumbel=b.sampling.gumbel=gumbel;
        std::vector<int32_t> out{admit(slots,0,a)};GenerationEvents events;
        events.has_foreign_work=[&]{return out.size()<8;};events.cooperate=[&]{out.push_back(slots.step(0).token);};
        run(gen,b,events);need(out==oracle(a,uint32_t(out.size())) && out.size()==8,"cooperative target/window path changed slot counters/history");
        auto fed=b.prompt;const auto expected=oracle(b,b.max_new);fed.insert(fed.end(),expected.begin(),expected.end()-1);
        need(model.ids==fed && model.mode,"foreign work was not restored before GEN continued");
    }
}
void failures_and_budget() {
    Model model;SlotSession* pointer=nullptr;Generation gen(32,128,{},model.gen_ops(pointer),options());SlotSession slots(2,32,128,{},model.slot_ops(gen),true);pointer=&slots;
    auto a=request({1,2}),b=request({8,9},false);run(gen,a);const auto fed=model.ids;admit(slots,0,b);
    auto follow=a;follow.prompt=fed;follow.prompt.push_back(5);model.fail_restore=true;bool emitted=false;GenerationEvents e;e.token=[&](int32_t){emitted=true;};
    rejects([&]{gen.run(follow,e);});need(!emitted && slots.step(0).token==oracle(b,2)[1],"failed main restore emitted a token or damaged an active image");
    Model tight(100);SlotSession* ptr=nullptr;Generation limited(32,128,{},tight.gen_ops(ptr),options());SlotSession active(2,32,128,{},tight.slot_ops(limited),false);ptr=&active;
    admit(active,0,b);uint32_t work=0;GenerationEvents no_room;no_room.has_foreign_work=[] {return true;};no_room.cooperate=[&]{++work;};
    run(limited,a,no_room);need(!work && tight.budget->stats().peak==100,"cooperative temporary image exceeded hard snapshot budget");
    need(active.step(0).token==oracle(b,2)[1],"unfunded main suspension damaged a running slot");
    const auto fresh=run(limited,a).first;need(!fresh.reused,"unfunded live metadata reused another session's Model");
    Model cancelled;cancelled.telemetry=false;SlotSession* cp=nullptr;Generation owner(32,128,{},cancelled.gen_ops(cp),options());SlotSession cs(2,32,128,{},cancelled.slot_ops(owner),true);cp=&cs;
    run(owner,a);const auto main=cancelled.ids;admit(cs,0,b);
    cancelled.telemetry=true;GenerationEvents none;none.token=[](int32_t){};
    need(owner.run(a,none,[]{return true;}).generated==0,"cancel-before-attach emitted");
    need(owner.cache_stats().entries==0,"cancel before attach published a foreign Model as the uncached main end");
    run(owner,request({20,21},false)); // select the old main from its CACHE next, not its live image
    auto again=a;again.prompt=main;again.prompt.push_back(7);need(run(owner,again).first.reused==0,"cancel before attach poisoned a parked main image with a foreign Model");
}
void numerical_recovery() {
    for(int failure=0;failure<3;++failure) {
        Model model;SlotSession* pointer=nullptr;auto config=options(failure==2?1:0);
        if(failure==0) config.prefill_chunk=0;
        Generation gen(32,128,{},model.gen_ops(pointer),config);
        const auto original=request({1,2,3},false),other=request({20,21},true);
        run(gen,original);const auto previous=model.ids;run(gen,other);
        need(gen.cache_stats().entries>=2,"nonfinite bank fixture missing independent conversations");
        model.bad_decode=failure==0;model.bad_prefill=failure==1;model.bad_verify=failure==2;
        GenerationEvents events;bool emitted=false;events.token=[&](int32_t){emitted=true;};
        auto bad=request({11,12,13,14},true,3);const auto failed=gen.run(bad,events);
        need(failed.finish=="nonfinite" && failed.generated==0 && !emitted && gen.nonfinite_pending(),"numerical failure did not end before its first window publication");
        rejects([&]{gen.attach_live();});model.bad_decode=model.bad_prefill=model.bad_verify=false;
        auto again=original;again.prompt=previous;again.prompt.push_back(7);
        need(run(gen,again).first.reused==0 && !gen.nonfinite_pending(),"next request reused a pre-nonfinite conversation");
        gen.mark_nonfinite();model.reset();gen.adopt({},{});need(!gen.nonfinite_pending(),"disk adoption retained the old numerical failure flag");
    }
    Model model;SlotSession* pointer=nullptr;Generation gen(32,128,{},model.gen_ops(pointer),options());
    SlotSession slots(3,32,128,{},model.slot_ops(gen),true);pointer=&slots;
    const auto idle=request({17,18},false),running=request({22,23},true),fresh=request({27,28},false);
    admit(slots,0,idle);slots.step(0);slots.finish(0);admit(slots,1,running);
    auto probe=idle;probe.prompt.push_back(oracle(idle,1).front());probe.prompt.push_back(5);
    need(bool(slots.select_prefix(probe.prompt,probe.prompt.size()-1,false)),"idle cache fixture missing");
    gen.mark_nonfinite();const auto resets=model.resets;need(gen.flush_nonfinite(),"pending numerical failure not flushed");slots.invalidate_caches();
    need(model.resets==resets && !slots.select_prefix(probe.prompt,probe.prompt.size()-1,false),"flush reset foreign Model or kept an idle owner");
    need(slots.step(1).token==oracle(running,2)[1],"global flush corrupted an active owned sequence");slots.finish(1);
    auto finished=running;finished.prompt.push_back(oracle(running,1).front());finished.prompt.push_back(5);
    need(!slots.select_prefix(finished.prompt,finished.prompt.size()-1,true),"tainted running reply became idle cache");
    admit(slots,1,fresh);slots.step(1);slots.finish(1);auto next=fresh;next.prompt.push_back(oracle(fresh,1).front());next.prompt.push_back(5);
    need(bool(slots.select_prefix(next.prompt,next.prompt.size()-1,false)),"taint leaked into a newly admitted owner");
}
void explicit_main_attachment() {
    Model model;SlotSession* pointer=nullptr;Generation gen(32,128,{},model.gen_ops(pointer),options());
    SlotSession slots(2,32,128,{},model.slot_ops(gen),true);pointer=&slots;
    rejects([&]{gen.attach_live();});
    const auto main=request({1,2,3},false),other=request({20,21},true);
    run(gen,main);const auto saved=model.ids;const auto captures=model.captures;
    admit(slots,0,other);need(model.ids!=saved && model.mode,"foreign attachment fixture did not switch Model");
    gen.attach_live();need(model.ids==saved && !model.mode && model.captures==captures+1,"disk main attachment copied or mounted the last slot");
    gen.release_checkpoints(); // save reclaim must not discard the attached state
    need(model.ids==saved && !model.mode && !gen.cache_stats().entries,"checkpoint reclaim changed attached main");
    need(slots.step(0).token==oracle(other,2)[1],"main attachment/reclaim damaged an owned active slot");
    gen.attach_live();need(model.ids==saved && !model.mode,"repeated disk attachment failed after cache release");
    auto one_shot=main;one_shot.checkpoint=false;run(gen,one_shot);rejects([&]{gen.attach_live();});
    run(gen,main);admit(slots,1,other);model.fail_restore=true;
    bool fatal=false;try {gen.attach_live();} catch(const SessionFatal&) {fatal=true;}
    need(fatal && !gen.cache_stats().entries,"failed disk attachment did not invalidate the main owner");
    rejects([&]{gen.attach_live();});need(slots.step(0).token==oracle(other,3)[2],"failed main attachment destroyed an independent image");
    Model tight(100);SlotSession* ptr=nullptr;Generation limited(32,128,{},tight.gen_ops(ptr),options());
    SlotSession active(1,32,128,{},tight.slot_ops(limited),false);ptr=&active;
    admit(active,0,other);run(limited,main);active.step(0);
    rejects([&]{limited.attach_live();});need(tight.budget->stats().peak==100,"unfunded disk attachment exceeded native snapshot pool");
}
void solo_handoff() {
    Model model;SlotSession* pointer=nullptr;auto config=options();config.cache_slots=3;config.cache_every=2;config.boundaries.turn_token=31;config.boundaries.root_min_tokens=2;
    Generation gen(32,128,{},model.gen_ops(pointer),config);SlotCheckpointOptions sc;sc.prefill_chunk=2;sc.short_read=0;sc.interval=2;sc.checkpoints=3;sc.boundaries=config.boundaries;
    SlotSession slots(2,32,128,{},model.slot_ops(gen),true,sc);pointer=&slots;
    auto other=request({21,22},false);admit(slots,0,other);
    auto prompt=request({1,2,31,3,4,5,6,7,8,9,10,11,12,13,31,14,15,16,17,18});
    GenerationEvents events;bool emitted=false;events.token=[&](int32_t){emitted=true;};
    events.yield=[&](const GenerationRead& read){return read.tokens.size()==8 && slots.park_partial(1,read)?1:-1;};
    const auto result=gen.run(prompt,events);
    need(!emitted && result.generated==0 && result.finish=="cancelled" && result.yielded_slot==1 && result.yielded_tokens==8 && result.prompt_read==8,"solo GEN did not yield an owned partial without tokens");
    need(gen.cache_stats().entries==0 && slots.yielded_reads()==1,"matching checkpoint chain was duplicated instead of handed off");
    admit(slots,1,prompt,8);need(gen.cache_stats().entries>0,"BGEN did not return its partial chain to the main bank before collapsing to one point");
    // Ordinary BSlot now keeps only its deepest point; the earlier root must
    // remain selectable from the main bank imported from the partial chain.
    auto branch=request({1,2,30,29,28},true,1);need(run(gen,branch).first.reused==2,"handoff/back-import lost the shared root");
    need(slots.step(0).token==oracle(other,2)[1],"partial-chain restores damaged another active owned image");
    // The first root is beyond the first yield. Clear this Generation's local
    // interrupted marker so only the receiving slot's from0 can re-elect it.
    Model late;SlotSession* lp=nullptr;auto lc=config;lc.cache_every=0;
    Generation owner(32,128,{},late.gen_ops(lp),lc);SlotSession held(1,32,128,{},late.slot_ops(owner),true,sc);lp=&held;
    auto longer=request({1,2,3,4,5,6,31,7,8,9,10,11,12,13,14,15,31,16,17,18,19,20});
    events.yield=[&](const GenerationRead& read){return read.tokens.size()==2 && held.park_partial(0,read)?0:-1;};
    const auto first=owner.run(longer,events);need(first.yielded_tokens==2 && first.prompt_read==2,"early solo yield fixture");
    late.reset();owner.adopt({},{});
    events.yield=[&](const GenerationRead& read){return read.tokens.size()==8 && held.park_partial(0,read)?0:-1;};
    const auto second=owner.run(longer,events);need(second.reused==2 && second.yielded_tokens==8 && second.prompt_read==6,"GEN reissue did not resume the held prefix");
    // Only root 6 is in the chain but the endpoint is 8. Import must restore
    // 8 after visiting 6 before feeding the rest of the prompt.
    need(run(owner,longer).first.reused==8,"checkpoint import did not reattach the held endpoint");
    auto changed=request({1,2,3,4,5,6,30,29,28},true,1);need(run(owner,changed).first.reused==6,"partial from0 did not carry across slot/main adoption to the late root");
    // Hard pool pressure refuses preemption, continues the actual prompt, and
    // protects the active image. The test prices synthetic images, not RSS.
    Model tight(100);SlotSession* tp=nullptr;Generation limited(32,128,{},tight.gen_ops(tp),options());SlotSession running(2,32,128,{},tight.slot_ops(limited),true);tp=&running;
    admit(running,0,other);uint32_t attempts=0;GenerationEvents no_room;
    no_room.yield=[&](const GenerationRead& read){++attempts;return running.park_partial(1,read)?1:-1;};
    const auto completed=run(limited,prompt,no_room).first;
    need(attempts && completed.yielded_slot<0 && tight.budget->stats().peak==100 && running.step(0).token==oracle(other,2)[1],"unfunded solo yield corrupted state or exceeded snapshot budget");
}
}
int main() {
    try {cross_bank();cooperative_paths();failures_and_budget();explicit_main_attachment();numerical_recovery();solo_handoff();std::cout<<"PASS shared native-session orchestration: GEN/main/parked/idle-slot prefixes, active exclusion, exact Philox/Gumbel and native verify windows, owned suspension/cooperation and hard-budget/restore failures\nPASS solo BYIELD partial/end/chain handoff, BGEN back-import, shared root, late-root from0 across main adoption and hard-pool refusal\n";}
    catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
