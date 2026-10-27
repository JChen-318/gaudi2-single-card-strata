#include "strata/hpu/slot_session.hpp"
#include "strata/hpu/snapshot_memory.hpp"
#include "strata/hpu/sampler.hpp"
#include <array>
#include <cmath>
#include <iostream>
#include <memory>

using namespace strata::hpu;
namespace {
void require(bool ok,const char* why) {if(!ok) throw std::runtime_error(why);}
template<class F> void rejects(F&& f) {bool caught=false;try {f();} catch(const std::exception&) {caught=true;}require(caught,"invalid slot operation accepted");}
struct FakeModel {
    struct Image {
        SnapshotReservation reservation;
        std::vector<int32_t> tokens;
        bool control=true;
        explicit Image(std::shared_ptr<AllocationBudget> budget):reservation(std::move(budget),100) {}
    };
    std::shared_ptr<AllocationBudget> budget=std::make_shared<AllocationBudget>();
    std::shared_ptr<Image> retained;
    std::vector<int32_t> tokens;
    std::vector<std::vector<int32_t>> reads;
    std::vector<size_t> capture_cells,window_rows;
    bool control=true,bad_logits=false,fail_restore=false,fail_capture=false;
    uint32_t resets=0,switches=0;
    size_t budget_capture_at=0;
    explicit FakeModel(uint64_t limit=10000) {budget->configure(limit);}
    std::vector<float> logits() const {
        int sum=control?7:19;for(auto id:tokens) sum=(sum*5+id)%31;
        std::vector<float> values(32);for(int i=0;i<32;++i) values[size_t(i)]=float((sum*13+i*3+i*i)%29)/7;
        if(bad_logits) values[0]=NAN;
        return values;
    }
    SlotSessionOperations operations(bool with_control=true,bool with_window=false) {
        SlotSessionOperations op;
        op.reset=[this]{++resets;tokens.clear();retained.reset();};
        op.prefill=[this](std::span<const int32_t> input){reads.emplace_back(input.begin(),input.end());tokens.insert(tokens.end(),input.begin(),input.end());return logits();};
        op.decode=[this](int32_t token){tokens.push_back(token);return logits();};
        op.checkpoint=[this]()->std::function<void()> {
            if(fail_capture) throw std::runtime_error("injected capture failure");
            if(budget_capture_at && tokens.size()==budget_capture_at) throw AllocationBudgetExceeded();
            auto image=retained && retained.use_count()==1?std::move(retained):std::make_shared<Image>(budget);
            image->tokens=tokens;image->control=control;
            capture_cells.push_back(tokens.size());
            return [this,image]{if(fail_restore) {fail_restore=false;throw std::runtime_error("injected restore failure");}tokens=image->tokens;control=image->control;retained=image;};
        };
        if(with_control) op.control_vector=[this](bool value){++switches;control=value;tokens.clear();};
        op.discard_retained=[this]{retained.reset();};
        if(with_window) op.prefill_window=[this,prefill=op.prefill](std::span<const int32_t> input){window_rows.push_back(input.size());return prefill(input);};
        return op;
    }
};
GenerationRequest request(std::initializer_list<int32_t> tokens,bool control=true) {
    GenerationRequest result;result.prompt=tokens;result.control_vector=control;
    result.sampling.seed=37;result.sampling.counter=11;result.sampling.penalty_last_n=16;
    result.sampling.penalty_repeat=1.2f;result.sampling.penalty_freq=.1f;result.sampling.penalty_present=.2f;
    return result;
}
int32_t admit(SlotSession& engine,int slot,const GenerationRequest& req,uint32_t count=8,uint32_t reuse=0) {
    int32_t first=-1;const auto result=engine.admit(slot,req,count,[&](int32_t token){require(first==-1,"admission emitted twice");first=token;});
    require(first>=0 && result.prompt_tokens==req.prompt.size() && result.reused_tokens==reuse,"admission history/reuse accounting");return first;
}
// No SlotSession, snapshot or scheduler in this oracle: feed one emitted
// token at a time and advance the real sampler's explicit draw index once.
std::vector<int32_t> sequential_oracle(const GenerationRequest& request,uint32_t count) {
    FakeModel model;model.tokens=request.prompt;model.control=request.control_vector;
    auto parameters=request.sampling;std::vector<int32_t> emitted;
    while(emitted.size()<count) {
        const auto next=sample(model.logits(),model.tokens,parameters,{});
        emitted.push_back(next);++parameters.counter;model.tokens.push_back(next);
    }
    return emitted;
}
void isolated_sampling() {
    for(bool gumbel:{false,true}) {
        FakeModel interleaved,a,b;SlotSession mixed(2,32,32,{},interleaved.operations(),true),solo_a(1,32,32,{},a.operations(),false),solo_b(1,32,32,{},b.operations(),false);
        auto ra=request({1,2},false),rb=request({8,9,10},true);ra.sampling.gumbel=gumbel;rb.sampling.gumbel=gumbel;rb.sampling.seed=81;rb.sampling.counter=4;
        const auto expected_a=sequential_oracle(ra,6),expected_b=sequential_oracle(rb,6);
        const auto first_a=admit(mixed,0,ra),first_b=admit(mixed,1,rb);
        require(first_a==admit(solo_a,0,ra) && first_a==expected_a[0],"slot A admission differs from explicit sampler oracle");
        require(first_b==admit(solo_b,0,rb) && first_b==expected_b[0],"slot B admission differs from explicit sampler oracle");
        for(int i=0;i<5;++i) {
            const auto next_a=mixed.step(0).token,next_b=mixed.step(1).token;
            require(next_a==solo_a.step(0).token && next_a==expected_a[size_t(i)+1] && !a.control,"slot A lost its seed/counter/penalty history or cvec mode");
            require(next_b==solo_b.step(0).token && next_b==expected_b[size_t(i)+1] && interleaved.control,"slot B inherited another conversation's state");
        }
    }
}
void reuse_and_modes() {
    FakeModel f;SlotSession e(3,32,32,{},f.operations(),true);
    const auto first=admit(e,0,request({1,2},false));e.step(0);const auto held=f.tokens;e.finish(0);
    auto follow=request({1,2,first,7},false);
    require(held==std::vector<int32_t>({1,2,first}),"cache includes an emitted token not yet consumed");
    const auto resets=f.resets;admit(e,1,follow,8,3);
    require(f.resets==resets && f.reads.back()==std::vector<int32_t>{7},"idle end-state was not restored before reading only its new suffix");
    e.finish(1);
    auto own=follow;own.prompt.push_back(6);admit(e,1,own,8,4);
    require(f.reads.back()==std::vector<int32_t>{6},"same-slot cache was released before restoration");e.finish(1);
    auto changed=own;changed.control_vector=true;changed.prompt.push_back(5);
    admit(e,2,changed);require(f.control && f.reads.back()==changed.prompt,"cvec mismatch reused incompatible arithmetic");e.release(2);
    // The longest valid end prefix wins. An active longer slot is excluded.
    auto from_short=request({1,2,first,7,6,5},false);admit(e,2,from_short,8,5);
    e.release(2);e.release(0);
    // Equal-length prompts cannot reuse the end: upstream always reads the last prompt token.
    const auto count=f.resets;admit(e,2,own);require(f.resets==count+1,"same-length slot end bypassed upstream's last-token rule");e.release(2);
    f.fail_restore=true;auto after_failure=own;after_failure.prompt.push_back(4);
    admit(e,2,after_failure);require(f.reads.back()==after_failure.prompt,"failed cached restore did not fall back to a fresh prompt");
    FakeModel no_cache;SlotSession off(1,32,32,{},no_cache.operations(),false);
    admit(off,0,request({1,2}));off.finish(0);admit(off,0,request({1,2,3}));require(no_cache.resets==2,"prompt-cache=0 retained an idle image");
    FakeModel absent;SlotSession no_vectors(1,32,32,{},absent.operations(false),true);
    admit(no_vectors,0,request({1,2},false));no_vectors.finish(0);admit(no_vectors,0,request({1,2,3},true),8,2);
    require(no_vectors.reused_tokens()==2,"cvec flag without loaded vectors separated identical arithmetic");
    FakeModel busy;SlotSession active(3,32,32,{},busy.operations(),true);
    admit(active,0,request({1,2}));admit(active,1,request({1,2,3}));
    require(busy.resets==2,"active slot lent its mutable prefix to another request");
    active.finish(0);admit(active,2,request({1,2,3,4}),8,2);
    require(busy.reads.back()==std::vector<int32_t>({3,4}),"idle prefix lost after excluding a longer active match");
}
void failures_and_budget() {
    FakeModel f(200);SlotSession e(3,32,32,{},f.operations(),true);
    admit(e,0,request({1,2}));e.finish(0);admit(e,1,request({3,4}));admit(e,2,request({5,6}));
    require(e.idle_evictions()==1 && f.budget->stats().used==200,"bounded snapshot admission failed to reclaim only the idle image");
    bool emitted=false;rejects([&]{e.admit(0,request({7,8}),8,[&](int32_t){emitted=true;});});
    require(!emitted,"capture failure emitted a token without a continuing image");
    require(!e.step(1).nonfinite && !e.step(2).nonfinite,"snapshot refusal evicted an active conversation");
    f.bad_logits=true;require(e.step(1).nonfinite,"non-finite row did not take the no-token termination path");e.finish(1);
    f.bad_logits=false;admit(e,1,request({3,4,9}));require(f.reads.back()==std::vector<int32_t>({3,4,9}),"non-finite slot became an idle cache");
    e.release(1);e.release(2);f.retained.reset();require(!f.budget->stats().used,"released slot images leaked a reservation");
    const auto resets=f.resets;
    auto invalid=request({1,2});invalid.bias.resize(32);
    rejects([&]{admit(e,0,invalid);});rejects([&]{admit(e,-1,request({1}));});rejects([&]{admit(e,0,request({32}));});
    require(f.resets==resets,"invalid request mutated the live model before validation");
    auto overflow=request({1,2});overflow.sampling.counter=UINT64_MAX;
    rejects([&]{admit(e,0,overflow,2);});
    auto bad_sampling=request({1,2});bad_sampling.sampling.temperature=NAN;
    rejects([&]{admit(e,0,bad_sampling);});
    require(f.resets==resets,"invalid batch sampling/counter changed the model before refusal");
    admit(e,0,overflow,1);e.release(0); // the last representable draw itself is valid
    f.fail_capture=true;emitted=false;rejects([&]{e.admit(0,request({1,2}),8,[&](int32_t){emitted=true;});});require(!emitted,"failed park published admission token");
    f.fail_capture=false;
    const auto used=f.budget->stats().used;admit(e,0,request({1,2}),1);e.release(0);require(f.budget->stats().used==used,"max_new=1 needlessly parked a slot");
    FakeModel eos_model;const auto eos_request=request({1,2});
    const auto first=sequential_oracle(eos_request,1).front();
    SlotSession eos(1,32,32,{first},eos_model.operations(),true);
    for(uint32_t count:{1u,8u}) {
        const auto stopped=eos.admit(0,eos_request,count,[&](int32_t token){require(token==first,"EOS admission oracle");});
        require(stopped.eos && !stopped.continues && !eos_model.budget->stats().used,"native slot result lost first EOS or parked its finished request");eos.release(0);
    }
    require(slot_session_host_bytes(3,32)==3*(32*4+8192) && !slot_session_host_bytes(0,32),"slot metadata envelope formula");
    rejects([]{(void)slot_session_host_bytes(65,32);});rejects([]{(void)slot_session_host_bytes(1,0);});
}
SlotCheckpointOptions turn_options() {
    SlotCheckpointOptions o;o.boundaries.turn_token=31;o.boundaries.root_min_tokens=2;o.interval=0;return o;
}
void turn_checkpoints() {
    const auto initial=request({1,2,31,3,4,31,5,6});
    auto follow=request({1,2,31,3,4,31,9,10});
    for(bool grouped:{false,true}) {
        FakeModel f;auto o=turn_options();if(grouped) {o.prefill_chunk=4;o.short_read=0;}
        SlotSession e(2,32,32,{},f.operations(true,grouped),true,o);
        admit(e,0,initial);require(e.checkpoint_captures()==2 && f.capture_cells==std::vector<size_t>({2,5,8}),"root/turn checkpoint positions differ from prompt boundaries");
        if(grouped) require(f.window_rows==std::vector<size_t>({2,3,3}),"native prefill windows crossed an explicit checkpoint cut");
        e.step(0);e.finish(0);const auto resets=f.resets;
        const auto sampled=admit(e,1,follow,8,5);
        require(sampled==sequential_oracle(follow,1).front() && f.tokens==follow.prompt && f.resets==resets,"re-rendered thinking history did not restore its checkpoint state");
        require(f.reads.back()==std::vector<int32_t>({31,9,10}),"checkpoint restore re-read the old prefix");
        e.finish(1);e.release(0);
        auto same=follow;same.prompt[6]=12;
        admit(e,1,same,8,5);require(f.tokens==same.prompt,"same-slot release destroyed the selected checkpoint");e.finish(1);
        auto incompatible=same;incompatible.control_vector=false;
        const auto prior=f.resets;admit(e,0,incompatible);require(f.resets==prior+1 && !f.control,"checkpoint reused another cvec arithmetic mode");e.release(0);
        auto rewritten=same;rewritten.prompt[0]=7;
        const auto before=f.resets;admit(e,0,rewritten);require(f.resets==before+1 && f.tokens==rewritten.prompt,"rewritten checkpoint tokens were reused");e.release(0);
        f.fail_restore=true;admit(e,0,same);require(f.tokens==same.prompt,"checkpoint restore error did not fall back to reading the prompt");
    }
    // ckpt=0 still allows the automatic system root, but no turn checkpoint.
    FakeModel root;SlotSession roots(1,32,32,{},root.operations(),true,turn_options());
    auto no_check=initial;no_check.checkpoint=false;admit(roots,0,no_check);roots.finish(0);
    auto sibling=request({1,2,31,11});sibling.checkpoint=false;
    admit(roots,0,sibling,8,2);require(root.tokens==sibling.prompt,"ckpt=0 lost its eligible shared root");
    FakeModel pin;auto disabled=turn_options();disabled.boundaries.turn_token=-1;
    SlotSession pins(1,32,32,{},pin.operations(),true,disabled);
    auto pinned=request({1,2,3,4,5,6});pinned.pin=3;admit(pins,0,pinned);pins.finish(0);
    admit(pins,0,request({1,2,3,9,10}),8,3);require(pin.tokens==std::vector<int32_t>({1,2,3,9,10}),"explicit prompt pin was not checkpointed");
    FakeModel busy;SlotSession active(2,32,32,{},busy.operations(),true,turn_options());
    admit(active,0,initial);admit(active,1,follow);require(busy.resets==2,"active slot checkpoint became an idle source");
}
void scheduled_checkpoints() {
    const auto prompt=request({1,2,3,4,5,6,7,8,9,10,11,12});
    SlotCheckpointOptions o;o.boundaries.turn_token=-1;o.interval=5;o.prefill_chunk=4;o.short_read=0;
    FakeModel f;SlotSession e(1,32,32,{},f.operations(true,true),true,o);
    admit(e,0,prompt);e.finish(0);
    require(f.window_rows==std::vector<size_t>({4,4,4}) && f.capture_cells==std::vector<size_t>({8,12}),"periodic checkpoint split the window or retained the final prompt row as a checkpoint");
    admit(e,0,request({1,2,3,4,5,6,7,8,20,21}),8,8);
    o.prefill_until=5;FakeModel until;SlotSession cut(1,32,32,{},until.operations(true,true),true,o);
    admit(cut,0,prompt);
    require(until.window_rows==std::vector<size_t>({4,1}) && until.capture_cells==std::vector<size_t>({5,10,12}),"prefill-until scalar suffix lost checkpoint scheduling");
    o.prefill_until=0;o.interval=100;o.tail=true;o.short_read=1;
    FakeModel tail;SlotSession tails(1,32,32,{},tail.operations(true,true),true,o);
    auto shorter=prompt;shorter.prompt.pop_back();admit(tails,0,shorter);
    require(tail.capture_cells==std::vector<size_t>({8,11}),"near-tail checkpoint law was not called on actual chunks");
    o.prefill_chunk=UINT32_MAX;o.prefill_until=UINT32_MAX;o.short_read=0;o.tail=false;o.interval=5;
    FakeModel wide;SlotSession bounded(1,32,32,{},wide.operations(true,true),true,o);
    admit(bounded,0,prompt);
    require(wide.window_rows==std::vector<size_t>({11}) && wide.capture_cells==std::vector<size_t>({11,12}),"wide prefill knobs overflowed a chunk or lost the final scalar position");
}
void checkpoint_pressure() {
    const auto initial=request({1,2,31,3,4,31,5,6});
    FakeModel tiny(100);SlotSession one(1,32,32,{},tiny.operations(),true,turn_options());
    admit(one,0,initial);require(one.checkpoint_evictions() && tiny.budget->stats().used==100 && tiny.budget->stats().peak<=100,"optional checkpoint blocked the required continuation or escaped the image budget");
    require(!one.step(0).nonfinite,"optional checkpoint eviction invalidated live continuation");
    FakeModel f(200);SlotSession two(2,32,32,{},f.operations(),true,turn_options());
    admit(two,0,initial);admit(two,1,request({20,21}));
    require(two.checkpoint_evictions() && f.budget->stats().used==200,"active prompt checkpoint was not reclaimable before refusing a second active end image");
    require(!two.step(0).nonfinite && !two.step(1).nonfinite,"reclaiming an optional boundary destroyed an active conversation");
    FakeModel denied;denied.budget_capture_at=5;SlotSession older(1,32,32,{},denied.operations(),true,turn_options());
    admit(older,0,initial);older.finish(0);admit(older,0,request({1,2,31,11}),8,2);
    require(denied.tokens==std::vector<int32_t>({1,2,31,11}),"failed newer checkpoint discarded the older valid root");
}
BatchSlotSession::Admission yield_at(SlotSession& slots,FakeModel& model,int slot,const GenerationRequest& req,uint32_t threshold) {
    BatchSlotSession::ReadControl read;bool asked=false,emitted=false;
    read.yield=[&]()->std::optional<int> {if(!asked && model.tokens.size()>=threshold) {asked=true;return slot;}return std::nullopt;};
    auto result=slots.admit_cooperative(slot,req,8,[&](int32_t){emitted=true;},read);
    require(!emitted && result.yielded_slot==slot && result.yielded_tokens==threshold && !result.continues,"BYIELD did not publish only an owned partial prompt");
    require(model.tokens==std::vector<int32_t>(req.prompt.begin(),req.prompt.begin()+threshold),"BYIELD saved the wrong prefix");
    return result;
}
void cooperative_yield() {
    SlotCheckpointOptions o=turn_options();o.prefill_chunk=2;o.short_read=0;o.interval=3;o.checkpoints=2;
    auto long_prompt=request({1,2,31,3,4,5,6,7,8,9,31,10,11,12,13,14,15,16});
    FakeModel f;SlotSession slots(2,32,32,{},f.operations(true,true),true,o);
    auto yielded=yield_at(slots,f,0,long_prompt,6);require(!yielded.reused_tokens,"fresh partial reused a nonexistent prefix");
    // Same-slot partial end resume must carry the whole chain. At capacity two
    // later leaves rotate while the shared root survives a second yield.
    yielded=yield_at(slots,f,0,long_prompt,12);require(yielded.reused_tokens==6,"reissued partial prompt was re-read");
    auto branched=request({1,2,30,29,28,27,26,25});
    admit(slots,1,branched,8,2);slots.release(1);
    int32_t first=-1;auto resumed=slots.admit(0,long_prompt,8,[&](int32_t id){first=id;});
    require(resumed.reused_tokens==12 && first==sequential_oracle(long_prompt,1).front(),"resume changed target sampling or lost the partial end");
    require(slots.yielded_reads()==2 && f.budget->stats().used<=10000,"partial image ownership/budget");
    // The first root is ahead of the first yield. resumed_from0 must still
    // elect it when the same prompt is issued again.
    FakeModel later;auto root=o;root.interval=0;root.checkpoints=3;
    auto late_prompt=request({1,2,3,4,5,6,31,7,8,9,10,11,12,13,14,15,31,16,17,18,19,20});
    SlotSession roots(2,32,32,{},later.operations(true,true),true,root);
    yield_at(roots,later,0,late_prompt,2);auto second=yield_at(roots,later,0,late_prompt,8);
    require(second.reused_tokens==2,"late-root partial resume length");
    admit(roots,1,request({1,2,3,4,5,6,30,29,28}),8,6);
    // New pin replaces old pins; root + explicit pin + rotating leaf fit in
    // three slots and the original retention policy chooses the victim.
    FakeModel pinned;auto pin_options=o;pin_options.checkpoints=3;pin_options.interval=2;pin_options.boundaries.turn_token=-1;
    SlotSession pins(2,32,32,{},pinned.operations(true,true),true,pin_options);
    auto pin_prompt=request({1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18});pin_prompt.pin=4;
    yield_at(pins,pinned,0,pin_prompt,12);
    admit(pins,1,request({1,2,3,4,30,29,28}),8,4);
    // Wrong target/cache-off/short remainder cannot create an unpublished
    // token or an unusable partial image. Snapshot-budget refusal continues.
    for(int mode=0;mode<4;++mode) {
        FakeModel reject;auto options=o;if(mode==2) options.short_read=32;
        SlotSession no_yield(2,32,32,{},reject.operations(true,true),mode!=1,options);
        BatchSlotSession::ReadControl read;bool asked=false;int32_t token=-1;
        read.yield=[&]()->std::optional<int> {if(asked) return {};asked=true;return mode==0?1:0;};
        if(mode==3) reject.budget_capture_at=2;
        const auto result=no_yield.admit_cooperative(0,long_prompt,8,[&](int32_t id){token=id;},read);
        require(result.yielded_slot<0 && token==sequential_oracle(long_prompt,1).front(),"invalid or unfunded yield truncated the prompt");
    }
}
void cancelled_reads() {
    for(uint64_t budget:{100ull,300ull}) {
        FakeModel model(budget);auto op=model.operations(true,true);
        op.prefill_cancellable=[&](std::span<const int32_t> input,bool,const PromptCommit& commit,const std::function<bool()>& stop) {
            for(uint32_t at=0;at<input.size();++at) {if(stop && stop()) throw Cancelled();model.tokens.push_back(input[at]);const auto values=model.logits();commit(at+1,values);}
        };
        auto config=turn_options();config.prefill_chunk=4;config.short_read=0;SlotSession slots(2,32,32,{},std::move(op),true,config);
        const auto other=request({22,23},false);const auto first=admit(slots,1,other);
        auto incoming=request({3,4,5,6,7,8,9,10},true);bool stop=false,emitted=false;BatchSlotSession::ReadControl read;
        read.cancelled=[&]{return stop;};read.progress=[&](uint32_t cells,uint32_t,double){if(cells==3) stop=true;};
        const auto result=slots.admit_cooperative(0,incoming,8,[&](int32_t){emitted=true;},read);
        require(result.cancelled && !result.continues && !emitted && result.prompt_read==3 && !result.reused_tokens && model.tokens==std::vector<int32_t>({3,4,5}),"cancel during native chunk lost committed prefix or emitted an admission token");
        require(first==sequential_oracle(other,2)[0] && slots.step(1).token==sequential_oracle(other,2)[1],"prompt STOP damaged an independent active slot");
        int32_t output=-1;const auto resumed=slots.admit(0,incoming,1,[&](int32_t id){output=id;});
        require(resumed.reused_tokens==(budget==300?3u:0u) && resumed.prompt_read==incoming.prompt.size()-resumed.reused_tokens && output==sequential_oracle(incoming,1)[0],"cancelled prompt did not resume only its actually owned partial image");
        require(model.budget->stats().peak<=budget,"cancel cache capture evicted active state or exceeded hard snapshot budget");
    }
    FakeModel model;SlotSession slots(1,32,32,{},model.operations(),true);auto incoming=request({3,4},true);bool stop=false;uint32_t published=0;BatchSlotSession::ReadControl read;read.cancelled=[&]{return stop;};
    const auto result=slots.admit_cooperative(0,incoming,8,[&](int32_t){++published;stop=true;},read);
    require(result.cancelled && !result.continues && published==1 && result.prompt_read==2,"STOP after admission publication lost the committed first token");
    rejects([&]{slots.step(0);});auto again=incoming;again.prompt.push_back(sequential_oracle(incoming,1)[0]);again.prompt.push_back(7);
    require(slots.admit(0,again,1,[](int32_t){}).reused_tokens==2,"stopped admission retained active scheduling or lost its healthy idle prefix");
}
void cooperative_interleave() {
    auto options=turn_options();options.prefill_chunk=2;options.short_read=0;options.interval=0;
    FakeModel f;SlotSession slots(2,32,32,{},f.operations(true,true),false,options);
    auto a=request({1,2},false),b=request({8,9,10,11,12,13,14,15},true);
    std::vector<int32_t> out{admit(slots,0,a)};BatchSlotSession::ReadControl read;
    read.has_work=[&]{return out.size()<4;};read.between=[&](double){out.push_back(slots.step(0).token);};
    int32_t first=-1;slots.admit_cooperative(1,b,8,[&](int32_t id){first=id;},read);
    require(out==sequential_oracle(a,4) && first==sequential_oracle(b,1).front() && f.tokens==b.prompt && f.control,"interleaving failed to restore the current prompt/MTP/control image");
    require(slots.interleaved_windows()==3,"running slots were stalled until prompt admission completed");
    read.has_work=[] {return true;};read.between=[&](double){slots.step(0);throw std::runtime_error("injected scheduler failure");};
    slots.release(1);rejects([&]{slots.admit_cooperative(1,b,8,[](int32_t){},read);});
    require(slots.step(0).token==sequential_oracle(a,6)[5],"failed prompt interleave damaged an active owned image");
    FakeModel tight(100);SlotSession limited(2,32,32,{},tight.operations(true,true),false,options);
    admit(limited,0,a);uint32_t calls=0;int32_t token=-1;
    read.has_work=[] {return true;};read.between=[&](double){++calls;};
    limited.admit_cooperative(1,b,1,[&](int32_t id){token=id;},read);
    require(!calls && !limited.interleaved_windows() && tight.budget->stats().peak==100 && token==sequential_oracle(b,1).front(),"temporary prompt park overcommitted the snapshot budget");
}
GenerationRequest pictured_request(uint64_t cells,uint64_t key) {
    auto result=request({1,2,3,4,5});
    result.embeddings=std::make_shared<VisionEmbeddings>();
    result.image_rows={-1,-1,0,1,-1};
    result.positions.resize(size_t(cells*3));
    for(uint64_t cell=0;cell<cells;++cell) for(uint32_t row=0;row<3;++row)
        result.positions[size_t(cell*3+row)]=int32_t(cell+row);
    result.images={{2,key}};
    result.checkpoint=false;
    return result;
}
void vision_requests() {
    const uint64_t cells=32+64;
    const auto pictured=pictured_request(cells,1234);
    FakeModel f;auto base=f.operations();
    std::vector<std::string> events;std::vector<GenerationRequest> installed;std::vector<size_t> seen_at_prefill;
    base.set_vision=[&](const GenerationRequest& value){events.push_back("vision");installed.push_back(value);};
    const auto reset=base.reset;base.reset=[&]{events.push_back("reset");reset();};
    const auto prefill=base.prefill;
    base.prefill=[&](std::span<const int32_t> input){events.push_back("prefill");seen_at_prefill.push_back(installed.size());return prefill(input);};
    SlotSession slots(2,32,32,{},std::move(base),true);
    admit(slots,0,pictured);
    require(!events.empty() && events.front()=="vision","slot admission ran Model work before installing the request's vision table");
    require(installed.size()==1 && installed[0].embeddings==pictured.embeddings && installed[0].image_rows==pictured.image_rows &&
            installed[0].positions==pictured.positions && installed[0].images==pictured.images,"slot admission changed the image request before installing it");
    require(seen_at_prefill.size()==1 && seen_at_prefill[0]==1,"image admission prefilled before installing its M-RoPE rows");
    admit(slots,1,request({6,7}));
    require(installed.size()==2 && !installed[1].embeddings && installed[1].image_rows.empty() && installed[1].positions.empty(),
            "text slot admission did not restore the identity table upstream rebuilds at :10746-10747");
    require(seen_at_prefill.back()==2,"text admission prefilled before restoring the identity table");
    slots.finish(0);slots.finish(1);
    // Upstream's idle-slot branch (:10935) only runs for a request without pictures.
    FakeModel g;auto reuse_op=g.operations();reuse_op.set_vision=[](const GenerationRequest&){};
    SlotSession reuse(2,32,32,{},std::move(reuse_op),true);
    admit(reuse,0,request({1,2,3,4}));reuse.finish(0);
    auto longer=pictured;longer.prompt.push_back(6);longer.image_rows.push_back(-1);
    admit(reuse,1,longer,8,0);
    require(g.reads.back()==longer.prompt,"pictured request reused an idle text slot without checking pictures");
    reuse.finish(1);reuse.release(0);
    // A GENI request needs the owner hook; a text request still runs without it.
    FakeModel plain;SlotSession no_vision(1,32,32,{},plain.operations(),true);
    bool refused=false;std::string why;
    try {admit(no_vision,0,pictured);} catch(const std::invalid_argument& error) {refused=true;why=error.what();}
    require(refused && why=="this engine was started without --vision","pictured slot request without --vision was not refused by name");
    require(plain.reads.empty() && plain.resets==0,"a refused vision request touched the Model");
    admit(no_vision,0,request({1,2}));
    // Generation's shape law runs before the hook; a malformed picture request never reaches the Model.
    FakeModel shape;uint32_t installs=0;auto op=shape.operations();op.set_vision=[&](const GenerationRequest&){++installs;};
    SlotSession checked(1,32,32,{},std::move(op),true);
    auto short_rows=pictured;short_rows.image_rows.pop_back();
    rejects([&]{admit(checked,0,short_rows);});
    auto short_positions=pictured;short_positions.positions.pop_back();
    rejects([&]{admit(checked,0,short_positions);});
    require(installs==0 && shape.reads.empty() && shape.resets==0,"malformed vision shapes reached the Model or its install hook");
}
}
int main() {
    try {isolated_sampling();reuse_and_modes();failures_and_budget();turn_checkpoints();scheduled_checkpoints();checkpoint_pressure();cooperative_yield();cancelled_reads();cooperative_interleave();vision_requests();std::cout<<"PASS production slot sessions: interleaved Philox/Gumbel/penalties/cvec versus independent sessions\nPASS consumed-prefix idle reuse, same-slot restore, control mismatch, last-token law, restore fallback\nPASS idle-only eviction with actual reservation accounting, active protection, non-finite and pre-publication failure\nPASS turn/root/pin and periodic/tail checkpoints, scalar/window schedules, re-rendered history and bounded optional image reclamation\nPASS BYIELD owned partial/end resume, full checkpoint chain/root/pin retention, resumed-from-zero root, invalid/unfunded yield and cooperative active windows\nPASS BGENI request vision install before Model work, identity restore for text, idle-slot picture gate and --vision/shape refusals\n";}
    catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
