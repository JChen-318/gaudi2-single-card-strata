#include "strata/hpu/slot_session.hpp"
#include "strata/hpu/allocation_budget.hpp"
#include "strata/hpu/sampler.hpp"
#include "strata/hpu/prefill_schedule.hpp"
#include "strata/program/conv_cache.hpp"
#include <chrono>
#include <algorithm>
#include <array>
#include <cmath>

namespace strata::hpu {
namespace {
bool finite_logits(std::span<const float> values,uint32_t vocabulary) {
    return values.size()==vocabulary && std::all_of(values.begin(),values.end(),[](float x){return std::isfinite(x);});
}
void validate_logits(std::span<const float> values,uint32_t vocabulary) {
    if(values.size()!=vocabulary) throw std::runtime_error("slot logits shape mismatch");
    for(float x:values) if(!std::isfinite(x)) throw NonFiniteOutput("non-finite slot logits");
}
}
SlotSession::SlotSession(uint32_t slots,uint32_t vocabulary,uint32_t context,std::vector<int32_t> stops,
                         SlotSessionOperations op,bool cache_idle,SlotCheckpointOptions options)
    :vocabulary_(vocabulary),context_(context),stops_(std::move(stops)),op_(std::move(op)),cache_idle_(cache_idle),options_(options) {
    if(!slots || !vocabulary || vocabulary>INT32_MAX || !op_.reset || !op_.prefill || !op_.decode || !op_.checkpoint)
        throw std::invalid_argument("incomplete slot session geometry/operations");
    (void)slot_session_host_bytes(slots,context);
    if(context>INT32_MAX || options.interval>INT32_MAX || options.short_read>INT32_MAX || options.checkpoints>64 || (cache_idle && !options.checkpoints))
        throw std::invalid_argument("invalid slot checkpoint schedule");
    if(options.canonical_prefill_rows && (options.canonical_prefill_rows>options.prefill_chunk || options.prefill_chunk%options.canonical_prefill_rows))
        throw std::invalid_argument("invalid canonical slot prefill rows");
    (void)prompt_boundaries({},0,options.boundaries,true);
    static_assert(sizeof(Slot)<=4096);
    for(auto token:stops_) if(token<0 || uint32_t(token)>=vocabulary) throw std::invalid_argument("invalid slot stop token");
    slots_.resize(slots);
    for(auto& slot:slots_) {
        slot.fed=std::vector<int32_t>(context);slot.fed.clear();
        if(slot.fed.capacity()!=context) throw std::runtime_error("slot token capacity exceeds admitted host envelope");
    }
}
SlotSession::Slot& SlotSession::at(int slot) {
    if(slot<0 || size_t(slot)>=slots_.size()) throw std::out_of_range("no such slot session");
    return slots_[size_t(slot)];
}
bool SlotSession::is_stop(int32_t token) const {return std::find(stops_.begin(),stops_.end(),token)!=stops_.end();}
void SlotSession::release(int slot) {
    auto& state=at(slot);state.image={};
    while(state.check_count) erase_point(state,state.check_count-1);
    state.fed.clear();state.images.clear();state.active=false;state.cached=false;state.partial=false;state.from0=false;state.tainted=false;
}
void SlotSession::erase_point(Slot& state,uint32_t index) {
    for(uint32_t i=index+1;i<state.check_count;++i) state.checks[i-1]=std::move(state.checks[i]);
    state.checks[--state.check_count]={};
}
void SlotSession::invalidate_caches() {
    for(size_t i=0;i<slots_.size();++i) {
        if(slots_[i].active) slots_[i].tainted=true;
        else release(int(i));
    }
    if(op_.discard_retained) op_.discard_retained();
}
void SlotSession::finish(int slot) {
    auto& state=at(slot);state.active=false;
    state.cached=cache_idle_ && !state.tainted && bool(state.image) && !state.fed.empty();
    if(!state.cached) release(slot);
}
void SlotSession::stop(int slot) {(void)at(slot);} // scheduler applies stop AFTER the next token
GenerationRead SlotSession::read_view(Slot& state) const {
    GenerationRead read;read.tokens=state.fed;read.control_vector=state.control;read.from0=state.from0;
    read.restore=state.image;read.logits=op_.logits;
    const auto below=images_below(state.images,state.fed.size());
    read.images.assign(below.begin(),below.end());
    read.checkpoints=[&state](const GenerationCheckpointVisitor& visit) {
        for(uint32_t j=0;j<state.check_count;++j) {
            const auto& point=state.checks[j];
            GenerationCheckpoint checkpoint{std::span<const int32_t>(state.fed).first(point.cells),point.image,point.age,point.pinned,point.root || j==0,point.tail};
            const auto pictures=images_below(state.images,point.cells);
            checkpoint.images=pictures;
            visit(checkpoint);
        }
    };
    return read;
}
SlotSession::Slot SlotSession::collect_points(const GenerationRead& read) {
    Slot result;
    if(read.checkpoints) read.checkpoints([&](const GenerationCheckpoint& point) {
        if(point.tokens.empty() || point.tokens.size()>read.tokens.size() || !point.restore || result.check_count==64 ||
           !std::equal(point.tokens.begin(),point.tokens.end(),read.tokens.begin()) ||
           !images_match(point.images,read.images,point.tokens.size()))
            throw SessionFatal("partial checkpoint chain contract mismatch");
        result.checks[result.check_count++]={point.restore,point.age,uint32_t(point.tokens.size()),point.pinned,point.tail,point.root};
        clock_=std::max(clock_,point.age);
    });
    std::stable_sort(result.checks.begin(),result.checks.begin()+result.check_count,[](const auto& a,const auto& b) {
        return a.root!=b.root?a.root>b.root:a.cells<b.cells;
    });
    while(result.check_count>options_.checkpoints) {
        std::array<uint64_t,64> stamps{};std::array<bool,64> tails{},pins{};bool any_pin=false;
        for(uint32_t j=0;j<result.check_count;++j) {stamps[j]=result.checks[j].age;tails[j]=result.checks[j].tail;pins[j]=result.checks[j].pinned;any_pin=any_pin || pins[j];}
        const auto victim=program::conv_cache::eviction_victim(stamps.data(),result.check_count,options_.checkpoints,tails.data(),any_pin?pins.data():nullptr);
        erase_point(result,uint32_t(victim));++checkpoint_evictions_;
    }
    return result;
}
bool SlotSession::park_partial(int slot,const GenerationRead& read) {
    auto& state=at(slot);
    if(state.active || !cache_idle_ || !read.can_yield || read.tokens.empty() || read.tokens.size()>=context_) return false;
    for(auto id:read.tokens) if(id<0 || uint32_t(id)>=vocabulary_) throw SessionFatal("invalid partial token history");
    auto incoming=collect_points(read);std::function<void()> image;
    for(uint32_t j=0;j<incoming.check_count;++j) if(incoming.checks[j].cells==read.tokens.size()) image=incoming.checks[j].image;
    release(slot); // upstream invalidates a half-written target if capture fails
    try {if(!image) image=park();} catch(const AllocationBudgetExceeded&) {return false;} catch(const std::bad_alloc&) {return false;}
    state.checks=std::move(incoming.checks);state.check_count=incoming.check_count;
    state.fed.assign(read.tokens.begin(),read.tokens.end());state.images.assign(read.images.begin(),read.images.end());
    state.control=op_.control_vector?read.control_vector:true;
    state.from0=read.from0;state.partial=true;state.cached=true;state.image=std::move(image);++yielded_reads_;return true;
}
std::optional<GenerationPrefix> SlotSession::select_prefix(std::span<const int32_t> prompt,uint64_t maximum,bool control,
                                                          std::span<const RequestImage> images) {
    // Upstream gates idle-slot reuse on req_imgs.empty() (:10935) and the whole slot-source loop only runs then;
    // the parked/live caches are searched by Generation itself and do compare pictures.
    if(!cache_idle_ || !op_.logits || prompt.empty() || !images.empty()) return {};
    maximum=std::min<uint64_t>(maximum,prompt.size()-1);
    Slot* source=nullptr;int point=-1;size_t cells=0;
    for(auto& state:slots_) {
        if(state.active || !state.cached || !state.image || state.control!=control) continue;
        // The end state has no picture identity of its own in upstream's BSlot and `starts_with(ids, {})`
        // accepts it; a checkpoint carries imgs and must then carry none for an image-free request.
        const auto consider=[&](size_t n,int check) {
            if(n && n<=maximum && n>cells && n<=state.fed.size() && std::equal(state.fed.begin(),state.fed.begin()+n,prompt.begin()) &&
               (check<0 || images_below(state.images,state.checks[size_t(check)].cells).empty())) {cells=n;source=&state;point=check;}
        };
        consider(state.fed.size(),-1);
        for(uint32_t j=0;j<state.check_count;++j) consider(state.checks[j].cells,int(j));
    }
    if(!source) return {};
    if(point>=0) source->checks[size_t(point)].age=++clock_;
    GenerationPrefix result{std::span<const int32_t>(source->fed).first(cells),point<0?source->image:source->checks[size_t(point)].image,op_.logits};
    if(point<0) result.images.assign(images_below(source->images,cells).begin(),images_below(source->images,cells).end());
    else {const auto pictures=images_below(source->images,source->checks[size_t(point)].cells);result.images.assign(pictures.begin(),pictures.end());}
    if(point<0 && source->partial) {
        // Only the image-free checkpoints of a yielded chain may enter a request without pictures; the same
        // filter the per-slot selection above applied point by point.
        Slot* state=source;result.from0=source->from0;
        result.checkpoints=[state](const GenerationCheckpointVisitor& visit) {
            for(uint32_t j=0;j<state->check_count;++j) {
                const auto& p=state->checks[j];
                const auto pictures=images_below(state->images,p.cells);
                if(!pictures.empty()) continue;
                GenerationCheckpoint checkpoint{std::span<const int32_t>(state->fed).first(p.cells),p.image,p.age,p.pinned,p.root || j==0,p.tail};
                checkpoint.images=pictures;visit(checkpoint);
            }
        };
    } else if(point>=0) result.checkpoints=[source,point](const GenerationCheckpointVisitor& visit) {
        const auto& p=source->checks[size_t(point)];
        GenerationCheckpoint checkpoint{std::span<const int32_t>(source->fed).first(p.cells),p.image,p.age,p.pinned,true,p.tail};
        const auto pictures=images_below(source->images,p.cells);
        checkpoint.images=pictures;visit(checkpoint);
    };
    return result;
}
std::function<void()> SlotSession::park(bool required) {
    // Active conversations are never evicted. Idle end images are an optional
    // cache: reclaim one at a time only if the bounded snapshot allocator
    // refuses a capture. Release the Model retained alias as well.
    for(size_t victim=0;;) {
        try {
            auto image=op_.checkpoint();
            if(!image) throw std::logic_error("empty slot checkpoint");
            return image;
        } catch(const AllocationBudgetExceeded&) {
            while(victim<slots_.size() && (!slots_[victim].cached || slots_[victim].active)) ++victim;
            if(victim==slots_.size()) {
                if(!required) return {};
                // Prompt images are optional cache, including a running
                // slot's old boundary. They must never block its required
                // continuation image. Its live/end state is not evicted.
                auto optional=std::find_if(slots_.begin(),slots_.end(),[](const auto& s){return s.check_count!=0;});
                if(optional==slots_.end()) throw;
                erase_point(*optional,optional->check_count-1);++checkpoint_evictions_;
                if(op_.discard_retained) op_.discard_retained();
                continue;
            }
            release(int(victim++));++idle_evictions_;
            if(op_.discard_retained) op_.discard_retained();
        }
    }
}
void SlotSession::capture_prompt(Slot& state,uint32_t cells,bool pinned,bool tail,bool root,std::span<const RequestImage> images) {
    if(!cache_idle_ || !cells) return;
    (void)images; // the slot's picture list below cells is the point's identity (upstream ConvCheckpoint::imgs)
    // Upstream's checkpoint_at reuses an existing checkpoint at this exact length without touching its imgs
    // (:8452-8457): the first capture's picture identity stays with the saved state.
    for(uint32_t i=0;i<state.check_count;++i) if(state.checks[i].cells==cells) {
        auto& point=state.checks[i];point.age=++clock_;point.pinned=point.pinned || pinned;
        if(!tail || pinned || root) point.tail=false;
        if(root) {
            for(uint32_t j=0;j<state.check_count;++j) state.checks[j].root=j==i;
            std::rotate(state.checks.begin(),state.checks.begin()+i,state.checks.begin()+i+1);
        }
        return;
    }
    // One near-tail item, unless it was promoted to an ordinary/pinned point.
    if(tail) for(uint32_t i=0;i<state.check_count;) {
        if(state.checks[i].tail && !state.checks[i].pinned) erase_point(state,i);else ++i;
    }
    if(state.check_count>=options_.checkpoints) {
        std::array<uint64_t,65> stamps{};std::array<bool,65> tails{},pins{};
        bool any_pin=pinned;
        std::array<uint32_t,65> order{};uint32_t n=0;
        if(root && options_.checkpoints>=2) order[n++]=state.check_count;
        for(uint32_t j=0;j<state.check_count;++j) order[n++]=j;
        if(n==state.check_count) order[n++]=state.check_count;
        for(uint32_t i=0;i<n;++i) {
            const auto j=order[i];
            stamps[i]=j==state.check_count?clock_+1:state.checks[j].age;
            tails[i]=j==state.check_count?tail:state.checks[j].tail;pins[i]=j==state.check_count?pinned:state.checks[j].pinned;any_pin=any_pin || pins[i];
        }
        const auto victim=order[program::conv_cache::eviction_victim(stamps.data(),n,options_.checkpoints,tails.data(),any_pin?pins.data():nullptr)];
        if(victim==state.check_count) return;
        erase_point(state,uint32_t(victim));++checkpoint_evictions_;
        if(op_.discard_retained) op_.discard_retained();
    }
    try {
        auto point=park(false);
        if(point) {
            state.checks[state.check_count++]={std::move(point),++clock_,cells,pinned,tail && !root,root};
            ++checkpoint_captures_;
            if(root) {
                for(uint32_t j=0;j+1<state.check_count;++j) state.checks[j].root=false;
                std::rotate(state.checks.begin(),state.checks.begin()+state.check_count-1,state.checks.begin()+state.check_count);
            }
        }
    } catch(const std::bad_alloc&) {} // optional indexing/owned-closure storage
}
SlotSession::ReadResult SlotSession::read_prompt(int slot,Slot& state,const GenerationRequest& request,uint32_t resume,const ReadControl& control) {
    const auto total=uint32_t(request.prompt.size());
    state.fed.assign(request.prompt.begin(),request.prompt.begin()+resume);
    const auto started=std::chrono::steady_clock::now();
    const auto boundaries=cache_idle_?prompt_boundaries(request.prompt,resume,options_.boundaries,request.checkpoint,state.from0):PromptBoundaries{};
    const auto pin=cache_idle_ && request.checkpoint && options_.checkpoints>=3?request.pin:0;
    const bool windowed=options_.prefill_chunk && total-resume>options_.short_read;
    const auto batched_end=prefill_batched_end(total,options_.prefill_until);
    PromptCheckpointSchedule schedule(resume,total,options_.interval,windowed && options_.tail,options_.short_read);
    std::vector<float> logits;
    uint32_t begin=resume;
    for(uint32_t i=resume;i<total;) {
        if(control.cancelled && control.cancelled()) throw Cancelled();
        const bool window=windowed && i<batched_end && total-i>options_.short_read;
        auto end=uint32_t(std::min<uint64_t>(total,uint64_t(i)+(window?options_.prefill_chunk:1)));
        if(window && options_.canonical_prefill_rows) end=std::min(end,i+canonical_prefill_count(total-i,options_.prefill_chunk,options_.short_read,options_.canonical_prefill_rows));
        if(window && end>batched_end) end=uint32_t(batched_end);
        for(auto stop:{pin,boundaries.root,boundaries.turn,boundaries.message}) if(stop>i && stop<end) end=stop;
        const auto flags=schedule.complete(end,end-i);
        const bool semantic=end==pin || end==boundaries.root || end==boundaries.turn || end==boundaries.message;
        const bool scheduled=request.checkpoint && (!options_.prefill_chunk || window || options_.prefill_until) && (flags.periodic || flags.tail);
        const bool capture=cache_idle_ && end<total && (semantic || scheduled);
        // Consecutive scalar logical chunks can share one prefill call until
        // an actual checkpoint cut. No array of per-token cuts is allocated.
        if(window || capture || end==total || control.yield || control.between || control.cancelled || control.progress) {
            const auto input=std::span<const int32_t>(request.prompt).subspan(begin,end-begin);
            const auto start=std::chrono::steady_clock::now();
            uint32_t committed=0;
            auto commit=[&](uint32_t done,std::span<const float> values) {
                if(done<=committed || done>input.size()) throw SessionFatal("slot prefill callback violated committed-prefix contract");
                if(values.size()!=vocabulary_) throw SessionFatal("slot prefill callback logits shape mismatch");
                validate_logits(values,vocabulary_);
                state.fed.assign(request.prompt.begin(),request.prompt.begin()+begin+done);
                logits.assign(values.begin(),values.end());committed=done;
                if(control.progress) control.progress(begin+done,total,std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count());
            };
            if(op_.prefill_cancellable) op_.prefill_cancellable(input,window,commit,control.cancelled);
            else {const auto values=window && op_.prefill_window?op_.prefill_window(input):op_.prefill(input);commit(uint32_t(input.size()),values);}
            if(committed!=input.size()) throw SessionFatal("slot prefill returned without its complete committed window");
            const auto ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
            validate_logits(logits,vocabulary_);
            if(control.cancelled && control.cancelled()) throw Cancelled();
            if(capture) capture_prompt(state,end,end==pin,flags.tail && !flags.periodic && !semantic,end==boundaries.root,
                                       images_below(request.images,end));
            begin=end;
            if(end<total) {
                // A yield must leave more than one chunk AND a short read in
                // the same semantic read segment (generate.cpp:11606-11609).
                uint32_t segment=total-1;
                for(auto cut:{pin,boundaries.root,boundaries.turn,boundaries.message}) if(cut>i && cut<segment) segment=cut;
                const auto remaining=segment>end?segment-end:0;
                if(control.yield) if(auto target=control.yield();target && *target==slot && cache_idle_ &&
                    remaining>std::max(std::max(1u,options_.prefill_chunk),options_.short_read)) {
                    std::function<void()> point;
                    try {point=park();} catch(const AllocationBudgetExceeded&) {} catch(const std::bad_alloc&) {}
                    if(point) {
                        state.image=std::move(point);state.fed.assign(request.prompt.begin(),request.prompt.begin()+end);
                        const auto below=images_below(request.images,end);
                        state.images.assign(below.begin(),below.end());
                        state.partial=true;state.cached=true;state.active=false;++yielded_reads_;
                        return {{},end};
                    }
                }
                if(control.between && control.has_work && control.has_work()) {
                    auto point=park(false); // budget refusal leaves the prompt running, never overcommits
                    if(point) {
                        try {control.between(ms);}
                        catch(...) {const auto error=std::current_exception();point();std::rethrow_exception(error);}
                        point();++interleaved_windows_;
                    }
                }
            }
        }
        i=end;
    }
    return {std::move(logits),0};
}
BatchSlotSession::Admission SlotSession::admit(int slot,const GenerationRequest& request,uint32_t max_new,
                                               const std::function<void(int32_t)>& token) {
    return admit_cooperative(slot,request,max_new,token,{});
}
BatchSlotSession::Admission SlotSession::admit_cooperative(int slot,const GenerationRequest& request,uint32_t max_new,
                                               const std::function<void(int32_t)>& token,const ReadControl& control_read) {
    auto& state=at(slot);
    if(state.active || !max_new || !token || request.prompt.empty() || request.prompt.size()>context_ || !request.bias.empty())
        throw std::invalid_argument("invalid slot admission");
    for(auto id:request.prompt) if(id<0 || uint32_t(id)>=vocabulary_) throw std::invalid_argument("slot token out of vocabulary");
    if(request.pin && uint64_t(request.pin)+1>=request.prompt.size()) throw std::invalid_argument("pinned prefix must end before the prompt's last token");
    // The admission is parsed as GEN 1, so the parser cannot validate the
    // remaining batch draws. Keep Generation's pre-mutation counter gate.
    if(uint64_t(max_new-1)>UINT64_MAX-request.sampling.counter)
        throw std::overflow_error("slot sampling draw counter overflow");
    const std::array<float,1> dummy{0};(void)sampling_distribution(dummy,{},request.sampling);
    if(control_read.cancelled && control_read.cancelled()) {Admission cancelled;cancelled.prompt_tokens=uint32_t(request.prompt.size());cancelled.cancelled=true;return cancelled;}
    // Generation's request law, applied here before the first Model touch: BGENI is upstream's `GENI 1`
    // rewrite (:10349-10362), so the image request needs the owner's per-request M-RoPE/row install, and a
    // text request installs the identity table so it never decodes with whatever the previous slot left.
    if(request.embeddings) {
        if(!op_.set_vision) throw std::invalid_argument("this engine was started without --vision");
        if(request.image_rows.size()!=request.prompt.size()) throw std::invalid_argument("image row list does not cover the prompt");
        if(request.positions.size()!=(uint64_t(context_)+64)*3) throw std::invalid_argument("image position table shape mismatch");
    }
    if(op_.set_vision) op_.set_vision(request);
    const bool control=op_.control_vector?request.control_vector:true;
    size_t resume=0;int source=-1,source_check=-1;
    if(op_.before_foreign) op_.before_foreign();
    auto external=cache_idle_ && op_.select_prefix?op_.select_prefix(request.prompt,request.prompt.size()-1,control,request.images):std::optional<GenerationPrefix>{};
    if(external) {
        if(external->tokens.empty() || external->tokens.size()>=request.prompt.size() || !external->restore ||
           !std::equal(external->tokens.begin(),external->tokens.end(),request.prompt.begin()) ||
           !images_match(external->images,request.images,external->tokens.size())) throw SessionFatal("external slot source contract mismatch");
        resume=external->tokens.size();source=-2;
    }
    // Upstream's idle-slot branch (:10935) only runs for a request without pictures; the end state carries no
    // picture list there, but a slot checkpoint does and must then carry none (:10939-10948).
    if(cache_idle_ && request.images.empty()) for(size_t i=0;i<slots_.size();++i) {
        const auto& candidate=slots_[i];
        // Upstream caps reuse at prompt length minus one. Ties keep the first
        // slot encountered; an active slot never lends its mutable state.
        if(candidate.active || !candidate.cached || candidate.control!=control || !candidate.image) continue;
        const auto consider=[&](size_t cells,int checkpoint) {
            if(!cells || cells>=request.prompt.size() || cells<=resume || cells>candidate.fed.size()) return;
            if(checkpoint>=0 && !images_below(candidate.images,candidate.checks[size_t(checkpoint)].cells).empty()) return;
            if(std::equal(candidate.fed.begin(),candidate.fed.begin()+cells,request.prompt.begin())) {resume=cells;source=int(i);source_check=checkpoint;}
        };
        consider(candidate.fed.size(),-1);
        for(uint32_t j=0;j<candidate.check_count;++j) consider(candidate.checks[j].cells,int(j));
    }
    std::array<PromptPoint,64> incoming{};uint32_t incoming_count=0;bool from0=!resume;std::function<void()> incoming_end;
    if(source==-2) {
        try {
            external->restore();GenerationRead read;read.tokens=external->tokens;read.checkpoints=external->checkpoints;
            read.images=external->images;
            auto chain=collect_points(read);incoming=std::move(chain.checks);incoming_count=chain.check_count;
        }
        catch(const SessionFatal&) {throw;}
        catch(const std::exception&) {resume=0;source=-1;from0=true;}
    }
    if(source>=0) {
        try {
            auto& candidate=slots_[size_t(source)];
            if(source_check>=0) {
                incoming[0]=candidate.checks[size_t(source_check)];incoming[0].age=++clock_;incoming_count=1;incoming[0].image();
                incoming_end=incoming[0].image;
            } else {
                candidate.image();
                if(source==slot && candidate.partial) {incoming=candidate.checks;incoming_count=candidate.check_count;from0=candidate.from0;incoming_end=candidate.image;}
            }
        }
        catch(const SessionFatal&) {throw;}
        catch(const std::exception&) {release(source);incoming={};incoming_count=0;incoming_end={};resume=0;source=-1;from0=true;}
    }
    external.reset(); // history has been consumed; do not pin an unused foreign image
    // Do not copy or reallocate the history buffer. The target can itself be
    // the source: restore it first, then release its owned image before park.
    release(slot);
    state.checks=std::move(incoming);state.check_count=incoming_count;state.from0=from0;state.control=control;
    state.images.assign(request.images.begin(),request.images.end());
    if(request.checkpoint && request.pin && options_.checkpoints>=3) for(uint32_t j=0;j<state.check_count;++j) state.checks[j].pinned=state.checks[j].cells==request.pin;
    try {
        if(incoming_end && op_.import_prefixes) {
            state.fed.assign(request.prompt.begin(),request.prompt.begin()+resume);
            auto read=read_view(state);read.restore=incoming_end;op_.import_prefixes(read);incoming_end={};
        }
        if(source==-1) {op_.reset();if(op_.control_vector) op_.control_vector(control);}
        const auto read=read_prompt(slot,state,request,uint32_t(resume),control_read);
        if(read.yielded) {reused_tokens_+=resume;Admission yielded{uint32_t(request.prompt.size()),false,uint32_t(resume),false,slot,read.yielded};yielded.prompt_read=read.yielded-uint32_t(resume);return yielded;}
        if(control_read.cancelled && control_read.cancelled()) throw Cancelled();
        state.fed.assign(request.prompt.begin(),request.prompt.end());
        state.control=control;state.sampling=request.sampling;
        state.pending=sample(read.logits,state.fed,state.sampling,{});
        // Ordinary BSlot keeps only its deepest point. A yielded read alone
        // keeps the full bounded chain (generate.cpp:11620-11624,12973-12982).
        if(state.check_count>1) {
            uint32_t best=0;for(uint32_t j=1;j<state.check_count;++j) if(state.checks[j].cells>state.checks[best].cells) best=j;
            auto point=std::move(state.checks[best]);while(state.check_count) erase_point(state,state.check_count-1);
            state.checks[0]=std::move(point);state.check_count=1;
        }
        const bool continues=max_new>1 && !is_stop(state.pending);
        if(continues) state.image=park();
        else {while(state.check_count) erase_point(state,state.check_count-1);if(op_.discard_retained) op_.discard_retained();}
        state.partial=false;
        if(control_read.cancelled && control_read.cancelled()) throw Cancelled();
        state.active=continues;
        ++state.sampling.counter;reused_tokens_+=resume;
        const auto first=state.pending;
        token(first);
        Admission admitted{uint32_t(request.prompt.size()),continues,uint32_t(resume),is_stop(first)};admitted.prompt_read=uint32_t(request.prompt.size()-resume);
        if(!admitted.eos && control_read.cancelled && control_read.cancelled()) {admitted.continues=false;admitted.cancelled=true;finish(slot);}
        return admitted;
    } catch(const Cancelled&) {
        Admission cancelled;cancelled.prompt_tokens=uint32_t(request.prompt.size());cancelled.reused_tokens=uint32_t(resume);cancelled.cancelled=true;
        reused_tokens_+=resume;
        cancelled.prompt_read=state.fed.size()>=resume?uint32_t(state.fed.size()-resume):0;
        if(cache_idle_ && request.checkpoint && !state.fed.empty()) {
            if(!state.image) state.image=park(false);
            if(state.image) {state.partial=true;state.cached=true;state.active=false;return cancelled;}
        }
        release(slot);return cancelled;
    } catch(const NonFiniteOutput&) {Admission failed;failed.prompt_tokens=uint32_t(request.prompt.size());failed.reused_tokens=uint32_t(resume);failed.nonfinite=true;failed.prompt_read=state.fed.size()>=resume?uint32_t(state.fed.size()-resume):0;reused_tokens_+=resume;release(slot);return failed;}
    catch(...) {release(slot);throw;}
}
BatchSlotSession::Step SlotSession::step(int slot) {
    return step_window(slot,1).rows[0];
}
BatchSlotSession::Window SlotSession::step_window(int slot,uint32_t limit) {
    if(!limit || limit>2) throw std::invalid_argument("invalid slot window limit");
    auto& state=at(slot);
    if(!state.active || !state.image || state.fed.size()>=context_) throw std::logic_error("slot has no valid continuation");
    if(op_.before_foreign) op_.before_foreign();
    try {
        state.image();state.image={};
        if(op_.speculate && limit==2 && state.fed.size()+3<=context_ && state.sampling.counter<UINT64_MAX) {
            auto result=op_.speculate(state.pending,state.fed,state.sampling);
            if(result) {
                const auto draft=std::span(&result->draft,1);auto& selected=result->selected;
                validate_drafts(draft,vocabulary_,1);
                if(!valid_verified_candidates(selected,draft,vocabulary_,limit,stops_))
                    throw SessionFatal("slot verify hook violated accepted-prefix/logit contract");
                state.fed.push_back(state.pending);
                if(selected.retained) state.fed.push_back(result->draft.token);
                Window output;output.count=uint32_t(selected.picks.size());
                for(uint32_t row=0;row<output.count;++row) output.rows[row]={selected.picks[row].token,is_stop(selected.picks[row].token),false};
                state.pending=selected.picks.back().token;state.sampling.counter+=output.count;
                state.image=park(); // atomic publication only after continuation is owned
                return output;
            }
        }
        const auto logits=op_.decode(state.pending);
        if(logits.size()!=vocabulary_) throw std::runtime_error("slot logits shape mismatch");
        if(!finite_logits(logits,vocabulary_)) throw NonFiniteOutput("non-finite slot logits");
        state.fed.push_back(state.pending);
        const auto next=sample(logits,state.fed,state.sampling,{});
        state.image=park();++state.sampling.counter;state.pending=next;
        Window output;output.count=1;output.rows[0]={next,is_stop(next),false};return output;
    } catch(const NonFiniteOutput&) {release(slot);Window output;output.count=1;output.rows[0].nonfinite=true;return output;}
    catch(...) {release(slot);throw;}
}
}
