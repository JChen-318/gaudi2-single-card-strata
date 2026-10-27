#include "strata/hpu/generation.hpp"
#include "strata/hpu/spec_gate.hpp"
#include "strata/hpu/prefill_schedule.hpp"
#include "strata/spec/suffix_drafter.hpp"
#include "strata/spec/draft_source.hpp"
#include "strata/core/conversation_memory.hpp"
#include "strata/hpu/allocation_budget.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <optional>
#include <cstdlib>
#include <limits>

namespace strata::hpu {
using Clock=std::chrono::steady_clock;
namespace {
double elapsed(Clock::time_point start) {return std::chrono::duration<double,std::milli>(Clock::now()-start).count();}
// Scalar reference evaluation is token-major. Only IDs need buffering to
// publish the same layer-major zero-weight window format as native verify.
// <=7 rows + one scalar export fit Model's routing single-export envelope.
class ScalarRoutingWindow {
public:
    ScalarRoutingWindow(uint32_t layers,uint32_t rows,uint32_t experts):layers_(layers),rows_(rows),experts_(experts) {
        if(!layers || layers>4096 || !rows || rows>7 || !experts || experts>4096) throw std::invalid_argument("scalar routing geometry exceeds bounded envelope");
    }
    void append(std::span<const RoutingRecord> records) {
        if(seen_>=rows_ || records.size()!=layers_ || records.empty()) throw std::logic_error("scalar routing source layer/row count mismatch");
        const auto k=records.front().ids.size();
        if(!k || k>64 || k>experts_ || (top_k_ && k!=top_k_)) throw std::logic_error("scalar routing top-k mismatch");
        for(uint32_t layer=0;layer<layers_;++layer) {
            const auto& record=records[layer];
            if(record.layer!=int32_t(layer) || record.ids.size()!=k || record.weights.size()!=k) throw std::logic_error("scalar routing source layout mismatch");
            for(size_t at=0;at<k;++at) if(record.ids[at]<0 || uint32_t(record.ids[at])>=experts_ || !std::isfinite(record.weights[at]) || record.weights[at]<0) throw std::logic_error("scalar routing source invalid expert/weight");
        }
        if(!top_k_) {
            top_k_=uint32_t(k);ids_.resize(uint64_t(layers_)*rows_*top_k_);
            if(ids_.capacity()!=ids_.size()) throw std::length_error("scalar routing storage exceeds admitted capacity");
        }
        for(uint32_t layer=0;layer<layers_;++layer) std::copy(records[layer].ids.begin(),records[layer].ids.end(),ids_.begin()+(uint64_t(layer)*rows_+seen_)*top_k_);
        ++seen_;
    }
    void emit(DumpSinks& sink) const {
        if(seen_!=rows_) throw std::logic_error("scalar routing source omitted physical rows");
        for(uint32_t layer=0;layer<layers_;++layer) sink.routing_window(int32_t(layer),top_k_,rows_,ids_.data()+uint64_t(layer)*rows_*top_k_);
        sink.flush_routing();
    }
private:
    uint32_t layers_,rows_,experts_,top_k_=0,seen_=0;std::vector<int32_t> ids_;
};
}
void validate_generation_options(const GenerationOptions& options) {
    if(options.resident_routing && !options.resident_routing->routing_session_matches(options.dumps,options.dump_shape))
        throw std::invalid_argument("resident routing writer must be open, routing-only and match the configured path/geometry");
    if(options.max_draft>7 || (options.suffix_min_match && (options.suffix_min_match<3 || options.suffix_min_match>64)) ||
       (options.verify_mode!=VerifyMode::Exact && options.verify_mode!=VerifyMode::Probabilistic))
        throw std::invalid_argument("invalid verify options: at most 7 drafts, suffix match 3..64");
    if(options.cache_slots>kMaxGenerationCheckpoints || options.cache_every>INT32_MAX || (options.cache_slots && !options.cache_bytes)) throw std::invalid_argument("invalid conversation cache limits");
    if(options.lookup_chain>7 || options.lookup_chain_min<3 || options.lookup_chain_min>64) throw std::invalid_argument("invalid lookup chain limits");
    if(options.prefill_chunk>32768 || options.short_read>INT32_MAX || (options.canonical_prefill_rows &&
       (options.canonical_prefill_rows>options.prefill_chunk || options.prefill_chunk%options.canonical_prefill_rows))) throw std::invalid_argument("invalid prefill window/short-read size");
    // Upstream clamps a request's spec_min_p into [0,1] (:10697); a start-up value outside that range could never
    // have been clamped by the parser, so it is refused here rather than silently compared against probabilities.
    if(!std::isfinite(options.spec_min_p) || options.spec_min_p<0 || options.spec_min_p>1)
        throw std::invalid_argument("invalid spec draft gate");
    (void)prompt_boundaries({},0,options.boundaries,true);
}
Generation::Generation(uint32_t vocabulary,uint32_t context,std::vector<int32_t> stops,GenerationOperations op,GenerationOptions options)
    :vocabulary_(vocabulary),context_(context),stop_ids_(std::move(stops)),operations_(std::move(op)),options_(options),
     draft_policy_(int(options.max_draft+1)),
     cache_(options.cache_slots,options.cache_bytes,operations_.available_host_bytes?operations_.available_host_bytes:core::available_host_bytes,options.cache_min_free,context,vocabulary,options.cache_min_tokens),
     window_hashes_(options.window_hashes) {
    validate_generation_options(options_);
    // Upstream hashes the window's T final residual rows (:13826-13831), so a record armed without a residual
    // producer, or without the row shape to size them with, would have to hash something else. Refusing is the
    // port's rule for every dump/diagnostic armed with no producer: a well-formed file over less data reads as a
    // determinism break, which is the one thing this file exists to detect.
    if(window_hashes_.armed()) {
        if(!operations_.residual) throw std::invalid_argument("--window-hashes needs the post-last-layer residual producer");
        if(options_.dump_shape.hc<=0 || options_.dump_shape.n_embd<=0)
            throw std::invalid_argument("--window-hashes needs the residual row shape (hc and n_embd)");
    }
    // Existing scalar-window metadata reserve is 4096 bytes outside its root
    // token/logit vectors. Policy and <=7 borrowed confidence cells fit it.
    static_assert(sizeof(spec::DraftPolicy)+7*sizeof(float)<1024);
    // The retained parked image lives with the state owner, but it is charged against this cache's byte budget
    // exactly as upstream counts reuse_ inside ConversationCache::bytes().
    cache_.set_retained(operations_.parked_retained_bytes,operations_.discard_parked);
    if(!vocabulary || vocabulary>INT32_MAX || !context || context>INT32_MAX || !operations_.reset || !operations_.decode ||
       (options_.cache_slots && (!operations_.checkpoint || !operations_.checkpoint_bytes))) throw std::invalid_argument("invalid generation geometry/operations");
    for(auto token:stop_ids_) if(token<0 || uint32_t(token)>=vocabulary_) throw std::invalid_argument("invalid stop token ID");
    fed_.reserve(context);logits_.reserve(vocabulary);
    interrupted_reads_.reserve(std::max(options_.cache_slots,1u));
    if(fed_.capacity()!=context || logits_.capacity()!=vocabulary || interrupted_reads_.capacity()!=std::max(options_.cache_slots,1u))
        throw std::length_error("generation STL capacity exceeds admitted metadata geometry");
}
void Generation::invalidate() {
    suspended_live_={};model_attached_=false;
    cache_.clear();fed_.clear();logits_.clear();valid_=false;interrupted_reads_.clear();sequence_images_.clear();
    // The retained parked image describes the conversation that just died (generate.cpp:10599).
    if(operations_.discard_parked) operations_.discard_parked();
}
void Generation::reset_sequence() {
    const bool detached=!model_attached_;
    suspended_live_={};fed_.clear();logits_.clear();valid_=false;sequence_images_.clear();operations_.reset();
    // reset() preserves the Model's arithmetic mode. A foreign slot may have
    // changed it while this owner's metadata still names its previous mode.
    if(detached && operations_.control_vector) operations_.control_vector(control_enabled_);
    valid_=true;model_attached_=true;
}
namespace {
// Upstream's --check-logits report (src/program/generate.cpp:13416-13419), byte for byte: the tool passes a
// "strata generate: " message through unchanged, so this is the exact operator-visible text. It runs only when the
// flag is on; without it the protocol path keeps its `NonFiniteOutput` -> `DONE nonfinite` behaviour.
void check_logits_row(const GenerationOptions& options,const std::span<const float> row,const int64_t position,
                      const uint32_t vocabulary) {
    if(!options.check_logits) return;
    uint64_t bad=0;for(const float value:row) if(!std::isfinite(value)) ++bad;
    if(bad!=0)
        throw std::runtime_error("strata generate: "+std::to_string(bad)+" of "+std::to_string(vocabulary)+
                                 " logits are not finite at position "+std::to_string(position));
    // Port addition: a row whose every logit is the same value (all zero in practice) is finite, so upstream's scan
    // passes it, yet it can only come from a broken forward pass (zero weights, an unbound input). Refuse it too.
    if(row.size()>1 && std::all_of(row.begin(),row.end(),[&](const float value){return value==row.front();}))
        throw std::runtime_error("strata generate: all "+std::to_string(vocabulary)+" logits equal "+
                                 std::to_string(row.front())+" at position "+std::to_string(position)+
                                 " (a constant row means the forward pass is broken)");
}
}
void Generation::reset() {cache_.clear();reset_sequence();interrupted_reads_.clear();nonfinite_pending_=false;if(operations_.discard_parked) operations_.discard_parked();}
bool Generation::flush_nonfinite() {
    if(!nonfinite_pending_) return false;
    invalidate();nonfinite_pending_=false;return true;
}
PrefixCacheStats Generation::cache_stats() const {return cache_.stats();}
Generation::MetadataStats Generation::metadata_stats() const {
    MetadataStats out;out.envelope=generation_cache_metadata_bytes(context_,vocabulary_);
    out.vector_bytes=cache_.metadata_vector_bytes()+interrupted_reads_.capacity()*sizeof(decltype(interrupted_reads_)::value_type);
    out.interrupted_entries=uint32_t(interrupted_reads_.size());
    for(const auto& read:interrupted_reads_) {
        out.vector_bytes=cache_metadata_add(out.vector_bytes,read.tokens.capacity()*sizeof(int32_t));
        out.vector_bytes=cache_metadata_add(out.vector_bytes,read.images.capacity()*sizeof(RequestImage));
    }
    return out;
}
void Generation::release_checkpoints() {
    cache_.clear();
    // Upstream's clear() drops the reuse slot with the parked conversations (conversation_cache.hpp:294-300).
    if(operations_.discard_parked) operations_.discard_parked();
}
PrefixCache::Reclaim Generation::reclaim_for_save(uint64_t need,std::span<const int32_t> selected) {
    return cache_.reclaim_for(need,selected);
}
bool Generation::cache_current(bool pinned,bool root,bool tail) {
    if(!options_.cache_slots || !valid_ || fed_.empty() || !model_attached_) return false;
    const uint64_t superseded_before=cache_.stats().superseded;
    const uint64_t short_before=cache_.stats().min_tokens_skipped;
    parked_reused_=0;
    // Upstream parks one image per layer-split stage (src/program/generate.cpp:8244-8285) and refuses to park a
    // split that shares one GPU (:2382-2386). A single-card backend can therefore never see a parkable stage
    // count above one; if a caller presents one anyway, parking is refused and said - not silently reduced to a
    // one-stage image that would then be restored as if it were the whole session.
    const uint32_t stages=operations_.parking_stages?operations_.parking_stages():0;
    if(stages>1) {
        park_refused_stages_=stages;
        if(park_events_&&park_events_->park_refused) park_events_->park_refused(stages);
        return false;
    }
    park_refused_stages_=0;
    bool stored=false;
    // The charge is the parked image's own size at this extent (upstream's capture-before-save estimate, which
    // equals the stored image because a grown image ends up exactly as large as a fresh park of the extent).
    const uint64_t cost=operations_.checkpoint_park_bytes?operations_.checkpoint_park_bytes(fed_.size()):operations_.checkpoint_bytes;
    try {stored=cache_.store(fed_,logits_,cost,operations_.checkpoint,pinned,root,tail,control_enabled_,
                             images_below(sequence_images_,fed_.size()));}
    catch(const std::bad_alloc&) {return false;} // optional host indexing allocation
    if(!stored) {
        // Upstream reports the min-tokens skip and still returns true (:8227-8236); the other skips (budget, RAM
        // floor) stay silent here exactly as they do upstream, where they are the paths this one is compared to.
        if(cache_.stats().min_tokens_skipped>short_before && park_events_ && park_events_->park_skipped_short)
            park_events_->park_skipped_short(fed_.size(),uint64_t(options_.cache_min_tokens),cache_.stats().entries);
        return false;
    }
    // Upstream's park line reports what the reuse carried over and the counters the cache keeps
    // (src/program/generate.cpp:8371); the superseded counter tells a grow-in-place park from one that took a
    // fresh slot, so nothing here has to be told twice.
    parked_reused_=operations_.parked_reused_bytes?operations_.parked_reused_bytes():0;
    if(park_events_&&park_events_->parked)
        park_events_->parked(fed_.size(),parked_reused_,cache_.stats().superseded>superseded_before);
    return true;
}
void Generation::set_spec_knobs(const SpecKnobRequest& request) {
    spec_knobs_.configure(request,vocabulary_,options_.max_draft);
}
void Generation::adopt(std::span<const int32_t> tokens,std::span<const float> logits,bool control_vector,std::span<const RequestImage> images) {
    if(tokens.size()>context_ || logits.size()!=(tokens.empty()?0:vocabulary_)) throw std::invalid_argument("restored generation shape mismatch");
    for(auto token:tokens) if(token<0 || uint32_t(token)>=vocabulary_) throw std::invalid_argument("restored token out of vocabulary");
    if(!all_finite(logits)) throw std::invalid_argument("non-finite restored logits");
    uint32_t previous=0;
    for(size_t i=0;i<images.size();++i) {
        if(uint64_t(images[i].start)>=tokens.size() || (i && images[i].start<=previous))
            throw std::invalid_argument("restored image identity shape mismatch");
        previous=images[i].start;
    }
    // Both capacities are reserved at construction, before model state changes.
    cache_.clear();
    suspended_live_={};model_attached_=true;nonfinite_pending_=false;
    // An adopted sequence was not read from token 0 by this process, so no partial-read marker carries over.
    fed_.assign(tokens.begin(),tokens.end());logits_.assign(logits.begin(),logits.end());control_enabled_=operations_.control_vector?control_vector:true;valid_=true;interrupted_reads_.clear();
    sequence_images_.assign(images.begin(),images.end());
}
void Generation::suspend_live() {
    if(foreign_scope_ || !model_attached_) return;
    if(!fed_.empty()) {
        const auto cached=cache_.peek(fed_,fed_.size(),control_enabled_,images_below(sequence_images_,fed_.size()));
        if(cached && cache_.entry(*cached).tokens.size()==fed_.size()) suspended_live_=cache_.entry(*cached).restore;
        else if(operations_.checkpoint) {
            try {suspended_live_=operations_.checkpoint();}
            catch(const AllocationBudgetExceeded&) {} catch(const std::bad_alloc&) {}
        }
    }
    model_attached_=false;
    // Live reuse is optional. Never retain metadata that refers to a Model
    // another session is about to overwrite when no owned image was possible.
    if(!suspended_live_) {fed_.clear();logits_.clear();valid_=false;}
}
void Generation::attach_live() {
    if(!valid_ || fed_.empty() || nonfinite_pending_) throw std::invalid_argument("no complete main session to save");
    if(model_attached_) return;
    if(!suspended_live_) throw std::invalid_argument("main session is no longer resident");
    try {suspended_live_();}
    catch(const std::exception& error) {invalidate();throw SessionFatal(std::string("main session attachment failed: ")+error.what());}
    suspended_live_={};model_attached_=true;
}
void Generation::report_picture_mismatch(std::span<const int32_t> prompt,std::span<const RequestImage> request,
                                        uint64_t reused,bool control) {
    if(!operations_.picture_mismatch || !options_.cache_slots || request.empty()) return;
    uint64_t deep=0;std::vector<RequestImage> stored;
    const auto consider=[&](std::span<const int32_t> tokens,std::span<const RequestImage> images) {
        // Upstream's consider() (:10966-10972): deeper than the reuse, this prompt's token prefix, other pictures.
        if(tokens.empty() || tokens.size()>=prompt.size() || tokens.size()<=std::max(deep,reused) ||
           !std::equal(tokens.begin(),tokens.end(),prompt.begin()) || images_match(images,request,tokens.size())) return;
        deep=tokens.size();stored.assign(images.begin(),images.end());
    };
    if(valid_ && control==control_enabled_) consider(fed_,images_below(sequence_images_,fed_.size()));
    cache_.visit_token_prefixes(prompt,control,[&](const PrefixCache::Entry& entry){consider(entry.tokens,entry.images);});
    if(!deep) return;
    const auto now=images_below(request,deep);
    size_t picture=0;
    while(picture<now.size() && picture<stored.size() && now[picture]==stored[picture]) ++picture;
    operations_.picture_mismatch(deep,uint64_t(picture)+1,picture<now.size()?int64_t(now[picture].start):-1);
}
std::optional<GenerationPrefix> Generation::select_prefix(std::span<const int32_t> prompt,uint64_t maximum,bool control,std::span<const RequestImage> images) {
    if(nonfinite_pending_ || !options_.cache_slots || prompt.empty()) return {};
    maximum=std::min<uint64_t>(maximum,prompt.size()-1);
    suspend_live();
    const auto cached=cache_.find(prompt,maximum,control,images);
    const auto live_images=images_below(sequence_images_,fed_.size());
    const bool live=valid_ && suspended_live_ && control==control_enabled_ && !fed_.empty() && fed_.size()<=maximum &&
        std::equal(fed_.begin(),fed_.end(),prompt.begin()) && images_match(live_images,images,fed_.size());
    std::optional<GenerationPrefix> result;
    if(live && (!cached || fed_.size()>=cache_.entry(*cached).tokens.size())) {
        result=GenerationPrefix{fed_,suspended_live_,{}};
        result->images.assign(live_images.begin(),live_images.end());
    } else if(cached) {
        const auto& entry=cache_.entry(*cached);result=GenerationPrefix{entry.tokens,entry.restore,{}};
        result->images=entry.images;
    }
    if(result) result->checkpoints=[this,tokens=result->tokens,control,images](const GenerationCheckpointVisitor& visit) {
        cache_.visit_prefixes(tokens,control,[&](const PrefixCache::Entry& point) {
            GenerationCheckpoint checkpoint{point.tokens,point.restore,point.age,point.pinned,point.root,point.tail};
            checkpoint.images=point.images;visit(checkpoint);
        },images);
    };
    report_picture_mismatch(prompt,images,result?result->tokens.size():0,control);
    return result;
}
void Generation::cooperate(const GenerationEvents& events) {
    if(!events.cooperate || !operations_.checkpoint || !model_attached_ || (events.has_foreign_work && !events.has_foreign_work())) return;
    std::function<void()> image;
    try {image=operations_.checkpoint();} catch(const AllocationBudgetExceeded&) {return;} catch(const std::bad_alloc&) {return;}
    if(!image) throw std::logic_error("empty cooperative generation image");
    auto restore=[&] {
        foreign_scope_=false;
        try {image();}
        catch(const std::exception& error) {invalidate();throw SessionFatal(std::string("cooperative generation restore failed: ")+error.what());}
    };
    foreign_scope_=true;
    try {events.cooperate();} catch(...) {const auto error=std::current_exception();restore();std::rethrow_exception(error);}
    restore();
}
void Generation::import_prefixes(const GenerationRead& read) {
    if(!options_.cache_slots || !read.checkpoints || !read.restore || !read.logits) return;
    auto restore_end=[&] {
        try {read.restore();}
        catch(const std::exception& error) {invalidate();throw SessionFatal(std::string("imported chain endpoint restore failed: ")+error.what());}
    };
    try {
        read.checkpoints([&](const GenerationCheckpoint& point) {
            if(point.tokens.empty() || point.tokens.size()>read.tokens.size() || !point.restore ||
               !std::equal(point.tokens.begin(),point.tokens.end(),read.tokens.begin()) ||
               !images_match(point.images,read.images,point.tokens.size())) throw SessionFatal("imported checkpoint prefix/picture contract mismatch");
            point.restore();auto values=read.logits();
            if(values.size()!=vocabulary_ || !std::all_of(values.begin(),values.end(),[](float x){return std::isfinite(x);})) throw SessionFatal("invalid imported checkpoint logits");
            const auto cost=operations_.checkpoint_park_bytes?operations_.checkpoint_park_bytes(point.tokens.size()):operations_.checkpoint_bytes;
            (void)cache_.store(point.tokens,values,cost,[&]{return point.restore;},point.pinned,point.root,point.tail,read.control_vector,point.images);
        });
    } catch(...) {const auto error=std::current_exception();restore_end();std::rethrow_exception(error);}
    restore_end();
}
Generation::ScalarCandidates Generation::scalar_candidates(std::span<const DraftRow> drafts,const GenerationRequest& request,
                                                            uint32_t emitted,const std::function<bool()>& cancelled) {
    if(drafts.empty() || drafts.size()>7 || !valid_ || !model_attached_ || (dumps_ && options_.dumps.token_positions()))
        throw std::logic_error("invalid scalar candidate transaction");
    const std::vector<int32_t> root_tokens(fed_);const std::vector<float> root_logits(logits_);
    if(root_tokens.capacity()!=root_tokens.size() || root_logits.capacity()!=root_logits.size())
        throw std::length_error("scalar verification metadata exceeds admitted capacity");
    std::function<void()> root;
    const auto cached=cache_.peek(fed_,fed_.size(),control_enabled_,images_below(sequence_images_,fed_.size()));
    if(cached && cache_.entry(*cached).tokens.size()==fed_.size()) root=cache_.entry(*cached).restore;
    else if(operations_.checkpoint) root=operations_.checkpoint();
    if(!root) throw std::logic_error("scalar candidate verification requires an owned bounded checkpoint");
    auto restore=[&] {
        try {root();}
        catch(const std::exception& error) {invalidate();throw SessionFatal(std::string("scalar candidate rollback failed: ")+error.what());}
        fed_.assign(root_tokens.begin(),root_tokens.end());logits_.assign(root_logits.begin(),root_logits.end());valid_=true;model_attached_=true;
    };
    ScalarCandidates selected;bool selecting=true;std::optional<ScalarRoutingWindow> routing;
    if(dumps_ && dumps_->wants_routing()) routing.emplace(uint32_t(options_.dump_shape.n_layers),uint32_t(drafts.size()),uint32_t(options_.dump_shape.n_expert));
    try {
        // Evaluate EVERY proposed physical row, even after an early mismatch
        // or EOS. A nonfinite later row invalidates the whole window upstream.
        for(size_t row=0;row<=drafts.size();++row) {
            if(cancelled && cancelled()) throw Cancelled();
            if(selecting) {
                auto parameters=request.sampling;parameters.counter+=uint64_t(emitted)+row;
                const auto* draft=row<drafts.size()?&drafts[row]:nullptr;
                auto pick=verify_pick(logits_,fed_,parameters,request.bias,draft,options_.verify_mode);
                if(spec_knobs_.follow_active()) {
                    const auto followed=spec_knobs_.follow_row(uint64_t(emitted)+row,pick.token);
                    pick.token=followed.token;if(draft) pick.accepted=pick.token==draft->token;
                    selected.follow_differ+=followed.differs;++selected.follow_emitted;
                }
                selected.picks[selected.count++]=pick;
                selected.stopped=options_.stop_eos && std::find(stop_ids_.begin(),stop_ids_.end(),pick.token)!=stop_ids_.end();
                if(draft && pick.accepted && !selected.stopped) ++selected.retained;
                else selecting=false;
            }
            if(row<drafts.size()) {
                feed(drafts[row].token,cancelled,false);
                if(routing) routing->append(operations_.routing());
            }
        }
        if(routing) {routing->emit(*dumps_);routing_written_+=uint64_t(options_.dump_shape.n_layers)*drafts.size();}
        restore();
        // Replay only the selected prefix from the identical owned root.
        // Cancellation/failure during replay also restores that same root.
        for(uint32_t row=0;row<selected.retained;++row) feed(drafts[row].token,cancelled,false);
        if(cancelled && cancelled()) throw Cancelled();
    } catch(const SessionFatal&) {invalidate();throw;}
    catch(...) {const auto error=std::current_exception();restore();std::rethrow_exception(error);}
    return selected;
}
void Generation::feed(int32_t token,const std::function<bool()>& cancelled,bool diagnostics) {
    // The position this token occupies in the sequence: one dumped record per position, from 0
    // (src/program/generate.cpp:6088-6095 and :13405-13432).
    const int64_t position=int64_t(fed_.size());
    auto result=operations_.decode(token,cancelled);
    if(result.size()!=vocabulary_) {reset();throw std::runtime_error("invalid model logits during generation");}
    check_logits_row(options_,result,position,vocabulary_);
    if(!all_finite(result)) throw NonFiniteOutput("non-finite model logits during generation");
    logits_=std::move(result);fed_.push_back(token);
    if(window_open_) {
        // Upstream hashes the window's final residual rows as copied down from the device (:13826-13831). Every row
        // this engine feeds while a window is open contributes exactly one row, in order.
        const auto residual=operations_.residual();
        residual_queue_.insert(residual_queue_.end(),residual.begin(),residual.end());
    }
    if(dumps_ && diagnostics) {
        // --dump-logits applies upstream's storage selection itself (:13405-13406); --dump-final-r is
        // appended after every processed position on the token path (:13363-13368); --dump-layers writes one
        // ladder per position from the in-model capture (:13381-13382).
        dumps_->logits(position,logits_);
        if(dumps_->wants_final_r()) {
            if(!operations_.residual) throw std::runtime_error("--dump-final-r needs the post-last-layer residual producer");
            dumps_->final_residual(position,token,operations_.residual());
        }
        if(dumps_->wants_layers()) {
            if(!operations_.layer_ladder) throw std::runtime_error("--dump-layers needs the in-model per-layer capture");
            dumps_->layers(operations_.layer_ladder());
        }
        if(dumps_->wants_halves()) dumps_->halves(operations_.halves());
        if(dumps_->wants_routing()) {
            if(!operations_.routing) throw std::runtime_error("--dump-routing needs the router's ids/weights producer");
            uint64_t written=0;
            for(const auto& record:operations_.routing()) {
                if(record.ids.size()!=record.weights.size()) throw std::runtime_error("routing record ids/weights length mismatch");
                dumps_->routing(record.layer,int64_t(record.ids.size()),record.ids.data(),record.weights.data());
                ++written;
            }
            dumps_->flush_routing();
            routing_written_+=written;
        }
    }
}
// Upstream's `--window-hashes` record set. A window is opened at the top of every round of the speculative loop and
// closed when that round leaves its scope - by continue, by break, or by an exception - so a round that never
// completes cannot be written as though it had, which is the failure mode a determinism baseline must not hide.
Generation::WindowRound::WindowRound(Generation* generation) : own(generation) { own->begin_window(); }
Generation::WindowRound::~WindowRound() { own->close_window(); }

void Generation::begin_window() {
    if(!window_hashes_.armed() || !valid_ || !model_attached_) return;
    window_position_=int64_t(fed_.size());
    window_tokens_.clear();
    window_open_=true;
}

void Generation::close_window() {
    if(!window_open_) return;
    window_open_=false;
    const uint32_t rows=uint32_t(window_tokens_.size());
    if(rows==0) return;                       // a round that emitted nothing has no window to record
    const uint64_t row_floats=uint64_t(options_.dump_shape.hc)*uint64_t(options_.dump_shape.n_embd);
    // Consume exactly the window's own rows: the round that just closed may already have fed the NEXT window's
    // first row, and a window whose rows were not all fed is a wiring break, not something to hash short.
    if(residual_queue_.size()<uint64_t(rows)*row_floats)
        throw std::logic_error("window hash has fewer residual rows than argmaxes: rows=" +
                               std::to_string(residual_queue_.size()/row_floats) + " tokens=" + std::to_string(rows));
    const std::vector<float> window_rows(residual_queue_.begin(),residual_queue_.begin()+std::ptrdiff_t(uint64_t(rows)*row_floats));
    residual_queue_.erase(residual_queue_.begin(),residual_queue_.begin()+std::ptrdiff_t(uint64_t(rows)*row_floats));
    // Upstream numbers windows from 0 in `rounds` and prints the position `p` the window started at (:13832).
    window_hashes_.record(int64_t(window_index_++),window_position_,rows,window_rows,window_tokens_,uint32_t(row_floats));
}

void Generation::emit_routing_window(const RoutingWindowView& view,bool window) {
    if(!dumps_) throw SessionFatal("routing window has no active writer");
    dumps_->routing_view(view,window);
    routing_written_+=uint64_t(view.layers)*view.rows;
}
GenerationResult Generation::run(const GenerationRequest& request,const GenerationEvents& events,const std::function<bool()>& cancelled) {
    if(request.prompt.empty() || request.prompt.size()>context_ || !request.max_new || request.max_new>context_-request.prompt.size() ||
       (!request.bias.empty() && request.bias.size()!=vocabulary_) || !events.token)
        throw std::invalid_argument("generation request exceeds context or lacks token sink");
    for(auto token:request.prompt) if(token<0 || uint32_t(token)>=vocabulary_) throw std::invalid_argument("generation token ID out of range");
    if(request.pin && uint64_t(request.pin)+1>=request.prompt.size()) throw std::invalid_argument("pinned prefix must end before the prompt's last token");
    // Parameter validation precedes reset, state changes and accelerator work.
    const std::array<float,1> dummy{0};(void)sampling_distribution(dummy,{},request.sampling);
    for(float b:request.bias) if(std::isnan(b) || b==INFINITY) throw std::invalid_argument("invalid generation bias");
    if(!request.bias.empty() && std::all_of(request.bias.begin(),request.bias.end(),[](float b){return b==-INFINITY;})) throw std::invalid_argument("all tokens are banned");
    if(request.max_new-1>UINT64_MAX-request.sampling.counter) throw std::overflow_error("sampling draw counter overflow");
    if(options_.verify_mode==VerifyMode::Probabilistic && request.sampling.counter+request.max_new-1>=(1ull<<61))
        throw std::invalid_argument("probabilistic verify counter overlaps reserved streams");
    // ---- upstream's per-position diagnostic dumps, armed before any state changes -------------------
    // Logits declares a full selected row count (:6077-6095). Raw layer/half
    // dumps and position-tagged final-r instead write ONLY the scalar token
    // loop (:13363-13384), after any batched prefill (:13265). Do not invent
    // records for skipped batched positions or force that computation scalar.
    // The park site reports through the caller's events for as long as this run owns the sequence.
    struct ParkEventsGuard {
        Generation* generation;
        const GenerationEvents* previous;
        ~ParkEventsGuard() {generation->park_events_=previous;}
    } park_guard{this,park_events_};
    park_events_=&events;
    std::optional<DumpSinks> sinks;routing_written_=0;
    struct DumpGuard {
        Generation* generation;
        ~DumpGuard() {generation->dumps_=nullptr;}
    } dump_guard{this};
    if(!options_.dumps.none()) {
        if(!options_.dumps.halves.empty() && !operations_.halves)
            throw std::invalid_argument("--dump-halves is CAPTURED into the layers; no complete half-frame producer is wired");
        // Only logits has a declared complete row-count contract. Raw files
        // retain upstream's token-loop suffix when prefill consumed a prefix.
        const bool per_position=options_.dumps.token_positions();
        if(!options_.dumps.logits.empty() && options_.prefill_chunk && request.prompt.size()>1)
            throw std::invalid_argument("the per-position dumps record the token path only: drop --prefill");
        // Drafts consume rows outside the per-position logits path. Any real draft SOURCE arms this refusal;
        // `max_draft` alone does not, because it defaults to the verification cap and drafts nothing when no
        // source is wired (the plain token path this backend runs by default).
        const bool may_draft = options_.max_draft != 0 &&
            (operations_.draft || spec_knobs_.oracle_active() || spec_knobs_.follow_active() ||
             spec_knobs_.corrupt_active() || options_.suffix_min_match != 0 || options_.lookup_chain != 0);
        if(per_position && may_draft)
            throw std::invalid_argument("the per-position dumps record the token path only: drop speculative drafts");
        // A resumed parked prefix would start the record set at a position > 0 while the header still
        // claims prompt_tokens-1+max_new rows, i.e. a file that describes a different file than the one
        // written. Rather than emit that, the combination is refused before any state changes.
        if(per_position && options_.cache_slots)
            throw std::invalid_argument("the per-position dumps record positions from 0, so they cannot run "
                                        "together with a prompt cache that may resume a parked prefix: drop --prompt-cache");
        if(!options_.dumps.routing.empty()) {
            if(options_.dump_shape.n_layers<1 || options_.dump_shape.n_layers>4096 || options_.dump_shape.n_expert<1 || options_.dump_shape.n_expert>4096)
                throw std::invalid_argument("--dump-routing geometry exceeds bounded layers/experts");
            if(!operations_.routing) throw std::invalid_argument("--dump-routing needs the token routing producer");
            if(options_.prefill_chunk && operations_.prefill && !operations_.prefill_routing)
                throw std::invalid_argument("--dump-routing with prefill needs a synchronous micro-window routing producer");
            if(may_draft && operations_.verify && !operations_.verify_routing)
                throw std::invalid_argument("--dump-routing with native verify needs its initial-pass routing producer");
        }
        auto geometry=options_.dump_shape;
        geometry.positions=int64_t(request.prompt.size())-1+int64_t(request.max_new);
        geometry.max_new=request.max_new;
        if(options_.resident_routing) {
            if(!options_.resident_routing->routing_session_matches(options_.dumps,geometry))
                throw std::invalid_argument("resident routing writer is closed or mismatched");
            dumps_=options_.resident_routing.get();
        } else {sinks.emplace(options_.dumps,geometry,options_.dump_producers);dumps_=&*sinks;}
        dump_positions_=geometry.positions;
    }
    // Upstream's elastic-K/V admission point: generate.cpp:13191/13227/13282 ask kvg_ensure for
    // n_prompt + max_new + 64 cells and a grow that cannot be paid for is refused at 12649 before any work
    // runs.  This backend commits the whole arena at plan time (no VMM range, no lendable expert-cache
    // slots - see strata/hpu/kv_residency.hpp's declared gaps), so the --kv byte cap is enforced in
    // plan_model and what is decided per request is whether the arena may grow at all.  With growth
    // disabled - upstream's default and the state --no-kv-grow selects - a request the committed cells
    // cannot hold is REFUSED here, before reset, before any pin is taken and before any parked state could
    // be evicted to pay for it.
    {
        KvArenaShape arena;
        arena.page_size=1;arena.capacity_cells=context_;arena.arrays=1;arena.per_page[0]=1;
        arena.granularity_bytes=options_.kv.granularity_bytes;
        const auto decision=kv_admit(arena,options_.kv_mapped_cells?options_.kv_mapped_cells:context_,0,
            kv_request_cells(uint64_t(request.prompt.size()),request.max_new,64),options_.kv);
        if(!decision.admits) {
            std::string reason=decision.refusal==KvRefusal::GrowthDisabled
                ?"--no-kv-grow keeps the K/V arena at "+std::to_string(decision.target_cells)+" cells"
                :decision.refusal==KvRefusal::CapExceeded
                    ?"the K/V arena would pass the --kv cap of "+std::to_string(options_.kv.cap_bytes)+" bytes"
                :"no room is left for the K/V to grow into";
            throw std::invalid_argument("the K/V cannot grow: "+reason+"; the request needs "+
                std::to_string(std::min<uint64_t>(context_,kv_request_cells(uint64_t(request.prompt.size()),request.max_new,64)))+" cells");
        }
    }
    // --vision's request law: install the M-RoPE table and the prompt's image rows before any prompt work.
    // A text request (no images) restores the identity table, exactly upstream's non-GENI branch. A GENI
    // request requires the owner's hook, since running it with the table but without the row override would
    // read the token embedding table where upstream reads the picture.
    if(request.embeddings) {
        if(!operations_.set_vision) throw std::invalid_argument("this engine was started without --vision");
        if(request.image_rows.size()!=request.prompt.size()) throw std::invalid_argument("image row list does not cover the prompt");
        if(request.positions.size()!=(uint64_t(context_)+64)*3) throw std::invalid_argument("image position table shape mismatch");
    }
    if(operations_.set_vision) operations_.set_vision(request);
    (void)flush_nonfinite();
    // Rebuild bounded lookup state from this request, including an adopted or
    // reused prompt. No stale proposal survives reset, branch or disk restore.
    std::optional<spec::SuffixDrafter> lookup;
    if(options_.max_draft && (options_.suffix_min_match || options_.lookup_chain)) {
        const auto minimum=options_.lookup_chain?(options_.suffix_min_match?std::min(options_.suffix_min_match,options_.lookup_chain_min):options_.lookup_chain_min):options_.suffix_min_match;
        lookup.emplace(int(minimum),64,context_);
        lookup->append(request.prompt.data(),request.prompt.size());
    }
    if(options_.lookup_chain) for(auto& source:spec::extra_draft_sources()) if(source) {source->reset();source->append(request.prompt.data(),request.prompt.size());}
    GenerationResult result;result.prompt_tokens=uint32_t(request.prompt.size());uint32_t read_reached=0;
    const auto prompt_start=Clock::now();
    try {
        if(cancelled && cancelled()) throw Cancelled();
        const bool want_control=operations_.control_vector?request.control_vector:true;
        if(want_control!=control_enabled_) {
            // Reset the active arithmetic state, retaining immutable snapshots
            // in the one bounded bank. Selection below requires the requested
            // mode, including identical token prefixes captured in both modes.
            try {reset_sequence();operations_.control_vector(want_control);control_enabled_=want_control;}
            catch(const std::exception& error) {invalidate();throw SessionFatal(std::string("control vector switch failed: ")+error.what());}
        }
        const bool wants_pin=options_.cache_slots && request.checkpoint && request.pin;
        const bool pin_exists=wants_pin && cache_.pin(std::span(request.prompt).first(request.pin),control_enabled_,
            images_below(request.images,request.pin));
        const uint64_t maximum=wants_pin && !pin_exists?request.pin:UINT64_MAX;
        // Upstream's live test is tokens AND pictures (starts_with(live, live_imgs), generate.cpp:10856-10861,
        // :10922): a request whose prompt cells carry other pictures cannot continue the live state even when
        // every token matches.
        const bool extends=valid_ && fed_.size()<=maximum && request.prompt.size()>=fed_.size() &&
            std::equal(fed_.begin(),fed_.end(),request.prompt.begin()) &&
            images_match(images_below(sequence_images_,fed_.size()),request.images,fed_.size());
        const auto cached=cache_.find(request.prompt,maximum,control_enabled_,request.images);
        auto external=options_.cache_slots && operations_.select_prefix?
            operations_.select_prefix(request.prompt,std::min<uint64_t>(maximum,request.prompt.size()-1),want_control,
                                      request.images):std::optional<GenerationPrefix>{};
        if(external && (external->tokens.empty() || external->tokens.size()>maximum || external->tokens.size()>=request.prompt.size() ||
            !external->restore || !external->logits || !std::equal(external->tokens.begin(),external->tokens.end(),request.prompt.begin()) ||
            !images_match(external->images,request.images,external->tokens.size())))
            throw SessionFatal("external prefix source contract mismatch");
        const auto best=std::max(extends?fed_.size():0,cached?cache_.entry(*cached).tokens.size():0);
        report_picture_mismatch(request.prompt,request.images,best,want_control);
        bool external_from0=false;
        if(external && external->tokens.size()>best) {
            try {
                external->restore();auto values=external->logits();
                if(values.size()!=vocabulary_ || !std::all_of(values.begin(),values.end(),[](float x){return std::isfinite(x);})) throw std::logic_error("invalid external prefix logits");
                fed_.assign(external->tokens.begin(),external->tokens.end());logits_=std::move(values);valid_=true;
                suspended_live_={};model_attached_=true;
                GenerationRead chain;chain.tokens=fed_;chain.control_vector=control_enabled_;chain.from0=external->from0;
                chain.checkpoints=external->checkpoints;chain.restore=external->restore;chain.logits=external->logits;
                chain.images.assign(request.images.begin(),request.images.end());
                import_prefixes(chain);external_from0=external->from0;
            } catch(const std::exception& error) {invalidate();throw SessionFatal(std::string("external prefix restore failed: ")+error.what());}
        } else if(cached && (!extends || cache_.entry(*cached).tokens.size()>fed_.size())) {
            // Upstream SERVES a parked conversation out of the cache: take() moves it out
            // (include/strata/core/conversation_cache.hpp:257-262), the model restores from that image and
            // retains it (retain(), :217-226), and every caller-side reference dies at the end of this block.
            // That is what lets the park at the end of the request GROW the image instead of re-copying the
            // whole K/V (src/core/conversation_snapshot.cpp:147-181); the re-park puts the conversation back.
            // Taking the conversation out is only what upstream can afford when the run will park it back: a
            // one-shot request (--ckpt 0) leaves its parked copy where it is, because a request that promises no
            // checkpoint must not cost the cache a conversation another request can still come back to.
            const bool running_out=request.checkpoint && cached && cache_.entry(*cached).tokens.size()<request.prompt.size();
            std::optional<PrefixCache::Entry> taken;
            if(running_out) taken=cache_.take(*cached);
            const auto& parked_entry=taken?*taken:cache_.entry(*cached);
            const auto restore=parked_entry.restore;
            try {restore();}
            catch(const std::exception& error) {invalidate();throw SessionFatal(std::string("cached state restore failed: ")+error.what());}
            fed_.assign(parked_entry.tokens.begin(),parked_entry.tokens.end());
            logits_.assign(parked_entry.logits.begin(),parked_entry.logits.end());valid_=true;
            suspended_live_={};model_attached_=true;
        } else if(!extends) reset_sequence();
        else if(!model_attached_) {
            if(!suspended_live_) reset_sequence();
            else {
                try {suspended_live_();suspended_live_={};model_attached_=true;}
                catch(const std::exception& error) {invalidate();throw SessionFatal(std::string("suspended live restore failed: ")+error.what());}
            }
        }
        external.reset(); // no borrowed image alias survives into eviction/cooperative windows
        // This run's sequence now owns whatever pictures the request carries; captures at any depth take the
        // sub-list below that depth (upstream re-derives imgs_below(req_imgs, L) at every checkpoint, :8449).
        sequence_images_.assign(request.images.begin(),request.images.end());
        result.reused=uint32_t(fed_.size());read_reached=result.reused;if(events.reused) events.reused(result.reused);
        // Upstream's `resumed_from0` (src/program/generate.cpp:11044, gate :11727): a prompt whose read was
        // interrupted and re-issued continues the SAME read that started at token 0, so the shared root is still
        // looked for even though tokens are already fed. The marker is transitive the same way upstream's is
        // (:11619 stores `read_from == 0 || resumed_from0` on the yielding slot).
        // The marker is keyed by what the interrupted read had reached: an entry matches when the tokens already
        // fed are exactly that prefix and this prompt continues it. A different conversation's entry is left
        // alone, which is what the per-slot table buys upstream; the table is bounded by --prompt-cache, and a
        // dropped entry only costs the root election, never correctness.
        const bool consumed = [&] {
            for(size_t i=0;i<interrupted_reads_.size();++i) {
                const auto& reached=interrupted_reads_[i].tokens;
                if(reached.size()!=fed_.size() || reached.size()>request.prompt.size() ||
                   !std::equal(reached.begin(),reached.end(),fed_.begin()) ||
                   !std::equal(reached.begin(),reached.end(),request.prompt.begin()) ||
                   !images_match(interrupted_reads_[i].images,request.images,reached.size())) continue;
                const bool from0=interrupted_reads_[i].from0;
                interrupted_reads_.erase(interrupted_reads_.begin()+std::ptrdiff_t(i));
                return from0;
            }
            return false;
        }();
        struct PartialReadGuard {
            using Reads=std::vector<InterruptedRead>;
            const std::vector<int32_t>* fed;size_t prompt;bool started;Reads* reads;size_t slots;
            const std::vector<RequestImage>* images;
            ~PartialReadGuard() {
                if(fed->size()>=prompt || fed->empty()) return;         // the read finished, or never started
                if(reads->size()<slots) try {
                    std::vector<int32_t> tokens(fed->size());
                    const auto below=images_below(*images,fed->size());
                    std::vector<RequestImage> pictures(below.size());
                    if(tokens.capacity()!=fed->size() || pictures.capacity()!=below.size()) return; // optional marker, never escape its admitted extent
                    std::copy(fed->begin(),fed->end(),tokens.begin());std::copy(below.begin(),below.end(),pictures.begin());
                    reads->push_back({std::move(tokens),std::move(pictures),started});
                } catch(const std::bad_alloc&) {} // optional bookkeeping must not terminate during cancellation
            }
        } partial_read{&fed_,request.prompt.size(),result.reused==0 || consumed || external_from0,&interrupted_reads_,
                       std::max<size_t>(options_.cache_slots,1),&sequence_images_};
        const auto boundaries=options_.cache_slots?
            prompt_boundaries(request.prompt,result.reused,options_.boundaries,request.checkpoint,consumed || external_from0):PromptBoundaries{};
        if(wants_pin && fed_.size()==request.pin) (void)cache_current(true);
        const bool windowed=options_.prefill_chunk && request.prompt.size()-fed_.size()>options_.short_read;
        const size_t batched_end=prefill_batched_end(request.prompt.size(),options_.prefill_until,
                                                   dumps_ && options_.dumps.token_positions());
        PromptCheckpointSchedule schedule(uint32_t(fed_.size()),uint32_t(request.prompt.size()),options_.cache_every,
            windowed && options_.cache_tail,options_.short_read);
        for(size_t i=fed_.size();i<request.prompt.size();) {
            const bool use_window=windowed && i<batched_end && request.prompt.size()-i>options_.short_read;
            size_t end=std::min(request.prompt.size(),i+(use_window?options_.prefill_chunk:1));
            if(use_window && options_.canonical_prefill_rows) end=std::min(end,i+canonical_prefill_count(uint32_t(request.prompt.size()-i),options_.prefill_chunk,options_.short_read,options_.canonical_prefill_rows));
            if(use_window && end>batched_end) end=batched_end; // the flag decides where batching stops
            // Explicit root/turn/message/pin stops are semantic boundaries.
            // Periodic and near-tail candidates never split an existing chunk.
            for(uint32_t stop:{wants_pin?request.pin:0,boundaries.root,boundaries.turn,boundaries.message})
                if(stop>i && stop<end) end=stop;
            const auto count=uint32_t(end-i);
            if(use_window && (operations_.prefill || operations_.prefill_routing)) {
                uint32_t committed=0;
                PromptCommit commit=[&](uint32_t done,std::span<const float> values) {
                    const bool good=valid_ && done>committed && done<=count && values.size()==vocabulary_;
                    if(!good) {invalidate();throw SessionFatal("prefill callback violated committed-prefix/logit contract");}
                    check_logits_row(options_,values,int64_t(read_reached)-1,vocabulary_);
            if(!all_finite(values)) throw NonFiniteOutput("non-finite prefill logits");
                    fed_.insert(fed_.end(),request.prompt.begin()+i+committed,request.prompt.begin()+i+done);
                    logits_.assign(values.begin(),values.end());committed=done;read_reached=uint32_t(i+done);
                    if(done<count && events.progress) events.progress(uint32_t(i+done),result.prompt_tokens,elapsed(prompt_start));
                };
                if(dumps_ && dumps_->wants_routing()) {
                    uint32_t traced=0;
                    operations_.prefill_routing(std::span(request.prompt).subspan(i,count),commit,cancelled,[&](const RoutingWindowView& view,bool window) {
                        if(view.rows>committed-traced) throw SessionFatal("prefill routing preceded its committed prefix");
                        emit_routing_window(view,window);traced+=view.rows;
                    });
                    if(traced!=count) throw SessionFatal("prefill routing omitted a committed micro-window");
                } else if(operations_.prefill) {
                    // The following chunk, when it will also be a window, as a prefetch hint.
                    if(operations_.prefill_next && end<batched_end && request.prompt.size()-end>options_.short_read)
                        operations_.prefill_next(std::span(request.prompt).subspan(end,std::min<size_t>(options_.prefill_chunk,batched_end-end)));
                    operations_.prefill(std::span(request.prompt).subspan(i,count),commit,cancelled);
                }
                else operations_.prefill_routing(std::span(request.prompt).subspan(i,count),commit,cancelled,{});
                if(committed!=count) {invalidate();throw SessionFatal("prefill returned without committing its complete window");}
            } else for(size_t at=i;at<end;++at) {
                feed(request.prompt[at],cancelled);
                read_reached=uint32_t(at+1);
                if(at+1<end && events.progress) events.progress(uint32_t(at+1),result.prompt_tokens,elapsed(prompt_start));
            }
            const auto flags=schedule.complete(uint32_t(end),count);
            if(wants_pin && fed_.size()==request.pin) (void)cache_current(true);
            if(fed_.size()==boundaries.root) (void)cache_current(false,true);
            else if(fed_.size()==boundaries.turn) (void)cache_current();
            else if(fed_.size()==boundaries.message) (void)cache_current(false,false,true);
            // With --prefill-until the remaining prompt deliberately runs through the token path, so the
            // periodic and tail checkpoints the schedule asks for must still be taken there.
            if(request.checkpoint && (!options_.prefill_chunk || use_window || options_.prefill_until) && (flags.periodic || flags.tail))
                (void)cache_current(false,false,flags.tail && !flags.periodic);
            if(events.progress) events.progress(uint32_t(end),result.prompt_tokens,elapsed(prompt_start));
            size_t segment=request.prompt.size()-1;
            for(uint32_t cut:{wants_pin?request.pin:0,boundaries.root,boundaries.turn,boundaries.message}) if(cut>i && cut<segment) segment=cut;
            if(events.yield && options_.cache_slots && end<segment) {
                GenerationRead read;read.tokens=fed_;read.control_vector=control_enabled_;read.from0=!result.reused || consumed || external_from0;
                read.can_yield=segment-end>std::max(std::max(1u,options_.prefill_chunk),options_.short_read);
                read.images.assign(images_below(sequence_images_,fed_.size()).begin(),
                                   images_below(sequence_images_,fed_.size()).end());
                read.checkpoints=[&](const GenerationCheckpointVisitor& visit) {
                    cache_.visit_prefixes(fed_,control_enabled_,[&](const PrefixCache::Entry& point) {
                        GenerationCheckpoint checkpoint{point.tokens,point.restore,point.age,point.pinned,point.root,point.tail};
                        checkpoint.images=point.images;visit(checkpoint);
                    },images_below(sequence_images_,fed_.size()));
                };
                const int target=events.yield(read);
                if(target>=0) {
                    if(!read.can_yield) throw SessionFatal("yield consumer accepted an ineligible prompt piece");
                    result.yielded_slot=target;result.yielded_tokens=uint32_t(fed_.size());
                    cache_.release_prefixes(fed_,control_enabled_,images_below(sequence_images_,fed_.size()));valid_=false;throw Cancelled();
                }
            }
            if(end<request.prompt.size()) cooperate(events);
            i=end;
        }
        if(request.checkpoint) (void)cache_current();
        result.prompt_ms=elapsed(prompt_start);const auto decode_start=Clock::now();
        // Upstream --spec-follow caps the run at the continuation's length (src/program/generate.cpp:13530);
        // with no follow set this is exactly request.max_new.
        const uint32_t max_new=uint32_t(spec_knobs_.follow_max_new(request.max_new));
        bool first_policy_window=options_.adaptive_drafts;
        const bool fixed_chain=std::getenv("STRATA_LOOKUP_CHAIN_FIXED")!=nullptr;
        auto policy_now=[&]() noexcept {
            try {return operations_.policy_clock_ms?operations_.policy_clock_ms():
                std::chrono::duration<double,std::milli>(Clock::now().time_since_epoch()).count();}
            catch(...) {return std::numeric_limits<double>::quiet_NaN();}
        };
        // A residual the target drew at a rejected DFlash row. It already IS the next token (rejection sampling
        // replaced the draft with it), so the next window takes it as its pending instead of drawing again from p.
        int32_t dflash_residual=-1;
        try {
            while(result.generated<max_new) {
                WindowRound window_round{this};   // opens this window; closes it on every exit
                if(cancelled && cancelled()) throw Cancelled();
                // STRATA_HPU_DFLASH_VERIFY: a request commits the pending token and the drafts the target's own
                // picks agree with in one model window, instead of feeding the pending token alone. The pick law is
                // verify_pick() row by row, exactly the plain loop's; a pending stop token and the last token of the
                // request take the plain path. Under --spec-prob a sampled request drafts point masses (the drafter's
                // greedy walk), so the verifier accepts a row with p(d) and draws the residual p-d on a reject: the
                // emitted tokens keep the request's own sampling chain.
                const bool dflash_greedy=request.sampling.greedy || request.sampling.temperature<=0;
                const bool dflash_sampled=!dflash_greedy && options_.verify_mode==VerifyMode::Probabilistic &&
                                          request.sampling.temperature>0.0f;
                if(operations_.dflash && (dflash_greedy||dflash_sampled) && !dumps_ && !window_hashes_.armed() &&
                   !spec_knobs_.oracle_active() && !spec_knobs_.follow_active() && !spec_knobs_.corrupt_active() &&
                   max_new-result.generated>=2) {
                    auto parameters=request.sampling;parameters.counter+=result.generated;
                    const auto pending=dflash_residual>=0?dflash_residual:
                        verify_pick(logits_,fed_,parameters,request.bias,nullptr,options_.verify_mode).token;
                    const auto stop=[&](int32_t token) {return options_.stop_eos && std::find(stop_ids_.begin(),stop_ids_.end(),token)!=stop_ids_.end();};
                    if(!stop(pending)) {
                        // Without bias or penalties the greedy pick is the argmax, which the device already found.
                        // A sampled row always needs its logits: the row's draw is not an argmax.
                        const auto& sp=request.sampling;
                        const bool plain=dflash_greedy && request.bias.empty() && !sp.logit_bias &&
                                         (sp.penalty_last_n==0 || (sp.penalty_repeat==1.0f && sp.penalty_freq==0.0f && sp.penalty_present==0.0f));
                        uint32_t selected=0;int32_t rejected=-1;
                        auto round=operations_.dflash(pending,max_new-result.generated,[&](const DflashRows& window) {
                            // Keep row k while the target's pick after rows 0..k-1 is the drafted token; nothing after a stop.
                            const auto ids=window.ids;std::vector<int32_t> seen;uint32_t keep=1;
                            if(!plain) {seen=fed_;seen.push_back(ids[0]);}
                            while(keep<ids.size() && !stop(ids[keep-1])) {
                                int32_t pick=window.argmax[keep-1];
                                if(!plain) {
                                    auto row_parameters=request.sampling;row_parameters.counter+=result.generated+keep;
                                    // The draft is a point mass (no q list): p(d) is the accept probability and a
                                    // reject yields the residual draw, the row's replacement token.
                                    const DraftRow point{ids[keep],{},{}};
                                    const auto verified=verify_pick(window.row(keep-1),seen,row_parameters,request.bias,
                                                                    dflash_sampled?&point:nullptr,options_.verify_mode);
                                    pick=verified.token;
                                    if(dflash_sampled && !verified.accepted) rejected=pick;
                                }
                                if(pick!=ids[keep]) break;
                                if(!plain) seen.push_back(ids[keep]);
                                ++keep;
                            }
                            selected=keep;return keep;
                        },cancelled);
                        if(!round.tokens.empty()) {
                            if(round.logits.size()!=vocabulary_ || !all_finite(round.logits)) {invalidate();throw SessionFatal("DFlash window returned invalid logits");}
                            check_logits_row(options_,round.logits,int64_t(fed_.size()+round.tokens.size()-1),vocabulary_);
                            ++result.verify_windows;result.drafts_proposed+=round.proposed;result.drafts_checked+=round.proposed;
                            result.drafts_accepted+=uint32_t(round.tokens.size()-1);
                            // Only a rejection the round committed up to leaves a token owed to the next window.
                            dflash_residual=(rejected>=0 && round.tokens.size()==selected)?rejected:-1;
                            bool stopped=false;
                            for(const auto token:round.tokens) {
                                fed_.push_back(token);events.token(token);++result.generated;
                                if(window_open_) window_tokens_.push_back(token);
                                if(lookup) lookup->append(token);
                                if(options_.lookup_chain) for(auto& source:spec::extra_draft_sources()) if(source) source->append(&token,1);
                                if(stop(token)) {stopped=true;break;}
                            }
                            logits_=std::move(round.logits);
                            if(stopped) {result.finish="stop";break;}
                            if(result.generated<max_new) cooperate(events);
                            continue;
                        }
                    }
                }
                if(dflash_residual>=0) {
                    // The gate did not take the carried residual (a stop token, a backoff skip or no draft rows):
                    // publish it and feed it like any other committed token, then sample the next one afresh.
                    const auto token=dflash_residual;dflash_residual=-1;
                    events.token(token);++result.generated;
                    if(window_open_) window_tokens_.push_back(token);
                    if(lookup) lookup->append(token);
                    if(options_.lookup_chain) for(auto& source:spec::extra_draft_sources()) if(source) source->append(&token,1);
                    if(options_.stop_eos && std::find(stop_ids_.begin(),stop_ids_.end(),token)!=stop_ids_.end()) {result.finish="stop";break;}
                    if(result.generated<max_new) feed(token,cancelled);
                    if(result.generated<max_new) cooperate(events);
                    continue;
                }
                const bool timed_policy=options_.adaptive_drafts && !first_policy_window;
                const double policy_start=timed_policy?policy_now():0;
                const auto accepted_before=result.drafts_accepted;
                // As upstream's first_window, prime the first output before
                // selecting/learning a speculative window. Fixed mode is unchanged.
                const auto maximum=first_policy_window?0u:std::min(options_.max_draft,max_new-result.generated-1);
                // Upstream's gate for THIS round: the request's spec_min_p when it named one, else the start-up
                // --spec-min-p (src/program/generate.cpp:10665 seeds the request value from the option).
                const float spec_gate_value=request.spec_min_p>=0.0f?request.spec_min_p:options_.spec_min_p;
                std::array<float,7> confidence;
                if(options_.adaptive_drafts || spec_gate_armed(spec_gate_value)) confidence.fill(std::numeric_limits<float>::quiet_NaN());
                bool from_lookup=false;int lookup_match=0,chain_match=0;uint32_t chain_count=0;
                std::vector<DraftRow> drafts;
                if(maximum && spec_knobs_.oracle_active()) {
                    // The oracle stands in for the drafter's own proposals (upstream's no-MTP/no-suffix
                    // branch of src/program/generate.cpp:13800-13801; the port replaces the source
                    // wholesale, declared).  Point-mass rows: the accept/reject math is untouched.
                    for(const auto token:spec_knobs_.oracle_window(result.generated,maximum)) drafts.push_back({token,{},{}});
                } else if(maximum && operations_.draft) {
                    auto parameters=request.sampling;parameters.counter+=result.generated;
                    DraftContext context{fed_,logits_,request.bias,parameters,options_.verify_mode};
                    // Upstream truncates the window on the drafter's OWN per-position probabilities (:12566-12568
                    // walks dprob, which the MTP run fills), so an armed gate has to ask the drafter for them exactly
                    // as --adaptive-drafts does. A row the drafter did not answer for cannot join: its confidence
                    // stays NaN and fails the comparison, which is the conservative reading of upstream's `avail`
                    // guard at :12041-12043. The oracle and --spec-corrupt diagnostics are the port's own sources and
                    // carry no drafter confidence, so they are not gated - declared, not silently skipped.
                    if(options_.adaptive_drafts || spec_gate_armed(spec_gate_value)) context.confidence=std::span(confidence).first(maximum);
                    drafts=operations_.draft(context,maximum,cancelled);
                    if(spec_gate_armed(spec_gate_value))
                        drafts.resize(gated_draft_count(std::span(confidence).first(maximum),uint32_t(drafts.size()),spec_gate_value));
                }
                validate_drafts(drafts,vocabulary_,maximum);
                auto external_count=drafts.size();
                if(options_.adaptive_drafts && maximum && lookup && options_.suffix_min_match) {
                    std::array<int32_t,7> ids{};auto count=lookup->propose(int(maximum),ids.data());
                    lookup_match=lookup->last_match();
                    if(lookup_match<int(options_.suffix_min_match)) count=0;
                    const bool mtp=operations_.draft && !spec_knobs_.oracle_active();
                    if(count>0 && (!mtp || (!drafts.empty() && ids[0]==drafts[0].token))) {
                        const auto pick=draft_policy_.choose(int(external_count+1),count,lookup_match);
                        if(pick.lookup) {
                            drafts.clear();for(int i=0;i<pick.t-1;++i) drafts.push_back({ids[size_t(i)],{},{}});
                            external_count=0;from_lookup=true;
                        }
                    }
                }
                if(external_count && external_count<maximum && lookup && options_.lookup_chain) {
                    std::vector<int32_t> tail;tail.reserve(64);
                    const auto before=std::min<size_t>(fed_.size(),64-external_count);
                    tail.insert(tail.end(),fed_.end()-before,fed_.end());for(const auto& row:drafts) tail.push_back(row.token);
                    std::array<int32_t,7> ids{};int match=0,source=-1;spec::PromptLookupSource own(*lookup);
                    auto count=spec::propose_from_sources(own,tail.data(),int(tail.size()),int(external_count),
                        int(std::min<uint32_t>(options_.lookup_chain,maximum-uint32_t(external_count))),int(options_.lookup_chain_min),ids.data(),&match,&source);
                    if(options_.adaptive_drafts && count>0 && !fixed_chain) {
                        double p=1;
                        for(size_t i=0;i<external_count;++i) {
                            float value=confidence[i];
                            if(std::isnan(value)) {
                                value=1; // point-mass external/oracle proposal
                                if(!drafts[i].ids.empty()) {
                                    const auto at=std::find(drafts[i].ids.begin(),drafts[i].ids.end(),drafts[i].token);
                                    value=drafts[i].probabilities[size_t(at-drafts[i].ids.begin())];
                                }
                            }
                            if(!std::isfinite(value) || value<0 || value>1) throw std::invalid_argument("invalid adaptive draft confidence");
                            p*=value;
                        }
                        count=draft_policy_.chain(int(external_count+1),p,count,match);
                    }
                    chain_count=uint32_t(count);chain_match=match;
                    for(int i=0;i<count;++i) drafts.push_back({ids[size_t(i)],{}, {}});
                    result.chain_proposed+=uint32_t(count);validate_drafts(drafts,vocabulary_,maximum);
                }
                if(!options_.adaptive_drafts && drafts.empty() && maximum && lookup && options_.suffix_min_match) {
                    std::array<int32_t,7> ids{};auto count=lookup->propose(int(maximum),ids.data());
                    if(lookup->last_match()<int(options_.suffix_min_match)) count=0;
                    for(int i=0;i<count;++i) drafts.push_back({ids[size_t(i)],{}, {}});
                }
                if(spec_knobs_.corrupt_active())
                    for(auto& row:drafts) if(spec_knobs_.corrupt_slot(row.token)) {
                        // Upstream's exact flip is (d+1) % n_vocab (src/program/generate.cpp:13803).  A
                        // corrupted row's drafter distribution no longer contains the flipped token, so
                        // keeping it would trip validate_drafts (verify.cpp:20); clearing it makes the
                        // row a point mass - the port's counterpart of upstream dropping the drafter's
                        // spec_q coupling whenever corruption is armed (13814: `&& o.spec_corrupt <= 0`).
                        row.ids.clear();row.probabilities.clear();
                    }
                ++result.verify_windows;result.drafts_proposed+=uint32_t(drafts.size());
                auto observe_policy=[&] {
                    first_policy_window=false;
                    if(!timed_policy || result.finish=="stop") return;
                    const double duration=policy_now()-policy_start;
                    const double ms=std::isfinite(duration) && duration>0?duration:0;
                    const auto accepted=int(result.drafts_accepted-accepted_before);
                    if(chain_count) draft_policy_.observe_chain(int(external_count+1),int(chain_count),accepted,chain_match,ms);
                    else draft_policy_.observe(from_lookup,int(drafts.size()+1),accepted,lookup_match,ms);
                };
                if(!drafts.empty() && operations_.verify) {
                    auto parameters=request.sampling;parameters.counter+=result.generated;
                    DraftContext context{fed_,logits_,request.bias,parameters,options_.verify_mode};
                    if(spec_knobs_.follow_active()) context.follow=std::span<const int32_t>(spec_knobs_.follow_tokens()).subspan(result.generated,drafts.size()+1);
                    uint32_t traced=0;
                    if(dumps_ && dumps_->wants_routing()) context.routing=[&](const RoutingWindowView& view,bool window) {
                        if(traced || !window || view.rows!=drafts.size()) throw SessionFatal("native routing omitted/duplicated physical candidate rows");
                        emit_routing_window(view,true);traced=view.rows;
                    };
                    auto window=operations_.verify(context,drafts,cancelled);
                    if(context.routing && traced!=drafts.size()) {invalidate();throw SessionFatal("native verify did not publish its routing window");}
                    // The backend already committed state. Malformed results
                    // are fatal; never reuse a potentially divergent prefix.
                    const bool follow_contract=context.follow.empty()
                        ?!window.follow_differ && !window.follow_emitted
                        :window.follow_emitted==window.picks.size() && window.follow_differ<=window.follow_emitted &&
                         window.picks.size()<=context.follow.size() && std::equal(window.picks.begin(),window.picks.end(),context.follow.begin(),
                             [](const VerifyPick& pick,int32_t token){return pick.token==token;});
                    if(!follow_contract || !valid_verified_candidates(window,drafts,vocabulary_,max_new-result.generated,stop_ids_)) {
                        invalidate();throw SessionFatal("native verify hook violated accepted-prefix/logit contract");
                    }
                    for(uint32_t row=0;row<window.retained;++row) fed_.push_back(drafts[row].token);
                    logits_=std::move(window.logits);
                    result.follow_differ+=window.follow_differ;result.follow_emitted+=window.follow_emitted;
                    // One <=8-token atomic publication after native commit.
                    // Cancellation before commit publishes nothing; after it,
                    // all selected tokens are streamed before the next poll.
                    for(size_t row=0;row<window.picks.size();++row) {
                        const auto pick=window.picks[row];
                        if(row<drafts.size()) {++result.drafts_checked;if(pick.accepted) {++result.drafts_accepted;if(external_count && row>=external_count) ++result.chain_accepted;}}
                        events.token(pick.token);++result.generated;
                        if(window_open_) window_tokens_.push_back(pick.token);
                        if(lookup) lookup->append(pick.token);
                        if(options_.lookup_chain) for(auto& source:spec::extra_draft_sources()) if(source) source->append(&pick.token,1);
                    }
                    if(options_.stop_eos && window.stopped) {result.finish="stop";break;}
                    if(result.generated<max_new) feed(window.picks.back().token,cancelled);
                    observe_policy();
                    if(result.generated<max_new) cooperate(events);
                    continue;
                }
                if(!drafts.empty()) {
                    const auto window=scalar_candidates(drafts,request,result.generated,cancelled);
                    result.follow_differ+=window.follow_differ;result.follow_emitted+=window.follow_emitted;
                    // Once committed, publish the entire selected window with
                    // no cancellation poll between its tokens (native parity).
                    for(uint32_t row=0;row<window.count;++row) {
                        const auto pick=window.picks[row];
                        if(row<drafts.size()) {++result.drafts_checked;if(pick.accepted) {++result.drafts_accepted;if(external_count && row>=external_count) ++result.chain_accepted;}}
                        events.token(pick.token);++result.generated;
                        if(window_open_) window_tokens_.push_back(pick.token);
                        if(lookup) lookup->append(pick.token);
                        if(options_.lookup_chain) for(auto& source:spec::extra_draft_sources()) if(source) source->append(&pick.token,1);
                    }
                    if(options_.stop_eos && window.stopped) {result.finish="stop";break;}
                    if(result.generated<max_new) feed(window.picks[window.count-1].token,cancelled);
                    observe_policy();
                    if(result.generated<max_new) cooperate(events);
                    continue;
                }
                for(size_t row=0;row<=drafts.size() && result.generated<max_new;++row) {
                    if(cancelled && cancelled()) throw Cancelled();
                    auto parameters=request.sampling;parameters.counter+=result.generated;
                    const auto* draft=row<drafts.size()?&drafts[row]:nullptr;
                    const auto pick=verify_pick(logits_,fed_,parameters,request.bias,draft,options_.verify_mode);
                    // Upstream --spec-follow (src/program/generate.cpp:13836-13842,13882,13844): the
                    // continuation stands in for the window's row, acceptance then compares the draft
                    // with the replaced row, and every emitted token counts its difference.  result.generated
                    // here is the run-relative index upstream calls k0+i.  With no follow this is exactly
                    // the old pick.
                    int32_t emitted_token=pick.token;bool accepted=pick.accepted;
                    if(spec_knobs_.follow_active()) {
                        const auto applied=spec_knobs_.follow_row(result.generated,pick.token);
                        emitted_token=applied.token;
                        if(draft) accepted=emitted_token==draft->token;
                        result.follow_differ+=applied.differs;++result.follow_emitted;
                    }
                    if(draft) {++result.drafts_checked;if(accepted) {++result.drafts_accepted;if(external_count && row>=external_count) ++result.chain_accepted;}}
                    events.token(emitted_token);++result.generated;
                    if(window_open_) window_tokens_.push_back(emitted_token);
                    if(lookup) lookup->append(emitted_token);
                    if(options_.lookup_chain) for(auto& source:spec::extra_draft_sources()) if(source) source->append(&emitted_token,1);
                    if(options_.stop_eos && std::find(stop_ids_.begin(),stop_ids_.end(),emitted_token)!=stop_ids_.end()) {result.finish="stop";break;}
                    if(result.generated<max_new) feed(emitted_token,cancelled);
                    if(draft && !accepted) break;
                }
                if(result.finish=="stop") break;
                observe_policy();
                if(result.generated<max_new) cooperate(events);
            }
        } catch(const NonFiniteOutput&) {result.finish="nonfinite";}
        catch(const Cancelled&) {result.finish="cancelled";}
        result.decode_ms=elapsed(decode_start);
    } catch(const NonFiniteOutput&) {result.prompt_ms=elapsed(prompt_start);result.finish="nonfinite";}
    catch(const Cancelled&) {result.prompt_ms=elapsed(prompt_start);result.finish="cancelled";}
    result.prompt_read=read_reached>=result.reused?read_reached-result.reused:0;
    if(result.finish=="nonfinite") {mark_nonfinite();valid_=false;}
    else if(request.checkpoint && result.yielded_slot<0) (void)cache_current();
    else valid_=false; // ckpt=0 neither parks nor keeps the live turn reusable
    if(dumps_) result.dumped_routing_records=routing_written_;
    if(sinks) {
        // The two ONE-SHOT records are written at the END of the run from the value in hand
        // (src/program/generate.cpp:14034-14078): --dump-mixed is the head input, --dump-residual the
        // final HC residual. A run whose producer is absent is refused rather than writing zeros.
        if(sinks->wants_mixed()) {
            if(!operations_.mixed) throw std::runtime_error("--dump-mixed needs the head-input producer");
            sinks->mixed(operations_.mixed());
        }
        if(sinks->wants_residual()) {
            if(!operations_.residual) throw std::runtime_error("--dump-residual needs the final-residual producer");
            sinks->residual(operations_.residual());
        }
        result.dumped_logits_rows=sinks->logits_rows_written();
        result.dumped_final_r_records=sinks->final_r_records();
        result.dumped_half_records=sinks->half_records();
        result.dumped_routing_records=sinks->routing_records();
        // Upstream checks this close and reports "cannot finish logits dump" (:13997-14000).
        if(!sinks->close()) throw std::runtime_error(sinks->close_error());
    }
    return result;
}
}
