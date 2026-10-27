#include "strata/hpu/protocol.hpp"
#include "strata/platform/aux_cpus.hpp"
#include "strata/hpu/batch_slots.hpp"
#include "strata/hpu/protocol_info.hpp"
#include "strata/hpu/qsa_phase_stats.hpp"
#include "strata/hpu/tensor.hpp"
#include "strata/hpu/state_io.hpp"
#include "strata/hpu/allocation_budget.hpp"
#include "strata/core/logit_bias.hpp"
#include "strata/kernels/verify_kernels.hpp"
#include <atomic>
#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <charconv>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <deque>
#include <memory>
#include <mutex>
#include <set>
#include <sstream>
#include <thread>
#include <poll.h>
#include <iostream>
#include <unistd.h>

namespace strata::hpu {
namespace {
template<class T> T number(const std::string& text) {
    T value{};const auto [end,error]=std::from_chars(text.data(),text.data()+text.size(),value);
    if(error!=std::errc{} || end!=text.data()+text.size()) throw std::invalid_argument("invalid request number: "+text);
    return value;
}
std::string one_line(std::string value) {for(auto& c:value) if(c=='\r' || c=='\n') c=' ';return value;}
// The same result/error grammar for both scheduling modes. SAVE first mounts
// the explicit main owner; RESTORE validates the file before replacing state
// in the session callback. A failed mount is fatal because transfer may have
// touched native state. No slot is inferred from the last Model operation.
enum class SessionOutcome {Rejected,Saved,Restored,Fatal};
SessionOutcome session_command(const std::string& line,ProtocolControl& control,Generation* main,
                     const std::function<void(const std::string&)>& emit) {
    const bool save=line.starts_with("SAVE ");const auto path=line.substr(save?5:8);
    const auto started=std::chrono::steady_clock::now();
    try {
        if(path.empty() || path.find_first_of("\r\n")!=std::string::npos || path.find('\0')!=std::string::npos || !control.session)
            throw std::invalid_argument("invalid session path or disk sessions unavailable");
        emit(std::string("SWAIT ")+(save?"saving":"restoring")+" 300");
        if(save && main) main->attach_live();
        const auto result=control.session(save,path);
        const auto ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();
        emit(std::string(save?"SAVED ":"RESTORED ")+std::to_string(result.tokens)+" "+std::to_string(result.bytes)+" "+std::to_string(ms));
        return save?SessionOutcome::Saved:SessionOutcome::Restored;
    } catch(const SessionFatal& error) {emit("FATAL "+one_line(error.what()));return SessionOutcome::Fatal;}
    catch(const StateIoError& error) {emit(std::string("SERR ")+(error.storage?"storage":"io")+" "+(error.published?"1 ":"0 ")+one_line(error.what()));}
    catch(const AllocationBudgetExceeded& error) {emit("SERR memory 0 "+one_line(error.what()));}
    catch(const std::bad_alloc&) {emit("SERR memory 0 snapshot allocation failed");}
    catch(const std::invalid_argument& error) {emit("SERR invalid 0 "+one_line(error.what()));}
    catch(const std::logic_error& error) {emit("SERR invalid 0 "+one_line(error.what()));}
    catch(const std::exception& error) {emit("SERR io 0 "+one_line(error.what()));}
    return SessionOutcome::Rejected;
}
}
void serve_protocol(int input_fd,std::ostream& output,Generation& generator,uint32_t vocabulary,uint32_t context,ProtocolControl control) {
    if(input_fd<0) throw std::invalid_argument("invalid protocol input file descriptor");
    struct Pending {std::string line;std::shared_ptr<std::atomic<bool>> cancelled;};
    std::mutex mutex,out_mutex;std::condition_variable ready;std::deque<Pending> queue;
    bool closed=false;std::shared_ptr<std::atomic<bool>> active;
    auto emit=[&](const std::string& line) {std::lock_guard lock(out_mutex);output<<line<<'\n'<<std::flush;};
    const auto vision_facts=protocol_info_append("",vision_info_facts(control.vision,control.vision_width));
    emit("INFO backend=hpu logit_bias=1 kv=cpu_fp16 slots=1"+vision_facts+(control.info.empty()?"":" "+one_line(control.info)));emit("READY "+std::to_string(context)+" stop");
    std::exception_ptr reader_error;
    std::jthread reader([&](std::stop_token stop) {
        strata::aux_cpus::pin_current_thread();
        try {
            std::string line;bool oversized=false;std::array<char,4096> bytes{};
            auto submit=[&] {
                if(!line.empty() && line.back()=='\r') line.pop_back();
                if(oversized) {emit("ERR request line too long");return;}
                if(line=="STOP") {std::lock_guard lock(mutex);if(active) active->store(true);return;}
                std::lock_guard lock(mutex);
                queue.push_back({line,{}});ready.notify_one();
            };
            while(!stop.stop_requested()) {
                pollfd fd{input_fd,POLLIN,0};const auto available=::poll(&fd,1,100);
                if(available<0) {if(errno==EINTR) continue;throw std::runtime_error("protocol input poll failed");}
                if(!available) continue;
                const auto count=::read(input_fd,bytes.data(),bytes.size());
                if(count<0) {if(errno==EINTR) continue;throw std::runtime_error("protocol input read failed");}
                if(!count) {if(!line.empty()) submit();break;}
                for(ssize_t i=0;i<count;++i) {
                    const auto c=bytes[size_t(i)];
                    if(c=='\n') {submit();line.clear();oversized=false;}
                    else if(line.size()<uint64_t(context)*12+65536) line.push_back(c);
                    else oversized=true;
                }
            }
        } catch(...) {std::lock_guard lock(mutex);reader_error=std::current_exception();}
        std::lock_guard lock(mutex);closed=true;ready.notify_one();
    });
    auto terminal=[&](const std::string& line) {
        {std::lock_guard lock(mutex);active.reset();}
        emit(line);
    };
        for(;;) {
            Pending pending;
            {
                std::unique_lock lock(mutex);ready.wait(lock,[&]{return closed || !queue.empty();});
                if(queue.empty()) break;
                pending=std::move(queue.front());queue.pop_front();
                pending.cancelled=std::make_shared<std::atomic<bool>>(false);active=pending.cancelled;
            }
            try {
                if(pending.line=="QUIT") {reader.request_stop();break;}
                if(pending.line.starts_with("SAVE ") || pending.line.starts_with("RESTORE ")) {
                    const auto outcome=session_command(pending.line,control,&generator,[&](const std::string& answer) {
                        if(answer.starts_with("SWAIT ")) emit(answer);else terminal(answer);
                    });
                    if(outcome==SessionOutcome::Fatal) {reader.request_stop();break;}
                    continue;
                }
                if(pending.line=="VRAM" || pending.line.starts_with("VRAM ")) {
                    if(!control.vram) throw std::invalid_argument("HPU engine was started without --vram-elastic");
                    std::optional<uint64_t> reserve;
                    if(pending.line!="VRAM") reserve=checked_product(number<uint64_t>(pending.line.substr(5)),1ull<<20);
                    terminal("VRAM "+one_line(control.vram(reserve)));continue;
                }
                const auto request=parse_generation_request(pending.line,vocabulary,context,{control.vision,control.vision_width});
                {
                    // Upstream's `aux_cpus=0|1` request key (:10698, applied at :11681-11690): where the non-pool,
                    // non-host threads run for THIS request, with -1 (absent) meaning the start-up --aux-cpus. The
                    // placement is re-swept at every request because threads created since the last one must be caught
                    // too, and a request that asked for a placement the machine cannot supply is said, not swallowed.
                    static const bool aux_startup=strata::aux_cpus::enabled();
                    const bool was=strata::aux_cpus::enabled();
                    strata::aux_cpus::set_enabled(request.aux_cpus<0?aux_startup:request.aux_cpus!=0);
                    if(strata::aux_cpus::enabled()!=was)
                        std::cerr<<"strata serve: aux cpus "<<(strata::aux_cpus::enabled()?"ON":"off")
                                 <<" for this request"<<(request.aux_cpus>0&&!strata::aux_cpus::enabled()?
                                     " (asked for, but no CPU is spare: see the start-up line)":"")<<"\n";
                }
                GenerationEvents events;
                events.reused=[&](uint32_t n) {emit("RESUME "+std::to_string(n));};
                events.progress=[&](uint32_t n,uint32_t total,double ms) {emit("PP "+std::to_string(n)+" "+std::to_string(total)+" "+std::to_string(ms));};
                events.token=[&](int32_t token) {emit("T "+std::to_string(token));};
                // Upstream's park line (src/program/generate.cpp:8371): the extent, the K/V bytes the reuse
                // carried over instead of re-reading, and whether this park replaced a superseded copy.
                events.parked=[&](uint64_t tokens,uint64_t reused,bool replaced) {
                    emit("PARK "+std::to_string(tokens)+" "+std::to_string(reused)+(replaced?" superseded":""));
                };
                // The park that upstream would split per stage: this backend says the refusal instead of parking
                // one stage's image and calling it the session.
                events.park_refused=[&](uint32_t stages) {
                    emit("PARK_REFUSED stage-count "+std::to_string(stages)+
                         ": a layer-split session parks one image per stage upstream, which one card cannot hold");
                };
                // Upstream's --conversation-cache-min-tokens skip (src/program/generate.cpp:8227-8236): a short
                // conversation is not worth one of the slots, the park is simply skipped, and the request still
                // succeeds. Upstream says so on stderr; this engine's report is its own line, because the client
                // reading the protocol has to be able to tell "skipped" from "parked and reused nothing".
                events.park_skipped_short=[&](uint64_t tokens,uint64_t minimum,uint64_t parked) {
                    emit("PARK_SKIPPED "+std::to_string(tokens)+" tokens, below the "+std::to_string(minimum)+
                         " minimum; parked="+std::to_string(parked));
                };
                const auto result=generator.run(request,events,[&]{return pending.cancelled->load();});
                terminal("DONE "+std::to_string(result.generated)+" "+std::to_string(result.prompt_tokens)+" "+std::to_string(result.prompt_ms)+" "+std::to_string(result.decode_ms)+" "+(result.finish=="cancelled"?"cancel":result.finish)+" "+std::to_string(result.drafts_accepted)+" "+std::to_string(result.drafts_proposed)+" "+std::to_string(result.reused)+" 0 0 0 0 0 "+std::to_string(result.prompt_read)+" 0");
            } catch(const SessionFatal& e) {terminal("ERR fatal model state: "+one_line(e.what()));reader.request_stop();break;}
            catch(const std::exception& e) {terminal("ERR "+one_line(e.what()));}
        }
    reader.join();
    if(reader_error) std::rethrow_exception(reader_error);
}

// ---- --batch: the slot protocol over any BatchSlotSession -------------------------------------------------
void serve_batch_protocol(int input_fd,std::ostream& output,BatchSlotSession& engine,int slots,uint32_t vocabulary,
                          uint32_t context,ProtocolControl control) {
    if(input_fd<0) throw std::invalid_argument("invalid protocol input file descriptor");
    if(slots<1 || slots>64) throw std::invalid_argument("invalid batch slot count");
    BatchSlots batch(slots,context);
    struct Pending {std::string line;std::shared_ptr<std::atomic<bool>> cancelled;};
    std::mutex mutex,out_mutex;std::condition_variable ready;std::deque<Pending> queue;
    bool closed=false;std::atomic<bool> quit{false};std::shared_ptr<std::atomic<bool>> active;
    auto emit=[&](const std::string& line) {std::lock_guard lock(out_mutex);output<<line<<'\n'<<std::flush;};
    // The INFO line says what this backend actually does: the slots are time-sliced through one Model, so the
    // figure is a slot COUNT, never a row-batched throughput claim.
    const auto vision_facts=protocol_info_append("",vision_info_facts(control.vision,control.vision_width));
    emit("INFO backend=hpu logit_bias=0 kv=cpu_fp16 slots="+std::to_string(slots)+
         " batch_slots="+std::to_string(slots)+" batch_schedule=time-sliced"+vision_facts+(control.info.empty()?"":" "+one_line(control.info)));
    emit("READY "+std::to_string(context)+" stop");

    std::vector<std::chrono::steady_clock::time_point> started(static_cast<size_t>(slots));
    std::exception_ptr reader_error;
    std::jthread reader([&](std::stop_token stop) {
        strata::aux_cpus::pin_current_thread();
        try {
            std::string line;bool oversized=false;std::array<char,4096> bytes{};
            auto submit=[&] {
                if(!line.empty() && line.back()=='\r') line.pop_back();
                if(line.empty()) return;
                if(oversized) {emit("ERR request line too long");return;}
                std::lock_guard lock(mutex);
                // Upstream STOP is out-of-band for the CURRENT ordinary
                // request/prompt, never a broadcast to other decoding slots.
                if(line=="STOP") {if(active) active->store(true);return;}
                queue.push_back({line,{}});ready.notify_one();
            };
            while(!stop.stop_requested() && !quit.load()) {
                pollfd fd{input_fd,POLLIN,0};const auto available=::poll(&fd,1,100);
                if(available<0) {if(errno==EINTR) continue;throw std::runtime_error("protocol input poll failed");}
                if(!available) continue;
                const auto count=::read(input_fd,bytes.data(),bytes.size());
                if(count<0) {if(errno==EINTR) continue;throw std::runtime_error("protocol input read failed");}
                if(!count) {if(!line.empty()) submit();break;}
                for(ssize_t i=0;i<count;++i) {
                    const auto c=bytes[size_t(i)];
                    if(c=='\n') {submit();line.clear();oversized=false;}
                    else if(line.size()<uint64_t(context)*12+65536) line.push_back(c);
                    else oversized=true;
                }
            }
        } catch(...) {std::lock_guard lock(mutex);reader_error=std::current_exception();}
        {std::lock_guard lock(mutex);closed=true;}ready.notify_one();
    });

    const auto clock_now=[] {return std::chrono::steady_clock::now();};
    auto elapsed_ms=[&](const int slot) {
        return std::chrono::duration<double,std::milli>(clock_now()-started[size_t(slot)]).count();
    };
    auto any_active=[&] {for(int slot=0;slot<slots;++slot) if(batch.active(slot)) return true;return false;};
    auto emit_events=[&](const std::vector<BatchSlots::Event>& events) {
        for(const auto& event:events) {
            if(event.kind==BatchSlots::Event::Kind::Token)
                emit("BT "+std::to_string(event.slot)+" "+std::to_string(event.token));
            else
                emit("BDONE "+std::to_string(event.slot)+" "+std::to_string(event.produced)+" "+
                     batch_finish_name(event.finish)+" "+std::to_string(elapsed_ms(event.slot)));
        }
    };
    auto settle=[&](const int slot) {
        // A healthy completed slot may cache its committed prefix. The error
        // paths below always release instead and can never preserve it.
        if(!batch.active(slot)) engine.finish(slot);
    };
    bool nonfinite_flush_pending=false;
    auto note_nonfinite=[&] {
        nonfinite_flush_pending=true;
        if(control.solo) control.solo->mark_nonfinite();
    };
    auto flush_nonfinite=[&] {
        if(!nonfinite_flush_pending && !(control.solo && control.solo->nonfinite_pending())) return;
        if(control.solo) (void)control.solo->flush_nonfinite();
        engine.invalidate_caches();nonfinite_flush_pending=false;
    };
    auto window=[&] {
        // Upstream's window plan over the active slots (:9957-9981), not a slot-at-a-time rotation: the
        // membership, the two-row mode and the cursor law come from the pure BatchSlots unit.
        const auto plan=batch.plan_window(control.batch_mtp,strata::kernels::kVerifyMaxT);
        for(size_t at=0;at<plan.size();) {
            const int slot=plan[at].slot;
            uint32_t planned=0;
            while(at<plan.size() && plan[at].slot==slot) {++planned;++at;}
            if(!batch.active(slot)) continue;
            const auto& current=batch.slot(slot);
            // Upstream keep=2 requires produced+2<=max_new, !stop and p+3<=ctx.
            // Restrict the producer BEFORE it runs; never discard hidden commits.
            const uint32_t allowed=!current.stop && current.max_new-current.produced>=2 &&
                current.position+3<=context?2:1;
            const uint32_t limit=std::min(planned,allowed);
            BatchSlotSession::Window output;
            try {
                output=engine.step_window(slot,limit);
                if(!output.count || output.count>limit) throw std::logic_error("invalid batch window size");
                for(uint32_t row=0;row<output.count;++row) {
                    const auto& step=output.rows[row];
                    if((step.eos || step.nonfinite) && row+1!=output.count) throw std::logic_error("nonterminal batch window stop");
                    if(!step.nonfinite && (step.token<0 || uint32_t(step.token)>=vocabulary)) throw std::logic_error("batch window token outside vocabulary");
                }
            }
            catch(const SessionFatal&) {engine.release(slot);throw;}
            catch(const std::exception& error) {
                // A step that failed cannot continue its slot: say why, close it, and free it.
                emit("ERR batch slot "+std::to_string(slot)+": "+one_line(error.what()));
                note_nonfinite();emit_events(batch.fail(slot));engine.release(slot);continue;
            }
            if(output.rows[output.count-1].nonfinite) {note_nonfinite();emit_events(batch.fail(slot));engine.release(slot);continue;}
            for(uint32_t row=0;row<output.count;++row) {
                const auto& step=output.rows[row];emit_events(batch.complete(slot,step.token,step.eos));
            }
            settle(slot);
        }
    };

    for(;;) {
        bool have=false;Pending pending;
        {
            std::unique_lock lock(mutex);active.reset();
            while(!quit.load() && !closed && queue.empty()) {
                if(any_active()) break;   // a pending window runs rather than waiting for input
                ready.wait_for(lock,std::chrono::milliseconds(50));
            }
            if(!queue.empty()) {
                pending=std::move(queue.front());queue.pop_front();have=true;
                if(pending.line.starts_with("GEN ") || pending.line.starts_with("BGEN ") || pending.line.starts_with("BGENI ")) {
                    pending.cancelled=std::make_shared<std::atomic<bool>>(false);active=pending.cancelled;
                }
            }
            else if(quit.load() || (closed && !any_active())) break;
        }
        if(!have) {
            if(any_active()) {
                try {window();}
                catch(const SessionFatal& e) {emit("FATAL "+one_line(e.what()));reader.request_stop();break;}
                continue;
            }
            break;
        }
        const std::string& line=pending.line;
        try {
            if(line=="QUIT") {quit.store(true);break;}
            if(line.starts_with("BSTOP ")) {
                const auto requested_slot=number<int64_t>(line.substr(6));
                if(requested_slot<0 || requested_slot>=slots) throw std::invalid_argument("BSTOP: no such slot");
                const int slot=int(requested_slot);
                if(batch.active(slot)) {batch.stop(slot);engine.stop(slot);}
                continue;
            }
            if(line=="BYIELD" || line.starts_with("BYIELD ")) continue; // no prompt read is in flight here (:10333)
            if(line=="VRAM" || line.starts_with("VRAM ")) {
                if(any_active()) {emit("ERR VRAM: not while batch slots are decoding");continue;}
                if(!control.vram) throw std::invalid_argument("HPU engine was started without --vram-elastic");
                std::optional<uint64_t> reserve;
                if(line!="VRAM") reserve=checked_product(number<uint64_t>(line.substr(5)),1ull<<20);
                emit("VRAM "+one_line(control.vram(reserve)));continue;
            }
            if(line.starts_with("SAVE ") || line.starts_with("RESTORE ")) {
                // Additive HPU idle support: upstream refuses ALL --batch
                // disk commands. The explicit main owner is unambiguous here.
                if(any_active()) {emit("SERR invalid 0 session files: not while batch slots are decoding");continue;}
                if(!control.solo) {emit("SERR invalid 0 disk sessions need an explicit main session");continue;}
                const bool pending_flush=nonfinite_flush_pending || control.solo->nonfinite_pending();
                const auto outcome=session_command(line,control,control.solo,emit);
                if(outcome==SessionOutcome::Fatal) {reader.request_stop();break;}
                // adopt() cleared the old main flag. Preserve the freshly
                // restored state while removing other pre-failure idle owners.
                if(outcome==SessionOutcome::Restored && pending_flush) {engine.invalidate_caches();nonfinite_flush_pending=false;}
                continue;
            }
            if((line.starts_with("GEN ") || line.starts_with("GENI ")) && control.solo) {
                const auto request=parse_generation_request(line,vocabulary,context,{control.vision,control.vision_width});
                if(!request.bias.empty()) {emit("ERR logit_bias: continuous batching is not supported");continue;}
                flush_nonfinite();
                auto poll_cancel=[&] {
                    // Native Model may hold its own lock here. BSTOP only
                    // changes host scheduler flags; no foreign Model work.
                    std::lock_guard lock(mutex);
                    for(auto it=queue.begin();it!=queue.end();) {
                        const auto& command=it->line;
                        if(command.starts_with("BSTOP ")) {
                            try {
                                const auto b=number<int64_t>(command.substr(6));
                                if(b<0 || b>=slots) throw std::invalid_argument("BSTOP: no such slot");
                                if(batch.active(int(b))) {batch.stop(int(b));engine.stop(int(b));}
                            } catch(const std::exception& e) {emit("ERR "+one_line(e.what()));}
                        } else {++it;continue;}
                        it=queue.erase(it);
                    }
                    return pending.cancelled->load();
                };
                GenerationEvents events;
                events.reused=[&](uint32_t n){emit("RESUME "+std::to_string(n));};
                events.progress=[&](uint32_t n,uint32_t total,double ms){emit("PP "+std::to_string(n)+" "+std::to_string(total)+" "+std::to_string(ms));};
                events.token=[&](int32_t token){emit("T "+std::to_string(token));};
                events.parked=[&](uint64_t tokens,uint64_t reused,bool replaced){emit("PARK "+std::to_string(tokens)+" "+std::to_string(reused)+(replaced?" superseded":""));};
                events.park_refused=[&](uint32_t stages){emit("PARK_REFUSED stage-count "+std::to_string(stages));};
                events.park_skipped_short=[&](uint64_t tokens,uint64_t minimum,uint64_t parked){
                    emit("PARK_SKIPPED "+std::to_string(tokens)+" tokens, below the "+std::to_string(minimum)+
                         " minimum; parked="+std::to_string(parked));
                };
                events.has_foreign_work=any_active;
                events.cooperate=[&]{window();};
                events.yield=[&](const GenerationRead& read) {
                    std::optional<int> target;
                    {
                        std::lock_guard lock(mutex);
                        for(auto it=queue.begin();it!=queue.end();) {
                            if(!it->line.starts_with("BYIELD ")) {++it;continue;}
                            try {
                                const auto value=number<int64_t>(it->line.substr(7));
                                if(value<0 || value>=slots) throw std::invalid_argument("BYIELD: no such slot");
                                target=int(value);
                            } catch(const std::exception& e) {emit("ERR "+one_line(e.what()));}
                            it=queue.erase(it);
                        }
                    }
                    if(!target || !read.can_yield || batch.active(*target)) return -1;
                    return engine.park_partial(*target,read)?*target:-1;
                };
                const auto result=control.solo->run(request,events,poll_cancel);
                if(result.finish=="nonfinite") note_nonfinite();
                if(result.yielded_slot>=0) {
                    if(result.generated || !result.yielded_tokens || result.yielded_tokens>=result.prompt_tokens) throw SessionFatal("invalid yielded GEN result");
                    emit("YIELDED "+std::to_string(result.yielded_slot)+" "+std::to_string(result.yielded_tokens));
                }
                emit("DONE "+std::to_string(result.generated)+" "+std::to_string(result.prompt_tokens)+" "+std::to_string(result.prompt_ms)+" "+
                    std::to_string(result.decode_ms)+" "+(result.finish=="cancelled"?"cancel":result.finish)+" "+
                    std::to_string(result.drafts_accepted)+" "+std::to_string(result.drafts_proposed)+" "+std::to_string(result.reused)+
                    " 0 0 0 0 0 "+std::to_string(result.prompt_read)+" 0");
                if(QsaPhaseStats::enabled()) QsaPhaseStats::instance().flush();
                continue;
            }
            const bool image=line.starts_with("BGENI ");
            if(!image && !line.starts_with("BGEN ")) {
                emit("ERR expected: GEN <max_new> <id,id,...> or BGEN <slot> <max_new> <id,id,...>");
                continue;
            }
            // BGEN <slot> <max_new> [keys] <ids>
            size_t at=image?6:5;
            const auto next_number=[&](const char* const what) {
                while(at<line.size() && line[at]==' ') ++at;
                const auto begin=at;
                while(at<line.size() && line[at]!=' ') ++at;
                if(begin==at) throw std::invalid_argument(std::string("BGEN: missing ")+what);
                return number<int64_t>(line.substr(begin,at-begin));
            };
            const auto slot=next_number("slot");
            const auto max_new=next_number("max_new");
            if(slot<0 || slot>=slots || max_new<1 || batch.active(int(slot))) {
                emit("ERR BGEN: no such free slot (--batch "+std::to_string(slots)+") or a bad max_new");
                continue;
            }
            // Upstream rewrites `BGENI <slot> <max_new> ...` to `GENI 1 ...` (:10349-10362) and lets the
            // ordinary GENI request law parse the file, keys and position records. SlotSession installs that
            // request's M-RoPE table before any prompt work, the same request-scoped seam as the solo path.
            const std::string body=line.substr(at);
            const auto request=parse_generation_request((image?"GENI 1":"GEN 1")+body,vocabulary,context,{control.vision,control.vision_width});
            if(!request.bias.empty()) {emit("ERR logit_bias: continuous batching is not supported");continue;}
            // Upstream accepts a signed64 budget, but our scheduler stores
            // uint32. Context always ends the slot first when the budget is
            // larger; cap in the wide type BEFORE converting, never wrap.
            const auto slot_budget=uint32_t(std::min<uint64_t>(uint64_t(max_new),context));
            flush_nonfinite();
            int32_t first=0;bool have_token=false;
            const auto admit_start=clock_now();
            BatchSlotSession::ReadControl read_control;
            read_control.cancelled=[&]{return pending.cancelled->load();};
            read_control.progress=[&](uint32_t n,uint32_t total,double ms){emit("PP "+std::to_string(n)+" "+std::to_string(total)+" "+std::to_string(ms));};
            read_control.yield=[&]()->std::optional<int> {
                std::optional<int> target;
                std::lock_guard lock(mutex);
                for(auto it=queue.begin();it!=queue.end();) {
                    const auto& command=it->line;
                    if(command.starts_with("BSTOP ") || command.starts_with("BYIELD ")) {
                        const bool yield=command.starts_with("BYIELD ");
                        try {
                            const auto value=number<int64_t>(command.substr(yield?7:6));
                            if(value<0 || value>=slots) throw std::invalid_argument(yield?"BYIELD: no such slot":"BSTOP: no such slot");
                            if(yield) target=int(value);
                            else if(batch.active(int(value))) {batch.stop(int(value));engine.stop(int(value));}
                        } catch(const std::exception& e) {emit("ERR "+one_line(e.what()));}
                    } else {++it;continue;}
                    it=queue.erase(it);
                }
                return target;
            };
            const auto* share=std::getenv("STRATA_BATCH_DECODE_SHARE");
            const double decode_share=share?std::max(0.,std::atof(share)):0.5;
            read_control.has_work=any_active;
            if(decode_share>0) read_control.between=[&](double prompt_ms) {
                const auto start=clock_now();
                do {window();} while(any_active() && std::chrono::duration<double,std::milli>(clock_now()-start).count()<decode_share*prompt_ms);
            };
            auto admission=engine.admit_cooperative(int(slot),request,slot_budget,[&](const int32_t token) {
                first=token;have_token=true;emit("T "+std::to_string(token));
            },read_control);
            if(admission.nonfinite) {
                if(have_token || admission.continues || admission.yielded_slot>=0 || admission.yielded_tokens) throw SessionFatal("nonfinite admission published state");
                note_nonfinite();engine.release(int(slot));
                const auto ms=std::chrono::duration<double,std::milli>(clock_now()-admit_start).count();
                emit("DONE 0 "+std::to_string(admission.prompt_tokens)+" "+std::to_string(ms)+" 0 nonfinite 0 0 "+std::to_string(admission.reused_tokens)+" 0 0 0 0 0 "+std::to_string(admission.prompt_read)+" 0");
                emit("BADM "+std::to_string(slot)+" 0");continue;
            }
            if(admission.cancelled) {
                if(admission.continues || admission.yielded_slot>=0 || admission.yielded_tokens) throw SessionFatal("cancelled admission still owns an active scheduler slot");
                const auto ms=std::chrono::duration<double,std::milli>(clock_now()-admit_start).count();
                emit("DONE "+std::to_string(have_token?1:0)+" "+std::to_string(admission.prompt_tokens)+" "+std::to_string(ms)+" 0 cancel 0 0 "+std::to_string(admission.reused_tokens)+" 0 0 0 0 0 "+std::to_string(admission.prompt_read)+" 0");
                emit("BADM "+std::to_string(slot)+" 0");continue; // producer retained only an optional idle/partial owner
            }
            if(admission.yielded_slot>=0 || admission.yielded_tokens) {
                if(have_token || admission.continues || admission.yielded_slot!=slot || !admission.yielded_tokens ||
                   admission.yielded_tokens>=admission.prompt_tokens || admission.reused_tokens>admission.yielded_tokens) {engine.release(int(slot));throw std::logic_error("invalid yielded admission");}
                const auto ms=std::chrono::duration<double,std::milli>(clock_now()-admit_start).count();
                emit("YIELDED "+std::to_string(slot)+" "+std::to_string(admission.yielded_tokens));
                emit("DONE 0 "+std::to_string(admission.prompt_tokens)+" "+std::to_string(ms)+" 0 cancel 0 0 "+
                     std::to_string(admission.reused_tokens)+" 0 0 0 0 0 "+
                     std::to_string(admission.yielded_tokens-admission.reused_tokens)+" 0");
                emit("BADM "+std::to_string(slot)+" 0");
                continue; // the engine owns the partial prefix; never release it here
            }
            if(!have_token) throw std::runtime_error("the admission produced no token");
            const auto ms=std::chrono::duration<double,std::milli>(clock_now()-admit_start).count();
            // The admission is an ordinary request the client also sees finish (:12984 is answered after it).
            emit("DONE 1 "+std::to_string(admission.prompt_tokens)+" "+std::to_string(ms)+" 0 "+(admission.eos?"stop":"length")+" 0 0 "+std::to_string(admission.reused_tokens)+" 0 0 0 0 0 "+std::to_string(admission.prompt_read)+" 0");
            const bool continues=admission.continues && !admission.eos && slot_budget>1;
            if(continues) {
                try {
                    batch.admit(int(slot),slot_budget,int64_t(admission.prompt_tokens),first);
                    started[size_t(slot)]=clock_now();
                } catch(...) {engine.release(int(slot));throw;}
            }
            emit("BADM "+std::to_string(slot)+" "+(continues?"1":"0"));
            if(!continues) engine.release(int(slot));
        } catch(const SessionFatal& e) {emit("FATAL "+one_line(e.what()));reader.request_stop();break;}
        catch(const std::exception& e) {emit("ERR "+one_line(e.what()));}
    }
    reader.request_stop();
    reader.join();
    if(reader_error) std::rethrow_exception(reader_error);
}
}
