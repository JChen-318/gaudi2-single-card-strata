// hpu/tests/batch_protocol.cpp - host evidence for the --batch SLOT PROTOCOL over a fake engine.
//
// The slot STATE LAW is pinned in hpu/tests/batch_slots.cpp; this test pins the wire grammar the engine and the
// client see (src/program/generate.cpp:566-568, :10326-10363, :12984, :10060-10089):
//   BGEN <slot> <max_new> <ids>  ->  T <first token>, the admission's own DONE, then BADM <slot> <1|0>;
//   a continuing slot            ->  BT <slot> <tok> per window, BDONE <slot> <produced> <finish> <ms>;
//   BSTOP <slot>                 ->  that slot's next window still emits its token, then BDONE ... cancel;
//   BGEN on a busy/absent slot   ->  ERR BGEN: no such free slot;
//   VRAM while slots decode      ->  ERR VRAM: not while batch slots are decoding  (:10336);
//   a bare GEN                   ->  ERR expected: GEN ... or BGEN ...
// The engine here is a script, so no model, SDK or device is involved.
#include "strata/hpu/protocol.hpp"
#include "strata/hpu/slot_session.hpp"
#include "strata/hpu/state_io.hpp"
#include "strata/hpu/allocation_budget.hpp"

#include <atomic>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

namespace {
using namespace strata::hpu;

int failures = 0;
void check(const bool ok, const std::string& what, const std::string& detail = {}) {
    if (!ok) {
        ++failures;
        std::cout << "FAIL " << what;
        // The observed values ride along, because a wire-law failure that only names the law cannot be
        // diagnosed from a CI log.
        if (!detail.empty()) std::cout << " [" << detail << "]";
        std::cout << "\n";
    }
}
std::string shown(const std::vector<uint32_t>& limits) {
    std::ostringstream out;
    for (const auto value : limits) out << value << ',';
    return out.str();
}

// A scripted engine: admission emits one token, each step emits the next of its own sequence, and a slot reports
// EOS at its scripted token so the protocol's BDONE "stop" path is reachable without a model.
struct FakeEngine : BatchSlotSession {
    struct Slot {
        uint32_t max_new = 0, produced = 0;
        int32_t next = 0, eos_at = 0;
        bool admitted = false;
    };
    std::vector<Slot> slots;
    std::atomic<int> steps{0};
    std::atomic<int> admits{0};
    std::atomic<int> finishes{0};
    int first_eos_slot=-1;
    std::vector<uint32_t> admission_budgets;

    explicit FakeEngine(const int count) : slots(size_t(count)) {}

    Admission admit(const int slot, const GenerationRequest& request, const uint32_t max_new,
                    const std::function<void(int32_t)>& token) override {
        ++admits;
        admission_budgets.push_back(max_new);
        Slot& state = slots[size_t(slot)];
        state = Slot{};
        state.admitted = true;
        state.max_new = max_new;
        state.produced = 1;
        state.next = 100 + 10 * slot;
        state.eos_at = 100 + 10 * slot + 2; // the third token of this slot is its EOS
        token(10 + slot);                   // the admission's first token
        const bool eos=slot==first_eos_slot;
        return Admission{uint32_t(request.prompt.size()), max_new > 1 && !eos, 0, eos};
    }
    Step step(const int slot) override {
        ++steps;
        Slot& state = slots[size_t(slot)];
        const int32_t token = state.next++;
        ++state.produced;
        return Step{token, token == state.eos_at, false};
    }
    void stop(const int slot) override { (void)slot; }
    void release(const int slot) override { slots[size_t(slot)] = Slot{}; }
    void finish(const int slot) override {++finishes;release(slot);}
};

std::vector<std::string> split_lines(const std::string& text) {
    std::vector<std::string> lines;
    size_t at = 0;
    while (at < text.size()) {
        const auto end = text.find('\n', at);
        if (end == std::string::npos) break;
        lines.push_back(text.substr(at, end - at));
        at = end + 1;
    }
    return lines;
}

bool starts_with(const std::string& text, const std::string& prefix) { return text.rfind(prefix, 0) == 0; }

// Consume the expected lines in order; anything else is a failure that names the surprise.
void expect(std::vector<std::string>& lines, size_t& cursor, const std::string& prefix) {
    if (cursor >= lines.size()) {
        ++failures;
        std::cout << "FAIL expected a line starting with \"" << prefix << "\" but the stream ended\n";
        return;
    }
    if (!starts_with(lines[cursor], prefix)) {
        ++failures;
        std::cout << "FAIL expected \"" << prefix << "\" at line " << cursor + 1 << ", got \"" << lines[cursor] << "\"\n";
        return;
    }
    ++cursor;
}
void boundaries() {
    int fds[2]{};if(::pipe(fds)!=0) {check(false,"boundary protocol pipe");return;}
    FakeEngine engine(2);engine.first_eos_slot=1;std::ostringstream output;
    const std::string commands=
        "BGEN 0 4294967297 1,2\n"
        "BSTOP 4294967296\nBSTOP -4294967296\nBSTOP 9223372036854775807\n"
        "BGEN 1 4 3\nBGEN 4294967296 4 3\n"
        "BGEN 1 9223372036854775807 3\nBGEN 1 4294967296 3\nBGEN 1 1 3\n"
        "BGEN 1 9223372036854775808 3\nBGEN 1 -1 3\n";
    check(::write(fds[1],commands.data(),commands.size())==ssize_t(commands.size()),"boundary command write");
    std::jthread server([&]{serve_batch_protocol(fds[0],output,engine,2,256,32);});
    const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(10);
    while(engine.finishes.load()<1 && std::chrono::steady_clock::now()<deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    const std::string quit="QUIT\n";check(::write(fds[1],quit.data(),quit.size())==ssize_t(quit.size()),"boundary QUIT write");
    server.join();::close(fds[0]);::close(fds[1]);
    const auto lines=split_lines(output.str());
    const auto count=[&](const std::string& prefix){return std::count_if(lines.begin(),lines.end(),[&](const auto& line){return starts_with(line,prefix);});};
    check(count("ERR BSTOP: no such slot")==3,"wide signed slot id wrapped into a valid BSTOP");
    check(engine.admission_budgets==std::vector<uint32_t>({32,4,32,32,1}),"signed64 output budget wrapped before the context cap");
    check(count("BADM 1 0")==4 && count("BT 1 ")==0,"first-token EOS continued an active batch slot");
    const auto eos_done=std::count_if(lines.begin(),lines.end(),[](const auto& line){return starts_with(line,"DONE 1 1 ") && line.find(" 0 stop 0 0 0")!=std::string::npos;});
    check(eos_done==4,"first-token EOS reported length instead of stop");
    check(count("BDONE 0 4 stop ")==1 && count("BDONE 0 2 cancel ")==0,"wide BSTOP changed a real slot's finish");
    check(engine.admits.load()==5,"invalid wide slot/budget reached the engine");
}
struct WindowEngine final:FakeEngine {
    std::array<std::vector<uint32_t>,6> limits;
    WindowEngine():FakeEngine(6) {}
    Admission admit(int slot,const GenerationRequest& r,uint32_t maximum,const std::function<void(int32_t)>& token) override {
        // Give the independent reader time to queue the prewritten STOP before
        // this prompt read finishes, just as an actual prompt would do.
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        return FakeEngine::admit(slot,r,maximum,token);
    }
    Window step_window(int slot,uint32_t limit) override {
        limits[size_t(slot)].push_back(limit);++steps;
        Window result;result.count=limit;
        for(uint32_t row=0;row<limit;++row) result.rows[row]={slots[size_t(slot)].next++,false,false};
        if(slot==4) result.rows[limit-1].nonfinite=true;
        if(slot==5) result.count=3; // reject envelope BEFORE accessing its rows
        return result;
    }
};
void windows() {
    int fds[2]{};if(::pipe(fds)!=0) {check(false,"window protocol pipe");return;}
    WindowEngine engine;std::ostringstream output;
    const std::string commands="BGEN 0 5 1,2\nBGEN 1 8 3\nBSTOP 1\n"
        "BGEN 2 8 1,2,3,4,5,6,7,8,9,10,11,12,13,14\nBGEN 3 2 3\nBGEN 4 4 3\nBGEN 5 4 3\n";
    check(::write(fds[1],commands.data(),commands.size())==ssize_t(commands.size()),"window command write");
    std::jthread server([&]{serve_batch_protocol(fds[0],output,engine,6,256,16);});
    const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(10);
    while(engine.finishes.load()<4 && std::chrono::steady_clock::now()<deadline) std::this_thread::sleep_for(std::chrono::milliseconds(1));
    const std::string quit="QUIT\n";check(::write(fds[1],quit.data(),quit.size())==ssize_t(quit.size()),"window QUIT write");
    server.join();::close(fds[0]);::close(fds[1]);
    const auto lines=split_lines(output.str());
    const auto count=[&](const std::string& p){return std::count_if(lines.begin(),lines.end(),[&](const auto& line){return starts_with(line,p);});};
    const auto observed=[&](const int slot){return "limits["+std::to_string(slot)+"]="+shown(engine.limits[size_t(slot)]);};
    const int before=failures;
    // Upstream's window plan gives a live slot TWO rows only under --batch-mtp ("Each MTP slot uses two rows",
    // src/program/generate.cpp:9957-9959's `S + (batch_mtp ? 2 : 1)`), so without that flag every pass is one row
    // per slot even when the slot's own budget could carry two.
    check(engine.limits[0]==std::vector<uint32_t>({1,1,1,1}) && count("BT 0 ")==4 && count("BDONE 0 5 length ")==1,
          "one row per live slot in the non-MTP window plan, slot 0 still reaches its length",
          observed(0)+" BT0="+std::to_string(count("BT 0 "))+" done0="+std::to_string(count("BDONE 0 5 length ")));
    check(engine.limits[1]==std::vector<uint32_t>({1}) && count("BT 1 ")==1 && count("BDONE 1 2 cancel ")==1,
          "BSTOP bound arrived after speculative producer",
          observed(1)+" BT1="+std::to_string(count("BT 1 "))+" cancel1="+std::to_string(count("BDONE 1 2 cancel ")));
    check(engine.limits[2]==std::vector<uint32_t>({1,1}) && count("BDONE 2 3 length ")==1,
          "context bound arrived after speculative producer",
          observed(2)+" done2="+std::to_string(count("BDONE 2 3 length ")));
    check(engine.limits[3]==std::vector<uint32_t>({1}) && count("BDONE 3 2 length ")==1,
          "output budget bound arrived after speculative producer",
          observed(3)+" done3="+std::to_string(count("BDONE 3 2 length ")));
    check(count("BT 4 ")==0 && count("BDONE 4 1 nonfinite ")==1,"nonfinite second row published a first token",
          "BT4="+std::to_string(count("BT 4 "))+" limits4="+observed(4));
    check(count("BT 5 ")==0 && count("ERR batch slot 5: invalid batch window size")==1 && count("BDONE 5 1 nonfinite ")==1,
          "malformed window published tokens",
          "BT5="+std::to_string(count("BT 5 "))+" limits5="+observed(5));
    if(failures>before) std::cout<<"window transcript:\n"<<output.str();
}
// The same window with --batch-mtp on: upstream gives each live slot TWO rows ("Each MTP slot uses two rows",
// src/program/generate.cpp:9957-9959), so slot 0's five outputs are the admitted token plus two atomic
// two-row windows. Without this case the two-row mode would only be pinned in the pure slot unit.
void mtp_two_row_window() {
    int fds[2]{};if(::pipe(fds)!=0) {check(false,"mtp window protocol pipe");return;}
    WindowEngine engine;std::ostringstream output;
    const std::string commands="BGEN 0 5 1,2\n";
    check(::write(fds[1],commands.data(),commands.size())==ssize_t(commands.size()),"mtp window command write");
    ProtocolControl control;control.batch_mtp=true;
    std::jthread server([&]{serve_batch_protocol(fds[0],output,engine,6,256,16,control);});
    const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(10);
    while(engine.finishes.load()<1 && std::chrono::steady_clock::now()<deadline) std::this_thread::sleep_for(std::chrono::milliseconds(1));
    const std::string quit="QUIT\n";check(::write(fds[1],quit.data(),quit.size())==ssize_t(quit.size()),"mtp window QUIT write");
    server.join();::close(fds[0]);::close(fds[1]);
    const auto lines=split_lines(output.str());
    const auto count=[&](const std::string& p){return std::count_if(lines.begin(),lines.end(),[&](const auto& line){return starts_with(line,p);});};
    const auto detail="limits="+shown(engine.limits[0])+" BT0="+std::to_string(count("BT 0 "))+
        " done0="+std::to_string(count("BDONE 0 5 length "));
    check(engine.limits[0]==std::vector<uint32_t>({2,2}) && count("BT 0 ")==4 && count("BDONE 0 5 length ")==1,
          "two-output atomic window/length protocol under --batch-mtp",detail);
    if(!detail.empty() && engine.limits[0]!=std::vector<uint32_t>({2,2})) std::cout<<"mtp window transcript:\n"<<output.str();
}
void cooperative_prompt() {
    int fds[2]{};if(::pipe(fds)!=0) {check(false,"cooperative pipe");return;}
    std::vector<int32_t> fed;bool slow=true;
    SlotSessionOperations op;op.reset=[&]{fed.clear();};
    op.prefill=[&](std::span<const int32_t> input) {
        if(slow) {std::this_thread::sleep_for(std::chrono::milliseconds(20));slow=false;}
        fed.insert(fed.end(),input.begin(),input.end());return std::vector<float>(32,0);
    };
    op.prefill_window=op.prefill;
    op.decode=[&](int32_t id){fed.push_back(id);return std::vector<float>(32,0);};
    op.checkpoint=[&]()->std::function<void()> {auto ids=fed;return [&,ids=std::move(ids)]{fed=ids;};};
    SlotCheckpointOptions options;options.prefill_chunk=2;options.short_read=0;options.interval=0;options.boundaries.turn_token=-1;
    SlotSession engine(2,32,32,{},std::move(op),true,options);std::ostringstream output;
    const std::string prompt="1,2,3,4,5,6,7,8,9,10,11,12";
    const std::string commands="BGEN 0 3 "+prompt+"\nBYIELD 4294967296\nBYIELD 0\nBGEN 1 2 3\nBGEN 0 3 "+prompt+"\n";
    check(::write(fds[1],commands.data(),commands.size())==ssize_t(commands.size()),"cooperative command write");::close(fds[1]);
    serve_batch_protocol(fds[0],output,engine,2,32,32);::close(fds[0]);
    const auto text=output.str();const auto lines=split_lines(text);
    const auto count=[&](const std::string& p){return std::count_if(lines.begin(),lines.end(),[&](const auto& line){return starts_with(line,p);});};
    check(count("YIELDED 0 2")==1 && count("BADM 0 0")==1 && engine.yielded_reads()==1,"real SlotSession BYIELD did not retain a partial read");
    check(count("ERR BYIELD: no such slot")==1 && count("T ")==2,"wide BYIELD wrapped or partial admission emitted a token");
    check(count("DONE 0 12 ")==1 && text.find(" cancel 0 0 0 0 0 0 0 0 2 0")!=std::string::npos,"YIELDED DONE did not report zero output/read count/cancel");
    check(engine.reused_tokens()==2 && count("BADM 0 1")==1,"protocol released the yielded image before the reissued BGEN");
    check(count("BT 1 ")==1 && text.find("BT 1 ")<text.rfind("\nT ") && engine.interleaved_windows()>0,"active slot stalled until the other prompt was read");
    check(count("BDONE 0 3 length ")==1 && count("BDONE 1 2 length ")==1,"cooperative prompt/window terminal responses");
}
void shared_main() {
    std::vector<int32_t> fed;bool mode=true;SlotSession* slots=nullptr;int cancel_writer=-1;
    auto logits=[&]{std::vector<float> row(32,-10);row[mode?1:2]=10;return row;};
    auto reset=[&]{fed.clear();};auto decode=[&](int32_t id){fed.push_back(id);return logits();};
    auto checkpoint=[&]()->std::function<void()> {auto ids=fed;const bool saved=mode;return [&,ids=std::move(ids),saved]{fed=ids;mode=saved;};};
    GenerationOperations go;go.reset=reset;go.decode=[&](int32_t id,const auto&){return decode(id);};
    go.prefill=[&](std::span<const int32_t> input,const PromptCommit& commit,const auto& cancel){
        if(cancel_writer>=0 && std::find(input.begin(),input.end(),25)!=input.end()) {
            const std::string stop="STOP\n";check(::write(cancel_writer,stop.data(),stop.size())==ssize_t(stop.size()),"STOP during GEN write");
            ::close(cancel_writer);cancel_writer=-1;
            const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(5);
            while(std::chrono::steady_clock::now()<deadline) {if(cancel && cancel()) throw Cancelled();std::this_thread::sleep_for(std::chrono::milliseconds(1));}
            throw std::runtime_error("STOP was not observed during GEN");
        }
        fed.insert(fed.end(),input.begin(),input.end());auto values=logits();commit(uint32_t(input.size()),values);
    };
    go.checkpoint=checkpoint;go.checkpoint_bytes=1000;go.available_host_bytes=[] {return std::optional<uint64_t>(UINT64_MAX);};go.control_vector=[&](bool value){mode=value;};
    go.select_prefix=[&](std::span<const int32_t> prompt,uint64_t maximum,bool control,std::span<const RequestImage> images)->std::optional<GenerationPrefix>{return slots?slots->select_prefix(prompt,maximum,control,images):std::nullopt;};
    GenerationOptions options;options.cache_slots=4;options.cache_bytes=100000;options.max_draft=0;options.prefill_chunk=2;options.short_read=0;options.cache_every=0;options.boundaries.turn_token=-1;
    Generation main(32,128,{},std::move(go),options);
    SlotSessionOperations so;so.before_foreign=[&]{main.suspend_live();};so.select_prefix=[&](std::span<const int32_t> prompt,uint64_t maximum,bool control,std::span<const RequestImage> images){return main.select_prefix(prompt,maximum,control,images);};
    so.logits=logits;so.reset=reset;so.decode=decode;so.prefill=[&](std::span<const int32_t> input){fed.insert(fed.end(),input.begin(),input.end());return logits();};so.prefill_window=so.prefill;
    so.import_prefixes=[&](const GenerationRead& read){main.import_prefixes(read);};
    so.checkpoint=checkpoint;so.control_vector=[&](bool value){mode=value;};
    SlotSession engine(2,32,128,{},std::move(so),true);slots=&engine;ProtocolControl control;control.solo=&main;
    const auto serve=[&](const std::string& commands,bool inject_stop=false) {
        int fds[2]{};if(::pipe(fds)!=0) throw std::runtime_error("shared protocol pipe");
        check(::write(fds[1],commands.data(),commands.size())==ssize_t(commands.size()),"shared command write");
        if(inject_stop) cancel_writer=fds[1];else ::close(fds[1]);
        std::ostringstream output;serve_batch_protocol(fds[0],output,engine,2,32,128,control);::close(fds[0]);return output.str();
    };
    const auto text=serve("BGEN 0 4 temperature=0 cvec=0 3,4,5\n"
        "GEN 3 temperature=0 cvec=1 10,11,12,13,14,15,16,17\n"
        "GEN 1 temperature=0 cvec=1 10,11,12,13,14,15,16,17,1,1,19\n"
        "BGEN 1 2 temperature=0 cvec=1 10,11,12,13,14,15,16,17,1,1,19,20\n"
        "GEN 1 temperature=0 cvec=0 3,4,5,2,2,2,6\n"
        "GEN 1 logit_bias=1:1 1\n");
    const auto lines=split_lines(text);const auto count=[&](const std::string& p){return std::count_if(lines.begin(),lines.end(),[&](const auto& line){return starts_with(line,p);});};
    check(count("T 1")==5 && count("T 2")==2 && count("BT 0 2")==3 && count("BT 1 1")==1,"GEN/BGEN streams inherited another session's mode/counter");
    check(text.find("BT 0 ")<text.find("\nT 1"),"GEN prompt blocked active slots");
    check(count("RESUME 10")==1 && count("RESUME 6")==1 && count("RESUME 0")==1,"GEN did not choose main live or longer idle slot");
    const auto borrowed=std::count_if(lines.begin(),lines.end(),[](const auto& line){return starts_with(line,"DONE 1 12 ") && line.find(" length 0 0 11 ")!=std::string::npos;});
    check(borrowed==1,"BGEN did not borrow the longest main prefix");
    check(count("ERR logit_bias: continuous batching is not supported")==1 && count("ERR ")==1,"batch GEN refused or hid unsupported bias");
    const auto stopped=serve("BGEN 0 4 temperature=0 cvec=0 23,24\nGEN 3 temperature=0 25,26,27,28\n",true);
    check(stopped.find("DONE 0 4 ")!=std::string::npos && stopped.find(" cancel 0 0 ")!=std::string::npos && stopped.find("BDONE 0 4 length ")!=std::string::npos,"STOP did not cancel only GEN while the independent slot completed its full reply");
    const auto yielded=serve("GEN 4 temperature=0 cvec=1 0,1,2,3,4,5,6,7,8,9,10,11\n"
        "BYIELD 4294967296\nBYIELD 1\nBGEN 0 2 temperature=0 cvec=0 23,24\n"
        "BGEN 1 3 temperature=0 cvec=1 0,1,2,3,4,5,6,7,8,9,10,11\n");
    const auto yl=split_lines(yielded);
    const auto yc=[&](const std::string& p){return std::count_if(yl.begin(),yl.end(),[&](const auto& line){return starts_with(line,p);});};
    check(yc("YIELDED 1 2")==1 && yc("DONE 0 12 ")==1 && yc("T ")==2 && yc("BADM 1 0")==0,"solo BYIELD emitted a token/BADM or did not own the reserved free slot");
    check(yc("ERR BYIELD: no such slot")==1 && yielded.find(" cancel 0 0 0 0 0 0 0 0 2 0")!=std::string::npos,"solo BYIELD wide target/read count contract");
    const auto yr=std::count_if(yl.begin(),yl.end(),[](const auto& line){return starts_with(line,"DONE 1 12 ") && line.find(" length 0 0 2 ")!=std::string::npos;});
    check(yr==1 && yc("BDONE 1 3 length ")==1 && yc("BDONE 0 2 length ")==1,"BGEN did not resume solo's held prefix after another request");
    const auto busy=serve("BGEN 0 8 temperature=0 cvec=0 25,26\nGEN 4 temperature=0 cvec=1 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15\nBYIELD 0\n");
    check(busy.find("YIELDED ")==std::string::npos && busy.find("DONE 4 16 ")!=std::string::npos,"solo BYIELD overwrote an active slot");

    // Actual POSIX protocol + production Gen/SlotSession + streaming state IO.
    // The payload is a host fixture, not a native Model/hardware claim.
    const auto directory=std::filesystem::temp_directory_path()/("strata-hpu-batch-disk-"+std::to_string(::getpid()));
    std::filesystem::create_directory(directory);
    struct Cleanup {std::filesystem::path path;~Cleanup(){std::error_code error;std::filesystem::remove_all(path,error);}} cleanup{directory};
    const auto first=(directory/"main one.state").string(),second=(directory/"after slot.state").string(),third=(directory/"after refusal.state").string();
    const auto corrupt=(directory/"corrupt.state").string();{std::ofstream file(corrupt);file<<"bad state";}
    std::vector<std::vector<int32_t>> saved;std::vector<bool> saved_modes;uint32_t callbacks=0;
    control.session=[&](bool save,const std::string& path) {
        ++callbacks;
        if(path=="memory") throw AllocationBudgetExceeded();
        if(path=="badalloc") throw std::bad_alloc();
        if(path=="published") throw StateIoError("published\nwith space failure",true,true);
        if(path=="logic") throw std::logic_error("invalid disk fixture");
        if(path=="fatal") throw SessionFatal("post-transfer failure");
        main.release_checkpoints();
        if(save) {
            saved.push_back(fed);saved_modes.push_back(mode);
            StateWriter writer(path,65536);writer.integer(mode);writer.integer(fed.size());writer.vector<int32_t>(fed);writer.vector<float>(logits());writer.finish();
        } else {
            StateReader reader(path,65536);const auto enabled=reader.integer(),count=reader.integer();
            if(enabled>1 || count>128) throw std::invalid_argument("invalid disk fixture shape");
            auto ids=reader.vector<int32_t>(count);auto values=reader.vector<float>(count?32:0);reader.finish();
            main.adopt(ids,values,enabled!=0);fed=std::move(ids);mode=enabled!=0;
        }
        return ProtocolSessionResult{fed.size(),std::filesystem::file_size(path)};
    };
    main.reset();const auto empty=serve("SAVE "+first+"\n");
    check(empty.find("SERR invalid 0 no complete main session")!=std::string::npos && callbacks==0 && !std::filesystem::exists(first),"empty main SAVE reached a foreign snapshot producer");
    const auto disk=serve("GEN 1 temperature=0 cvec=0 1,2,3\nSAVE "+first+"\n"
        "BGEN 0 1 temperature=0 cvec=1 7,8,9\nSAVE "+second+"\n"
        "BGEN 1 1 temperature=0 cvec=1 10,11\nRESTORE "+first+"\n"
        "GEN 1 temperature=0 cvec=0 1,2,3,4\nRESTORE "+corrupt+"\nRESTORE "+(directory/"missing.state").string()+"\nSAVE "+third+"\n"
        "RESTORE memory\nRESTORE badalloc\nRESTORE published\nRESTORE logic\nSAVE \n"
        "BGEN 0 4 temperature=0 cvec=1 20,21\nSAVE active\nRESTORE active\n");
    check(saved==std::vector<std::vector<int32_t>>{{1,2,3},{1,2,3},{1,2,3,4}} && saved_modes==std::vector<bool>{false,false,false},"SAVE guessed the last idle slot or lost main after failed RESTORE");
    check(disk.find("RESTORED 3 ")!=std::string::npos && disk.find("RESUME 3")!=std::string::npos,"RESTORE did not adopt the complete main before the next GEN");
    const auto dl=split_lines(disk);const auto dc=[&](const std::string& p){return std::count_if(dl.begin(),dl.end(),[&](const auto& line){return starts_with(line,p);});};
    check(dc("SAVED ")==3 && dc("SERR memory 0 ")==2 && dc("SERR storage 1 published with space failure")==1 && dc("SERR invalid 0 ")==5 && dc("SERR io 0 ")==1,"batch disk error categories/publication/newline contract changed");
    check(callbacks==10 && dc("SERR invalid 0 session files: not while batch slots are decoding")==2,"active batch disk command reached a live state producer");
    check(dc("BDONE 0 4 length ")==1,"disk refusal disrupted the active slot's remaining decode budget");
    const auto fatal=serve("RESTORE fatal\nGEN 1 temperature=0 1,2,3\n");
    check(fatal.find("FATAL post-transfer failure")!=std::string::npos && fatal.find("\nT ")==std::string::npos,"fatal disk transfer continued protocol processing");
}
} // namespace

int main() {
    shared_main();
    cooperative_prompt();
    windows();
    mtp_two_row_window();
    int fds[2]{};
    if (::pipe(fds) != 0) {
        std::cout << "FAIL cannot create the protocol pipe\n";
        return 1;
    }
    FakeEngine engine(2);
    std::ostringstream output;
    std::jthread server([&] { serve_batch_protocol(fds[0], output, engine, 2, 256, 16); });

    auto send=[&](const std::string& line){const auto text=line+'\n';(void)!::write(fds[1],text.data(),text.size());};
    // ---- phase 1: two admissions, a stopped slot, a busy slot, a bad line and VRAM while decoding
    send("BGEN 0 5 5,6");   // slot 0, five tokens, prompt of two ids
    send("BGEN 1 2 7");     // slot 1, two tokens
    send("BSTOP 1");        // slot 1 ends at its next window, AFTER that window's token
    send("BGEN 0 3 1");     // slot 0 is busy
    send("BGEN 7 3 1");     // no such slot
    send("GEN 1 1");        // the single-request form is not a batch command
    send("VRAM");           // refused while slots decode
    // wait until the windows have advanced every active slot past its scripted end
    const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(10);
    while(engine.steps.load()<4 && std::chrono::steady_clock::now()<deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    send("QUIT");
    server.join();
    auto lines=split_lines(output.str());

    size_t cursor=0;
    expect(lines,cursor,"INFO backend=hpu logit_bias=0");
    if(!lines.empty() && lines[0].find("slots=2")==std::string::npos) {++failures;std::cout<<"FAIL the INFO line does not report the slot count\n";}
    if(!lines.empty() && lines[0].find("batch_schedule=time-sliced")==std::string::npos) {++failures;std::cout<<"FAIL the INFO line hides that the slots are time-sliced\n";}
    expect(lines,cursor,"READY 16 stop");
    // admission 0: the ordinary token + the admission's own DONE, then BADM
    expect(lines,cursor,"T 10");
    expect(lines,cursor,"DONE 1 2 ");
    if(cursor<lines.size() && lines[cursor-1].find("length")==std::string::npos) {++failures;std::cout<<"FAIL the admission DONE does not report length\n";}
    expect(lines,cursor,"BADM 0 1");
    // admission 1
    expect(lines,cursor,"T 11");
    expect(lines,cursor,"DONE 1 1 ");
    expect(lines,cursor,"BADM 1 1");
    // the refusals, in the order the commands arrived
    expect(lines,cursor,"ERR BGEN: no such free slot (--batch 2) or a bad max_new");
    expect(lines,cursor,"ERR BGEN: no such free slot (--batch 2) or a bad max_new");
    expect(lines,cursor,"ERR expected: GEN <max_new> <id,id,...> or BGEN <slot> <max_new> <id,id,...>");
    expect(lines,cursor,"ERR VRAM: not while batch slots are decoding");
    // window 1 serves both slots: slot 0's first token, then slot 1's - which the BSTOP closes as "cancel"
    expect(lines,cursor,"BT 0 100");
    expect(lines,cursor,"BT 1 110");
    expect(lines,cursor,"BDONE 1 2 cancel ");
    // window 2 and 3 serve the remaining slot: its third token is the scripted EOS -> "stop"
    expect(lines,cursor,"BT 0 101");
    expect(lines,cursor,"BT 0 102");
    expect(lines,cursor,"BDONE 0 4 stop ");
    if(cursor!=lines.size()) {
        ++failures;
        std::cout << "FAIL " << (lines.size()-cursor) << " unexpected trailing line(s), first: \"" << lines[cursor] << "\"\n";
    }
    check(engine.admits.load()==2,"exactly the two accepted BGENs reached the engine");
    check(engine.finishes.load()==2,"healthy stop/cancel must reach the idle-cache finish seam");
    ::close(fds[0]);
    ::close(fds[1]);
    boundaries();
    if(failures!=0) {std::cout<<"FAILED batch protocol checks: "<<failures<<"\n";return 1;}
    std::cout << "PASS batch protocol: BGEN answers with the admission's own T and DONE plus BADM, a continuing slot "
                 "emits BT per window and BDONE with upstream's finish names (one window's token then cancel for a "
                 "BSTOP, stop on EOS), a busy or absent slot is refused, the single-request GEN form is not a batch "
                 "command, and VRAM is refused while slots decode; no device\n";
    return 0;
}
