#pragma once
#include "strata/hpu/generation.hpp"
#include "strata/hpu/protocol_info.hpp"
#include <ostream>
#include <optional>
#include <array>
namespace strata::hpu {
struct ProtocolSessionResult {uint64_t tokens=0,bytes=0;};
// What GENI needs from the engine: whether --vision was given and the model width the SVE1 records must match.
struct VisionLoad {bool enabled=false; uint64_t width=0;};
GenerationRequest parse_generation_request(const std::string&,uint32_t vocabulary,uint32_t context,VisionLoad={});
struct ProtocolControl {
    std::string info;
    // Returns original key=value VRAM response fields, or throws a reason.
    std::function<std::string(std::optional<uint64_t>)> vram;
    std::function<ProtocolSessionResult(bool save,const std::string& path)> session;
    Generation* solo=nullptr; // same native Model; batch slots have owned images
    bool batch_mtp=false;     // the effective --batch-mtp: two rows per live slot in the window plan
    bool vision=false;        // --vision: GENI requests are accepted
    uint64_t vision_width=0;  // model n_embd the SVE1 record width must match
};
// Original Strata text protocol: READY, GEN, T, PP, RESUME, DONE, STOP, QUIT.
// Serial FIFO requests; QUIT is a barrier after earlier requests, and EOF
// drains accepted work. STOP targets only the current ordinary request/read,
// never a future request or all batch slots. Every started GEN has a terminal.
void serve_protocol(int input_fd,std::ostream&,Generation&,uint32_t vocabulary,uint32_t context,ProtocolControl={});

// ---- --batch: the same text protocol with slot commands (src/program/generate.cpp:566-568) -----------
// A request sent as `BGEN <slot> <max_new> [keys] <ids>` reads its prompt and its FIRST token through the
// ordinary token path (it answers with that `T <tok>` and its own `DONE`), then `BADM <slot> <1|0>` says
// whether the slot continues in the batch windows. A continuing slot emits `BT <slot> <tok>` per window and
// closes with `BDONE <slot> <produced> <finish> <ms>`; `BSTOP <slot>` ends it at its next window (after that
// window's token, finish "cancel"). A per-request logit_bias is refused in batch mode (:10716-10719), and VRAM
// is refused while any slot decodes (:10336). Pinned upstream refuses disk
// sessions for ALL --batch runs (:10446); HPU additionally permits them while
// idle, explicitly targeting control.solo rather than whichever slot last ran.
//
// The window MEMBERSHIP is upstream's (:9957-9981): every active slot from the cursor, two rows per slot while
// --batch-mtp is effective, at most kVerifyMaxT rows, with the cursor advancing only in that mode. The ENGINE
// still advances one slot at a time through its single Model, so this is a time-sliced schedule with NO row
// batching and no throughput claim.
class BatchSlotSession {
public:
    virtual ~BatchSlotSession()=default;
    struct Admission {
        uint32_t prompt_tokens=0;
        // Upstream continues only when the read produced exactly one token, max_new > 1, the finish is
        // "length" and the live ids are the whole prompt (:12944-12947).
        bool continues=false;
        uint32_t reused_tokens=0;
        bool eos=false; // first emitted token, before a slot is admitted
        int yielded_slot=-1;
        uint32_t yielded_tokens=0; // no output token; owned partial prompt
        bool nonfinite=false; // no admission token was published
        bool cancelled=false; // may contain the already published first token
        uint32_t prompt_read=0; // committed NEW prompt cells, excludes reused
    };
    // Runs the request's prompt and its FIRST token; `token` receives that token so the protocol can answer
    // `T <tok>` exactly like the ordinary request path. max_new is the slot's own budget from BGEN.
    virtual Admission admit(int slot,const GenerationRequest& request,uint32_t max_new,
                            const std::function<void(int32_t)>& token)=0;
    struct ReadControl {
        // Called only at committed prompt piece boundaries. Consumes queued
        // BSTOPs as well; an out-of-range yield target is never cast to int.
        std::function<std::optional<int>()> yield;
        std::function<bool()> has_work;
        std::function<void(double prompt_ms)> between;
        std::function<bool()> cancelled; // current ordinary STOP, not BSTOP
        // May be called under Model's lock; output/metadata only, no reentry.
        std::function<void(uint32_t,uint32_t,double)> progress;
    };
    virtual Admission admit_cooperative(int slot,const GenerationRequest& request,uint32_t max_new,
        const std::function<void(int32_t)>& token,const ReadControl&) {return admit(slot,request,max_new,token);}
    virtual bool park_partial(int,const GenerationRead&) {return false;}
    struct Step {
        int32_t token=0;
        bool eos=false;
        bool nonfinite=false; // a row with non-finite logits ends the slot emitting nothing (:10060-10071)
    };
    virtual Step step(int slot)=0;
    // A fixed two-row envelope; the producer MUST obey the hard bound before
    // committing. Stop/length/context limits cannot truncate committed work.
    struct Window {std::array<Step,2> rows{};uint32_t count=0;};
    virtual Window step_window(int slot,uint32_t limit) {
        if(!limit || limit>2) throw std::invalid_argument("invalid slot window limit");
        Window result;result.rows[0]=step(slot);result.count=1;return result;
    }
    virtual void stop(int slot)=0;     // BSTOP for an active slot; its next window still emits that token
    virtual void release(int slot)=0;  // the slot is free again: drop whatever it parked
    // Ordinary completion may keep an idle conversation image. Errors always
    // use release(), so an invalid or half-written state cannot become cache.
    virtual void finish(int slot) {release(slot);}
    // #879: drop idle/partial caches next request; running owners finish but
    // cannot become idle cache afterward. Engines with caches override this.
    virtual void invalidate_caches() {}
};
void serve_batch_protocol(int input_fd,std::ostream&,BatchSlotSession&,int slots,uint32_t vocabulary,
                          uint32_t context,ProtocolControl={});
}
