#pragma once
// hpu/include/strata/hpu/batch_slots.hpp - upstream's `--batch` slots for this single-card backend, as one pure
// unit: the ADMISSION laws and the per-slot state machine the batch windows drive.
//
// AUTHORITIES (pinned source, commit 296b59d, src/program/generate.cpp unless noted):
//   566-578    the option field and its documentation, including the protocol grammar: a request sent as
//              `BGEN <slot> <max_new> ...` reads its prompt and its first token through the usual path
//              (`T <tok>`), answers `BADM <slot> 1`, and then continues in that slot of the batch windows
//              (`BT <slot> <id>` lines, then `BDONE <slot> ...`); `BSTOP <slot>` ends a slot at its next window.
//   4493-4514  --batch-mtp's gate: needs --batch >= 2, --mtp, --spec T >= 2, a layer split whose stages are on
//              their own GPUs, and --serve; when it cannot run it is said ("WARNING: --batch-mtp is off: ...")
//              and left OFF, plain batching still starting.
//   4516-4527  --batch's own gate: needs --serve, at least 2 slots, and each stage of a layer split on its own
//              GPU; above the window's row cap (kVerifyMaxT) it is clamped to the cap unless --batch-mtp is on.
//   4549-4553  --batch-groups needs a layer split and must divide --batch; otherwise one group, said once.
//   10716-10719 logit_bias is refused in batch mode ("continuous batching is not supported").
//   12944-12986 admission: the slot takes over only when the read produced exactly one token, max_new > 1, the
//              finish is "length" and the live ids are the whole prompt; it keeps the token just produced as
//              the token its first window feeds, produced = 1, and the prompt's length as that token's position.
//   10055-10089 the window's per-slot result: a non-finite row ends the slot with NOTHING emitted
//              (BDONE ... nonfinite); otherwise `BT <slot> <tok>` and BDONE with the first of
//              eos -> "stop", BSTOP -> "cancel" (AFTER that token), produced >= max_new -> "length",
//              position + 2 > context -> "length".
//
// This header must not reach the Synapse headers: it holds no device resource, so a host test can pin every law
// above. The window MEMBERSHIP is upstream's (:9957-9981): two rows per live slot when --batch-mtp is effective,
// at most kVerifyMaxT rows, and the cursor advances only in that mode. The execution is still one slot at a
// time through this backend's single Model, so the schedule says so and makes no throughput claim (declared in
// hpu/PROGRESS.md and in the tool's INFO line).
#include "strata/kernels/verify_kernels.hpp"
#include <cstdint>
#include <string>
#include <vector>

namespace strata::hpu {

// ---- admission ---------------------------------------------------------------------------------------
struct BatchOptions {
    bool serve = false;            // --serve
    int batch = 0;                 // --batch / --slots (0 = off)
    int groups = 1;                // --batch-groups (as given; 1 = one window)
    bool groups_set = false;       // --batch-groups was given
    bool groups_auto = false;      // "--batch-groups auto"
    bool batch_mtp = false;        // --batch-mtp or STRATA_BATCH_MTP=1
    bool has_layer_split = false;  // a --layer-split is configured (stages non-empty)
    bool split_same = false;       // ... and two stages share one GPU
    bool mtp = false;              // --mtp given
    int spec = 0;                  // --spec T
    int cap = strata::kernels::kVerifyMaxT;   // rows per window (:4519)
};

struct BatchResolution {
    int slots = 0;                  // effective slot count (0 = --batch off)
    int groups = 1;                 // effective group count
    bool batch_mtp = false;         // effective --batch-mtp
    std::vector<std::string> warnings;      // upstream's own WARNING lines, in order
    std::string batch_mtp_off_reason;       // why --batch-mtp stayed off (empty when it is on)
    // Upstream refuses a per-request logit_bias in batch mode (:10716-10719): the batched rows have no
    // per-row bias plumbing.
    bool logit_bias_allowed() const { return slots == 0; }
};
BatchResolution resolve_batch(const BatchOptions&);

// ---- --layer-split / --split-device admission (:2355-2406) -------------------------------------------
// Upstream reads the device count from the runtime (`cudaGetDeviceCount`, :2362-2364); this unit takes it as an
// argument so a host test can pin the law for 1, 2, 4 and 8 cards without a device. SplitDrive::kMax caps the
// stage count (:1052-1053) and is named here rather than reached through a device header.
inline constexpr int kSplitDriveMax = 8;   ///< src/program/generate.cpp:1053

struct LayerSplitRequest {
    std::string spec;       // --layer-split: "auto" or "K[,K2..]"; empty means the flag was not given
    std::string devices;    // --split-device: "D[,D2..]"; empty means not given
    bool serve = false;     // upstream accepts a split only under --serve (:2376)
    int device_count = 1;   // the visible device count; this backend has one Gaudi2 card
    bool parking = false;   // conversation parking is on (:2382's three-part condition)
};

struct LayerSplit {
    bool given = false;      // --layer-split was passed
    bool accepted = false;   // a split the run applies (multi-stage, or the same-GPU form)
    bool automatic = false;  // "auto" survived
    bool same_gpu = false;   // upstream's split_same: every stage on device 0 (:2380)
    bool dropped = false;    // upstream cleared it - "auto" with one GPU visible (:2388-2391)
    bool fatal = false;      // upstream prints and exits 2 (:2382-2386, :2398-2405)
    std::vector<int64_t> at;             // first layer of each later stage (:2376)
    std::vector<int> device_ids;         // one device per stage after the first (:2377-2379)
    std::string message;                 // upstream's own text; empty when there is nothing to say
};
// The law, transcribed line for line: parse, fill devices, recognise the one-GPU form, refuse a same-GPU split
// that would park conversations, clear "auto" when there is nothing to split, then validate the stage boundaries
// and the distinctness/range of every device id.
LayerSplit resolve_layer_split(const LayerSplitRequest&);

// ---- per-slot state machine --------------------------------------------------------------------------
enum class BatchFinish { Stop, Length, Cancel, NonFinite };
const char* batch_finish_name(BatchFinish); // "stop" | "length" | "cancel" | "nonfinite"

struct BatchSlotInfo {
    bool active = false;
    bool stop = false;         // BSTOP seen: ends at the next window, after that window's token
    int32_t token = 0;         // the token the next window feeds (upstream's sl.x)
    int64_t position = 0;      // that token's position (upstream's sl.p)
    uint32_t produced = 0;     // tokens emitted for this request, including the admitted first one
    uint32_t max_new = 0;
};

class BatchSlots {
public:
    BatchSlots(int slots, uint32_t context);

    int size() const { return int(slots_.size()); }
    bool free(int slot) const { return !slot_at(slot).active; }
    bool active(int slot) const { return slot_at(slot).active; }
    const BatchSlotInfo& slot(int slot) const;

    // Admission after the request's prompt and FIRST token went through the token path. `prompt_tokens` is the
    // committed prompt length; upstream stores it as the first token's position. Refuses an active/bad slot or
    // max_new < 1, exactly the conditions its BGEN handler answers with an ERR.
    void admit(int slot, uint32_t max_new, int64_t prompt_tokens, int32_t first_token);
    void stop(int slot);   // BSTOP

    // One row of the next batch window (:9953-9956): `slot` and whether it is the pending token or its draft.
    // `position` is the stored token's position; the draft row is one past it.
    struct WindowRow { int slot = 0; bool draft = false; int64_t position = 0; };
    // Upstream's window plan (:9957-9981): start at the cursor and take every active slot, one row per slot
    // (two when --batch-mtp is effective), stopping before a slot that would not fit `cap` rows. Only the
    // two-row mode advances the cursor, to the slot after the last one served.
    std::vector<WindowRow> plan_window(bool batch_mtp, int cap);
    // After the window: `token` is the slot's row output and `eos` says whether it is a stop token.
    struct Event {
        enum class Kind { Token, Done };
        Kind kind = Kind::Token;
        int slot = 0;
        int32_t token = 0;
        uint32_t produced = 0;
        BatchFinish finish = BatchFinish::Length;
    };
    std::vector<Event> complete(int slot, int32_t token, bool eos);
    // A window whose row for this slot was not finite: BDONE ... nonfinite, nothing emitted (:10060-10071).
    std::vector<Event> fail(int slot);

private:
    const BatchSlotInfo& slot_at(int slot) const;
    BatchSlotInfo& slot_at(int slot);
    std::vector<BatchSlotInfo> slots_;
    uint32_t context_ = 0;
    size_t cursor_ = 0;
};
} // namespace strata::hpu
