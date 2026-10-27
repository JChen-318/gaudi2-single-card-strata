// Implementation of the --batch admission laws and per-slot state machine; every rule cites the pinned
// src/program/generate.cpp lines it was transcribed from (see the header).
#include "strata/hpu/batch_slots.hpp"

#include <algorithm>
#include <cstdlib>
#include <stdexcept>

namespace strata::hpu {

BatchResolution resolve_batch(const BatchOptions& options) {
    BatchResolution result;
    result.slots = std::max(options.batch, 0);
    result.groups = std::max(options.groups, 1);

    // ---- --batch-mtp (:4493-4514): the gates are said and the flag is left OFF, never forced on.
    const char* why = result.slots < 2 ? "it needs --batch 2 or more"
                    : !options.mtp     ? "it needs --mtp"
                    : options.spec < 2 ? "it needs --spec T (T >= 2)"
                    : options.split_same ? "it needs each stage of a layer split on its own GPU"
                    : !options.serve   ? "it needs --serve"
                                       : nullptr;
    if (options.batch_mtp) {
        if (why == nullptr && options.has_layer_split && options.groups > 1)
            why = "it does not combine with --batch-groups (pipelined slot groups) yet";
        if (why == nullptr) result.batch_mtp = true;
        else {
            result.batch_mtp_off_reason = why;
            result.warnings.push_back(std::string("WARNING: --batch-mtp is off: ") + why);
        }
    }

    // ---- --batch (:4516-4527).
    if (result.slots != 0) {
        const char* off = !options.serve ? "it needs --serve"
                        : result.slots < 2 ? "it needs 2 or more slots"
                        : options.split_same ? "it needs each stage of a layer split on its own GPU"
                                            : nullptr;
        if (off != nullptr) {
            result.warnings.push_back("WARNING: --batch " + std::to_string(result.slots) + " is off: " + off);
            result.slots = 0;
        } else if (result.slots > options.cap && !result.batch_mtp) {
            result.warnings.push_back("WARNING: --batch " + std::to_string(result.slots) +
                                      ": a batch window holds at most " + std::to_string(options.cap) +
                                      " rows, so " + std::to_string(options.cap) + " slots");
            result.slots = options.cap;
        }
    }

    // ---- --batch-groups (:4549-4553): a layer split is required and the groups must divide the slots.
    if (result.slots > 0 && result.groups > 1 &&
        (!options.has_layer_split || result.slots % result.groups != 0)) {
        result.warnings.push_back("WARNING: --batch-groups " + std::to_string(result.groups) +
                                  " needs a layer split and to divide --batch " + std::to_string(result.slots) +
                                  "; one group");
        result.groups = 1;
    }
    // On one card there are no pipelined groups: a group count above one is refused above, and "auto" has
    // exactly one stage to hand out, so it resolves to one group (declared divergence: upstream's auto spreads
    // one group per split stage).
    if (result.slots > 0 && options.groups_auto && !options.has_layer_split) result.groups = 1;
    return result;
}

namespace {
// Upstream's `ints` lambda (:2365-2375): every comma-separated field must be non-empty and made only of digits,
// and the list must not come out empty. It appends as upstream's does, so a half-parsed list stays visible.
template <typename V> bool ints(const std::string& text, std::vector<V>& out) {
    size_t a = 0;
    while (a < text.size()) {
        size_t b = text.find(',', a);
        if (b == std::string::npos) b = text.size();
        const std::string field = text.substr(a, b - a);
        if (field.empty() || field.find_first_not_of("0123456789") != std::string::npos) return false;
        out.push_back(static_cast<V>(std::atoll(field.c_str())));
        a = b + 1;
    }
    return !out.empty();
}
}

LayerSplit resolve_layer_split(const LayerSplitRequest& request) {
    LayerSplit split;
    split.given = !request.spec.empty();
    if (!split.given) return split;
    split.automatic = request.spec == "auto";
    // Upstream asks the runtime how many devices exist and clamps a failed query to one (:2362-2364); a
    // non-positive answer is meaningless, so the same clamp is applied here.
    const int devices = request.device_count > 1 ? request.device_count : 1;
    bool ok = request.serve && (split.automatic || ints(request.spec, split.at));
    if (ok && !request.devices.empty()) ok = ints(request.devices, split.device_ids);
    else if (ok)
        for (int d = 1; d < devices && (split.automatic || split.device_ids.size() < split.at.size()); ++d)
            split.device_ids.push_back(d);
    // One boundary with no --split-device is the same-GPU form upstream still accepts (:2380).
    if (ok && !split.automatic && split.device_ids.empty() && split.at.size() == 1) split.device_ids.push_back(0);
    split.same_gpu = ok && split.device_ids.size() == 1 && split.device_ids[0] == 0 && !split.automatic;
    if (split.same_gpu && request.parking) {   // :2382-2386 - parking cannot follow a split that shares a card
        split.fatal = true;
        split.message = "conversation parking does not support a layer split on one GPU (--split-device 0); "
                        "disable parking with --conversation-cache-mib 0";
        return split;
    }
    if (ok && split.automatic && split.device_ids.empty()) {
        split.dropped = true;                 // :2388-2391: cleared, and the run continues unsplitted
        split.automatic = false;
        split.message = "--layer-split auto: one GPU visible, so no split";
    } else if (ok) {
        ok = (split.automatic || split.at.size() == split.device_ids.size()) &&
             split.device_ids.size() < static_cast<size_t>(kSplitDriveMax);
        for (size_t i = 0; ok && i < split.at.size(); ++i)
            ok = split.at[i] >= 2 && (i == 0 || split.at[i] > split.at[i - 1]);
        for (size_t i = 0; ok && !split.same_gpu && i < split.device_ids.size(); ++i) {
            ok = split.device_ids[i] > 0 && split.device_ids[i] < devices;
            for (size_t j = 0; ok && j < i; ++j) ok = split.device_ids[i] != split.device_ids[j];
        }
    }
    if (!ok) {                                 // :2398-2405, upstream's own wording
        split.fatal = true;
        split.message = "--layer-split K[,K2..]|auto needs --serve, rising K from 2, and one distinct GPU per K in "
                        "--split-device (1.." + std::to_string(devices - 1) +
                        "; or 0 with one K: the same GPU). K is the FIRST LAYER of each later GPU, not a count of "
                        "layers per card: \"24,36,42\" for 4 GPUs, not \"24,12,6,6\" (got \"" + request.spec + "\")";
        return split;
    }
    split.accepted = !split.dropped;
    return split;
}

const char* batch_finish_name(const BatchFinish finish) {
    switch (finish) {
        case BatchFinish::Stop: return "stop";
        case BatchFinish::Length: return "length";
        case BatchFinish::Cancel: return "cancel";
        case BatchFinish::NonFinite: return "nonfinite";
    }
    return "length";
}

BatchSlots::BatchSlots(const int slots, const uint32_t context) : slots_(size_t(std::max(slots, 0))), context_(context) {
    if (slots < 1 || slots > 64 || context == 0) throw std::invalid_argument("invalid batch slot geometry");
}

const BatchSlotInfo& BatchSlots::slot_at(const int slot) const {
    if (slot < 0 || slot >= size()) throw std::out_of_range("batch slot out of range");
    return slots_[size_t(slot)];
}
BatchSlotInfo& BatchSlots::slot_at(const int slot) {
    if (slot < 0 || slot >= size()) throw std::out_of_range("batch slot out of range");
    return slots_[size_t(slot)];
}
const BatchSlotInfo& BatchSlots::slot(const int slot) const { return slot_at(slot); }

void BatchSlots::admit(const int slot, const uint32_t max_new, const int64_t prompt_tokens, const int32_t first_token) {
    auto& info = slot_at(slot);
    // Upstream's BGEN refusal (:10355): an active slot, a bad index or max_new < 1.
    if (info.active || max_new < 1 || prompt_tokens < 1 || prompt_tokens > int64_t(context_))
        throw std::invalid_argument("no such free slot or a bad max_new/geometry");
    info = BatchSlotInfo{};
    info.active = true;
    info.max_new = max_new;
    info.produced = 1;                 // the token the admission produced through the ordinary path
    info.token = first_token;          // the next window feeds it (:12956-12963)
    info.position = prompt_tokens;     // its position is the prompt's length
}

void BatchSlots::stop(const int slot) {
    auto& info = slot_at(slot);
    if (!info.active) return; // a BSTOP for a free slot is a no-op upstream
    info.stop = true;
}

std::vector<BatchSlots::WindowRow> BatchSlots::plan_window(const bool batch_mtp, const int cap) {
    std::vector<WindowRow> rows;
    const int limit = std::max(cap, 0);
    const int per_slot = batch_mtp ? 2 : 1;
    if (size() == 0 || per_slot > limit) return rows;
    int last_active = -1;
    for (int offset = 0; offset < size() && int(rows.size()) + per_slot <= limit; ++offset) {
        const int candidate = int((cursor_ + size_t(offset)) % size_t(size()));
        const auto& info = slots_[size_t(candidate)];
        if (!info.active) continue;
        last_active = candidate;
        rows.push_back({candidate, false, info.position});
        if (batch_mtp) rows.push_back({candidate, true, info.position + 1});
    }
    if (batch_mtp && last_active >= 0) cursor_ = size_t((last_active + 1) % size());
    return rows;
}

std::vector<BatchSlots::Event> BatchSlots::complete(const int slot, const int32_t token, const bool eos) {
    auto& info = slot_at(slot);
    if (!info.active) throw std::logic_error("completing a free batch slot");
    std::vector<Event> events;
    Event emitted;
    emitted.kind = Event::Kind::Token;
    emitted.slot = slot;
    emitted.token = token;
    events.push_back(emitted);
    ++info.produced;
    // (:10078-10079) the FIRST of these decides: eos, then BSTOP's cancel - which is AFTER this token - then
    // produced >= max_new, then the context bound.
    const bool context_full = info.position + 2 > int64_t(context_);
    const BatchFinish* finish = nullptr;
    BatchFinish chosen = BatchFinish::Length;
    if (eos) { chosen = BatchFinish::Stop; finish = &chosen; }
    else if (info.stop) { chosen = BatchFinish::Cancel; finish = &chosen; }
    else if (info.produced >= info.max_new || context_full) { chosen = BatchFinish::Length; finish = &chosen; }
    if (finish != nullptr) {
        Event done;
        done.kind = Event::Kind::Done;
        done.slot = slot;
        done.produced = info.produced;
        done.finish = *finish;
        events.push_back(done);
        info.active = false;
        info.stop = false;
        return events;
    }
    info.token = token;
    ++info.position;
    return events;
}

std::vector<BatchSlots::Event> BatchSlots::fail(const int slot) {
    auto& info = slot_at(slot);
    if (!info.active) throw std::logic_error("failing a free batch slot");
    std::vector<Event> events;
    Event done;
    done.kind = Event::Kind::Done;
    done.slot = slot;
    done.produced = info.produced;
    done.finish = BatchFinish::NonFinite;
    events.push_back(done);
    info.active = false;
    info.stop = false;
    return events;
}
} // namespace strata::hpu
