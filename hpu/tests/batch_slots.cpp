// hpu/tests/batch_slots.cpp - host evidence for --batch's admission laws and slot state machine.
//
// The rules are transcribed from the pinned src/program/generate.cpp (see strata/hpu/batch_slots.hpp for the
// line numbers). The test compares the port against an INDEPENDENT transcription written from those lines, over
// a sweep of the option combinations, and then drives the slot machine through the window outcomes upstream
// documents: a token then BDONE with the first of eos -> "stop", BSTOP -> "cancel" (after that token),
// produced >= max_new -> "length" and the context bound, plus the non-finite row that emits NOTHING.
#include "strata/hpu/batch_slots.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <tuple>
#include <vector>

namespace {
using strata::hpu::BatchFinish;
using strata::hpu::BatchOptions;
using strata::hpu::BatchResolution;
using strata::hpu::BatchSlots;

int failures = 0;
void check(const bool ok, const std::string& what) {
    if (!ok) {
        ++failures;
        std::cout << "FAIL " << what << "\n";
    }
}
template <class F>
bool refused(F&& call) {
    try {
        call();
    } catch (const std::exception&) {
        return true;
    }
    return false;
}

// ---- the independent transcription of upstream's admission block (:4493-4553) -------------------------
struct Expected {
    int slots = 0;
    int groups = 1;
    bool batch_mtp = false;
    bool warned_batch_mtp = false;
    bool warned_batch_off = false;
    bool warned_clamped = false;
    bool warned_groups = false;
};
Expected upstream(const BatchOptions& o) {
    Expected expected;
    expected.slots = std::max(o.batch, 0);
    expected.groups = std::max(o.groups, 1);
    // --batch-mtp (:4493-4514)
    const char* why = expected.slots < 2 ? "batch"
                    : !o.mtp ? "mtp"
                    : o.spec < 2 ? "spec"
                    : o.split_same ? "split"
                    : !o.serve ? "serve"
                               : nullptr;
    if (o.batch_mtp) {
        if (why == nullptr && o.has_layer_split && expected.groups > 1) why = "groups";
        if (why == nullptr) expected.batch_mtp = true;
        else expected.warned_batch_mtp = true;
    }
    // --batch (:4516-4527)
    if (expected.slots != 0) {
        const char* off = !o.serve ? "serve" : expected.slots < 2 ? "slots" : o.split_same ? "split" : nullptr;
        if (off != nullptr) {
            expected.warned_batch_off = true;
            expected.slots = 0;
        } else if (expected.slots > o.cap && !expected.batch_mtp) {
            expected.warned_clamped = true;
            expected.slots = o.cap;
        }
    }
    // --batch-groups (:4549-4553)
    if (expected.slots > 0 && expected.groups > 1 &&
        (!o.has_layer_split || expected.slots % expected.groups != 0)) {
        expected.warned_groups = true;
        expected.groups = 1;
    }
    return expected;
}

bool same(const BatchResolution& got, const Expected& want) {
    bool ok = got.slots == want.slots && got.groups == want.groups && got.batch_mtp == want.batch_mtp;
    auto has = [&got](const std::string& needle) {
        return std::any_of(got.warnings.begin(), got.warnings.end(),
                           [&needle](const std::string& w) { return w.find(needle) != std::string::npos; });
    };
    ok = ok && has("--batch-mtp is off") == want.warned_batch_mtp;
    // The --batch gate's line is "WARNING: --batch N is off: ..." and the --batch-mtp line also says "is off:",
    // so the two are told apart by the flag name in front of it.
    const bool batch_off_line = std::any_of(got.warnings.begin(), got.warnings.end(), [](const std::string& w) {
        return w.find("is off:") != std::string::npos && w.find("--batch-mtp") == std::string::npos;
    });
    ok = ok && batch_off_line == want.warned_batch_off;
    ok = ok && has("holds at most") == want.warned_clamped;
    ok = ok && has("needs a layer split and to divide") == want.warned_groups;
    return ok;
}

void admission_sweep() {
    int combinations = 0;
    for (const bool serve : {false, true})
        for (const int batch : {-3, 0, 1, 2, 3, 8, 9, 16})
            for (const int groups : {0, 1, 2, 4})
                for (const bool groups_set : {false, true})
                    for (const bool auto_groups : {false, true})
                        for (const bool mtp_flag : {false, true})
                            for (const bool has_split : {false, true})
                                for (const bool same_gpu : {false, true})
                                    for (const bool mtp_on : {false, true})
                                        for (const int spec : {0, 2}) {
                                            BatchOptions o;
                                            o.serve = serve;
                                            o.batch = batch;
                                            o.groups = groups;
                                            o.groups_set = groups_set;
                                            o.groups_auto = auto_groups;
                                            o.batch_mtp = mtp_flag;
                                            o.has_layer_split = has_split;
                                            o.split_same = same_gpu;
                                            o.mtp = mtp_on;
                                            o.spec = spec;
                                            o.cap = strata::kernels::kVerifyMaxT;
                                            ++combinations;
                                            const auto got = strata::hpu::resolve_batch(o);
                                            const auto want = upstream(o);
                                            if (!same(got, want)) {
                                                ++failures;
                                                std::cout << "FAIL admission case serve=" << serve << " batch=" << batch
                                                          << " groups=" << groups << " mtp_flag=" << mtp_flag
                                                          << " split=" << has_split << " same=" << same_gpu
                                                          << " mtp=" << mtp_on << " spec=" << spec
                                                          << ": got slots=" << got.slots << " groups=" << got.groups
                                                          << " mtp=" << got.batch_mtp << " warnings=" << got.warnings.size()
                                                          << " want slots=" << want.slots << " groups=" << want.groups
                                                          << " mtp=" << want.batch_mtp << "\n";
                                                if (failures > 6) return;
                                            }
                                        }
    check(combinations == 2 * 8 * 4 * 2 * 2 * 2 * 2 * 2 * 2 * 2, "the sweep covers every combination it claims");
    check(combinations > 5000, "the sweep is wide enough to be evidence (" + std::to_string(combinations) + " cases)");

    // the single-card shape the tool actually runs: serve, no layer split
    BatchOptions one_card;
    one_card.serve = true;
    one_card.batch = 4;
    one_card.groups = 4;          // given, but there is no layer split to pipeline
    one_card.groups_set = true;
    const auto resolved = strata::hpu::resolve_batch(one_card);
    check(resolved.slots == 4 && resolved.groups == 1, "a single card keeps 4 slots and collapses --batch-groups to 1");
    check(any_of(resolved.warnings.begin(), resolved.warnings.end(),
                 [](const std::string& w) { return w.find("needs a layer split") != std::string::npos; }),
          "the collapsed group count is reported with upstream's warning");
    check(!resolved.logit_bias_allowed(), "logit_bias is refused while batch slots are on (:10716-10719)");
    check(strata::hpu::resolve_batch(BatchOptions{}).logit_bias_allowed(), "with --batch off a bias is allowed again");
}

void slot_machine() {
    // ---- admission refusals and the admitted state
    BatchSlots slots(2, 16);
    check(slots.free(0) && slots.free(1), "two fresh slots are free");
    check(refused([&] { slots.admit(2, 4, 3, 7); }), "a slot index outside the batch is refused");
    check(refused([&] { slots.admit(0, 0, 3, 7); }), "max_new 0 is refused (upstream's BGEN law)");
    check(refused([&] { slots.admit(0, 4, 0, 7); }), "an empty prompt has no first token to store");
    slots.admit(0, 4, 3, 7);
    const auto& admitted = slots.slot(0);
    check(admitted.active && admitted.produced == 1 && admitted.token == 7 && admitted.position == 3,
          "admission keeps the produced token and the prompt length as its position (:12956-12963)");
    check(refused([&] { slots.admit(0, 4, 3, 8); }), "an active slot cannot be admitted again");

    // ---- window rows: every active slot once, in cursor order while the cap holds them all
    slots.admit(1, 4, 5, 9);
    const auto slots_of = [](const auto& rows) {
        std::vector<int> ids;
        for (const auto& row : rows) ids.push_back(row.slot);
        return ids;
    };
    check((slots_of(slots.plan_window(false, 8)) == std::vector<int>{0, 1}), "one window holds both active slots");
    check((slots_of(slots.plan_window(false, 1)) == std::vector<int>{0}), "a cap of one row starts at the lowest active slot");

    // ---- length: produced reaches max_new (admitted 1, so three more tokens end a max_new=4 slot)
    auto events = slots.complete(0, 11, false);
    check(events.size() == 1 && events[0].kind == BatchSlots::Event::Kind::Token && events[0].token == 11,
          "a live window emits BT and no BDONE");
    check(slots.slot(0).token == 11 && slots.slot(0).position == 4, "the emitted token becomes the next window's input");
    events = slots.complete(0, 12, false);
    check(events.size() == 1 && slots.slot(0).produced == 3, "two more tokens leave the slot active");
    events = slots.complete(0, 13, false);
    check(events.size() == 2 && events[1].kind == BatchSlots::Event::Kind::Done &&
              events[1].finish == BatchFinish::Length && events[1].produced == 4,
          "produced == max_new closes the slot with BDONE length AFTER emitting its last token");
    check(slots.free(0), "a finished slot is free again");

    // ---- EOS wins over a pending BSTOP, and BSTOP still emits one token first (:10072-10079)
    slots.admit(0, 6, 2, 21);
    slots.stop(0);
    events = slots.complete(0, 22, false);
    check(events.size() == 2 && events[0].kind == BatchSlots::Event::Kind::Token &&
              events[1].finish == BatchFinish::Cancel,
          "BSTOP ends the slot at its next window, after that window's token (cancel)");
    slots.admit(0, 6, 2, 21);
    slots.stop(0);
    events = slots.complete(0, 22, true);
    check(events.size() == 2 && events[1].finish == BatchFinish::Stop, "an EOS token reports stop even while stopping");
    slots.admit(0, 6, 2, 21);
    slots.stop(0);
    slots.stop(0); // idempotent
    check(slots.slot(0).stop, "a repeated BSTOP leaves the flag set");
    slots.complete(0, 23, false);

    // ---- the context bound: position + 2 > context closes with length (:10079)
    BatchSlots tight(1, 6);
    tight.admit(0, 100, 5, 1);   // position 5 with a context of 6: 5 + 2 > 6
    events = tight.complete(0, 2, false);
    check(events.size() == 2 && events[1].finish == BatchFinish::Length,
          "the context bound ends the slot with length instead of overrunning it");
    BatchSlots roomy(1, 6);
    roomy.admit(0, 100, 4, 1);   // position 4: 4 + 2 > 6 is false, so one more token fits
    events = roomy.complete(0, 2, false);
    check(events.size() == 1 && roomy.slot(0).position == 5, "a slot with room left keeps decoding");

    // ---- a non-finite row emits NOTHING and ends the slot (:10060-10071)
    BatchSlots bad(1, 16);
    bad.admit(0, 4, 2, 5);
    events = bad.fail(0);
    check(events.size() == 1 && events[0].kind == BatchSlots::Event::Kind::Done &&
              events[0].finish == BatchFinish::NonFinite && events[0].produced == 1,
          "a non-finite window row closes the slot as nonfinite without a token");
    check(bad.free(0), "the failed slot is free again");

    // ---- a BSTOP for a free slot is a no-op, and the finish names are upstream's
    bad.stop(0);
    check(bad.free(0) && !bad.slot(0).stop, "BSTOP on a free slot does nothing");
    check(std::string(strata::hpu::batch_finish_name(BatchFinish::Stop)) == "stop" &&
              std::string(strata::hpu::batch_finish_name(BatchFinish::Length)) == "length" &&
              std::string(strata::hpu::batch_finish_name(BatchFinish::Cancel)) == "cancel" &&
              std::string(strata::hpu::batch_finish_name(BatchFinish::NonFinite)) == "nonfinite",
          "the BDONE finish names are upstream's four");

    // ---- the window membership and cursor of :9957-9981
    BatchSlots many(8, 64);
    for (int slot = 0; slot < 8; ++slot) many.admit(slot, 8, 2, 5);
    auto rows = many.plan_window(true, strata::kernels::kVerifyMaxT);
    check(rows.size() == 8 && slots_of(rows) == std::vector<int>({0, 0, 1, 1, 2, 2, 3, 3}),
          "--batch-mtp takes two rows per live slot, four slots per kVerifyMaxT window");
    check(rows.size() == 8 && rows[1].draft && rows[1].position == 3 && !rows[0].draft && rows[0].position == 2,
          "the draft row is one position past the stored token");
    rows = many.plan_window(true, strata::kernels::kVerifyMaxT);
    check(slots_of(rows) == std::vector<int>({4, 4, 5, 5, 6, 6, 7, 7}),
          "the two-row cursor continues after the last slot served");
    rows = many.plan_window(false, strata::kernels::kVerifyMaxT);
    check(slots_of(rows) == std::vector<int>({0, 1, 2, 3, 4, 5, 6, 7}),
          "one row per slot without --batch-mtp, with the cursor left where the two-row mode put it");
    BatchSlots half(8, 64);
    for (int slot = 0; slot < 8; ++slot) half.admit(slot, 8, 2, 5);
    check(slots_of(half.plan_window(true, 4)) == std::vector<int>({0, 0, 1, 1}) &&
              slots_of(half.plan_window(true, 4)) == std::vector<int>({2, 2, 3, 3}),
          "a four-row window rotates over two slots at a time");
}

// ---- the window plan against a second transcription of upstream's batch_step (:9957-9981) ------------
struct PlannedRow {
    int slot = 0;
    bool draft = false;
    int64_t position = 0;
    bool operator==(const PlannedRow&) const = default;
};
int plan_cases = 0;
std::vector<PlannedRow> transcribe_plan(const std::vector<bool>& active, const std::vector<int64_t>& positions,
                                        const bool batch_mtp, const int cap, size_t& next_slot) {
    std::vector<PlannedRow> rows;
    const int per_slot = batch_mtp ? 2 : 1;
    int S = 0;
    int A = 0;
    int last = -1;
    for (size_t offset = 0; offset < active.size() && S + per_slot <= cap; ++offset) {
        const int b = int((next_slot + offset) % active.size());
        if (!active[size_t(b)]) continue;
        last = b;
        rows.push_back({b, false, positions[size_t(b)]});
        ++S;
        if (batch_mtp) { rows.push_back({b, true, positions[size_t(b)] + 1}); ++S; }
        ++A;
    }
    if (batch_mtp && A > 0) next_slot = (size_t(last) + 1) % active.size();
    return rows;
}

void window_plan_sweep() {
    using strata::hpu::BatchSlots;
    for (int count = 1; count <= 9; ++count)
        for (int mask = 0; mask < (1 << count); ++mask)
            for (const bool batch_mtp : {false, true})
                for (const int cap : {1, 2, 4, 5, 8}) {
                    if (mask == 0) continue;
                    BatchSlots slots(count, 64);
                    std::vector<bool> active(size_t(count), false);
                    std::vector<int64_t> positions(size_t(count), 0);
                    for (int slot = 0; slot < count; ++slot) {
                        active[size_t(slot)] = (mask >> slot) & 1;
                        if (!active[size_t(slot)]) continue;
                        positions[size_t(slot)] = 2 + slot;
                        slots.admit(slot, 8, positions[size_t(slot)], 5);
                    }
                    size_t cursor = 0;
                    for (int window = 0; window < 4; ++window) {
                        const auto got = slots.plan_window(batch_mtp, cap);
                        const auto want = transcribe_plan(active, positions, batch_mtp, cap, cursor);
                        std::vector<PlannedRow> converted;
                        for (const auto& row : got) converted.push_back({row.slot, row.draft, row.position});
                        ++plan_cases;
                        if (!(converted == want)) {
                            ++failures;
                            std::cout << "FAIL window plan count=" << count << " mask=" << mask
                                      << " mtp=" << batch_mtp << " cap=" << cap << " window=" << window
                                      << ": got " << converted.size() << " rows, want " << want.size() << "\n";
                            if (failures > 6) return;
                        }
                    }
                }
    check(plan_cases == 0 || plan_cases > 1000, "the window-plan sweep is wide enough to be evidence (" +
          std::to_string(plan_cases) + " windows)");
}
// ---- --layer-split / --split-device -------------------------------------------------------------------
size_t split_cases = 0;   // reported through the PASS line, which must stay the first line of stdout
// A second transcription of generate.cpp:2355-2406, written again from the pinned source rather than derived
// from resolve_layer_split, so agreement is evidence about the transcription and not a tautology.
struct SplitShape {
    bool given = false, accepted = false, automatic = false, same_gpu = false, dropped = false, fatal = false;
    std::vector<int64_t> at;
    std::vector<int> devs;
};

SplitShape transcribe_split(const std::string& spec, const std::string& devices_in, bool serve, int n_dev,
                           bool parking) {
    auto digits = [](const std::string& text, auto& out) {
        size_t a = 0;
        while (a < text.size()) {
            size_t b = text.find(',', a);
            if (b == std::string::npos) b = text.size();
            const std::string field = text.substr(a, b - a);
            if (field.empty() || field.find_first_not_of("0123456789") != std::string::npos) return false;
            out.push_back(static_cast<typename std::decay_t<decltype(out)>::value_type>(
                std::atoll(field.c_str())));   // upstream's own decay form (:2366)
            a = b + 1;
        }
        return !out.empty();
    };
    SplitShape s;
    s.given = !spec.empty();
    if (!s.given) return s;
    bool aut = spec == "auto";
    bool ok = serve && (aut || digits(spec, s.at));
    if (ok && !devices_in.empty()) ok = digits(devices_in, s.devs);
    else if (ok) for (int d = 1; d < n_dev && (aut || s.devs.size() < s.at.size()); ++d) s.devs.push_back(d);
    if (ok && !aut && s.devs.empty() && s.at.size() == 1) s.devs.push_back(0);
    s.same_gpu = ok && s.devs.size() == 1 && s.devs[0] == 0 && !aut;
    if (s.same_gpu && parking) { s.fatal = true; s.automatic = aut; return s; }
    if (ok && aut && s.devs.empty()) { s.dropped = true; aut = false; }   // upstream clears split_auto (:2390)
    else if (ok) {
        ok = (aut || s.at.size() == s.devs.size()) && s.devs.size() < static_cast<size_t>(8);   // SplitDrive::kMax (:1053)
        for (size_t i = 0; ok && i < s.at.size(); ++i)
            ok = s.at[i] >= 2 && (i == 0 || s.at[i] > s.at[i - 1]);
        for (size_t i = 0; ok && !s.same_gpu && i < s.devs.size(); ++i) {
            ok = s.devs[i] > 0 && s.devs[i] < n_dev;
            for (size_t j = 0; ok && j < i; ++j) ok = s.devs[i] != s.devs[j];
        }
    }
    // split_auto keeps its value on the exit path upstream never reaches again, so the flag is carried here too.
    s.automatic = aut;
    if (!ok) { s.fatal = true; return s; }
    s.accepted = !s.dropped;
    return s;
}

void layer_split_sweep() {
    using strata::hpu::resolve_layer_split;
    using strata::hpu::LayerSplitRequest;
    const std::vector<std::string> specs = {"", "auto", "24", "24,36", "24,36,42", "1", "24,12", "24,,36", "x",
                                            "24,36,42,48,52,56,60,64", "24,36,42,48,52,56,60,64,68"};
    const std::vector<std::string> devices = {"", "0", "1", "1,2", "1,1", "3", "0,1"};
    const std::vector<int> counts = {1, 2, 4, 8};
    int cases = 0;
    for (const auto& spec : specs) for (const auto& dev : devices) for (const int n_dev : counts)
        for (int flags = 0; flags < 4; ++flags) {
            const bool serve = flags & 1, parking = flags & 2;
            LayerSplitRequest request;
            request.spec = spec; request.devices = dev; request.serve = serve;
            request.device_count = n_dev; request.parking = parking;
            const auto got = resolve_layer_split(request);
            const auto want = transcribe_split(spec, dev, serve, n_dev, parking);
            ++cases;
            const bool same_shape = got.given == want.given && got.accepted == want.accepted &&
                got.automatic == want.automatic && got.same_gpu == want.same_gpu && got.dropped == want.dropped &&
                got.fatal == want.fatal && got.at == want.at && got.device_ids == want.devs;
            check(same_shape, "the layer-split law equals the independent transcription over every flag shape");
            // a refusal always says why, and an accepted split never does
            check(got.fatal || got.dropped || got.message.empty(), "only a said case speaks");
            check(!got.fatal || !got.message.empty(), "a refusal carries upstream's message");
        }

    // the shapes a single Gaudi2 card can actually reach
    LayerSplitRequest one; one.serve = true; one.device_count = 1;
    one.spec = "auto";
    const auto dropped = resolve_layer_split(one);
    check(dropped.given && dropped.dropped && !dropped.accepted && !dropped.fatal &&
              dropped.message == "--layer-split auto: one GPU visible, so no split",
          "auto on one card clears with upstream's no-split line");
    one.spec = "24";
    const auto same = resolve_layer_split(one);
    check(same.accepted && same.same_gpu && !same.fatal, "one boundary with no --split-device is the same-GPU split");
    one.parking = true;
    const auto parked = resolve_layer_split(one);
    check(parked.fatal && parked.message.find("conversation parking does not support a layer split on one GPU") == 0,
          "a same-GPU split with parking is refused before any device work");
    one.parking = false; one.spec = "24,36";
    check(resolve_layer_split(one).fatal, "two boundaries need a device 1..0, which one card cannot give");
    one.spec = "24"; one.devices = "1";
    check(resolve_layer_split(one).fatal, "device 1 does not exist on one card");

    // the wiring the fields exist for: an accepted split reaches the --batch gates
    BatchOptions options; options.serve = true; options.batch = 4; options.groups = 2; options.groups_set = true;
    const auto without_split = strata::hpu::resolve_batch(options);
    check(without_split.groups == 1 &&
          std::any_of(without_split.warnings.begin(), without_split.warnings.end(),
                      [](const std::string& w) { return w.find("needs a layer split") != std::string::npos; }),
          "--batch-groups without a split collapses and says so");
    options.has_layer_split = true; options.split_same = false;
    const auto with_split = strata::hpu::resolve_batch(options);
    check(with_split.groups == 2 && with_split.slots == 4 &&
          without_split.warnings.size() == with_split.warnings.size() + 1,
          "a real split (multi-device shape) makes the group law apply instead of collapse");
    options.split_same = true;
    const auto same_card_split = strata::hpu::resolve_batch(options);
    check(same_card_split.slots == 0 &&
          std::any_of(same_card_split.warnings.begin(), same_card_split.warnings.end(),
                      [](const std::string& w) { return w.find("each stage of a layer split on its own GPU") !=
                                                     std::string::npos; }),
          "a same-GPU split turns --batch off with upstream's reason");
    // The recorder reads the FIRST line of stdout as the verdict, so the sweep reports through the PASS line
    // instead of printing its own summary first.
    split_cases = cases;
}
} // namespace

int main() {
    admission_sweep();
    slot_machine();
    window_plan_sweep();
    layer_split_sweep();
    if (failures != 0) {
        std::cout << "FAILED batch slot checks: " << failures << "\n";
        return 1;
    }
    std::cout << "PASS batch slots: the --batch admission gates equal an independent transcription of upstream's "
                 "block over the full option sweep (serve/off, capacity clamp, --batch-mtp's WARNING-and-off path, "
                 "--batch-groups collapsing to one group on a single card, logit_bias refusal), and the slot machine "
                 "reproduces the window law - one BT per window, BDONE with eos -> stop, BSTOP -> cancel after that "
                 "token, produced >= max_new -> length, the context bound, and the non-finite row that emits nothing; "
                 "the window membership/cursor law equals a second transcription of :9957-9981 over " +
                 std::to_string(plan_cases) + " windows (two rows per slot while --batch-mtp is effective, "
                 "kVerifyMaxT cap, cursor advancing only in that mode), "
                 "the layer-split law agrees with a second transcription over " +
                 std::to_string(split_cases) +
                 " (spec, devices, serve, device count, parking) shapes, with auto clearing on one card, a same-GPU "
                 "split refused while parking is on, two boundaries fatal, and an accepted split reaching the --batch "
                 "gates; no device\n";
    return 0;
}
