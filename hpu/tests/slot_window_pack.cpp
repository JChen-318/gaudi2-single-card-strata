// hpu/tests/slot_window_pack.cpp - host evidence for the admission law between the slot scheduler and a
// multi-sequence window: which slots contribute a row, in which order, and which requests are refused.
#include "strata/hpu/slot_window_pack.hpp"

#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using strata::hpu::pack_slot_window;
using strata::hpu::SlotWindowSlot;

int failures = 0;
void check(const bool ok, const std::string& what) {
    if (!ok) {
        ++failures;
        std::cout << "FAIL " << what << "\n";
    }
}
std::string refusal(auto&& call) {
    try {
        call();
    } catch (const std::invalid_argument& error) {
        return error.what();
    }
    return {};
}
bool has(const std::string& message, const std::string& needle) {
    return message.find(needle) != std::string::npos;
}
} // namespace

int main() {
    // 1. only ready slots contribute, in ascending slot order regardless of the input order
    {
        const std::vector<SlotWindowSlot> slots{{.slot = 5, .kv_base = 500, .position = 7, .ready = true},
                                                {.slot = 2, .kv_base = 200, .position = 0, .ready = true},
                                                {.slot = 3, .kv_base = 300, .position = 3, .ready = false},
                                                {.slot = 9, .kv_base = 900, .position = 11, .ready = true}};
        const auto packed = pack_slot_window(slots, 4096, 4096, 8);
        check(packed.plan.rows() == 3, "three ready slots give three rows");
        check((packed.slots == std::vector<uint32_t>{2, 5, 9}), "rows come in ascending slot order");
        check((packed.plan.row_position == std::vector<uint32_t>{0, 7, 11}), "each row keeps its own position");
        check((packed.plan.kv_base == std::vector<uint32_t>{200, 500, 900}), "each row keeps its own KV base");
        check((packed.plan.sample_rows == std::vector<uint32_t>{0, 1, 2}), "each one-row sequence samples its row");
        check(packed.plan.segment_rows == std::vector<uint32_t>{1, 1, 1}, "one row per sequence");
        // the same set in a different input order must give the same plan
        const std::vector<SlotWindowSlot> reordered{slots[3], slots[1], slots[0], slots[2]};
        const auto again = pack_slot_window(reordered, 4096, 4096, 8);
        check(again.slots == packed.slots && again.plan.row_position == packed.plan.row_position,
              "input order does not change the plan");
    }

    // 2. max_rows truncates in slot order: the caller runs the next window for the rest
    {
        std::vector<SlotWindowSlot> slots;
        for (uint32_t s = 0; s < 12; ++s)
            slots.push_back({.slot = s, .kv_base = s * 100, .position = s, .ready = true});
        const auto first = pack_slot_window(slots, 4096, 4096, 8);
        check(first.plan.rows() == 8, "the first window takes the native chunk");
        check(first.slots.front() == 0 && first.slots.back() == 7, "the first window is slots 0..7");
        std::vector<SlotWindowSlot> rest;
        for (uint32_t s = 8; s < 12; ++s)
            rest.push_back({.slot = s, .kv_base = s * 100, .position = s, .ready = true});
        const auto second = pack_slot_window(rest, 4096, 4096, 8);
        check(second.plan.rows() == 4 && second.slots.front() == 8, "the second window carries the rest");
    }

    // 3. refusals
    {
        const std::vector<SlotWindowSlot> duplicate{{.slot = 4, .kv_base = 0, .position = 0, .ready = true},
                                                    {.slot = 4, .kv_base = 100, .position = 1, .ready = true}};
        check(has(refusal([&] { (void)pack_slot_window(duplicate, 4096, 4096, 8); }), "duplicate slot id"),
              "a duplicate slot id is refused");
        const std::vector<SlotWindowSlot> idle{{.slot = 1, .kv_base = 0, .position = 0, .ready = false}};
        check(has(refusal([&] { (void)pack_slot_window(idle, 4096, 4096, 8); }), "no ready slot"),
              "a window with no ready slot is refused");
        const std::vector<SlotWindowSlot> past{{.slot = 1, .kv_base = 0, .position = 4096, .ready = true}};
        check(has(refusal([&] { (void)pack_slot_window(past, 4096, 4096, 8); }), "at or past the context"),
              "a slot at the context is refused");
        const std::vector<SlotWindowSlot> one{{.slot = 1, .kv_base = 0, .position = 0, .ready = true}};
        check(has(refusal([&] { (void)pack_slot_window(one, 4096, 4096, 0); }), "max_rows"),
              "max_rows outside 1..kWindowRowsMax is refused");
        const std::vector<SlotWindowSlot> pool{{.slot = 1, .kv_base = 4000, .position = 100, .ready = true}};
        {
            const auto message = refusal([&] { (void)pack_slot_window(pool, 4096, 4096, 8); });
            check(has(message, "pool"), "a row past the KV pool is refused by the plan: " + message);
        }
    }

    // 4. random ready sets against a naive transcription
    {
        std::mt19937_64 rng(7);
        for (int round = 0; round < 2000; ++round) {
            const uint32_t count = 1 + uint32_t(rng() % 16);
            std::vector<SlotWindowSlot> slots;
            for (uint32_t s = 0; s < count; ++s)
                slots.push_back({.slot = s * 2 + uint32_t(rng() % 2), .kv_base = s * 64,
                                 .position = uint32_t(rng() % 40), .ready = (rng() % 3) != 0});
            std::vector<SlotWindowSlot> want;
            for (const auto& slot : slots)
                if (slot.ready) want.push_back(slot);
            if (want.empty()) {
                check(!refusal([&] { (void)pack_slot_window(slots, 4096, 4096, 8); }).empty(),
                      "an all-idle random window is refused");
                continue;
            }
            std::sort(want.begin(), want.end(),
                      [](const SlotWindowSlot& a, const SlotWindowSlot& b) { return a.slot < b.slot; });
            bool unique = true;
            for (size_t i = 1; i < want.size(); ++i) if (want[i].slot == want[i - 1].slot) unique = false;
            if (!unique) {
                check(!refusal([&] { (void)pack_slot_window(slots, 4096, 4096, 8); }).empty(),
                      "a random duplicate slot is refused");
                continue;
            }
            if (want.size() > 8) want.resize(8);
            const auto packed = pack_slot_window(slots, 4096, 4096, 8);
            bool same = packed.plan.rows() == want.size();
            for (size_t i = 0; same && i < want.size(); ++i)
                same = packed.slots[i] == want[i].slot && packed.plan.row_position[i] == want[i].position &&
                       packed.plan.kv_base[i] == want[i].kv_base;
            if (!same) {
                ++failures;
                std::cout << "FAIL random window " << round << " disagrees with the naive packing\n";
                break;
            }
        }
    }

    if (failures != 0) {
        std::cout << "FAILED slot-window checks: " << failures << "\n";
        return 1;
    }
    std::cout << "PASS slot window pack: only ready slots contribute (one row each), rows come in ascending slot order "
                 "whatever the input order, max_rows truncates in that same order, every row keeps its own position and "
                 "KV base and samples itself; duplicate ids, an all-idle window, a slot at the context, a bad max_rows "
                 "and a row past the pool are refused by name; 2000 random sets match a naive packing; no device\n";
    return 0;
}
