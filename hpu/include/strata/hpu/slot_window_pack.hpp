#pragma once
// hpu/include/strata/hpu/slot_window_pack.hpp - pack the RUNNABLE slots into one multi-sequence window.
//
// Upstream's batched step takes one row per active slot: the linear layers run once for the whole window, while each
// row's attention sees only its own sequence (the per-row mask does that - `stage_arena_rows` fills it). This header is
// the admission law between the slot scheduler and that window:
//
//   * only slots that are READY contribute a row; the others are skipped, not padded with zeros (a zero row would
//     enter a sequence's history);
//   * the rows come in ascending slot order, so the same set of slots always produces the same plan (a window's byte
//     identity must not depend on iteration order);
//   * at most `max_rows` slots go into one window (the native chunk is 8); the caller runs a second window for the
//     rest, in the same order;
//   * a slot whose position is at or past the context is refused, as is a duplicate slot id;
//   * the plan is built by `build_window_row_plan`, so every row-table law (contiguous segments, absolute positions,
//     per-sequence KV base, own last row) applies unchanged.
//
// Host-only and SDK-free, so a test can pin it without a device.
#include "strata/hpu/window_rows.hpp"

#include <algorithm>
#include <cstdint>
#include <span>
#include <vector>

namespace strata::hpu {
struct SlotWindowSlot {
    uint32_t slot = 0;       // the scheduler's slot id (unique per window)
    uint32_t kv_base = 0;    // that slot's first KV cell in the arena
    uint32_t position = 0;   // the absolute position the new row will occupy
    bool ready = false;      // a slot that is not ready contributes no row
};

struct PackedSlotWindow {
    std::vector<WindowSequence> sequences;   // the rows, in ascending slot order
    std::vector<uint32_t> slots;             // the slot id of each sequence, same order
    WindowRowPlan plan;                      // the row table every kernel consumes
};

inline PackedSlotWindow pack_slot_window(const std::span<const SlotWindowSlot> slots, const uint32_t context,
                                         const uint32_t cell_capacity, const uint32_t max_rows) {
    if (max_rows == 0 || max_rows > kWindowRowsMax)
        throw std::invalid_argument("slot window: max_rows must be in 1..kWindowRowsMax");
    std::vector<SlotWindowSlot> ready;
    for (const auto& slot : slots)
        if (slot.ready) ready.push_back(slot);
    // Deterministic order: the same ready set must always produce the same plan bytes.
    std::sort(ready.begin(), ready.end(),
              [](const SlotWindowSlot& left, const SlotWindowSlot& right) { return left.slot < right.slot; });
    for (size_t i = 1; i < ready.size(); ++i)
        if (ready[i].slot == ready[i - 1].slot)
            throw std::invalid_argument("slot window: duplicate slot id " + std::to_string(ready[i].slot));
    if (ready.size() > max_rows) ready.resize(max_rows);   // the caller runs the next window for the rest
    if (ready.empty()) throw std::invalid_argument("slot window: no ready slot for this window");

    PackedSlotWindow packed;
    packed.sequences.reserve(ready.size());
    for (const auto& slot : ready) {
        if (slot.position >= context)
            throw std::invalid_argument("slot window: slot " + std::to_string(slot.slot) +
                                        " is at or past the context");
        packed.sequences.push_back(
            {.sequence = slot.slot, .kv_base = slot.kv_base, .first_position = slot.position, .rows = 1});
        packed.slots.push_back(slot.slot);
    }
    // One row per sequence, so `sequence` is already unique and increasing: the plan's own laws then hold, and its
    // sample row per sequence is that single row.
    packed.plan = build_window_row_plan(packed.sequences, context, cell_capacity, max_rows);
    return packed;
}
} // namespace strata::hpu
