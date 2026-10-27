#pragma once
// hpu/include/strata/hpu/multi_slot_window.hpp - one window that decodes ONE row for each READY slot.
//
// This is the integration layer the row-batching pieces were built for:
//
//   pack_slot_window  ->  the rows (which slot, which position, whose KV range)
//   stage_arena_rows  ->  each row appended into ITS OWN sequence pool and gathered against its own prefix
//   attention + head  ->  one logits row per window row (the callbacks are the model's projections)
//   sample_window_rows->  one token per sequence, from its own row
//
// The order is fixed and the failure modes are explicit: a window with no ready slot, a model callback that returns the
// wrong shape, and a per-sequence input that does not match the packing are all refused before a token is produced.
// Header-only over the existing pieces; the callbacks keep it independent of the Model so a host test can drive it.
#include "strata/hpu/dense_window_flow.hpp"
#include "strata/hpu/kv_arena.hpp"
#include "strata/hpu/slot_window_pack.hpp"
#include "strata/hpu/window_sampling.hpp"

#include <cstdint>
#include <functional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace strata::hpu {
struct MultiSlotRow {
    SlotWindowSlot window;                 // slot id, KV base, position, ready
    int32_t token = 0;                     // the token this row feeds
    std::span<const float> embedding;      // [hidden] the model's embedding for that token
    std::span<const int32_t> history;      // that sequence's recent tokens (penalties)
    strata::kernels::SamplerParams params;
    std::span<const float> bias;
};

struct MultiSlotOperations {
    std::function<DenseProjected(std::span<const float>)> project;      // [T,hidden] -> projections
    std::function<std::vector<float>(const QsaWindowBatch&)> attention; // -> [T,hidden]
    std::function<std::vector<float>(std::span<const float>)> head;     // [T,hidden] -> [T,vocabulary]
};

struct MultiSlotWindowResult {
    std::vector<int32_t> tokens;    // [sequences] in ascending slot order
    std::vector<uint32_t> slots;    // [sequences] the slot id of each token
    std::vector<uint32_t> rows;     // [sequences] the window row each token came from
    uint32_t window_rows = 0;       // rows this window carried
    uint32_t bucket = 0;            // the padded attention bucket they shared
};

// `hidden` is the model width, `vocabulary` the head width. The window's rows are the READY slots in ascending slot
// order, at most `max_rows` of them (the caller runs the next window for the rest).
inline MultiSlotWindowResult run_multi_slot_window(KvArena& arena, const QsaConfig& config,
                                                   const MultiSlotOperations& operations,
                                                   const std::span<const MultiSlotRow> rows, const uint32_t hidden,
                                                   const uint32_t vocabulary, const DenseNorms norms,
                                                   const uint32_t window, const uint32_t max_rows,
                                                   const uint32_t context, const uint32_t cell_capacity) {
    if (!operations.project || !operations.attention || !operations.head)
        throw std::invalid_argument("multi-slot window: every model stage is required");
    if (hidden == 0 || vocabulary == 0) throw std::invalid_argument("multi-slot window: hidden/vocabulary is zero");
    std::vector<SlotWindowSlot> slots;
    slots.reserve(rows.size());
    for (const auto& row : rows) slots.push_back(row.window);
    const auto packed = pack_slot_window(slots, context, cell_capacity, max_rows);

    // Join the packed (slot-ordered) window with the caller's per-row inputs, which may arrive in any order.
    std::vector<const MultiSlotRow*> inputs;
    inputs.reserve(packed.sequences.size());
    for (const uint32_t slot : packed.slots) {
        const MultiSlotRow* found = nullptr;
        for (const auto& row : rows)
            if (row.window.slot == slot) { found = &row; break; }
        if (found == nullptr) throw std::invalid_argument("multi-slot window: no input for slot " + std::to_string(slot));
        if (found->embedding.size() != hidden)
            throw std::invalid_argument("multi-slot window: slot " + std::to_string(slot) +
                                        " embedding does not match the hidden width");
        inputs.push_back(found);
    }

    std::vector<float> embeddings;
    embeddings.reserve(size_t(inputs.size()) * hidden);
    for (const auto* input : inputs)
        embeddings.insert(embeddings.end(), input->embedding.begin(), input->embedding.end());
    const auto projected = operations.project(embeddings);
    const auto batch = stage_arena_rows(arena, packed.plan, config, projected, window, norms, /*cancelled=*/{});
    const auto attended = operations.attention(batch);
    if (attended.size() != size_t(packed.sequences.size()) * hidden)
        throw std::invalid_argument("multi-slot window: the attention stage did not return one hidden row per sequence");
    const auto logits = operations.head(attended);
    if (logits.size() != size_t(packed.sequences.size()) * vocabulary)
        throw std::invalid_argument("multi-slot window: the head did not return one logits row per sequence");

    std::vector<WindowSampleRequest> requests;
    requests.reserve(inputs.size());
    for (const auto* input : inputs)
        requests.push_back({.history = input->history, .params = input->params, .bias = input->bias});
    auto sampled = sample_window_rows(packed.plan, logits, vocabulary, requests);

    MultiSlotWindowResult result;
    result.tokens = std::move(sampled.tokens);
    result.rows = std::move(sampled.rows);
    result.slots = packed.slots;
    result.window_rows = packed.plan.rows();
    result.bucket = batch.bucket;
    return result;
}
} // namespace strata::hpu
