#pragma once
// hpu/include/strata/hpu/multi_slot_loop.hpp - the batched decode LOOP over several slots, one window at a time.
//
// `run_multi_slot_window` decodes one row per ready slot for ONE window. This is the loop around it: it keeps the slots
// that still need tokens, packs up to `max_rows` of them per window (ascending slot order), feeds each slot's own last
// token's embedding, and stops a slot on its own stop token or when it has produced `max_new` tokens - without stopping
// the others. The laws:
//
//   * a window carries at most `max_rows` slots, taken in ASCENDING SLOT ORDER (the lowest ready slots go first),
//     so a slot drains over consecutive windows before a higher slot starts;
//   * a slot's history is that slot's own tokens: penalties never see another slot's text;
//   * the stop check runs BEFORE the token is appended for the next window, so a stopped slot contributes no more rows;
//   * `position` advances by exactly the rows a slot contributed (one per window it was in);
//   * the loop refuses an empty slot set, a slot with max_new == 0, and a model callback that violates its shapes.
//
// Header-only; the caller owns the model (embed/project/attention/head) and the KV pools.
#include "strata/hpu/multi_slot_window.hpp"

#include <algorithm>
#include <cstdint>
#include <deque>
#include <functional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace strata::hpu {
struct MultiSlotSequence {
    uint32_t slot = 0;             // scheduler's slot id (unique)
    uint32_t kv_base = 0;          // its first KV cell in the arena
    uint32_t position = 0;         // the next absolute position it will write
    int32_t next_token = 0;        // the token to feed in its next window
    uint32_t max_new = 0;          // tokens this slot may generate
    uint32_t generated = 0;        // tokens it has generated so far
    std::vector<int32_t> history;  // its own tokens (prompt + generated): penalties and the emit trail
    strata::kernels::SamplerParams params;
    std::span<const float> bias;
    std::vector<int32_t> stops;    // this slot's stop tokens (empty = none)
    bool done = false;
    bool stopped_by_token = false;
};

struct MultiSlotLoopReport {
    uint32_t windows = 0;
    uint32_t slots_finished = 0;
    uint32_t tokens_generated = 0;
};

inline MultiSlotLoopReport run_multi_slot_loop(KvArena& arena, const QsaConfig& config,
                                               const MultiSlotOperations& operations, const DenseNorms& norms,
                                               const std::function<std::vector<float>(uint32_t slot, int32_t token)> &embed,
                                               std::vector<MultiSlotSequence>& sequences, const uint32_t hidden,
                                               const uint32_t vocabulary, const uint32_t max_rows,
                                               const uint32_t context, const uint32_t cell_capacity,
                                               const uint32_t max_windows = 1u << 20) {
    if (!embed) throw std::invalid_argument("multi-slot loop: an embedding source is required");
    if (sequences.empty()) throw std::invalid_argument("multi-slot loop: no sequence to decode");
    if (max_rows == 0 || max_rows > kWindowRowsMax) throw std::invalid_argument("multi-slot loop: bad max_rows");
    for (size_t i = 0; i < sequences.size(); ++i) {
        if (sequences[i].max_new == 0)
            throw std::invalid_argument("multi-slot loop: slot " + std::to_string(sequences[i].slot) +
                                        " has max_new == 0");
        for (size_t j = 0; j < i; ++j)
            if (sequences[j].slot == sequences[i].slot)
                throw std::invalid_argument("multi-slot loop: duplicate slot " + std::to_string(sequences[i].slot));
    }
    MultiSlotLoopReport report;
    while (report.windows < max_windows) {
        std::vector<size_t> ready;
        for (size_t i = 0; i < sequences.size(); ++i)
            if (!sequences[i].done) ready.push_back(i);
        if (ready.empty()) break;
        std::sort(ready.begin(), ready.end(),
                  [&sequences](const size_t left, const size_t right) {
                      return sequences[left].slot < sequences[right].slot;
                  });
        if (ready.size() > max_rows) ready.resize(max_rows);

        std::vector<MultiSlotRow> rows;
        rows.reserve(ready.size());
        // The row's embedding SPAN must outlive the loop body: `MultiSlotRow` is built here and consumed after the loop,
        // so each embedding lives in this container until the window finishes.
        std::deque<std::vector<float>> embeddings;
        for (const size_t index : ready) {
            auto& sequence = sequences[index];
            embeddings.push_back(embed(sequence.slot, sequence.next_token));
            const auto& embedding = embeddings.back();
            if (embedding.size() != hidden)
                throw std::invalid_argument("multi-slot loop: slot " + std::to_string(sequence.slot) +
                                            " embedding does not match the hidden width");
            rows.push_back({.window = {.slot = sequence.slot, .kv_base = sequence.kv_base,
                                      .position = sequence.position, .ready = true},
                            .token = sequence.next_token,
                            .embedding = std::span<const float>(embedding),
                            .history = sequence.history,
                            .params = sequence.params,
                            .bias = sequence.bias});
        }
        const auto result = run_multi_slot_window(arena, config, operations, rows, hidden, vocabulary, norms,
                                                  /*window=*/0, max_rows, context, cell_capacity);
        ++report.windows;
        for (size_t i = 0; i < result.tokens.size(); ++i) {
            auto& sequence = sequences[ready[i]];
            const int32_t token = result.tokens[i];
            sequence.history.push_back(token);
            sequence.next_token = token;
            ++sequence.position;
            ++sequence.generated;
            ++report.tokens_generated;
            const bool stop = std::find(sequence.stops.begin(), sequence.stops.end(), token) != sequence.stops.end();
            if (stop) {
                sequence.done = true;
                sequence.stopped_by_token = true;
            } else if (sequence.generated >= sequence.max_new) {
                sequence.done = true;
            }
        }
    }
    for (const auto& sequence : sequences) if (sequence.done) ++report.slots_finished;
    return report;
}
} // namespace strata::hpu
