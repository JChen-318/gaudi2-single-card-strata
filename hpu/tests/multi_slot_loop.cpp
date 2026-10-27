// hpu/tests/multi_slot_loop.cpp - host evidence for the batched decode loop over several slots.
//
// Laws: N ready slots need ceil(N/max_rows) windows; every slot generates exactly its own max_new tokens (or fewer when
// its own stop token appears); a stop token stops ONLY that slot; each slot's history is its own tokens; positions
// advance by the rows the slot contributed; and the loop refuses an empty set, max_new == 0, duplicate slots, and an
// embedding of the wrong width.
#include "strata/hpu/multi_slot_loop.hpp"

#include <algorithm>
#include <deque>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using strata::hpu::DenseNorms;
using strata::hpu::DenseProjected;
using strata::hpu::KvArena;
using strata::hpu::MultiSlotLoopReport;
using strata::hpu::MultiSlotOperations;
using strata::hpu::MultiSlotSequence;
using strata::hpu::QsaConfig;
using strata::hpu::run_multi_slot_loop;

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
    } catch (const std::exception& error) {
        return error.what();
    }
    return "";
}
bool has(const std::string& message, const std::string& needle) {
    return message.find(needle) != std::string::npos;
}
const uint32_t kHidden = 8, kVocabulary = 16;
QsaConfig config(const uint32_t capacity) {
    QsaConfig c;
    c.query_heads = 1;
    c.kv_heads = 1;
    c.width = 8;
    c.rotary = 8;
    c.index_heads = 1;
    c.index_width = 8;
    c.block = 4;
    c.top_k = capacity;
    c.capacity = capacity;
    c.position_base = 0;
    c.validate();
    return c;
}
// The fake model makes window row i's argmax equal to `salt_base + i`, and the loop's window order is ascending slot
// order, so the token trail of each slot is reproducible: slot s's k-th token is salt_base + (its row index each window).
struct FakeModel {
    uint32_t windows = 0, head_calls = 0;
    MultiSlotOperations operations() {
        MultiSlotOperations op;
        op.project = [](std::span<const float> embeddings) {
            const uint32_t rows = uint32_t(embeddings.size() / kHidden);
            DenseProjected projected;
            projected.q_gate.assign(size_t(rows) * 2 * kHidden, 0.5f);
            projected.key.assign(size_t(rows) * kHidden, 0.25f);
            projected.value.assign(size_t(rows) * kHidden, 0.75f);
            return projected;
        };
        op.attention = [](const strata::hpu::QsaWindowBatch& batch) {
            return std::vector<float>(size_t(batch.rows) * kHidden, 0.125f);
        };
        op.head = [this](std::span<const float> attended) {
            ++head_calls;
            const uint32_t rows = uint32_t(attended.size() / kHidden);
            std::vector<float> logits(size_t(rows) * kVocabulary, -5.0f);
            // row 0 -> token 1, row 1 -> token 2 (always inside the vocabulary)
            for (uint32_t row = 0; row < rows; ++row) logits[size_t(row) * kVocabulary + ((row % 2) + 1)] = 5.0f;
            return logits;
        };
        return op;
    }
};
struct UnitNorms {
    std::vector<float> query, key;
    explicit UnitNorms(const uint32_t width) : query(width, 1.0f), key(width, 1.0f) {}
    DenseNorms view() const { return {.query = query, .key = key}; }
};
} // namespace

int main() {
    const auto c = config(128);
    const UnitNorms gamma(c.width);
    auto embed = [](uint32_t, int32_t token) { return std::vector<float>(kHidden, float(token) / 16.0f); };

    // 1. five ready slots, two rows per window: three windows, and every slot gets exactly its max_new tokens
    {
        KvArena arena;
        std::deque<strata::hpu::DenseKvState> pools;   // DenseKvState is neither copyable nor movable: a deque never relocates its elements
        std::vector<MultiSlotSequence> sequences;
        for (uint32_t slot = 0; slot < 5; ++slot) {
            pools.emplace_back(c.capacity, c.kv_heads, c.width);
            arena.add_region(pools.back(), slot * c.capacity);
            sequences.push_back({.slot = slot, .kv_base = slot * c.capacity, .position = 0, .next_token = 7,
                                 .max_new = 2, .generated = 0, .history = {7}, .params = {}, .bias = {},
                                 .stops = {}, .done = false, .stopped_by_token = false});
        }
        for (auto& sequence : sequences) {
            sequence.params.greedy = true;
            sequence.params.top_p = 1;
            sequence.params.top_k = 64;
        }
        FakeModel model;
        const auto report = run_multi_slot_loop(arena, c, model.operations(), gamma.view(), embed, sequences, kHidden,
                                                kVocabulary, /*max_rows=*/2, /*context=*/4096,
                                                /*cell_capacity=*/4096);
        // The policy is "lowest ready slots first": slots 0,1 drain in two windows, then 2,3, then 4.
        check(report.windows == 6, "five slots at two rows per window take six windows under lowest-slot-first");
        check(report.slots_finished == 5, "every slot finished");
        check(report.tokens_generated == 10, "five slots times two tokens");
        for (const auto& sequence : sequences) {
            check(sequence.generated == sequence.max_new, "slot " + std::to_string(sequence.slot) +
                                                              " generated exactly its max_new");
            check(sequence.history.size() == 3, "the slot's history is its prompt token plus its own two tokens");
            check(sequence.position == 2, "the slot's position advanced by the rows it contributed");
        }
        check(model.head_calls == 6, "the head ran once per window (six windows)");
    }

    // 2. a stop token stops ONLY the slot that produced it
    {
        KvArena arena;
        std::deque<strata::hpu::DenseKvState> pools;   // DenseKvState is neither copyable nor movable: a deque never relocates its elements
        std::vector<MultiSlotSequence> sequences;
        for (uint32_t slot = 0; slot < 2; ++slot) {
            pools.emplace_back(c.capacity, c.kv_heads, c.width);
            arena.add_region(pools.back(), slot * c.capacity);
            sequences.push_back({.slot = slot, .kv_base = slot * c.capacity, .position = 0, .next_token = 7, .max_new = 4,
                                 .generated = 0, .history = {7}, .params = {}, .bias = {}, .stops = {},
                                 .done = false, .stopped_by_token = false});
        }
        sequences[0].params.greedy = true;
        sequences[0].params.top_p = 1;
        sequences[0].params.top_k = 64;
        sequences[1].params = sequences[0].params;
        // window row 0 is slot 0 and produces token 1: make token 1 slot 0's stop token
        sequences[0].stops = {1};
        FakeModel model;
        const auto report = run_multi_slot_loop(arena, c, model.operations(), gamma.view(), embed, sequences, kHidden,
                                                kVocabulary, /*max_rows=*/1, 4096, 4096);
        check(sequences[0].done && sequences[0].stopped_by_token, "the stopped slot says it stopped on its token");
        check(sequences[0].generated == 1, "the stopped slot generated only the stop token");
        check(!sequences[1].done || sequences[1].generated == 4, "the other slot kept going to its max_new");
        check(report.windows >= 4, "the loop kept running windows for the surviving slot");
    }

    // 3. refusals
    {
        KvArena arena;
        strata::hpu::DenseKvState pool(c.capacity, c.kv_heads, c.width);
        arena.add_region(pool, 0);
        FakeModel model;
        std::vector<MultiSlotSequence> none{};
        check(has(refusal([&] {
                  (void)run_multi_slot_loop(arena, c, model.operations(), gamma.view(), embed, none, kHidden,
                                            kVocabulary, 2, 4096, 4096);
              }), "no sequence"),
              "an empty slot set is refused");
        std::vector<MultiSlotSequence> zero{{.slot = 1, .kv_base = 0, .position = 0, .next_token = 1, .max_new = 0,
                                             .generated = 0, .history = {1}, .params = {}, .bias = {}, .stops = {},
                                             .done = false, .stopped_by_token = false}};
        check(has(refusal([&] {
                  (void)run_multi_slot_loop(arena, c, model.operations(), gamma.view(), embed, zero, kHidden,
                                            kVocabulary, 2, 4096, 4096);
              }), "max_new == 0"),
              "a slot with max_new == 0 is refused");
        std::vector<MultiSlotSequence> two{{.slot = 3, .kv_base = 0, .position = 0, .next_token = 1, .max_new = 1,
                                            .generated = 0, .history = {1}, .params = {}, .bias = {}, .stops = {},
                                            .done = false, .stopped_by_token = false},
                                           {.slot = 3, .kv_base = 0, .position = 0, .next_token = 1, .max_new = 1,
                                            .generated = 0, .history = {1}, .params = {}, .bias = {}, .stops = {},
                                            .done = false, .stopped_by_token = false}};
        check(has(refusal([&] {
                  (void)run_multi_slot_loop(arena, c, model.operations(), gamma.view(), embed, two, kHidden,
                                            kVocabulary, 2, 4096, 4096);
              }), "duplicate slot"),
              "duplicate slots are refused");
        std::vector<MultiSlotSequence> one{{.slot = 4, .kv_base = 0, .position = 0, .next_token = 1, .max_new = 1,
                                            .generated = 0, .history = {1}, .params = {}, .bias = {}, .stops = {},
                                            .done = false, .stopped_by_token = false}};
        auto narrow = [](uint32_t, int32_t) { return std::vector<float>(kHidden - 1, 0.0f); };
        check(has(refusal([&] {
                  (void)run_multi_slot_loop(arena, c, model.operations(), gamma.view(), narrow, one, kHidden,
                                            kVocabulary, 2, 4096, 4096);
              }), "hidden width"),
              "an embedding of the wrong width is refused");
    }

    if (failures != 0) {
        std::cout << "FAILED multi-slot loop checks: " << failures << "\n";
        return 1;
    }
    std::cout << "PASS multi-slot loop: five slots at two rows per window take six windows (lowest slot first), every slot "
                 "generates exactly its own max_new (or stops early on its own stop token while the others continue), each slot's "
                 "history is its own and its position advances by its own rows, the head runs once per window, and an "
                 "empty set, max_new == 0, duplicate slots and a mis-sized embedding are refused; no device\n";
    return 0;
}
