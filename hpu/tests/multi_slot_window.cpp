// hpu/tests/multi_slot_window.cpp - host evidence for the integration layer of multi-slot row batching.
//
// One window, several READY slots, ONE row each. The laws checked here:
//   * every ready slot produces exactly one token, in ascending slot order, and an idle slot produces none;
//   * each row is appended into its own slot's pool at its own position (the arena's next-cell law), so the pools grow
//     by exactly their own rows;
//   * the model callbacks see ONE window (projection, attention and head each called once for the packed rows), and the
//     per-row logits decide each slot's token - verified by making the fake head's argmax the row's position;
//   * a token is sampled from the slot's own row: the token of the slot at position p is p % vocabulary;
//   * refusals: no ready slot, an embedding of the wrong width, an input missing for a packed slot, an attention or head
//     result that is not one row per sequence.
#include "strata/hpu/multi_slot_window.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using strata::hpu::DenseNorms;
using strata::hpu::DenseProjected;
using strata::hpu::KvArena;
using strata::hpu::MultiSlotOperations;
using strata::hpu::MultiSlotRow;
using strata::hpu::MultiSlotWindowResult;
using strata::hpu::QsaConfig;
using strata::hpu::run_multi_slot_window;

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
    return {};
}
bool has(const std::string& message, const std::string& needle) {
    return message.find(needle) != std::string::npos;
}
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
const uint32_t kHidden = 8, kVocabulary = 16;
// DenseNorms holds SPANS, so the gamma vectors must outlive them: a helper returning the struct by value would hand
// back spans into vectors that die at the end of the full expression (that dangling span was a real bug in this test).
struct UnitNorms {
    std::vector<float> query, key;
    explicit UnitNorms(const uint32_t width) : query(width, 1.0f), key(width, 1.0f) {}
    DenseNorms view() const { return {.query = query, .key = key}; }
};
// The fake model: the projection turns each embedding row into a q/key/value row; attention passes the row through;
// the head makes the argmax of row i equal to a value the test chose per row (carried through the embedding's first
// element, so the head's output depends on the ROW, not on hidden state).
struct FakeModel {
    uint32_t project_calls = 0, attention_calls = 0, head_calls = 0, rows_seen = 0;
    MultiSlotOperations operations() {
        MultiSlotOperations op;
        op.project = [this](std::span<const float> embeddings) {
            ++project_calls;
            const uint32_t rows = uint32_t(embeddings.size() / kHidden);
            rows_seen = rows;
            DenseProjected projected;
            projected.q_gate.assign(size_t(rows) * 2 * kHidden, 0.5f);
            projected.key.assign(size_t(rows) * kHidden, 0.25f);
            projected.value.assign(size_t(rows) * kHidden, 0.75f);
            return projected;
        };
        op.attention = [this](const strata::hpu::QsaWindowBatch& batch) {
            ++attention_calls;
            return std::vector<float>(size_t(batch.rows) * kHidden, 0.125f);
        };
        op.head = [this](std::span<const float> attended) {
            ++head_calls;
            const uint32_t rows = uint32_t(attended.size() / kHidden);
            std::vector<float> logits(size_t(rows) * kVocabulary, -5.0f);
            for (uint32_t row = 0; row < rows; ++row) logits[size_t(row) * kVocabulary + (row % kVocabulary)] = 5.0f;
            return logits;
        };
        return op;
    }
};
std::vector<float> embedding(const float salt) { return std::vector<float>(kHidden, salt); }
} // namespace

int main() {
    const auto c = config(64);
    const UnitNorms gamma(c.width);
    const auto norms = gamma.view();

    // 1. two ready slots and one idle: two rows, two tokens, ascending slot order
    {
        KvArena arena;
        strata::hpu::DenseKvState pool_a(64, 1, c.width), pool_b(64, 1, c.width);
        // give each pool a prefix so the positions are not zero
        for (int i = 0; i < 3; ++i) {
            std::vector<float> key(c.width, 0.1f), value(c.width, 0.2f);
            pool_a.append(key, value);
        }
        for (int i = 0; i < 1; ++i) {
            std::vector<float> key(c.width, 0.3f), value(c.width, 0.4f);
            pool_b.append(key, value);
        }
        arena.add_region(pool_a, 0);
        arena.add_region(pool_b, 100);

        const auto emb_a = embedding(1.0f), emb_b = embedding(2.0f), emb_idle = embedding(3.0f);
        std::vector<int32_t> history{};
        strata::kernels::SamplerParams params;
        params.greedy = true;
        params.top_p = 1;
        params.top_k = 64;
        std::vector<MultiSlotRow> rows{
            {.window = {.slot = 5, .kv_base = 100, .position = 1, .ready = true}, .token = 11,
             .embedding = emb_b, .history = history, .params = params, .bias = {}},
            {.window = {.slot = 2, .kv_base = 0, .position = 3, .ready = true}, .token = 22,
             .embedding = emb_a, .history = history, .params = params, .bias = {}},
            {.window = {.slot = 9, .kv_base = 0, .position = 0, .ready = false}, .token = 33,
             .embedding = emb_idle, .history = history, .params = params, .bias = {}},
        };
        FakeModel model;
        const auto result = run_multi_slot_window(arena, c, model.operations(), rows, kHidden, kVocabulary, norms,
                                                  /*window=*/0, /*max_rows=*/8, /*context=*/4096,
                                                  /*cell_capacity=*/4096);
        check(result.window_rows == 2, "two ready slots give a two-row window");
        check((result.slots == std::vector<uint32_t>{2, 5}), "the window is in ascending slot order");
        check(result.tokens.size() == 2, "one token per ready slot");
        // the fake head makes window row i's argmax i: row 0 is slot 2, row 1 is slot 5.
        check((result.tokens == std::vector<int32_t>{0, 1}), "each slot samples its own window row's argmax");
        check((result.rows == std::vector<uint32_t>{0, 1}), "the sampled rows are the window rows");
        check(pool_a.size() == 4, "slot 2's pool grew by exactly its one row");
        check(pool_b.size() == 2, "slot 5's pool grew by exactly its one row");
        check(model.project_calls == 1 && model.attention_calls == 1 && model.head_calls == 1,
              "the model ran ONCE for the packed window, not once per slot");
        check(model.rows_seen == 2, "the projection saw the window's two rows");
    }

    // 2. refusals
    {
        KvArena arena;
        strata::hpu::DenseKvState pool(64, 1, c.width);
        arena.add_region(pool, 0);
        const auto emb = embedding(1.0f);
        std::vector<int32_t> history{};
        strata::kernels::SamplerParams params;
        params.greedy = true;
        params.top_p = 1;
        params.top_k = 64;
        FakeModel model;
        std::vector<MultiSlotRow> idle{
            {.window = {.slot = 1, .kv_base = 0, .position = 0, .ready = false}, .token = 1, .embedding = emb,
             .history = history, .params = params, .bias = {}}};
        check(has(refusal([&] {
                  (void)run_multi_slot_window(arena, c, model.operations(), idle, kHidden, kVocabulary, norms, 0, 8,
                                              4096, 4096);
              }), "no ready slot"),
              "a window with no ready slot is refused");
        std::vector<MultiSlotRow> narrow{
            {.window = {.slot = 1, .kv_base = 0, .position = 0, .ready = true}, .token = 1,
             .embedding = std::vector<float>(kHidden - 1, 0.5f), .history = history, .params = params, .bias = {}}};
        check(has(refusal([&] {
                  (void)run_multi_slot_window(arena, c, model.operations(), narrow, kHidden, kVocabulary, norms, 0, 8,
                                              4096, 4096);
              }), "hidden width"),
              "an embedding of the wrong width is refused");
        // a head that returns one extra row
        MultiSlotOperations broken = model.operations();
        broken.head = [](std::span<const float> attended) {
            return std::vector<float>(attended.size() / kHidden + 1, 0.0f);
        };
        std::vector<MultiSlotRow> one{
            {.window = {.slot = 1, .kv_base = 0, .position = 0, .ready = true}, .token = 1, .embedding = emb,
             .history = history, .params = params, .bias = {}}};
        check(has(refusal([&] {
                  (void)run_multi_slot_window(arena, c, broken, one, kHidden, kVocabulary, norms, 0, 8, 4096, 4096);
              }), "the head did not return"),
              "a head result of the wrong shape is refused");
    }

    if (failures != 0) {
        std::cout << "FAILED multi-slot window checks: " << failures << "\n";
        return 1;
    }
    std::cout << "PASS multi-slot window: one window decodes one row per READY slot in ascending slot order, each row "
                 "lands in its own pool at its own position, the model callbacks run once for the packed window, each "
                 "slot samples its own row, and an empty window, a mis-sized embedding and a mis-shaped head result "
                 "are refused; no device\n";
    return 0;
}
