// hpu/tests/arena_window_flow.cpp - host evidence that a window carrying SEVERAL sequences stages exactly the rows the
// sequential path stages, and that every failure rolls each pool back.
//
// The reference is `stage_dense_window`, the single-sequence path the port already uses: for a sequence whose prefix is
// already in its pool, `start + row` IS the plan's absolute position, so the two paths must produce identical per-row
// queries/gates and identical gathered key/value/mask prefixes. Nothing here needs a device: the dense KV cells live in
// host memory and the attention gather is host code.
#include "strata/hpu/dense_window_flow.hpp"
#include "strata/hpu/kv_arena.hpp"
#include "strata/hpu/session_flow.hpp"  // Cancelled
#include "strata/hpu/window_rows.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using strata::hpu::build_window_row_plan;
using strata::hpu::DenseKvState;
using strata::hpu::DenseProjected;
using strata::hpu::KvArena;
using strata::hpu::QsaConfig;
using strata::hpu::stage_arena_rows;
using strata::hpu::stage_dense_window;
using strata::hpu::WindowSequence;

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
    c.query_heads = 2;
    c.kv_heads = 1;
    c.width = 8;
    c.rotary = 8;
    c.index_heads = 1;
    c.index_width = 8;
    c.block = 4;
    c.top_k = capacity;
    c.capacity = capacity;
    c.position_base = 3;
    c.validate();
    return c;
}
// `gamma` is a namespace-scope vector, so these spans stay valid for the whole test (a helper returning a DenseNorms
// built from a temporary vector would dangle - the same bug the multi-slot test had).
const std::vector<float> gamma{.9f, 1.1f, 1.0f, 1.2f, .8f, 1.3f, .95f, 1.05f};
strata::hpu::DenseNorms norms() {
    return {.query = gamma, .key = gamma};
}
// Deterministic projected rows: q_gate has 2*query_heads*width per row, key/value kv_heads*width.
DenseProjected project(const uint32_t rows, const float salt) {
    DenseProjected p;
    p.q_gate.resize(size_t(rows) * 2 * 2 * 8);
    p.key.resize(size_t(rows) * 8);
    p.value.resize(size_t(rows) * 8);
    for (size_t row = 0; row < rows; ++row) {
        for (size_t i = 0; i < p.q_gate.size() / rows; ++i) p.q_gate[row * (p.q_gate.size() / rows) + i] =
            salt + float((i * 7 + row * 3) % 13) / 11.0f;
        for (size_t i = 0; i < 8; ++i) {
            p.key[row * 8 + i] = salt + float((i + row) % 5) / 7.0f;
            p.value[row * 8 + i] = salt - float((i + 2 * row) % 3) / 9.0f;
        }
    }
    return p;
}
void fill(DenseKvState& state, const uint32_t rows, const float salt) {
    std::vector<float> key(8), value(8);
    for (uint32_t row = 0; row < rows; ++row) {
        for (uint32_t i = 0; i < 8; ++i) {
            key[i] = salt + float((i + row) % 4) / 5.0f;
            value[i] = salt - float((i + row) % 3) / 7.0f;
        }
        state.append(key, value);
    }
}
bool close(const std::vector<float>& a, const std::vector<float>& b, const float tolerance = 1e-5f) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i)
        if (!(std::fabs(a[i] - b[i]) <= tolerance * (1.0f + std::fabs(b[i])))) return false;
    return true;
}
} // namespace

int main() {
    const uint32_t capacity = 64, window = 0;
    const auto c = config(capacity);

    // Sequence A holds 4 prefix cells, sequence B holds 2; the window appends 2 rows to A and 1 to B.
    const std::vector<WindowSequence> sequences{{.sequence = 0, .kv_base = 0, .first_position = 4, .rows = 2},
                                                {.sequence = 1, .kv_base = 1000, .first_position = 2, .rows = 1}};
    const auto plan = build_window_row_plan(sequences, 4096, 4096, 8);
    const auto projected = project(plan.rows(), 0.25f);

    // The arena path.
    DenseKvState a_arena(capacity, 1, 8), b_arena(capacity, 1, 8);
    fill(a_arena, 4, 0.10f);
    fill(b_arena, 2, 0.50f);
    KvArena arena;
    const auto ra = arena.add_region(a_arena, 0);
    const auto rb = arena.add_region(b_arena, 1000);
    check(ra == 0 && rb == 1, "two regions");
    const auto batch = stage_arena_rows(arena, plan, c, projected, window, norms());
    check(batch.rows == 3, "the batch carries the plan's three rows");
    check(a_arena.size() == 6 && b_arena.size() == 3, "each sequence grew by exactly its own rows");

    // The reference: the same rows through the single-sequence path, on pools with the same prefixes.
    DenseKvState a_ref(capacity, 1, 8), b_ref(capacity, 1, 8);
    fill(a_ref, 4, 0.10f);
    fill(b_ref, 2, 0.50f);
    // Rows 0,1 belong to A and row 2 to B: feed each sequence its own contiguous projected slice.
    DenseProjected a_projected, b_projected;
    a_projected.q_gate.assign(projected.q_gate.begin(), projected.q_gate.begin() + 2 * 32);
    a_projected.key.assign(projected.key.begin(), projected.key.begin() + 2 * 8);
    a_projected.value.assign(projected.value.begin(), projected.value.begin() + 2 * 8);
    b_projected.q_gate.assign(projected.q_gate.begin() + 2 * 32, projected.q_gate.begin() + 3 * 32);
    b_projected.key.assign(projected.key.begin() + 2 * 8, projected.key.begin() + 3 * 8);
    b_projected.value.assign(projected.value.begin() + 2 * 8, projected.value.begin() + 3 * 8);
    const auto a_batch = stage_dense_window(a_ref, c, a_projected, 2, window, norms());
    const auto b_batch = stage_dense_window(b_ref, c, b_projected, 1, window, norms());
    check(a_ref.size() == 6 && b_ref.size() == 3, "the reference pools grew the same way");

    // Queries and gates are bucket-independent: they must match row for row.
    check(close(std::vector<float>(batch.query.begin(), batch.query.begin() + 2 * 16),
                std::vector<float>(a_batch.query.begin(), a_batch.query.begin() + 2 * 16)),
          "sequence A's queries match the sequential path");
    check(close(std::vector<float>(batch.gate.begin(), batch.gate.begin() + 2 * 16),
                std::vector<float>(a_batch.gate.begin(), a_batch.gate.begin() + 2 * 16)),
          "sequence A's gates match the sequential path");
    check(close(std::vector<float>(batch.query.begin() + 2 * 16, batch.query.end()),
                std::vector<float>(b_batch.query.begin(), b_batch.query.end())),
          "sequence B's queries match the sequential path");

    // The gathered prefix of each row must equal that sequence's own prefix from its own pool. Rows of A see 5 and 6
    // cells; B's row sees 3 - and never A's cells, even though both share one padded bucket.
    for (uint32_t row = 0; row < 3; ++row) {
        const uint32_t sequence = plan.row_sequence[row];
        const uint32_t cells = plan.row_position[row] + 1;
        DenseKvState& pool = sequence == 0 ? a_ref : b_ref;
        const auto expected = pool.gather_prefix(cells, batch.bucket, window);
        const auto at = size_t(row) * size_t(c.kv_heads) * batch.bucket * c.width;
        const std::vector<float> got(batch.key.begin() + at, batch.key.begin() + at + expected.key.size());
        check(close(got, expected.key), "row " + std::to_string(row) + " gathers its own sequence's key prefix");
        const std::vector<float> got_value(batch.value.begin() + at, batch.value.begin() + at + expected.value.size());
        check(close(got_value, expected.value), "row " + std::to_string(row) + " gathers its own value prefix");
        // the padded tail beyond `cells` must be masked out for that row
        for (uint32_t cell = cells; cell < batch.bucket; ++cell) {
            const auto mask_at = size_t(row) * c.kv_heads * batch.bucket + cell;
            if (batch.mask[mask_at] > -1e30f) {
                ++failures;
                std::cout << "FAIL row " << row << ": padding cell " << cell << " is not masked\n";
                break;
            }
        }
    }

    // Refusals.
    {
        DenseKvState x(capacity, 1, 8), y(capacity, 1, 8);
        KvArena bad;
        bad.add_region(x, 0);
        bad.add_region(y, 100);   // disjoint, so the plan below can name a base the arena does not hold
        const std::vector<WindowSequence> wrong_base{{.sequence = 0, .kv_base = 1000, .first_position = 0, .rows = 1}};
        const auto bad_plan = build_window_row_plan(wrong_base, 4096, 4096, 8);
        check(has(refusal([&] { (void)stage_arena_rows(bad, bad_plan, c, project(1, 0.1f), window, norms()); }),
                  "no region is registered at base"),
              "a plan whose KV base is not the arena's region is refused");
        DenseKvState z(capacity, 1, 8);
        KvArena one;
        one.add_region(z, 0);
        const std::vector<WindowSequence> one_seq{{.sequence = 0, .kv_base = 0, .first_position = 0, .rows = 9}};
        check(has(refusal([&] {
                  const auto wide = build_window_row_plan(one_seq, 4096, 4096, 128);
                  (void)stage_arena_rows(one, wide, c, project(9, 0.1f), window, norms());
              }), "outside 1..kQsaWindowRows"),
              "a window wider than the native chunk is refused");
        const std::vector<WindowSequence> skip{{.sequence = 0, .kv_base = 0, .first_position = 2, .rows = 1}};
        const auto skipped = build_window_row_plan(skip, 4096, 4096, 8);
        check(has(refusal([&] { (void)stage_arena_rows(one, skipped, c, project(1, 0.1f), window, norms()); }),
                  "next cell"),
              "a row that is not its pool's next cell is refused");
    }

    // A cancellation in the middle of the window leaves every pool exactly as it was.
    {
        DenseKvState x(capacity, 1, 8), y(capacity, 1, 8);
        fill(x, 4, 0.2f);
        fill(y, 2, 0.6f);
        KvArena cancel_arena;
        cancel_arena.add_region(x, 0);
        cancel_arena.add_region(y, 1000);
        uint32_t polls = 0;
        bool rolled_back = false;
        try {
            (void)stage_arena_rows(cancel_arena, plan, c, projected, window, norms(), [&] { return ++polls > 2; });
        } catch (const strata::hpu::Cancelled&) {
            rolled_back = x.size() == 4 && y.size() == 2;
        }
        check(rolled_back, "a cancellation restores every pool to its starting size");
    }

    if (failures != 0) {
        std::cout << "FAILED arena-window checks: " << failures << "\n";
        return 1;
    }
    std::cout << "PASS arena window: two sequences staged in one call match the single-sequence path row for row "
                 "(queries, gates, gathered key/value), each row gathers only its own causal prefix and its padding is "
                 "masked, the pools grow by exactly their own rows, and a base mismatch, an over-wide window, a "
                 "non-next cell and a mid-window cancellation are all refused/rolled back; no device\n";
    return 0;
}
