#pragma once
// hpu/include/strata/hpu/window_rows.hpp - the ROW PLAN of a window that carries more than one sequence.
//
// Today a window belongs to one sequence: `SessionFlow::prefill_window` takes 1..128 rows and `stage_dense_window`
// appends them contiguously into one KV state. Multi-SEQUENCE row batching needs the same graph to carry rows from
// several sequences, each row appending to its own KV range, and each sequence sampling its own last row. This header
// is the pure part of that: the row table, its invariants, and where each sequence's sampling row is.
//
// It is deliberately SDK-free and allocation-light so a host test can pin the laws before any kernel changes. The
// invariants below are the ones a batched attention pass depends on; a plan that violates one would silently mix two
// sequences' histories, which is why every check throws instead of clamping.
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace strata::hpu {
inline constexpr uint32_t kWindowRowsMax = 128;      // prefill_window's bound
inline constexpr uint32_t kWindowVerifyRowsMax = 7;  // verify_window's bound
inline constexpr uint32_t kWindowSequencesMax = 64;  // same bound as the slot count

// One sequence's contribution to the window. `first_position` is the absolute position of the segment's first row, so
// the segment covers positions [first_position, first_position + rows); `kv_base` is that sequence's first KV cell, so
// the same segment covers cells [kv_base + first_position, kv_base + first_position + rows) - the layout upstream's
// per-slot state has (BSlot.ids/cvec, src/program/generate.cpp:9823-9850).
struct WindowSequence {
    uint32_t sequence = 0;        // caller's id; must be unique inside the plan
    uint32_t kv_base = 0;         // first KV cell of this sequence in its pool
    uint32_t first_position = 0;  // absolute position of the segment's first row
    uint32_t rows = 0;            // 1..(max_rows - already planned rows)
};

struct WindowRowPlan {
    std::vector<uint32_t> row_sequence;   // [rows] index into `sequences`
    std::vector<uint32_t> row_position;   // [rows] absolute position
    std::vector<uint32_t> segment_begin;  // [N] first row of each sequence's segment
    std::vector<uint32_t> segment_rows;   // [N] rows of each segment (contiguous, in plan order)
    std::vector<uint32_t> sample_rows;    // [N] the row whose logits that sequence samples (its last)
    std::vector<uint32_t> kv_base;        // [N] copied from the input for the caller's convenience

    uint32_t rows() const { return static_cast<uint32_t>(row_sequence.size()); }
    uint32_t sequences() const { return static_cast<uint32_t>(segment_rows.size()); }
};

namespace window_rows_detail {
inline void require(const bool condition, const std::string& why) {
    if (!condition) throw std::invalid_argument("window row plan: " + why);
}
} // namespace window_rows_detail

// Builds the table. Refusals (all std::invalid_argument, each naming the violated law):
//   * no sequence, or more sequences than kWindowSequencesMax;
//   * max_rows outside 1..kWindowRowsMax, or a total that exceeds max_rows;
//   * a sequence with zero rows, or a duplicate sequence id;
//   * a segment that would pass `context` (positions) or `cell_capacity` (KV cells);
//   * a segment whose positions would wrap (first_position + rows overflows 32 bits).
inline WindowRowPlan build_window_row_plan(const std::span<const WindowSequence> sequences, const uint32_t context,
                                           const uint32_t cell_capacity, const uint32_t max_rows) {
    using window_rows_detail::require;
    require(!sequences.empty(), "a window needs at least one sequence");
    require(sequences.size() <= kWindowSequencesMax, "too many sequences for one window");
    require(max_rows >= 1 && max_rows <= kWindowRowsMax, "max_rows must be in 1..kWindowRowsMax");

    WindowRowPlan plan;
    plan.segment_begin.reserve(sequences.size());
    plan.segment_rows.reserve(sequences.size());
    plan.sample_rows.reserve(sequences.size());
    plan.kv_base.reserve(sequences.size());

    uint32_t rows = 0;
    for (size_t s = 0; s < sequences.size(); ++s) {
        const WindowSequence& sequence = sequences[s];
        require(sequence.rows >= 1, "a sequence contributes at least one row");
        for (size_t earlier = 0; earlier < s; ++earlier)
            require(sequences[earlier].sequence != sequence.sequence, "duplicate sequence id in one window");
        require(sequence.first_position <= context, "a segment starts past the context");
        require(sequence.first_position + sequence.rows > sequence.first_position, "a segment's positions wrap");
        require(sequence.first_position + sequence.rows <= context, "a segment passes the context");
        require(sequence.kv_base <= cell_capacity, "a segment's KV base is past the pool");
        require(sequence.kv_base + sequence.first_position <= cell_capacity, "a segment's KV start is past the pool");
        require(sequence.kv_base + sequence.first_position + sequence.rows >
                    sequence.kv_base + sequence.first_position, "a segment's KV range wraps");
        require(sequence.kv_base + sequence.first_position + sequence.rows <= cell_capacity,
                "a segment passes the KV pool");
        require(rows + sequence.rows <= max_rows, "the window exceeds max_rows");

        plan.segment_begin.push_back(rows);
        plan.segment_rows.push_back(sequence.rows);
        plan.sample_rows.push_back(rows + sequence.rows - 1);   // the segment's LAST row
        plan.kv_base.push_back(sequence.kv_base);
        for (uint32_t row = 0; row < sequence.rows; ++row) {
            plan.row_sequence.push_back(static_cast<uint32_t>(s));
            plan.row_position.push_back(sequence.first_position + row);
        }
        rows += sequence.rows;
    }
    require(plan.row_sequence.size() == rows, "the row table and the segment table disagree");
    require(plan.row_position.size() == rows, "the position table and the row table disagree");
    return plan;
}
} // namespace strata::hpu
