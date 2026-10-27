#pragma once
// hpu/include/strata/hpu/kv_arena.hpp - several sequences' dense KV pools behind ONE global cell space.
//
// The dense KV state (`DenseKvState`) owns ONE contiguous pool and appends rows in order; a window over several
// sequences needs each row to land in its own sequence's range. Rather than teach the storage class about sequences,
// an arena hands out disjoint cell ranges and routes an absolute cell to the region that owns it:
//
//     KvArena arena;                       // global cell space [0, sum of capacities)
//     arena.add_region(state_a, 0);        // sequence A: cells [0, cap_a)
//     arena.add_region(state_b, cap_a);    // sequence B: cells [cap_a, cap_a + cap_b)
//     arena.append_at(global_cell, key, value);   // the row plan's kv_base + position
//     arena.gather_span(region_id, cells, bucket); // that sequence's causal prefix, through its own state
//
// `DenseKvState` keeps its own KV FORMAT (fp16 / int8 / q4_0 / k8v4) and its own checkpoint/snapshot paths, so the
// arena adds addressing only. It is host-only (the dense KV cells live in host memory), which is why a host test can
// pin the addressing laws without a device. The laws:
//   * regions are disjoint, non-empty, and their bases are strictly increasing;
//   * a cell belongs to exactly one region;
//   * append_at refuses a cell that is not the owning region's NEXT cell (the row plan never interleaves a sequence's
//     rows with another sequence's, so a gap means the plan and the arena disagree - refuse rather than guess);
//   * gather_span refuses a span wider than the region's owner holds.
#include "strata/hpu/dense_kv.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace strata::hpu {
class KvArena {
public:
    struct Region {
        DenseKvState* state = nullptr;
        uint32_t base = 0;
    };

    // Registers `state`'s pool at `base`. Refusals: a null state, a pool with zero capacity, a base that is not
    // strictly past the previously registered region, or any overlap with it.
    uint32_t add_region(DenseKvState& state, const uint32_t base) {
        if (state.capacity() == 0) throw std::invalid_argument("KV arena: a region needs a non-empty pool");
        if (!regions_.empty()) {
            const Region& last = regions_.back();
            if (base <= last.base) throw std::invalid_argument("KV arena: region bases must strictly increase");
            if (base < last.base + last.state->capacity())
                throw std::invalid_argument("KV arena: region ranges overlap");
        }
        regions_.push_back({&state, base});
        return static_cast<uint32_t>(regions_.size() - 1);
    }

    uint32_t regions() const { return static_cast<uint32_t>(regions_.size()); }

    const Region& region(const uint32_t id) const {
        if (id >= regions_.size()) throw std::invalid_argument("KV arena: region id out of range");
        return regions_[id];
    }

    // The region registered at exactly `base`: the row plan names a base, and a window may carry only SOME slots, so
    // the window's sequence index is not the arena's region index.
    uint32_t region_with_base(const uint32_t base) const {
        for (uint32_t r = 0; r < regions_.size(); ++r)
            if (regions_[r].base == base) return r;
        throw std::invalid_argument("KV arena: no region is registered at base " + std::to_string(base));
    }

    // The region that owns `cell`, or a refusal when the cell is outside every range.
    uint32_t region_of(const uint32_t cell) const {
        for (uint32_t r = 0; r < regions_.size(); ++r) {
            const Region& candidate = regions_[r];
            if (cell >= candidate.base && cell < candidate.base + candidate.state->capacity()) return r;
        }
        throw std::invalid_argument("KV arena: cell " + std::to_string(cell) + " belongs to no region");
    }

    // Appends one row at an absolute cell. The cell must be the owning region's next free cell: a row plan writes each
    // sequence's rows in order, so a gap means the plan and the arena disagree.
    void append_at(const uint32_t cell, const std::span<const float> key, const std::span<const float> value) {
        const Region& owner = region(region_of(cell));
        if (cell - owner.base != owner.state->size())
            throw std::invalid_argument("KV arena: cell " + std::to_string(cell) + " is not region " +
                                        std::to_string(region_of(cell)) + "'s next cell (it holds " +
                                        std::to_string(owner.state->size()) + ")");
        owner.state->append(key, value);
    }

    // One sequence's causal prefix through its own state (so its KV format applies), refusing a span the state cannot
    // hold - a batch that asks for another sequence's rows must fail here, not silently read them.
    SelectedKv gather_span(const uint32_t id, const uint32_t cells, const uint32_t bucket, const uint32_t window = 0) const {
        const Region& owner = region(id);
        if (cells > owner.state->size())
            throw std::invalid_argument("KV arena: region " + std::to_string(id) + " holds " +
                                        std::to_string(owner.state->size()) + " cells, not " + std::to_string(cells));
        return owner.state->gather_prefix(cells, bucket, window);
    }

    // The absolute cell a sequence's next row lands on: the row plan's `kv_base + position`.
    uint32_t next_cell(const uint32_t id) const {
        const Region& owner = region(id);
        return owner.base + owner.state->size();
    }

private:
    std::vector<Region> regions_;
};
} // namespace strata::hpu
