#include "strata/hpu/kv_residency.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>

// The upstream KV-cache residency controls, transcribed.  Every rule below names the upstream line it
// comes from; where this backend has no mechanism to copy, the function says so instead of inventing one
// (see the DECLARED GAPS block in strata/hpu/kv_residency.hpp).
namespace strata::hpu {
namespace {

uint64_t capacity_pages(const KvArenaShape& s) {
    // layer.cpp:534 `p.n_slots`: the arena's page slots, i.e. what --max-context rounds up to.
    const uint64_t page = s.page_size ? s.page_size : 1;
    return (s.capacity_cells + page - 1) / page;
}

uint64_t round_up(uint64_t value, uint64_t unit) { return unit ? (value + unit - 1) / unit * unit : value; }

bool shape_valid(const KvArenaShape& s) {
    if (!s.page_size || s.page_size > UINT64_MAX / 2 || !s.capacity_cells || s.capacity_cells > UINT64_MAX / 2)
        return false;
    if (!s.granularity_bytes || !s.arrays || s.arrays > 4) return false;
    for (uint32_t a = 0; a < s.arrays; ++a)
        if (!s.per_page[a] || s.per_page[a] > UINT64_MAX / (capacity_pages(s) ? capacity_pages(s) : 1)) return false;
    return true;
}

}  // namespace

bool kv_growth_enabled(const KvResidencyOptions& o) {
    // generate.cpp:4398: `asked && ... && o.kv_resident <= 0 && ...`.  A resident floor and an elastic
    // arena are mutually exclusive upstream, so the floor wins and growth is reported off.
    return o.grow && o.resident_cells <= 0;
}

uint64_t kv_pool_slots(const KvArenaShape& s, uint64_t cells) {
    const uint64_t page = s.page_size ? s.page_size : 1;
    const uint64_t wanted = (cells + page - 1) / page;   // layer.cpp:534
    return std::min<uint64_t>(capacity_pages(s), std::max<uint64_t>(wanted, 1));
}

uint64_t kv_pool_chunks(const KvArenaShape& s, uint64_t array, uint64_t slots) {
    // layer.cpp:529-532: `(int64_t)(((uint64_t)slots * p.per_slot[a] + G - 1) / G)`
    if (array >= s.arrays) return 0;
    const uint64_t G = s.granularity_bytes ? s.granularity_bytes : 1;
    return ((uint64_t)slots * s.per_page[array] + G - 1) / G;
}

uint64_t kv_elastic_need(const KvArenaShape& s, uint64_t mapped_cells, uint64_t target_cells) {
    // layer.cpp:585-596 (qsa_kv_elastic_need) over this backend's contiguous arena: the chunks the target
    // range has and the mapped range has not.  Upstream counts literally-unmapped chunks; pool_grow maps
    // from chunk c0 upwards in order (layer.cpp:552-558), so the unmapped part is exactly the suffix past
    // what is mapped, and clamping at zero keeps the difference honest if a caller reports more than the
    // target.
    const uint64_t mapped_slots = kv_pool_slots(s, mapped_cells), target_slots = kv_pool_slots(s, target_cells);
    uint64_t need = 0;
    for (uint32_t a = 0; a < s.arrays; ++a)
        need += kv_pool_chunks(s, a, target_slots) > kv_pool_chunks(s, a, mapped_slots)
                    ? uint64_t(kv_pool_chunks(s, a, target_slots) - kv_pool_chunks(s, a, mapped_slots)) : 0;
    return need;
}

uint64_t kv_grow_target(const KvArenaShape& s, const KvResidencyOptions& o, uint64_t request_cells) {
    // generate.cpp:6790: `std::min<int64_t>(o.max_context, (cells + kvg.step - 1) / kvg.step * kvg.step)`
    const uint64_t stepped = round_up(request_cells, o.step ? o.step : 1);
    return std::min<uint64_t>(s.capacity_cells, stepped);
}

uint64_t kv_arena_bytes(const KvArenaShape& s, uint64_t cells) {
    // layer.cpp:652-661 (kv_pool_bytes) for the fp16 rows this backend stores, plus the pool's 64-byte
    // tail that every upstream branch of that function adds.
    if (!cells) return 0;
    const uint64_t slots = kv_pool_slots(s, cells);
    uint64_t bytes = 0;
    for (uint32_t a = 0; a < s.arrays; ++a) bytes += slots * s.per_page[a];
    return bytes + 64;
}

bool kv_cap_admits(const KvResidencyOptions& o, uint64_t arena_bytes) {
    // The cap is INCLUSIVE: `--kv` of exactly N MiB admits N MiB of arena and refuses N MiB + 1 byte.
    return !o.cap_bytes || arena_bytes <= o.cap_bytes;
}

uint64_t kv_request_cells(uint64_t prompt_cells, uint64_t max_new, uint64_t slack) {
    // generate.cpp:13191/13230/13282 `kvg_ensure(n_prompt + o.max_new + 64, ...)`; the token path asks
    // `n + 256` (10885) and a session load `live.ids.size() + 256` (10595/11065).
    return prompt_cells + max_new + slack;
}

KvDecision kv_admit(const KvArenaShape& s, uint64_t mapped_cells, uint64_t free_bytes, uint64_t request_cells,
                    const KvResidencyOptions& o) {
    KvDecision d;
    if (!shape_valid(s) || o.resident_cells < 0) { d.refusal = KvRefusal::InvalidGeometry; return d; }
    const uint64_t held = std::min<uint64_t>(mapped_cells, s.capacity_cells);
    if (!kv_growth_enabled(o)) {
        // The arena is what it is: upstream's non-elastic path allocated --max-context up front
        // (generate.cpp:6754-6760), so kvg_ensure is a no-op for anything the fixed pools already hold
        // (6789).  6790's min-with-max_context clamp applies here too: a request past --max-context is the
        // context admission's business, never paid by taking someone else's cells.  What refuses here is a
        // request past what the arena COMMITS, and nothing is evicted on this path at all.
        const uint64_t wanted = std::min<uint64_t>(request_cells, s.capacity_cells);
        d.target_cells = held;
        d.arena_bytes = kv_arena_bytes(s, held);
        d.refusal = wanted > held ? KvRefusal::GrowthDisabled : KvRefusal::None;
        d.admits = wanted <= held;
        d.pinned_released = 0;
        d.slots_released = 0;
        return d;
    }
    const uint64_t grown = kv_grow_target(s, o, request_cells);
    const uint64_t target = std::max<uint64_t>(held, grown);   // generate.cpp:6789: never shrinks here
    const uint64_t need = kv_elastic_need(s, held, target);    // generate.cpp:6791
    d.target_cells = target;
    d.chunks_needed = need;
    d.bytes_needed = need > UINT64_MAX / s.granularity_bytes ? UINT64_MAX : need * s.granularity_bytes;
    d.arena_bytes = kv_arena_bytes(s, target);
    if (!kv_cap_admits(o, d.arena_bytes)) { d.refusal = KvRefusal::CapExceeded; return d; }
    if (need == 0) {   // generate.cpp:6792: nothing to map, accept without growing
        d.admits = true;
        d.refusal = KvRefusal::None;
        return d;
    }
    // generate.cpp:6816-6850 pays for the chunks by giving expert-cache slots up, down to kvg.floor
    // (6764, 6816).  This backend has no lendable slot range (gap 1), so a grow can only be paid for out
    // of room the allocator still has - exactly the branch upstream falls into when the cache is already at
    // its floor and the chunks must be fresh memory (6871 "the cache is at its floor: new memory"), which
    // then fails at layer.cpp:558-562 and is reported as "the K/V cannot grow: no VRAM is left"
    // (generate.cpp:12649).
    d.slots_released = 0;
    d.pinned_released = 0;
    if (d.bytes_needed > free_bytes) { d.refusal = KvRefusal::NoRoom; return d; }
    d.admits = true;
    d.grows = true;
    d.refusal = KvRefusal::None;
    return d;
}

uint64_t kv_resident_pages(uint64_t max_cells, uint64_t resident_cells, uint64_t page_size) {
    const uint64_t page = page_size ? page_size : 1;
    const uint64_t pages = (max_cells + page - 1) / page;
    if (resident_cells == 0) return pages;   // layer.cpp:638: 0 keeps every cell in the arena
    // layer.cpp:647-648: the floor is max(N, 20480) rounded up to pages, and a context of N cells or fewer
    // is not streamed at all (726-728).
    const uint64_t r = (std::max<uint64_t>(resident_cells, kv_resident_min_cells) + page - 1) / page;
    return r < pages ? r : pages;
}

uint64_t kv_resident_pinned_cells(const KvArenaShape& s, const KvResidencyOptions& o) {
    if (!shape_valid(s) || o.resident_cells < 0) return 0;
    const uint64_t page = s.page_size ? s.page_size : 1;
    const uint64_t pinned_pages = kv_resident_pages(s.capacity_cells, uint64_t(std::max<int64_t>(o.resident_cells, 0)), page);
    return std::min<uint64_t>(pinned_pages * page, kv_arena_commit_cells(s));
}

uint64_t kv_arena_commit_cells(const KvArenaShape& s) { return capacity_pages(s) * (s.page_size ? s.page_size : 1); }

uint64_t kv_trim_target(uint64_t request_cells, uint64_t hold_cells, const KvArenaShape& s,
                        const KvResidencyOptions& o) {
    if (!shape_valid(s)) return 0;
    // generate.cpp:6897-6898: the K/V keeps the longest parked conversation's cells (#1128) and rounds down
    // to whole steps, never below one step.
    const uint64_t step = o.step ? o.step : 1;
    const uint64_t want = std::max<uint64_t>(request_cells, hold_cells);
    const uint64_t trimmed = std::max<uint64_t>(step, round_up(want, step));
    // Upstream has no residency pin to respect here - the two mechanisms are exclusive (4398) - so this
    // clamp is this backend's own rule: a pin, once set, is never handed back.
    const uint64_t pinned = o.resident_cells > 0 ? kv_resident_pinned_cells(s, o) : 0;
    // The ceiling is the arena's page-rounded capacity, the same quantity the pin is clamped by: a trim can
    // never hand back cells the arena does not have, and never the pinned ones.
    const uint64_t arena = kv_arena_commit_cells(s);
    return std::min<uint64_t>(arena, std::max<uint64_t>(trimmed, pinned));
}

void validate_kv_residency(const KvArenaShape& s, const KvResidencyOptions& o) {
    // generate.cpp:2559-2562: `--kv-resident must be >= 0`.
    if (o.resident_cells < 0) throw std::invalid_argument("--kv-resident must be >= 0");
    if (o.cap_bytes && !s.capacity_cells) throw std::invalid_argument("--kv needs a KV arena to cap");
    if (!shape_valid(s)) throw std::invalid_argument("invalid KV arena geometry for the residency policy");
    // The cap is checked against the arena as configured, because this backend allocates it whole: what
    // upstream would discover mid-run at layer.cpp:558-562 is a plan-time refusal here.
    if (!kv_cap_admits(o, kv_arena_bytes(s, kv_arena_commit_cells(s))))
        throw std::invalid_argument("--kv: the K/V arena needs " + std::to_string(kv_arena_bytes(s, kv_arena_commit_cells(s))) +
                                    " bytes, the cap is " + std::to_string(o.cap_bytes));
}

namespace {
KvResidencyOptions& storage() {
    static KvResidencyOptions options;
    return options;
}
}

void set_kv_residency(const KvResidencyOptions& o) {
    // Upstream's own shape for these knobs: a process-level setter called by the flag parser
    // (layer.cpp:664 `qsa_set_kv_resident`, layer.cpp:575-578 `qsa_set_kv_elastic`).
    if (o.resident_cells < 0) throw std::invalid_argument("--kv-resident must be >= 0");
    storage() = o;
}
const KvResidencyOptions& kv_residency() { return storage(); }
uint64_t kv_resident_cells() { return std::max<int64_t>(storage().resident_cells, 0); }
uint64_t kv_resident_minimum() { return kv_resident_min_cells; }

}
