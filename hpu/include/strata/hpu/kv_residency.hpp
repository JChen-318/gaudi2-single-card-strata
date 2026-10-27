#pragma once
// The upstream KV-cache residency controls, as one pure policy unit: --kv (the hard byte cap),
// --kv-grow / --no-kv-grow (elastic growth) and --kv-resident (the resident floor).
//
// Upstream authorities (pinned source, commit 296b59d):
//   src/program/generate.cpp:405-407      the three option fields (kv encoding, resident cells, grow ask)
//   src/program/generate.cpp:1950-1953    the parse order; --no-kv-grow simply clears the ask, last flag wins
//   src/program/generate.cpp:2547,2559-2563  --kv takes fp16|int8|q4_0|k8v4; --kv-resident must be >= 0
//   src/program/generate.cpp:4394-4408    STRATA_KV_GROW=1/0 overrides the flag; elastic growth REQUIRES
//                                         --kv-resident <= 0 (4398), so the two mechanisms are exclusive
//   src/program/generate.cpp:6761-6785    struct KvGrow: step 8192 cells (6765), the cache keeps a floor of
//                                         128 slots (6764), the arena starts at 16384 cells (4407)
//   src/program/generate.cpp:6788-6889    kvg_ensure: the growth decision itself
//                                         cells <= held -> nothing to do (6789); target = min(max_context,
//                                         round_up(cells,step)) (6790); need = unmapped chunks for that
//                                         target (6791); need == 0 -> accept without growing (6792); else the
//                                         cache hands out slots down to its floor (6816-6850) and the pool
//                                         maps the chunks, failing only when the driver has no VRAM left
//                                         (src/core/layer.cpp:558-562, reported at generate.cpp:12649)
//   src/program/generate.cpp:6892-6900    kvg_trim: a shorter request hands cells back, never below one step
//   src/core/layer.cpp:529-535,537-548,585-596  the exact slot/chunk arithmetic this unit transcribes
//   src/core/layer.cpp:633-651,664-666    kv_plan/qsa_set_kv_resident: what --kv-resident pins, and the
//                                         20480-cell minimum; a context of <= that many cells is not streamed
//
// DECLARED GAPS (not simulated here, and not faked anywhere in this backend):
//   1. This backend has no CUDA VMM range for the K/V.  The arena is ONE host allocation sized by
//      --max-context (hpu/src/model_plan.cpp:87-91, hpu/src/qsa_state.cpp:103-106); nothing maps or
//      unmaps chunks at runtime.  So kvg_ensure's slot-lending/eviction loop (generate.cpp:6816-6850)
//      has no counterpart, --kv-grow's "the expert cache gives slots back as the context grows"
//      (generate.cpp:729-732) cannot happen, and the trim path (6892-6900) has nothing to hand back.
//      What IS real, and implemented: the hard cap, the no-growth admission refusal (the arena is what it
//      is: a request past it is refused before any pinned state is touched), and the residency pin.
//   2. Upstream's --kv selects a cell ENCODING (fp16/int8/q4_0/k8v4, src/core/layer.cpp:652-661), not a
//      size; there is no `kv_mib` anywhere upstream (greps clean).  This backend stores FP16 cells only,
//      so --kv is bound here to the enforceable half of the same idea: a hard, inclusive MiB cap on the
//      K/V arena.  The encoding selector is NOT ported and is not pretended to be.
//   3. --kv-resident's second tier (the whole K/V in pinned RAM, src/core/layer.cpp:677) does not exist
//      here: the cells already live in host RAM.  The floor is therefore implemented as a pin the
//      allocator may never exceed and never trim below, not as a RAM/VRAM split.
#include <cstdint>

namespace strata::hpu {

// generate.cpp:6764 (kvg.floor) and layer.cpp:666 (qsa_kv_resident_min).
inline constexpr uint64_t kv_resident_min_cells = 20480;
inline constexpr uint64_t kv_cache_floor_slots = 128;
inline constexpr uint64_t kv_grow_step_cells = 8192;      // generate.cpp:6765 (kvg.step)
inline constexpr uint64_t kv_grow_init_cells = 16384;     // generate.cpp:4407 (STRATA_KV_GROW_INIT default)

/// The four knobs.  Defaults are upstream's: `--kv` uncapped, growth OFF (generate.cpp:407 "kv_grow = false"),
/// nothing streamed (generate.cpp:406 "kv_resident = 0").
struct KvResidencyOptions {
    uint64_t cap_bytes = 0;      // --kv: hard INCLUSIVE cap on the K/V arena in bytes; 0 = no cap
    bool grow = false;           // --kv-grow sets it, --no-kv-grow clears it (last flag wins, 1952-1953)
    int64_t resident_cells = 0;  // --kv-resident N: 0 keeps every cell in the arena; negative is refused
    uint64_t step = kv_grow_step_cells;
    uint64_t granularity_bytes = 2ull << 20;  // strata::core::vmm_granularity(): growth is paid per chunk
    uint64_t cache_floor_slots = kv_cache_floor_slots;
};

/// Upstream's ElasticPool geometry reduced to what the decision reads: the page size, the chunk
/// granularity, the cell count the arena can never exceed, and the per-page bytes of each array
/// (layer.cpp:524 `per_slot`, the arrays the K/V pool is made of).
struct KvArenaShape {
    uint64_t page_size = 4;
    uint64_t capacity_cells = 0;   // o.max_context, the clamp at generate.cpp:6790
    uint32_t arrays = 1;
    uint64_t per_page[4] = {0, 0, 0, 0};
    uint64_t granularity_bytes = 2ull << 20;
};

enum class KvRefusal : uint8_t { None, GrowthDisabled, CapExceeded, NoRoom, InvalidGeometry };

struct KvDecision {
    bool admits = false;
    bool grows = false;
    uint64_t target_cells = 0;
    uint64_t chunks_needed = 0;
    uint64_t bytes_needed = 0;
    uint64_t arena_bytes = 0;
    uint64_t slots_released = 0;   // cache slots the growth paid itself out of (generate.cpp:6828-6849)
    uint64_t pinned_released = 0;  // always 0: nothing here, and nothing upstream, evicts pinned state
    KvRefusal refusal = KvRefusal::InvalidGeometry;
};

/// generate.cpp:4398-4402: the elastic arena needs a resident floor of zero.  With --kv-resident set,
/// growth is off whatever the flag says.
bool kv_growth_enabled(const KvResidencyOptions&);

/// layer.cpp:533-535: `pool_slots` - page slots for `cells`, at least one, never past the arena.
uint64_t kv_pool_slots(const KvArenaShape&, uint64_t cells);
/// layer.cpp:529-532: `pool_chunks` - whole chunks one array needs for that many page slots.
uint64_t kv_pool_chunks(const KvArenaShape&, uint64_t array, uint64_t slots);
/// layer.cpp:585-596: `qsa_kv_elastic_need` - chunks the target range has that the arena has not mapped.
uint64_t kv_elastic_need(const KvArenaShape&, uint64_t mapped_cells, uint64_t target_cells);
/// generate.cpp:6790: `min(max_context, round_up(cells, step))`.
uint64_t kv_grow_target(const KvArenaShape&, const KvResidencyOptions&, uint64_t request_cells);
/// src/core/layer.cpp:652-661 with the fp16 row: every array's bytes for that many cells.
uint64_t kv_arena_bytes(const KvArenaShape&, uint64_t cells);
/// The cap is INCLUSIVE: exactly `cap_bytes` of arena is admitted, one byte more is not.
bool kv_cap_admits(const KvResidencyOptions&, uint64_t arena_bytes);
/// generate.cpp:13191/13230/13282 ask for `n_prompt + max_new + 64` cells, 10885 for `n + 256`.
uint64_t kv_request_cells(uint64_t prompt_cells, uint64_t max_new, uint64_t slack);

/// The growth decision for one request.  `mapped_cells` is what the arena commits now
/// (qsa_kv_elastic_cells(), layer.cpp:580-584) and `free_bytes` the room a grow could still take
/// (the cache's spare chunks plus fresh memory, generate.cpp:6871).
KvDecision kv_admit(const KvArenaShape&, uint64_t mapped_cells, uint64_t free_bytes, uint64_t request_cells,
                   const KvResidencyOptions&);

/// layer.cpp:647-648: the page slots --kv-resident pins, clamped by the 20480 minimum and by the arena
/// itself (a context that small or smaller is not streamed at all).
uint64_t kv_resident_pages(uint64_t max_cells, uint64_t resident_cells, uint64_t page_size);

/// The cells the pin holds, never more than the arena itself.
uint64_t kv_resident_pinned_cells(const KvArenaShape&, const KvResidencyOptions&);
/// What this backend commits for the arena at start: the page-rounded capacity, because the arena is one
/// fixed allocation (gap 1 above).  The pin is always a subset of it, never larger.
uint64_t kv_arena_commit_cells(const KvArenaShape&);

/// generate.cpp:6897-6898 (kvg_trim): where a shorter request hands cells back to, never past one step
/// and never below the resident pin.  Returns the cells the arena keeps.
uint64_t kv_trim_target(uint64_t request_cells, uint64_t hold_cells, const KvArenaShape&,
                        const KvResidencyOptions&);

/// Refuses what upstream refuses: a negative --kv-resident (generate.cpp:2559-2562) and a cap that the
/// arena as configured cannot fit.
void validate_kv_residency(const KvArenaShape&, const KvResidencyOptions&);

/// The process-level knob, in upstream's own shape (layer.cpp:575-579 `qsa_set_kv_elastic`,
/// layer.cpp:664-666 `qsa_set_kv_resident`/`qsa_kv_resident`): the tool sets it once, the plan and the
/// request admission read it.
void set_kv_residency(const KvResidencyOptions&);
const KvResidencyOptions& kv_residency();
uint64_t kv_resident_cells();      // qsa_kv_resident()
uint64_t kv_resident_minimum();    // qsa_kv_resident_min()

}
