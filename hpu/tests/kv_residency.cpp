// Host test for the KV-cache residency controls (--kv / --kv-grow / --no-kv-grow / --kv-resident).
// The growth decision is compared, as one whole set, against an independently transcribed copy of the
// upstream expression (src/program/generate.cpp:6788-6792 + src/core/layer.cpp:529-535,585-596), so the
// port's policy cannot drift from the pinned source without this test failing.
#include "strata/hpu/kv_residency.hpp"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <string>
#include <tuple>
#include <vector>

namespace {
using strata::hpu::KvArenaShape;
using strata::hpu::KvDecision;
using strata::hpu::KvRefusal;
using strata::hpu::KvResidencyOptions;

int failures = 0;
void check(bool ok, const std::string& what) {
    if (!ok) { ++failures; std::cout << "FAIL " << what << "\n"; }
}

// The port's decision and the transcription reduced to the same comparable tuple:
// admits, grows, target cells, chunks needed, refusal code (0 none, 1 growth off, 2 cap, 3 no room).
struct Outcome {
    bool admits, grows;
    uint64_t target, chunks;
    int reason;
};
bool operator==(const Outcome& a, const Outcome& b) {
    return std::tie(a.admits, a.grows, a.target, a.chunks, a.reason) ==
           std::tie(b.admits, b.grows, b.target, b.chunks, b.reason);
}
Outcome from(const KvDecision& d) {
    int reason = 0;
    switch (d.refusal) {
        case KvRefusal::None: reason = 0; break;
        case KvRefusal::GrowthDisabled: reason = 1; break;
        case KvRefusal::CapExceeded: reason = 2; break;
        case KvRefusal::NoRoom: reason = 3; break;
        case KvRefusal::InvalidGeometry: reason = 4; break;
    }
    return {d.admits, d.grows, d.target_cells, d.chunks_needed, reason};
}

// ---- the independently transcribed upstream expression -------------------------------------------
// Nothing here calls into strata/hpu/kv_residency.hpp: it is written straight from the pinned source.
struct UpstreamArena {
    uint64_t page, granularity, max_context, step;
    std::vector<uint64_t> per_slot;   // ElasticPool::per_slot, layer.cpp:524

    // layer.cpp:533-535: pool_slots = min(n_slots, max((cells + page - 1) / page, 1))
    uint64_t slots(uint64_t cells) const {
        const uint64_t n_slots = (max_context + page - 1) / page;
        const uint64_t wanted = std::max<uint64_t>((cells + page - 1) / page, 1);
        return std::min(n_slots, wanted);
    }
    // layer.cpp:529-532: pool_chunks = (slots * per_slot[a] + G - 1) / G
    uint64_t chunks(size_t array, uint64_t page_slots) const {
        return (page_slots * per_slot[array] + granularity - 1) / granularity;
    }
    // layer.cpp:585-596: qsa_kv_elastic_need counts the chunks of the target range that are not mapped.
    // The arena maps from chunk 0 upwards (pool_grow, layer.cpp:552-558), so the unmapped part is the suffix.
    uint64_t need(uint64_t mapped_cells, uint64_t target_cells) const {
        const uint64_t ms = slots(mapped_cells), ts = slots(target_cells);
        uint64_t total = 0;
        for (size_t a = 0; a < per_slot.size(); ++a) {
            const uint64_t c0 = chunks(a, ms), c1 = chunks(a, ts);
            total += c1 > c0 ? c1 - c0 : 0;
        }
        return total;
    }
    // layer.cpp:652-661 (kv_pool_bytes) for fp16 rows, with the +64 tail every branch of it adds.
    uint64_t bytes(uint64_t cells) const {
        if (!cells) return 0;
        const uint64_t page_slots = slots(cells);
        uint64_t total = 64;
        for (size_t a = 0; a < per_slot.size(); ++a) total += page_slots * per_slot[a];
        return total;
    }

    // generate.cpp:6788-6792 (kvg_ensure) plus the fresh-memory failure (layer.cpp:558-562, reported at
    // generate.cpp:12649) and this backend's --kv cap.
    Outcome ensure(uint64_t mapped, uint64_t free_bytes, uint64_t request, bool grow_enabled, uint64_t cap,
                   uint64_t* arena_bytes) const {
        if (!grow_enabled) {   // 6789, with the whole --max-context arena allocated at start (6754-6760)
            // 6790's min-with-max_context clamp: the arena can never hold past --max-context, so a request
            // past the context is the context admission's business; what refuses here is a request past
            // what the arena commits.
            const uint64_t wanted = std::min<uint64_t>(max_context, request);
            const bool fits = wanted <= mapped;
            if (arena_bytes) *arena_bytes = bytes(mapped);
            return {fits, false, mapped, 0, fits ? 0 : 1};
        }
        const uint64_t stepped = std::min<uint64_t>(max_context, (request + step - 1) / step * step);  // 6790
        const uint64_t target = std::max<uint64_t>(mapped, stepped);
        const uint64_t need_chunks = need(mapped, target);   // 6791
        const uint64_t arena = bytes(target);
        if (arena_bytes) *arena_bytes = arena;
        if (cap && arena > cap) return {false, false, target, need_chunks, 2};
        if (need_chunks == 0) return {true, false, target, 0, 0};   // 6792
        if (need_chunks > UINT64_MAX / granularity || need_chunks * granularity > free_bytes)
            return {false, false, target, need_chunks, 3};
        return {true, true, target, need_chunks, 0};
    }
};

KvArenaShape shape(uint64_t capacity, uint64_t page = 4) {
    KvArenaShape s;
    s.page_size = page;
    s.capacity_cells = capacity;
    s.arrays = 2;                             // the K array and the V array (qsa_state keys_/values_)
    s.per_page[0] = page * 2 * 256 * 2;       // kv_heads * head_dim, FP16
    s.per_page[1] = page * 2 * 256 * 2;
    s.granularity_bytes = 2ull << 20;
    return s;
}

UpstreamArena upstream(uint64_t capacity, uint64_t page = 4) {
    return UpstreamArena{page, 2ull << 20, capacity, strata::hpu::kv_grow_step_cells,
                         {page * 2 * 256 * 2, page * 2 * 256 * 2}};
}
}  // namespace

int main() {
    // ---- 1. the growth decision over a sweep of (used, free, request), as one whole set ----
    const uint64_t capacity = 131072;
    const std::vector<uint64_t> used = {0, 1, 2048, 16383, 16384, 40960, 65536, capacity};
    const std::vector<uint64_t> free_bytes = {0, 1ull << 20, 2ull << 20, 8ull << 20, 40ull << 20, UINT64_MAX};
    const std::vector<uint64_t> request = {0, 64, 4096, 8192, 8193, 20480, 65536, capacity, 200000};
    const std::vector<uint64_t> caps = {0, 16ull << 20, 64ull << 20, 256ull << 20};
    const UpstreamArena reference = upstream(capacity);
    std::vector<Outcome> ported, transcribed;
    std::vector<uint64_t> ported_arena_bytes, reference_arena_bytes;
    std::vector<bool> pinned_clean, slots_clean;
    for (bool grow : {false, true}) {
        for (uint64_t cap : caps) {
            KvResidencyOptions options;
            options.grow = grow;
            options.cap_bytes = cap;
            for (uint64_t u : used) {
                for (uint64_t f : free_bytes) {
                    for (uint64_t r : request) {
                        const KvDecision d = strata::hpu::kv_admit(shape(capacity), u, f, r, options);
                        uint64_t arena = 0;
                        const auto o = reference.ensure(u, f, r, grow, cap, &arena);
                        ported.push_back(from(d));
                        transcribed.push_back(o);
                        ported_arena_bytes.push_back(d.arena_bytes);
                        reference_arena_bytes.push_back(arena);
                        // nothing here, and nothing upstream, evicts pinned state to admit a request
                        pinned_clean.push_back(d.pinned_released == 0);
                        slots_clean.push_back(d.slots_released == 0);   // this port has no lendable slots
                        if (d.admits) check(d.refusal == KvRefusal::None, "an admitted decision names a refusal");
                        else check(d.refusal != KvRefusal::None, "a refused decision names no reason");
                    }
                }
            }
        }
    }
    const size_t cases = 2 * caps.size() * used.size() * free_bytes.size() * request.size();
    check(ported.size() == cases && transcribed.size() == cases,
          "the sweep did not cover every (used, free, request) triple twice over");
    check(ported == transcribed, "the growth decision diverged from the transcribed upstream expression");
    check(ported_arena_bytes == reference_arena_bytes, "the arena byte accounting diverged from kv_pool_bytes");
    check(std::all_of(pinned_clean.begin(), pinned_clean.end(), [](bool v) { return v; }),
          "a grow decision evicted pinned state");
    check(std::all_of(slots_clean.begin(), slots_clean.end(), [](bool v) { return v; }),
          "a grow decision released expert-cache slots this backend does not have");

    // ---- 2. --no-kv-grow: a request that needs growth is refused, never paid with pinned state ----
    {
        KvResidencyOptions off, on;
        off.grow = false;
        on.grow = true;
        const uint64_t held = 16384, need_cells = 40960;
        const auto refused = strata::hpu::kv_admit(shape(capacity), held, UINT64_MAX, need_cells, off);
        check(!refused.admits && refused.refusal == KvRefusal::GrowthDisabled, "--no-kv-grow refused the request");
        check(refused.pinned_released == 0 && refused.slots_released == 0,
              "--no-kv-grow evicted something instead of refusing");
        check(refused.target_cells == held && refused.chunks_needed == 0, "--no-kv-grow touched the arena anyway");
        const auto admitted = strata::hpu::kv_admit(shape(capacity), held, UINT64_MAX, need_cells, on);
        check(admitted.admits && admitted.grows, "--kv-grow admits the same request when there is room");
        check(admitted.chunks_needed == reference.need(held, admitted.target_cells),
              "--kv-grow mapped a different number of chunks than upstream's need()");
        check(admitted.target_cells == std::min<uint64_t>(capacity, (need_cells + 8191) / 8192 * 8192),
              "--kv-grow ignored generate.cpp:6790's round-up-to-step / min-with-max_context target");
        check(admitted.pinned_released == 0, "--kv-grow evicted pinned state");
        const auto starved = strata::hpu::kv_admit(shape(capacity), held, admitted.bytes_needed - 1, need_cells, on);
        check(!starved.admits && starved.refusal == KvRefusal::NoRoom,
              "a grow one byte short of room was admitted instead of refused");
        check(strata::hpu::kv_admit(shape(capacity), held, 0, held, off).admits,
              "--no-kv-grow refused a request the arena already holds");
        // upstream turns growth off whenever a resident floor is set (generate.cpp:4398)
        KvResidencyOptions floored;
        floored.grow = true;
        floored.resident_cells = 32768;
        check(!strata::hpu::kv_growth_enabled(floored), "--kv-resident left the elastic arena on");
        check(!strata::hpu::kv_admit(shape(capacity), held, UINT64_MAX, need_cells, floored).admits,
              "a resident floor with --kv-grow grew the arena anyway");
    }

    // ---- 3. the --kv cap is inclusive at its boundary and refuses the byte above ----
    {
        const uint64_t mib = 1ull << 20;
        const auto full = shape(capacity);
        const uint64_t arena = strata::hpu::kv_arena_bytes(full, strata::hpu::kv_arena_commit_cells(full));
        check(arena == reference.bytes(capacity), "the port's arena bytes disagree with kv_pool_bytes");
        KvResidencyOptions at_cap, over_cap;
        at_cap.grow = true;
        over_cap.grow = true;
        at_cap.cap_bytes = arena;
        over_cap.cap_bytes = arena - 1;
        check(strata::hpu::kv_cap_admits(at_cap, arena) && !strata::hpu::kv_cap_admits(at_cap, arena + 1),
              "the arena cap is not inclusive at its boundary");
        check(strata::hpu::kv_cap_admits(at_cap, arena - 1), "the cap refused a smaller arena than it allows");
        for (uint64_t whole : {100ull, 512ull, 65536ull}) {   // the MiB the flag is given in
            KvResidencyOptions o;
            o.cap_bytes = whole * mib;
            check(strata::hpu::kv_cap_admits(o, whole * mib) && !strata::hpu::kv_cap_admits(o, whole * mib + 1),
                  "the MiB cap boundary is not exact");
        }
        const auto fits = strata::hpu::kv_admit(full, 16384, UINT64_MAX, capacity, at_cap);
        const auto above = strata::hpu::kv_admit(full, 16384, UINT64_MAX, capacity, over_cap);
        check(fits.admits && fits.arena_bytes == at_cap.cap_bytes, "a grow to exactly the cap was refused");
        check(!above.admits && above.refusal == KvRefusal::CapExceeded, "a grow one byte past the cap was admitted");
        bool threw = false;
        try {
            strata::hpu::validate_kv_residency(full, over_cap);   // the arena is allocated whole here
        } catch (const std::exception&) { threw = true; }
        check(threw, "the plan-time cap check let an arena that does not fit through");
        threw = false;
        try {
            strata::hpu::validate_kv_residency(full, at_cap);
        } catch (const std::exception&) { threw = true; }
        check(!threw, "the plan-time cap check refused the very boundary it must admit");
        // a negative floor is refused exactly as upstream refuses it (generate.cpp:2559-2562)
        KvResidencyOptions negative;
        negative.resident_cells = -1;
        threw = false;
        try {
            strata::hpu::set_kv_residency(negative);
        } catch (const std::exception&) { threw = true; }
        check(threw, "--kv-resident accepted a negative value");
        check(strata::hpu::kv_admit(shape(capacity), 0, 0, 10, negative).refusal == KvRefusal::InvalidGeometry,
              "a negative floor slipped past the decision");
    }

    // ---- 4. the resident floor: pinned, and never exceeded by the allocator ----
    {
        using Row = std::tuple<uint64_t, uint64_t, uint64_t, uint64_t, uint64_t>;   // page, max, resident, pinned, commit
        std::vector<Row> ported_floor, transcribed_floor;
        for (uint64_t page : {1ull, 4ull, 16ull}) {
            for (uint64_t max_cells : {1024ull, 20479ull, 20480ull, 20481ull, 32768ull, 131072ull}) {
                for (uint64_t resident : {0ull, 1ull, 20479ull, 20480ull, 40960ull, 1000000ull}) {
                    const auto s = shape(max_cells, page);
                    KvResidencyOptions o;
                    o.resident_cells = int64_t(resident);
                    const uint64_t pinned = strata::hpu::kv_resident_pinned_cells(s, o);
                    const uint64_t commit = strata::hpu::kv_arena_commit_cells(s);
                    // layer.cpp:638,647-648, transcribed straight from the source: 0 pins everything, else
                    // max(N, 20480) rounded up to pages, and a context that small is not streamed at all.
                    const uint64_t pages = (max_cells + page - 1) / page;
                    const uint64_t floor_pages = resident == 0
                        ? pages
                        : std::min<uint64_t>((std::max<uint64_t>(resident, strata::hpu::kv_resident_minimum()) + page - 1) / page,
                                             pages);
                    ported_floor.push_back(Row{page, max_cells, resident, pinned, commit});
                    transcribed_floor.push_back(Row{page, max_cells, resident, floor_pages * page, commit});
                    check(pinned == floor_pages * page, "the resident floor does not follow layer.cpp:647-648");
                    check(pinned <= commit, "the resident floor exceeds what the allocator commits");
                    check(pinned <= pages * page, "the resident floor exceeds the arena's own pages");
                    if (resident > 0)
                        check(pinned >= std::min<uint64_t>(resident, commit),
                              "the pin dropped below the cells --kv-resident asked for");
                    else
                        check(pinned == commit, "with no floor set the arena is not fully pinned");
                    // kvg_trim (generate.cpp:6897-6898) hands cells back, never below the pin
                    // uint64_t(0), not 0ull: on Linux uint64_t is `unsigned long` and a braced list mixing it
                    // with `unsigned long long` fails deduction ("unable to deduce std::initializer_list<auto>"),
                    // which only the real target compiler reports.
                    for (uint64_t r : {uint64_t(0), uint64_t(1024), uint64_t(20480), max_cells}) {
                        for (uint64_t hold : {uint64_t(0), uint64_t(4096), max_cells}) {
                            const uint64_t kept = strata::hpu::kv_trim_target(r, hold, s, o);
                            check(kept >= (resident > 0 ? pinned : 0), "a trim went below the resident pin");
                            check(kept <= commit, "a trim kept more cells than the arena commits");
                            check(kept >= std::min<uint64_t>(std::max<uint64_t>(r, hold), commit),
                                  "a trim went below the request it was answering");
                        }
                    }
                }
            }
        }
        check(ported_floor == transcribed_floor, "the residency pin diverged from the transcribed kv_plan expression");
        check(strata::hpu::kv_resident_minimum() == 20480, "the residency minimum moved off layer.cpp:666");
    }

    // the process-level knob keeps upstream's setter/getter shape (layer.cpp:664-666, 575-579)
    {
        KvResidencyOptions o;
        o.grow = true;
        o.resident_cells = 40960;
        o.cap_bytes = 1ull << 30;
        strata::hpu::set_kv_residency(o);
        check(strata::hpu::kv_residency().cap_bytes == o.cap_bytes && strata::hpu::kv_resident_cells() == 40960,
              "the residency setter did not take");
        check(strata::hpu::kv_request_cells(100, 32, 64) == 196, "the request-cell ask moved off generate.cpp:13191");
        strata::hpu::set_kv_residency(KvResidencyOptions{});
        check(!strata::hpu::kv_growth_enabled(strata::hpu::kv_residency()), "growth is not off by default");
    }

    if (failures) {
        std::cout << failures << " residency checks failed\n";
        return 1;
    }
    std::cout << "PASS kv residency: " << cases
              << " (used,free,request) growth decisions equal the transcribed upstream expression as whole sets, "
                 "--no-kv-grow refuses instead of evicting pinned state, the MiB arena cap admits its boundary and "
                 "refuses the byte above, and the resident floor is pinned but never exceeded by the allocator; "
                 "no device\n";
    return 0;
}
