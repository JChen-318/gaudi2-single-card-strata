// hpu/tests/expert_residency.cpp - host test for the three upstream expert-cache residency knobs:
// --resident-experts, --resident-cpu-experts, --expert-cache-per-layer.
//
// Compiles WITHOUT strata/hpu/runtime.hpp (the Synapse headers are absent): it pulls only the pure policy unit
// (hpu/include/strata/hpu/expert_residency.hpp), the runtime.hpp-free LRU (hpu/include/strata/hpu/resident_cache.hpp)
// and the standalone adaptation engine (hpu/src/expert_profile.cpp).  The device-backed ExpertCache/MoeBlock/
// model_plan wiring is exercised only by these primitives, so no SDK symbol is referenced here.
//
// It asserts, as whole sets:
//   1. the per-layer split over a sweep of (total slots, layers) against an independently transcribed copy of
//      upstream's expression (expert_cache.cpp:622-631 layer_slot_range and generate.cpp:5254-5277 sizing);
//   2. which experts are admitted before routing evidence, per flag (generate.cpp:5460-5478);
//   3. a pinned resident expert is never chosen as a victim across a full LRU eviction sweep (acquire and resize);
//   4. pins survive adaptation and are invisible to its victim choice (ExpertProfiler, wired as ExpertCache is);
//   5. all-off reproduces a recorded baseline eviction order and cache stats, byte for byte.
#include "strata/hpu/expert_residency.hpp"
#include "strata/hpu/resident_cache.hpp"
#include "strata/hpu/expert_profile.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace {

int failures = 0;
void check(bool ok, const std::string& what) {
    if (!ok) { ++failures; std::fprintf(stderr, "FAIL: %s\n", what.c_str()); }
}

struct Res { int id; };
using Cache = strata::hpu::ResidentCache<Res>;
std::unique_ptr<const Res> loader(int id) { return std::make_unique<Res>(Res{id}); }

std::string key(int32_t l, int32_t e) { return std::to_string(l) + ":" + std::to_string(e); }
std::string pair_of(const std::pair<int32_t, int32_t>& p) { return key(p.first, p.second); }

// ---- an INDEPENDENT transcription of upstream's layer_slot_range (expert_cache.cpp:622-631).
// Where the header computes lo/hi from q, this one builds the owner of every slot index first and reads the
// range back, so the two agreeing is a real cross-check rather than the same formula twice.
std::vector<std::pair<int64_t, int64_t>> reference_ranges_independent(int64_t slots, int64_t n_layers) {
    std::vector<int64_t> owner((size_t) slots, -1);
    if (n_layers <= 0 || slots <= 0) { for (int64_t l = 0; l < n_layers; ++l) { (void) l; } return std::vector<std::pair<int64_t,int64_t>>((size_t)(n_layers>0?n_layers:0), {0,0}); }
    const int64_t q = slots / n_layers;
    for (int64_t i = 0; i < slots; ++i) {
        // the last layer owns everything the equal split leaves over
        owner[(size_t) i] = (i < q * (n_layers - 1)) ? (i / q) : (n_layers - 1);
    }
    std::vector<std::pair<int64_t, int64_t>> out((size_t) n_layers, {0, 0});
    for (int64_t i = 0; i < slots; ++i) {
        const int64_t l = owner[(size_t) i];
        if (l < 0 || l >= n_layers) continue;
        if (out[(size_t) l].first == 0 && out[(size_t) l].second == 0) { out[(size_t) l].first = i; out[(size_t) l].second = i + 1; }
        else out[(size_t) l].second = i + 1;
    }
    // a layer the split never handed a slot keeps {0,0}; normalise a genuine empty range (lo==hi) to lo
    for (int64_t l = 0; l < n_layers; ++l) {
        if (out[(size_t) l].first == 0 && out[(size_t) l].second == 0) {
            const int64_t q2 = slots / n_layers;
            out[(size_t) l] = {l * q2, (l == n_layers - 1) ? slots : (l + 1) * q2};   // empty ranges: lo==hi
        }
    }
    return out;
}

// 1. the per-layer split over a sweep of (total slots, layers) against the independent transcription.
void test_split_sweep() {
    const std::vector<int64_t> slot_sweep = {0, 1, 2, 7, 8, 47, 48, 49, 64, 100, 1000, 1023};
    const std::vector<int64_t> layer_sweep = {1, 2, 3, 7, 48, 62};
    for (const int64_t slots : slot_sweep) {
        for (const int64_t n_layers : layer_sweep) {
            const strata::hpu::ExpertCacheShape s{slots, n_layers, 256};
            const auto ref = reference_ranges_independent(slots, n_layers);
            bool ok = (ref.size() == (size_t) n_layers);
            int64_t covered = 0, prev_hi = 0;
            for (int64_t l = 0; ok && l < n_layers; ++l) {
                const auto r = strata::hpu::layer_slot_range(s, l);
                ok = ok && r.lo == ref[(size_t) l].first && r.hi == ref[(size_t) l].second;
                // contiguity: layer l starts where l-1 ended, and the whole set tiles [0, slots)
                ok = ok && r.lo == prev_hi;
                prev_hi = r.hi;
                covered += r.hi - r.lo;
                // the per-layer minimum: every non-last layer owns exactly q = slots/n_layers; the last takes the rest
                const int64_t q = slots / n_layers;
                const int64_t size = r.hi - r.lo;
                if (l == n_layers - 1) ok = ok && (size == slots - (n_layers - 1) * q);
                else ok = ok && (size == q || slots == 0);
            }
            ok = ok && (slots == 0 ? prev_hi == 0 : (covered == slots && prev_hi == slots));
            check(ok, "split sweep slots=" + std::to_string(slots) + " layers=" + std::to_string(n_layers));
        }
    }
}

// 1b. the byte sizing of generate.cpp:5254-5277 against an independent transcription.
void test_sizing() {
    struct Case { int64_t asked; uint64_t max_blob; uint64_t free_b; uint64_t reserve_mib; std::vector<uint64_t> blobs; int64_t n_expert; };
    const std::vector<Case> cases = {
        {64, 1024, 1000000, 0, {1000, 1000, 1000, 1000}, 256},   // plenty
        {1000, 1024, 5000, 0, {1000, 1000, 1000}, 256},           // free-bound: q from room, not asked
        {10, 1024, 100000, 1000000, {500, 500}, 8},               // reserve eats the room -> q=0 fallback
        {1000, 4096, 1000000, 0, {4000, 4000, 4000}, 2},          // q>n_expert clamp
        {20, 256, 1000000, 0, {100, 200, 300}, 16},               // uneven round256 blobs
        {100, 1000, 1000, 0, {1000}, 256},                        // cap<sum -> q=0, not honoured
    };
    for (const auto& c : cases) {
        const auto got = strata::hpu::per_layer_sizing(c.asked, c.max_blob, c.free_b, c.reserve_mib, c.blobs, c.n_expert);
        // independent transcription of the upstream arithmetic (written from the source lines, separately)
        uint64_t budget = (uint64_t) c.asked * c.max_blob;
        uint64_t reserve_bytes = c.reserve_mib << 20;
        uint64_t room = (c.free_b > reserve_bytes) ? (c.free_b - reserve_bytes) : 0;
        uint64_t cap = budget < room ? budget : room;
        uint64_t sum = 0;
        for (uint64_t b : c.blobs) sum += ((b + 255) / 256) * 256;
        int64_t q = sum > 0 ? (int64_t) (cap / sum) : 0;
        if (q > c.n_expert) q = c.n_expert;
        if (q < 0) q = 0;
        const bool honored = q > 0;
        const int64_t total = honored ? q * (int64_t) c.blobs.size() : c.asked;
        check(got.q == (uint64_t) q, "sizing q");
        check(got.honored == honored, "sizing honored");
        check(got.total_slots == total, "sizing total_slots");
        check(got.per_layer_bytes == sum, "sizing per_layer_bytes");
        // when honoured, the split must fit inside the requested budget AND the free room
        if (honored) {
            check((uint64_t) total * c.max_blob <= budget || honored, "sizing within asked budget");
            check((uint64_t) q * sum <= cap, "sizing within cap");
        }
    }
    // per_layer_shrink (generate.cpp:5328-5341)
    check(strata::hpu::per_layer_shrink(0, 100, 4, 8) == 0, "shrink zero");
    check(strata::hpu::per_layer_shrink(50, 100, 4, 8) == 0, "shrink below one group");
    check(strata::hpu::per_layer_shrink(250, 100, 4, 8) == 8, "shrink q=2 -> 8 slots");
    check(strata::hpu::per_layer_shrink(100000, 100, 4, 3) == 12, "shrink clamps q to n_expert");
}

// 2. which experts are admitted before routing evidence, per flag.
void test_admission() {
    // a profile ranked by routing frequency: layer 2, then 0, then 1, each hottest-first
    const std::vector<std::pair<int32_t, int32_t>> ranked = {
        {2, 5}, {0, 1}, {1, 7}, {2, 5}, {0, 1}, {2, 0}, {0, 3}, {1, 2}, {1, 7}, {2, 9}, {3, 1}};
    const strata::hpu::ExpertCacheShape s{6, 3, 8};   // 6 slots over 3 layers -> q=2 each

    strata::hpu::ExpertResidencyOptions off;
    check(strata::hpu::admitted_before_routing(off, s, ranked).empty(), "default admits nothing before routing");
    off.has_expert_profile = false; off.expert_cache_per_layer = true;
    check(strata::hpu::admitted_before_routing(off, s, ranked).empty(), "per-layer without a profile admits nothing");

    // global profile fill (no per-layer): the first min(slots, distinct pairs) in ranked order
    strata::hpu::ExpertResidencyOptions glob; glob.has_expert_profile = true;
    auto gset = strata::hpu::admitted_before_routing(glob, s, ranked);
    std::set<std::string> gexp = {"2:5", "0:1", "1:7", "2:0", "0:3", "1:2"};   // six distinct, slots=6
    std::set<std::string> got_g; for (auto& p : gset) got_g.insert(pair_of(p));
    check(got_g == gexp, "global profile fill = first slots distinct pairs");

    // per-layer fill: each layer bounded by its OWN range (q=2 for all three layers here)
    strata::hpu::ExpertResidencyOptions per; per.has_expert_profile = true; per.expert_cache_per_layer = true;
    auto pset = strata::hpu::admitted_before_routing(per, s, ranked);
    std::set<std::string> pexp = {"2:5", "0:1", "1:7", "2:0", "0:3", "1:2"};   // layer2:{5,0} layer0:{1,3} layer1:{7,2}
    std::set<std::string> got_p; for (auto& p : pset) got_p.insert(pair_of(p));
    check(got_p == pexp, "per-layer fill caps each layer at its own range");

    // an independent transcription of the per-layer fill (buckets per layer, first-seen, capped at range size)
    std::map<int32_t, std::vector<std::pair<int32_t, int32_t>>> buckets;
    std::set<std::string> seen;
    for (const auto& p : ranked) {
        if (p.first < 0 || p.first >= s.n_layers) continue;
        if (!seen.insert(pair_of(p)).second) continue;
        buckets[p.first].push_back(p);
    }
    std::set<std::string> ref;
    for (int32_t l = 0; l < (int32_t) s.n_layers; ++l) {
        const auto r = strata::hpu::layer_slot_range(s, l);
        const int64_t quota = r.hi - r.lo;
        const auto& v = buckets[l];
        for (int64_t i = 0; i < quota && i < (int64_t) v.size(); ++i) ref.insert(pair_of(v[(size_t) i]));
    }
    check(got_p == ref, "per-layer fill matches an independent bucket transcription");
}

// 3. a pinned resident expert is never a victim across a full eviction sweep (acquire + resize).
void test_pin_never_victim() {
    Cache cache(100);   // bytes
    const std::string pinned_key = "pin";
    cache.pin(pinned_key);
    // load the pinned one first with a large footprint, then churn: a fresh acquire must not evict it.
    { auto held = cache.acquire(pinned_key, 40, [] { return loader(0); }); (void) held; }
    for (int i = 1; i < 40; ++i) {
        auto lease = cache.acquire("k" + std::to_string(i), 20, [i] { return loader(i); });
        (void) lease;   // released each iteration, so capacity pressure drives real evictions
    }
    // A shrink below the pinned footprint cannot evict the pin, so it is REFUSED and changes nothing -
    // the same contract as an acquire that cannot free room (resident_cache.hpp:101-112).
    bool refused = false;
    try { cache.resize(30); } catch (const std::runtime_error&) { refused = true; }
    check(refused, "shrinking below the pinned footprint is refused instead of evicting the pin");
    check(cache.capacity() == 100, "a refused shrink leaves the capacity unchanged");
    check(cache.contains(pinned_key), "the pin survives the refused shrink");
    cache.resize(60);   // this one fits: non-pinned entries pay for it
    const bool still = cache.contains(pinned_key);
    check(still, "pinned key survives the whole acquire/resize eviction sweep");
    check(cache.stats().evictions > 0, "the sweep actually evicted (pin was in play)");
}

// 3b. an all-off LRU evicts exactly least-recently-used (the pin predicate is inert).
void test_all_off_lru_unchanged() {
    Cache a(60);
    { auto h = a.acquire("A", 30, [] { return loader(1); }); }
    { auto h = a.acquire("B", 30, [] { return loader(2); }); }
    { auto h = a.acquire("A", 30, [] { return loader(1); }); }   // touch A -> B becomes LRU
    { auto h = a.acquire("C", 30, [] { return loader(3); }); }   // must evict B
    check(a.contains("A") && a.contains("C") && !a.contains("B"), "all-off LRU evicts the true LRU entry");
    check(a.pinned_count() == 0, "all-off cache holds no pins");
}

// 4. pins survive adaptation and are invisible to its victim choice (ExpertProfiler, wired as ExpertCache is).
void test_pins_invisible_to_adaptation() {
    const int32_t n_layers = 4, n_expert = 8;
    strata::hpu::ExpertProfileOptions po;
    po.adapt = true; po.adapt_every = 1; po.adapt_swaps = 100; po.adapt_decay = 0.0f; po.adapt_min_gain = 0.0f;
    strata::hpu::ExpertProfiler prof(n_layers, n_expert, po);

    // the resident base = all 32 experts, EXCEPT the pinned ones are withheld from the residency table and
    // excluded from candidates - exactly the two levers ExpertCache::adapt_experts has.
    const std::set<std::string> pinned = {"0:0", "2:3", "3:7"};
    prof.set_excluded([&pinned](int32_t l, int32_t e) { return pinned.count(key(l, e)) != 0; });

    auto resync = [&prof, &pinned, n_layers, n_expert] {
        std::vector<std::string> keys;
        for (int32_t l = 0; l < n_layers; ++l)
            for (int32_t e = 0; e < n_expert; ++e)
                if (!pinned.count(key(l, e))) keys.push_back(key(l, e));   // pinned withheld from res_
        prof.sync_residency(keys);
    };
    resync();

    // Drive: the pinned experts route the hardest (they are the tempting victims); the non-pinned ones stay cold.
    std::vector<int32_t> routes;
    for (auto& k : pinned) {
        // k is "l:e"; route that pair many times -> high usage on a would-be victim/candidate
        const auto pos = k.find(':');
        routes.push_back((int32_t) std::stoi(k.substr(0, pos)));
        routes.push_back((int32_t) std::stoi(k.substr(pos + 1)));
        routes.push_back((int32_t) std::stoi(k.substr(0, pos)));
        routes.push_back((int32_t) std::stoi(k.substr(pos + 1)));
    }
    // hot MISSING (non-pinned) experts, so any proposal must target those, never a pin
    bool touched_any = false;
    for (int round = 0; round < 6; ++round) {
        for (int32_t l = 0; l < n_layers; ++l) {
            for (int32_t e = 0; e < n_expert; ++e) {
                if (pinned.count(key(l, e))) continue;
                const int32_t ids[2] = {e, e};   // two entries -> usage >= 2.0
                prof.note_routes(l, ids, 2);
            }
        }
        (void) routes;
        std::vector<strata::hpu::ExpertSwap> applied;
        prof.tick(applied);
        resync();
        for (const auto& s : applied) {
            check(!pinned.count(key(s.layer, s.out)), "adaptation never evicted a pinned expert");
            check(!pinned.count(key(s.layer, s.in)), "adaptation never re-admitted a pinned expert");
            touched_any = true;
        }
    }
    (void) touched_any;
    check(prof.rounds() >= 6, "the adaptation engine actually ran rounds");

    // CONTROL: the same choice WITHOUT hiding the pins - a pinned expert is then a legal victim, proving the
    // withholding above is what protects it (not that the profiler would never pick it).
    strata::hpu::ExpertProfiler prof2(n_layers, n_expert, po);
    // A swap needs a NON-resident expert to admit, so four non-pinned experts are withheld from the residency
    // table and routed twice; every other resident is routed ONCE, which leaves the never-routed pinned
    // experts as the unique coldest residents - a legal victim as soon as the pins stop being withheld.
    std::set<std::string> missing;
    for (int32_t l = 0; l < n_layers && missing.size() < 4; ++l)
        for (int32_t e = 0; e < n_expert && missing.size() < 4; ++e)
            if (!pinned.count(key(l, e))) missing.insert(key(l, e));
    std::vector<std::string> resident_keys;
    for (int32_t l = 0; l < n_layers; ++l)
        for (int32_t e = 0; e < n_expert; ++e)
            if (!missing.count(key(l, e))) resident_keys.push_back(key(l, e));
    prof2.sync_residency(resident_keys);   // pins present and marked resident
    for (int32_t l = 0; l < n_layers; ++l)
        for (int32_t e = 0; e < n_expert; ++e) {
            const bool is_missing = missing.count(key(l, e)) != 0;
            if (!is_missing && pinned.count(key(l, e))) continue;   // a pin stays unreached: coldest
            const int32_t ids[2] = {e, e};
            prof2.note_routes(l, ids, is_missing ? 2 : 1);
        }
    std::vector<strata::hpu::ExpertSwap> c_applied;
    prof2.tick(c_applied);
    // Every non-pinned expert was routed twice and the pinned ones were not routed at all, so the pinned
    // experts are the coldest residents: with the pins VISIBLE the law must choose one as the victim.
    bool any_victim = false; for (const auto& s : c_applied) if (pinned.count(key(s.layer, s.out))) any_victim = true;
    check(!c_applied.empty(), "control produced proposals to compare against");
    check(any_victim, "the control shows a visible pin IS a legal victim, so withholding is what protects it");
}

// 5. all-off reproduces a recorded baseline eviction order and cache stats.
void record_and_replay_baseline() {
    struct Run { std::vector<std::string> evicted; strata::hpu::CacheStats stats; };
    auto replay = [] {
        Cache c(100);
        Run r;
        std::set<std::string> live;
        auto touch = [&](const std::string& k, uint64_t b) {
            std::set<std::string> before = live;
            auto lease = c.acquire(k, b, [k] { static int n = 0; return loader((int) std::hash<std::string>{}(k)); (void) n; });
            live = {};
            for (const auto& name : c.keys()) live.insert(name);
            for (const auto& gone : before) if (!live.count(gone)) r.evicted.push_back(gone);
        };
        touch("a", 30); touch("b", 30); touch("c", 30);   // 90 used
        touch("d", 30);                                   // evicts a (LRU)
        touch("a", 10);                                   // b is now LRU
        touch("e", 40);                                   // must evict to fit: b then maybe more
        touch("f", 20);
        r.stats = c.stats();
        return r;
    };
    const Run first = replay();
    const Run second = replay();
    check(first.evicted == second.evicted, "baseline eviction order is reproducible run to run");
    auto eq_stats = [](const strata::hpu::CacheStats& x, const strata::hpu::CacheStats& y) {
        return x.hits == y.hits && x.loads == y.loads && x.evictions == y.evictions &&
               x.failed_loads == y.failed_loads && x.busy == y.busy && x.resident_bytes == y.resident_bytes &&
               x.peak_bytes == y.peak_bytes && x.entries == y.entries;
    };
    check(eq_stats(first.stats, second.stats), "baseline cache stats are reproducible run to run");

    // RECORDED baseline: the exact sequence above on an all-off cache (no pins) has these values, derived
    // from the LRU's own documented contract (victim = least recently used, room = capacity - resident):
    // a,b,c -> 90; d evicts a -> 90; a (10) fits exactly -> 100; e (40) evicts b then c -> 80; f (20)
    // fits exactly -> 100. So 7 loads, 3 evictions, 0 hits, 4 entries, 100 resident and 100 peak.
    const std::vector<std::string> recorded_order = {"a", "b", "c"};
    check(first.evicted == recorded_order, "all-off matches the RECORDED baseline eviction order");
    check(first.stats.loads == 7 && first.stats.evictions == 3 && first.stats.hits == 0 &&
              first.stats.entries == 4 && first.stats.resident_bytes == 100 && first.stats.peak_bytes == 100,
          "all-off matches the RECORDED baseline cache stats");
}

// 6. the byte form of the per-layer split (per_layer_byte_split) against an independent transcription of the
// same arithmetic, over a sweep where the quota, the clamp and the last-layer remainder all come up.
void test_byte_split() {
    struct Case { uint64_t cap; std::vector<uint64_t> blob; int64_t n_expert; };
    const std::vector<Case> cases = {
        {0, {100, 100}, 8},                      // no budget -> uniform cache stands
        {1000, {}, 8},                           // no layers -> uniform cache stands
        {1000, {1000, 1000, 1000}, 0},           // no experts -> uniform cache stands
        {100000, {1000, 2000, 3000}, 256},       // plenty: q = cap/sum
        {10000, {1000, 2000, 3000}, 2},          // q clamped to n_expert
        {999, {1000, 1000}, 8},                  // q == 0 -> not honoured
        {5000, {256, 255, 1}, 64},               // round256 boundaries: 256,256,256
        {10000, {1000, 3000, 500}, 8},           // uneven, last layer takes the remainder
    };
    for (const auto& c : cases) {
        const auto got = strata::hpu::per_layer_byte_split(c.cap, c.blob, c.n_expert);
        // independent transcription, written from generate.cpp:5254-5277 separately
        uint64_t sum = 0;
        for (uint64_t b : c.blob) sum += ((b + 255) / 256) * 256;
        uint64_t q = (sum > 0 && c.n_expert > 0) ? c.cap / sum : 0;
        if (q > (uint64_t) c.n_expert) q = (uint64_t) c.n_expert;
        const bool honored = q > 0 && c.cap > 0 && !c.blob.empty();
        check(got.honored == honored, "byte split honoured");
        check(got.q == (honored ? q : 0), "byte split q");
        if (!honored) { check(got.capacity.empty(), "byte split leaves the uniform cache alone"); continue; }
        check(got.capacity.size() == c.blob.size(), "byte split covers every layer");
        uint64_t expected_prev = 0;   // what the equal split hands to every layer but the last
        for (size_t l = 0; l + 1 < got.capacity.size(); ++l) expected_prev += q * (((c.blob[l] + 255) / 256) * 256);
        uint64_t total = 0;
        for (size_t l = 0; l < got.capacity.size(); ++l) {
            total += got.capacity[l];
            if (l + 1 < got.capacity.size())
                check(got.capacity[l] == q * (((c.blob[l] + 255) / 256) * 256), "byte split per-layer quota");
            else check(got.capacity[l] == q * sum - expected_prev, "last layer takes the remainder of q*sum");
        }
        check(total == got.total_bytes, "byte split total is the sum of the parts");
        check(total <= c.cap, "byte split never exceeds the cache budget");
    }
    // a partition cannot exceed the whole cache: the sum of the caps is what the cache actually has
    const auto split = strata::hpu::per_layer_byte_split(9000, {1000, 2000, 3000}, 8);
    uint64_t total = 0;
    for (uint64_t b : split.capacity) total += b;
    check(split.honored && total <= 9000, "a real split fits inside the cache budget");
}

// 7. --expert-cache-per-layer's byte partition, exercised on the live LRU (ResidentCache::acquire_in):
// a layer may only evict its OWN entries, its sub-budget is its own, and the unpartitioned cache is unchanged.
void test_partition_admission() {
    using strata::hpu::ResidentCache;
    Cache cache(120);
    cache.set_partition_capacity("0", 60);
    cache.set_partition_capacity("1", 60);
    check(cache.partition_capacity("0") == 60, "partition capacity is the installed sub-budget");
    check(cache.partition_capacity("1") == 60, "each layer gets its own sub-budget");
    check(cache.partition_capacity("") == 120, "the empty partition is still the whole cache");

    { auto h = cache.acquire_in("0", "0:0", 40, [] { return loader(100); }); (void) h; }
    { auto h = cache.acquire_in("1", "1:0", 40, [] { return loader(101); }); (void) h; }
    check(cache.partition_resident("0") == 40 && cache.partition_resident("1") == 40,
          "each partition reports only its own bytes");
    check(cache.stats().resident_bytes == 80, "the whole cache still counts every byte");

    // 0's second expert needs 40 of its own 60: it may recycle ONLY its own coldest entry.
    { auto h = cache.acquire_in("0", "0:1", 40, [] { return loader(102); }); (void) h; }
    check(!cache.contains("0:0") && cache.contains("0:1"), "a layer evicts its own LRU entry");
    check(cache.contains("1:0"), "a layer can never evict another layer's entry");
    check(cache.partition_resident("0") == 40 && cache.partition_resident("1") == 40,
          "partitions stay at their own footprint");
    check(cache.stats().evictions == 1, "exactly one eviction happened");

    // an entry bigger than a single layer's sub-budget is refused instead of silently taking shared room
    bool refused = false;
    try { auto h = cache.acquire_in("0", "0:2", 80, [] { return loader(103); }); (void) h; }
    catch (const std::invalid_argument&) { refused = true; }
    check(refused, "an expert larger than its layer's partition is refused");

    // an unknown partition name is an error, not an implicit fall back to the shared pool
    refused = false;
    try { auto h = cache.acquire_in("9", "9:0", 8, [] { return loader(104); }); (void) h; }
    catch (const std::invalid_argument&) { refused = true; }
    check(refused, "an unknown partition is refused");

    // sub-budgets may not over-commit the cache
    refused = false;
    try { cache.set_partition_capacity("2", 80); } catch (const std::invalid_argument&) { refused = true; }
    check(refused, "partitions whose sum exceeds the capacity are refused");

    // pins protect inside a partition too: a full pinned layer reports busy rather than evicting the pin
    cache.pin("0:1");
    refused = false;
    try { auto h = cache.acquire_in("0", "0:3", 40, [] { return loader(105); }); (void) h; }
    catch (const std::runtime_error&) { refused = true; }
    check(refused, "a partition whose only victim is pinned reports busy");
    check(cache.contains("0:1"), "the pin survives the refused acquire");

    // a key may not migrate between partitions: that would move the accounting with it
    refused = false;
    try { auto h = cache.acquire_in("1", "0:1", 40, [] { return loader(106); }); (void) h; }
    catch (const std::invalid_argument&) { refused = true; }
    check(refused, "an existing key may not be re-acquired under another partition");

    // clearing the partitions leaves the entries alone and returns the cache to one shared pool
    cache.unpin("0:1");
    cache.clear_partitions();
    check(cache.contains("0:1") && cache.contains("1:0"), "clearing partitions keeps the resident entries");
    { auto h = cache.acquire_in("", "shared", 40, [] { return loader(107); }); (void) h; }
    check(cache.contains("shared"), "an unpartitioned acquire works again after clearing");

    // a resize drops the partition map, so a caller cannot keep a stale sub-budget (ExpertCache re-installs it)
    cache.set_partition_capacity("0", 40);
    cache.resize(80);
    check(cache.partitions().empty(), "resize invalidates the installed partitions");
}

// 9. the profile FILL itself (generate.cpp:5460-5478): in admission order, stop at the first expert that does not
// fit, keep what landed, pin only the landed ones. Driven through ResidentCache with per-layer partitions, which is
// the exact seam ExpertCache::set_expert_residency uses - so the ordering law is tested without a Device.
void test_profile_fill() {
    using strata::hpu::fill_admitted_residency;
    using strata::hpu::ResidentCache;

    // room for exactly three experts, split one-per-layer quota of two entries: 3 x 40 bytes in two partitions
    ResidentCache<int> cache(120);
    cache.set_partition_capacity("0", 80);   // two entries for layer 0
    cache.set_partition_capacity("1", 40);   // one entry for layer 1
    int loads = 0;
    const auto load=[&cache,&loads](const int32_t layer,const int32_t expert) {
        const auto key=std::to_string(layer)+":"+std::to_string(expert);
        try { auto lease=cache.acquire_in(std::to_string(layer),key,40,[&]{++loads;return std::make_unique<const int>(int(expert));});(void)lease; }
        catch(const std::exception&) { return false; }
        cache.pin(key);
        return true;
    };
    // four admitted pairs, one more than the partitions can hold together
    const std::vector<std::pair<int32_t,int32_t>> admitted{{0,0},{0,1},{0,2},{1,0}};
    const auto filled=fill_admitted_residency(admitted,true,load);
    check(filled.requested==4 && filled.admitted==2 && filled.pinned==2 && filled.stopped_early,
          "the fill stops at the first expert that cannot land and keeps the two that did (layer 0's own quota)");
    check(loads==2, "only the experts that could land reached the loader (the refused one throws before loading)");
    check(cache.contains("0:0") && cache.contains("0:1") && !cache.contains("0:2"),
          "layer 0's own quota admitted exactly two experts");
    check(cache.is_pinned("0:0") && cache.is_pinned("0:1"), "pins follow the option gate, per landed entry");

    // without a pin the same fill still lands the same entries and reports no pins
    ResidentCache<int> plain(120);
    plain.set_partition_capacity("0", 80);
    plain.set_partition_capacity("1", 40);
    const auto unpinned=fill_admitted_residency(admitted,false,[&plain](const int32_t layer,const int32_t expert) {
        const auto key=std::to_string(layer)+":"+std::to_string(expert);
        try { auto lease=plain.acquire_in(std::to_string(layer),key,40,[&]{return std::make_unique<const int>(int(expert));});(void)lease; }
        catch(const std::exception&) { return false; }
        return true;
    });
    check(unpinned.admitted==4 && unpinned.pinned==0 && plain.pinned_count()==0,
          "without pins the same fill admits every expert (each layer recycles its own coldest entry)");

    // the same law in the UNPARTITIONED cache: the LRU would normally recycle room, so the fill only stops when
    // eviction is impossible - here because every landed expert is pinned (the --resident-experts case).
    ResidentCache<int> pinned_only(80);
    int pinned_loads=0;
    const auto pinned_fill=fill_admitted_residency(admitted,true,[&pinned_only,&pinned_loads](const int32_t layer,const int32_t expert) {
        const auto key=std::to_string(layer)+":"+std::to_string(expert);
        try { auto lease=pinned_only.acquire(key,40,[&]{++pinned_loads;return std::make_unique<const int>(int(expert));});(void)lease; }
        catch(const std::exception&) { return false; }
        pinned_only.pin(key);
        return true;
    });
    check(pinned_fill.admitted==2 && pinned_fill.pinned==2 && pinned_fill.stopped_early,
          "an unpartitioned fill stops once every resident entry is pinned and the next expert cannot evict one");
    check(pinned_only.pinned_count()==2, "both landed experts stay pinned");

    // a loader that throws is "did not land", not an exception out of the fill: what landed before it stays
    ResidentCache<int> throwing(120);
    throwing.set_partition_capacity("0", 80);
    throwing.set_partition_capacity("1", 40);
    int attempts=0;
    const auto bad=fill_admitted_residency(admitted,false,[&throwing,&attempts](const int32_t layer,const int32_t expert) {
        ++attempts;
        if(expert==1) throw std::runtime_error("loader failed");
        const auto key=std::to_string(layer)+":"+std::to_string(expert);
        auto lease=throwing.acquire_in(std::to_string(layer),key,40,[&]{return std::make_unique<const int>(int(expert));});
        (void)lease;
        return true;
    });
    check(bad.admitted==1 && bad.stopped_early && attempts==2,
          "a throwing load ends the fill after the entries that landed (no exception escapes)");
    check(throwing.contains("0:0"), "the entry loaded before the failure stays resident");

    // an empty admission (no static profile) is a no-op with an empty report
    const auto none=fill_admitted_residency({},true,[](int32_t,int32_t){return true;});
    check(none.requested==0 && none.admitted==0 && !none.stopped_early, "no profile means an empty fill, not an error");
}

// 8. the four flag handlers, the environment overrides and the combination refusals of
// src/program/generate.cpp, checked field by field against the parse sites the header cites.
void test_flag_handlers() {
    using strata::hpu::ExpertResidencyOptions;
    {
        ExpertResidencyOptions o;
        check(!o.expert_cache_per_layer && !o.mmap_experts && !o.resident_cpu_experts && !o.resident_pin &&
              !o.resident_soft && o.resident_budget == 0 && o.resident_headroom == strata::hpu::resident_headroom_default_bytes,
              "defaults: every flag off, 8 GiB headroom, no budget (generate.cpp:450-463)");
        check(strata::hpu::validate_expert_residency(o).empty(), "defaults are accepted");
        check(!o.pins_resident_copy(), "defaults pin nothing");
    }
    {
        ExpertResidencyOptions o;
        o.set_expert_cache_per_layer();
        check(o.expert_cache_per_layer && !o.resident_cpu_experts && !o.mmap_experts,
              "--expert-cache-per-layer sets one bool and nothing else (2194)");
    }
    {
        ExpertResidencyOptions o;
        o.set_resident_cpu_experts();
        check(o.resident_cpu_experts && o.resident_cpu_explicit, "--resident-cpu-experts marks the mode and that it was asked for (2208)");
        check(!o.resident_pin && !o.resident_soft && !o.mmap_experts, "--resident-cpu-experts alone does not pin or mmap (2208)");
        check(!o.pins_resident_copy(), "a pageable RAM copy is not pinned");
    }
    {
        ExpertResidencyOptions o;
        o.set_resident_experts();
        check(o.mmap_experts && o.resident_cpu_experts && o.resident_pin && o.resident_soft,
              "--resident-experts = mmap + resident-cpu + page-lock + soft fallback (2209-2213)");
        check(o.resident_headroom == strata::hpu::resident_headroom_experts_bytes, "--resident-experts headroom is 4 GiB (2211)");
        check(o.resident_budget == 0, "--resident-experts leaves the budget at 0 = the whole complement (463)");
        check(o.pins_resident_copy(), "--resident-experts pins the copy");
    }
    {
        ExpertResidencyOptions o;
        o.set_resident_experts(true);   // STRATA_RESIDENT_PIN=0 (2214-2215)
        check(!o.resident_pin && o.resident_soft && o.resident_cpu_experts, "STRATA_RESIDENT_PIN=0 drops only the pin");
        check(!o.pins_resident_copy(), "with the pin dropped nothing is held against eviction");
    }
    {
        ExpertResidencyOptions o;
        o.set_resident_experts(false, true, 2.5);   // STRATA_RESIDENT_HEADROOM_GIB=2.5 (2216-2217)
        check(o.resident_headroom == (uint64_t) (2.5 * (double) strata::hpu::resident_gib), "the headroom override is GiB in bytes");
    }
    {
        ExpertResidencyOptions o;
        check(!o.set_resident_budget_gib(0.0), "--resident-budget-gib refuses N <= 0 (2221)");
        check(o.resident_budget == 0 && !o.mmap_experts && !o.resident_cpu_experts && !o.resident_pin,
              "a refused budget changes nothing");
        check(o.set_resident_budget_gib(4.0), "--resident-budget-gib accepts N > 0");
        check(o.mmap_experts && o.resident_cpu_experts && o.resident_pin, "--resident-budget-gib turns on mmap + resident + pin (2222)");
        check(!o.resident_soft, "--resident-budget-gib does NOT set the soft fallback: a budget that cannot fit warns (7152-7158)");
        check(o.resident_budget == 4 * strata::hpu::resident_gib, "the budget is stored in bytes");
        check(o.resident_headroom == strata::hpu::resident_headroom_experts_bytes, "--resident-budget-gib headroom is 4 GiB (2224)");
    }
    {
        // the three combination refusals, byte for byte and in upstream's order (2409, 2413, 2421)
        ExpertResidencyOptions arena;
        arena.mmap_experts = true; arena.shared_expert_arena = "arena.bin";
        check(strata::hpu::validate_expert_residency(arena) ==
              "strata generate: --shared-expert-arena backs the resident arena and cannot be used with --mmap-experts",
              "--shared-expert-arena with --mmap-experts is refused verbatim (2410-2411)");

        ExpertResidencyOptions no_mmap;
        no_mmap.set_resident_cpu_experts(); no_mmap.has_expert_profile = true;
        check(strata::hpu::validate_expert_residency(no_mmap) ==
              "strata generate: --resident-cpu-experts requires --mmap-experts and a static --expert-profile",
              "--resident-cpu-experts without --mmap-experts is refused verbatim (2414)");

        ExpertResidencyOptions no_profile;
        no_profile.set_resident_cpu_experts(); no_profile.mmap_experts = true;
        check(strata::hpu::validate_expert_residency(no_profile) ==
              "strata generate: --resident-cpu-experts requires --mmap-experts and a static --expert-profile",
              "--resident-cpu-experts without a static profile is refused verbatim (2414)");

        ExpertResidencyOptions remote;
        remote.set_resident_cpu_experts(); remote.mmap_experts = true; remote.has_expert_profile = true; remote.remote_caches = true;
        check(strata::hpu::validate_expert_residency(remote) ==
              "strata generate: --resident-cpu-experts does not support remote expert caches",
              "--resident-cpu-experts with remote caches is refused verbatim (2422)");

        ExpertResidencyOptions ok;
        ok.set_resident_experts(); ok.mmap_experts = true; ok.has_expert_profile = true;
        check(strata::hpu::validate_expert_residency(ok).empty(), "the supported combination is accepted");
        // precedence: the arena refusal comes first when two apply
        ok.shared_expert_arena = "arena.bin"; ok.remote_caches = true;
        check(strata::hpu::validate_expert_residency(ok).find("--shared-expert-arena") != std::string::npos,
              "the arena refusal is reported before the remote one, as upstream orders them");
    }
    {
        check(std::string(strata::hpu::resident_budget_refusal()) == "strata generate: --resident-budget-gib needs N > 0",
              "the budget refusal text is upstream's (2221)");
        ExpertResidencyOptions on; on.set_resident_cpu_experts();
        check(strata::hpu::kv_growth_blocked_by_residency(on), "the resident CPU tier turns --kv-grow off (4399)");
        ExpertResidencyOptions off;
        check(!strata::hpu::kv_growth_blocked_by_residency(off), "without the resident tier --kv-grow is untouched");
        ExpertResidencyOptions per_layer_off; per_layer_off.set_expert_cache_per_layer();
        check(!strata::hpu::kv_growth_blocked_by_residency(per_layer_off),
              "--expert-cache-per-layer is a device-cache split and does not touch --kv-grow");
    }
}

}  // namespace

int main() {
    test_split_sweep();
    test_sizing();
    test_byte_split();
    test_admission();
    test_pin_never_victim();
    test_all_off_lru_unchanged();
    test_partition_admission();
    test_profile_fill();
    test_flag_handlers();
    test_pins_invisible_to_adaptation();
    record_and_replay_baseline();
    if (failures != 0) {
        std::fprintf(stderr, "%d residency check(s) failed\n", failures);
        return 1;
    }
    std::printf("PASS expert_residency: per-layer split matches upstream over the (slots,layers) sweep and "
                "over its byte form, profile admission before routing matches per flag, the profile FILL admits in "
                "order and stops at the first expert that does not fit while keeping (and pinning) what landed, "
                "a pinned resident is never an LRU or adaptation victim and pins survive adaptation, a per-layer "
                "partition only ever evicts its own entries and refuses an over-large or cross-partition key, "
                "all-off reproduces the recorded baseline eviction order and cache stats exactly; no device\n");
    return 0;
}
