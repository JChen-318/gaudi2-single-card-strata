// hpu/tests/expert_profile.cpp - host test for the ported expert-cache profiling and adaptation knobs.
//
// Pins, against transcriptions of upstream written independently of hpu/src/expert_profile.cpp:
//   * the per-routed-entry count (src/core/expert_source.cpp:3717-3719) over a fixed access sequence,
//     plus the cache's own per-expert hit/miss tallies,
//   * the decay law `for (float& v : usage) v *= adapt_decay` (generate.cpp:8987, 9271, 13703) over a whole
//     sweep of (decay, seed, rounds), each expected value obtained as the correctly-rounded single product
//     via an exact double product - a different formulation of the same IEEE step, so a paraphrase of the
//     law (double accumulation, an EMA, an FMA) fails rather than passing quietly,
//   * the STRP profile's bytes (src/core/expert_cache.cpp:72-210): a hand-derived golden string, a round
//     trip, and every refusal upstream makes,
//   * the --adapt-min-gain gate and the break that follows it (generate.cpp:8854) and the --adapt-swaps cap
//     (:8859 / :9121),
//   * what --adapt-async changes about WHEN the update happens (generate.cpp:9264-9271 against 8976-8987),
//   * --expert-profile-save's ranking (:9414-9432) and --expert-profile-save-every's throttle (:10364-10367),
//   * and that a disabled configuration observes exactly the eviction order an uninstrumented cache produces.
// CPU only: nothing here opens a device, allocates HBM or touches the Synapse SDK.
#include "strata/hpu/expert_profile.hpp"
#include "strata/hpu/resident_cache.hpp"

#include <bit>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace strata::hpu;
using Clock = ExpertProfiler::Clock;
using TP = Clock::time_point;

int checks = 0;
void require(bool ok, const char* message) {
    ++checks;
    if (!ok) throw std::runtime_error(message);
}

/// Bit-exact float comparison: the decay law is a rounding law, so `==` on values is not enough (-0.0 and
/// 0.0, and the last mantissa bit, are both observable here).
bool same_bits(float a, float b) { return std::bit_cast<uint32_t>(a) == std::bit_cast<uint32_t>(b); }

/// The count, as a plain loop over ids: upstream's writer is `for (i < n) if (ids[i] >= 0 && ids[i] <
/// n_expert) usage[layer*n_expert + ids[i]] += 1.0f`.
void bump(ExpertProfiler& prof, int64_t layer, int32_t expert, int count) {
    if (count <= 0) return;
    std::vector<int32_t> ids((size_t) count, expert);
    prof.note_routes(layer, ids.data(), (int64_t) count);
}

ExpertProfileOptions tier(float decay, float min_gain, int every, int swaps) {
    ExpertProfileOptions o;
    o.adapt = true;   // the port's opt-in: upstream's numbers alone must not start a tier here
    o.adapt_every = every;
    o.adapt_swaps = swaps;
    o.adapt_decay = decay;
    o.adapt_min_gain = min_gain;
    return o;
}

std::vector<uint8_t> slurp(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}
void spit(const std::string& path, const std::vector<uint8_t>& bytes) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.write(reinterpret_cast<const char*>(bytes.data()), (std::streamsize) bytes.size());
}
/// Plain C, not <filesystem>: this MinGW release cannot launch even a one-call std::filesystem::exists
/// program without -static, and the port's target (Gaudi2, Linux) has no such problem.
bool exists(const std::string& path) {
    if (std::FILE* f = std::fopen(path.c_str(), "rb")) { std::fclose(f); return true; }
    return false;
}

// ---- A. the counters over a fixed access sequence (src/core/expert_source.cpp:3717-3719) ------------------
void counters_over_a_fixed_sequence() {
    ExpertProfiler prof(2, 4, tier(1.0f, 1.5f, 4, 96));
    require(prof.counting(), "the tier's routing counts must exist while the tier is enabled");
    const int32_t first[] = {1, 1, 2, -1, 7, 0};       // the shared expert (-1) and an id past n_expert do not count
    prof.note_routes(0, first, 6);
    const int32_t second[] = {3, 3, 3, -1};
    prof.note_routes(1, second, 4);
    const int32_t third[] = {2, 2};
    prof.note_routes(0, third, 2);
    const float expected[2][4] = {{1.0f, 2.0f, 3.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 3.0f}};
    for (int64_t l = 0; l < 2; ++l)
        for (int32_t e = 0; e < 4; ++e)
            require(same_bits(prof.usage(l, e), expected[l][e]), "per-expert routing count over the fixed sequence");
    // out-of-range lookups are ignored rather than counted somewhere adjacent
    prof.note_routes(-1, first, 6);
    prof.note_routes(2, first, 6);
    prof.note_routes(0, nullptr, 6);
    require(same_bits(prof.usage(0, 1), 2.0f) && prof.usage(1, 3) == 3.0f, "an out-of-range layer/ids row counts nothing");

    // the hit/miss tallies, fed at the lookup point; they report, they never feed the adaptation law
    prof.note_hit(0, 1); prof.note_hit(0, 1); prof.note_miss(0, 2);
    prof.note_hit(1, 3); prof.note_miss(0, 0); prof.note_miss(0, 0);
    prof.note_hit(0, 9); prof.note_miss(7, 0);
    require(prof.hits(0, 1) == 2 && prof.hits(1, 3) == 1 && prof.hits(0, 2) == 0, "per-expert hit tally");
    require(prof.misses(0, 2) == 1 && prof.misses(0, 0) == 2 && prof.misses(0, 1) == 0, "per-expert miss tally");
    require(same_bits(prof.usage(0, 1), 2.0f) && same_bits(prof.usage(0, 2), 3.0f),
            "a hit/miss tally must not move a routing count");
    // an expert routed k times by one activation takes k, which is what the >= 2.0f candidate threshold reads
    ExpertProfiler single(1, 2, tier(1.0f, 1.5f, 4, 96));
    const int32_t same[] = {0, 0};
    single.note_routes(0, same, 2);
    require(same_bits(single.usage(0, 0), 2.0f) && same_bits(single.usage(0, 1), 0.0f),
            "repeats within one window each count, per entry not per distinct expert");
}

// ---- B. the decay law over a whole sweep -----------------------------------------------------------------
/// The expected step, transcribed a second way: `v *= decay` is one correctly-rounded IEEE-754 single
/// product, and the exact product of two floats is representable in a double, so double(x)*double(y)
/// rounded to float IS the single-precision product.  Same law, different mechanism, so a paraphrase of
/// the implementation shows up as a difference instead of cancelling out.
float decay_step_reference(float v, float decay) { return (float) ((double) v * (double) decay); }

void decay_sweep() {
    const float decays[] = {0.0f, 0.125f, 0.25f, 0.5f, 0.7f, 0.75f, 0.9f, 0.99f, 1.0f, 1.5f, 3.0f, 1e-6f, -0.5f};
    const int seeds[] = {0, 1, 2, 3, 5, 7, 10, 17, 64, 99};
    int paraphrase_differences = 0;
    for (float decay : decays) {
        for (int seed : seeds) {
            ExpertProfiler prof(1, 1, tier(decay, 1.5f, 1, 96));
            bump(prof, 0, 0, seed);
            float expected = (float) seed;
            // the paraphrase to rule out: carry the product in double and round only when the table is
            // stored, instead of re-rounding once per round as `for (float& v : usage) v *= decay` does
            double carried = (double) seed;
            for (int round = 0; round <= 6; ++round) {
                require(same_bits(prof.usage(0, 0), expected), "decay sweep: the count before this round");
                std::vector<ExpertSwap> applied;
                require(prof.run_round(applied) && applied.empty(), "a decay-only round proposes nothing when nothing is resident");
                expected = decay_step_reference(expected, decay);
                carried *= (double) decay;
                require(same_bits(prof.usage(0, 0), expected), "decay sweep: the count after this round");
                if (!same_bits((float) carried, expected)) ++paraphrase_differences;
            }
        }
    }
    require(paraphrase_differences > 0,
            "the decay sweep must be able to tell single-rounding from double accumulation");

    // the same law with routing added between the decays, which is how the engine actually composes it
    for (float decay : {0.5f, 0.7f, 0.25f, 1.5f, 0.9f}) {
        for (int per_round : {1, 2, 3}) {
            ExpertProfiler prof(1, 1, tier(decay, 1.5f, 1, 96));
            float expected = 0.0f;
            for (int round = 0; round < 5; ++round) {
                bump(prof, 0, 0, per_round);
                for (int i = 0; i < per_round; ++i) expected += 1.0f;   // the count is a float += 1.0f per entry
                require(same_bits(prof.usage(0, 0), expected), "decay sweep with routing: the count added per entry");
                std::vector<ExpertSwap> applied;
                prof.run_round(applied);
                expected = decay_step_reference(expected, decay);
                require(same_bits(prof.usage(0, 0), expected), "decay sweep with routing: the count decayed once per round");
            }
        }
    }
}

// ---- C. the STRP profile's bytes (src/core/expert_cache.cpp:72-210) ---------------------------------------
void profile_format(const std::string& dir) {
    const std::string path = dir + "/golden.bin";
    std::vector<std::pair<int32_t, int32_t>> ranked = {{1, 2}, {0, 0}, {1, 0}};
    std::string err;
    require(write_expert_profile(path, 2, 3, ranked, err), "the profile writer succeeds");
    // the hand-derived golden bytes: STRP, then uint32 {version, n_layers, n_expert, slots, n_ranked},
    // then the ranked pairs as uint16 (layer, expert), then the 2x3 int32 rank table
    static const uint8_t golden[] = {
        'S', 'T', 'R', 'P',
        0x01, 0x00, 0x00, 0x00,   // version 1
        0x02, 0x00, 0x00, 0x00,   // n_layers 2
        0x03, 0x00, 0x00, 0x00,   // n_expert 3
        0x03, 0x00, 0x00, 0x00,   // slots   = n_ranked
        0x03, 0x00, 0x00, 0x00,   // n_ranked
        0x01, 0x00, 0x02, 0x00,   // pair (1,2)
        0x00, 0x00, 0x00, 0x00,   // pair (0,0)
        0x01, 0x00, 0x00, 0x00,   // pair (1,0)
        0x01, 0x00, 0x00, 0x00,   // table[0][0] rank 1
        0xff, 0xff, 0xff, 0xff,   // table[0][1] not ranked
        0xff, 0xff, 0xff, 0xff,   // table[0][2] not ranked
        0x02, 0x00, 0x00, 0x00,   // table[1][0] rank 2
        0xff, 0xff, 0xff, 0xff,   // table[1][1] not ranked
        0x00, 0x00, 0x00, 0x00,   // table[1][2] rank 0
    };
    const std::vector<uint8_t> bytes = slurp(path);
    require(bytes.size() == sizeof golden, "the profile's length is 24 + 4*n_ranked + 4*n_layers*n_expert");
    require(std::memcmp(bytes.data(), golden, sizeof golden) == 0, "the profile's bytes match the golden string");
    require(!exists(path + ".tmp"), "the profile is written through a tmp that is renamed away");

    std::vector<std::pair<int32_t, int32_t>> back;
    int64_t slots = -1;
    require(read_expert_profile(path, 2, 3, back, slots, err), "the profile reads back");
    require(back == ranked && slots == 3, "the round trip returns the ranking and the slot count");

    // an empty ranking: the header and the table only
    const std::string empty_path = dir + "/empty.bin";
    require(write_expert_profile(empty_path, 2, 3, {}, err) && slurp(empty_path).size() == 48,
            "an empty ranking writes the header plus the table");
    require(read_expert_profile(empty_path, 2, 3, back, slots, err) && back.empty() && slots == 0,
            "an empty ranking reads back empty");

    // every refusal upstream makes, with the same message
    require(!read_expert_profile(dir + "/absent.bin", 2, 3, back, slots, err) && err.rfind("read_expert_profile: cannot open ", 0) == 0,
            "a missing profile is refused by name");
    auto refuse = [&](const std::string& name, const std::vector<uint8_t>& bytes, const char* needle, const char* what) {
        const std::string p = dir + "/" + name;
        spit(p, bytes);
        std::vector<std::pair<int32_t, int32_t>> r;
        int64_t s = -1;
        std::string e;
        require(!read_expert_profile(p, 2, 3, r, s, e), what);
        require(e.find(needle) != std::string::npos, what);
    };
    const std::vector<uint8_t> short_file = {'S', 'T', 'R', 'P', 0x01, 0x00};
    refuse("short.bin", short_file, "too short to hold a header", "a header shorter than 24 bytes is refused");
    {
        std::vector<uint8_t> bad(golden, golden + sizeof golden);
        std::memcpy(bad.data(), "XTRP", 4);
        refuse("badmagic.bin", bad, "does not start with STRP", "a wrong magic is refused");
        std::vector<uint8_t> v2(golden, golden + sizeof golden);
        v2[4] = 2;
        refuse("v2.bin", v2, "is a version 2 profile but this engine reads version 1", "a later version is refused by its number");
        std::vector<uint8_t> dims(golden, golden + sizeof golden);
        dims[8] = 9; dims[12] = 9;
        refuse("dims.bin", dims, "different artifact", "a profile for another layout is refused");
        std::vector<uint8_t> many(golden, golden + sizeof golden);
        many[16] = 4;                       // slots 4, n_ranked 3 -> fine; flip the other way
        many[16] = 3; many[20] = 4;
        refuse("ranked.bin", many, "more ranked pairs than slots", "n_ranked past the slot count is refused");
        std::vector<uint8_t> trunc(golden, golden + sizeof golden);
        trunc.resize(30);                   // the ranked list cut short
        refuse("trunc.bin", trunc, "the ranked list is truncated", "a truncated ranked list is refused");
        std::vector<uint8_t> oor(golden, golden + sizeof golden);
        oor[24] = 5;                        // pair 0's layer, past n_layers
        refuse("oor.bin", oor, "out of range", "a ranked pair outside the layout is refused");
    }
    require(!write_expert_profile(dir + "/nope.bin", 2, 3, {{5, 0}}, err) &&
                err == "write_expert_profile: a ranked pair is out of range",
            "a ranked pair outside the layout is refused before anything is written");
    require(!write_expert_profile(dir + "/nope.bin", 0, 3, ranked, err) &&
                err == "write_expert_profile: the model's layout does not fit the format",
            "a layout that does not fit the format is refused");
    require(!write_expert_profile(dir + "/nope.bin", 2, 70000, ranked, err),
            "an n_expert past the uint16 field is refused");

    // replaces an existing file in place, and leaves nothing behind
    require(write_expert_profile(path, 2, 3, {{0, 0}}, err), "the profile is written over the old one");
    require(read_expert_profile(path, 2, 3, back, slots, err) && back.size() == 1 && back[0] == std::make_pair(0, 0),
            "the replaced profile is what the next start reads");
    require(!exists(path + ".tmp"), "a re-write leaves no tmp file");
}

// ---- D. the min-gain gate and the swap cap --------------------------------------------------------------
void min_gain_and_swap_limits() {
    // one layer, experts 0 and 1 resident, 2 and 3 missing; usage = [6, 4, 9, 8]
    auto build = [](float min_gain) {
        ExpertProfiler prof(1, 4, tier(1.0f, min_gain, 4, 96));
        bump(prof, 0, 0, 6); bump(prof, 0, 1, 4); bump(prof, 0, 2, 9); bump(prof, 0, 3, 8);
        prof.set_resident(0, 0, 0); prof.set_resident(0, 1, 0);
        return prof;
    };
    {
        const ExpertProfiler prof = build(1.5f);
        const auto p = prof.ranked_proposals();
        require(p.size() == 2, "min-gain 1.5 admits both changes");
        require(same_bits(p[0].gain, 5.0f) && p[0].in == 2 && p[0].out == 1, "the coldest resident expert is the first victim");
        require(same_bits(p[1].gain, 2.0f) && p[1].in == 3 && p[1].out == 0, "the gain-descending order is kept");
    }
    {   // the second pair's gain is 2.0, so min-gain 3.0 admits it and the break stops the layer there
        const ExpertProfiler prof = build(3.0f);
        const auto p = prof.ranked_proposals();
        require(p.size() == 1 && same_bits(p[0].gain, 5.0f) && p[0].in == 2 && p[0].out == 1,
                "min-gain 3.0 refuses the weaker change and stops the layer there");
    }
    {   // the gate is `cand < vict + min_gain`: a gain exactly equal to min_gain is admitted
        const ExpertProfiler prof = build(5.0f);
        const auto p = prof.ranked_proposals();
        require(p.size() == 1 && same_bits(p[0].gain, 5.0f), "a gain equal to --adapt-min-gain is admitted");
    }
    {   // ... and one less than that is not, which also means the layer proposes nothing at all
        const ExpertProfiler prof = build(5.5f);
        require(prof.ranked_proposals().empty(), "a gain below --adapt-min-gain is refused outright");
    }
    {   // turning a knob later does not lose what was already counted
        ExpertProfiler prof = build(1.5f);
        ExpertProfileOptions o = prof.options();
        o.adapt_min_gain = 3.0f;
        prof.set_options(o);
        require(same_bits(prof.usage(0, 2), 9.0f) && prof.is_resident(0, 1),
                "--adapt-min-gain changed later: the counts and the residency survived");
        require(prof.ranked_proposals().size() == 1, "and the new gate is what the next round uses");
    }
    {   // the >= 2.0f candidate threshold (generate.cpp:8845): one count is never a candidate, two are
        ExpertProfiler prof(1, 3, tier(1.0f, 0.0f, 4, 96));
        bump(prof, 0, 2, 1);   // expert 0 resident and never routed, expert 2 missing with one routing
        prof.set_resident(0, 0, 0);
        require(prof.ranked_proposals().empty(), "a missing expert with one routing is not a candidate");
        bump(prof, 0, 2, 1);
        const auto p = prof.ranked_proposals();
        require(p.size() == 1 && p[0].in == 2 && p[0].out == 0 && same_bits(p[0].gain, 2.0f),
                "the same expert becomes a candidate at two routings");
    }
    {   // a layer with nothing resident can evict nothing, and one with no missing expert gains nothing
        ExpertProfiler prof(2, 2, tier(1.0f, 0.0f, 4, 96));
        bump(prof, 0, 0, 5); bump(prof, 1, 0, 5);
        prof.set_resident(1, 0, 0); prof.set_resident(1, 1, 0);
        require(prof.ranked_proposals().empty(), "a layer with no resident expert is skipped and a full layer has no candidate");
    }
    {   // --adapt-swaps: at most N changes per round, and the ones kept are the highest-gain
        ExpertProfiler prof(3, 2, tier(1.0f, 1.5f, 4, 2));
        for (int64_t l = 0; l < 3; ++l) {
            bump(prof, l, 0, (int) (l == 0 ? 10 : l == 1 ? 9 : 8));   // the missing expert of that layer
            prof.set_resident(l, 1, 0);                                // its only resident expert, never routed
        }
        require(prof.ranked_proposals().size() == 3, "all three layers qualify before the cap");
        std::vector<ExpertSwap> applied;
        require(prof.run_round(applied), "the round completes");
        require(applied.size() == 2, "--adapt-swaps caps a round at two residency changes");
        require(same_bits(applied[0].gain, 10.0f) && applied[0].layer == 0 && applied[0].in == 0 && applied[0].out == 1,
                "the cap keeps the highest gain");
        require(same_bits(applied[1].gain, 9.0f) && applied[1].layer == 1, "and the second highest");
        require(prof.swaps() == 2 && prof.rounds() == 1, "the round's swaps are counted once");
        require(prof.is_resident(0, 0) && !prof.is_resident(0, 1) && prof.is_resident(2, 1) && !prof.is_resident(2, 0),
                "a capped round leaves the third layer's residency untouched");
    }
    {   // an expert another tier holds is not swapped in here (upstream's peer/remote/helper terms, :8845)
        ExpertProfiler prof(1, 3, tier(1.0f, 0.0f, 4, 96));
        bump(prof, 0, 2, 9);
        prof.set_resident(0, 0, 0);
        prof.set_excluded([](int32_t, int32_t e) { return e == 2; });
        require(prof.ranked_proposals().empty(), "an expert another tier owns is never a candidate");
        prof.set_excluded({});
        require(prof.ranked_proposals().size() == 1, "and with no other tier it is");
    }
}

// ---- E. --adapt-every's cadence and --adapt-async's WHEN ----------------------------------------------
void cadence_and_async(const std::string& dir) {
    {   // adapt_every = 0 with the tier asked for: static residency, nothing counted, no round ever runs
        ExpertProfileOptions o; o.adapt = true; o.adapt_every = 0; o.adapt_swaps = 96;
        ExpertProfiler prof(1, 4, o);
        bump(prof, 0, 0, 5);
        std::vector<ExpertSwap> applied;
        for (int i = 0; i < 8; ++i) require(prof.tick(applied), "a tick without a tier is not an error");
        require(!prof.counting() && prof.usage_counts().empty() && prof.windows() == 0 && prof.rounds() == 0,
                "--adapt-every 0 leaves the counters and the rounds absent");
        require(prof.usage(0, 0) == 0.0f, "--adapt-every 0 counts no routing");
    }
    {   // the default tier: a round on every 4th call and none in between (generate.cpp:10031 / 13863)
        ExpertProfiler prof(1, 4, tier(0.5f, 1.5f, 4, 96));
        int64_t expected_rounds = 0;
        for (int64_t call = 1; call <= 12; ++call) {
            std::vector<ExpertSwap> applied;
            require(prof.tick(applied) && applied.empty(), "a tick with nothing resident proposes nothing");
            if (call % 4 == 0) ++expected_rounds;
            require(prof.rounds() == expected_rounds, "--adapt-every fires on exactly every Nth call");
            require(prof.windows() == call, "the tick counts every window");
        }
        require(prof.rounds() == 3 && prof.windows() == 12, "twelve windows at --adapt-every 4 is three rounds");
    }
    {   // the blocking tier: choose, apply, then decay - all inside one tick
        ExpertProfiler prof(2, 2, tier(0.5f, 1.5f, 1, 96));
        bump(prof, 0, 1, 9); prof.set_resident(0, 0, 0);   // layer 0: expert 1 missing with 9 routings
        bump(prof, 1, 1, 7); prof.set_resident(1, 0, 0);   // layer 1: ... and 7
        const auto before = prof.ranked_proposals();
        require(before.size() == 2, "both layers qualify before the round");
        std::vector<ExpertSwap> applied;
        require(prof.tick(applied), "the tick runs the round");
        require(applied.size() == 2 && applied[0].layer == 0 && applied[1].layer == 1,
                "the blocking tier publishes its residency changes in the same call");
        require(prof.is_resident(0, 1) && !prof.is_resident(0, 0) && prof.is_resident(1, 1) && !prof.is_resident(1, 0),
                "the blocking tier evicts the victim and admits the candidate as it chooses");
        require(same_bits(prof.usage(0, 1), 4.5f) && same_bits(prof.usage(1, 1), 3.5f),
                "the blocking tier decays after the picks it just fed (generate.cpp:8987)");
    }
    {   // --adapt-async: the decay lands at the round's START (generate.cpp:9271) and the choice is made
        // later, over the snapshot - so the counts the copies were chosen from are already gone from the
        // live table, and a second round cannot start before the first one settles (:10031's `!ajob`).
        ExpertProfileOptions o = tier(0.5f, 1.5f, 2, 96);
        o.adapt_async = true;
        ExpertProfiler prof(1, 2, o);
        bump(prof, 0, 0, 3);                      // the missing expert, 3 routings
        prof.set_resident(0, 1, 0);               // the resident one, never routed
        std::vector<ExpertSwap> applied;
        require(prof.tick(applied) && applied.empty(), "the first window runs no round at --adapt-every 2");
        require(prof.tick(applied) && applied.empty(), "--adapt-async: the tick begins the round and applies nothing yet");
        require(prof.round_in_flight(), "--adapt-async: the round is in flight after its start");
        require(prof.rounds() == 1, "--adapt-async: the round is counted when it starts");
        require(same_bits(prof.usage(0, 0), 1.5f), "--adapt-async decays at the round's start, before the choice");
        require(!prof.is_resident(0, 0) && prof.is_resident(0, 1), "--adapt-async publishes no residency at the start");
        require(prof.ranked_proposals().empty(),
                "the live counts have already fallen under 2.0f, so a choice from them would find nothing");
        // a due round cannot stack on the one in flight (generate.cpp:10031's `!ajob`)
        prof.tick(applied);                                    // window 3: not due
        const int64_t rounds_before = prof.rounds();
        require(prof.tick(applied) && applied.empty(), "window 4 is due but a round is in flight");
        require(prof.rounds() == rounds_before && prof.round_in_flight(),
                "--adapt-async: one round at a time, the due tick is skipped");
        prof.settle_async_round(applied);
        require(!prof.round_in_flight(), "--adapt-async: the round settles");
        require(applied.size() == 1 && applied[0].in == 0 && applied[0].out == 1 && same_bits(applied[0].gain, 3.0f),
                "--adapt-async chooses from the snapshot taken before the decay, not from the decayed table");
        require(prof.is_resident(0, 0) && !prof.is_resident(0, 1), "--adapt-async publishes residency when it settles");
        require(prof.swaps() == 1 && prof.rounds() == 1, "one round, one swap");
    }
    {   // --expert-profile P pre-loads the residency tier from the ranking (generate.cpp:3657-3698)
        const std::string path = dir + "/preload.bin";
        std::string err;
        require(write_expert_profile(path, 2, 3, {{1, 2}, {0, 0}}, err), "the preload profile is written");
        // widen its slot field past the ranked list so the truncation branch is exercised
        std::vector<uint8_t> bytes = slurp(path);
        bytes[16] = 5;   // slots 5 >= n_ranked 2
        spit(path, bytes);
        ExpertProfileOptions o = tier(0.7f, 1.5f, 4, 96);
        o.profile = path;
        ExpertProfiler prof(2, 3, o);
        require(prof.preload_profile(err), "the profile loads");
        require(prof.is_resident(1, 2) && prof.is_resident(0, 0) && !prof.is_resident(0, 1) && !prof.is_resident(1, 0),
                 "--expert-profile marks exactly its ranked pairs resident, hottest first");
        int64_t slots = 0;
        std::vector<std::pair<int32_t, int32_t>> ranked;
        require(read_expert_profile(path, 2, 3, ranked, slots, err) && slots == 5 && ranked.size() == 2,
                "the widened slot field survives the read");
        require(!write_expert_profile(dir + "/nested/absent.bin", 2, 3, ranked, err) &&
                    err.find("cannot create") != std::string::npos,
                "a profile whose directory does not exist is refused");
    }
}

// ---- F. --expert-profile-save's ranking and throttle ----------------------------------------------------
void save_policy(const std::string& dir) {
    // rank_learned_profile (src/core/expert_cache.hpp:57-66): resident first, then heat descending, then the
    // profile the run started from, then the index
    const std::vector<uint8_t> resident = {1, 0, 1, 0, 0, 1};
    const std::vector<double> heat = {5.0, 9.0, 5.0, 0.0, 0.0, 5.0};
    const std::vector<std::pair<int32_t, int32_t>> prior = {{1, 0}, {1, 1}};
    const auto ranked = rank_learned_profile(2, 3, resident, heat, prior);
    const std::vector<std::pair<int32_t, int32_t>> expected = {{0, 0}, {0, 2}, {1, 2}, {0, 1}, {1, 0}, {1, 1}};
    require(ranked == expected,
            "the saved ranking puts the resident tier first (even under a hotter miss), then heat, then the prior, then the index");
    // an expert this run never routed keeps its old place in the prior (the heat-0 tail above), and no heat at
    // all leaves the prior to decide the whole order
    const auto by_prior = rank_learned_profile(2, 3, {}, {}, {{1, 1}, {0, 0}});
    require(by_prior[0] == std::make_pair(1, 1) && by_prior[1] == std::make_pair(0, 0), "the prior breaks a heat tie");
    const std::vector<uint8_t> one_resident = {0, 0, 0, 0, 1, 0};
    const auto resident_first = rank_learned_profile(2, 3, one_resident, {}, {});
    require(resident_first[0] == std::make_pair(1, 1), "the resident set comes first with no counts at all");

    const std::string path = dir + "/learned.bin";
    ExpertProfileOptions o = tier(0.5f, 1.5f, 1, 96);
    o.save = path;
    o.save_every_minutes = 10.0;
    ExpertProfiler prof(2, 3, o);
    require(prof.saving(), "--expert-profile-save turns the heat counters on (generate.cpp:8711-8716)");
    bump(prof, 0, 0, 3); bump(prof, 1, 2, 1);
    prof.set_resident(1, 2, 0);
    std::vector<ExpertSwap> applied;
    prof.run_round(applied);   // round 1: heat takes this round's counts before the decay
    prof.run_round(applied);   // round 2: and again
    require(same_bits(prof.usage(0, 0), 0.75f), "the count decayed twice: 3 * 0.5 * 0.5");
    require(prof.heat(0, 0) == 4.5 && prof.heat(1, 2) == 1.5 && prof.heat(0, 2) == 0.0,
            "the heat is the sum of each round's pre-decay count: 3 + 1.5, and 1 + 0.5");

    // the exact statement of upstream's note at generate.cpp:8973-8974: counted this way, one routing is
    // worth 1/(1-decay) over all its decays.  At decay 0.5 six rounds of one routing give 1+1/2+...+1/32 = 63/32.
    ExpertProfileOptions g = tier(0.5f, 1.5f, 1, 96);
    g.save = dir + "/geo.bin";
    ExpertProfiler geo(1, 1, g);
    bump(geo, 0, 0, 1);
    for (int round = 0; round < 6; ++round) geo.run_round(applied);
    require(geo.heat(0, 0) == 63.0 / 32.0, "one routing contributes 1 + d + d^2 ... = 1/(1-decay) to the heat");
    require(same_bits(geo.usage(0, 0), 0.015625f), "and the live count is only the last decaying remnant");

    // --expert-profile-save-every: 10 minutes between saves (generate.cpp:10364-10367)
    const TP t0 = Clock::now();
    std::string err;
    require(!prof.save_due(t0 + std::chrono::minutes(9)), "a save is not due before --expert-profile-save-every minutes");
    require(prof.save_due(t0 + std::chrono::minutes(10)), "a save is due at exactly --expert-profile-save-every minutes");
    require(prof.save_profile(t0, err), "the profile is saved");
    const std::vector<uint8_t> saved = slurp(path);
    require(saved.size() == 24 + 6u * 4u + 6u * 4u, "the saved profile ranks every (layer, expert) pair");
    std::vector<std::pair<int32_t, int32_t>> back;
    int64_t slots = 0;
    require(read_expert_profile(path, 2, 3, back, slots, err) && back.size() == 6 && slots == 6,
            "what --expert-profile-save writes, --expert-profile reads");
    require(back[0] == std::make_pair(1, 2), "the saved ranking starts where this run's cache ended");
    require(back[1] == std::make_pair(0, 0), "then the hottest expert that is not resident");
    require(!exists(path + ".tmp"), "the save is atomic: no tmp file survives it");
    require(!prof.save_due(t0 + std::chrono::minutes(9)), "the throttle restarts at the save, not at the start");
    require(prof.save_due(t0 + std::chrono::minutes(20)), "and the next save is due ten minutes after that");

    // 0 = on shutdown only
    prof.set_options([&] { auto n = prof.options(); n.save_every_minutes = 0.0; return n; }());
    require(prof.saving() && !prof.save_due(t0 + std::chrono::hours(24 * 365)),
            "--expert-profile-save-every 0 never saves between requests");

    // without --expert-profile-save nothing is counted for the file and nothing is written
    ExpertProfiler quiet(2, 3, tier(0.5f, 1.5f, 1, 96));
    bump(quiet, 0, 0, 3);
    require(!quiet.saving() && quiet.heat_counts().empty(), "--expert-profile-save off: no heat counters at all");
    require(!quiet.save_due(Clock::now()) && !quiet.save_profile(Clock::now(), err),
            "--expert-profile-save off: nothing is ever saved");
    require(!exists(dir + "/learned-off.bin"), "the off path writes no file");
}

// ---- G. a disabled configuration observes the same eviction order --------------------------------------
struct Payload { int id; };

/// One pass over the cache with the same operations the port's lookup point runs (`find`, then `acquire`
/// only on a miss), optionally with the profiler fed at that point.  Records the resident key set after
/// every step, which is what an eviction order is observable through without perturbing it.
std::vector<std::string> drive_cache(ExpertProfiler* prof, uint64_t capacity, std::string& trace) {
    ResidentCache<Payload> cache(capacity);
    const char* keys[] = {"0:0", "0:1", "0:2", "0:0", "0:3", "0:1", "0:4", "0:2", "0:5", "0:0", "0:6", "0:3"};
    std::vector<std::string> observed;
    int id = 0;
    for (const char* key : keys) {
        ++id;
        const int32_t layer = (int32_t) std::strtol(key, nullptr, 10);
        const char* colon = std::strchr(key, ':');
        const int32_t expert = (int32_t) std::strtol(colon + 1, nullptr, 10);
        auto lease = cache.find(key);
        if (lease) {
            if (prof) prof->note_hit(layer, expert);
        } else {
            if (prof) prof->note_miss(layer, expert);
            lease = cache.acquire(key, 4, [id]() -> std::unique_ptr<const Payload> { return std::make_unique<Payload>(id); });
        }
        if (prof && prof->active()) prof->sync_residency(cache.keys());
        lease = {};
        // the resident set after this step, in the cache's own order, plus which keys are there
        for (const std::string& k : cache.keys()) observed.push_back(k);
        observed.push_back(std::string("|"));
        if (!cache.contains(key)) observed.push_back("?");
    }
    trace = std::to_string(cache.stats().hits) + "/" + std::to_string(cache.stats().loads) + "/" +
            std::to_string(cache.stats().evictions) + "/" + std::to_string(cache.stats().resident_bytes) + "/" +
            std::to_string(cache.stats().peak_bytes) + "/" + std::to_string(cache.stats().busy) + "/" +
            std::to_string(cache.stats().entries);
    return observed;
}

void disabled_configuration_is_unchanged() {
    // a plain-constructed options struct is "no flags": upstream's numbers are its defaults, and none of
    // them starts a tier, allocates a counter or opens a file in this port
    const ExpertProfileOptions none;
    require(none.adapt_every == 4 && none.adapt_swaps == 96 && same_bits(none.adapt_decay, 0.7f) &&
                same_bits(none.adapt_min_gain, 1.5f) && none.save_every_minutes == 10.0 && !none.adapt_async &&
                !none.adapt,
            "the defaults are upstream's (generate.cpp:598, 599, 602, 606, 628 and 506)");
    require(!none.tier_enabled() && !none.monitoring(), "no flags means no tier and nothing counted");
    ExpertProfiler off(1, 7, none);
    require(!off.active() && !off.counting() && !off.saving(), "with no flags the profiler holds nothing");
    require(off.usage_counts().empty() && off.heat_counts().empty(), "with no flags no counters are allocated");
    // every hook is a no-op, and nothing it reports exists to report
    bump(off, 0, 0, 5);
    off.note_hit(0, 0); off.note_miss(0, 1);
    require(off.usage(0, 0) == 0.0f && off.hits(0, 0) == 0 && off.misses(0, 1) == 0, "a disabled profiler counts nothing");
    off.set_resident(0, 0, 0);
    require(!off.is_resident(0, 0), "a disabled profiler holds no residency table");
    std::vector<ExpertSwap> applied;
    for (int i = 0; i < 20; ++i) require(off.tick(applied) && applied.empty(), "a disabled tier never runs a round");
    require(off.rounds() == 0 && off.windows() == 0 && off.swaps() == 0, "a disabled tier never advances");

    std::string plain_trace, off_trace, on_trace;
    const std::vector<std::string> plain = drive_cache(nullptr, 12, plain_trace);
    const std::vector<std::string> with_off = drive_cache(&off, 12, off_trace);
    require(plain == with_off, "a disabled configuration reports exactly the same eviction order as before the change");
    require(plain_trace == off_trace, "a disabled configuration reports exactly the same cache statistics as before the change");
    // and the counters, when they are on, are observation only: same order, same stats, same payload
    ExpertProfiler on(1, 7, tier(0.7f, 1.5f, 4, 96));
    const std::vector<std::string> with_on = drive_cache(&on, 12, on_trace);
    require(plain == with_on, "counting hit/miss and residency does not reorder the cache's evictions");
    require(plain_trace == on_trace, "counting hit/miss and residency does not change the cache's statistics");
    require(on.counting() && on.usage(0, 0) == 0.0f, "the residency hooks never touch the routing counts");
    require(on.hits(0, 0) + on.hits(0, 1) + on.hits(0, 2) + on.hits(0, 3) > 0 && on.misses(0, 4) == 1,
            "the per-expert tallies recorded the same lookups the cache did");
    require(!plain.empty() && plain.front() == "0:0" && plain.size() > 12, "the eviction trace was actually recorded");
}

}  // namespace

int main(int argc, char** argv) {
    const std::string dir = argc > 1 ? argv[1] : "work/tmp-expert";
    try {
        // The fixtures below write into `dir`; a ctest working directory does not contain it, so the test
        // creates its own scratch directory instead of relying on the runner to prepare one.
        std::filesystem::create_directories(dir);
        counters_over_a_fixed_sequence();
        decay_sweep();
        profile_format(dir);
        min_gain_and_swap_limits();
        cadence_and_async(dir);
        save_policy(dir);
        disabled_configuration_is_unchanged();
        // the scratch files are this test's only side effect; the directory itself is left for the runner to drop
        for (const char* name : {"golden.bin", "empty.bin", "short.bin", "preload.bin", "learned.bin", "geo.bin",
                                 "absent.bin", "nope.bin", "badmagic.bin", "v2.bin", "dims.bin", "ranked.bin",
                                 "trunc.bin", "oor.bin", "magic.bin"})
            std::remove((dir + "/" + name).c_str());
        std::cout << "PASS expert routing counts per entry, decay law over a 13x10x7 sweep against an independent "
                     "single-rounding transcription, golden STRP bytes with round trip and every upstream refusal, "
                     "min-gain and --adapt-swaps gates with the break and the 2.0f threshold, --adapt-every cadence, "
                     "--adapt-async decaying at the round's start and choosing from its snapshot, "
                     "--expert-profile-save ranking with its minutes throttle and atomic write, "
                     "--expert-profile pre-load, and a disabled configuration reporting the identical eviction order "
                     "and cache statistics ("
                  << checks << " checks); no device\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "expert_profile test failed: " << e.what() << " (" << checks << " checks before the failure)\n";
        return 1;
    }
}
