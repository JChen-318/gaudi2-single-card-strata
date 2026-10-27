#pragma once
// hpu/include/strata/hpu/expert_profile.hpp - the upstream expert-cache profiling and adaptation knobs
// for the Gaudi2 port: --expert-profile, --expert-profile-save, --expert-profile-save-every, --adapt-every,
// --adapt-swaps, --adapt-decay, --adapt-min-gain and --adapt-async.
//
// THE LAWS ARE TRANSCRIBED FROM UPSTREAM, NOT PARAPHRASED.  Each one says which lines it came from:
//   src/core/expert_source.cpp:3717-3719   the per-routed-entry routing count (the only writer of `usage`)
//   src/program/generate.cpp:8833-8988     the blocking adaptive round: choose, apply, accumulate heat, decay
//   src/program/generate.cpp:9095-9130     the asynchronous round's choice, over its snapshots
//   src/program/generate.cpp:9264-9271     the asynchronous round's start: snapshot, heat, decay
//   src/program/generate.cpp:10031/12133/13863  the --adapt-every gate
//   src/program/generate.cpp:8708-8717, 9414-9432, 10364-10367  --expert-profile-save and its minutes throttle
//   src/core/expert_cache.cpp:72-210       the STRP profile's read / rank / write (the file format)
//
// THIS HEADER MUST NOT REACH strata/hpu/runtime.hpp (the Synapse SDK headers).  strata/hpu/expert_cache.hpp
// does reach it (through weight_transfer.hpp), so the profiling logic lives here, standalone, and a host
// test can compile it without the SDK.
//
// DEFAULT BEHAVIOUR IS UNCHANGED.  With no flags set, `active()` is false: every vector below stays empty,
// every note*() returns at its first statement, `tick()` never runs a round and no file is written.  That is
// the same shape as upstream's `if (o.adapt_every > 0 && o.adapt_swaps > 0) drive.d.usage.assign(...)`
// (generate.cpp:13573) with `if (!d.usage.empty())` guarding the count (expert_source.cpp:3717).

#include <chrono>
#include <cstdint>
#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace strata::hpu {

/// `(layer, expert)` is not resident.  Upstream's `strata::core::kNotResident` (core/expert_cache.hpp:38);
/// named separately here because this header cannot include that one (it pulls in CUDA).
inline constexpr int32_t kExpertNotResident = -1;

/// The knobs, one field per upstream option, with upstream's defaults and upstream's parsing:
/// generate.cpp:2092-2095, 2192, 2201-2204 into the fields at 501-506, 596-606, 628.
/// Nothing here turns itself on: an empty `profile`/`save` and the disabled tier are the default.
struct ExpertProfileOptions {
    std::string profile;               ///< --expert-profile P: pre-load the residency tier from P (2201)
    std::string save;                  ///< --expert-profile-save P: write what the tier learned (2202)
    double save_every_minutes = 10.0;  ///< --expert-profile-save-every MIN; 0 = on shutdown only (2203-2204, 506)
    int adapt_every = 4;               ///< a round every N calls of the tick; 0 = static residency (598)
    float adapt_decay = 0.7f;          ///< `usage` is multiplied by this after each adaptation (599)
    float adapt_min_gain = 1.5f;       ///< swap in only when the decayed count beats the victim by more (602)
    int adapt_swaps = 96;              ///< at most this many residency changes per round (628)
    bool adapt_async = false;          ///< --adapt-async 1: the round runs between windows, not on one (606)
    /// **DECLARED DIVERGENCE FROM UPSTREAM, AND WHY.**  Upstream gates the counters on
    /// `o.adapt_every > 0 && o.adapt_swaps > 0` alone (generate.cpp:8667, 13573) because that line sits inside
    /// the serve and decode paths that already have an expert cache and a residency table - a plain run never
    /// reaches it.  These knobs are reachable from code that runs today with no adaptation at all, so
    /// inheriting that gate would change every existing run.  `adapt` says the tier was asked for; the numbers
    /// above stay exactly upstream's defaults, so `--adapt-every 0` and `--adapt-swaps 0` still mean "static
    /// residency" the way they do there.
    bool adapt = false;
    /// The adaptive tier exists when it was asked for and neither gate is zero - upstream's two conditions.
    bool tier_enabled() const { return adapt && adapt_every > 0 && adapt_swaps > 0; }
    /// Anything is counted at all: the tier, or one of the two profile paths asked for it.
    bool monitoring() const { return tier_enabled() || !profile.empty() || !save.empty(); }
};

/// One proposed residency change: expert `in` takes the place of the resident expert `out` in `layer`.
/// Field order and types are upstream's `struct Swap { float gain; int32_t layer, in, out; }` (generate.cpp:8836).
struct ExpertSwap {
    float gain;      ///< `in`'s decayed routing count less `out`'s (generate.cpp:8855)
    int32_t layer;
    int32_t in;      ///< the expert to admit
    int32_t out;     ///< the expert to evict
};

/// The STRP profile's header, as src/core/expert_cache.cpp:81 reads it and :189-198 writes it.
inline constexpr char expert_profile_magic[4] = {'S', 'T', 'R', 'P'};
inline constexpr uint32_t expert_profile_version = 1u;

/// **THE FORMAT, BYTE FOR BYTE.**  `STRP`, then five little-endian uint32
/// `version, n_layers, n_expert, slots, n_ranked`, then `n_ranked` pairs of little-endian uint16
/// `(layer, expert)` ranked by descending routing, then an `n_layers x n_expert` int32 table holding each
/// pair's rank (-1: not ranked).  Transcribed from src/core/expert_cache.cpp:170-210 (write) and :72-139 (read),
/// and refused with the same messages.  Written atomically: `<path>.tmp` then renamed over `path` (:191-207).
bool write_expert_profile(const std::string& path, int64_t n_layers, int64_t n_expert,
                          const std::vector<std::pair<int32_t, int32_t>>& ranked, std::string& err);
bool read_expert_profile(const std::string& path, int64_t n_layers, int64_t n_expert,
                         std::vector<std::pair<int32_t, int32_t>>& ranked, int64_t& slots, std::string& err);

/// The order `--expert-profile-save` writes: resident experts first, then the rest; inside each by `heat`
/// descending, then by the profile the run started from, then by index.  src/core/expert_cache.cpp:141-168.
std::vector<std::pair<int32_t, int32_t>> rank_learned_profile(
    int64_t n_layers, int64_t n_expert, const std::vector<uint8_t>& resident, const std::vector<double>& heat,
    const std::vector<std::pair<int32_t, int32_t>>& prior);

/// The counters, the decay law and the adaptation policy.  Host-side only: it holds no device resource and
/// makes no SDK call, so it is both testable and safe to feed from the token path.
class ExpertProfiler {
public:
    using Clock = std::chrono::steady_clock;

    ExpertProfiler(int64_t n_layers, int64_t n_expert, ExpertProfileOptions options = {});

    const ExpertProfileOptions& options() const { return options_; }
    /// Set the knobs.  Changing the law (decay, min-gain, every, swaps, async) keeps what was counted; only a
    /// counter set appearing or disappearing - turning the tier or a profile path on or off - allocates it,
    /// so the tool calls this once at startup and a later call cannot lose the run's counts.
    /// `--expert-profile`'s pre-load goes through preload_profile().
    void set_options(ExpertProfileOptions options);

    int64_t n_layers() const { return n_layers_; }
    int64_t n_expert() const { return n_expert_; }
    /// True when anything is counted: no flags at all, and this is false and every hook below is free.
    bool active() const { return !hits_.empty(); }
    /// True when the adaptive tier's routing counts exist - upstream's `!drive.d.usage.empty()` guard.
    bool counting() const { return !usage_.empty(); }

    // ---- the counters, fed at the lookup point --------------------------------------------

    /// The routing count, upstream's law verbatim (src/core/expert_source.cpp:3717-3719): one float
    /// increment PER ROUTED ENTRY, so an expert an activation routes k times takes k, and only ids in
    /// [0, n_expert) count (the port's shared expert, -1, does not).
    void note_routes(int64_t layer, const int32_t* ids, int64_t n);
    /// The cache's own hit/miss tallies, one per lookup of (layer, expert) at ExpertCache::acquire /
    /// ::resident.  Reporting only: they never feed the adaptation law, which reads `usage` alone.
    void note_hit(int64_t layer, int64_t expert);
    void note_miss(int64_t layer, int64_t expert);

    float usage(int64_t layer, int64_t expert) const;
    uint64_t hits(int64_t layer, int64_t expert) const;
    uint64_t misses(int64_t layer, int64_t expert) const;
    double heat(int64_t layer, int64_t expert) const;
    const std::vector<float>& usage_counts() const { return usage_; }
    const std::vector<double>& heat_counts() const { return heat_; }

    /// The residency table, as upstream's `host_res`: slot >= 0 resident, `kExpertNotResident` not.  A
    /// byte-budget LRU has no slot indices, so the port passes 0 for resident.
    void set_resident(int64_t layer, int64_t expert, int32_t slot);
    bool is_resident(int64_t layer, int64_t expert) const;
    /// Rebuild the whole table from the cache's key set ("layer:expert"); experts it does not name go missing.
    void sync_residency(const std::vector<std::string>& keys);
    void clear_residency();

    /// Experts another tier already holds and that may not be swapped in here.  Upstream asks the peer cache,
    /// the remote helpers and the arena (`peer.has` / `remote_opt->owns` / `helper_holds`, generate.cpp:8845);
    /// on one Gaudi2 there is none of those, so this is normally empty.
    void set_excluded(std::function<bool(int32_t layer, int32_t expert)> excludes) {
        excludes_ = std::move(excludes);
    }

    // ---- the adaptation policy -------------------------------------------------------------

    /// The choices, `std::vector<Swap> swaps` at generate.cpp:8837-8858, with no cap applied yet: the
    /// blocking tier truncates it (`--adapt-swaps`, :8859) and the asynchronous one walks it, skipping a
    /// victim that lost its slot since the snapshot (:9120-9123).
    std::vector<ExpertSwap> ranked_proposals() const;

    /// One blocking round: choose, apply the residency changes, add the counts to `heat`, decay them
    /// (generate.cpp:8833-8988, in that order - the decay is LAST, after the picks it just fed).
    bool run_round(std::vector<ExpertSwap>& applied);

    /// --adapt-async: the round's START (generate.cpp:9264-9271) - snapshot the counts and the residency,
    /// fold the counts into `heat`, and decay, all BEFORE the choice is made.  The choice itself happens
    /// later, on the snapshot, in settle_async_round().
    bool begin_async_round();
    /// ... and the round's second half: choose over the snapshot (:9095-9130) and publish the residency.
    void settle_async_round(std::vector<ExpertSwap>& applied);
    bool round_in_flight() const { return async_in_flight_; }

    /// The --adapt-every gate, as the serve loop runs it (generate.cpp:10031): one call per verify window /
    /// decode round; a round on every `adapt_every`-th call; a round never starts while one is in flight
    /// (:10031's `!ajob`, :8834's `pending.empty()`).  Returns false only when a round could not complete.
    bool tick(std::vector<ExpertSwap>& applied);
    int64_t windows() const { return windows_; }
    int64_t rounds() const { return rounds_; }
    int64_t swaps() const { return swaps_; }

    /// The decay, verbatim: `for (float& v : drive.d.usage) v *= o.adapt_decay;` (generate.cpp:8987, 9271,
    /// 13703).  Public so the invariant can be checked against an independent transcription.
    void apply_decay();

    // ---- --expert-profile / --expert-profile-save ------------------------------------------

    /// Load `--expert-profile P` and pre-mark its first `slots` pairs resident (:2429/:3657-3698 and the
    /// tier fill): a start from a saved profile begins where the last run ended.
    bool preload_profile(std::string& err);

    /// `--expert-profile-save-every MIN` between saves, 0 = only at shutdown (generate.cpp:2203-2204, 506,
    /// and the check at :10364-10367 before the next request).
    bool save_due(Clock::time_point now) const;
    /// Rank what the tier learned and write it; then restart the throttle window (:9414-9432).  False, with
    /// `err`, when there is nothing counted or the file could not be written.
    bool save_profile(Clock::time_point now, std::string& err);
    bool saving() const { return !heat_.empty(); }

private:
    std::vector<ExpertSwap> proposals_from(const std::vector<float>& usage, const std::vector<int32_t>& res) const;
    void allocate();
    void accumulate_heat();

    int64_t n_layers_ = 0, n_expert_ = 0;
    ExpertProfileOptions options_;
    std::vector<float> usage_;                ///< [n_layers * n_expert] decayed routing counts (expert_source.hpp:470)
    std::vector<uint64_t> hits_, misses_;     ///< the cache's own per-expert lookup tallies
    std::vector<double> heat_;                ///< [n_layers * n_expert] routing since the start (:8716)
    std::vector<int32_t> res_;                ///< [n_layers * n_expert] slot or kExpertNotResident (`host_res`)
    std::vector<std::pair<int32_t, int32_t>> prior_;   ///< --expert-profile's ranking, as loaded (:3700-3703)
    std::vector<float> snap_usage_;           ///< --adapt-async: the round's count snapshot (:9267)
    std::vector<int32_t> snap_res_;           ///< ... and its residency snapshot (:9268)
    bool async_in_flight_ = false;
    int64_t windows_ = 0, rounds_ = 0, swaps_ = 0;
    Clock::time_point saved_at_;
    std::function<bool(int32_t, int32_t)> excludes_;
};

}  // namespace strata::hpu
