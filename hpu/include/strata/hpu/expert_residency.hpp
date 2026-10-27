#pragma once
// hpu/include/strata/hpu/expert_residency.hpp - the upstream expert-cache residency controls for the
// Gaudi2 single-card port: --resident-experts, --resident-cpu-experts and --expert-cache-per-layer.
//
// ONE PURE POLICY UNIT, THE SAME SHAPE AS hpu/include/strata/hpu/kv_residency.hpp: it pulls in no Synapse
// header, so a host test compiles it against `resident_cache.hpp` (also runtime.hpp-free) without the SDK.
// strata/hpu/expert_cache.hpp DOES reach runtime.hpp (through weight_transfer.hpp), which is why the split
// arithmetic and the admission set live here rather than in that header.
//
// UPSTREAM AUTHORITIES (pinned source, commit 296b59d, src/program/generate.cpp unless noted):
//   450-458   the option fields: mmap_experts (450), resident_cpu_experts (452), resident_pin (456),
//             resident_headroom (457, default 8 GiB), resident_soft (458), resident_cpu_explicit (459)
//   462-463   resident_budget = 0 (0 = the whole complement)
//   483       expert_cache_per_layer = false; the comment at 478-482 is the R4.2g measurement
//   2194      --expert-cache-per-layer            sets expert_cache_per_layer
//   2208      --resident-cpu-experts              sets resident_cpu_experts AND resident_cpu_explicit (no pin)
//   2209-2218 --resident-experts                  sets mmap_experts, resident_cpu_experts, resident_pin,
//                                                 resident_soft; headroom 4 GiB; STRATA_RESIDENT_PIN=0 drops the pin;
//                                                 STRATA_RESIDENT_HEADROOM_GIB=N overrides the headroom
//   2219-2229 --resident-budget-gib N             N>0 else refused (2221); sets mmap+resident_cpu+pin, headroom 4 GiB
//   2409-2411 refusal: --shared-expert-arena with --mmap-experts
//   2413-2416 refusal: --resident-cpu-experts without --mmap-experts or without a static --expert-profile
//   2421-2424 refusal: --resident-cpu-experts with remote expert caches
//   4399      --kv-grow is off when --resident-cpu-experts is on (the two RAM tiers are exclusive)
//   5254-5277 the per-layer slot sizing (generate.cpp): budget = asked*max_blob, cap = min(budget, free_room),
//             sum = sum over layers of round256(blob_bytes(l)), q = cap/sum, q clamped to n_expert, and the
//             per-layer list is q copies of each layer's own blob (q*n_layers slots); q==0 -> the split is not
//             honoured and the uniform sizing stands
//   5328-5341 shrink_to's per-layer arm: q = keep_bytes/per_layer_bytes, clamped to n_expert, q<=0 -> 0 slots
//   5433,5444 set_per_layer_admission(o.expert_cache_per_layer) and the "each layer owns N slots" report
//   5460-5478 the profile fill: per-layer, each layer takes its own hottest experts until its range is full;
//             global, the first min(profile.size(), slots) pairs.  This is the admission that happens BEFORE
//             any routing evidence (the profile, not the run, decides it).
//   src/core/expert_cache.cpp:622-631 layer_slot_range: q = slots/n_layers, layer l owns [l*q,(l+1)*q),
//             the LAST layer takes whatever the division leaves, so the ranges cover 0..slots exactly.
//   src/core/expert_cache.cpp:638-656 admit: the per-layer path shares the global path's "no eviction" rule;
//             a layer admits only inside its own range and is refused (kNotResident) once that range is full.
//
// WHAT "RESIDENT" MEANS, disambiguated from the three lines above (usage 934-939):
//   * --resident-cpu-experts  holds the experts the GPU cache does NOT keep in ordinary RAM, and requires a
//     static --expert-profile so WHICH experts are held is decided by the profile (routing counted over a whole
//     trace), not by the run's live routing.  The copy is pageable unless pinned.
//   * --resident-experts      = --mmap-experts --resident-cpu-experts PLUS page-lock (resident_pin), 4 GiB of
//     headroom, and a plain-mmap fallback (resident_soft) when the copy does not fit.  "resident" = admitted from
//     the profile before routing AND, when resident_pin, held against eviction (page-locked).
//   * --expert-cache-per-layer changes only WHICH slot a layer may take: one budget split across the layers by a
//     per-layer quota q = slots/n_layers, the last layer taking the remainder.  It does NOT evict; the per-layer
//     minimum is q slots per layer, and the total is honoured only as far as the bytes fit (q = floor(cap/sum)).
//
// DEFAULTS LEAVE TODAY'S BEHAVIOUR ALONE.  With no flags every field is false/zero, validate() is empty,
// per_layer_sizing reports the uniform cache, admitted_before_routing is empty and pinned_residents is empty -
// so ExpertCache keeps its plain lazy acquire and the LRU keeps its plain LRU order.
#include <algorithm>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace strata::hpu {

/// generate.cpp:457 (resident_headroom default 8 GiB) and 2211/2224 (--resident-experts / --resident-budget-gib
/// override it to 4 GiB).  A GiB in bytes, verbatim from the parse sites (2211, 2217, 2222, 2224).
inline constexpr uint64_t resident_headroom_default_bytes = 8ull << 30;
inline constexpr uint64_t resident_headroom_experts_bytes = 4ull << 30;
inline constexpr uint64_t resident_gib = 1073741824ull;

/// The three knobs plus the fields they set, one per upstream option.  Parsing mirrors the flag handler
/// exactly (2194, 2208, 2209-2218, 2219-2229); nothing here turns itself on.
struct ExpertResidencyOptions {
    bool expert_cache_per_layer = false;   ///< --expert-cache-per-layer (2194, field 483)
    bool mmap_experts = false;             ///< --mmap-experts (2206, field 450); implied by both resident modes
    bool resident_cpu_experts = false;     ///< --resident-cpu-experts (2208, field 452)
    bool resident_cpu_explicit = false;    ///< #384: --resident-cpu-experts given by itself, not only implied (459)
    bool resident_pin = false;             ///< page-locked copy (456); set by --resident-experts and --resident-budget-gib
    bool resident_soft = false;            ///< plain-mmap fallback when the copy does not fit (458)
    uint64_t resident_headroom = resident_headroom_default_bytes;  ///< RAM left free (457)
    uint64_t resident_budget = 0;          ///< GiB-sized resident budget in bytes; 0 = the whole complement (463)
    std::string shared_expert_arena;       ///< --shared-expert-arena backing file (451); only read by validate()
    bool remote_caches = false;            ///< any --expert-cache-remote slot > 0 (2417); only read by validate()
    bool has_expert_profile = false;       ///< --expert-profile non-empty (501); decides which experts are admitted

    /// --expert-cache-per-layer (2194): one bool, nothing else.
    void set_expert_cache_per_layer() { expert_cache_per_layer = true; }
    /// --resident-cpu-experts (2208): marks the mode AND that it was asked for by itself.  It does NOT pin;
    /// only --resident-experts / --resident-budget-gib page-lock the copy.  It implies neither mmap nor the
    /// profile here - validate() refuses the combination that lacks them, exactly as upstream does.
    void set_resident_cpu_experts() {
        resident_cpu_experts = true;
        resident_cpu_explicit = true;
    }
    /// --resident-experts (2209-2218): mmap + resident-cpu + page-lock + soft fallback, headroom 4 GiB.
    /// `env_pin_zero` is STRATA_RESIDENT_PIN=="0" (2214-2215); `headroom_gib` is a parsed
    /// STRATA_RESIDENT_HEADROOM_GIB >= 0 (2216-2217), nullopt-style via `has_headroom_override`.
    void set_resident_experts(bool env_pin_zero = false, bool has_headroom_override = false,
                              double env_headroom_gib = 0.0) {
        mmap_experts = resident_cpu_experts = resident_pin = resident_soft = true;
        resident_headroom = resident_headroom_experts_bytes;
        if (env_pin_zero) resident_pin = false;
        if (has_headroom_override && env_headroom_gib >= 0.0)
            resident_headroom = (uint64_t) (env_headroom_gib * (double) resident_gib);
    }
    /// --resident-budget-gib N (2219-2229): N>0 is enforced by the caller (the parse site returns 2 otherwise);
    /// here it stores the byte budget and turns on mmap + resident-cpu + pin, headroom 4 GiB.  It does NOT set
    /// resident_soft, so a budget that cannot fit is a warning, not a stop (7152-7158) - the flag that differs
    /// from --resident-experts.
    bool set_resident_budget_gib(double gib, bool env_pin_zero = false, bool has_headroom_override = false,
                                 double env_headroom_gib = 0.0) {
        if (!(gib > 0.0)) return false;   // generate.cpp:2221 refuses N <= 0
        resident_budget = (uint64_t) (gib * (double) resident_gib);
        mmap_experts = resident_cpu_experts = resident_pin = true;
        resident_headroom = resident_headroom_experts_bytes;
        if (env_pin_zero) resident_pin = false;
        if (has_headroom_override && env_headroom_gib >= 0.0)
            resident_headroom = (uint64_t) (env_headroom_gib * (double) resident_gib);
        return true;
    }

    /// The combination that page-locks a RAM copy: only --resident-experts and --resident-budget-gib set the pin,
    /// and both go through resident_cpu_experts.  Upstream's --kv-grow is off for this mode (4399).
    bool pins_resident_copy() const { return resident_pin && resident_cpu_experts; }
};

/// The cache's slot geometry, the three integers the arithmetic reads.
struct ExpertCacheShape {
    int64_t total_slots = 0;   ///< xcache.slots()
    int64_t n_layers = 0;      ///< g.n_layers
    int64_t n_expert = 0;      ///< g.n_expert
};

struct SlotRange { int64_t lo = 0, hi = 0; };

/// src/core/expert_cache.cpp:622-631 verbatim.  Out-of-range layer / empty cache -> {0,0}, the same as upstream's
/// early return after zeroing both.
inline SlotRange layer_slot_range(const ExpertCacheShape& s, int64_t layer) {
    SlotRange r;
    if (s.n_layers <= 0 || s.total_slots <= 0 || layer < 0 || layer >= s.n_layers) return r;
    const int64_t q = s.total_slots / s.n_layers;
    r.lo = layer * q;
    r.hi = (layer == s.n_layers - 1) ? s.total_slots : (layer + 1) * q;
    return r;
}

/// generate.cpp:5254-5277, the per-layer slot sizing.  Reports what the split WOULD produce; `honored` is the
/// upstream `if (q > 0)` gate - when it is false the split cannot give even one slot per layer and the uniform
/// cache (the plain `asked` slots) stands, exactly as generate.cpp falls through the whole block.
struct PerLayerSizing {
    uint64_t q = 0;                 ///< slots each non-last layer owns
    int64_t total_slots = 0;        ///< q*n_layers when honored, else the caller keeps the uniform asked value
    uint64_t per_layer_bytes = 0;   ///< the sum of the round256 blob sizes (upstream per_layer_bytes)
    bool honored = false;           ///< q > 0
};

inline uint64_t round256(uint64_t bytes) { return (bytes + 255) / 256 * 256; }

/// `blob_bytes_per_layer` must have `n_layers` entries (lay.blob_bytes(l)); `free_bytes` is the raw device_free_bytes
/// reading and `reserve_mib` the --vram-reserve-mib.  max_blob is lay.max_blob.  Transcribed, not paraphrased.
inline PerLayerSizing per_layer_sizing(int64_t asked, uint64_t max_blob, uint64_t free_bytes, uint64_t reserve_mib,
                                       const std::vector<uint64_t>& blob_bytes_per_layer, int64_t n_expert) {
    PerLayerSizing out;
    if (asked <= 0 || blob_bytes_per_layer.empty()) return out;
    const uint64_t budget = (uint64_t) asked * max_blob;
    const uint64_t reserve_bytes = reserve_mib << 20;
    const uint64_t free_room = free_bytes > reserve_bytes ? free_bytes - reserve_bytes : 0;
    const uint64_t cap = std::min<uint64_t>(budget, free_room);
    uint64_t sum = 0;
    for (const uint64_t b : blob_bytes_per_layer) sum += round256(b);
    const int64_t n_layers = (int64_t) blob_bytes_per_layer.size();
    const int64_t q = sum > 0 ? (int64_t) (cap / sum) : 0;
    out.q = (uint64_t) (q > n_expert ? n_expert : (q < 0 ? 0 : q));
    out.per_layer_bytes = sum;
    out.honored = out.q > 0;
    if (out.honored) out.total_slots = (int64_t) (out.q * (uint64_t) n_layers);
    else out.total_slots = asked;   // the uniform cache the fall-through keeps
    return out;
}

/// generate.cpp:5328-5341, shrink_to's per-layer arm: the quota a byte figure buys, clamped to n_expert, and the
/// slot count it rebuilds (0 when the figure cannot fit even one per-layer group).
inline int64_t per_layer_shrink(int64_t keep_bytes, uint64_t per_layer_bytes, int64_t n_layers, int64_t n_expert) {
    if (per_layer_bytes == 0 || n_layers <= 0) return 0;
    int64_t q = keep_bytes / (int64_t) per_layer_bytes;
    if (q > n_expert) q = n_expert;
    if (q <= 0) return 0;
    return q * n_layers;
}

/// The byte form of the same split, which is what a byte-budget cache can actually install.  Upstream sizes its
/// slot array (:5254-5277): budget = asked*max_blob, cap = min(budget, free room), sum = sum of round256(blob_l),
/// q = cap/sum clamped to n_expert, and layer l then owns q slots OF ITS OWN BLOB.  This backend's cache is a
/// byte budget with no slot array, so `cache_capacity` takes the place of `cap` (the caller has already bounded it
/// by --expert-cache-mib and the HBM plan) and every layer gets q * round256(blob_l) bytes; the last layer takes
/// whatever q*sum leaves over, exactly as it takes the remainder of the slot count upstream.  q == 0 keeps the
/// uniform cache, which is upstream's `if (q > 0)` fall-through.
struct PerLayerByteSplit {
    uint64_t q = 0;                 ///< experts each layer keeps (upstream's per-layer quota)
    std::vector<uint64_t> capacity; ///< per layer: q * round256(blob_l); the last layer takes the remainder
    uint64_t total_bytes = 0;       ///< sum of `capacity`, never above cache_capacity
    bool honored = false;           ///< q > 0
};

inline PerLayerByteSplit per_layer_byte_split(uint64_t cache_capacity,
                                              const std::vector<uint64_t>& blob_bytes_per_layer, int64_t n_expert) {
    PerLayerByteSplit out;
    if (cache_capacity == 0 || blob_bytes_per_layer.empty() || n_expert <= 0) return out;
    uint64_t sum = 0;
    for (const uint64_t b : blob_bytes_per_layer) sum += round256(b);
    if (sum == 0) return out;
    uint64_t q = cache_capacity / sum;
    if (q > (uint64_t) n_expert) q = (uint64_t) n_expert;   // upstream clamps the quota to the expert count
    if (q == 0) return out;
    out.q = q;
    out.honored = true;
    out.total_bytes = q * sum;
    out.capacity.assign(blob_bytes_per_layer.size(), 0);
    uint64_t handed_out = 0;
    for (size_t l = 0; l + 1 < blob_bytes_per_layer.size(); ++l) {
        out.capacity[l] = q * round256(blob_bytes_per_layer[l]);
        handed_out += out.capacity[l];
    }
    out.capacity.back() = out.total_bytes - handed_out;   // the last layer takes the remainder (expert_cache.cpp:626)
    return out;
}

/// DECLARED DIVERGENCE, and the reason this unit exists rather than a claim of equivalence: upstream's per-layer
/// admission REFUSES a (layer, expert) whose layer range is already full (core/expert_cache.cpp:638-656 returns
/// kNotResident) and then executes that expert outside the cache.  Every routed expert in this backend is executed
/// THROUGH the cache - there is no non-resident execution path to fall back to - so per-layer admission here means
/// a per-layer byte partition: a layer may only evict entries of its own partition (ResidentCache::acquire_in) and
/// can therefore never take another layer's room, but it keeps admitting by evicting ITS OWN coldest expert.  The
/// partition cap is the same number upstream's slot range implies, so what is shared between layers is identical;
/// what differs is that a full layer recycles its own slot instead of missing every time.  This is measured by
/// hpu/tests/expert_residency.cpp and named here so a reader cannot mistake it for the upstream rule.

/// Which experts are admitted BEFORE any routing evidence - the profile fill of generate.cpp:5460-5478.  A run
/// with no static profile admits nothing up front (compulsory-miss fills with whatever it routes first), so the
/// empty vector is both the default and the honest answer for the no-profile case.  `ranked` is the profile's
/// (layer, expert) pairs in descending routing order; the result is deduplicated and, per layer, capped by that
/// layer's own range under --expert-cache-per-layer, or capped globally by the slot count otherwise.
inline std::vector<std::pair<int32_t, int32_t>>
admitted_before_routing(const ExpertResidencyOptions& o, const ExpertCacheShape& s,
                        const std::vector<std::pair<int32_t, int32_t>>& ranked) {
    std::vector<std::pair<int32_t, int32_t>> admitted;
    if (!o.has_expert_profile || s.total_slots <= 0 || s.n_layers <= 0) return admitted;
    std::vector<std::pair<int32_t, int32_t>> seen;
    auto already = [&seen](int32_t l, int32_t e) {
        return std::find(seen.begin(), seen.end(), std::make_pair(l, e)) != seen.end();
    };
    if (o.expert_cache_per_layer) {
        std::vector<int64_t> taken((size_t) s.n_layers, 0);
        for (const auto& p : ranked) {
            const int64_t l = p.first;
            if (l < 0 || l >= s.n_layers || already(p.first, p.second)) continue;
            const SlotRange r = layer_slot_range(s, l);
            const int64_t quota = r.hi - r.lo;
            if (taken[(size_t) l] >= quota) continue;   // this layer's range is full (expert_cache.cpp:648)
            taken[(size_t) l]++;
            seen.push_back(p);
            admitted.push_back(p);
        }
    } else {
        int64_t filled = 0;
        for (const auto& p : ranked) {
            if (p.first < 0 || p.first >= s.n_layers || already(p.first, p.second)) continue;
            if (filled >= s.total_slots) break;         // the global cursor is full (expert_cache.cpp:653)
            filled++;
            seen.push_back(p);
            admitted.push_back(p);
        }
    }
    return admitted;
}

/// The set the port holds against the LRU's eviction: the profile-admitted experts, and only when the copy is
/// page-locked.  --resident-cpu-experts alone keeps a pageable RAM copy (2208 does not set the pin), so its
/// device-resident experts remain ordinary LRU entries; --resident-experts / --resident-budget-gib pin them.
inline std::vector<std::pair<int32_t, int32_t>>
pinned_residents(const ExpertResidencyOptions& o, const ExpertCacheShape& s,
                 const std::vector<std::pair<int32_t, int32_t>>& ranked) {
    if (!o.pins_resident_copy()) return {};
    return admitted_before_routing(o, s, ranked);
}

/// The refusal messages, in upstream's order (2409, 2413, 2421), byte-for-byte including the "strata generate: "
/// prefix.  Empty means accepted.  The --resident-budget-gib N>0 refusal is the caller's (it is a parse error,
/// 2221) and is carried separately as `budget_refused` below.
inline std::string validate_expert_residency(const ExpertResidencyOptions& o) {
    if (o.mmap_experts && !o.shared_expert_arena.empty())
        return "strata generate: --shared-expert-arena backs the resident arena and cannot be used with "
               "--mmap-experts";   // generate.cpp:2410-2411
    if (o.resident_cpu_experts && (!o.mmap_experts || !o.has_expert_profile))
        return "strata generate: --resident-cpu-experts requires --mmap-experts and a static --expert-profile";   // 2414
    if (o.resident_cpu_experts && o.remote_caches)
        return "strata generate: --resident-cpu-experts does not support remote expert caches";   // 2422
    return {};
}

/// The exact --resident-budget-gib refusal (2221).  Kept separate because upstream emits it from the argument
/// parser (return 2), not the post-parse validation block.
inline const char* resident_budget_refusal() {
    return "strata generate: --resident-budget-gib needs N > 0";
}

/// The --kv-grow gate (4399): the elastic K/V arena is off whenever the resident CPU tier is on, because the two
/// RAM tiers would compete for the same memory.  Reported so a plan can say which is in force.
inline bool kv_growth_blocked_by_residency(const ExpertResidencyOptions& o) { return o.resident_cpu_experts; }

/// The FILL ExpertCache::set_expert_residency runs (generate.cpp:5460-5478): walk the admitted pairs IN ORDER,
/// load each into the cache, and stop at the first one that does not fit, KEEPING what already landed - upstream's
/// fill does exactly that (and so does its report of how many are resident).  Pinning follows the same option gate
/// the caller uses (pins_resident_copy).  `load` returns true when the entry landed and must not throw for a full
/// cache; a throwing load is treated as "did not land", so a refusal never loses the entries before it.
struct ResidencyFillResult {
    uint64_t admitted = 0;     ///< entries that landed
    uint64_t pinned = 0;       ///< of those, the ones the caller pinned
    uint64_t requested = 0;    ///< entries the admission asked for
    bool stopped_early = false;///< the fill ran out of room (upstream's `break`)
};
template <class Load>
ResidencyFillResult fill_admitted_residency(const std::vector<std::pair<int32_t, int32_t>>& admitted, const bool pin,
                                            Load&& load) {
    ResidencyFillResult result;
    result.requested = admitted.size();
    for (const auto& entry : admitted) {
        bool landed = false;
        try {
            landed = load(entry.first, entry.second);
        } catch (...) {
            landed = false;
        }
        if (!landed) {
            result.stopped_early = result.admitted < result.requested;
            break;
        }
        ++result.admitted;
        if (pin) ++result.pinned;
    }
    return result;
}

}  // namespace strata::hpu
