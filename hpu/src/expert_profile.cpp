// hpu/src/expert_profile.cpp - the upstream expert-cache profiling and adaptation laws, for the Gaudi2 port.
// Every block below names the upstream lines it transcribes; keep the two in step when either changes.
#include "strata/hpu/expert_profile.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

namespace strata::hpu {
namespace {

// The STRP fields are little-endian on the wire (core/expert_cache.cpp:197).  Assembling and decoding the
// bytes by hand rather than fwrite-ing native integers keeps that promise on a host that is not LE; on the
// LE machines this runs on it produces exactly upstream's bytes.  Declared divergence, not a format change.
void put16(std::vector<uint8_t>& out, uint16_t v) {
    out.push_back((uint8_t) (v & 0xffu));
    out.push_back((uint8_t) (v >> 8));
}
void put32(std::vector<uint8_t>& out, uint32_t v) {
    out.push_back((uint8_t) (v & 0xffu));
    out.push_back((uint8_t) ((v >> 8) & 0xffu));
    out.push_back((uint8_t) ((v >> 16) & 0xffu));
    out.push_back((uint8_t) (v >> 24));
}
uint16_t get16(const uint8_t* p) { return (uint16_t) ((uint16_t) p[0] | ((uint16_t) p[1] << 8)); }
uint32_t get32(const uint8_t* p) {
    return (uint32_t) p[0] | ((uint32_t) p[1] << 8) | ((uint32_t) p[2] << 16) | ((uint32_t) p[3] << 24);
}

}  // namespace

// ---- src/core/expert_cache.cpp:72-139, read_expert_profile, transcribed -------------------------------
bool read_expert_profile(const std::string& path, int64_t n_layers, int64_t n_expert,
                         std::vector<std::pair<int32_t, int32_t>>& ranked, int64_t& slots, std::string& err) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) {
        err = "read_expert_profile: cannot open " + path;
        return false;
    }
    // the whole file: the header is 24 bytes and the body is 4 bytes per ranked pair plus 4 per table entry
    std::fseek(f, 0, SEEK_END);
    const long length = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> bytes;
    if (length > 0) {
        bytes.resize((size_t) length);
        if (std::fread(bytes.data(), 1, (size_t) length, f) != (size_t) length) {
            std::fclose(f);
            err = "read_expert_profile: " + path + " is too short to hold a header";
            return false;
        }
    }
    std::fclose(f);
    if (bytes.size() < 24) {
        err = "read_expert_profile: " + path + " is too short to hold a header";
        return false;
    }
    if (std::memcmp(bytes.data(), expert_profile_magic, 4) != 0) {
        err = "read_expert_profile: " + path + " does not start with STRP";
        return false;
    }
    uint32_t hdr[5];
    for (int i = 0; i < 5; ++i) hdr[i] = get32(bytes.data() + 4 + (size_t) i * 4);
    const uint32_t version = hdr[0], nl = hdr[1], ne = hdr[2], want = hdr[3], n_ranked = hdr[4];
    // the version comes first: the fields after it are only known to be these in a version 1 file (upstream :92-93)
    if (version != expert_profile_version) {
        char buf[256];
        std::snprintf(buf, sizeof buf,
                      "read_expert_profile: %s is a version %u profile but this engine reads version 1 - it was "
                      "written by a different release's tools/make_profile.py or --expert-profile-save",
                      path.c_str(), version);
        err = buf;
        return false;
    }
    if ((int64_t) nl != n_layers || (int64_t) ne != n_expert) {
        char buf[256];
        std::snprintf(buf, sizeof buf,
                      "read_expert_profile: %s is %ux%u but this model is %lldx%lld - it is a profile for a "
                      "different artifact", path.c_str(), nl, ne, (long long) n_layers, (long long) n_expert);
        err = buf;
        return false;
    }
    if (n_ranked > want) {
        err = "read_expert_profile: the header claims more ranked pairs than slots";
        return false;
    }
    // upstream reads only the header and the ranked pairs and never touches the trailing table
    // (core/expert_cache.cpp:118-124; the header's note at core/expert_cache.hpp:52-53 says why), so a short
    // TABLE is not a read failure here either - only a short ranked list is.
    if (bytes.size() < 24 + (size_t) n_ranked * 4) {
        err = "read_expert_profile: the ranked list is truncated";
        return false;
    }
    ranked.assign(n_ranked, {0, 0});
    for (uint32_t i = 0; i < n_ranked; ++i) {
        const int32_t l = (int32_t) get16(bytes.data() + 24 + (size_t) i * 4);
        const int32_t e = (int32_t) get16(bytes.data() + 24 + (size_t) i * 4 + 2);
        if (l < 0 || l >= n_layers || e < 0 || e >= n_expert) {
            char buf[256];
            std::snprintf(buf, sizeof buf, "read_expert_profile: pair %u is (layer %d, expert %d), out of range",
                          i, l, e);
            err = buf;
            return false;
        }
        ranked[(size_t) i] = {l, e};
    }
    slots = (int64_t) want;
    return true;
}

// ---- src/core/expert_cache.cpp:141-168, rank_learned_profile, transcribed -------------------------------
std::vector<std::pair<int32_t, int32_t>> rank_learned_profile(int64_t n_layers, int64_t n_expert,
                                                             const std::vector<uint8_t>& resident,
                                                             const std::vector<double>& heat,
                                                             const std::vector<std::pair<int32_t, int32_t>>& prior) {
    const size_t n = (size_t) (n_layers * n_expert);
    std::vector<int64_t> prior_rank(n, std::numeric_limits<int64_t>::max());
    for (size_t r = 0; r < prior.size(); ++r) {
        const int32_t l = prior[r].first, e = prior[r].second;
        if (l >= 0 && l < n_layers && e >= 0 && e < n_expert) {
            int64_t& pr = prior_rank[(size_t) (l * n_expert + e)];
            if (pr == std::numeric_limits<int64_t>::max()) pr = (int64_t) r;
        }
    }
    std::vector<int64_t> order(n);
    for (size_t i = 0; i < n; ++i) order[i] = (int64_t) i;
    auto res = [&](int64_t i) { return (size_t) i < resident.size() && resident[(size_t) i] != 0; };
    auto ht = [&](int64_t i) { return (size_t) i < heat.size() ? heat[(size_t) i] : 0.0; };
    std::stable_sort(order.begin(), order.end(), [&](int64_t a, int64_t b) {
        if (res(a) != res(b)) return res(a);
        if (ht(a) != ht(b)) return ht(a) > ht(b);
        if (prior_rank[(size_t) a] != prior_rank[(size_t) b]) return prior_rank[(size_t) a] < prior_rank[(size_t) b];
        return a < b;
    });
    std::vector<std::pair<int32_t, int32_t>> ranked(n);
    for (size_t r = 0; r < n; ++r)
        ranked[r] = {(int32_t) (order[r] / n_expert), (int32_t) (order[r] % n_expert)};
    return ranked;
}

// ---- src/core/expert_cache.cpp:170-210, write_expert_profile, transcribed -------------------------------
bool write_expert_profile(const std::string& path, int64_t n_layers, int64_t n_expert,
                          const std::vector<std::pair<int32_t, int32_t>>& ranked, std::string& err) {
    if (n_layers <= 0 || n_expert <= 0 || n_layers > 65535 || n_expert > 65535) {
        err = "write_expert_profile: the model's layout does not fit the format";
        return false;
    }
    std::vector<int32_t> table((size_t) (n_layers * n_expert), -1);
    std::vector<uint8_t> body;
    body.reserve(24 + ranked.size() * 4 + table.size() * 4);
    for (size_t r = 0; r < ranked.size(); ++r) {
        const int32_t l = ranked[r].first, e = ranked[r].second;
        if (l < 0 || l >= n_layers || e < 0 || e >= n_expert) {
            err = "write_expert_profile: a ranked pair is out of range";
            return false;
        }
        table[(size_t) (l * n_expert + e)] = (int32_t) r;
    }
    body.insert(body.end(), expert_profile_magic, expert_profile_magic + 4);   // :198
    put32(body, expert_profile_version);
    put32(body, (uint32_t) n_layers);
    put32(body, (uint32_t) n_expert);
    put32(body, (uint32_t) ranked.size());   // the slots field: upstream writes ranked.size() (:189-190)
    put32(body, (uint32_t) ranked.size());
    for (const auto& p : ranked) { put16(body, (uint16_t) p.first); put16(body, (uint16_t) p.second); }
    for (const int32_t r : table) put32(body, (uint32_t) r);
    // atomically: a `<path>.tmp` beside it, renamed over `path` once complete (:191-207)
    const std::string tmp = path + ".tmp";
    std::FILE* f = std::fopen(tmp.c_str(), "wb");
    if (f == nullptr) {
        err = "write_expert_profile: cannot create " + tmp;
        return false;
    }
    const bool ok = body.empty() || std::fwrite(body.data(), 1, body.size(), f) == body.size();
    const bool closed = std::fclose(f) == 0;
    // **DECLARED DIVERGENCE, AND IT IS ONLY ABOUT THE CALL.**  Upstream uses std::filesystem::rename, whose
    // POSIX rename(2) replaces an existing file; the C rename does the same here, and where it does not (a
    // Windows host, where a re-save over an existing profile would otherwise be refused) the target is dropped
    // and the rename retried.  Nothing partial is ever left at `path`: the tmp is only ever renamed or removed.
    bool replaced = false;
    if (ok && closed) {
        replaced = std::rename(tmp.c_str(), path.c_str()) == 0;
        if (!replaced) {
            std::remove(path.c_str());
            replaced = std::rename(tmp.c_str(), path.c_str()) == 0;
        }
    }
    if (!replaced) {
        std::remove(tmp.c_str());
        err = "write_expert_profile: cannot write " + path;
        return false;
    }
    return true;
}

// ---- the profiler --------------------------------------------------------------------------------------
ExpertProfiler::ExpertProfiler(int64_t n_layers, int64_t n_expert, ExpertProfileOptions options)
    : n_layers_(n_layers < 0 ? 0 : n_layers), n_expert_(n_expert < 0 ? 0 : n_expert),
      options_(std::move(options)), saved_at_(Clock::now()) {
    if (n_layers_ <= 0 || n_expert_ <= 0) throw std::invalid_argument("expert profiler needs a positive layout");
    allocate();
}

void ExpertProfiler::set_options(ExpertProfileOptions options) {
    options_ = std::move(options);
    allocate();
}

void ExpertProfiler::allocate() {
    const size_t n = (size_t) n_layers_ * (size_t) n_expert_;
    // Upstream sizes the routing counts by the tier, not by the file options: `usage` exists only under
    // `adapt_every > 0 && adapt_swaps > 0` (generate.cpp:13573), and `heat` only with --expert-profile-save
    // and a live tier (generate.cpp:8711-8717).  The residency table exists with either.
    const size_t usage_n = options_.tier_enabled() ? n : 0;
    const size_t heat_n = (options_.tier_enabled() && !options_.save.empty()) ? n : 0;
    const size_t watch_n = options_.monitoring() ? n : 0;
    // Re-sized only when a counter set appears or disappears: moving --adapt-decay, --adapt-min-gain,
    // --adapt-every, --adapt-swaps or --adapt-async after the fact must not throw away what was counted.
    if (usage_.size() != usage_n) usage_.assign(usage_n, 0.0f);
    if (heat_.size() != heat_n) heat_.assign(heat_n, 0.0);
    if (hits_.size() != watch_n) hits_.assign(watch_n, 0);
    if (misses_.size() != watch_n) misses_.assign(watch_n, 0);
    if (res_.size() != watch_n) res_.assign(watch_n, kExpertNotResident);
    snap_usage_.clear();
    snap_res_.clear();
    async_in_flight_ = false;
}

void ExpertProfiler::note_routes(int64_t layer, const int32_t* ids, int64_t n) {
    if (usage_.empty()) return;   // upstream's guard is `if (!d.usage.empty())` (expert_source.cpp:3717)
    if (ids == nullptr || layer < 0 || layer >= n_layers_ || n <= 0) return;
    float* u = usage_.data() + (size_t) layer * (size_t) n_expert_;
    for (int64_t i = 0; i < n; ++i)
        if (ids[i] >= 0 && ids[i] < n_expert_) u[(size_t) ids[i]] += 1.0f;   // :3719
}

void ExpertProfiler::note_hit(int64_t layer, int64_t expert) {
    if (hits_.empty() || layer < 0 || layer >= n_layers_ || expert < 0 || expert >= n_expert_) return;
    ++hits_[(size_t) layer * (size_t) n_expert_ + (size_t) expert];
}

void ExpertProfiler::note_miss(int64_t layer, int64_t expert) {
    if (misses_.empty() || layer < 0 || layer >= n_layers_ || expert < 0 || expert >= n_expert_) return;
    ++misses_[(size_t) layer * (size_t) n_expert_ + (size_t) expert];
}

float ExpertProfiler::usage(int64_t layer, int64_t expert) const {
    if (usage_.empty() || layer < 0 || layer >= n_layers_ || expert < 0 || expert >= n_expert_) return 0.0f;
    return usage_[(size_t) layer * (size_t) n_expert_ + (size_t) expert];
}
uint64_t ExpertProfiler::hits(int64_t layer, int64_t expert) const {
    if (hits_.empty() || layer < 0 || layer >= n_layers_ || expert < 0 || expert >= n_expert_) return 0;
    return hits_[(size_t) layer * (size_t) n_expert_ + (size_t) expert];
}
uint64_t ExpertProfiler::misses(int64_t layer, int64_t expert) const {
    if (misses_.empty() || layer < 0 || layer >= n_layers_ || expert < 0 || expert >= n_expert_) return 0;
    return misses_[(size_t) layer * (size_t) n_expert_ + (size_t) expert];
}
double ExpertProfiler::heat(int64_t layer, int64_t expert) const {
    if (heat_.empty() || layer < 0 || layer >= n_layers_ || expert < 0 || expert >= n_expert_) return 0.0;
    return heat_[(size_t) layer * (size_t) n_expert_ + (size_t) expert];
}

void ExpertProfiler::set_resident(int64_t layer, int64_t expert, int32_t slot) {
    if (res_.empty() || layer < 0 || layer >= n_layers_ || expert < 0 || expert >= n_expert_) return;
    res_[(size_t) layer * (size_t) n_expert_ + (size_t) expert] = slot;
}
bool ExpertProfiler::is_resident(int64_t layer, int64_t expert) const {
    if (res_.empty() || layer < 0 || layer >= n_layers_ || expert < 0 || expert >= n_expert_) return false;
    return res_[(size_t) layer * (size_t) n_expert_ + (size_t) expert] >= 0;
}
void ExpertProfiler::clear_residency() { std::fill(res_.begin(), res_.end(), kExpertNotResident); }

void ExpertProfiler::sync_residency(const std::vector<std::string>& keys) {
    if (res_.empty()) return;
    clear_residency();
    for (const std::string& key : keys) {
        const size_t sep = key.find(':');
        if (sep == std::string::npos) continue;
        const int64_t layer = std::strtoll(key.substr(0, sep).c_str(), nullptr, 10);
        const int32_t expert = (int32_t) std::strtol(key.substr(sep + 1).c_str(), nullptr, 10);
        if (layer >= 0 && layer < n_layers_ && expert >= 0 && expert < n_expert_)
            res_[(size_t) layer * (size_t) n_expert_ + (size_t) expert] = 0;
    }
}

/// generate.cpp:8839-8858 (the per-layer choice) and :8858 (the gain sort), over the given counts and
/// residency table.  The cap is NOT applied here: the two tiers apply it differently (see run_round /
/// settle_async_round), so sharing the choice and not the cap keeps both transcriptions literal.
std::vector<ExpertSwap> ExpertProfiler::proposals_from(const std::vector<float>& usage,
                                                      const std::vector<int32_t>& res) const {
    std::vector<ExpertSwap> swaps;
    if (usage.empty() || res.empty()) return swaps;
    // the same element type upstream sorts (`std::pair<float, int32_t>`, generate.cpp:8838), so the
    // tie-breaking of the sorts below is the same sort on the same type, not a lookalike
    std::vector<std::pair<float, int32_t>> cand, vict;
    for (int64_t l = 0; l < n_layers_; ++l) {
        cand.clear();
        vict.clear();
        const float* u = usage.data() + l * n_expert_;
        const int32_t* r = res.data() + l * n_expert_;
        for (int32_t e = 0; e < (int32_t) n_expert_; ++e) {
            // the same law, the same threshold: a missing expert is a candidate only at 2.0 counts or more
            if (r[e] < 0) {
                if (u[e] >= 2.0f && !(excludes_ && excludes_((int32_t) l, e))) cand.emplace_back(u[e], e);
            } else vict.emplace_back(u[e], e);
        }
        if (cand.empty() || vict.empty()) continue;
        std::sort(cand.begin(), cand.end(),
                  [](const std::pair<float, int32_t>& a, const std::pair<float, int32_t>& b) { return a.first > b.first; });
        const size_t nc = std::min(cand.size(), vict.size());
        std::partial_sort(vict.begin(), vict.begin() + (std::ptrdiff_t) nc, vict.end(),
                          [](const std::pair<float, int32_t>& a, const std::pair<float, int32_t>& b) { return a.first < b.first; });
        for (size_t i = 0; i < nc; ++i) {
            if (cand[i].first < vict[i].first + options_.adapt_min_gain) break;   // :8854 - break, not continue
            swaps.push_back({cand[i].first - vict[i].first, (int32_t) l, cand[i].second, vict[i].second});
        }
    }
    std::sort(swaps.begin(), swaps.end(), [](const ExpertSwap& a, const ExpertSwap& b) { return a.gain > b.gain; });
    return swaps;
}

std::vector<ExpertSwap> ExpertProfiler::ranked_proposals() const { return proposals_from(usage_, res_); }

void ExpertProfiler::apply_decay() {
    // generate.cpp:8987 / 9271 / 13703, verbatim: a per-element float multiply over the whole table.
    for (float& v : usage_) v *= options_.adapt_decay;
}

void ExpertProfiler::accumulate_heat() {
    // generate.cpp:8976 / 9270, verbatim.  `heat` is empty unless --expert-profile-save, so this is free
    // otherwise; the count it adds is the round's, taken BEFORE the decay (upstream's comment at :8973-8974
    // notes each count is therefore worth up to 1/(1-decay) over its decays).
    for (size_t i = 0; i < heat_.size(); ++i) heat_[i] += (double) usage_[i];
}

bool ExpertProfiler::run_round(std::vector<ExpertSwap>& applied) {
    applied.clear();
    if (!counting()) return true;
    // generate.cpp:8834: a round does not stack on copies that have not landed
    if (!res_.empty()) {
        applied = proposals_from(usage_, res_);
        // :8859: at most --adapt-swaps changes per round, and the ones kept are the highest-gain
        if ((int) applied.size() > options_.adapt_swaps) applied.resize((size_t) options_.adapt_swaps);
        for (const ExpertSwap& s : applied) {
            const size_t in = (size_t) s.layer * (size_t) n_expert_ + (size_t) s.in;
            const size_t out = (size_t) s.layer * (size_t) n_expert_ + (size_t) s.out;
            res_[out] = kExpertNotResident;   // :8941 evicted now, the CPU computes it meanwhile
            res_[in] = 0;                     // :8943 resident once the copy has landed
        }
        swaps_ += (int64_t) applied.size();
    }
    accumulate_heat();   // :8976, after the picks it just fed
    apply_decay();       // :8987, last
    ++rounds_;
    return true;
}

bool ExpertProfiler::begin_async_round() {
    if (!counting()) return true;
    // generate.cpp:9264-9271: the round starts by taking its snapshots, folding this round's counts into
    // `heat` and decaying IMMEDIATELY - before the choice is made.  That is what --adapt-async changes about
    // WHEN the update happens; the blocking tier decays only after its copies have landed (:8987).
    snap_usage_ = usage_;   // :9267
    snap_res_ = res_;       // :9268
    accumulate_heat();      // :9270
    apply_decay();          // :9271
    async_in_flight_ = true;
    ++rounds_;
    return true;
}

void ExpertProfiler::settle_async_round(std::vector<ExpertSwap>& applied) {
    applied.clear();
    if (!async_in_flight_) return;
    const std::vector<ExpertSwap> all = proposals_from(snap_usage_, snap_res_);   // :9095-9118
    // :9120-9123: walk the gain-ordered list, skip a victim that lost its slot since the snapshot, and stop
    // at the cap.  (Upstream's cap is `a_arena ? adapt_swaps : min(adapt_swaps, 96)`; the 96 is the CUDA
    // exchange-buffer reservation at :7130/:7137, which this port has no equivalent of, so the cap here is
    // --adapt-swaps.  Declared divergence.)
    for (const ExpertSwap& c : all) {
        if ((int64_t) applied.size() >= options_.adapt_swaps) break;
        const size_t out = (size_t) c.layer * (size_t) n_expert_ + (size_t) c.out;
        if (snap_res_[out] < 0) continue;   // :9122-9123
        const size_t in = (size_t) c.layer * (size_t) n_expert_ + (size_t) c.in;
        res_[in] = 0;
        res_[out] = kExpertNotResident;
        applied.push_back(c);
    }
    swaps_ += (int64_t) applied.size();
    snap_usage_.clear();
    snap_res_.clear();
    async_in_flight_ = false;
}

bool ExpertProfiler::tick(std::vector<ExpertSwap>& applied) {
    if (!counting()) return true;   // no tier: nothing to run, and nothing was ever counted
    ++windows_;
    // generate.cpp:10031 `(++ba_window % o.adapt_every) == 0`, 12133 `(rounds % o.adapt_every) == 0`,
    // 13863 `((rounds + 1) % o.adapt_every) == 0`: a round on every --adapt-every-th call.
    if (options_.adapt_every <= 0 || windows_ % (int64_t) options_.adapt_every != 0) return true;
    if (async_in_flight_) return true;   // :10031's `!ajob` - one asynchronous round at a time
    if (options_.adapt_async) return begin_async_round();
    return run_round(applied);
}

bool ExpertProfiler::preload_profile(std::string& err) {
    if (options_.profile.empty()) {
        err = "expert profiler: --expert-profile was not given";
        return false;
    }
    int64_t slots = 0;
    prior_.clear();
    if (!read_expert_profile(options_.profile, n_layers_, n_expert_, prior_, slots, err)) return false;
    // the ranked pairs are the tier, hottest first: the first `slots` of them are what the run starts resident
    clear_residency();
    const size_t keep = (size_t) slots < prior_.size() ? (size_t) slots : prior_.size();
    for (size_t r = 0; r < keep; ++r)
        res_[(size_t) prior_[r].first * (size_t) n_expert_ + (size_t) prior_[r].second] = 0;
    return true;
}

bool ExpertProfiler::save_due(Clock::time_point now) const {
    if (!saving()) return false;
    // generate.cpp:10364-10367: only when a MIN > 0 was asked for, and only once MIN minutes have passed
    // since the last save.  0 = on shutdown only (the option's own doc, :871-872).
    if (options_.save_every_minutes <= 0) return false;
    return now - saved_at_ >= std::chrono::duration<double>(options_.save_every_minutes * 60.0);
}

bool ExpertProfiler::save_profile(Clock::time_point now, std::string& err) {
    if (!saving()) {
        err = "expert profiler: --expert-profile-save needs the adaptive tier: nothing will be saved";
        return false;
    }
    if (options_.save.empty()) {
        err = "expert profiler: --expert-profile-save has no path";
        return false;
    }
    // generate.cpp:9418-9419: the resident set as the table says it, an in-flight swap's expert already counted
    std::vector<uint8_t> resident(res_.size(), 0);
    for (size_t i = 0; i < res_.size(); ++i) resident[i] = res_[i] >= 0;
    if (async_in_flight_)
        for (const ExpertSwap& s : proposals_from(snap_usage_, snap_res_))
            resident[(size_t) s.layer * (size_t) n_expert_ + (size_t) s.in] = 1;   // :9421-9422
    const auto ranked = rank_learned_profile(n_layers_, n_expert_, resident, heat_, prior_);   // :9424
    if (!write_expert_profile(options_.save, n_layers_, n_expert_, ranked, err)) return false;
    saved_at_ = now;   // :9431, and it restarts the --expert-profile-save-every window
    return true;
}

}  // namespace strata::hpu
