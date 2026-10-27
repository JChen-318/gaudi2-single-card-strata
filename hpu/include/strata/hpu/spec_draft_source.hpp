#pragma once
// Upstream's speculative-decoding verification/diagnostic knobs, ported to this backend:
//   --spec-oracle  src/program/generate.cpp:559-560,579,2085,13521-13524,13800-13801
//   --spec-follow  src/program/generate.cpp:581-584,2087,13525-13528,13530,13836-13842,13882,13917-13919
//   --spec-corrupt src/program/generate.cpp:560,580,2086,13730,13803,13814
// This header is host-pure (no runtime.hpp, no Synapse), matching the pattern of prefill_schedule.hpp,
// so Generation and the tests can share it without the device toolchain.  Everything here substitutes
// or post-processes DRAFT tokens and the emitted rows; it never touches the accept/reject math
// (verify_pick / select_verified_candidates are consumed as they are).
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace strata::hpu {
// Upstream's file reader is read_ids (src/program/generate.cpp:13514-13519) over parse_i64_list
// (965-983): comma-or-whitespace separated integers in [0,2147483647], rejected only as
// "invalid token id: expected an integer in [0, 2147483647]" (975) or "token list was empty" (981),
// and both --spec-oracle and --spec-follow failures print the single message "cannot read ..." (13522,13526).
// This port keeps the same accepted grammar but names the refusal reasons distinctly (declared
// refinement; upstream conflates them).  A NUL byte is upstream an "invalid token id" (from_chars
// fails on the extracted token); here it is its own reason, as the task's diagnostic demands.
enum class SpecTokenFileIssue { Ok, Unreadable, NulByte, MalformedId, Empty, BelowWindow, OutsideVocabulary };
struct SpecTokenFile {
    SpecTokenFileIssue issue=SpecTokenFileIssue::Ok;
    std::string message; // distinct, non-empty for every refusal
    std::vector<int32_t> ids;
};
// `vocabulary` bounds the ids upstream only bounds at 2^31 (validate_drafts, hpu/src/verify.cpp:11,
// is the port's real in-vocabulary gate); `minimum_count` is the refusal threshold for the oracle's
// window (upstream instead pads exhausted positions with token 0 - kept verbatim at runtime by
// spec_oracle_window, so the check only gates configuration, declared).  `kind` names the knob in
// the refusal ("oracle" for --spec-oracle, "follow" for --spec-follow, matching the flag upstream
// quotes at 13522/13526).
SpecTokenFile load_spec_tokens(const std::string& path,uint32_t vocabulary,size_t minimum_count,const std::string& kind);

// The exact corruptor of src/program/generate.cpp:13803:
//   if (o.spec_corrupt > 0 && (++corrupt_counter % o.spec_corrupt) == 0) d = (d + 1) % (int32_t) n_vocab;
// The counter is global (declared at 13730) and advances once per assembled draft slot in window
// order across all rounds of the run, whatever produced the slot (oracle, MTP, suffix or chain).
class SpecCorruptor {
public:
    SpecCorruptor(int64_t every,int32_t vocabulary);
    bool active() const { return every_>0; }
    // Advances the counter (upstream's ++corrupt_counter fires even on slots later dropped by an
    // earlier rejection, because assembly precedes verification); rewrites `draft` in place and
    // returns true when this slot is the Nth.
    bool next(int32_t& draft);
    int64_t counter() const { return counter_; }
private:
    int64_t every_,counter_=0;
    int32_t vocabulary_;
};

// The exact draft selection of src/program/generate.cpp:13800-13801:
//   const size_t at = produced.size() - 1 + i;   // i is the 1-based window slot, slot 0 is `x`
//   int32_t d = ... at < oracle.size() ? (int32_t) oracle[at] : 0;
// With the port's 0-based draft rows (row r is upstream's slot i=r+1) and `base` tokens emitted
// this round's run so far, at = base + r.  An exhausted oracle yields token 0, verbatim.
std::vector<int32_t> spec_oracle_window(const std::vector<int32_t>& oracle,size_t base,size_t rows);

// The exact follow rule, transcribed:
//   13530: max_new = follow.empty() ? o.max_new : std::min<int64_t>(o.max_new, (int64_t) follow.size())
//   13836-13841, per emitted row i of the window with k0 = produced.size():
//          fdiff[i] = outv[i] != follow[k0+i]; outv[i] = follow[k0+i];   // rows past the file keep argmax
//   13844: acceptance compares the window against the REPLACED rows: while (a < T-1 && window[a+1]==outv[a]) ++a
//   13882: follow_differ += fdiff[i]; ++follow_emitted;   // accumulated per emitted token only
uint64_t spec_follow_max_new(uint64_t requested,const std::vector<int32_t>& follow);
struct SpecFollowRow { int32_t token; bool differs; };
// `emitted_index` is the run-relative 0-based index of the row's position (= upstream k0+i).
SpecFollowRow spec_follow_row(const std::vector<int32_t>& follow,size_t emitted_index,int32_t argmax);

// The knob set the CLI hands to Generation::set_spec_knobs.  Default-constructed means "no flags":
// every hook then degenerates to today's behaviour, unchanged.
struct SpecKnobRequest {
    std::string oracle_path; // --spec-oracle FILE
    std::string follow_path; // --spec-follow FILE (a token file, NOT a count: src/program/generate.cpp:581-584)
    int64_t corrupt=0;       // --spec-corrupt N (N>0 flips every Nth assembled draft)
};
class SpecKnobs {
public:
    // Parses and validates both files here, before any device work, throwing std::invalid_argument
    // with the distinct message for each refusal (duplicates upstream's one "cannot read" surface).
    // `window` is the per-round draft count the oracle must be able to cover (GenerationOptions::max_draft).
    void configure(const SpecKnobRequest& request,uint32_t vocabulary,uint32_t window);
    bool oracle_active() const { return !oracle_.empty(); }
    bool follow_active() const { return !follow_.empty(); }
    bool corrupt_active() const { return corruptor_.active(); }
    const std::vector<int32_t>& oracle_tokens() const { return oracle_; }
    const std::vector<int32_t>& follow_tokens() const { return follow_; }
    // Draft rows for one window from the oracle (point-mass tokens; the drafter's own source is
    // replaced wholesale - declared: upstream's oracle only fills the no-MTP/no-suffix branch of 13800).
    std::vector<int32_t> oracle_window(size_t base,size_t rows) const { return spec_oracle_window(oracle_,base,rows); }
    // One assembled draft slot through the shared corrupt counter.
    bool corrupt_slot(int32_t& draft) { return corruptor_.next(draft); }
    uint64_t follow_max_new(uint64_t requested) const { return spec_follow_max_new(requested,follow_); }
    SpecFollowRow follow_row(size_t emitted_index,int32_t argmax) const { return spec_follow_row(follow_,emitted_index,argmax); }
    int64_t corrupt_counter() const { return corruptor_.counter(); }
private:
    std::vector<int32_t> oracle_,follow_;
    SpecCorruptor corruptor_{0,0};
};
}
