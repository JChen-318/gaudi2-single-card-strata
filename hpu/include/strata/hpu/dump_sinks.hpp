#pragma once
// Upstream's per-position diagnostic dumps, ported for the single-card Gaudi2 backend:
// --dump-logits, --dump-layers, --dump-halves, --dump-mixed, --dump-residual, --dump-final-r,
// --dump-routing.  These are VALIDATION plumbing, so the byte layout and the record ORDER are the
// contract: two runs must be diffable, and a dump file must be readable by upstream's own tools
// (tools/logits_identical.py, tools/make_profile.py, llama-debug's `l_last-<il>` ladder).
//
// Authority for every layout below is src/program/generate.cpp at commit 296b59d:
//   --dump-logits   decl :347   parse :1930   open/geometry :6072-6097   write :13405-13432
//                   close :13997-14000   summary :14171   stride law :2631-2633
//                   header int32 {n_vocab, n_rows} written WITH THE FIRST ROW, then one row of
//                   n_vocab float32 per emitted position; `n_rows` counts positions the way
//                   :6077/:6095 do (tokens.size()-1+max_new, stride-selected).
//   --dump-layers   decl :408-416  parse :1940  open :6103-6117  write :13382  close :14001-14006
//                   refusal with --no-capture :5996-6001.  float32, (n_layers+1)*hc*n_embd floats
//                   per position, appended in position order; slot 0 is the INPUT residual, slots
//                   1..n_layers are R after layer l (src/core/session.cpp:664-672, :759-766).
//   --dump-halves   decl :420  parse :1941  stride :3252-3254  open :3257-3269  write :13383-13385
//                   close :14007-14012  refusal with --no-capture :6002-6007.  float32,
//                   n_layers*half_stride per position, half_stride =
//                   2*n_embd + 2*hc + n_head*head_dim + 5*n_head_kv*head_dim + 8.
//   --dump-mixed    decl :411  parse :1939  open+write :14034-14052 (ONE file, ONE shot, at the end
//                   of the run, NO header): n_embd float32 of the head input `bb.mixed`.
//   --dump-residual decl :408  parse :1938  open+write :14055-14078 (ONE shot, at the end):
//                   int32 {hc, n_embd} then hc*n_embd float32 of `ss.R`.
//   --dump-final-r  decl :555-557 parse :2073 open :13162  write :13363-13368 (TOKEN PATH ONLY):
//                   per processed position, int64 {pos, tok} then hc*n_embd float32 - the residual
//                   AFTER the last layer, the MTP draft head's input.
//   --dump-routing  decl :421-425 parse :1942 open :5926-5939 write :1007-1023 (token) and
//                   :1039-1047 (verify window, one record per token, weights are ZEROS because the
//                   fused multi-dispatch does not surface them) close :14013-14018.  Record =
//                   int32 {layer, k}, k int32 ids, k float32 weights; a layer outside the model's
//                   0..n_layers-1 is reported and the record is NOT written (:1014-1018).
//
// This header is deliberately standalone: it pulls in nothing from the Synapse runtime
// (strata/hpu/runtime.hpp) so a host test can compile and run it without a device.
#include "strata/program/logits_selection.hpp" // upstream's stride law, :7-29 of that header

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace strata::hpu {

// One sampled routing record: the layer, the k experts it routed to and their weights. Upstream's record is
// int32 {layer,k}, k int32 ids, k float32 weights (src/program/generate.cpp:1007-1023); the token path writes
// the real weights, the verify-window form writes zeros (:1035-1045).
struct RoutingRecord {
    int32_t layer = 0;
    std::vector<int32_t> ids;
    std::vector<float> weights;
};
// Borrowed layer-major slabs from ONE physical micro-window. The stride is
// measured in expert entries, not rows. Valid only during its synchronous sink.
struct RoutingWindowView {
    uint32_t layers=0,rows=0,top_k=0;
    uint64_t layer_stride=0;
    std::span<const int32_t> ids;
    std::span<const float> weights;
};
// window=true uses upstream multi-dispatch's zero-weight wire convention;
// token fallback publishes its true router weights. Never re-enter Model.
using RoutingWindowSink=std::function<void(const RoutingWindowView&,bool window)>;

// One path per flag.  An empty path means that dump is OFF, and off must cost nothing: no file is
// opened, no buffer is reserved, and every writer is a single bool test away from returning.
struct DumpOptions {
    std::string logits;    // --dump-logits
    std::string layers;    // --dump-layers
    std::string halves;    // --dump-halves
    std::string mixed;     // --dump-mixed
    std::string residual;  // --dump-residual
    std::string final_r;   // --dump-final-r
    std::string routing;   // --dump-routing
    bool per_position() const {
        return !logits.empty() || !layers.empty() || !halves.empty() ||
               !final_r.empty() || !routing.empty();
    }
    bool token_positions() const {
        // These writers run in the scalar token loop; raw files do not promise
        // records for a prefix consumed by batched prefill. Logits separately
        // carries a declared row count and retains its coverage restrictions.
        return !logits.empty() || !layers.empty() || !halves.empty() || !final_r.empty();
    }
    bool none() const {
        return logits.empty() && layers.empty() && halves.empty() && mixed.empty() &&
               residual.empty() && final_r.empty() && routing.empty();
    }
    int enabled() const {
        return int(!logits.empty()) + int(!layers.empty()) + int(!halves.empty()) +
               int(!mixed.empty()) + int(!residual.empty()) + int(!final_r.empty()) +
               int(!routing.empty());
    }
    bool routing_only() const { return !routing.empty() && enabled()==1; }
};

// The model shape every record is sized against, plus the run's position budget.  Mirrors
// core::ModelGeometry's fields the upstream writers read (g.n_vocab, g.n_layers, g.hc, g.n_embd,
// g.n_head, g.head_dim, g.n_head_kv) and the two logits terms upstream derives at :6077/:6095.
struct DumpGeometry {
    int64_t n_vocab = 0;
    int64_t n_layers = 0;
    int64_t hc = 0;
    int64_t n_embd = 0;
    int64_t n_head = 0;
    int64_t head_dim = 0;
    int64_t n_head_kv = 0;
    int64_t n_expert = 0;    // bounds a routing record's k, as upstream's 48 bounds its layer
    int64_t positions = 0;   // upstream dump_positions (:6077) = prompt tokens - 1 + max_new
    int64_t stride = 1;      // --logits-stride
    uint32_t max_new = 1;    // :2631's law only permits a stride > 1 with max_new == 1

    // 2*n_embd + 2*hc + n_head*head_dim + 5*n_head_kv*head_dim + 8, exactly :3252-3254.
    uint64_t half_stride() const {
        return uint64_t(2) * uint64_t(n_embd) + uint64_t(2) * uint64_t(hc) +
               uint64_t(n_head) * uint64_t(head_dim) +
               uint64_t(5) * uint64_t(n_head_kv) * uint64_t(head_dim) + uint64_t(8);
    }
    uint64_t layer_floats() const { return uint64_t(n_layers + 1) * uint64_t(hc) * uint64_t(n_embd); }
    uint64_t final_r_floats() const { return uint64_t(hc) * uint64_t(n_embd); }
    // n_rows of the logits header: row_count over the same position count upstream uses (:6095).
    uint64_t logits_rows() const {
        return uint64_t(strata::program::logits_selection::row_count(positions, stride));
    }
};

// Which source paths in THIS session can actually feed a kind.  Upstream refuses a flag whose
// producer never runs rather than writing a file of uninitialised memory - "a file of zeros ...
// reads as a wrong answer rather than as a mistake" (:5993-6007, and the --no-pool law at
// :5928-5932).  The port's equivalent of `--no-capture` is "no hook was wired".
struct DumpProducers {
    bool logits = true;          // the token path returns logits (GenerationOperations::decode)
    bool layer_ladder = false;   // in-model per-layer residual hook (hpu/src/model.cpp)
    bool half_capture = false;   // in-model per-layer half outputs (captured graphs)
    bool routing = false;        // hpu/src/moe.cpp router hook
    bool final_r = false;        // post-last-layer residual available per position
    bool head_residual = false;  // the same residual readable at the END of the run
    bool head_mixed = false;     // the head input `mixed` readable at the END of the run
    // A kind a producer cannot feed must be refused at open; the refusal text is upstream's rule
    // that a producer-less dump would write a file of uninitialised memory (:5993-6007).
};

// The sinks, in the order upstream opens them: per-position files at arm time, the two one-shot
// files only when the run is over and the value is in hand (:14034-14078).
class DumpSinks {
public:
    DumpSinks() = default;
    DumpSinks(const DumpOptions& options, const DumpGeometry& geometry, const DumpProducers& producers);
    DumpSinks(DumpSinks&&) noexcept;
    DumpSinks& operator=(DumpSinks&&) noexcept;
    DumpSinks(const DumpSinks&) = delete;
    DumpSinks& operator=(const DumpSinks&) = delete;
    ~DumpSinks();

    const DumpGeometry& geometry() const { return geometry_; }
    bool active() const { return any_; }                        // false => every writer returns at once
    bool wants_logits() const { return logits_ != nullptr; }
    bool wants_layers() const { return layers_ != nullptr; }
    bool wants_halves() const { return halves_ != nullptr; }
    bool wants_mixed() const { return !options_.mixed.empty(); }
    bool wants_residual() const { return !options_.residual.empty(); }
    bool wants_final_r() const { return final_r_ != nullptr; }
    bool wants_routing() const { return routing_ != nullptr; }
    // The process owns this writer across GEN/BGEN/slot calls. Only routing
    // has a headerless physical-dispatch format suitable for this lifetime.
    bool routing_session_matches(const DumpOptions&,const DumpGeometry&) const;

    // ---- the per-position writers, in upstream's record order -------------------------------
    // `position` is the processed position; the stride law and the header-with-first-row rule are
    // applied here, exactly as :13405-13427.  A row of the wrong length is a geometry mismatch.
    void logits(int64_t position, std::span<const float> row);
    void layers(std::span<const float> stage);     // (n_layers+1)*hc*n_embd floats, layer 0 first
    void halves(std::span<const float> stage);     // n_layers*half_stride floats, layer 0 first
    void final_residual(int64_t position, int64_t token, std::span<const float> residual);
    void routing(int32_t layer, int64_t k, const int32_t* ids, const float* weights);
    // The verify-window form: one record per token, weights written as zeros - upstream cannot
    // surface the fused router weights and documents the convention at :1035-1045.
    void routing_window(int32_t layer, int64_t k, int64_t tokens, const int32_t* ids);
    // Validate the complete borrowed physical view before writing any records.
    // Used by both Generation and the shared resident/slot writer.
    void routing_view(const RoutingWindowView&,bool window);
    // A routing callback must surface buffered IO errors before its numeric
    // transaction commits/publishes. Optional debug IO, never a Model reentry.
    void flush_routing();

    // ---- the two one-shot end-of-run writers (opened at write time, as upstream does) --------
    void mixed(std::span<const float> head_input);                 // n_embd floats, no header
    void residual(std::span<const float> r);                       // {hc,n_embd} header + hc*n_embd

    // Close the per-position files.  Never throws (the destructor calls it): a failed logit-dump
    // close is recorded and reported, which is what upstream's :13997-14000 check does.
    bool close();
    bool closed() const { return closed_; }
    // "cannot finish logits dump" (:13998) when the last flush of the logit file failed.
    const std::string& close_error() const { return close_error_; }
    // Records written per kind, for the summary lines upstream prints (:14004-14018, :14171).
    uint64_t logits_rows_written() const { return logits_rows_; }
    uint64_t layer_records() const { return layer_records_; }
    uint64_t half_records() const { return half_records_; }
    uint64_t final_r_records() const { return final_r_records_; }
    uint64_t routing_records() const { return routing_records_; }

private:
    DumpOptions options_;
    DumpGeometry geometry_;
    std::FILE* logits_ = nullptr;
    std::FILE* layers_ = nullptr;
    std::FILE* halves_ = nullptr;
    std::FILE* final_r_ = nullptr;
    std::FILE* routing_ = nullptr;
    int32_t logits_header_[2] = {0, 0}; // {n_vocab, n_rows}, written with the FIRST row (:6075, :13422)
    bool logits_header_written_ = false;
    bool any_ = false;
    bool closed_ = false;
    std::string close_error_;
    uint64_t logits_rows_ = 0, layer_records_ = 0, half_records_ = 0, final_r_records_ = 0,
             routing_records_ = 0;
};

} // namespace strata::hpu
