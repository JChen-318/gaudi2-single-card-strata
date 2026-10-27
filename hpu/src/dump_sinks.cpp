// Implementation of the ported diagnostic dumps.  Every byte layout, record order and refusal here
// is transcribed from src/program/generate.cpp at commit 296b59d; the line numbers in the comments
// are that file's.  Nothing in this translation unit touches the device or strata/hpu/runtime.hpp.
#include "strata/hpu/dump_sinks.hpp"
#include "strata/hpu/session_flow.hpp"

#include <cstddef>
#include <array>
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace strata::hpu {
namespace {

constexpr int64_t kInt32Max = 2147483647;

// Upstream's refusal texts, without the "strata generate: " the tool itself prefixes (:2632,
// :5930, :5998, :6004, :6080 and the `cannot write %s` family at :3260, :5935, :6085, :6109,
// :14043, :14063).
[[noreturn]] void refuse(const std::string& message) { throw std::invalid_argument(message); }

std::string cannot_write(const std::string& path) { return "cannot write " + path; }

// A product of the model shape has to fit the record it sizes; upstream trusts its own buffers,
// the port validates a value handed in by a hook.
bool product_fits(uint64_t a, uint64_t b, uint64_t limit) {
    return a == 0 || b == 0 || a <= limit / b;
}

std::FILE* open_binary_or_throw(const std::string& path) {
    std::FILE* f = path.empty() ? nullptr : std::fopen(path.c_str(), "wb");
    if (f == nullptr) refuse(cannot_write(path));
    return f;
}

// An unbuffered-looking writer still has to be checked: a short write mid-record would desynchronise
// every following record, which is the one thing a diffable dump may not do.
void write_all(std::FILE* f, const void* data, size_t count, const std::string& message) {
    if (std::fwrite(data, count, 1, f) != 1) refuse(message);
}

} // namespace

DumpSinks::DumpSinks(const DumpOptions& options, const DumpGeometry& geometry,
                     const DumpProducers& producers)
    : options_(options), geometry_(geometry) {
    any_ = !options_.none();
    if (!any_) return; // off means off: no validation, no open, no allocation, no cost

    // ---- geometry that cannot match the model shape is refused before a single byte is written.
    if (geometry_.n_vocab < 0 || geometry_.n_layers < 0 || geometry_.hc < 0 || geometry_.n_embd < 0 ||
        geometry_.n_head < 0 || geometry_.head_dim < 0 || geometry_.n_head_kv < 0 ||
        geometry_.n_expert < 0 || geometry_.positions < 0 || geometry_.stride < 1 ||
        geometry_.max_new < 1)
        refuse("dump geometry requires non-negative dimensions, a positive logits stride and max_new >= 1");
    if (!product_fits(uint64_t(geometry_.hc), uint64_t(geometry_.n_embd), kInt32Max))
        refuse("dump geometry hc*n_embd exceeds int32");

    // Upstream opens the per-position files in this order: halves (:3257, armed before capture),
    // routing (:5933), logits (:6083), layers (:6107), final_r (:13162).  Keeping it means a run
    // with two bad paths refuses the one upstream refuses, and the files it created match.
    if (!options_.halves.empty()) {
        if (!producers.half_capture)
            refuse("--dump-halves is CAPTURED into the layer graphs, so it needs the captured path; "
                   "this session never records it. Drop one of the two.");
        if (geometry_.n_layers < 1 || geometry_.half_stride() == 0 ||
            !product_fits(uint64_t(geometry_.n_layers), geometry_.half_stride(), kInt32Max))
            refuse("--dump-halves needs a model with layers and a positive half_stride that fits int32");
        halves_ = open_binary_or_throw(options_.halves);
    }
    if (!options_.routing.empty()) {
        if (!producers.routing)
            refuse("--dump-routing needs the expert pool; the routed ids reach the host through the "
                   "router. No router hook is wired into this session. Drop one of the two.");
        if (geometry_.n_layers < 1 || geometry_.n_expert < 1)
            refuse("--dump-routing needs a MoE model: n_layers and n_expert must both be positive");
        routing_ = open_binary_or_throw(options_.routing);
    }
    if (!options_.logits.empty()) {
        if (!producers.logits) refuse("--dump-logits is written by the token path; this session has no "
                                      "logit producer for it.");
        // :2631-2633's law, kept because a stride with no dump is a silent no-op upstream refuses.
        if (geometry_.stride > 1 && geometry_.max_new != 1)
            refuse("--logits-stride > 1 requires --max-new 1 and --dump-logits");
        if (geometry_.positions > kInt32Max || geometry_.n_vocab > kInt32Max || geometry_.n_vocab < 1)
            refuse("logits dump dimensions exceed int32"); // :6080
        logits_ = open_binary_or_throw(options_.logits);
        logits_header_[0] = int32_t(geometry_.n_vocab); // {n_vocab, n_rows}, deferred to row 0 (:6075, :13422)
        logits_header_[1] = int32_t(geometry_.logits_rows());
    }
    if (!options_.layers.empty()) {
        if (!producers.layer_ladder)
            refuse("--dump-layers is written by the layer loop; this session runs a path that never "
                   "fills the staging buffer. Drop one of the two.");
        if (geometry_.n_layers < 1 || geometry_.hc < 1 || geometry_.n_embd < 1 ||
            !product_fits(uint64_t(geometry_.n_layers + 1), uint64_t(geometry_.hc * geometry_.n_embd),
                          uint64_t(0x7FFFFFFFFFFFFFFF) / 4))
            refuse("--dump-layers needs n_layers, hc and n_embd to be positive");
        layers_ = open_binary_or_throw(options_.layers);
    }
    if (!options_.final_r.empty()) {
        if (!producers.final_r)
            refuse("--dump-final-r is appended after every processed position on the token path; this "
                   "session never publishes the post-last-layer residual.");
        if (geometry_.hc < 1 || geometry_.n_embd < 1) refuse("--dump-final-r needs hc and n_embd positive");
        final_r_ = open_binary_or_throw(options_.final_r);
    }
    // --dump-mixed and --dump-residual are ONE-SHOT: upstream reads the value back and opens the
    // file at the END of the run (:14034-14052, :14055-14078), so a run that never gets there leaves
    // no file.  Only the producer/shape law is checked here.
    if (!options_.mixed.empty()) {
        if (!producers.head_mixed)
            refuse("--dump-mixed is the head input read at the end of the run; this session publishes no "
                   "readable mixed state.");
        if (geometry_.n_embd < 1) refuse("--dump-mixed needs n_embd positive");
    }
    if (!options_.residual.empty()) {
        if (!producers.head_residual)
            refuse("--dump-residual is the final R read at the end of the run; this session publishes no "
                   "readable residual state.");
        if (geometry_.hc < 1 || geometry_.n_embd < 1) refuse("--dump-residual needs hc and n_embd positive");
    }
}

bool DumpSinks::routing_session_matches(const DumpOptions& options,const DumpGeometry& shape) const {
    return !closed_ && routing_ && options_.routing_only() && options.routing_only() && options.routing==options_.routing &&
        shape.n_layers==geometry_.n_layers && shape.n_expert==geometry_.n_expert;
}
void DumpSinks::routing_view(const RoutingWindowView& view,bool window) {
    const auto cells=uint64_t(view.rows)*view.top_k;
    if(!routing_ || closed_ || !view.layers || view.layers>4096 || int64_t(view.layers)!=geometry_.n_layers ||
       !view.rows || view.rows>128 || !view.top_k || view.top_k>64 || int64_t(view.top_k)>geometry_.n_expert ||
       view.layer_stride<cells || view.ids.size()!=view.weights.size() ||
       view.layer_stride>UINT64_MAX/view.layers || uint64_t(view.layers-1)*view.layer_stride+cells>view.ids.size())
        throw SessionFatal("routing window producer violated bounded geometry contract");
    for(uint32_t layer=0;layer<view.layers;++layer) for(uint64_t at=0;at<cells;++at) {
        const auto offset=uint64_t(layer)*view.layer_stride+at;
        if(view.ids[offset]<0 || int64_t(view.ids[offset])>=geometry_.n_expert || !std::isfinite(view.weights[offset]) || view.weights[offset]<0)
            throw SessionFatal("routing window producer returned invalid expert/weight");
    }
    for(uint32_t layer=0;layer<view.layers;++layer) {
        const auto offset=uint64_t(layer)*view.layer_stride;
        if(window) routing_window(int32_t(layer),view.top_k,view.rows,view.ids.data()+offset);
        else for(uint32_t row=0;row<view.rows;++row) routing(int32_t(layer),view.top_k,view.ids.data()+offset+uint64_t(row)*view.top_k,view.weights.data()+offset+uint64_t(row)*view.top_k);
    }
    flush_routing();
}

DumpSinks::DumpSinks(DumpSinks&& other) noexcept
    : options_(std::move(other.options_)), geometry_(other.geometry_), logits_(other.logits_),
      layers_(other.layers_), halves_(other.halves_), final_r_(other.final_r_),
      routing_(other.routing_), logits_header_written_(other.logits_header_written_),
      any_(other.any_), closed_(other.closed_), close_error_(std::move(other.close_error_)),
      logits_rows_(other.logits_rows_),
      layer_records_(other.layer_records_), half_records_(other.half_records_),
      final_r_records_(other.final_r_records_), routing_records_(other.routing_records_) {
    other.logits_ = other.layers_ = other.halves_ = other.final_r_ = other.routing_ = nullptr;
    other.any_ = false;
}

DumpSinks& DumpSinks::operator=(DumpSinks&& other) noexcept {
    if (this != &other) {
        close();
        options_ = std::move(other.options_);
        geometry_ = other.geometry_;
        logits_ = other.logits_;
        layers_ = other.layers_;
        halves_ = other.halves_;
        final_r_ = other.final_r_;
        routing_ = other.routing_;
        logits_header_[0] = other.logits_header_[0];
        logits_header_[1] = other.logits_header_[1];
        logits_header_written_ = other.logits_header_written_;
        any_ = other.any_;
        closed_ = other.closed_;
        close_error_ = std::move(other.close_error_);
        logits_rows_ = other.logits_rows_;
        layer_records_ = other.layer_records_;
        half_records_ = other.half_records_;
        final_r_records_ = other.final_r_records_;
        routing_records_ = other.routing_records_;
        other.logits_ = other.layers_ = other.halves_ = other.final_r_ = other.routing_ = nullptr;
        other.any_ = false;
    }
    return *this;
}

DumpSinks::~DumpSinks() { close(); }

void DumpSinks::logits(const int64_t position, const std::span<const float> row) {
    if (logits_ == nullptr) return; // zero cost when off
    if (row.size() != size_t(geometry_.n_vocab))
        refuse("logits row of " + std::to_string(row.size()) + " floats does not match the model's n_vocab " +
               std::to_string(geometry_.n_vocab));
    // :13405-13406: storage selection only; a row outside the selection writes nothing at all.
    if (!strata::program::logits_selection::selected(position, geometry_.positions, geometry_.stride)) return;
    // The header rides the FIRST row (:13421-13427): a run that dumps nothing leaves an empty file,
    // and a header promising rows that were never written is worse than no file (:6073-6074).
    if (!logits_header_written_) {
        write_all(logits_, logits_header_, sizeof logits_header_, "cannot write logits header"); // :14023
        logits_header_written_ = true;
    }
    write_all(logits_, row.data(), row.size() * sizeof(float),
              "cannot write logits at position " + std::to_string(position)); // :14429
    ++logits_rows_;
}

void DumpSinks::layers(const std::span<const float> stage) {
    if (layers_ == nullptr) return;
    // One snapshot per layer per position, layer 0 first, slot 0 the INPUT residual (:13381-13382,
    // src/core/session.cpp:660-672).  A short buffer is a geometry mismatch, not a partial record.
    if (stage.size() != size_t(geometry_.layer_floats()))
        refuse("layer ladder record of " + std::to_string(stage.size()) + " floats does not match (n_layers+1)*hc*n_embd " +
               std::to_string(geometry_.layer_floats()));
    write_all(layers_, stage.data(), stage.size() * sizeof(float), "cannot write layer ladder");
    ++layer_records_;
}

void DumpSinks::halves(const std::span<const float> stage) {
    if (halves_ == nullptr) return;
    const uint64_t expected = uint64_t(geometry_.n_layers) * geometry_.half_stride();
    if (stage.size() != size_t(expected))
        refuse("half record of " + std::to_string(stage.size()) + " floats does not match n_layers*half_stride " +
               std::to_string(expected));
    write_all(halves_, stage.data(), stage.size() * sizeof(float), "cannot write half dump"); // :13384
    if(std::fflush(halves_)!=0) refuse("cannot finish half record");
    ++half_records_;
}

void DumpSinks::final_residual(const int64_t position, const int64_t token, const std::span<const float> r) {
    if (final_r_ == nullptr) return;
    if (r.size() != size_t(geometry_.final_r_floats()))
        refuse("final-r record of " + std::to_string(r.size()) + " floats does not match hc*n_embd " +
               std::to_string(geometry_.final_r_floats()));
    // int64 {pos, tok} then hc*n_embd float32, appended after each processed POSITION, token path
    // only (:13363-13368; the flag's own note at :555-556).
    const int64_t posrec[2] = {position, token};
    write_all(final_r_, posrec, sizeof posrec, "cannot write final-r position record");
    write_all(final_r_, r.data(), r.size() * sizeof(float), "cannot write final-r residual");
    ++final_r_records_;
}

void DumpSinks::routing(const int32_t layer, const int64_t k, const int32_t* const ids,
                        const float* const weights) {
    if (routing_ == nullptr) return;
    // Upstream bounds the layer at 0..47 - its n_layers - reports and writes NOTHING for the record
    // (:1014-1018).  The same law against the port's geometry, and the same "skip, do not fail".
    if (layer < 0 || layer >= int32_t(geometry_.n_layers)) {
        std::fprintf(stderr, "strata generate: the routing trace saw layer %d, outside 0..%lld\n", layer,
                     static_cast<long long>(geometry_.n_layers - 1));
        return;
    }
    if (k < 1 || k > geometry_.n_expert || ids == nullptr || weights == nullptr) {
        refuse("routing record: k must be in 1..n_expert and ids/weights must be readable");
    }
    const int32_t rec[2] = {layer, int32_t(k)}; // int32 layer, int32 k, k int32 ids, k float weights
    write_all(routing_, rec, sizeof rec, "cannot write routing record");
    write_all(routing_, ids, size_t(k)*sizeof(int32_t), "cannot write routing ids");
    write_all(routing_, weights, size_t(k)*sizeof(float), "cannot write routing weights");
    ++routing_records_;
}

void DumpSinks::routing_window(const int32_t layer, const int64_t k, const int64_t tokens,
                               const int32_t* const ids) {
    if (routing_ == nullptr) return;
    if (layer < 0 || layer >= int32_t(geometry_.n_layers)) return; // :1039's silent bound
    if (k < 1 || k > 64 || k > geometry_.n_expert || tokens < 1 || tokens > 128 || ids == nullptr)
        refuse("routing window record: k, tokens and ids must be readable");
    // The verify-window form: the same record, ONE PER TOKEN, weights ZEROS because the fused
    // dispatch does not surface them - make_profile.py reads zeros as unit weights (:1035-1045).
    const std::array<float,64> zero{};
    for (int64_t tok = 0; tok < tokens; ++tok) {
        const int32_t rec[2] = {layer, int32_t(k)};
        write_all(routing_, rec, sizeof rec, "cannot write routing record");
        write_all(routing_, ids + tok * k, size_t(k)*sizeof(int32_t), "cannot write routing window ids");
        write_all(routing_, zero.data(), size_t(k)*sizeof(float), "cannot write routing window weights");
        ++routing_records_;
    }
}

void DumpSinks::flush_routing() {
    if(routing_ && std::fflush(routing_)!=0) refuse("cannot finish routing window");
}

void DumpSinks::mixed(const std::span<const float> head_input) {
    if (options_.mixed.empty()) return;
    if (head_input.size() != size_t(geometry_.n_embd))
        refuse("mixed dump of " + std::to_string(head_input.size()) + " floats does not match n_embd " +
               std::to_string(geometry_.n_embd));
    std::FILE* mf = open_binary_or_throw(options_.mixed); // :14041-14044, opened at the END
    write_all(mf, head_input.data(), head_input.size() * sizeof(float), cannot_write(options_.mixed));
    std::fclose(mf);
}

void DumpSinks::residual(const std::span<const float> r) {
    if (options_.residual.empty()) return;
    if (r.size() != size_t(geometry_.final_r_floats()))
        refuse("residual dump of " + std::to_string(r.size()) + " floats does not match hc*n_embd " +
               std::to_string(geometry_.final_r_floats()));
    std::FILE* rf = open_binary_or_throw(options_.residual); // :14061-14064
    const int32_t hdr[2] = {int32_t(geometry_.hc), int32_t(geometry_.n_embd)}; // :14066
    write_all(rf, hdr, sizeof hdr, cannot_write(options_.residual));
    write_all(rf, r.data(), r.size() * sizeof(float), cannot_write(options_.residual));
    std::fclose(rf);
}

bool DumpSinks::close() {
    if (closed_) return close_error_.empty();
    closed_ = true;
    // Only the logit dump's close is checked upstream (:13997-14000 "cannot finish logits dump");
    // the rest are closed without a check (:14002, :14008, :14014).  Recorded rather than thrown,
    // because the destructor calls this and a throwing destructor is a terminate.
    if (logits_ != nullptr && std::fclose(logits_) != 0 && close_error_.empty())
        close_error_ = "cannot finish logits dump";
    logits_ = nullptr;
    if (layers_ != nullptr) std::fclose(layers_);
    layers_ = nullptr;
    if (halves_ != nullptr && std::fclose(halves_) != 0 && close_error_.empty()) close_error_="cannot finish half dump";
    halves_ = nullptr;
    if (final_r_ != nullptr) std::fclose(final_r_); // upstream never closes it (:13162-13367): the port does
    final_r_ = nullptr;
    // Declared IO refinement: routing callbacks now promise that their bytes
    // were accepted, so an ignored final close failure would be misleading.
    if (routing_ != nullptr && std::fclose(routing_) != 0 && close_error_.empty()) close_error_="cannot finish routing dump";
    routing_ = nullptr;
    return close_error_.empty();
}

} // namespace strata::hpu
