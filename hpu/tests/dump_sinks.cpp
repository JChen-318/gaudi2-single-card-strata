// hpu/tests/dump_sinks.cpp - host evidence for the ported per-position diagnostic dumps.
//
// hpu/src/dump_sinks.cpp transcribes upstream's seven --dump-* record layouts from src/program/generate.cpp
// at commit 296b59d; this test checks the BYTES and the REFUSALS against independent expectations, because a
// dump that is silently wrong reads as a wrong answer rather than as a mistake (:5993-6007).
//
// Authorities used here:
//   * the pinned `strata::program::logits_selection::selected/row_count` (include/strata/program/
//     logits_selection.hpp) - the same unmodified header the module uses - so the stride law is checked
//     against upstream's own code rather than a paraphrase;
//   * hand-built expected byte buffers for every layout: int32 {n_vocab,n_rows} + rows; (n_layers+1)*hc*n_embd
//     floats; int64 {pos,tok} + hc*n_embd floats; int32 {layer,k} + k int32 + k float32; n_embd floats with
//     no header; int32 {hc,n_embd} + hc*n_embd floats.
// No device, no SDK header: strata/hpu/dump_sinks.hpp is deliberately standalone.
#include "strata/hpu/dump_sinks.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using strata::hpu::DumpGeometry;
using strata::hpu::DumpOptions;
using strata::hpu::DumpProducers;
using strata::hpu::DumpSinks;

int failures = 0;
void check(const bool ok, const std::string& what) {
    if (!ok) {
        ++failures;
        std::cout << "FAIL " << what << "\n";
    }
}

std::filesystem::path temp_dir() {
    static const auto dir = std::filesystem::temp_directory_path() / "strata-hpu-dump-test";
    std::filesystem::create_directories(dir);
    return dir;
}
std::string path_for(const char* const name) { return (temp_dir() / name).string(); }

std::vector<unsigned char> read_bytes(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    return std::vector<unsigned char>(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
}
template <class T>
void append(std::vector<unsigned char>& out, const T& value) {
    const auto* const bytes = reinterpret_cast<const unsigned char*>(&value);
    out.insert(out.end(), bytes, bytes + sizeof(T));
}
template <class T>
void append_all(std::vector<unsigned char>& out, const std::vector<T>& values) {
    for (const auto& value : values) append(out, value);
}

// The model shape every record is sized against: two layers, three streams, four-wide hidden, tiny vocab.
constexpr int64_t kVocab = 4, kLayers = 2, kStreams = 3, kHidden = 4, kHead = 2, kHeadDim = 2, kExpert = 5;

DumpGeometry geometry(const int64_t positions, const int64_t stride = 1, const uint32_t max_new = 1) {
    DumpGeometry g;
    g.n_vocab = kVocab;
    g.n_layers = kLayers;
    g.hc = kStreams;
    g.n_embd = kHidden;
    g.n_head = kHead;
    g.head_dim = kHeadDim;
    g.n_head_kv = 1;
    g.n_expert = kExpert;
    g.positions = positions;
    g.stride = stride;
    g.max_new = max_new;
    return g;
}

DumpProducers producers() {
    DumpProducers p;
    p.logits = true;
    p.layer_ladder = true;
    p.half_capture = false;   // no per-layer half capture is wired in this backend
    p.routing = true;
    p.final_r = true;
    p.head_residual = true;
    p.head_mixed = true;
    return p;
}

bool refused(const std::function<void()>& call, const std::string& needle) {
    try {
        call();
    } catch (const std::invalid_argument& error) {
        return std::string(error.what()).find(needle) != std::string::npos;
    }
    return false;
}

// ---- 1. off means off -------------------------------------------------------------------------------
void off_is_free() {
    DumpSinks sinks{DumpOptions{}, geometry(5), producers()};
    check(!sinks.active(), "an empty DumpOptions is inactive");
    check(!sinks.wants_logits() && !sinks.wants_layers() && !sinks.wants_halves() && !sinks.wants_mixed() &&
              !sinks.wants_residual() && !sinks.wants_final_r() && !sinks.wants_routing(),
          "every writer reports off");
    // Every writer must return without touching a file that was never opened.
    const std::vector<float> row(kVocab, 0.0f);
    sinks.logits(0, row);
    sinks.layers(std::vector<float>(size_t(kLayers + 1) * kStreams * kHidden, 0.0f));
    sinks.final_residual(0, 0, std::vector<float>(size_t(kStreams) * kHidden, 0.0f));
    sinks.mixed(std::vector<float>(kHidden, 0.0f));
    sinks.residual(std::vector<float>(size_t(kStreams) * kHidden, 0.0f));
    check(sinks.close(), "closing an inactive sink succeeds");
    check(sinks.logits_rows_written() == 0 && sinks.layer_records() == 0 && sinks.final_r_records() == 0 &&
              sinks.routing_records() == 0,
          "nothing was recorded");
}

// ---- 2. a dump whose producer never runs is refused ---------------------------------------------------
void producers_are_required() {
    const auto base = geometry(4);
    {
        DumpOptions o;
        o.halves = path_for("halves.bin");
        check(refused([&] { DumpSinks s{o, base, producers()}; }, "CAPTURED"), "--dump-halves needs the captured path");
    }
    DumpProducers no_ladder = producers();
    no_ladder.layer_ladder = false;
    {
        DumpOptions o;
        o.layers = path_for("layers-refused.bin");
        check(refused([&] { DumpSinks s{o, base, no_ladder}; }, "staging buffer"),
              "--dump-layers needs the layer loop");
    }
    DumpProducers no_routing = producers();
    no_routing.routing = false;
    {
        DumpOptions o;
        o.routing = path_for("routing-refused.bin");
        check(refused([&] { DumpSinks s{o, base, no_routing}; }, "routing"), "--dump-routing needs the router hook");
    }
    DumpProducers no_final = producers();
    no_final.final_r = false;
    {
        DumpOptions o;
        o.final_r = path_for("final-refused.bin");
        check(refused([&] { DumpSinks s{o, base, no_final}; }, "post-last-layer residual"),
              "--dump-final-r needs its producer");
    }
    DumpProducers no_mixed = producers();
    no_mixed.head_mixed = false;
    {
        DumpOptions o;
        o.mixed = path_for("mixed-refused.bin");
        check(refused([&] { DumpSinks s{o, base, no_mixed}; }, "mixed"), "--dump-mixed needs its producer");
    }
    DumpProducers no_head = producers();
    no_head.head_residual = false;
    {
        DumpOptions o;
        o.residual = path_for("residual-refused.bin");
        check(refused([&] { DumpSinks s{o, base, no_head}; }, "residual"), "--dump-residual needs its producer");
    }
}

// ---- 3. geometry and the stride law ------------------------------------------------------------------
void stride_and_geometry() {
    {
        DumpOptions o;
        o.logits = path_for("stride-refused.bin");
        check(refused([&] { DumpSinks s{o, geometry(6, 2, 2), producers()}; }, "stride"),
              "a stride > 1 with max_new != 1 is refused (:2631-2633)");
    }
    {
        DumpOptions o;
        o.logits = path_for("vocab-refused.bin");
        auto g = geometry(6);
        g.n_vocab = 0;
        check(refused([&] { DumpSinks s{o, g, producers()}; }, "logits dump dimensions"),
              "a zero vocabulary cannot size a logits row");
    }

    // stride 1: one row per position, header {n_vocab, n_rows} with n_rows == row_count(positions,1).
    const int64_t positions = 5;
    const std::string path = path_for("logits.bin");
    {
        DumpOptions o;
        o.logits = path;
        DumpSinks sinks{o, geometry(positions), producers()};
        std::vector<std::vector<float>> rows;
        for (int64_t p = 0; p < positions; ++p) {
            std::vector<float> row(kVocab);
            for (int64_t v = 0; v < kVocab; ++v) row[size_t(v)] = float(p * 10 + v) + 0.5f;
            rows.push_back(row);
            sinks.logits(p, row);
        }
        // A position past the count writes NOTHING (:13405-13406's selection). The file must not grow.
        sinks.logits(positions, std::vector<float>(kVocab, 7.0f));
        check(sinks.close(), "logits dump closes cleanly");
    }
    std::vector<unsigned char> expected;
    append(expected, int32_t(kVocab));
    append(expected, int32_t(strata::program::logits_selection::row_count(positions, 1)));
    for (int64_t p = 0; p < positions; ++p)
        for (int64_t v = 0; v < kVocab; ++v) append(expected, float(p * 10 + v) + 0.5f);
    check(read_bytes(path) == expected, "logits bytes are {n_vocab,n_rows} then one float32 row per position");

    // stride 3 with max_new 1: upstream's zero-based multiples of the stride plus the final position.
    const int64_t strided_positions = 10;
    const std::string strided = path_for("logits-stride3.bin");
    {
        DumpOptions o;
        o.logits = strided;
        DumpSinks sinks{o, geometry(strided_positions, 3, 1), producers()};
        for (int64_t p = 0; p < strided_positions; ++p) sinks.logits(p, std::vector<float>(kVocab, float(p)));
        check(sinks.logits_rows_written() == uint64_t(strata::program::logits_selection::row_count(strided_positions, 3)),
              "the stride law's row count matches upstream's row_count");
        check(sinks.close(), "strided dump closes cleanly");
    }
    std::vector<unsigned char> strided_expected;
    append(strided_expected, int32_t(kVocab));
    append(strided_expected, int32_t(strata::program::logits_selection::row_count(strided_positions, 3)));
    for (int64_t p = 0; p < strided_positions; ++p)
        if (strata::program::logits_selection::selected(p, strided_positions, 3))
            for (int64_t v = 0; v < kVocab; ++v) append(strided_expected, float(p));
    check(read_bytes(strided) == strided_expected, "the strided rows are exactly upstream's selected positions");
}

// ---- 4. the ladder, final-r, routing and the two one-shot records -------------------------------------
void record_layouts() {
    const uint64_t ladder = uint64_t(kLayers + 1) * uint64_t(kStreams) * uint64_t(kHidden);
    const std::string layers_path = path_for("layers.bin");
    const std::string final_path = path_for("final-r.bin");
    const std::string routing_path = path_for("routing.bin");
    {
        DumpOptions o;
        o.layers = layers_path;
        o.final_r = final_path;
        o.routing = routing_path;
        DumpSinks sinks{o, geometry(3), producers()};
        for (int64_t p = 0; p < 2; ++p) {
            std::vector<float> stage(ladder);
            for (size_t i = 0; i < stage.size(); ++i) stage[i] = float(p) + float(i) * 0.25f;
            sinks.layers(stage);
            std::vector<float> residual(size_t(kStreams) * kHidden);
            for (size_t i = 0; i < residual.size(); ++i) residual[i] = float(p) - float(i);
            sinks.final_residual(p, 100 + p, residual);
        }
        // A wrong-size ladder is a geometry mismatch, not a partial record.
        check(refused([&] { sinks.layers(std::vector<float>(ladder - 1)); }, "layer ladder record"),
              "a short ladder is refused");
        // Routing: a record per layer; a layer outside 0..n_layers-1 is reported and NOT written (:1014-1018).
        const std::vector<int32_t> ids{1, 3};
        const std::vector<float> weights{0.75f, 0.25f};
        sinks.routing(0, 2, ids.data(), weights.data());
        sinks.routing(int32_t(kLayers), 2, ids.data(), weights.data());
        check(sinks.routing_records() == 1, "an out-of-range routing layer is skipped, not written");
        // The window form carries tokens*k ids, one record per token, weights written as zeros.
        const std::vector<int32_t> window_ids{1, 3, 2, 4};
        sinks.routing_window(1, 2, 2, window_ids.data());
        check(sinks.routing_records() == 3, "the window form writes one record per token");
        check(sinks.close(), "the three files close");
    }
    std::vector<unsigned char> expected_layers;
    for (int64_t p = 0; p < 2; ++p)
        for (size_t i = 0; i < ladder; ++i) append(expected_layers, float(p) + float(i) * 0.25f);
    check(read_bytes(layers_path) == expected_layers, "the ladder is one (n_layers+1)*hc*n_embd block per position");

    std::vector<unsigned char> expected_final;
    for (int64_t p = 0; p < 2; ++p) {
        append(expected_final, int64_t(p));
        append(expected_final, int64_t(100 + p));
        for (size_t i = 0; i < size_t(kStreams) * kHidden; ++i) append(expected_final, float(p) - float(i));
    }
    check(read_bytes(final_path) == expected_final, "final-r is int64 {pos,tok} plus hc*n_embd float32");

    std::vector<unsigned char> expected_routing;
    append(expected_routing, int32_t(0));
    append(expected_routing, int32_t(2));
    for (auto id : std::vector<int32_t>{1, 3}) append(expected_routing, id);
    for (auto w : std::vector<float>{0.75f, 0.25f}) append(expected_routing, w);
    // The window form: one record per token, zeros where the fused dispatch cannot surface weights.
    for (int64_t t = 0; t < 2; ++t) {
        append(expected_routing, int32_t(1));
        append(expected_routing, int32_t(2));
        for (auto id : std::vector<int32_t>{t == 0 ? 1 : 2, t == 0 ? 3 : 4}) append(expected_routing, id);
        for (int64_t k = 0; k < 2; ++k) append(expected_routing, 0.0f);
    }
    check(read_bytes(routing_path) == expected_routing, "routing records are int32 {layer,k} + ids + weights");

    // ---- the two one-shot end-of-run records: opened at WRITE time, so an unwritten one leaves no file.
    const std::string mixed_path = path_for("mixed.bin");
    const std::string residual_path = path_for("residual.bin");
    std::filesystem::remove(mixed_path);
    std::filesystem::remove(residual_path);
    {
        DumpOptions o;
        o.mixed = mixed_path;
        o.residual = residual_path;
        DumpSinks sinks{o, geometry(3), producers()};
        check(!std::filesystem::exists(mixed_path) && !std::filesystem::exists(residual_path),
              "a one-shot dump creates no file until the run ends");
        std::vector<float> mixed(kHidden);
        for (int64_t i = 0; i < kHidden; ++i) mixed[size_t(i)] = 1.5f + float(i);
        sinks.mixed(mixed);
        std::vector<float> residual(size_t(kStreams) * kHidden);
        for (size_t i = 0; i < residual.size(); ++i) residual[i] = -float(i) * 0.5f;
        sinks.residual(residual);
        check(refused([&] { sinks.mixed(std::vector<float>(kHidden - 1)); }, "mixed dump"), "a short mixed row is refused");
        check(refused([&] { sinks.residual(std::vector<float>(1)); }, "residual dump"), "a short residual is refused");
    }
    std::vector<unsigned char> expected_mixed;
    for (int64_t i = 0; i < kHidden; ++i) append(expected_mixed, 1.5f + float(i));
    check(read_bytes(mixed_path) == expected_mixed, "--dump-mixed is n_embd float32 and no header");
    std::vector<unsigned char> expected_residual;
    append(expected_residual, int32_t(kStreams));
    append(expected_residual, int32_t(kHidden));
    for (size_t i = 0; i < size_t(kStreams) * kHidden; ++i) append(expected_residual, -float(i) * 0.5f);
    check(read_bytes(residual_path) == expected_residual, "--dump-residual is int32 {hc,n_embd} then the residual");
}
} // namespace

int main() {
    off_is_free();
    producers_are_required();
    stride_and_geometry();
    record_layouts();
    if (failures != 0) {
        std::cout << "FAILED dump sink checks: " << failures << "\n";
        return 1;
    }
    std::cout << "PASS dump sinks: the seven --dump-* record layouts are byte-checked (logits header and stride "
                 "law against upstream's own logits_selection, the (n_layers+1) ladder, final-r's int64 {pos,tok}, "
                 "routing's ids+weights and its bounds rule, the one-shot mixed/residual forms), every missing "
                 "producer is REFUSED with its reason, and an empty DumpOptions opens nothing; no device\n";
    std::filesystem::remove_all(temp_dir());
    return 0;
}
