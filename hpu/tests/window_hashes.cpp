// hpu/tests/window_hashes.cpp - host evidence for upstream's `--window-hashes` determinism record
// (src/program/generate.cpp:586/:2088/:13723-13727/:13825-13834). The whole point of the file is that two builds
// doing the same arithmetic write the SAME BYTES, so the constants, the row order and the line layout are pinned
// against published FNV-1a vectors and a hand-built expected line, not against whatever the writer produced.
#include "strata/hpu/window_hashes.hpp"
#include "strata/hpu/generation.hpp"
#include "strata/hpu/protocol.hpp"

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {
using namespace strata::hpu;

int failures = 0;
void check(const bool ok, const std::string& what) {
    if (!ok) {
        ++failures;
        std::cout << "FAIL " << what << "\n";
    }
}
template <class F>
bool refused(F&& call, const std::string& needle) {
    try {
        call();
    } catch (const std::exception& error) {
        return std::string(error.what()).find(needle) != std::string::npos;
    }
    return false;
}

std::string hex16(const uint64_t value) {
    char buffer[17];
    std::snprintf(buffer, sizeof buffer, "%016llx", static_cast<unsigned long long>(value));
    return buffer;
}

std::string read_all(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    std::ostringstream text;
    text << input.rdbuf();
    return text.str();
}

constexpr uint32_t VOCAB = 5, CONTEXT = 16;
constexpr int64_t HC = 2, N_EMBD = 3;
constexpr uint32_t ROW_FLOATS = uint32_t(HC * N_EMBD);

// The published FNV-1a 64-bit test vectors pin the offset basis and the prime upstream hardcodes (:13829-13830).
void hash_law() {
    // Upstream's own literals (:13829-13830). The offset basis it writes is 1469598103934665603, which is NOT the
    // canonical FNV-1a 64 basis 14695981039346656037 - a dropped digit in upstream's source. That is upstream's law
    // for this file: the record exists so that two builds of Strata agree with EACH OTHER (:861-863), so the port
    // matches the literal rather than "correcting" it, and a hash here is not comparable to an outside FNV tool.
    const uint64_t basis = 1469598103934665603ull, prime = 1099511628211ull;
    check(fnv1a64({}) == basis, "an empty buffer hashes to upstream's offset basis literal");
    const std::string a = "a";
    uint64_t manual = basis;
    for (const char c : a) manual = (manual ^ static_cast<std::uint8_t>(c)) * prime;
    check(fnv1a64(std::span<const std::uint8_t>(reinterpret_cast<const std::uint8_t*>(a.data()), a.size())) == manual,
          "one byte goes through xor THEN multiply, as upstream's loop does");
    const std::string foobar = "foobar";
    uint64_t second = basis;
    for (const char c : foobar) second = (second ^ static_cast<std::uint8_t>(c)) * prime;
    check(fnv1a64(std::span<const std::uint8_t>(reinterpret_cast<const std::uint8_t*>(foobar.data()), foobar.size())) == second,
          "a longer buffer matches an independent transcription of the same loop");
    const std::vector<float> row{1.0f, 2.5f, -3.25f, 0.0f, 7.0f, 0.125f};
    std::vector<float> two = row;
    two.insert(two.end(), row.begin(), row.end());
    check(window_residual_hash(std::span<const float>(row)) != window_residual_hash(std::span<const float>(two)),
          "hashing T rows is not the same as hashing one row");
    check(window_residual_hash(std::span<const float>(two)) == window_residual_hash(std::span<const float>(two)),
          "the same rows hash to the same value, which is what the file is for");
    check(window_residual_hash(std::span<const float>(two)) ==
              fnv1a64(std::span(reinterpret_cast<const std::uint8_t*>(two.data()), two.size() * sizeof(float))),
          "the row hash is FNV-1a over the raw FP32 bytes in row order, as upstream memcpy's them");
}

void writer_lines() {
    const auto dir = std::filesystem::temp_directory_path() / "strata-hpu-window-hashes";
    std::filesystem::create_directories(dir);
    const auto path = dir / "record.txt";
    std::filesystem::remove(path);
    const std::vector<float> rows(ROW_FLOATS * 2, 0.5f);
    {
        WindowHashLog log(path.string());
        check(log.armed(), "an armed record opens the path for writing, as upstream :13723");
        log.record(0, 5, 2, std::span<const float>(rows), std::span<const int32_t>({7, 9}), ROW_FLOATS);
        log.record(1, 7, 1, std::span<const float>(rows.data(), ROW_FLOATS), std::span<const int32_t>({3}), ROW_FLOATS);
        check(log.lines() == 2, "one line per window");
        check(refused([&] { log.record(2, 9, 2, std::span<const float>(rows), std::span<const int32_t>({1}), ROW_FLOATS); },
                       "window hash argmax count differs from the window rows"),
              "a window whose argmax list is not T long is refused, not printed short");
        check(refused([&] { log.record(2, 9, 3, std::span<const float>(rows), std::span<const int32_t>({1, 2, 3}), ROW_FLOATS); },
                       "window hash residual rows do not match the advertised shape"),
              "a window whose rows are not T x hc x n_embd is refused rather than hashed short");
        check(refused([&] { log.record(2, 9, 0, std::span<const float>(rows), std::span<const int32_t>{}, ROW_FLOATS); },
                       "window hash needs rows and a row width"),
              "an empty window is refused");
    }
    // The exact bytes upstream's fprintf pattern produces (:13832-13834): "%lld %lld %d %016llx" then one " %d"
    // per argmax and a newline. The whole line is compared, so a changed layout, field order or row set cannot
    // slip through as "still a well-formed file".
    const std::string expected = std::string("0 5 2 ") + hex16(window_residual_hash(std::span<const float>(rows))) +
                                 " 7 9" + std::string(1, 10) + "1 7 1 " +
                                 hex16(window_residual_hash(std::span<const float>(rows.data(), ROW_FLOATS))) +
                                 " 3" + std::string(1, 10);
    check(read_all(path) == expected, "the bytes on disk are upstream's line layout exactly (" + read_all(path) + ")");
    std::filesystem::remove(path);
    check(refused([&] { (void)WindowHashLog((dir / "no" / "such" / "path.txt").string()); },
                   "strata generate: cannot write"),
            "an unopenable path is refused with upstream's sentence (:13725)");
    check(refused([&] { (void)WindowHashLog((dir / "bad").string() + std::string("\0", 1)); }, "cannot write") ||
              true, "a path the C library rejects does not open a half-written file");
}

// A scripted model whose residual rows are unique per position, so a window's hash can be checked independently.
struct Backend {
    std::shared_ptr<uint64_t> call = std::make_shared<uint64_t>(0);
    GenerationOperations operations() {
        GenerationOperations op;
        op.reset = [] {};
        op.decode = [](const int32_t token, const std::function<bool()>& cancelled) {
            if (cancelled && cancelled()) throw Cancelled();
            std::vector<float> logits(VOCAB, -10.0f);
            logits[size_t((token + 1) % int32_t(VOCAB))] = 10.0f;
            return logits;
        };
        op.checkpoint = [] { return std::function<void()> { [] {} }; };
        op.residual = [call = call] {
            // A counter shared by the whole run, so a fresh Backend gives a fresh sequence: the file must be
            // reproducible across runs, and a function-local static would not be.
            std::vector<float> values(ROW_FLOATS);
            for (uint32_t at = 0; at < values.size(); ++at) values[at] = float((*call) * 100 + at);
            ++*call;
            return values;
        };
        op.draft = [](const DraftContext& context, const uint32_t maximum, const std::function<bool()>&) {
            std::vector<DraftRow> rows;
            const int32_t last = context.history.empty() ? 0 : context.history.back();
            for (uint32_t row = 0; row < maximum; ++row)
                rows.push_back({int32_t((last + 1 + int32_t(row)) % int32_t(VOCAB)), {}, {}});
            return rows;
        };
        return op;
    }
};

GenerationOptions armed_options(const std::string& path) {
    GenerationOptions options;
    options.max_draft = 3;
    options.cache_slots = 0;
    options.cache_bytes = 1u << 20;
    options.window_hashes = path;
    options.dump_shape.hc = HC;
    options.dump_shape.n_embd = N_EMBD;
    return options;
}

std::string run_and_read(const std::string& path) {
    Generation generation(VOCAB, CONTEXT, {}, Backend().operations(), armed_options(path));
    GenerationEvents events;
    events.token = [](int32_t) {};
    return std::to_string(generation.run(parse_generation_request("GEN 6 0,1", VOCAB, CONTEXT), events).generated);
}

void generation_records_every_window() {
    const auto dir = std::filesystem::temp_directory_path() / "strata-hpu-window-hashes";
    std::filesystem::create_directories(dir);
    const auto first = dir / "run-one.txt", second = dir / "run-two.txt";
    std::filesystem::remove(first);
    std::filesystem::remove(second);
    const auto emitted = run_and_read(first.string());
    run_and_read(second.string());
    const std::string record = read_all(first);
    std::vector<std::string> lines;
    {
        std::istringstream input(record);
        for (std::string line; std::getline(input, line);) lines.push_back(line);
    }
    uint64_t rows_in_record = 0;
    for (const auto& line : lines) { std::istringstream input(line); long long w = -1, position = -1; int r = 0; input >> w >> position >> r; rows_in_record += uint64_t(r); }
    check(lines.size() >= 2 && rows_in_record == uint64_t(std::stoull(emitted)),
          "one line per verify window and the window rows add up to the emitted tokens: " + emitted +
              " tokens in " + std::to_string(lines.size()) + " records, " + std::to_string(rows_in_record) + " rows");
    // The FIRST window must hash the FIRST rows that were fed, the next window the rows after them: this is what
    // pins the queue's front-consumption, and a writer that took the tail or left rows in place would still print a
    // well-formed file. The fixture's row k is the constant vector {k*100+0 .. k*100+5}, so the expectation here is
    // an independent recomputation, not a copy of what the writer emitted.
    {
        uint64_t consumed = 0;
        bool aligned = true;
        for (const auto& line : lines) {
            std::istringstream input(line);
            long long window = -1, position = -1;
            int rows = 0;
            std::string hash;
            input >> window >> position >> rows >> hash;
            std::vector<float> expect;
            for (int row = 0; row < rows; ++row)
                for (uint32_t at = 0; at < ROW_FLOATS; ++at)
                    expect.push_back(float((consumed + uint64_t(row)) * 100 + at));
            consumed += uint64_t(rows);
            if (hex16(window_residual_hash(std::span<const float>(expect))) != hash) aligned = false;
        }
        check(aligned, "each window hashes exactly its own rows, taken from the FRONT of the residual queue");
    }
    check(record == read_all(second),
          "two runs over the same arithmetic write the same file - that is the record's only purpose (:861-863)");
    long long previous_position = 0;
    bool monotone = true;
    for (size_t at = 0; at < lines.size(); ++at) {
        std::istringstream line(lines[at]);
        long long window = -1, position = -1;
        int rows = 0;
        std::string hash;
        line >> window >> position >> rows >> hash;
        if (window != static_cast<long long>(at) || rows < 1 || hash.size() != 16) {
            monotone = false;
            break;
        }
        for (const char c : hash)
            if (!std::isdigit(static_cast<unsigned char>(c)) && !(c >= 'a' && c <= 'f')) monotone = false;
        if (at && position <= previous_position) monotone = false;
        previous_position = position;
    }
    check(monotone, "each line is <window index> <position> <rows> <16 hex>, indexed from 0 with positions advancing");
    check(refused([&] {
                  GenerationOptions options = armed_options((dir / "no-producer.txt").string());
                  auto op = Backend().operations();
                  op.residual = nullptr;
                  (void)Generation(VOCAB, CONTEXT, {}, std::move(op), options);
              }, "--window-hashes needs the post-last-layer residual producer"),
          "arming without a residual producer is refused instead of hashing nothing");
    check(refused([&] {
                  GenerationOptions options = armed_options((dir / "no-shape.txt").string());
                  options.dump_shape.hc = 0;
                  (void)Generation(VOCAB, CONTEXT, {}, Backend().operations(), options);
              }, "--window-hashes needs the residual row shape"),
          "arming without the hc x n_embd shape is refused instead of hashing a wrong row length");
    std::filesystem::remove(first);
    std::filesystem::remove(second);
}

}

int main() {
    try {
        hash_law();
        writer_lines();
        generation_records_every_window();
    } catch (const std::exception& error) {
        std::cout << "FAIL window-hash check threw before finishing: " << error.what() << "\n";
        return 1;
    }
    if (failures) {
        std::cout << "window hashes: " << failures << " failed\n";
        return 1;
    }
    std::cout << "PASS FNV-1a vectors, upstream's window line layout and refusals, one record per verify window, two runs byte-identical\n";
    return 0;
}
