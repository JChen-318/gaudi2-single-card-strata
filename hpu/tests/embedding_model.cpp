// hpu/tests/embedding_model.cpp - host evidence for the MODEL seam's consumption of upstream's
// `--embd-gguf` (src/program/generate.cpp:397 field, :1972 parse, :3032-3037 load and install, :8155
// session binding; loader src/core/native_head.cpp:120-153; gathers src/core/layer.cpp:1195-1200,
// src/core/verify.cpp:944-946, src/core/mtp.cpp:893-895, src/prefill/prefill.cpp:1420-1424;
// checkpoint fingerprint src/core/conversation_file.cpp:935).
//
// hpu/tests/embedding_input.cpp already covers the LOADER. This test covers the law hpu/src/model.cpp
// puts in front of every seam - strata/hpu/embedding_route.hpp - driven the way the seams drive it:
//   (1) option empty           -> rows equal the canonical embedding table, and nothing else is asked;
//   (2) vocabulary-covering    -> every row equals the file, by token ID, and the table callback is
//                                 never consulted (upstream's single law);
//   (3) file shorter           -> the file's rows for the positions it claims, the table after them;
//   (4) hidden-width mismatch  -> refused at load (EmbeddingInput::load) and again at the seam;
//   (5) prompt, window, verify and MTP draft replay all read ONE route, so their rows are equal by
//       construction and equal to the transcriptions below.
// The session's embedding callback receives only a token ID (strata/hpu/session_flow.hpp:14), so
// positional mode reads its position from EmbeddingWindow; that cursor is exercised here too,
// including the partial-acceptance replay session_flow.cpp performs.
// No device: nothing here opens a card, and Model itself cannot be constructed on this host because
// strata/hpu/model.hpp reaches <synapse_api.h> through layer.hpp -> runtime.hpp.
#include "strata/hpu/embedding_route.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using strata::hpu::EmbeddingInput;
using strata::hpu::EmbeddingInputMode;
using strata::hpu::EmbeddingRowMode;
using strata::hpu::EmbeddingRows;
using strata::hpu::EmbeddingWindow;
using Row = std::vector<float>;
using TableLookup = EmbeddingRows::TableLookup;

int failures = 0;
void require(const bool ok, const char* const what) {
    if (!ok) {
        std::printf("FAIL %s\n", what);
        ++failures;
    }
}

constexpr uint64_t HIDDEN = 4, VOCAB = 8;
const std::vector<int32_t> PROMPT{5, 2, 7, 0, 3};            // five prompt rows, positions 0..4
const std::vector<int32_t> DRAFTED{1, 6, 4};                  // the MTP drafter's picks, positions 5..7

// The canonical embedding table, transcribed by hand (half-integers keep BF16 exact).
const float TABLE[VOCAB][HIDDEN] = {
    {0.0f, 0.5f, -0.5f, 1.0f},      {1.5f, -1.5f, 2.0f, -2.0f}, {2.5f, 3.0f, -3.0f, 3.5f},   {-4.0f, 4.5f, 5.0f, -5.5f},
    {6.0f, -6.5f, 7.0f, 7.5f},      {-8.0f, 8.5f, -9.0f, 9.5f}, {10.0f, -10.5f, 11.0f, 11.5f}, {-12.0f, 12.5f, 13.0f, -13.5f},
};
// A vocabulary-covering embedding file: DIFFERENT values per row, so any leak of the table is visible.
const float FILE_ROWS[VOCAB][HIDDEN] = {
    {-0.25f, 0.25f, 0.75f, -0.75f}, {1.25f, -1.75f, 2.25f, 2.75f},  {-3.25f, 3.75f, -4.25f, 4.75f}, {5.25f, 5.75f, -6.25f, -6.75f},
    {7.25f, -7.75f, 8.25f, 8.75f},  {-9.25f, 9.75f, 10.25f, -10.75f}, {11.25f, 11.75f, -12.25f, 12.75f}, {-13.25f, -13.75f, 14.25f, 14.75f},
};
// A three-row (shorter than the vocabulary) embedding file, consumed positionally.
const float SHORT[3][HIDDEN] = {
    {20.0f, -20.5f, 21.0f, 21.5f}, {-22.0f, 22.5f, -23.0f, 23.5f}, {24.0f, 24.5f, -25.0f, 25.5f},
};

Row table_row(const int32_t token) {
    const float* row = TABLE[token];
    return {row[0], row[1], row[2], row[3]};
}
Row file_row(const uint64_t index) {
    const float* row = FILE_ROWS[index];
    return {row[0], row[1], row[2], row[3]};
}
Row short_row(const uint64_t index) {
    const float* row = SHORT[index];
    return {row[0], row[1], row[2], row[3]};
}
bool same(const Row& a, const float (&expected)[HIDDEN]) {
    if (a.size() != HIDDEN) return false;
    for (uint64_t c = 0; c < HIDDEN; ++c) if (a[c] != expected[c]) return false;
    return true;
}
// The same comparison for a FLAT [T,HIDDEN] block: the seam hands the session a row-major block, not rows.
bool same_at(const std::vector<float>& flat, const size_t row, const float (&expected)[HIDDEN]) {
    if (flat.size() < (row + 1) * HIDDEN) return false;
    for (uint64_t c = 0; c < HIDDEN; ++c) if (flat[row * HIDDEN + c] != expected[c]) return false;
    return true;
}
bool same_block(const std::vector<float>& flat, const std::vector<int32_t>& tokens, const TableLookup& expect) {
    if (flat.size() != tokens.size() * HIDDEN) return false;
    for (size_t row = 0; row < tokens.size(); ++row)
        for (uint64_t c = 0; c < HIDDEN; ++c)
            if (flat[row * HIDDEN + c] != expect(tokens[row])[c]) return false;
    return true;
}

// A table lookup that counts its calls: the vocabulary law says the table must never be consulted.
TableLookup counting_table(int* const calls) {
    return [calls](const int32_t token) {
        ++*calls;
        return table_row(token);
    };
}

void append_u32(std::string& out, const uint32_t v) { out.append(reinterpret_cast<const char*>(&v), 4); }
void append_u64(std::string& out, const uint64_t v) { out.append(reinterpret_cast<const char*>(&v), 8); }
void append_text(std::string& out, const std::string& s) { append_u64(out, uint64_t(s.size())); out += s; }
std::string f32_payload(const float* values, const size_t count) {
    std::string out;
    out.resize(count * 4);
    std::memcpy(out.data(), values, count * 4);
    return out;
}
// The one-tensor GGUF tools/embd_bf16_pack.py writes, in F32 so the transcriptions stay exact here.
std::string gguf(const uint64_t width, const uint64_t rows, const std::string& payload) {
    std::string out;
    append_u32(out, 0x46554747u);
    append_u32(out, 3);
    append_u64(out, 1);
    append_u64(out, 1);
    append_text(out, "general.architecture");
    append_u32(out, 8);
    append_text(out, "strata-embd");
    append_text(out, "token_embd.weight");
    append_u32(out, 2);
    append_u64(out, width);
    append_u64(out, rows);
    append_u32(out, 0);  // F32
    append_u64(out, 0);
    out.resize(((out.size() + 31) / 32) * 32, '\0');
    out += payload;
    return out;
}
void write_file(const std::filesystem::path& path, const std::string& bytes) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.write(bytes.data(), std::streamsize(bytes.size()));
    out.close();
    if (!out) throw std::runtime_error("cannot write fixture " + path.string());
}
template <class Call>
void must_refuse(Call&& call, const char* const needle, const char* const what) {
    std::string message;
    bool refused = false;
    try {
        call();
    } catch (const std::exception& error) {
        refused = true;
        message = error.what();
    }
    if (!refused) {
        std::printf("FAIL no refusal: %s\n", what);
        ++failures;
    } else if (message.find(needle) == std::string::npos) {
        std::printf("FAIL wrong refusal reason for %s: %s\n", what, message.c_str());
        ++failures;
    }
}

// The seam, exactly as hpu/src/model.cpp wires it: the session's embedding callback asks the route for
// (token, position), the position coming from the window the caller opened or from the committed
// history length (session_flow.cpp records a decoded token only after its embedding was gathered).
struct Seam {
    const EmbeddingRows& rows;
    EmbeddingWindow& window;
    uint64_t history = 0;
    Row embedding(const int32_t token) { return rows.row(token, window.next(token, history)); }
    // SessionFlow::decode / Model::prefill: one token, position == committed history length.
    Row decode(const int32_t token) {
        auto row = embedding(token);
        ++history;
        return row;
    }
    // SessionFlow::run_window (prefill_window / verify_window) row loop, then the commit of the window.
    std::vector<float> window_rows(const std::vector<int32_t>& tokens) {
        window.open(history, tokens);
        std::vector<float> block;
        for (const int32_t token : tokens) {
            const Row row = embedding(token);
            block.insert(block.end(), row.begin(), row.end());
        }
        window.close();
        history += tokens.size();
        return block;
    }
    // Model::draft: the MTP drafter re-embeds each picked token at its own sequence position.
    Row draft(const int32_t token) {
        const Row row = rows.row(token, history);
        ++history;
        return row;
    }
};
}  // namespace

int main() {
    const std::filesystem::path dir = "work/tmp-embd-model";
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
    std::filesystem::create_directories(dir);
    const TableLookup table = table_row;
    try {
        write_file(dir / "full.gguf", gguf(HIDDEN, VOCAB, f32_payload(&FILE_ROWS[0][0], VOCAB * HIDDEN)));
        write_file(dir / "short.gguf", gguf(HIDDEN, 3, f32_payload(&SHORT[0][0], 3 * HIDDEN)));
        write_file(dir / "wide.gguf", gguf(HIDDEN * 2, VOCAB, f32_payload(&FILE_ROWS[0][0], VOCAB * HIDDEN)));
        const auto path = [&dir](const char* name) { return (dir / name).string(); };

        // ---- (1) the option is empty: rows equal the table, and only the table is asked.
        const EmbeddingInput absent;
        int table_calls = 0;
        const EmbeddingRows off = EmbeddingRows::for_input(absent, VOCAB, HIDDEN, counting_table(&table_calls));
        require(off.mode() == EmbeddingRowMode::Table && !off.active() && !off.positional(),
                "an absent --embd-gguf leaves the seam on the canonical table law");
        EmbeddingWindow cursor;
        Seam seam{off, cursor};
        bool off_table = true;
        for (const int32_t token : PROMPT)
            if (!same(seam.decode(token), TABLE[token])) off_table = false;
        require(off_table, "option empty: every prompt row equals the embedding-table row for the same token ID");
        const auto off_window = seam.window_rows(DRAFTED);
        require(same_block(off_window, DRAFTED, table_row) && table_calls == int(PROMPT.size() + DRAFTED.size()),
                "option empty: window rows come from the table too, with no other source consulted");
        must_refuse([&] { EmbeddingRows::for_input(absent, VOCAB, 0, table); }, "positive", "a route built without a model width");

        // ---- (4) a hidden-width mismatch is refused at load, and again where the seam is built.
        must_refuse([&] { EmbeddingInput::load(path("wide.gguf"), HIDDEN, VOCAB); }, "hidden width",
                    "a file whose hidden width differs from the model's is refused at load");
        const EmbeddingInput full = EmbeddingInput::load(path("full.gguf"), HIDDEN, VOCAB);
        require(full.mode() == EmbeddingInputMode::Vocabulary && full.rows() == VOCAB, "a vocabulary-covering file is loaded");
        must_refuse([&] { EmbeddingRows::for_input(full, VOCAB, HIDDEN * 2, table); }, "hidden width",
                    "the seam re-checks the width before it routes a single row");

        // ---- (2) a vocabulary-covering file: every row, prompt or decoded, comes from the file by ID.
        table_calls = 0;
        const EmbeddingRows by_id = EmbeddingRows::for_input(full, VOCAB, HIDDEN, counting_table(&table_calls));
        require(by_id.mode() == EmbeddingRowMode::FileByToken && by_id.rows() == VOCAB,
                "the route built from a vocabulary-covering file is the token-ID law");
        bool id_rows = true;
        for (uint32_t token = 0; token < VOCAB; ++token)
            for (const uint64_t position : {uint64_t(0), uint64_t(token), uint64_t(VOCAB + 41)})  // position is inert
                if (by_id.row(int32_t(token), position) != file_row(token)) id_rows = false;
        require(id_rows && table_calls == 0,
                "vocabulary-covering file: every token ID gathers the file's row and the table is never consulted");
        EmbeddingWindow id_cursor;
        Seam id_seam{by_id, id_cursor};
        const auto id_prompt = id_seam.window_rows(PROMPT);
        require(same_block(id_prompt, PROMPT, [&](const int32_t token) { return file_row(uint64_t(token)); }),
                "vocabulary-covering file: the prefill window block is the file's rows in prompt order");
        const auto id_verify = id_seam.window_rows(DRAFTED);
        const auto id_draft = [&] {
            std::vector<float> flat;
            for (const int32_t token : DRAFTED) {
                const Row row = id_seam.draft(token);
                flat.insert(flat.end(), row.begin(), row.end());
            }
            return flat;
        }();
        require(id_verify.size() == DRAFTED.size() * HIDDEN && id_verify == id_draft,
                "vocabulary-covering file: the candidate window and the MTP draft replay gather identical rows");
        require(table_calls == 0, "vocabulary-covering file: no seam ever falls back to the embedding table");
        must_refuse([&] { by_id.row(int32_t(VOCAB), 0); }, "outside", "a token ID the covering file cannot serve");
        must_refuse([&] { by_id.row(-1, 0); }, "outside", "a negative token ID");

        // ---- (3) a shorter file: the file's rows for the positions it claims, the table after them.
        const EmbeddingInput short_file = EmbeddingInput::load(path("short.gguf"), HIDDEN, VOCAB);
        require(short_file.mode() == EmbeddingInputMode::Prompt && short_file.rows() == 3,
                "a sub-vocabulary file loads as the positional law");
        table_calls = 0;
        const EmbeddingRows positional = EmbeddingRows::for_input(short_file, VOCAB, HIDDEN, counting_table(&table_calls));
        require(positional.mode() == EmbeddingRowMode::FileByPosition && positional.positional(),
                "the route built from a shorter file is positional");
        bool claimed = true;
        for (uint64_t position = 0; position < 3; ++position)
            if (positional.row(PROMPT[size_t(position)], position) != short_row(position)) claimed = false;
        require(claimed && table_calls == 0, "shorter file: each claimed position comes from the file, never from the table");
        bool after = true;
        for (uint64_t position = 3; position < PROMPT.size(); ++position)
            if (positional.row(PROMPT[size_t(position)], position) != table_row(PROMPT[size_t(position)])) after = false;
        require(after && table_calls == int(PROMPT.size()) - 3, "shorter file: positions past the claim keep the table");
        EmbeddingWindow pos_cursor;
        Seam pos_seam{positional, pos_cursor};
        const auto pos_prompt = pos_seam.window_rows(PROMPT);
        require(pos_prompt.size() == PROMPT.size() * HIDDEN &&
                    same_at(pos_prompt, 0, SHORT[0]) && same_at(pos_prompt, 1, SHORT[1]) && same_at(pos_prompt, 2, SHORT[2]) &&
                    same_at(pos_prompt, 3, TABLE[PROMPT[3]]) && same_at(pos_prompt, 4, TABLE[PROMPT[4]]),
                "shorter file: the prefill window is the file's rows then the table's rows, position by position");
        const std::vector<int32_t> later{4, 1, 6};  // positions 5..7: the file claims none of them
        const auto pos_window = pos_seam.window_rows(later);
        require(same_block(pos_window, later, table_row), "shorter file: a later window keeps the table for every row");
        const auto pos_mtp = [&] {
            std::vector<float> flat;
            for (const int32_t token : DRAFTED) {
                const Row row = positional.row(token, PROMPT.size() + size_t(&token - &DRAFTED[0]));
                flat.insert(flat.end(), row.begin(), row.end());
            }
            return flat;
        }();
        require(same_block(pos_mtp, DRAFTED, table_row), "shorter file: MTP draft replay reads the table for its own positions");

        // ---- (5) one law for every seam: the same route re-driven from position 0 reproduces the same bytes.
        EmbeddingWindow replay_cursor;
        Seam replay{positional, replay_cursor};
        require(replay.window_rows(PROMPT) == pos_prompt, "re-driving the same window from its base position is reproducible");
        replay_cursor.open(0, PROMPT);
        bool rewind_ok = true;
        for (size_t row = 0; row < 3; ++row) rewind_ok = rewind_ok && same_at(pos_prompt, row, SHORT[row]);
        require(rewind_ok, "a partially replayed window re-reads the file's rows at the same positions");

        // ---- the cursor's own law: positional files need a position, and a desync is refused, not guessed.
        EmbeddingWindow strict;
        require(!strict.open() && strict.next(3, 17) == 17, "no window open: the seam falls back to the committed history length");
        strict.open(2, {4, 1});
        require(strict.next(4, 0) == 2 && strict.next(1, 0) == 3, "an open window hands out consecutive positions");
        strict.next(4, 0);  // the window is spent; a replay rewinds it
        must_refuse([&] { strict.next(7, 0); }, "desync", "a token that is not the window's next row is refused, never guessed");
        strict.close();
        require(strict.next(7, 9) == 9, "a closed window returns to the history-length position");

        // ---- a row of the wrong width is refused in the file modes instead of reaching the trunk.
        const EmbeddingRows narrow = EmbeddingRows::positional([](const uint64_t) { return Row{1.0f, 2.0f, 3.0f}; }, 3, VOCAB, HIDDEN,
                                                              [](const int32_t) { return Row(HIDDEN + 1, 1.0f); });
        must_refuse([&] { narrow.row(0, 0); }, "floats wide", "a file row of the model's wrong width is refused");
        must_refuse([&] { narrow.row(0, 5); }, "floats wide", "a table row of the model's wrong width is refused in the file modes");
        must_refuse([&] { EmbeddingRows::from_file(file_row, VOCAB + 1, VOCAB, HIDDEN); }, "row count",
                    "a route whose file row count is not the vocabulary is refused at construction");
        must_refuse([&] { EmbeddingRows::positional(file_row, VOCAB, VOCAB, HIDDEN, table); }, "fewer rows",
                    "a positional route must claim fewer rows than the vocabulary");

        // ---- the checkpoint binding the model relies on (generate.cpp:8155, conversation_file.cpp:935).
        require(!full.file_identity().empty() && full.file_identity() != short_file.file_identity(),
                "distinct embedding files carry distinct identities for hpu/src/model.cpp's file_identity() binding");
    } catch (const std::exception& error) {
        std::printf("FAIL unexpected exception escaped the fixtures: %s\n", error.what());
        ++failures;
    }
    std::filesystem::remove_all(dir, ec);
    if (failures) {
        std::printf("%d embedding_model checks failed\n", failures);
        return 1;
    }
    std::printf("PASS embedding_model: the model seam's row law for --embd-gguf - option empty gives the table's rows, a vocabulary-covering file gives its own rows by token ID with the table never consulted, a shorter file gives its rows for the positions it claims and the table's after them, a hidden-width mismatch is refused at load and again at the seam, and prefill, window, verify and MTP draft replay read one identical route; no device\n");
    return 0;
}
