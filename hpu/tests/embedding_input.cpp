// hpu/tests/embedding_input.cpp - host evidence for the port of upstream's `--embd-gguf`
// (src/program/generate.cpp:397/:1972/:3032-3037/:8155; loader src/core/native_head.cpp:120-153).
//
// Two claims, both without a device:
//  (a) the validation refusals hold as a whole set - wrong hidden width, wrong row count against the
//      model's vocabulary, zero rows, non-finite values, an embedding file that claims more rows than
//      the prompt, an absent token_embd.weight, a truncated payload, and a token ID outside a
//      full-table file's rows (upstream's shape law, native_head.cpp:130-135/:139-142); and
//  (b) for the same token IDs, the embedding path's inputs equal an equivalent embedding-table
//      lookup - compared against independently transcribed EXPECTED rows (retyped from the token IDs
//      they belong to), never against the module's own output. A full [n_embd, n_vocab] BF16 file
//      (the form tools/embd_bf16_pack.py ships) replaces the lookup by token ID exactly as upstream;
//      the absent-flag default passes the table's rows through unchanged.
#include "strata/hpu/embedding_input.hpp"

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

int failures = 0;
void require(const bool ok, const char* const what) {
    if (!ok) {
        std::printf("FAIL %s\n", what);
        ++failures;
    }
}

constexpr uint64_t HIDDEN = 4, VOCAB = 16;
const std::vector<int32_t> PROMPT{7, 0, 9};

// The canonical embedding table, transcribed by hand: 16 token IDs of width-4 rows, every value on
// the half-integers so BF16 (upstream's shipped form) is exact. This is the test's own reference.
const float REF[VOCAB][HIDDEN] = {
    {0.0f, 0.5f, -1.0f, 1.5f},     {2.0f, -2.5f, 3.0f, 3.5f},    {-4.0f, 4.5f, 5.0f, -5.5f},  {6.0f, -6.5f, 7.0f, 7.5f},
    {-8.0f, 8.5f, -9.0f, 9.5f},    {10.0f, -10.5f, 11.0f, -11.5f}, {-12.0f, 12.5f, 13.0f, -13.5f}, {-14.0f, -14.5f, -15.0f, -15.5f},
    {16.0f, 16.5f, -17.0f, 17.5f}, {-18.0f, 18.5f, 19.0f, -19.5f}, {20.0f, -20.5f, -21.0f, 21.5f}, {-22.0f, 22.5f, -23.0f, 23.5f},
    {24.0f, 24.5f, 25.0f, -25.5f}, {-26.0f, -26.5f, 27.0f, 27.5f}, {28.0f, -28.5f, 29.0f, -29.5f}, {-30.0f, 30.5f, -31.0f, 31.5f},
};

// The separately transcribed expectation: rows for tokens 7, 0, 9 retyped from the token IDs they
// belong to, not copied from REF. A divergence here is a test bug the checks below surface.
const float EXPECTED[3][HIDDEN] = {
    {-14.0f, -14.5f, -15.0f, -15.5f},  // token 7
    {0.0f, 0.5f, -1.0f, 1.5f},          // token 0
    {-18.0f, 18.5f, 19.0f, -19.5f},     // token 9
};
const float EXPECTED_12[HIDDEN] = {24.0f, 24.5f, 25.0f, -25.5f};  // token 12, the fallback position

// The equivalent embedding-table lookup: a naive index into the transcription, never the loader.
std::vector<float> table_lookup(const int32_t token) {
    const float* row = REF[token];
    return {row[0], row[1], row[2], row[3]};
}
const std::function<std::vector<float>(int32_t)> NO_TABLE{};

bool equals(const std::vector<float>& flat, const float (&expected)[3][HIDDEN]) {
    if (flat.size() != 12) return false;
    for (size_t i = 0; i < 12; ++i)
        if (flat[i] != (&expected[0][0])[i]) return false;
    return true;
}
bool equals_row(const std::vector<float>& row, const float (&expected)[HIDDEN]) {
    if (row.size() != HIDDEN) return false;
    for (size_t c = 0; c < HIDDEN; ++c)
        if (row[c] != expected[c]) return false;
    return true;
}

void append_u32(std::string& out, const uint32_t v) { out.append(reinterpret_cast<const char*>(&v), 4); }
void append_u64(std::string& out, const uint64_t v) { out.append(reinterpret_cast<const char*>(&v), 8); }
void append_text(std::string& out, const std::string& s) { append_u64(out, uint64_t(s.size())); out += s; }

// The one-tensor GGUF shape tools/embd_bf16_pack.py writes: general.architecture "strata-embd" and
// token_embd.weight with ne = [n_embd, rows], data past the header aligned to 32.
std::string gguf(const std::string& tensor, const uint64_t width, const uint64_t rows, const uint32_t type,
                 const std::string& payload) {
    std::string out;
    append_u32(out, 0x46554747u);
    append_u32(out, 3);
    append_u64(out, 1);
    append_u64(out, 1);
    append_text(out, "general.architecture");
    append_u32(out, 8);  // GGUF_TYPE_STRING
    append_text(out, "strata-embd");
    append_text(out, tensor);
    append_u32(out, 2);
    append_u64(out, width);
    append_u64(out, rows);
    append_u32(out, type);  // 0 = F32, 30 = BF16 (ggml)
    append_u64(out, 0);
    out.resize(((out.size() + 31) / 32) * 32, '\0');
    out += payload;
    return out;
}

std::string f32_payload(const float* values, const size_t count) {
    std::string out;
    out.resize(count * 4);
    std::memcpy(out.data(), values, count * 4);
    return out;
}
std::string bf16_payload(const float* values, const size_t count) {
    std::string out;
    out.reserve(count * 2);
    for (size_t i = 0; i < count; ++i) {
        uint32_t bits = 0;
        std::memcpy(&bits, &values[i], 4);
        const uint16_t half = uint16_t(bits >> 16);  // exact for the half-integer transcriptions
        out.append(reinterpret_cast<const char*>(&half), 2);
    }
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
}  // namespace

int main() {
    const std::filesystem::path dir = "work/tmp-embd";
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
    std::filesystem::create_directories(dir);
    try {
        const auto path = [&dir](const char* name) { return (dir / name).string(); };

        // ---- fixtures
        write_file(dir / "table.gguf", gguf("token_embd.weight", HIDDEN, VOCAB, 30, bf16_payload(&REF[0][0], VOCAB * HIDDEN)));
        write_file(dir / "prompt.gguf", gguf("token_embd.weight", HIDDEN, 3, 0, f32_payload(&EXPECTED[0][0], 3 * HIDDEN)));
        float five[5][HIDDEN] = {{0.125f, 0.25f, 0.375f, 0.5f}, {0.5f, 0.625f, 0.75f, 0.875f}, {1.0f, 1.125f, 1.25f, 1.375f},
                                 {1.5f, 1.625f, 1.75f, 1.875f}, {2.0f, 2.125f, 2.25f, 2.375f}};
        write_file(dir / "five.gguf", gguf("token_embd.weight", HIDDEN, 5, 0, f32_payload(&five[0][0], 5 * HIDDEN)));
        five[2][3] = std::numeric_limits<float>::quiet_NaN();
        five[4][0] = std::numeric_limits<float>::infinity();
        write_file(dir / "nan.gguf", gguf("token_embd.weight", HIDDEN, 5, 0, f32_payload(&five[0][0], 5 * HIDDEN)));
        write_file(dir / "wide.gguf", gguf("token_embd.weight", 8, 8, 30, bf16_payload(&REF[0][0], 8 * HIDDEN)));
        write_file(dir / "rows17.gguf", gguf("token_embd.weight", HIDDEN, 17, 0, f32_payload(&REF[0][0], 16 * HIDDEN) + f32_payload(&EXPECTED_12[0], HIDDEN)));
        write_file(dir / "empty.gguf", gguf("token_embd.weight", HIDDEN, 0, 0, std::string()));
        write_file(dir / "wrongname.gguf", gguf("output.weight", HIDDEN, VOCAB, 30, bf16_payload(&REF[0][0], VOCAB * HIDDEN)));
        std::string truncated = gguf("token_embd.weight", HIDDEN, VOCAB, 30, bf16_payload(&REF[0][0], VOCAB * HIDDEN));
        truncated.resize(truncated.size() - 96);  // header and its alignment intact, payload short by 96 B
        write_file(dir / "truncated.gguf", truncated);

        // ---- (b) upstream-shaped file: a full [n_embd, n_vocab] BF16 table replaces the lookup by token ID
        const EmbeddingInput full = EmbeddingInput::load(path("table.gguf"), HIDDEN, VOCAB);
        require(full.active() && full.mode() == EmbeddingInputMode::Vocabulary, "a full table is consumed by token ID, as upstream");
        require(full.rows() == VOCAB && full.hidden() == HIDDEN, "the file's row count and width reach the caller");
        const auto table_path = table_lookup(PROMPT[0]);  // (shape sample for the equivalence checks)
        require(table_path.size() == HIDDEN, "the reference lookup yields one row");
        const std::vector<float> flat = full.prompt_inputs(PROMPT, NO_TABLE);  // the table is never consulted
        require(equals(flat, EXPECTED), "a full-table embedding file yields, per token ID, the transcribed embedding-table inputs");
        bool all_ids = true;
        for (uint32_t token = 0; token < VOCAB; ++token)
            if (full.row(token) != table_lookup(int32_t(token))) all_ids = false;
        require(all_ids, "every token ID gathers the same row the embedding-table lookup would return");
        const std::function<std::vector<float>(int32_t)> gather =
            [&](const int32_t token) { return full.row(uint64_t(token)); };
        require(gather(7) == table_lookup(7) && equals_row(gather(9), EXPECTED[2]), "the gathered rows match the transcription");
        const auto decode = full.decode_lookup(NO_TABLE);  // an empty table proves the lookup is replaced
        bool decode_same = decode != nullptr;
        for (uint32_t token = 0; token < VOCAB && decode; ++token)
            if (decode(int32_t(token)) != table_lookup(int32_t(token))) decode_same = false;
        require(decode_same, "the token-path seam with a full file gathers from it, not from the table");

        // ---- (b, port mode) a prompt-length file replaces the lookup for the provided rows, positionally
        const EmbeddingInput provided = EmbeddingInput::load(path("prompt.gguf"), HIDDEN, VOCAB);
        require(provided.mode() == EmbeddingInputMode::Prompt && provided.rows() == 3, "a sub-vocabulary file is consumed positionally");
        require(equals(provided.prompt_inputs(PROMPT, NO_TABLE), EXPECTED), "the provided rows are exactly the embedding-table inputs for tokens 7, 0, 9");
        const std::vector<int32_t> longer{7, 0, 9, 12};
        const auto grown = provided.prompt_inputs(longer, table_lookup);
        require(equals(std::vector<float>(grown.begin(), grown.end() - 4), EXPECTED) &&
                    equals_row(std::vector<float>(grown.end() - 4, grown.end()), EXPECTED_12),
                "positions past the provided rows keep the embedding-table lookup");
        require(provided.decode_lookup(table_lookup)(5) == table_lookup(5), "a positional file leaves the token path on the table");

        // ---- default (flag absent): byte-identical to the table path of today
        const EmbeddingInput absent;
        require(!absent.active() && absent.rows() == 0, "no flag, no embedding input");
        require(equals(absent.prompt_inputs(PROMPT, table_lookup), EXPECTED), "the absent-flag default consumes exactly what the embedding table yields");
        require(absent.decode_lookup(table_lookup)(9) == table_lookup(9), "the absent-flag decode path is the table itself");

        // ---- (a) the refusals, as a whole set, each with its own reason
        must_refuse([&] { EmbeddingInput::load(path("wide.gguf"), HIDDEN, VOCAB); }, "hidden width", "a file of another hidden width than the model's");
        must_refuse([&] { EmbeddingInput::load(path("rows17.gguf"), HIDDEN, VOCAB); }, "row count", "a file with more rows than the model's vocabulary");
        must_refuse([&] { EmbeddingInput::load(path("empty.gguf"), HIDDEN, VOCAB); }, "zero rows", "a file with zero rows");
        must_refuse([&] { EmbeddingInput::load(path("nan.gguf"), HIDDEN, VOCAB); }, "non-finite", "a file holding a NaN or infinity");
        must_refuse([&] { EmbeddingInput::load(path("wrongname.gguf"), HIDDEN, VOCAB); }, "absent", "a file without token_embd.weight");
        must_refuse([&] { EmbeddingInput::load(path("truncated.gguf"), HIDDEN, VOCAB); }, "truncated", "a file whose payload is cut short");
        const EmbeddingInput five_rows = EmbeddingInput::load(path("five.gguf"), HIDDEN, VOCAB);
        must_refuse([&] { five_rows.prompt_inputs(PROMPT, NO_TABLE); }, "more rows", "a file that claims more rows than the prompt has");
        must_refuse([&] { full.prompt_inputs(std::vector<int32_t>{999999}, NO_TABLE); }, "outside", "a token ID outside the full table");
        must_refuse([&] { full.prompt_inputs(std::vector<int32_t>{-1}, NO_TABLE); }, "outside", "a negative token ID");

        // ---- checkpoint binding (upstream generate.cpp:8155 + conversation_file.cpp:935): the file joins
        // the model identity, so a swapped embedding file cannot silently reuse a saved checkpoint.
        require(!full.file_identity().empty() && full.file_identity() != provided.file_identity(),
                "distinct embedding files carry distinct identities for the session binding");
    } catch (const std::exception& error) {
        std::printf("FAIL unexpected exception escaped the fixtures: %s\n", error.what());
        ++failures;
    }
    std::filesystem::remove_all(dir, ec);
    if (failures) {
        std::printf("%d embedding_input checks failed\n", failures);
        return 1;
    }
    std::printf("PASS embedding_input: --embd-gguf refuses {hidden width, row count, zero rows, non-finite, rows beyond the prompt, absent tensor, truncated payload} as a set and its rows equal the equivalent embedding-table lookup for the same token IDs against independently transcribed expectations; no device\n");
    return 0;
}
