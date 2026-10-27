#pragma once
// strata/hpu/embedding_route.hpp - the row-selection law the generation seam applies to `--embd-gguf`.
//
// Upstream installs one table and every gather reads it: `embed_row` (src/core/layer.cpp:1195-1200)
// gathers one row per token ID, the verify window gathers the candidate rows by token ID
// (src/core/verify.cpp:944-946), the MTP drafter gathers the prompt's rows by token ID
// (src/core/mtp.cpp:893-895, src/prefill/prefill.cpp:1420-1424), and the flag itself is
// `o.embd_gguf` (src/program/generate.cpp:397, parsed at :1972, loaded and installed at :3032-3037,
// bound into the session's model-file identity at :8155 / src/core/conversation_file.cpp:935).
// Upstream's file always covers the vocabulary (src/core/native_head.cpp:130 demands
// shape[1] == n_vocab), so upstream has exactly one law: the file IS the table, keyed by token ID.
//
// strata/hpu/embedding_input.hpp additionally admits a file SHORTER than the vocabulary and documents
// its consumption as positional. This header is that law as one pure function of (token, position),
// shared by every seam so prompt, decode, MTP draft replay and candidate verification cannot diverge:
//   * Table          - the option is absent. The canonical token_embd.weight gather, unchanged.
//   * FileByToken    - rows == vocabulary: upstream's law. Every row, prompt or decoded, comes from
//                      the file by token ID and the table is never consulted (layer.cpp:1195-1200).
//   * FileByPosition - 0 < rows < vocabulary: the file claims sequence positions [0, rows). A lookup
//                      at a claimed position MUST come from the file (never a silent table fallback);
//                      later positions and decoded tokens keep the table.
// Row width is checked against the model's hidden width in both file modes; Table mode keeps the
// seam's existing checks untouched so the default path stays byte-for-byte what it was.
//
// Why the seam needs a position: SessionOperations::embedding (strata/hpu/session_flow.hpp:14) is
// per token only. EmbeddingWindow below is the seam's position source: the owner opens a window with
// the consecutive tokens it is about to submit, and the callback consumes it in order. Positional
// mode is the only mode that consults it, so the default and upstream-shaped paths gain no new state.
//
// Standalone by design for the host: it pulls in only strata/hpu/embedding_input.hpp (and through it
// weights.hpp/tensor.hpp, all GGUF/host code), never a Synapse header - unlike strata/hpu/model.hpp,
// whose layer.hpp/session_flow.hpp chain reaches hpu/runtime.hpp and <synapse_api.h>. That is why
// hpu/tests/embedding_model.cpp can execute this law and the loader's refusals on a machine with no
// card. The row SOURCES are injected: a WeightSlice gather for the table, EmbeddingInput::row for the
// file.
#include <cstdint>
#include <functional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "strata/hpu/embedding_input.hpp"

namespace strata::hpu {
enum class EmbeddingRowMode { Table, FileByToken, FileByPosition };

inline const char* embedding_row_mode_name(const EmbeddingRowMode mode) {
    switch (mode) {
        case EmbeddingRowMode::Table: return "table";
        case EmbeddingRowMode::FileByToken: return "file-by-token";
        case EmbeddingRowMode::FileByPosition: return "file-by-position";
    }
    return "unknown";
}

// The seam's position source for positional files only. Open/close is scoped by the caller that owns
// the token span (prefill window, candidate verification); every embedding callback the session makes
// while a window is open consumes one entry of it, in order. A window that is fully consumed and then
// asked again is a replay of the same window's prefix (session_flow.cpp's partial-acceptance replay),
// so the cursor rewinds. A token that does not match the opened window is a seam desync and is
// refused loudly: guessing a position here is exactly the silent fallback this module forbids.
class EmbeddingWindow {
public:
    bool open() const { return !tokens_.empty(); }
    uint64_t base() const { return base_; }
    void open(const uint64_t base, std::vector<int32_t> tokens) {
        base_ = base;
        index_ = 0;
        tokens_ = std::move(tokens);
    }
    void close() {
        base_ = 0;
        index_ = 0;
        tokens_.clear();
    }
    // The absolute position of `token`, or `fallback` (the session's committed length, which is the
    // position of a single decoded token) when no window is open.
    uint64_t next(const int32_t token, const uint64_t fallback) {
        if (tokens_.empty()) return fallback;
        if (index_ == tokens_.size()) index_ = 0;  // a replayed window re-embeds from its first row
        if (tokens_[index_] != token) throw std::logic_error("embedding window row/token desync");
        return base_ + index_++;
    }

private:
    uint64_t base_ = 0, index_ = 0;
    std::vector<int32_t> tokens_;
};

// One row source law for every seam. Copyable and immutable once built; the callbacks it holds are the
// model's own gathers (a WeightSlice row and an EmbeddingInput row), never a re-read of the file.
class EmbeddingRows {
public:
    using Row = std::vector<float>;
    using TableLookup = std::function<Row(int32_t)>;
    using FileLookup = std::function<Row(uint64_t)>;

    EmbeddingRows() = default;  // no option: the canonical table lookup

    // The canonical table only. `hidden` is kept for diagnostics; the gather stays what the seam does today.
    static EmbeddingRows from_table(TableLookup table, const uint64_t hidden) {
        if (!table) throw std::invalid_argument("embedding route: the model supplied no token_embd.weight lookup");
        return EmbeddingRows(EmbeddingRowMode::Table, std::move(table), FileLookup(), 0, hidden, hidden);
    }
    // A vocabulary-covering file: upstream's law, the file replaces the table for every token ID.
    static EmbeddingRows from_file(FileLookup file, const uint64_t rows, const uint64_t vocabulary, const uint64_t hidden) {
        if (!file) throw std::invalid_argument("embedding route: a vocabulary-covering file supplied no row reader");
        if (!rows || rows != vocabulary)
            throw std::invalid_argument("embedding route: a vocabulary-covering file must have exactly the model's row count");
        return EmbeddingRows(EmbeddingRowMode::FileByToken, TableLookup(), std::move(file), rows, vocabulary, hidden);
    }
    // A file shorter than the vocabulary: it claims positions [0, rows); the table keeps the rest.
    static EmbeddingRows positional(FileLookup file, const uint64_t rows, const uint64_t vocabulary, const uint64_t hidden,
                                    TableLookup table) {
        if (!file || !table) throw std::invalid_argument("embedding route: a positional file needs both a file and a table reader");
        if (!rows || rows >= vocabulary)
            throw std::invalid_argument("embedding route: a positional file must claim fewer rows than the model's vocabulary");
        return EmbeddingRows(EmbeddingRowMode::FileByPosition, std::move(table), std::move(file), rows, vocabulary, hidden);
    }

    // The seam's single construction point, used by hpu/src/model.cpp and by the host test, so the
    // model and the test cannot drift apart: an inactive (absent-flag) input yields the canonical
    // table lookup, a vocabulary-covering one yields upstream's law, a shorter one yields positional.
    static EmbeddingRows for_input(const EmbeddingInput& input, const uint64_t vocabulary, const uint64_t hidden,
                                   TableLookup table) {
        if (!hidden || !vocabulary)
            throw std::invalid_argument("embedding route: the model's hidden width and vocabulary must be positive");
        if (!table) throw std::invalid_argument("embedding route: the model supplied no token_embd.weight lookup");
        if (!input.active()) return from_table(std::move(table), hidden);
        if (input.hidden() != hidden)
            throw std::invalid_argument("embedding route: the embedding file's hidden width " +
                                        std::to_string(input.hidden()) + " is not the model's " + std::to_string(hidden));
        auto file = [input](const uint64_t index) { return input.row(index); };
        if (input.mode() == EmbeddingInputMode::Vocabulary)
            return from_file(std::move(file), input.rows(), vocabulary, hidden);
        return positional(std::move(file), input.rows(), vocabulary, hidden, std::move(table));
    }

    EmbeddingRowMode mode() const { return mode_; }
    uint64_t rows() const { return rows_; }
    uint64_t hidden() const { return hidden_; }
    // The model vocabulary the route was built against; the file's row count says which law applies.
    uint64_t vocabulary() const { return vocabulary_; }
    bool positional() const { return mode_ == EmbeddingRowMode::FileByPosition; }
    bool active() const { return mode_ != EmbeddingRowMode::Table; }

    // The law. `position` is the token's absolute sequence position and is consulted only in
    // positional mode; in every other mode it is inert, which is what keeps the default path inert too.
    Row row(const int32_t token, const uint64_t position) const {
        switch (mode_) {
            case EmbeddingRowMode::Table:
                return table_(token);  // exactly what the seam produced before this option existed
            case EmbeddingRowMode::FileByToken:
                // layer.cpp:1195-1200 / verify.cpp:944-946 / mtp.cpp:893-895: the file IS the table.
                if (token < 0 || uint64_t(token) >= rows_)
                    throw std::invalid_argument("embedding route: token " + std::to_string(token) +
                                                " is outside the embedding file's " + std::to_string(rows_) +
                                                " rows (the model's vocabulary is " + std::to_string(vocabulary_) + ")");
                return checked(file_(uint64_t(token)));
            case EmbeddingRowMode::FileByPosition:
                // The claimed prefix MUST come from the file; a claimed row never falls back to the table.
                if (position < rows_) {
                    if (position > uint64_t(UINT32_MAX))
                        throw std::invalid_argument("embedding route: the claimed position exceeds the file's row bound");
                    return checked(file_(position));
                }
                return checked(table_(token));
        }
        throw std::logic_error("embedding route: unreachable mode");
    }

    // The [count, hidden] row block a window consumes: the same law, row by row, from base_position.
    std::vector<float> gather(std::span<const int32_t> tokens, const uint64_t base_position) const {
        std::vector<float> out;
        out.reserve(tokens.size() * (mode_ == EmbeddingRowMode::Table ? 1 : hidden_));
        for (size_t index = 0; index < tokens.size(); ++index) {
            const Row row_value = row(tokens[index], base_position + index);
            out.insert(out.end(), row_value.begin(), row_value.end());
        }
        return out;
    }

private:
    EmbeddingRows(const EmbeddingRowMode mode, TableLookup table, FileLookup file, const uint64_t rows,
                  const uint64_t vocabulary, const uint64_t hidden)
        : mode_(mode), table_(std::move(table)), file_(std::move(file)), rows_(rows), vocabulary_(vocabulary), hidden_(hidden) {}

    Row checked(Row value) const {
        if (value.size() != hidden_)
            throw std::invalid_argument("embedding route: a row is " + std::to_string(value.size()) +
                                        " floats wide, the model wants " + std::to_string(hidden_));
        return value;
    }

    EmbeddingRowMode mode_ = EmbeddingRowMode::Table;
    TableLookup table_;
    FileLookup file_;
    uint64_t rows_ = 0, vocabulary_ = 0, hidden_ = 0;
};
}  // namespace strata::hpu
