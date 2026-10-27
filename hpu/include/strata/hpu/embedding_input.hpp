#pragma once
// strata/hpu/embedding_input.hpp - the upstream `--embd-gguf` for this backend.
//
// Upstream (src/program/generate.cpp:397 field, :1972 parse, :3032-3037 load and install, :8155 session
// binding; src/core/native_head.cpp:120-153 loader) takes a GGUF holding `token_embd.weight` in the form
// the checkpoint ships it (tools/embd_bf16_pack.py: one tensor, ne = [n_embd, n_vocab], BF16, architecture
// "strata-embd") and uses it instead of the canonical embedding table for every row it covers: `embed_row`
// (src/core/layer.cpp:1195-1200) gathers one row by token ID, the verify window (src/core/verify.cpp:944-946)
// and the MTP drafter (src/core/mtp.cpp:893-895) gather the prompt's rows by token ID. Shape and length
// mismatches are refused at load (src/core/native_head.cpp:130-135 absent/shape/type, :139-142 truncated
// payload), and the file joins the checkpoint's model identity (src/core/conversation_file.cpp:935).
//
// This module is the port's loader and the seam's row source; the generation path consumes it through
// prompt_inputs() and decode_lookup(), whose callback type matches SessionOperations::embedding
// (strata/hpu/session_flow.hpp:14). Two consumption modes, resolved by the file's row count against the
// model's vocabulary:
//   * Vocabulary (rows == n_vocab): exactly upstream - the file replaces the embedding-table lookup for
//     every token ID, prompt rows and decoded tokens alike, and the table callback is not consulted.
//   * Prompt (0 < rows < n_vocab): a declared extension upstream has no analogue - the file's rows are
//     consumed POSITIONALLY and replace the embedding-table lookup for the provided rows, the prompt's
//     first `rows` positions; later prompt positions and decoded tokens keep the table. A file with more
//     rows than the prompt's length is refused (that is what bounds this mode), never silently truncated.
// Absent flag: an inactive input whose prompt_inputs() is the plain table lookup, so the default path is
// byte-identical to today. Reading uses the port's existing GGUF facilities only (strata::GgufFile for
// metadata, WeightSource/WeightSlice for the rows); no second GGUF parser or dequantizer is invented.
#include <cstdint>
#include <functional>
#include <span>
#include <string>
#include <vector>

#include "strata/hpu/weights.hpp"

namespace strata::hpu {
enum class EmbeddingInputMode { Inactive, Vocabulary, Prompt };

// An immutable embedding row file. Copies share one read-only mapping.
class EmbeddingInput {
public:
    EmbeddingInput() = default;  // inactive: flag absent, the embedding table is used as today

    // The parsing entry point for a `--embd-gguf PATH` value: opens the GGUF and refuses, with
    // std::invalid_argument, every file upstream's loader refuses plus this port's row-count law:
    // no general.architecture or one that is neither "strata-embd" nor an architecture the engine
    // accepts (upstream native_head.cpp:123-126); `token_embd.weight` absent or not rank 2, a hidden
    // width (GGUF shape[0]) other than the model's n_embd, or zero rows (upstream :130-135); more rows
    // than the model's vocabulary (the port's generalisation of upstream's exact shape[1] == n_vocab,
    // which the Vocabulary mode below still enforces as the consumption law); a payload truncated or
    // shorter than rows x hidden in the file's blocks (upstream :139-142); a type the port cannot
    // convert; or any non-finite value in any row (a declared port addition: upstream keeps the raw
    // blocks and dequantizes per token, this port materialises and checks once at load).
    static EmbeddingInput load(const std::string& path, uint64_t hidden, uint64_t vocabulary);
    static EmbeddingInput load(const std::string& path, const core::ModelGeometry& model) {
        return load(path, uint64_t(model.n_embd), uint64_t(model.n_vocab));
    }

    bool active() const { return mode_ != EmbeddingInputMode::Inactive; }
    EmbeddingInputMode mode() const { return mode_; }
    uint64_t rows() const { return rows_; }
    uint64_t hidden() const { return hidden_; }
    const std::string& path() const { return path_; }
    // One row as FP32, by file row index (token ID in Vocabulary mode, position in Prompt mode).
    std::vector<float> row(uint64_t index) const;
    // The local immutable-file identity to bind checkpoints with, mirroring upstream's
    // `in.embedding = o.embd_gguf` session fingerprint entry (generate.cpp:8155,
    // conversation_file.cpp:935) in this port's WeightTable::file_identity style.
    const std::string& file_identity() const { return identity_; }

    // The [T, hidden] embedding inputs the generation path consumes for `prompt`, replacing the
    // embedding-table lookup for the provided rows. `table` is the canonical per-token lookup (the
    // existing SessionOperations::embedding); it may be empty when nothing falls back to it.
    // Refusals: a Prompt-mode file with more rows than the prompt's length, a Vocabulary-mode token
    // ID outside the table's rows, a missing fallback callback where one is needed, or a fallback
    // row of the wrong width. Inactive returns exactly what `table` yields today, per token.
    std::vector<float> prompt_inputs(std::span<const int32_t> prompt,
                                     const std::function<std::vector<float>(int32_t)>& table) const;

    // The token-path (decode) seam: a SessionOperations::embedding-shaped lookup. Vocabulary mode
    // gathers from the file for every token ID, like upstream's embed_row; Prompt and Inactive keep
    // `table` untouched, because those rows were only provided for the prompt.
    std::function<std::vector<float>(int32_t)> decode_lookup(std::function<std::vector<float>(int32_t)> table) const;

private:
    EmbeddingInputMode mode_ = EmbeddingInputMode::Inactive;
    uint64_t hidden_ = 0, rows_ = 0;
    std::string path_, identity_;
    WeightSlice slice_;  // keeps the read-only mapping alive
};

}  // namespace strata::hpu
