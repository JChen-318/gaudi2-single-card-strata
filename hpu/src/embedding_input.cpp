#include "strata/hpu/embedding_input.hpp"

// Same header-hygiene as hpu/src/weights.cpp: the pinned upstream reader's snprintf format checker
// warning is a GCC-only artifact of its optimized path, scoped to the include, not weakened.
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat-truncation"
#endif
#include "strata/artifact/gguf_reader.hpp"
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif
#include "mapped_file.hpp"  // detail::local_identity, the port's immutable-file identity
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <limits>
#include <stdexcept>

namespace strata::hpu {
namespace {
constexpr const char* embedding_tensor = "token_embd.weight";

// Upstream prefixes every refusal from this loader with "native embedding: " (src/core/native_head.cpp:125,
// :132, :140); the port's prefix plays the same role and every refusal is a std::invalid_argument.
std::invalid_argument refuse(const std::string& message) { return std::invalid_argument("embedding input: " + message); }
}  // namespace

// The row-count law and the file checks, mirroring NativeEmbed::load (src/core/native_head.cpp:120-153).
EmbeddingInput EmbeddingInput::load(const std::string& path, uint64_t hidden, uint64_t vocabulary) {
try {
    if (!hidden || !vocabulary || hidden > uint64_t(std::numeric_limits<int64_t>::max()) ||
        vocabulary > uint64_t(std::numeric_limits<int64_t>::max()))
        throw refuse("the model hidden width and vocabulary must be positive and below 2^63");
    // The existing reader does the parse; no second GGUF parser here. Unreadable, non-GGUF and
    // non-v3 files throw from its constructor and load() turns those into refusals.
    const strata::GgufFile file(path);
    // Architecture guard, upstream native_head.cpp:123-126: the one-tensor file from
    // tools/embd_bf16_pack.py says "strata-embd" and only its tensor is checked; any other
    // architecture must be one the engine itself accepts.
    const auto* arch = file.get("general.architecture");
    if (!arch || arch->type != strata::MetaType::STRING)
        throw refuse("the file declares no general.architecture string");
    if (arch->s != "strata-embd") {
        const std::string error = strata::check_engine_architecture(file);
        if (!error.empty())
            throw refuse("the file is neither a strata-embd embedding input nor an architecture the engine accepts: " + error);
    }
    const auto* tensor = file.find(embedding_tensor);
    if (!tensor) throw refuse(std::string(embedding_tensor) + " is absent from " + path);
    if (tensor->shape.size() != 2)
        throw refuse(std::string(embedding_tensor) + " is rank " + std::to_string(tensor->shape.size()) +
                     ", a token embedding table is rank 2");
    if (tensor->shape[0] != hidden)
        throw refuse(std::string(embedding_tensor) + "'s hidden width " + std::to_string(tensor->shape[0]) +
                     " is not the model's " + std::to_string(hidden));
    if (tensor->shape[1] == 0)
        throw refuse(std::string(embedding_tensor) + " has zero rows: an embedding input must provide at least one row");
    // Upstream demands shape[1] == n_vocab exactly (native_head.cpp:130); the port admits a smaller
    // table and consumes it positionally (see the header), but never one that claims rows beyond the
    // model's vocabulary.
    if (tensor->shape[1] > vocabulary)
        throw refuse(std::string(embedding_tensor) + "'s row count " + std::to_string(tensor->shape[1]) +
                     " exceeds the model's vocabulary " + std::to_string(vocabulary));
    // Truncation, upstream native_head.cpp:139-142 and GgufModel::in_bounds: the table is copied out
    // of the mapping, so a short shard must be an error, not a read past EOF.
    const uint64_t payload = strata::tensor_payload_bytes(*tensor);
    const uint64_t available = file.file_size() > file.data_start() ? file.file_size() - file.data_start() : 0;
    if (!payload || tensor->offset > available || payload > available - tensor->offset)
        throw refuse(std::string(embedding_tensor) + "'s payload is truncated in " + path + " (needs " +
                     std::to_string(payload) + " B of " + strata::ggml_type_name(tensor->type) + ", has " +
                     std::to_string(available > tensor->offset ? available - tensor->offset : 0) + " B)");
    // Rows are read and dequantized through the port's existing slice machinery, exactly as
    // hpu/src/model.cpp's embedding lookup does today (convert_rows_f32, one row at a time).
    const WeightSource source(path);
    const WeightSlice slice = source.matrix(embedding_tensor);
    if (slice.columns() != tensor->shape[0] || slice.rows() != tensor->shape[1])
        throw refuse(std::string(embedding_tensor) + "'s slice disagrees with its directory entry");
    // The port's declared addition: upstream keeps the raw GGUF blocks and dequantizes per token, so
    // it has no finiteness law; this port materialises rows for host-side consumption, so it refuses
    // a file holding a non-finite value at load, one scratch row at a time (bounded memory).
    std::vector<float> scratch(hidden);
    for (uint64_t row = 0; row < slice.rows(); ++row) {
        slice.convert_rows_f32(row, scratch);
        for (const float value : scratch)
            if (!std::isfinite(value))
                throw refuse(std::string(embedding_tensor) + " row " + std::to_string(row) +
                             " holds a non-finite value");
    }
    EmbeddingInput result;
    result.mode_ = slice.rows() == vocabulary ? EmbeddingInputMode::Vocabulary : EmbeddingInputMode::Prompt;
    result.hidden_ = hidden;
    result.rows_ = slice.rows();
    result.path_ = path;
    // The checkpoint binding upstream gets from `in.embedding = o.embd_gguf` (generate.cpp:8155,
    // conversation_file.cpp:935): the embedding file joins the model identity a session is bound to,
    // in this port's WeightTable::file_identity shape (weights.cpp:178-183).
    result.identity_ = "gguf-local-files-v1|" + detail::local_identity(path, file.file_size());
    if (result.identity_.size() > 65536) throw refuse("file identity exceeds the state header bound");
    result.slice_ = slice;  // keeps the read-only mapping alive
    return result;
} catch (const std::invalid_argument&) {
    throw;
} catch (const std::exception& error) {  // upstream wraps the loader's exceptions the same way (native_head.cpp:196-199)
    throw refuse(error.what());
}
}

std::vector<float> EmbeddingInput::row(const uint64_t index) const {
    if (!active()) throw refuse("the embedding input is inactive; the flag was not given");
    if (index >= rows_) throw std::out_of_range("embedding input row");
    std::vector<float> out(hidden_);
    slice_.convert_rows_f32(index, out);
    return out;
}

std::vector<float> EmbeddingInput::prompt_inputs(std::span<const int32_t> prompt,
                                                 const std::function<std::vector<float>(int32_t)>& table) const {
    std::vector<float> out;
    out.reserve(prompt.size() * hidden_);
    const auto append = [&](const std::vector<float>& value, const bool check_width) {
        if (check_width && value.size() != hidden_)
            throw refuse("an embedding row is " + std::to_string(value.size()) + " floats wide, the model wants " +
                         std::to_string(hidden_));
        out.insert(out.end(), value.begin(), value.end());
    };
    if (mode_ == EmbeddingInputMode::Inactive) {
        // Flag absent: the generation path consumes exactly what the embedding table yields today -
        // byte-identical, with no new width law the token path did not have before.
        if (!table && !prompt.empty()) throw refuse("the generation path supplied no embedding table lookup");
        for (const int32_t token : prompt) append(table(token), false);
        return out;
    }
    if (mode_ == EmbeddingInputMode::Vocabulary) {
        // Upstream semantics (layer.cpp:1195-1200, verify.cpp:944-946): the file IS the embedding
        // table; every prompt row is gathered by token ID and the canonical lookup is not consulted.
        for (const int32_t token : prompt) {
            if (token < 0 || uint64_t(token) >= rows_)
                throw refuse("token ID " + std::to_string(token) + " is outside the embedding file's " +
                             std::to_string(rows_) + " rows");
            append(row(uint64_t(token)), true);
        }
        return out;
    }
    // Prompt mode: the rows replace the embedding-table lookup for the provided rows - the prompt's
    // first `rows` positions, positionally. The count must fit the prompt; it is refused, never
    // silently truncated. Fallback widths are checked as upstream's prompt path does (prefill.cpp:1422).
    if (rows_ > prompt.size())
        throw refuse("the embedding file claims more rows (" + std::to_string(rows_) + ") than the prompt (" +
                     std::to_string(prompt.size()) + " tokens)");
    for (uint64_t at = 0; at < rows_; ++at) append(row(at), true);
    if (rows_ < prompt.size() && !table)
        throw refuse("the generation path supplied no embedding table lookup for the positions past the file's rows");
    for (size_t at = rows_; at < prompt.size(); ++at) append(table(prompt[static_cast<size_t>(at)]), true);
    return out;
}

std::function<std::vector<float>(int32_t)> EmbeddingInput::decode_lookup(
    std::function<std::vector<float>(int32_t)> table) const {
    if (mode_ != EmbeddingInputMode::Vocabulary) return table;  // Inactive and Prompt: the token path keeps the table
    EmbeddingInput input = *this;
    return [input](const int32_t token) {
        if (token < 0 || uint64_t(token) >= input.rows_)
            throw refuse("token ID " + std::to_string(token) + " is outside the embedding file's " +
                         std::to_string(input.rows_) + " rows");
        return input.row(uint64_t(token));
    };
}
}  // namespace strata::hpu
