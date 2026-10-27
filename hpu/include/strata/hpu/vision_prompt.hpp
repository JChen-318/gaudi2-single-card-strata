#pragma once
// --vision's request law: the GENI embedding file, the prompt's per-cell M-RoPE positions and the
// image rows that replace the embedding table.
//
// Upstream parses a GENI request at src/program/generate.cpp:9682-9686/:10635-10845: the file is one or
// more strata-vision records (int32 magic 'SVE1' 0x31455653, n, nx, ny, n_embd, then n x n_embd floats)
// in prompt order; each record's rows go to the prompt's next run of <|image_pad|> tokens. M-RoPE is
// mtmd's: t = p, h = p + y, w = p + x, and text after the image continues at p + max(nx, ny). The table
// is [cells][3] (t, h, w) where cells is the engine's max_context + 64; every rope kernel reads a cell
// index and rotates pair i by the table's (i % 3) stream (include/strata/kernels/mrope.hpp). Cells past
// the prompt are text continuing from the last image. A text request rebuilds the identity table.
//
// This module is the host-side reader/planner only. The refusals carry upstream's exact sentences; the
// caller prints them as "ERR <sentence>". A failed plan leaves the caller to restore the identity table,
// which is what upstream does at :10820-10828.
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "strata/hpu/qsa_state.hpp"

namespace strata::hpu {
// qwen4exp.ple.image_token_id (upstream's kImagePad): the prompt cell whose PLE hash reads 248056.
inline constexpr int32_t kVisionImagePad = 248056;

// Upstream's local fnv1a (src/program/generate.cpp:1502-1506): the conversation-cache picture identity.
uint64_t vision_fnv1a(const void* data, std::size_t bytes, uint64_t seed = 1469598103934665603ull);

// One SVE1 record: n = nx*ny rows of n_embd floats, starting at `row` in the file's row store.
struct VisionRecord {
    uint32_t rows = 0, nx = 0, ny = 0;
    uint64_t row = 0;
};

// An immutable, fully validated SVE1 file. Copies share the row store.
class VisionEmbeddings {
public:
    VisionEmbeddings() = default;
    // Refusals: "cannot open <path>", "bad embeddings file (expected strata-vision records of width N)"
    // (missing/short header, wrong magic, n/nx/ny < 1, nx*ny != n, n_embd != hidden), "short embeddings
    // file" (payload shorter than n*n_embd floats, including a trailing partial record).
    static VisionEmbeddings load(const std::string& path, uint64_t hidden);

    uint64_t rows() const { return rows_; }
    uint64_t hidden() const { return hidden_; }
    const std::vector<VisionRecord>& records() const { return records_; }
    const std::string& path() const { return path_; }
    // Row `index` as FP32, or an out_of_range throw.
    std::span<const float> row(uint64_t index) const;

private:
    uint64_t hidden_ = 0, rows_ = 0;
    std::string path_;
    std::vector<float> data_;
    std::vector<VisionRecord> records_;
};

// One picture placed in the prompt: its run of image-pad cells, its file rows, and the identity the
// conversation cache compares (the grid, then the server's img_keys[k] when the key count matches, else
// the embedding bytes) exactly as upstream builds it at :10774-10786.
struct VisionImage {
    uint32_t start = 0, rows = 0, nx = 0, ny = 0;
    uint64_t row = 0, key = 0;
};

// The request's position table and per-cell image rows. `positions` is [cells][3] flattened (t, h, w);
// `rows` has one entry per prompt cell (an index into VisionEmbeddings, or -1 for a table lookup).
struct VisionPlan {
    uint64_t cells = 0;
    std::vector<int32_t> positions;
    std::vector<int64_t> rows;
    std::vector<VisionImage> images;

    // The rotary position pair `pair` rotates at for `cell` (pair % 3: 0 t, 1 h, 2 w).
    int32_t position(uint64_t cell, uint32_t pair) const;
    // The image row claimed by prompt cell `cell`, or -1 (text / decoded token).
    int64_t image_row(uint64_t cell) const { return cell < rows.size() ? rows[cell] : -1; }
};

// Upstream's non-GENI branch: every cell rotates at its own index, no image rows.
VisionPlan plan_text_prompt(uint64_t cells, uint64_t prompt_cells);

// The GENI branch. Refusals (upstream's sentences): "the prompt has more images than the embeddings
// file", "image k has n rows but fewer <|image_pad|> tokens", "the embeddings file has more images than
// the prompt", "the prompt cannot end in an image". A prompt longer than the table is refused before
// any write (upstream has no such request: its table is max_context + 64 and the context check follows).
VisionPlan plan_vision_prompt(std::span<const int32_t> ids, const VisionEmbeddings& embeddings,
                              uint64_t cells, std::span<const uint64_t> img_keys = {});

}  // namespace strata::hpu
