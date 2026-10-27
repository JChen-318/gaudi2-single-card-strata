#include "strata/hpu/vision_prompt.hpp"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace strata::hpu {
namespace {
constexpr int32_t kSve1Magic = 0x31455653;  // 'SVE1'
constexpr uint64_t kFnvOffset = 1469598103934665603ull, kFnvPrime = 1099511628211ull;
}

uint64_t vision_fnv1a(const void* data, std::size_t bytes, uint64_t seed) {
    const auto* p = static_cast<const uint8_t*>(data);
    for (std::size_t i = 0; i < bytes; ++i) { seed ^= p[i]; seed *= kFnvPrime; }
    return seed;
}

VisionEmbeddings VisionEmbeddings::load(const std::string& path, uint64_t hidden) {
    if (!hidden) throw std::invalid_argument("vision embeddings need a positive width");
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) throw std::invalid_argument("cannot open " + path);
    const auto end = in.tellg();
    if (end < 0) throw std::invalid_argument("cannot open " + path);
    const auto size = uint64_t(end);
    std::vector<uint8_t> raw(static_cast<std::size_t>(size));
    in.seekg(0);
    if (size && !in.read(reinterpret_cast<char*>(raw.data()), std::streamsize(size))) throw std::invalid_argument("cannot open " + path);

    VisionEmbeddings out;
    out.hidden_ = hidden;
    out.path_ = path;
    std::size_t at = 0;
    const auto bad = [&] {
        return std::invalid_argument("bad embeddings file (expected strata-vision records of width " + std::to_string(hidden) + ")");
    };
    while (at < raw.size()) {
        if (raw.size() - at < sizeof(int32_t) * 5) throw bad();
        int32_t hdr[5];
        std::memcpy(hdr, raw.data() + at, sizeof hdr);
        at += sizeof hdr;
        const int64_t rows = hdr[1], nx = hdr[2], ny = hdr[3];
        if (hdr[0] != kSve1Magic || rows < 1 || nx < 1 || ny < 1 || nx * ny != rows || hdr[4] < 0 ||
            uint64_t(hdr[4]) != hidden)
            throw bad();
        const auto count = uint64_t(rows) * hidden;
        const auto bytes = count * sizeof(float);
        if (count / hidden != uint64_t(rows) || bytes / sizeof(float) != count || bytes > raw.size() - at)
            throw std::invalid_argument("short embeddings file");
        const auto first = out.rows_;
        out.data_.resize(std::size_t(out.rows_ + uint64_t(rows)) * hidden);
        std::memcpy(out.data_.data() + std::size_t(first * hidden), raw.data() + at, std::size_t(bytes));
        out.rows_ += uint64_t(rows);
        out.records_.push_back({uint32_t(rows), uint32_t(nx), uint32_t(ny), first});
        at += std::size_t(bytes);
    }
    return out;
}

std::span<const float> VisionEmbeddings::row(uint64_t index) const {
    if (index >= rows_) throw std::out_of_range("vision embedding row out of range");
    return std::span<const float>(data_).subspan(std::size_t(index * hidden_), std::size_t(hidden_));
}

int32_t VisionPlan::position(uint64_t cell, uint32_t pair) const {
    if (cell >= cells) throw std::out_of_range("vision position cell out of range");
    return positions[std::size_t(cell * 3 + pair % 3)];
}

namespace {
void put_position(std::vector<int32_t>& table, uint64_t cells, int64_t cell, int64_t t, int64_t h, int64_t w) {
    if (cell < 0 || uint64_t(cell) >= cells) throw std::logic_error("image position cell escaped the table");
    for (const auto value : {t, h, w})
        if (value < 0 || value > int64_t(INT32_MAX)) throw std::runtime_error("image position exceeds the int32 table");
    const auto at = std::size_t(uint64_t(cell) * 3);
    table[at] = int32_t(t);
    table[at + 1] = int32_t(h);
    table[at + 2] = int32_t(w);
}
}  // namespace

VisionPlan plan_text_prompt(uint64_t cells, uint64_t prompt_cells) {
    if (prompt_cells > cells) throw std::invalid_argument("prompt exceeds the image position table");
    VisionPlan plan;
    plan.cells = cells;
    plan.positions.resize(std::size_t(cells * 3));
    for (uint64_t cell = 0; cell < cells; ++cell) put_position(plan.positions, cells, int64_t(cell), int64_t(cell), int64_t(cell), int64_t(cell));
    plan.rows.assign(std::size_t(prompt_cells), -1);
    return plan;
}

VisionPlan plan_vision_prompt(std::span<const int32_t> ids, const VisionEmbeddings& embeddings, uint64_t cells,
                              std::span<const uint64_t> img_keys) {
    const auto n = int64_t(ids.size());
    if (uint64_t(n) > cells) throw std::invalid_argument("prompt exceeds the image position table");
    std::vector<VisionImage> imgs;
    imgs.reserve(embeddings.records().size());
    for (const auto& record : embeddings.records())
        imgs.push_back({0, record.rows, record.nx, record.ny, record.row, 0});

    VisionPlan plan;
    plan.cells = cells;
    plan.positions.resize(std::size_t(cells * 3));
    plan.rows.assign(std::size_t(n), -1);
    int64_t p = 0, i = 0;
    std::size_t k = 0;
    while (i < n) {
        if (ids[std::size_t(i)] != kVisionImagePad) {
            put_position(plan.positions, cells, i, p, p, p);
            ++p;
            ++i;
            continue;
        }
        if (k >= imgs.size()) throw std::invalid_argument("the prompt has more images than the embeddings file");
        auto& im = imgs[k++];
        // The conversation cache's picture identity: the grid, then the server's key when one was
        // supplied per record, else the embedding bytes (src/program/generate.cpp:10777-10786).
        const int64_t grid[3] = {int64_t(im.rows), int64_t(im.nx), int64_t(im.ny)};
        uint64_t key = vision_fnv1a(grid, sizeof grid);
        if (img_keys.size() == imgs.size())
            key = vision_fnv1a(&img_keys[k - 1], sizeof(uint64_t), key);
        else {
            const auto first = embeddings.row(im.row);
            key = vision_fnv1a(first.data(), std::size_t(im.rows) * std::size_t(embeddings.hidden()) * sizeof(float), key);
        }
        im.start = uint32_t(i);
        im.key = key;
        plan.images.push_back(im);
        for (uint32_t j = 0; j < im.rows; ++j)
            if (i + int64_t(j) >= n || ids[std::size_t(i + j)] != kVisionImagePad)
                throw std::invalid_argument("image " + std::to_string(k) + " has " + std::to_string(im.rows) +
                                            " rows but fewer <|image_pad|> tokens");
        for (uint32_t j = 0; j < im.rows; ++j) {
            const uint32_t y = j / im.nx, x = j % im.nx;
            put_position(plan.positions, cells, i + int64_t(j), p, p + int64_t(y), p + int64_t(x));
            plan.rows[std::size_t(i + j)] = int64_t(im.row + j);
        }
        i += int64_t(im.rows);
        p += std::max(im.nx, im.ny);
    }
    if (k != imgs.size()) throw std::invalid_argument("the embeddings file has more images than the prompt");
    if (n > 0 && ids[std::size_t(n - 1)] == kVisionImagePad) throw std::invalid_argument("the prompt cannot end in an image");
    for (int64_t c = n; c < int64_t(cells); ++c) put_position(plan.positions, cells, c, p + (c - n), p + (c - n), p + (c - n));
    return plan;
}

}  // namespace strata::hpu
