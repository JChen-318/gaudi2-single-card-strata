#pragma once

#include "strata/hpu/tensor.hpp"
#include "strata/core/geometry.hpp"
#include <memory>
#include <optional>
#include <span>
#include <utility>

namespace strata::hpu {
namespace detail { struct MappedWeights; }
struct WeightInfo {
    std::string name;
    uint32_t ggml_type = 0;
    // Original GGUF axis order: columns, rows, experts (axis 0 is contiguous).
    std::vector<uint64_t> gguf_shape;
    uint64_t source_bytes = 0;
    bool bf16_conversion_supported = false;
};
struct WeightLocation {std::string path;uint64_t offset=0,file_bytes=0;};

class WeightSlice {
public:
    // Checked conventional row storage from a non-GGUF source, e.g. MTP
    // dense.bin or one losslessly relaid expert. Owner must retain these bytes.
    static WeightSlice raw_rows(std::shared_ptr<const void> owner,std::span<const uint8_t>,uint32_t ggml_type,uint64_t columns,uint64_t rows);
    uint64_t columns() const { return columns_; }
    uint64_t rows() const { return rows_; }
    uint64_t source_bytes() const { return source_.size(); }
    uint32_t ggml_type() const {return type_;}
    uint64_t row_bytes() const {return row_bytes_;}
    // Read-only raw rows. Keep this owning slice alive while the span is used.
    std::span<const uint8_t> bytes() const {return source_;}
    // Row-major [rows,columns], no implicit transpose. Only output.size() BF16
    // elements and one <=256-float quant block are materialized in host RAM.
    void convert_rows(uint64_t first_row, std::span<uint16_t> output) const;
    void convert_rows_f32(uint64_t first_row, std::span<float> output) const;
    // Requantize supported mapped formats into Q4_0, one decoded float row
    // at a time. Destination contains complete rows in source order.
    void convert_rows_q4(uint64_t first_row,std::span<uint8_t> output,std::span<float> scratch={}) const;
private:
    void convert_impl(uint64_t first_row, void* output, uint64_t elements, bool bf16) const;
    std::shared_ptr<const void> owner_;
    std::span<const uint8_t> source_;
    uint64_t columns_ = 0, rows_ = 0, row_bytes_ = 0;
    uint32_t type_ = 0;
    friend class WeightSource;
};

class WeightSource {
public:
    // Read-only mmap. No prefault, pinning or full-model conversion.
    explicit WeightSource(const std::string& path);
    std::vector<WeightInfo> inventory() const;
    WeightSlice matrix(const std::string& name) const;
    WeightSlice vector(const std::string& name) const;
    // GGUF [K,N,E] => expert slice [N,K], without copying adjacent experts.
    WeightSlice expert(const std::string& name, uint64_t expert_id) const;
private:
    explicit WeightSource(std::shared_ptr<const detail::MappedWeights> owner):owner_(std::move(owner)) {}
    WeightSlice slice(const std::string&, uint64_t, unsigned rank) const;
    std::shared_ptr<const detail::MappedWeights> owner_;
    friend class WeightTable;
};
// Complete split-family catalog. Opens any correctly named shard, validates
// the split metadata, rejects duplicate tensor names, validates qwen4exp
// geometry against this port's target and resolves each tensor to its shard.
// Read-only mappings are shared with slices; no second mapping or full decode.
class WeightTable {
public:
    explicit WeightTable(const std::string& any_shard);
    explicit WeightTable(const std::vector<std::string>& shards_in_split_order);
    const core::ModelGeometry& geometry() const;
    uint32_t shard_count() const;
    std::vector<WeightInfo> inventory() const;
    const WeightInfo& info(const std::string&) const;
    uint32_t shard(const std::string&) const;
    std::string tensor_path(const std::string&) const;
    WeightLocation location(const std::string&) const;
    std::vector<std::pair<std::string,uint64_t>> shard_files() const;
    // Exact integer metadata, never routed through double. Missing keys return
    // nullopt; wrong types, negative values or arrays >64 are rejected.
    std::optional<std::vector<uint64_t>> integers(const std::string& key) const;
    std::optional<double> number(const std::string& key) const;
    std::optional<std::string> text(const std::string& key) const;
    // Local immutable-file identity: canonical path, inode/device, size and
    // nanosecond mtime for every shard. This is not a content SHA256.
    std::string file_identity() const;
    WeightSlice matrix(const std::string&) const;
    WeightSlice vector(const std::string&) const;
    WeightSlice expert(const std::string&,uint64_t expert_id) const;
    // Explicit small constant materialization (rank 1/2); no implicit 1+w fold
    // or transpose. Shape and a maximum element count are checked first.
    std::vector<float> constant(const std::string&,uint64_t elements,uint64_t max_elements=1048576) const;
private:
    struct Impl;
    std::shared_ptr<const Impl> impl_;
};
// Explicit GGUF row-major [rows,columns] -> HPU [columns,rows], retaining
// FP32 constants. Destination is caller-owned; temporary storage is bounded.
void transpose_f32(const WeightSlice&, std::span<float> output, uint64_t scratch_budget_bytes=64*1024);
} // namespace strata::hpu
