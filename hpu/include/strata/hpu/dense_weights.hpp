#pragma once
#include "strata/hpu/weight_transfer.hpp"
#include "strata/hpu/recipe.hpp"

namespace strata::hpu {
// Startup-only immutable dense weights. Enforces a payload limit before each
// allocation and bounded pinned staging. Does not count recipe/state/KV/driver
// overhead; the model planner must reserve those separately from this budget.
class DenseWeights {
public:
    DenseWeights(const Device&,uint64_t payload_budget,uint64_t staging_budget=1024*1024);
    ~DenseWeights();
    DenseWeights(const DenseWeights&)=delete;
    DenseWeights& operator=(const DenseWeights&)=delete;
    void bf16(const std::string& binding,const WeightSlice&,uint64_t columns,uint64_t rows);
    void bf16_subset(const std::string& binding,const WeightSlice&,std::span<const int32_t> source_rows);
    // Preserve Q2_0/Q4_0/Q8_0 bytes, with optional ordered subset. Uploads use
    // two bounded pinned slots, never materialize/dequantize a full host matrix.
    void packed(const std::string& binding,const WeightSlice&,std::span<const int32_t> source_rows={});
    void requantized_q4(const std::string& binding,const WeightSlice&,std::span<const int32_t> source_rows={});
    void f32(const std::string& binding,std::span<const float>);
    // Validates exact GGUF axis order before conversion; constants stay F32.
    void matrix(const std::string& binding,const WeightTable&,const std::string& tensor,uint64_t columns,uint64_t rows);
    void constant(const std::string& binding,const WeightTable&,const std::string& tensor,uint64_t elements);
    void transposed(const std::string& binding,const WeightTable&,const std::string& tensor,uint64_t source_columns,uint64_t source_rows);
    // Moves the BF16 matrices `parts` (one column count) back to back into one buffer `name` [their rows, columns]
    // and frees theirs; binding(part) then addresses its rows inside it. No extra device memory is kept.
    void fuse(const std::string& name,std::initializer_list<std::string> parts);
    // Adds an FP8 (fp8_143) copy of the BF16 matrix `name` [rows, columns] with one F32 scale per row,
    // max|row|/240: name+"_f8" (F8) and name+"_f8_scale" (F32 [1, rows]). The BF16 matrix and every binding
    // already taken stay valid (the graphs that are not FP8 keep reading it).
    void to_fp8(const std::string& name);
    bool contains(const std::string& name) const;
    std::vector<Binding> bindings() const;
    Binding binding(const std::string&) const;
    uint64_t payload_bytes() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
