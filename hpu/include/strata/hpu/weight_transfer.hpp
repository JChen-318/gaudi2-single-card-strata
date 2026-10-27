#pragma once
#include "strata/hpu/weights.hpp"
#include "strata/hpu/runtime.hpp"

namespace strata::hpu {
// Synchronous immutable packed copy, shared by dense and expert owners. Exact
// selected rows, two bounded staging slots, wait/drain before returning/freeing.
// Optional Q4 uses one reusable float row plus the pinned slots, not a matrix.
DeviceBuffer upload_packed_weights(const Device&,const WeightSlice&,const Stream&,uint64_t staging_budget,
                                   std::span<const int32_t> source_rows={},bool requantize_q4=false);
// Cold-load one mapped matrix/expert slice into BF16 HBM. Owns the HBM result,
// two bounded pinned staging slots and transfer completion event. Conversion
// overlaps already queued H2D copies; the host waits only before reusing a slot.
// Construction may wait for earlier copies. Consumers must wait(completion()).
// This reference path does not preserve ggml's CPU activation quantization.
class Bf16WeightUpload {
public:
    Bf16WeightUpload(const Device&, const WeightSlice&, const Stream& copy_stream,
                     uint64_t pinned_budget_bytes = 1024 * 1024);
    // Nonempty IDs gather source rows in that order; empty means all rows.
    Bf16WeightUpload(const Device&,const WeightSlice&,std::span<const int32_t> source_rows,const Stream& copy_stream,
                    uint64_t pinned_budget_bytes=1024*1024);
    ~Bf16WeightUpload();
    Bf16WeightUpload(const Bf16WeightUpload&) = delete;
    Bf16WeightUpload& operator=(const Bf16WeightUpload&) = delete;
    const DeviceBuffer& buffer() const;
    const Event& completion() const;
    uint64_t pinned_bytes() const;
    uint64_t transfer_count() const;
    // Wait for upload completion, release both pinned staging slots and transfer
    // unique HBM ownership to a resident cache. This object is consumed once.
    DeviceBuffer finish();
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace strata::hpu
