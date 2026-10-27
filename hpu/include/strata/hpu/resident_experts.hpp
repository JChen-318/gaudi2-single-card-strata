#pragma once
#include "strata/hpu/recipe.hpp"
#include "strata/hpu/runtime.hpp"
#include "strata/hpu/weights.hpp"
#include "strata/hpu/model_abi.hpp"
#include <array>
#include <span>

namespace strata::hpu {
// Direct per-slot kernels have no fixed routing slab. The separate shared-read
// pairing kernel still has its original 128-slot limit.
inline constexpr uint32_t kResidentMoeSlots=abi::kMaxMoeSlots;
// STRATA_HPU_RESIDENT_IQ4 (strata/hpu/performance_modes.hpp): every expert of a layer resident in HBM in the
// layout tpc/iq4q8_gemv.c reads. A GGUF IQ4_NL (20) or IQ4_XS (23) matrix [rows, columns] becomes 4-bit codebook
// codes plus one F32 scale per row and 32 columns (d, or d * (ls - 32) for IQ4_XS: exact in F32), so
// kv[code] * scale reproduces the GGUF dequantizer bit for bit. Rows come in groups of five 256-row tiles:
//   codes  at (((g*blocks + b)*16 + c/2)*5 + t)*256 + L: column c (low nibble) and c+1 (high nibble) of row
//          (5g+t)*256 + 64*(L%4) + L/4;
//   scales at codes_bytes + (((g*blocks + b)*5 + t)*256 + i)*4: row (5g+t)*256 + i of block b.
// rows must be a multiple of 1280 and columns of 32 (256 for IQ4_XS).
uint64_t iq4_resident_bytes(uint64_t rows,uint64_t columns);
// Writes `parts` stacked row-wise (e.g. gate then up) into out[0, iq4_resident_bytes(total rows, columns)).
void pack_iq4_resident(std::span<const WeightSlice> parts,std::span<uint8_t> out);
// The IQ4 codebook as the kernel's shuffle table: kv[i & 15] at byte i of 256.
std::array<uint8_t,256> iq4_codebook_table();
// Why `table` cannot run resident (empty when it can): routed and shared gate/up must be IQ4_XS or IQ4_NL,
// down IQ4_NL or IQ4_XS, with the geometry the kernel tiles.
std::string resident_iq4_unsupported(const WeightTable& table);
// Why `table` cannot run the resident U4 store (empty when it can): the six-region layout needs 128-byte-aligned
// blocks, and every routed and shared gate/up/down matrix must be IQ4_NL or IQ4_XS.
std::string resident_u4_unsupported(const WeightTable& table);

// The device FFN half of one decode row in the current graph scope: router logits [1, experts] and shared gate
// [1, 1] -> strata_route_topk_f32 -> Q8 x -> gate/up GEMV -> silu(gate)*up -> Q8 -> down GEMV -> weighted sum.
// Declares the persistent "table", "w_gu", "w_down", "s_gu", "s_down" (ResidentLayerExperts::bindings()) and
// returns the expert block [hidden] for the caller's HC write. x is F32 [1, hidden].
// rows > 1 (target verification): x [rows, hidden], logits [rows, experts], shared gate [rows, 1]; block [rows, hidden].
// unreduced: returns the slots' expert rows "expert_rows" [rows*(top_k+1), h] before the route weighting ("route_weights"
// [slots]) and the per-row sum, for a consumer that folds both in (STRATA_HPU_HC_FUSED's HC write).
// x_codes/x_scales: x already quantized as strata_q8_quant_packed_f32 would (STRATA_HPU_HC_DECODE_TPC's FFN read).
TensorId define_resident_moe(GraphBuilder&,const core::ModelGeometry&,TensorId x,TensorId logits,TensorId shared_gate,uint32_t rows=1,bool unreduced=false,
                             TensorId x_codes=kNoTensor,TensorId x_scales=kNoTensor);
// One layer's resident expert store: the bindings its FFN graph declares (names without the "moe." scope) and
// the HBM it holds. Two layouts implement it: IQ4 codebook rows (ResidentLayerExperts) and the U4 grouped-MoE
// store (U4LayerExperts, u4_moe.hpp).
class ResidentExpertStore {
public:
    virtual ~ResidentExpertStore()=default;
    virtual std::vector<Binding> bindings() const=0;
    virtual uint64_t device_bytes() const=0;
};
// One layer's experts: all routed experts as rows of one buffer per projection, and the shared expert.
// Bindings: "w_gu" {experts, gu_bytes}, "w_down" {experts, down_bytes}, "s_gu" {1, gu_bytes}, "s_down"
// {1, down_bytes}, "table" (the codebook, one copy per device).
class ResidentLayerExperts:public ResidentExpertStore {
public:
    ResidentLayerExperts(const Device&,const WeightTable&,uint32_t layer,unsigned threads);
    std::vector<Binding> bindings() const override;
    uint64_t gate_up_bytes() const {return gu_bytes_;}
    uint64_t down_bytes() const {return down_bytes_;}
    uint64_t device_bytes() const override {return gu_.size()+down_.size()+shared_gu_.size()+shared_down_.size();}
private:
    uint64_t gu_bytes_,down_bytes_;
    DeviceBuffer gu_,down_,shared_gu_,shared_down_;
    std::shared_ptr<const DeviceBuffer> table_;
};
}
