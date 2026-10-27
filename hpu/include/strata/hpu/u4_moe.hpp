#pragma once
// STRATA_HPU_RESIDENT_U4: the grouped-MoE MME path over the resident U4 store (strata/hpu/u4_resident.hpp).
// A prefill window's routed slots are bucketed per expert, the bucket gathers the activation into the MME's
// [batch=(expert,group)][rows][group] A operand, one batch GEMM per projection decompresses every expert's U4
// codes, and a fused epilogue applies the per-channel affine, SiLU(gate)*up and the route weight. The shared
// expert is the store's last block, driven by the same four nodes at batch=groups. One read of the layer's
// weights per window instead of one read per slot.
#include "strata/hpu/recipe.hpp"
#include "strata/hpu/resident_experts.hpp"
#include "strata/hpu/u4_resident.hpp"
#include "strata/hpu/weights.hpp"
#include <cstdint>
#include <vector>

namespace strata::hpu {
// Fixed 64-row per-expert bucket. Overflow must be checked and replayed; total
// capacity alone does not bound the number of rows routed to a hot expert.
uint32_t u4_bucket_rows(uint32_t rows,uint32_t top_k,uint32_t experts);
// The FFN half of one window, dispatching on the selected resident store format.
TensorId define_ffn_moe(GraphBuilder&,const core::ModelGeometry&,TensorId x,TensorId logits,TensorId shared_gate,
                        uint32_t rows=1,bool unreduced=false);
// The U4 graph itself; `store` must be U4LayerStore::make(geometry.n_expert, n_embd, n_ff).
TensorId define_resident_u4_moe(GraphBuilder&,const core::ModelGeometry&,const U4LayerStore&,TensorId x,
                                TensorId logits,TensorId shared_gate,uint32_t rows=1,bool unreduced=false);
// One layer's U4 store in HBM: one buffer holding all six regions (routed experts then the shared block), the
// shared expert's static row-order map. Bindings are
// "gu_codes", "gu_scale", "gu_offset", "down_codes", "down_scale", "down_offset" (routed prefix of each region)
// and "shared_gu_codes"/"shared_gu_scale"/"shared_gu_offset"/"shared_down_codes"/"shared_down_scale"/
// "shared_down_offset" (the last block), plus "shared_map". 13 external bindings;
// place is a graph-internal tensor and raw-code scales are graph constants.
class U4LayerExperts:public ResidentExpertStore {
public:
    U4LayerExperts(const Device&,const WeightTable&,uint32_t layer,unsigned threads);
    std::vector<Binding> bindings() const override;
    const U4LayerStore& store() const {return store_;}
    uint64_t device_bytes() const override {return layer_.size()+maps_.size();}
private:
    U4LayerStore store_;
    DeviceBuffer layer_,maps_;
};
}
