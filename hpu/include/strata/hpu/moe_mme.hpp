#pragma once
// STRATA_HPU_MOE_MME: the prefill window's MoE as grouped MME GEMMs over the resident IQ4 store
// (ResidentLayerExperts, resident_experts.hpp), instead of one GEMV read of an expert's weights per routed slot.
// The window's routed slots are sorted by expert into tiles of `tile_rows` rows: expert e owns ceil(rows_e /
// tile_rows) consecutive tiles, so the tile count is at most slots / tile_rows + experts whatever the routing (a
// skewed window, one expert taking most rows, only fills more of that expert's tiles; nothing is ever dropped).
// Per window and layer every used tile gets its expert's IQ4 rows expanded once to BF16 (strata_iq4_expand_tiles),
// the activation rows are gathered into [tile][tile row][hidden] (strata_moe_mme_gather), and two batch GEMMs with
// SiLU(gate)*up between them give each slot's expert row; the shared expert is the same chain at batch 1 over all
// window rows. Not bit-identical to the GEMV path: activations are BF16, not Q8, and each weight kv*scale is
// rounded to BF16 (the scale too).
#include "strata/hpu/recipe.hpp"
#include "strata/core/geometry.hpp"
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace strata::hpu {
// Rows per tile (STRATA_HPU_MOE_MME_TILE overrides: 32, 64 or 128; default 64).
uint32_t moe_mme_bucket_rows(uint32_t rows,const core::ModelGeometry&);
// The tile capacity of a window: enough for any routing.
uint32_t moe_mme_tiles(uint32_t rows,uint32_t tile_rows,const core::ModelGeometry&);
// Same contract as define_resident_moe (bindings "table", "w_gu", "w_down", "s_gu", "s_down"; returns the expert
// block [rows, hidden], or with `unreduced` the per-slot rows "expert_rows" [slots, hidden] and "route_weights").
// The graph reads the persistent constants moe_mme_constants() names (under the current scope); the caller sets them.
TensorId define_resident_mme_moe(GraphBuilder&,const core::ModelGeometry&,TensorId x,TensorId logits,
                                 TensorId shared_gate,uint32_t rows,bool unreduced,uint32_t tile_rows);
std::vector<std::pair<std::string,std::vector<float>>> moe_mme_constants(const core::ModelGeometry&);
}
