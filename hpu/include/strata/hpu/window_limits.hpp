#pragma once
#include <cstdint>
#include "strata/hpu/model_abi.hpp"
namespace strata::hpu {
// Resident fused graphs retain a state per row; generic causal executors keep
// their independent8-row cap. Memory admission prices states beyond8 explicitly.
// Resident kernels have a separate cap from generic causal executors. Admission
// must price actual QSA chunk slabs, full-window FFN and GDN state/workspace.
// The IQ4 path still issues slot GEMVs; a wider window alone does not guarantee
// one physical weight read per expert.
inline constexpr uint32_t kResidentWindowRows=abi::kMaxWindowRows;
}
