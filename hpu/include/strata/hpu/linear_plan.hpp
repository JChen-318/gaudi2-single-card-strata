#pragma once
#include <cstdint>
#include "strata/hpu/packed_quant.hpp"
namespace strata::hpu {
// Exact persistent non-weight slots of a linear graph; weights/workspace are
// separate. The runtime also enforces the process-wide pinned/HBM limits.
// The activation contract decides whether the input slot is packed int8 codes
// plus per-32-value metadata or an ordinary BF16 row set.
// device_quantizer must match the graph: the FP32 activation slot exists only when the recipe
// really contains the quantizer kernel.
uint64_t linear_pinned_bytes(uint32_t columns,uint32_t rows,uint32_t batch,uint32_t type,PackedActivation mode=PackedActivation::Q8_1,bool device_quantizer=false);
}
