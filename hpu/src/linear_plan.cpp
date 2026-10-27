#include "strata/hpu/linear_plan.hpp"
#include "../tpc/packed_codebook_layout.inc"
#include "strata/hpu/packed_quant.hpp"
#include "strata/hpu/tensor.hpp"
#include "strata/hpu/window_limits.hpp"
namespace strata::hpu {
uint64_t linear_pinned_bytes(uint32_t columns,uint32_t rows,uint32_t batch,uint32_t type,PackedActivation mode,bool device_quantizer) {
    if(!columns || columns>65536 || !rows || rows>1048576 || !batch || batch>kResidentWindowRows)
        throw std::invalid_argument("invalid linear slot budget geometry");
    uint64_t input;
    if(packed_activation_supported(type,mode)) {
        (void)packed_row_bytes(type,columns);
        input=checked_product(batch,columns)+checked_product(checked_product(batch,columns/32),8)+STRATA_PACKED_CODEBOOK_BYTES;
        // The quantizer reads an FP32 activation, so that slot only exists in the graph that
        // contains it. The codes and metadata it writes were already counted: the dot reads them
        // either way. Callers must pass the flag matching the graph they build; head_window_graph
        // checks the budget against the real slot list for both shapes.
        if(device_quantizer) input=checked_product(batch,columns)*4+input;
    } else input=checked_product(checked_product(batch,columns),2);
    return input+checked_product(checked_product(batch,rows),4);
}
}
