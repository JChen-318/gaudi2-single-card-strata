#pragma once
#include <cstdint>
namespace strata::hpu::abi {
struct PackedMatmul {int32_t columns,rows,batch,type,activation;};
static_assert(sizeof(PackedMatmul)==20);
// The device Q8_1 activation quantizer: one FP32 activation in, the U8 codes and the per-block
// metadata out, so a projection no longer needs the host to quantize its input first.
struct ActivationQuant {int32_t columns,batch;};
static_assert(sizeof(ActivationQuant)==8);
}
