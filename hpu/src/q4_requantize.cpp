#include "strata/hpu/q4_requantize.hpp"
#include <ggml.h>
#include <cmath>
#include <stdexcept>
namespace strata::hpu {
void quantize_q4_rows(std::span<const float> input,uint32_t columns,std::span<uint8_t> output) {
    if(!columns || columns>65536 || columns%32 || input.empty() || input.size()%columns ||
       input.size()/columns>1048576 || output.size()!=input.size()/32*18)
        throw std::invalid_argument("invalid Q4 requantization shape/output");
    for(size_t base=0;base<input.size();base+=32) {
        float maximum=0,signed_maximum=0;
        for(uint32_t i=0;i<32;++i) {
            const float value=input[base+i];if(!std::isfinite(value)) throw std::invalid_argument("Q4 requantization requires finite weights");
            if(maximum<std::abs(value)) {maximum=std::abs(value);signed_maximum=value;}
        }
        // Same signed first-maximum and reciprocal as quantize_row_q4_0_ref.
        const float scale=signed_maximum/-8.0f,inverse=scale?1.0f/scale:0;
        if(!std::isfinite(inverse) || !std::isfinite(ggml_fp16_to_fp32(ggml_fp32_to_fp16(scale))))
            throw std::overflow_error("Q4 requantization reciprocal/FP16 scale overflow");
    }
    const auto bytes=ggml_quantize_chunk(GGML_TYPE_Q4_0,input.data(),output.data(),0,int64_t(input.size()/columns),columns,nullptr);
    if(bytes!=output.size()) throw std::runtime_error("ggml Q4 requantization returned wrong byte count");
}
}
