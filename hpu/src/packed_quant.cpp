#include "strata/hpu/packed_quant.hpp"
#include "strata/hpu/window_limits.hpp"
#include "../tpc/packed_codebook_layout.inc"
#include "strata/kernels/f16_bits.hpp"
#include <algorithm>
#include <array>
#include <cmath>

namespace strata::hpu {
uint32_t packed_codebook_slot_bytes() {return STRATA_PACKED_CODEBOOK_BYTES;}
bool packed_supported(uint32_t type) {return type==2 || type==3 || type==6 || type==7 || type==8 || type==11 || type==12 || type==13 || type==14 || type==20 || type==23 || type==16 || type==17 || type==18 || type==21 || type==22 || type==29 || type==40 || type==42;}
uint32_t packed_values_per_block(uint32_t type) {
    if(type==11 || type==12 || type==13 || type==14 || type==16 || type==17 || type==18 || type==21 || type==22 || type==29 || type==23) return 256; // K-quant superblock
    if(type==42 || type==40) return 64;
    if(packed_supported(type)) return 32; // IQ4_NL keeps the 32-value block
    throw std::invalid_argument("unsupported packed format");
}
uint32_t packed_block_bytes(uint32_t type) {
    switch(type) {case 2:case 20:case 42:return 18;case 3:return 20;case 6:return 22;case 7:return 24;case 8:return 34;
        case 11:return 110;case 12:return 144;case 13:return 176;case 14:return 210;case 23:return 136;case 16:return 66;case 17:return 74;case 18:return 98;case 21:return 110;case 22:return 82;case 29:return 56;case 40:return 36;
        default:throw std::invalid_argument("unsupported packed format");}
}
uint64_t packed_row_bytes(uint32_t type,uint64_t columns) {
    const uint32_t block=packed_values_per_block(type);
    if(!packed_supported(type) || !columns || columns>65536 || columns%block) throw std::invalid_argument("invalid packed matrix type/columns");
    return checked_product(columns/block,packed_block_bytes(type));
}
namespace {
float half_finite(float x) {
    if(std::isnan(x)) throw std::overflow_error("Q8_1 raw reduction became NaN");
    x=std::clamp(x,-65504.0f,65504.0f);
    return strata::kernels::f32_from_f16(strata::kernels::f16_from_f32(x));
}
}
Q8Activation quantize_q8_activation(std::span<const float> input,uint32_t columns) {
    if(!columns || columns>65536 || columns%32 || input.empty() || input.size()%columns || input.size()/columns>kResidentWindowRows)
        throw std::invalid_argument("invalid Q8_1 activation shape");
    for(float x:input) if(!std::isfinite(x)) throw std::invalid_argument("Q8_1 activation requires finite input");
    Q8Activation out;out.codes.resize(input.size());out.metadata.resize(input.size()/16);
    for(size_t base=0;base<input.size();base+=32) {
        std::array<float,32> sum;float maximum=0;
        for(uint32_t lane=0;lane<32;++lane) {sum[lane]=input[base+lane];maximum=std::max(maximum,std::abs(sum[lane]));}
        for(uint32_t offset=16;offset;offset/=2) {
            const auto before=sum;
            for(uint32_t lane=0;lane<32;++lane) sum[lane]=before[lane]+before[lane^offset];
        }
        const float scale=std::min(maximum/127.0f,65504.0f);
        // CUDA's zero divided by an underflowed nonzero block scale has no
        // portable int8 result. Refuse explicitly instead of invoking C++ UB.
        if(maximum && !scale) throw std::overflow_error("Q8_1 activation scale underflow");
        out.metadata[base/16]=half_finite(scale);out.metadata[base/16+1]=half_finite(sum[0]);
        for(uint32_t lane=0;lane<32;++lane) {
            const float q=scale?std::clamp(std::round(input[base+lane]/scale),-127.0f,127.0f):0;
            out.codes[base+lane]=static_cast<uint8_t>(static_cast<int>(q));
        }
    }
    return out;
}
void validate_packed_rows(std::span<const uint8_t> bytes,uint32_t type,uint64_t columns,uint64_t rows) {
    const auto row=packed_row_bytes(type,columns);const uint32_t block=packed_block_bytes(type);
    if(!rows || rows>1048576 || bytes.size()!=checked_product(row,rows)) throw std::invalid_argument("invalid packed matrix payload/rows");
    // Q3_K and Q6_K store their single signed super-block scale at the END of the
    // block (hmask/qs/scales and ql/qh/scales respectively), unlike the rest.
    const uint32_t scale_at=type==11?108:type==14?208:0;
    for(uint64_t offset=0;offset<bytes.size();offset+=block) {
        // NVFP4 keeps no floating-point half anywhere in the block: its four UE4M3
        // sub-block scales decode to a finite value for every encoding (ggml-impl.h:514
        // maps both 0x00 and 0x7F to zero), so a weight byte cannot make it non-finite.
        if(type==40) continue;
        // IQ1_M stores no half at the start of the block: the fp16 super-block scale is
        // assembled from nibbles of the four scale words, so it is read back that way.
        const uint16_t assembled=[&]() -> uint16_t {
            if(type!=29) return uint16_t(bytes[offset+scale_at])|(uint16_t(bytes[offset+scale_at+1])<<8);
            const uint32_t w0=uint16_t(bytes[offset+48])|(uint16_t(bytes[offset+49])<<8);
            const uint32_t w1=uint16_t(bytes[offset+50])|(uint16_t(bytes[offset+51])<<8);
            const uint32_t w2=uint16_t(bytes[offset+52])|(uint16_t(bytes[offset+53])<<8);
            const uint32_t w3=uint16_t(bytes[offset+54])|(uint16_t(bytes[offset+55])<<8);
            return uint16_t((w0>>12)|((w1>>8)&0xf0u)|((w2>>4)&0xf00u)|(w3&0xf000u));
        }();
        const uint16_t bits=assembled;
        if(!std::isfinite(strata::kernels::f32_from_f16(bits))) throw std::invalid_argument("non-finite packed weight scale");
        if(type==3 || type==7 || type==12 || type==13) {
            const uint16_t minimum=uint16_t(bytes[offset+2])|(uint16_t(bytes[offset+3])<<8);
            if(!std::isfinite(strata::kernels::f32_from_f16(minimum))) throw std::invalid_argument("non-finite packed weight minimum");
        }
    }
}
Q8Activation quantize_s2_activation(std::span<const float> input,uint32_t columns) {
    if(!columns || columns>65536 || columns%32 || input.empty() || input.size()%columns || input.size()/columns>kResidentWindowRows)
        throw std::invalid_argument("invalid routed S2 activation shape");
    for(float x:input) if(!std::isfinite(x)) throw std::invalid_argument("routed S2 activation requires finite input");
    Q8Activation out;out.codes.resize(input.size());out.metadata.resize(input.size()/16);
    for(size_t base=0;base<input.size();base+=32) {
        float maximum=0;for(uint32_t i=0;i<32;++i) maximum=std::max(maximum,std::abs(input[base+i]));
        const float scale=maximum>0?maximum/127.0f:0,inverse=scale>0?1.0f/scale:0;
        if((maximum && !scale) || !std::isfinite(inverse)) throw std::overflow_error("routed S2 activation reciprocal/scale underflow");
        int sum=0;
        for(uint32_t i=0;i<32;++i) {
            const float product=input[base+i]*inverse;
            const float rounded=product+(product>=0?0.5f:-0.5f);
            const int code=static_cast<int>(std::clamp(rounded,-127.0f,127.0f));
            out.codes[base+i]=static_cast<uint8_t>(code);sum+=code;
        }
        const float correction=scale*float(sum);
        if(!std::isfinite(correction)) throw std::overflow_error("routed S2 activation correction overflow");
        out.metadata[base/16]=scale;out.metadata[base/16+1]=correction;
    }
    return out;
}
}
