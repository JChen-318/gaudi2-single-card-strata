#include "strata/hpu/cpu_quant_guard.hpp"
#include "strata/kernels/f16_bits.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

extern "C" void quantize_row_q8_K_ref(const float*,void*,int64_t);
namespace strata::hpu {
namespace {
struct Block {float scale;int8_t values[256];int16_t sums[16];};
static_assert(sizeof(Block)==292);
bool exceptional(const float* input) noexcept {
    float maximum=0;
    for(int i=0;i<256;++i) {if(!std::isfinite(input[i])) return true;maximum=std::max(maximum,std::abs(input[i]));}
    return maximum>0 && !std::isfinite(127.0f/maximum);
}
enum class Q8Case {Normal,Tiny,Invalid};
Q8Case q8_case(const float* input) noexcept {
    float maximum=0;
    for(int i=0;i<32;++i) {if(!std::isfinite(input[i])) return Q8Case::Invalid;maximum=std::max(maximum,std::abs(input[i]));}
    const float scale=maximum/127.0f;
    if((strata::kernels::f16_from_f32(scale)&0x7c00u)==0x7c00u) return Q8Case::Invalid;
    if(maximum && (!scale || !std::isfinite(1.0f/scale))) return Q8Case::Tiny;
    return Q8Case::Normal;
}
}
bool guard_q8k(const float* input,void* output,int64_t elements) noexcept {
    bool needs_guard=false;
    for(int64_t start=0;start<elements;start+=256) needs_guard|=exceptional(input+start);
    if(!needs_guard) return false;
    auto* destination=static_cast<uint8_t*>(output);
    for(int64_t start=0;start<elements;start+=256,destination+=sizeof(Block)) {
        const auto* x=input+start;
        if(!exceptional(x)) {quantize_row_q8_K_ref(x,destination,256);continue;}
        Block block{};bool invalid=false;float maximum=0,signed_maximum=0;
        for(int i=0;i<256;++i) {
            if(!std::isfinite(x[i])) invalid=true;
            if(std::abs(x[i])>maximum) {maximum=std::abs(x[i]);signed_maximum=x[i];}
        }
        if(invalid) block.scale=std::numeric_limits<float>::quiet_NaN();
        else if(maximum>0) {
            // Double reciprocal stays finite for every nonzero FP32 input.
            // Q8_K's FP32 scale may underflow for the very smallest values;
            // their representable result is zero, with no undefined integer cast.
            const double inverse=-127.0/double(signed_maximum);block.scale=float(1.0/inverse);
            for(int i=0;i<256;++i) {
                const auto value=int(std::nearbyint(float(inverse*double(x[i]))));
                block.values[i]=int8_t(std::clamp(value,-127,127));
                block.sums[i/16]+=block.values[i];
            }
        }
        std::memcpy(destination,&block,sizeof(block));
    }
    return true;
}
bool guard_q8_0(const float* input,void* output,int64_t elements,QuantizeRow original) noexcept {
    bool exceptional=false;
    for(int64_t offset=0;offset<elements;offset+=32) exceptional|=q8_case(input+offset)!=Q8Case::Normal;
    if(!exceptional) return false;
    auto* target=static_cast<uint8_t*>(output);
    for(int64_t offset=0;offset<elements;offset+=32,target+=34) {
        const auto kind=q8_case(input+offset);
        if(kind==Q8Case::Normal) {original(input+offset,target,32);continue;}
        std::memset(target,0,34);
        if(kind==Q8Case::Invalid) {const uint16_t marker=0x7e00;std::memcpy(target,&marker,2);}
    }
    return true;
}
}
