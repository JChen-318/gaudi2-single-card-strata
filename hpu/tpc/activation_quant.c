// The device side of the Q8_1 activation contract: FP32 activations in, the int8 codes and the
// fp16-rounded scale/sum metadata out - the exact two buffers the packed integer dots already read
// (hpu/tpc/packed_matmul.c reads activation as {columns,batch} U8 and metadata as
// {2,columns/32,batch} F32). Two authorities define this law and both are pinned by
// hpu/tests/activation_quant_device.cpp: the pinned CUDA quantizer
// (src/kernels/cuda/native_mmvq.cu:155 native_quantize_q8_1_kernel with
// include/strata/kernels/q8_1_finite.hpp:18/23/29) and this backend's host quantizer
// (hpu/src/packed_quant.cpp quantize_q8_activation), which is what feeds the graph today. They must
// agree BYTE for byte: q8_1_finite.hpp exists because an inf in a stored scale turned a residual into
// NaN, and quantize_act.cu's own header records that dividing by the fp16-ROUNDED scale instead of
// the fp32 one shifted 18 of 80 blocks - the quantized value always divides by the unrounded fp32
// scale, and only the stored metadata is fp16.
//
// The fp16 converter below is a transcription of include/strata/kernels/f16_bits.hpp:37 (bits out)
// and :73 (bits back to float). That file documents why a fourth copy is dangerous: round 198 found
// a private copy conflating a finite-but-too-large value with an inf/NaN, which no O(1) fixture could
// see. The test therefore pins this copy against the pinned one over all 65536 fp16 patterns plus the
// overflow, subnormal and exact-tie cases, so a wrong bit cannot pass.
//
// Scalar reference scheduling, executed here through the host TPC emulator: no ISA execution,
// no device and no performance claim.
static unsigned int actq_f16_bits(float value) {
    union {float f;unsigned int i;} in;in.f=value;
    const unsigned int sign=(in.i>>16)&0x8000u;
    const unsigned int rawexp=(in.i>>23)&0xffu;
    const int exp=(int)rawexp-127+15;
    unsigned int man=in.i&0x7fffffu;
    if(rawexp==0xffu) return sign|0x7c00u|(man?0x200u:0u);   // inf, or a quiet NaN
    if(exp>=31) return sign|0x7c00u;                        // a FINITE overflow saturates to inf
    if(exp<=0) {
        if(exp<-10) return sign;                            // too small: a signed zero
        man|=0x800000u;                                     // restore the implicit 1
        const unsigned int sh=(unsigned int)(14-exp);
        unsigned int h=(man>>sh)&0x3ffu;
        const unsigned int rem=man&((1u<<sh)-1u);
        if(rem>(1u<<(sh-1)) || (rem==(1u<<(sh-1)) && (h&1u))) ++h;
        return sign|h;
    }
    unsigned int h=sign|((unsigned int)exp<<10)|(man>>13);
    const unsigned int rem=man&0x1fffu;
    if(rem>0x1000u || (rem==0x1000u && (h&1u))) ++h;        // the carry may bump the exponent
    return h;
}
static float actq_f32(unsigned int bits) {
    unsigned int sign=(bits&32768u)<<16,exponent=(bits>>10)&31u,mantissa=bits&1023u;
    union {unsigned int bits;float value;} out;
    if(exponent==0) {
        if(!mantissa) out.bits=sign;
        else {
            int shifts=0;while(!(mantissa&1024u)) {mantissa<<=1;++shifts;}
            out.bits=sign|((unsigned int)(113-shifts)<<23)|((mantissa&1023u)<<13);
        }
    } else if(exponent==31) out.bits=sign|2139095040u|(mantissa<<13);
    else out.bits=sign|((exponent+112u)<<23)|(mantissa<<13);
    return out.value;
}
// The metadata field is what a reader multiplies by, so it is the fp16 value, widened back to the
// float the dot loads. Every finite block is bit-identical to before the clamp; only a value past
// the largest finite half is pulled back to it, which is what keeps inf * 0 = NaN out of the dot.
static float actq_metadata(float value) {
    const float clamped=value>65504.0f?65504.0f:value<-65504.0f?-65504.0f:value;
    return actq_f32(actq_f16_bits(clamped));
}
// Round half AWAY from zero, then hold to the int8 range the CUDA clamps to. No double is involved:
// no other TPC kernel in this tree uses it, so its support here would only be proven by an SDK build
// that cannot be run without the card. The clamp happens first - the host path clamps too, and an
// unclamped quotient can be astronomically large once the scale has been pulled back to 65504, where
// a conversion to int would be undefined. After that |value| <= 127, so the truncation and the
// remainder are both exact and `>= 0.5` is the same decision roundf makes.
static int actq_code(float quotient) {
    const float clamped=quotient>127.0f?127.0f:quotient<-127.0f?-127.0f:quotient;
    const float magnitude=clamped<0.0f?-clamped:clamped;
    float floor_value=(float)(int)magnitude;
    if(magnitude-floor_value>=0.5f) floor_value+=1.0f;
    const int rounded=(int)(clamped<0.0f?-floor_value:floor_value);
    return rounded>127?127:rounded<-127?-127:rounded;
}
void main(tensor input,tensor codes,tensor metadata,int columns,int batch) {
    const int5 start=get_index_space_offset(),end=start+get_index_space_size();
    for(int token=start[1];token<end[1] && token<batch;++token)
        for(int block=start[0];block<end[0] && block<columns/32;++block) {
            float raw[32],sum[32],maximum=0;
            for(int lane=0;lane<32;++lane) {
                int5 p={block*32+lane,token,0,0,0};
                const float value=s_f32_ld_g((__global__ float*)gen_addr(p,input));
                raw[lane]=value;sum[lane]=value;
                const float magnitude=value<0?-value:value;
                if(magnitude>maximum) maximum=magnitude;   // the pinned scan starts at zero
            }
            // The block sum is the CUDA warp's XOR butterfly, and lane 0's association order is the
            // one the host path stores; a plain serial sum can differ in the last bit.
            for(int offset=16;offset;offset>>=1) {
                float before[32];
                for(int lane=0;lane<32;++lane) before[lane]=sum[lane];
                for(int lane=0;lane<32;++lane) sum[lane]=before[lane]+before[lane^offset];
            }
            float scale=maximum/127.0f;
            if(scale>65504.0f) scale=65504.0f;
            int5 m={0,block,token,0,0};
            s_f32_st_g((__global__ float*)gen_addr(m,metadata),actq_metadata(scale));m[0]=1;
            s_f32_st_g((__global__ float*)gen_addr(m,metadata),actq_metadata(sum[0]));
            // An all-zero block quantizes to zero codes with a zero scale, as q8_1_quant returns for
            // amax == 0. A nonzero maximum whose fp32 scale flushes to zero never reaches this kernel:
            // the host gate refuses that input before the buffers are uploaded.
            for(int lane=0;lane<32;++lane) {
                const int code=scale?actq_code(raw[lane]/scale):0;
                int5 p={block*32+lane,token,0,0,0};
                s_u8_st_g((__global__ unsigned char*)gen_addr(p,codes),(unsigned char)code);
            }
        }
}
