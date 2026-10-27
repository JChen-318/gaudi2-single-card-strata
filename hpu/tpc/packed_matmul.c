// Packed weights stay in HBM as bytes. Integer products precede FP16 scale
// widening; shared/head Q4 correction uses RAW Q8_1 sum, routed Q4 centers
// integer codes, routed K-quants use their own 6-bit group scales and minima.
// Scalar reference scheduling:
// output reductions are ordered here, not CUDA warp/CTA-bitwise reductions.
#include "packed_codebook_layout.inc"
#include "fp4_values.inc"
static unsigned char packed_byte(tensor value,int column,int row) {
    int5 p={column,row,0,0,0};
    return s_u8_ld_g((__global__ unsigned char*)gen_addr(p,value));
}
static int packed_signed(unsigned char value) {return value<128?(int)value:(int)value-256;}
static float packed_half(unsigned int bits) {
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
// IQ4_NL is a nonlinear codebook, not an affine range: the four-bit index selects
// one of these sixteen int8 values. Byte-for-byte identical to the pinned CUDA
// iq4nl_values (native_mmvq.cu:449) and to kvalues_iq4nl in the fixed ggml tree, so
// a wrong index is a wrong weight value, which is why the host test permutes it.
static const int packed_iq4nl_values[16]={-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113};
static int packed_iq4nl(int index) {return packed_iq4nl_values[index&15];}
static float packed_s2_dot(tensor weight,tensor activation,tensor metadata,int columns,int row,int token) {
    float acc[8]={0,0,0,0,0,0,0,0};float correction=0;
    for(int block=0;block<columns/64;++block) {
        const int at=block*18;
        const unsigned int bits=(unsigned int)packed_byte(weight,at,row)|((unsigned int)packed_byte(weight,at+1,row)<<8);
        const float d=packed_half(bits);int5 m={0,block*2,token,0,0};
        const float lo=s_f32_ld_g((__global__ float*)gen_addr(m,metadata));m[1]++;
        const float hi=s_f32_ld_g((__global__ float*)gen_addr(m,metadata));m[0]=1;
        const float hx_hi=s_f32_ld_g((__global__ float*)gen_addr(m,metadata));m[1]--;
        const float hx_lo=s_f32_ld_g((__global__ float*)gen_addr(m,metadata));
#ifndef STRATA_PACKED_EMULATION
#pragma unroll
#endif
        for(int lane=0;lane<8;++lane) {
            int dot_lo=0,dot_hi=0;
            const int low=packed_byte(weight,at+2+lane,row),high=packed_byte(weight,at+10+lane,row);
#ifndef STRATA_PACKED_EMULATION
#pragma unroll
#endif
            for(int j=0;j<4;++j) {
                dot_lo+=((low>>(j*2))&3)*packed_signed(packed_byte(activation,block*64+lane*4+j,token));
                dot_hi+=((high>>(j*2))&3)*packed_signed(packed_byte(activation,block*64+32+lane*4+j,token));
            }
            acc[lane]=s_f32_mac(d*lo,(float)dot_lo,acc[lane]);
            acc[lane]=s_f32_mac(d*hi,(float)dot_hi,acc[lane]);
        }
        // The original AVX2 Release kernel contracts this scalar update too.
        correction=s_f32_mac(d,hx_lo+hx_hi,correction);
    }
    // Original AVX2: low/high 128-bit add, movehl add, movehdup add
    // (offsets 4,2,1). Adjacent horizontal adds change cancellation cases.
    return ((acc[0]+acc[4])+(acc[2]+acc[6]))+((acc[1]+acc[5])+(acc[3]+acc[7]))-correction;
}
// K-quant superblock: 256 values in eight 32-value groups with 6-bit scale and
// minimum pairs packed into 12 bytes (pinned ggml get_scale_min_k4). The
// minimum multiplies the sum of the QUANTIZED activations, as in
// iq_kernels.cu vec_dot_q4_K_q8_1_impl_vmmq, never the Q8_1 raw-input sum.
static int packed_k_field(tensor weight,int scales_at,int row,int group,int minimum) {
    // get_scale_min_k4: groups 0..3 keep six low bits in their own byte while
    // groups 4..7 take four bits from scales[8..11] plus two high bits that the
    // low bytes reserve above bit six. Bytes are read from HBM directly because
    // TPC C rejects local aggregates of sub-32-bit elements.
    if(group<4) return packed_byte(weight,scales_at+(minimum?group+4:group),row)&63;
    const unsigned int low=(unsigned int)(minimum?(packed_byte(weight,scales_at+group+4,row)>>4)
                                                 :(packed_byte(weight,scales_at+group+4,row)&15));
    const unsigned int high=(unsigned int)(packed_byte(weight,scales_at+(minimum?group:group-4),row)>>6);
    return (int)(low|((high&3u)<<4));
}
static float packed_k_dot(tensor weight,tensor activation,tensor metadata,int columns,int row,int token,int type) {
    const int size=type==12?144:176,qs=type==12?16:48;float total=0;
    for(int block=0;block<columns/256;++block) {
        const int at=block*size,scales_at=at+4;
        const unsigned int d_bits=(unsigned int)packed_byte(weight,at,row)|((unsigned int)packed_byte(weight,at+1,row)<<8);
        const unsigned int m_bits=(unsigned int)packed_byte(weight,at+2,row)|((unsigned int)packed_byte(weight,at+3,row)<<8);
        const float d=packed_half(d_bits),dmin=packed_half(m_bits);float sum_d=0,sum_m=0;
        for(int group=0;group<8;++group) {
            // Chunk c holds values 64c..64c+63: low nibbles of bytes 32c..32c+31
            // are group 2c, high nibbles of the same bytes are group 2c+1.
            const int chunk=group/2,high=group&1;int dot=0,code_sum=0;
            for(int l=0;l<32;++l) {
                const int byte=packed_byte(weight,at+qs+chunk*32+l,row);
                int code=high?(byte>>4):(byte&15);
                if(type==13) code+=((packed_byte(weight,at+16+l,row)>>(chunk*2+high))&1)<<4;
                const int a=packed_signed(packed_byte(activation,block*256+group*32+l,token));
                dot+=code*a;code_sum+=a;
            }
            int5 m={0,block*8+group,token,0,0};
            const float scale=s_f32_ld_g((__global__ float*)gen_addr(m,metadata));
            // The original CUDA kernel contracts both group accumulations.
            sum_d=s_f32_mac(scale,(float)(dot*packed_k_field(weight,scales_at,row,group,0)),sum_d);
            sum_m=s_f32_mac(scale,(float)(code_sum*packed_k_field(weight,scales_at,row,group,1)),sum_m);
        }
        total=s_f32_mac(d,sum_d,total);total=s_f32_mac(-dmin,sum_m,total);
    }
    return total;
}
// Q3_K superblock (pinned ggml block_q3_K + dequantize_row_q3_K and native_mmvq
// q3_q8_dot_impl): hmask[32], qs[64], scales[12] at 96 and ONE signed half d at
// offset 108 - no learned minimum. Sixteen 16-value sub-blocks each carry a 6-bit
// scale: sub-block s takes its low nibble from scales[s%8] at shift 4*(s/8) and its
// top two bits from scales[8+(s%4)] at shift 2*(s/4), then uses (raw-32). Value v is
// (low two bits of qs byte 32*(v/128)+v%32 at shift 2*((v%128)/32)) plus the third
// bit from hmask byte v%32 bit v/32, minus four. Two sub-blocks share one Q8_1
// activation block, so d8 is indexed by s/2 and the whole half-superblock sum is
// multiplied by d. The activation factor is always the int8 code dot, never a sum of
// the raw inputs.
static int packed_q3_scale(tensor weight,int scales_at,int row,int sub) {
    const int low=(packed_byte(weight,scales_at+(sub%8),row)>>(4*(sub/8)))&15;
    const int high=(packed_byte(weight,scales_at+8+(sub%4),row)>>(2*(sub/4)))&3;
    return (low|(high<<4))-32;
}
static float packed_q3k_dot(tensor weight,tensor activation,tensor metadata,int columns,int row,int token) {
    float total=0;
    for(int block=0;block<columns/256;++block) {
        const int at=block*110,scales_at=at+96;
        const unsigned int bits=(unsigned int)packed_byte(weight,at+108,row)|((unsigned int)packed_byte(weight,at+109,row)<<8);
        const float d=packed_half(bits);
        // Upstream multiplies d per half-superblock (four q8_1 blocks each), so the
        // eight sub-block terms of that half are grouped before the multiply.
        for(int section=0;section<2;++section) {
            float sum=0;
            for(int sub=section*8;sub<section*8+8;++sub) {
                int dot=0;
                for(int l=0;l<16;++l) {
                    const int v=sub*16+l;
                    // qs starts after hmask: byte 32*(v/128)+v%32, plane (v%128)/32.
                    const int low=(packed_byte(weight,at+32+32*(v/128)+(v%32),row)>>(2*((v%128)/32)))&3;
                    const int high=(packed_byte(weight,at+(v%32),row)>>(v/32))&1;
                    const int code=(low|(high<<2))-4;
                    dot+=code*packed_signed(packed_byte(activation,block*256+v,token));
                }
                int5 m={0,block*8+sub/2,token,0,0};
                const float d8=s_f32_ld_g((__global__ float*)gen_addr(m,metadata));
                sum=s_f32_mac(d8,(float)(dot*packed_q3_scale(weight,scales_at,row,sub)),sum);
            }
            total=s_f32_mac(d,sum,total);
        }
    }
    return total;
}
// Q6_K superblock (pinned ggml block_q6_K + dequantize_row_q6_K and native_mmvq
// q6_q8_dot_impl): ql[128] at 0, qh[64] at 128, sixteen SIGNED int8 sub-block
// scales at 192 and one signed half d at 208, so 210 bytes. The field map was
// measured with work/validation/q6k_layout_probe.cpp against ggml's own decoder:
// for value v, j=v/128 and i=v%128 put the low nibble at ql byte 64*j+i%64 shift
// 4*(i/64) and the two high bits at qh byte 32*j+i%32 shift 2*(i/32); sub-block
// s=v/16 supplies the signed scale. Decoded weight is d*scale*((code)-32), so as
// in Q3_K there is no learned minimum and the activation factor is the int8 code
// dot only, with two sub-blocks sharing one Q8_1 block (d8 index s/2).
static float packed_q6k_dot(tensor weight,tensor activation,tensor metadata,int columns,int row,int token) {
    float total=0;
    for(int block=0;block<columns/256;++block) {
        const int at=block*210;
        const unsigned int bits=(unsigned int)packed_byte(weight,at+208,row)|((unsigned int)packed_byte(weight,at+209,row)<<8);
        const float d=packed_half(bits);
        for(int section=0;section<2;++section) {
            float sum=0;
            for(int sub=section*8;sub<section*8+8;++sub) {
                const int scale=packed_signed(packed_byte(weight,at+192+sub,row));
                int dot=0;
                for(int l=0;l<16;++l) {
                    const int v=sub*16+l,j=v/128,i=v%128;
                    const int low=(packed_byte(weight,at+64*j+(i%64),row)>>(4*(i/64)))&15;
                    const int high=(packed_byte(weight,at+128+32*j+(i%32),row)>>(2*(i/32)))&3;
                    dot+=((low|(high<<4))-32)*packed_signed(packed_byte(activation,block*256+v,token));
                }
                int5 m={0,block*8+sub/2,token,0,0};
                const float d8=s_f32_ld_g((__global__ float*)gen_addr(m,metadata));
                sum=s_f32_mac(d8,(float)(dot*scale),sum);
            }
            total=s_f32_mac(d,sum,total);
        }
    }
    return total;
}
// IQ4_XS superblock (pinned ggml block_iq4_xs and dequantize_row_iq4_xs, measured with
// work/validation/iq4xs_layout_probe_v2.cpp): 136 bytes per 256 values with half d at 0,
// a 16-bit scales_h at 2, scales_l[4] at 4 and qs[128] at 8. Each of the eight 32-value
// sub-blocks carries a 6-bit scale - the low nibble of scales_l[sub/2] for an even
// sub-block and the high nibble for an odd one, with two more bits from scales_h at
// 2*sub - used as (raw-32), and one d8 of its own. Values are codebook indices: byte
// 8+16*sub+j holds value 32*sub+j in its low nibble and value 32*sub+16+j in its high
// nibble, so this is not the Q3_K/Q4_K bit-plane layout.
static int packed_iq4xs_scale(tensor weight,int scales_at,int row,int sub) {
    const int low=(packed_byte(weight,scales_at+sub/2,row)>>(4*(sub%2)))&15;
    const int high=((packed_byte(weight,scales_at-2,row)|(packed_byte(weight,scales_at-1,row)<<8))>>(2*sub))&3;
    return (low|(high<<4))-32;
}
static float packed_iq4xs_dot(tensor weight,tensor activation,tensor metadata,int columns,int row,int token) {
    float total=0;
    for(int block=0;block<columns/256;++block) {
        const int at=block*136;
        const float d=packed_half((unsigned int)packed_byte(weight,at,row)|((unsigned int)packed_byte(weight,at+1,row)<<8));
        float sum=0;
        for(int sub=0;sub<8;++sub) {
            const int scale=packed_iq4xs_scale(weight,at+4,row,sub);
            int dot=0;
            for(int j=0;j<16;++j) {
                const int code=packed_byte(weight,at+8+sub*16+j,row);
                dot+=packed_iq4nl(code&15)*packed_signed(packed_byte(activation,block*256+sub*32+j,token));
                dot+=packed_iq4nl(code>>4)*packed_signed(packed_byte(activation,block*256+sub*32+16+j,token));
            }
            int5 m={0,block*8+sub,token,0,0};
            const float d8=s_f32_ld_g((__global__ float*)gen_addr(m,metadata));
            sum=s_f32_mac(d8,(float)(dot*scale),sum);
        }
        total=s_f32_mac(d,sum,total);
    }
    return total;
}
// IQ2_XXS (GGML type 16) superblock: 66 bytes per 256 values, measured against the
// unmodified pinned ggml row decoder with work/validation/iq2xxs_layout_probe.cpp. Half d
// sits at 0, then eight 8-byte groups at 2+8*group: the first four bytes are indices into
// packed_iq2xxs_grid (byte +lane holds values group*32+lane*8+j for j 0..7, eight positive
// one-byte magnitudes each) and the last four are a little-endian word whose bits 7*lane
// through 7*lane+6 are that lane's sign field and whose bits 28..31 are the 4-bit group
// scale. A lane's value is negative when pattern bit j is set, with pattern = bits XOR
// ((popcount(bits) AND 1) << 7): the parity fold is how ggml keeps a negated zero at zero.
// The scale is the pinned CUDA integer of iq_kernels.cu:98, sumi*(2*sc+1)/8 with C
// truncation toward zero. Upstream writes (word>>27)|1, which is the same number because
// bit 27 belongs to lane 3's sign field, so this matches CUDA exactly and differs from the
// decoder's float (0.5+sc)*0.25 form only by that truncation, by at most one integer unit.
static int packed_popcount7(int value) {
    int count=0;
    for(int bit=0;bit<7;++bit) if(value&(1<<bit)) ++count;
    return count;
}
static float packed_iq2xxs_dot(tensor weight,tensor activation,tensor metadata,tensor codebook,int columns,int row,int token) {
    float total=0;
    for(int block=0;block<columns/256;++block) {
        const int at=block*66;
        const float d=packed_half((unsigned int)packed_byte(weight,at,row)|((unsigned int)packed_byte(weight,at+1,row)<<8));
        for(int group=0;group<8;++group) {
            const int word_at=at+2+group*8;
            const unsigned int word=(unsigned int)packed_byte(weight,word_at+4,row)|((unsigned int)packed_byte(weight,word_at+5,row)<<8)
                |((unsigned int)packed_byte(weight,word_at+6,row)<<16)|((unsigned int)packed_byte(weight,word_at+7,row)<<24);
            const int ls=2*(int)(word>>28)+1;
            int sumi=0;
            for(int lane=0;lane<4;++lane) {
                const int entry_at=STRATA_PACKED_CODEBOOK_IQ2XXS_AT+packed_byte(weight,word_at+lane,row)*8;
                const int bits=(int)((word>>(7*lane))&127u);
                const int pattern=bits^((packed_popcount7(bits)&1)<<7);
                for(int j=0;j<8;++j) {
                    const int magnitude=packed_byte(codebook,entry_at+j,0);
                    sumi+=((pattern&(1<<j))?-magnitude:magnitude)*packed_signed(packed_byte(activation,block*256+group*32+lane*8+j,token));
                }
            }
            int5 m={0,block*8+group,token,0,0};
            const float d8=s_f32_ld_g((__global__ float*)gen_addr(m,metadata));
            total+=(d*d8)*(float)(sumi*ls/8);
        }
    }
    return total;
}
// IQ3_XXS (GGML type 18) superblock, 98 bytes per 256 values, measured against the
// unmodified pinned ggml row decoder with work/validation/iq3xxs_layout_probe.cpp. Half d
// sits at 0; the 64 grid bytes start at 2 as eight groups of 8, and byte 2+8*group+2*lane
// carries the four magnitudes of values group*32+lane*8+j while the next byte carries the
// four of +4+j. The 32 sign/scale bytes start at 66: word = little-endian at 66+4*group,
// lane l uses bits 7*l..7*l+6 with the same popcount parity fold as IQ2_XXS (bit 7 of the
// pattern negates the lane eighth value), and bits 28..31 are the group scale.
// The scale is the pinned CUDA integer of iq_kernels.cu:124, (scale*sumi + sumi/2)/2 with C
// truncation, which the decoder writes as the float d*(0.5+scale)*0.5; the two agree except
// for that truncation, at most one integer unit per group.
static float packed_iq3xxs_dot(tensor weight,tensor activation,tensor metadata,tensor codebook,int columns,int row,int token) {
    float total=0;
    for(int block=0;block<columns/256;++block) {
        const int at=block*98;
        const float d=packed_half((unsigned int)packed_byte(weight,at,row)|((unsigned int)packed_byte(weight,at+1,row)<<8));
        for(int group=0;group<8;++group) {
            const int word_at=at+66+group*4;
            const unsigned int word=(unsigned int)packed_byte(weight,word_at,row)|((unsigned int)packed_byte(weight,word_at+1,row)<<8)
                |((unsigned int)packed_byte(weight,word_at+2,row)<<16)|((unsigned int)packed_byte(weight,word_at+3,row)<<24);
            const int ls=(int)(word>>28);
            int sumi=0;
            for(int lane=0;lane<4;++lane) {
                const int low_at=STRATA_PACKED_CODEBOOK_IQ3XXS_AT+packed_byte(weight,at+2+group*8+lane*2,row)*4;
                const int high_at=STRATA_PACKED_CODEBOOK_IQ3XXS_AT+packed_byte(weight,at+2+group*8+lane*2+1,row)*4;
                const int bits=(int)((word>>(7*lane))&127u);
                const int pattern=bits^((packed_popcount7(bits)&1)<<7);
                for(int j=0;j<4;++j) {
                    const int first=packed_byte(codebook,low_at+j,0);
                    sumi+=((pattern&(1<<j))?-first:first)*packed_signed(packed_byte(activation,block*256+group*32+lane*8+j,token));
                    const int second=packed_byte(codebook,high_at+j,0);
                    sumi+=((pattern&(1<<(j+4)))?-second:second)*packed_signed(packed_byte(activation,block*256+group*32+lane*8+4+j,token));
                }
            }
            int5 m={0,block*8+group,token,0,0};
            const float d8=s_f32_ld_g((__global__ float*)gen_addr(m,metadata));
            total+=(d*d8)*(float)((ls*sumi+sumi/2)/2);
        }
    }
    return total;
}
// IQ2_XS (GGML type 17) superblock, 74 bytes per 256 values, measured against the
// unmodified pinned ggml row decoder with work/validation/iq2xs_layout_probe.cpp. Half d
// sits at 0; 32 little-endian uint16 codes start at 2 and code c carries values 8c..8c+7,
// its bits 0..8 indexing packed_iq2xs_grid (eight one-byte magnitudes) and bits 9..15
// holding the 7-bit sign field with the same popcount parity fold as the other IQ2/IQ3
// tables. The scale is one byte per 32-value group at 66: its low nibble scales that
// group's first two codes, its high nibble the last two. The pinned CUDA form is the
// INTEGER (sumi0*ls0 + sumi1*ls1 + (sumi0+sumi1)/2)/4, which matches the decoder's float
// (0.5+ls)*0.25 factor except for that truncation, at most one integer unit per group.
static float packed_iq2xs_dot(tensor weight,tensor activation,tensor metadata,tensor codebook,int columns,int row,int token) {
    float total=0;
    for(int block=0;block<columns/256;++block) {
        const int at=block*74;
        const float d=packed_half((unsigned int)packed_byte(weight,at,row)|((unsigned int)packed_byte(weight,at+1,row)<<8));
        for(int group=0;group<8;++group) {
            const unsigned int scale=packed_byte(weight,at+66+group,row);
            const int ls0=(int)(scale&15u),ls1=(int)(scale>>4);
            int sumi0=0,sumi1=0;
            for(int lane=0;lane<4;++lane) {
                const int code_at=at+2+group*8+lane*2;
                const unsigned int code=(unsigned int)packed_byte(weight,code_at,row)|((unsigned int)packed_byte(weight,code_at+1,row)<<8);
                const int entry_at=STRATA_PACKED_CODEBOOK_IQ2XS_AT+(code&511u)*8;
                const int bits=(int)((code>>9)&127u);
                const int pattern=bits^((packed_popcount7(bits)&1)<<7);
                int lane_sum=0;
                for(int j=0;j<8;++j) {
                    const int magnitude=packed_byte(codebook,entry_at+j,0);
                    lane_sum+=((pattern&(1<<j))?-magnitude:magnitude)*packed_signed(packed_byte(activation,block*256+group*32+lane*8+j,token));
                }
                if(lane<2) sumi0+=lane_sum;else sumi1+=lane_sum;
            }
            int5 m={0,block*8+group,token,0,0};
            const float d8=s_f32_ld_g((__global__ float*)gen_addr(m,metadata));
            total+=(d*d8)*(float)((sumi0*ls0+sumi1*ls1+(sumi0+sumi1)/2)/4);
        }
    }
    return total;
}
// IQ3_S (GGML type 21) superblock, 110 bytes per 256 values, measured with
// work/validation/iq3s_layout_probe.cpp against the unmodified pinned ggml row decoder.
// Code (group*8 + lane*2 + half) is qs byte 8g+2l+half ORed with qh[group] bit
// (2l+half) shifted to bit 8, and its grid entry holds the four magnitudes of values
// 4*code..4*code+3. signs[4g+l] carries the plain 8-bit sign field of values
// 32g+8l..32g+8l+7. The scale byte 106+g/2 uses its low nibble for the even group and its
// high nibble for the odd one, and the multiplier is the integer (1 + 2*sc) exactly as the
// pinned iq_kernels.cu:215 applies it, so no truncation separates this format from the
// decoder - only the float accumulation order does.
static float packed_iq3s_dot(tensor weight,tensor activation,tensor metadata,tensor codebook,int columns,int row,int token) {
    float total=0;
    for(int block=0;block<columns/256;++block) {
        const int at=block*110;
        const float d=packed_half((unsigned int)packed_byte(weight,at,row)|((unsigned int)packed_byte(weight,at+1,row)<<8));
        for(int group=0;group<8;++group) {
            const int sc=((int)(packed_byte(weight,at+106+group/2,row)>>(4*(group%2)))&15);
            const int qh=packed_byte(weight,at+66+group,row);
            int dot=0;
            for(int lane=0;lane<4;++lane) {
                const int signs=packed_byte(weight,at+74+group*4+lane,row);
                for(int section=0;section<2;++section) {
                    const int index=packed_byte(weight,at+2+group*8+lane*2+section,row)|(((qh>>(lane*2+section))&1)<<8);
                    const int grid_at=STRATA_PACKED_CODEBOOK_IQ3S_AT+index*4;
                    for(int j=0;j<4;++j) {
                        const int magnitude=packed_byte(codebook,grid_at+j,0);
                        dot+=((signs&(1<<(section*4+j)))?-magnitude:magnitude)
                            *packed_signed(packed_byte(activation,block*256+group*32+lane*8+section*4+j,token));
                    }
                }
            }
            int5 m={0,block*8+group,token,0,0};
            const float d8=s_f32_ld_g((__global__ float*)gen_addr(m,metadata));
            total+=(d*d8)*(float)((1+2*sc)*dot);
        }
    }
    return total;
}
// IQ2_S (GGML type 22) superblock, 82 bytes per 256 values, measured with
// work/validation/iq2s_layout_probe.cpp against the unmodified pinned ggml row decoder.
// Grid byte 4*group+lane at 2+4*group+lane indexes packed_iq2s_grid with the two bits
// qh[group]>>(2*lane) extended to positions 8 and 9; its entry holds the eight magnitudes of
// values 32*group+8*lane..+7. The sign byte at 34+4*group+lane negates value j of that lane
// when bit j is set, plain, with no parity fold. scales[74+group] drives the two halves of
// the 32-value group with the pinned CUDA integer (sumi0*ls0+sumi1*ls1+(sumi0+sumi1)/2)/4,
// which truncates where the decoder multiplies by (0.5+ls)*0.25 in float.
static float packed_iq2s_dot(tensor weight,tensor activation,tensor metadata,tensor codebook,int columns,int row,int token) {
    float total=0;
    for(int block=0;block<columns/256;++block) {
        const int at=block*82;
        const float d=packed_half((unsigned int)packed_byte(weight,at,row)|((unsigned int)packed_byte(weight,at+1,row)<<8));
        for(int group=0;group<8;++group) {
            const unsigned int scale=packed_byte(weight,at+74+group,row);
            const int ls0=(int)(scale&15u),ls1=(int)(scale>>4);
            const int qh=packed_byte(weight,at+66+group,row);
            int sumi0=0,sumi1=0;
            for(int lane=0;lane<4;++lane) {
                const int index=packed_byte(weight,at+2+group*4+lane,row)|(((qh>>(lane*2))&3)<<8);
                const int entry_at=STRATA_PACKED_CODEBOOK_IQ2S_AT+index*8;
                const int signs=packed_byte(weight,at+34+group*4+lane,row);
                int lane_sum=0;
                for(int j=0;j<8;++j) {
                    const int magnitude=packed_byte(codebook,entry_at+j,0);
                    lane_sum+=((signs&(1<<j))?-magnitude:magnitude)
                        *packed_signed(packed_byte(activation,block*256+group*32+lane*8+j,token));
                }
                if(lane<2) sumi0+=lane_sum;else sumi1+=lane_sum;
            }
            int5 m={0,block*8+group,token,0,0};
            const float d8=s_f32_ld_g((__global__ float*)gen_addr(m,metadata));
            total+=(d*d8)*(float)((sumi0*ls0+sumi1*ls1+(sumi0+sumi1)/2)/4);
        }
    }
    return total;
}
// IQ1_M (GGML type 29) superblock, 56 bytes per 256 values, measured with
// work/validation/iq1m_layout_probe.cpp against the unmodified pinned ggml row decoder.
// The block carries no d: the fp16 scale is assembled from nibbles of its own eight scale
// bytes (ggml-quants.c:2679, iq_kernels.cu:266). qs[4g+l] holds the low index bits of the
// 8-value lane 32g+8l..32g+8l+7, qh[2g+l/2] bits 4*(l%2)..+2 hold index bits 8..10 and bit
// 4*(l%2)+3 that lane's delta sign, and the grid entry stores the ternary pattern as eight
// nibbles of value+1. Each 16-value half is summed as the pinned CUDA does,
// (sumi + delta*sumy)*(2*field+1) with delta = -1 + 0.125 - (bit ? 0.25 : 0), which is the
// decoder's (grid +/- 0.125)*d*(2*field+1) term by term - no truncation, only float order.
static float packed_iq1m_dot(tensor weight,tensor activation,tensor metadata,tensor codebook,int columns,int row,int token) {
    float total=0;
    for(int block=0;block<columns/256;++block) {
        const int at=block*56;
        const unsigned int w0=(unsigned int)packed_byte(weight,at+48,row)|((unsigned int)packed_byte(weight,at+49,row)<<8);
        const unsigned int w1=(unsigned int)packed_byte(weight,at+50,row)|((unsigned int)packed_byte(weight,at+51,row)<<8);
        const unsigned int w2=(unsigned int)packed_byte(weight,at+52,row)|((unsigned int)packed_byte(weight,at+53,row)<<8);
        const unsigned int w3=(unsigned int)packed_byte(weight,at+54,row)|((unsigned int)packed_byte(weight,at+55,row)<<8);
        const unsigned int words[4]={w0,w1,w2,w3};
        const float d=packed_half((w0>>12)|((w1>>8)&0xf0u)|((w2>>4)&0xf00u)|(w3&0xf000u));
        for(int group=0;group<8;++group) {
            const unsigned int tmp=words[group/2]>>(6*(group%2));
            const int sc0=2*(int)((tmp>>0)&7u)+1,sc1=2*(int)((tmp>>3)&7u)+1;
            int sumi0=0,sumi1=0;
            float sumf0=0,sumf1=0;
            for(int lane=0;lane<4;++lane) {
                const int qs=packed_byte(weight,at+4*group+lane,row);
                const int qhl=packed_byte(weight,at+32+2*group+lane/2,row)>>(4*(lane%2));
                const int grid_at=STRATA_PACKED_CODEBOOK_IQ1S_AT+(qs|((qhl&7)<<8))*4;
                const float delta=-1.0f+0.125f-((qhl&8)?0.25f:0.0f);
                int nibble_dot=0,code_sum=0;
                for(int j=0;j<8;++j) {
                    const int grid_byte=packed_byte(codebook,grid_at+(j<4?j:j-4),0);const int nibble=(j<4?(grid_byte&15):(grid_byte>>4));
                    const int code=packed_signed(packed_byte(activation,block*256+group*32+lane*8+j,token));
                    nibble_dot+=nibble*code;code_sum+=code;
                }
                if(lane<2) {sumi0+=nibble_dot;sumf0+=delta*(float)code_sum;}
                else {sumi1+=nibble_dot;sumf1+=delta*(float)code_sum;}
            }
            int5 m={0,block*8+group,token,0,0};
            const float d8=s_f32_ld_g((__global__ float*)gen_addr(m,metadata));
            total+=(d*d8)*(((float)sumi0+sumf0)*(float)sc0+((float)sumi1+sumf1)*(float)sc1);
        }
    }
    return total;
}
// NVFP4 (GGML type 40) 64-value block, 36 bytes, measured with
// work/validation/nvfp4_layout_probe.cpp against the unmodified pinned ggml row decoder.
// The UE4M3 sub-block scale is transcribed from ggml-impl.h:514 - exp=(x>>3)&15, man=x&7,
// exp 0 gives man*2^-9 and otherwise (1+man/8)*2^(exp-7), with the helper's closing half
// folded into 2^(exp-8) - and 0x00/0x7F are zero. The nibble magnitudes come from the pinned
// doubled e2m1 table, so the dot stays integer and the scale carries the fraction.
static float packed_ue4m3(unsigned int x) {
    if(x==0u || x==0x7fu) return 0.0f;
    const int exp=(int)((x>>3)&15u),man=(int)(x&7u);
    const int e=exp==0?-10:exp-8;   // the helper's closing half applies to both branches
    float scale=1.0f;
    for(int k=0;k<e;++k) scale+=scale;
    for(int k=0;k<-e;++k) scale*=0.5f;
    return (exp==0?(float)man:1.0f+(float)man*0.125f)*scale;
}
static float packed_nvfp4_dot(tensor weight,tensor activation,tensor metadata,int columns,int row,int token) {
    float total=0;
    for(int block=0;block<columns/64;++block) {
        const int at=block*36;
        for(int sub=0;sub<4;++sub) {
            const float scale=packed_ue4m3((unsigned int)packed_byte(weight,at+sub,row));
            int dot=0;
            for(int j=0;j<8;++j) {
                const int pair=packed_byte(weight,at+4+sub*8+j,row);
                dot+=packed_kvalues_fp4[pair&15]*packed_signed(packed_byte(activation,block*64+sub*16+j,token));
                dot+=packed_kvalues_fp4[pair>>4]*packed_signed(packed_byte(activation,block*64+sub*16+8+j,token));
            }
            int5 m={0,block*2+sub/2,token,0,0};
            const float d8=s_f32_ld_g((__global__ float*)gen_addr(m,metadata));
            total+=(scale*d8)*(float)dot;
        }
    }
    return total;
}
void main(tensor weight,tensor activation,tensor metadata,tensor codebook,tensor output,int columns,int rows,int batch,int type,int activation_mode) {
    const int5 start=get_index_space_offset(),end=start+get_index_space_size();
    for(int t=start[1];t<end[1] && t<batch;++t) for(int row=start[0];row<end[0] && row<rows;++row) {
        if(activation_mode==1) {
            const float result=packed_s2_dot(weight,activation,metadata,columns,row,t);
            int5 p={row,t,0,0,0};s_f32_st_g((__global__ float*)gen_addr(p,output),result);continue;
        }
        if(type==40) {
            const float result=packed_nvfp4_dot(weight,activation,metadata,columns,row,t);
            int5 p={row,t,0,0,0};s_f32_st_g((__global__ float*)gen_addr(p,output),result);continue;
        }
        if(type==11 || type==12 || type==13 || type==14 || type==16 || type==17 || type==18 || type==21 || type==22 || type==23 || type==29) {
            const float result=type==11?packed_q3k_dot(weight,activation,metadata,columns,row,t)
                :type==16?packed_iq2xxs_dot(weight,activation,metadata,codebook,columns,row,t)
                :type==17?packed_iq2xs_dot(weight,activation,metadata,codebook,columns,row,t)
                :type==18?packed_iq3xxs_dot(weight,activation,metadata,codebook,columns,row,t)
                :type==21?packed_iq3s_dot(weight,activation,metadata,codebook,columns,row,t)
                :type==22?packed_iq2s_dot(weight,activation,metadata,codebook,columns,row,t)
                :type==29?packed_iq1m_dot(weight,activation,metadata,codebook,columns,row,t)
                :type==14?packed_q6k_dot(weight,activation,metadata,columns,row,t)
                :type==23?packed_iq4xs_dot(weight,activation,metadata,columns,row,t)
                :packed_k_dot(weight,activation,metadata,columns,row,t,type);
            int5 p={row,t,0,0,0};s_f32_st_g((__global__ float*)gen_addr(p,output),result);continue;
        }
        float total=0;
        for(int block=0;block<columns/32;++block) {
            const int at=type==42?(block/2)*18:block*(type==3?20:type==6?22:type==7?24:type==8?34:18);
            const unsigned int bits=(unsigned int)packed_byte(weight,at,row)|((unsigned int)packed_byte(weight,at+1,row)<<8);
            const float d=packed_half(bits);int5 m={0,block,t,0,0};
            const float scale=s_f32_ld_g((__global__ float*)gen_addr(m,metadata));m[0]=1;
            const float sum=s_f32_ld_g((__global__ float*)gen_addr(m,metadata));
            if(type==2) {
                for(int part=0;part<2;++part) {
                    int dot=0;
                    for(int j=part*8;j<part*8+8;++j) {
                        const int w=packed_byte(weight,at+2+j,row);
                        const int center=activation_mode==2?8:0;
                        dot+=((w&15)-center)*packed_signed(packed_byte(activation,block*32+j,t));
                        dot+=((w>>4)-center)*packed_signed(packed_byte(activation,block*32+j+16,t));
                    }
                    if(activation_mode==2) total+=(d*scale)*(float)dot;
                    else total+=d*((float)dot*scale-4.0f*sum);
                }
            } else if(type==3 || type==6 || type==7) {
                const int affine=type==3 || type==7,high=type==6 || type==7;
                const int high_at=at+(affine?4:2),codes_at=high_at+(high?4:0);
                const unsigned int minimum_bits=affine?((unsigned int)packed_byte(weight,at+2,row)|((unsigned int)packed_byte(weight,at+3,row)<<8)):0;
                const float minimum=packed_half(minimum_bits);
                for(int part=0;part<2;++part) {
                    int dot=0,code_sum=0;
                    for(int j=part*8;j<part*8+8;++j) for(int side=0;side<2;++side) {
                        const int index=j+side*16;
                        int code=(packed_byte(weight,codes_at+j,row)>>(side*4))&15;
                        if(high) code|=((packed_byte(weight,high_at+index/8,row)>>(index%8))&1)<<4;
                        const int a=packed_signed(packed_byte(activation,block*32+index,t));dot+=code*a;code_sum+=a;
                    }
                    if(affine) total+=(float)dot*(d*scale)+(float)code_sum*(minimum*scale);
                    else total+=d*((float)dot*scale-8.0f*sum);
                }
            } else if(type==8) {
                for(int part=0;part<4;++part) {
                    int dot=0;
                    for(int j=part*8;j<part*8+8;++j) dot+=packed_signed(packed_byte(weight,at+2+j,row))*packed_signed(packed_byte(activation,block*32+j,t));
                    total+=(d*scale)*(float)dot;
                }
            } else if(type==20) {
                // Pinned small_q8_dot(IQ4NLBlock): the low nibble of qs[j] and the high
                // nibble of qs[j+16] are codebook indices, so no raw-input sum and no
                // learned minimum ever enters this dot; d is the single block scale.
                for(int part=0;part<2;++part) {
                    int dot=0;
                    for(int j=part*8;j<part*8+8;++j) {
                        const int code=packed_byte(weight,at+2+j,row);
                        dot+=packed_iq4nl(code&15)*packed_signed(packed_byte(activation,block*32+j,t));
                        dot+=packed_iq4nl(code>>4)*packed_signed(packed_byte(activation,block*32+j+16,t));
                    }
                    total+=(d*scale)*(float)dot;
                }
            } else {
                int dot=0;
                for(int j=0;j<32;++j) {
                    const int index=(block%2)*32+j;
                    const int code=(packed_byte(weight,at+2+index/4,row)>>((index%4)*2))&3;
                    dot+=(code-1)*packed_signed(packed_byte(activation,block*32+j,t));
                }
                total+=(d*scale)*(float)dot;
            }
        }
        int5 result={row,t,0,0,0};s_f32_st_g((__global__ float*)gen_addr(result,output),total);
    }
}
