// STRATA_HPU_MOE_FP8: iq4_expand_tiles.c into FP8 E4M3 (bias 7, max 240) with one F32 scale per output row. The
// codes are scaled in BF16 exactly as there, by s_b[n] / scale[n] instead of s_b[n], with scale[n] = max_b |s_b[n]|
// * 127/240 (the row's largest code lands at 240), and converted to FP8 with the native (interleaving) convert,
// which returns the resident order: within each 256-row tile, lane L is row 64*(L%4) + L/4 (moe_mme_fp8.h undoes
// it on the GEMM outputs). scale {rows, count} F32 is in natural row order.
// table {256} U8, w {bytes, experts} U8, tile_expert {tiles} I32 -> out {rows, columns, count} F8, scale; an unused
// tile (-1) is skipped. Members are (256-row tile, local tile).
void main(tensor table,tensor w,tensor tile_expert,tensor out,tensor scale,int rows,int columns,int variant,int first) {
    const int5 begin=get_index_space_offset(),end=begin+get_index_space_size();
    (void)variant;
    const int blocks=columns/32,codes_bytes=rows/2*columns;
    int5 zero={0,0,0,0,0};
    const uchar256 lut=v_u8_ld_tnsr_b(zero,table);
    for(int local=begin[1];local<end[1];++local) {
        int5 ep={first+local,0,0,0,0};
        const int e=s_i32_ld_g((__global__ int*)gen_addr(ep,tile_expert));
        if(e<0) continue;
        for(int tile=begin[0];tile<end[0];++tile) {
            const int group=tile/5,t=tile-5*group;
            // s0..s3 are the block scales of natural rows 0-63, 64-127, 128-191 and 192-255 of this tile.
            float64 m0=0,m1=0,m2=0,m3=0;
            for(int b=0;b<blocks;++b) {
                int5 sp={codes_bytes+((group*blocks+b)*5+t)*1024,e,0,0,0};
                const float64 s0=(float64)v_u8_ld_tnsr_b(sp,w);sp[0]+=256;
                const float64 s1=(float64)v_u8_ld_tnsr_b(sp,w);sp[0]+=256;
                const float64 s2=(float64)v_u8_ld_tnsr_b(sp,w);sp[0]+=256;
                const float64 s3=(float64)v_u8_ld_tnsr_b(sp,w);
                m0=v_f32_max_b(m0,v_f32_sel_less_f32_b(s0,0.0f,-s0,s0));m1=v_f32_max_b(m1,v_f32_sel_less_f32_b(s1,0.0f,-s1,s1));
                m2=v_f32_max_b(m2,v_f32_sel_less_f32_b(s2,0.0f,-s2,s2));m3=v_f32_max_b(m3,v_f32_sel_less_f32_b(s3,0.0f,-s3,s3));
            }
            const float64 S0=m0*(127.0f/240.0f)+1e-30f,S1=m1*(127.0f/240.0f)+1e-30f,S2=m2*(127.0f/240.0f)+1e-30f,S3=m3*(127.0f/240.0f)+1e-30f;
            const float64 i0=1.0f/S0,i1=1.0f/S1,i2=1.0f/S2,i3=1.0f/S3;
            {
                int5 so={tile*256,local,0,0,0};
                v_f32_st_tnsr(so,scale,S0);so[0]+=64;v_f32_st_tnsr(so,scale,S1);so[0]+=64;
                v_f32_st_tnsr(so,scale,S2);so[0]+=64;v_f32_st_tnsr(so,scale,S3);
            }
            for(int b=0;b<blocks;++b) {
                int5 sp={codes_bytes+((group*blocks+b)*5+t)*1024,e,0,0,0};
                const float64 r0=(float64)v_u8_ld_tnsr_b(sp,w)*i0;sp[0]+=256;
                const float64 r1=(float64)v_u8_ld_tnsr_b(sp,w)*i1;sp[0]+=256;
                const float64 r2=(float64)v_u8_ld_tnsr_b(sp,w)*i2;sp[0]+=256;
                const float64 r3=(float64)v_u8_ld_tnsr_b(sp,w)*i3;
                // The same packing as iq4_expand_tiles.c's sa / sb.
                float128 a,c;a.v1=r0;a.v2=r2;c.v1=r1;c.v2=r3;
                const bfloat128 sa=v_convert_f32_to_bf16_all_b(a,0),sb=v_convert_f32_to_bf16_all_b(c,0);
                int5 at={((group*blocks+b)*80+t)*256,e,0,0,0};
                int5 o={tile*256,b*32,local,0,0};
                // Each code vector is loaded right before its two nibbles are used (a 16-vector array spilled to VLM).
                #pragma unroll
                for(int i=0;i<16;++i) {
                    const uchar256 codes=v_u8_ld_tnsr_b(at,w);at[0]+=1280;
                    #pragma unroll
                    for(int nibble=0;nibble<2;++nibble) {
                        const uchar256 code=nibble?codes>>4:codes;
                        const char256 decoded=(char256)v_u8_shuffle_b(lut,code|(unsigned char)0x80,0,lut);
                        const bfloat256 q=v_convert_i8_to_bf16_all_b(decoded,0);
                        bfloat256 x;x.v1=v_bf16_mul_b(q.v1,sa,0);x.v2=v_bf16_mul_b(q.v2,sb,0);
                        v_f8_st_tnsr(o,out,v_convert_bf16_to_f8_all_b(x,0));
                        o[1]+=1;
                    }
                }
            }
        }
    }
}
