// STRATA_HPU_MOE_MME tiles: iq4_expand_bf16.c variant 4 (one i8 -> BF16 convert and a BF16 scale multiply, rows
// interleaved within each 256-row tile as moe_mme_act.c / moe_mme_join_perm.c expect) for the experts of tiles
// [first, first+count): local tile l gets the weights of expert tile_expert[first+l], or nothing for an unused tile
// (-1). table {256} U8, w {bytes, experts} U8, tile_expert {tiles} I32 -> out {rows, columns, count} BF16.
// Members are (256-row tile, local tile): one expert id load per member.
void main(tensor table,tensor w,tensor tile_expert,tensor out,int rows,int columns,int variant,int first) {
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
            for(int b=0;b<blocks;++b) {
                int5 sp={codes_bytes+((group*blocks+b)*5+t)*1024,e,0,0,0};
                const float64 s0=(float64)v_u8_ld_tnsr_b(sp,w);sp[0]+=256;
                const float64 s1=(float64)v_u8_ld_tnsr_b(sp,w);sp[0]+=256;
                const float64 s2=(float64)v_u8_ld_tnsr_b(sp,w);sp[0]+=256;
                const float64 s3=(float64)v_u8_ld_tnsr_b(sp,w);
                float128 a,c;a.v1=s0;a.v2=s2;c.v1=s1;c.v2=s3;
                const bfloat128 sa=v_convert_f32_to_bf16_all_b(a,0),sb=v_convert_f32_to_bf16_all_b(c,0);
                int5 at={((group*blocks+b)*80+t)*256,e,0,0,0};
                int5 o={tile*256,b*32,local,0,0};
                // Each code vector is loaded right before its two nibbles are used (a sixteen-vector array was spilled
                // to VLM and reloaded). Same values, same store order.
                #pragma unroll
                for(int i=0;i<16;++i) {
                    const uchar256 codes=v_u8_ld_tnsr_b(at,w);at[0]+=1280;
                    #pragma unroll
                    for(int nibble=0;nibble<2;++nibble) {
                        const uchar256 code=nibble?codes>>4:codes;
                        const char256 decoded=(char256)v_u8_shuffle_b(lut,code|(unsigned char)0x80,0,lut);
                        const bfloat256 q=v_convert_i8_to_bf16_all_b(decoded,0);
                        o[0]=tile*256;v_bf16_st_tnsr(o,out,v_bf16_mul_b(q.v1,sa,0));
                        o[0]+=128;v_bf16_st_tnsr(o,out,v_bf16_mul_b(q.v2,sb,0));
                        o[1]+=1;
                    }
                }
            }
        }
    }
}
