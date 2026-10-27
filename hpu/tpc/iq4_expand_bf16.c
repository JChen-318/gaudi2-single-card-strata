// STRATA_HPU_MOE_MME: resident IQ4 rows (strata/hpu/resident_experts.hpp layout) of experts [first, first+count)
// expanded once per window into the grouped MME's BF16 B operand. Graph shape [count][columns][rows] = TPC
// {rows, columns, count}, so one 256-lane code vector of tile t is one 256-row stretch of one column.
// variant 4 (production): one i8 -> BF16 convert and a BF16 multiply by the scale rounded to BF16 (two roundings).
//   The convert interleaves the rows: within each 256-row tile, position 128*h + 2*i + p holds row
//   64*h + 128*p + i (moe_mme_act.c and moe_mme_join_perm.c read them back in natural order).
// variant 1 (exact): kv[code] * scale in F32 (exact: |kv| <= 127, scale = d or d*(ls-32)) rounded once to BF16,
//   rows in natural order: lane L of the i8 -> F32 convert is row 64*(L%4)+L/4, returned as four contiguous
//   64-row vectors. About twice the instructions of variant 4.
// Variants 0, 2, 3 are diagnostics: 0 is 1 without the preloaded code vectors, 2 loads and stores only (garbage
// output: the memory floor), 3 is 1 without the scale multiply (wrong values).
// One member per (256-row tile, 32-column block, expert), flattened so every TPC gets work even at count 1.
#define NIBBLE(PACKED,SHIFT,MUL) { \
    const uchar256 code=SHIFT?(PACKED)>>4:(PACKED); \
    const char256 decoded=(char256)v_u8_shuffle_b(lut,code|(unsigned char)0x80,0,lut); \
    const float256 f=v_convert_i8_to_f32_all_b(decoded,0); \
    float128 lo,hi; \
    if(MUL) {lo.v1=f.v1*s0;lo.v2=f.v2*s1;hi.v1=f.v3*s2;hi.v2=f.v4*s3;} else {lo.v1=f.v1;lo.v2=f.v2;hi.v1=f.v3;hi.v2=f.v4;} \
    o[0]=tile*256;v_bf16_st_tnsr(o,out,v_convert_f32_to_bf16_all_b(lo,SW_LINEAR)); \
    o[0]+=128;v_bf16_st_tnsr(o,out,v_convert_f32_to_bf16_all_b(hi,SW_LINEAR)); \
    o[1]+=1; }
#define RAW(PACKED) { \
    o[0]=tile*256;v_bf16_st_tnsr(o,out,(bfloat128)(PACKED));o[0]+=128;v_bf16_st_tnsr(o,out,(bfloat128)(PACKED));o[1]+=1; \
    o[0]=tile*256;v_bf16_st_tnsr(o,out,(bfloat128)(PACKED));o[0]+=128;v_bf16_st_tnsr(o,out,(bfloat128)(PACKED));o[1]+=1; }
void main(tensor table,tensor w,tensor out,int rows,int columns,int variant,int first) {
    const int5 begin=get_index_space_offset(),end=begin+get_index_space_size();
    const int blocks=columns/32,tiles=rows/256,codes_bytes=rows/2*columns;
    int5 zero={0,0,0,0,0};
    const uchar256 lut=v_u8_ld_tnsr_b(zero,table);
    for(int member=begin[0];member<end[0];++member) {
        const int tile=member%tiles,rest=member/tiles,b=rest%blocks,local=rest/blocks,e=first+local;
        const int group=tile/5,t=tile-5*group;
        int5 sp={codes_bytes+((group*blocks+b)*5+t)*1024,e,0,0,0};
        const float64 s0=(float64)v_u8_ld_tnsr_b(sp,w);sp[0]+=256;
        const float64 s1=(float64)v_u8_ld_tnsr_b(sp,w);sp[0]+=256;
        const float64 s2=(float64)v_u8_ld_tnsr_b(sp,w);sp[0]+=256;
        const float64 s3=(float64)v_u8_ld_tnsr_b(sp,w);
        int5 at={((group*blocks+b)*80+t)*256,e,0,0,0};
        int5 o={tile*256,b*32,local,0,0};
        if(variant==4) {
            float128 a,c;a.v1=s0;a.v2=s2;c.v1=s1;c.v2=s3;
            const bfloat128 sa=v_convert_f32_to_bf16_all_b(a,0),sb=v_convert_f32_to_bf16_all_b(c,0);
            for(int half_=0;half_<2;++half_) {
                uchar256 p[8];
                for(int i=0;i<8;++i) {p[i]=v_u8_ld_tnsr_b(at,w);at[0]+=1280;}
                for(int i=0;i<8;++i) for(int nibble=0;nibble<2;++nibble) {
                    const uchar256 code=nibble?p[i]>>4:p[i];
                    const char256 decoded=(char256)v_u8_shuffle_b(lut,code|(unsigned char)0x80,0,lut);
                    const bfloat256 q=v_convert_i8_to_bf16_all_b(decoded,0);
                    o[0]=tile*256;v_bf16_st_tnsr(o,out,v_bf16_mul_b(q.v1,sa,0));
                    o[0]+=128;v_bf16_st_tnsr(o,out,v_bf16_mul_b(q.v2,sb,0));
                    o[1]+=1;
                }
            }
            continue;
        }
        if(variant==0) {
            for(int c=0;c<32;c+=2) {
                const uchar256 packed=v_u8_ld_tnsr_b(at,w);at[0]+=5*256;
                NIBBLE(packed,0,1) NIBBLE(packed,1,1)
            }
            continue;
        }
        // Two halves of eight code vectors each, all eight loads issued before the first use.
        for(int half_=0;half_<2;++half_) {
            const uchar256 p0=v_u8_ld_tnsr_b(at,w);at[0]+=1280;
            const uchar256 p1=v_u8_ld_tnsr_b(at,w);at[0]+=1280;
            const uchar256 p2=v_u8_ld_tnsr_b(at,w);at[0]+=1280;
            const uchar256 p3=v_u8_ld_tnsr_b(at,w);at[0]+=1280;
            const uchar256 p4=v_u8_ld_tnsr_b(at,w);at[0]+=1280;
            const uchar256 p5=v_u8_ld_tnsr_b(at,w);at[0]+=1280;
            const uchar256 p6=v_u8_ld_tnsr_b(at,w);at[0]+=1280;
            const uchar256 p7=v_u8_ld_tnsr_b(at,w);at[0]+=1280;
            if(variant==2) {RAW(p0) RAW(p1) RAW(p2) RAW(p3) RAW(p4) RAW(p5) RAW(p6) RAW(p7)}
            else if(variant==3) {
                NIBBLE(p0,0,0) NIBBLE(p0,1,0) NIBBLE(p1,0,0) NIBBLE(p1,1,0) NIBBLE(p2,0,0) NIBBLE(p2,1,0) NIBBLE(p3,0,0) NIBBLE(p3,1,0)
                NIBBLE(p4,0,0) NIBBLE(p4,1,0) NIBBLE(p5,0,0) NIBBLE(p5,1,0) NIBBLE(p6,0,0) NIBBLE(p6,1,0) NIBBLE(p7,0,0) NIBBLE(p7,1,0)
            } else {
                NIBBLE(p0,0,1) NIBBLE(p0,1,1) NIBBLE(p1,0,1) NIBBLE(p1,1,1) NIBBLE(p2,0,1) NIBBLE(p2,1,1) NIBBLE(p3,0,1) NIBBLE(p3,1,1)
                NIBBLE(p4,0,1) NIBBLE(p4,1,1) NIBBLE(p5,0,1) NIBBLE(p5,1,1) NIBBLE(p6,0,1) NIBBLE(p6,1,1) NIBBLE(p7,0,1) NIBBLE(p7,1,1)
            }
        }
    }
}
