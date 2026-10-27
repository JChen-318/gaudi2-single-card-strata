// STRATA_HPU_RESIDENT_IQ4 expert GEMV: IQ4 codebook weights x Q8 activations, experts addressed by id.
// One member = (slot, column part, group of 5 row tiles of 256 rows).
//   out[part][slot][r] = sum over the part's 32-column blocks b of scale[r][b] * dx[b] * sum_c kv[code[r][c]] * xq[c]
// The block sum is an exact int32 (int8 codebook values times int8 activation codes); scales are F32.
// Expert buffer (one row of `w` per expert id; `shared` holds the last slot's expert), group-major so every member
// streams one contiguous range (layout written by strata/hpu/resident_experts.hpp):
//   codes  at (((g*blocks + b)*16 + c/2)*5 + t)*256: byte L holds column c (low nibble) and c+1 (high nibble) of
//          tile row 64*(L%4) + L/4, so the four int32 accumulator vectors (acc.vK[j] = lane 4j+K-1) are rows
//          64(K-1)..64K-1 in order;
//   scales at codes_bytes + ((g*blocks + b)*5 + t)*1024: 256 F32, tile rows in natural order.
// x: U8 {256, columns, xrows}, column c's int8 code in every lane; dx: F32 {64, blocks, xrows} (block scale
// repeated); ids: I32 {slots}; table: kv[i & 15] at lane i.
// u8 shuffle reads mask bits 0-4 (bit 5 swaps to the neighbouring group, bit 6 is ignored) and needs bit 7, so a
// code byte | 0x80 indexes its low nibble directly against the replicated table. mac_x2 is not used: the compiler's
// schedule corrupts its second operand pair in this loop.
__local__ float64 partial[5][4];
void main(tensor x,tensor dx,tensor table,tensor ids,tensor w,tensor shared,tensor out,
          int columns,int rows,int x_group,int parts,int slots,int period) {
    const int5 begin=get_index_space_offset(),end=begin+get_index_space_size();
    const int tiles=rows/256,groups=tiles/5,blocks=columns/32,per_part=blocks/parts;
    const int codes_bytes=rows/2*columns;
    int5 zero={0,0,0,0,0};
    const uchar256 lut=v_u8_ld_tnsr_b(zero,table);
    for(int member=begin[0];member<end[0];++member) {
        const int slot=member/(parts*groups),rest=member-slot*parts*groups,part=rest/groups,group=rest-part*groups;
        int5 ip={slot,0,0,0,0};
        const int id=s_i32_ld_g((__global__ int*)gen_addr(ip,ids));
        // Slot s reads x row s/x_group; every period-th slot (the last of each row's group) is the shared expert.
        const bool is_shared=slot%period==period-1;
        const int W=is_shared?5:4,erow=is_shared?0:id,xrow=slot/x_group;
        for(int t=0;t<5;++t) for(int k=0;k<4;++k) partial[t][k]=0;
        for(int b=part*per_part;b<(part+1)*per_part;++b) {
            int5 dp={0,b,xrow,0,0};
            const float64 dxv=v_f32_ld_tnsr_b(dp,dx);
            int5 xp={0,b*32,xrow,0,0};
            int256 acc0={0},acc1={0},acc2={0},acc3={0},acc4={0};
            int5 at={((group*blocks+b)*16*5)*256,erow,0,0,0};
            for(int c=0;c<32;c+=2) {
                const char256 x0=v_i8_ld_tnsr_b(xp,x);xp[1]+=1;
                const char256 x1=v_i8_ld_tnsr_b(xp,x);xp[1]+=1;
                #define STEP(ACC) { \
                    const uchar256 v=v_u8_ld_tnsr_b(at,W);at[0]+=256; \
                    const char256 w0=(char256)v_u8_shuffle_b(lut,v|(unsigned char)0x80,0,lut); \
                    const char256 w1=(char256)v_u8_shuffle_b(lut,(v>>4)|(unsigned char)0x80,0,lut); \
                    ACC=v_i8_mac_b(w1,x1,v_i8_mac_b(w0,x0,ACC,0,1,0),0,1,0); }
                STEP(acc0) STEP(acc1) STEP(acc2) STEP(acc3) STEP(acc4)
                #undef STEP
            }
            int5 sp={codes_bytes+((group*blocks+b)*5)*1024,erow,0,0,0};
            #define SCALE(T,ACC) { \
                const float64 s0=(float64)v_u8_ld_tnsr_b(sp,W);sp[0]+=256;const float64 s1=(float64)v_u8_ld_tnsr_b(sp,W);sp[0]+=256; \
                const float64 s2=(float64)v_u8_ld_tnsr_b(sp,W);sp[0]+=256;const float64 s3=(float64)v_u8_ld_tnsr_b(sp,W);sp[0]+=256; \
                partial[T][0]=v_f32_mac_b(v_convert_i32_to_f32_b(ACC.v1,0)*s0,dxv,partial[T][0]); \
                partial[T][1]=v_f32_mac_b(v_convert_i32_to_f32_b(ACC.v2,0)*s1,dxv,partial[T][1]); \
                partial[T][2]=v_f32_mac_b(v_convert_i32_to_f32_b(ACC.v3,0)*s2,dxv,partial[T][2]); \
                partial[T][3]=v_f32_mac_b(v_convert_i32_to_f32_b(ACC.v4,0)*s3,dxv,partial[T][3]); }
            SCALE(0,acc0) SCALE(1,acc1) SCALE(2,acc2) SCALE(3,acc3) SCALE(4,acc4)
            #undef SCALE
        }
        for(int t=0;t<5;++t) for(int k=0;k<4;++k) {
            int5 o={(group*5+t)*256+64*k,slot,part,0,0};
            v_f32_st_tnsr(o,out,partial[t][k]);
        }
    }
}
