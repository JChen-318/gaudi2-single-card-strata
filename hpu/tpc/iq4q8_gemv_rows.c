// STRATA_HPU_DFLASH verify windows: iq4q8_gemv.c with each expert's weights read once for the slots sharing it
// (route_share.c). Same tensors, parameters, index space and arithmetic per slot as iq4q8_gemv.c, plus
//   share: I32 {16, slots}; share[slot][0] = 2 (this slot and partner share[slot][1]), 1 (this slot alone), 0 (skip).
// A pair is computed over the five row tiles in two passes (tiles 0-2, then 3-4) so both slots' accumulators fit the
// vector registers, decoding each weight vector once for both; a lone slot takes the loop of iq4q8_gemv.c. Every
// member does at most one pair. Per slot every block sum and its scaling are the same operations in the same order
// as there, so the outputs are identical.
__local__ float64 partial[5][4];
__local__ float64 pa[3][4],pb[3][4];
void main(tensor x,tensor dx,tensor table,tensor ids,tensor w,tensor shared,tensor share,tensor out,
          int columns,int rows,int x_group,int parts,int slots,int period) {
    const int5 begin=get_index_space_offset(),end=begin+get_index_space_size();
    const int tiles=rows/256,groups=tiles/5,blocks=columns/32,per_part=blocks/parts;
    const int codes_bytes=rows/2*columns;
    int5 zero={0,0,0,0,0};
    const uchar256 lut=v_u8_ld_tnsr_b(zero,table);
    for(int member=begin[0];member<end[0];++member) {
        const int slot=member/(parts*groups),rest=member-slot*parts*groups,part=rest/groups,group=rest-part*groups;
        int5 sp0={0,slot,0,0,0};
        const int count=s_i32_ld_g((__global__ int*)gen_addr(sp0,share));
        if(count<=0) continue;
        int5 ip={slot,0,0,0,0};
        const int id=s_i32_ld_g((__global__ int*)gen_addr(ip,ids));
        const bool is_shared=slot%period==period-1;
        const int W=is_shared?5:4,erow=is_shared?0:id;
        {
            const int slot_a=slot;
            if(count==1) {
                // ---- one slot: the loop of iq4q8_gemv.c ----
                const int xrow=slot_a/x_group;
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
                    int5 o={(group*5+t)*256+64*k,slot_a,part,0,0};
                    v_f32_st_tnsr(o,out,partial[t][k]);
                }
                continue;
            }
            // ---- two slots sharing the expert ----
            int5 sb={1,slot,0,0,0};
            const int slot_b=s_i32_ld_g((__global__ int*)gen_addr(sb,share));
            const int xrow_a=slot_a/x_group,xrow_b=slot_b/x_group;
            for(int pass=0;pass<2;++pass) {
                const int t0=pass*3,nt=pass?2:3;
                for(int t=0;t<3;++t) for(int k=0;k<4;++k) {pa[t][k]=0;pb[t][k]=0;}
                for(int b=part*per_part;b<(part+1)*per_part;++b) {
                    int5 dpa={0,b,xrow_a,0,0},dpb={0,b,xrow_b,0,0};
                    const float64 dxa=v_f32_ld_tnsr_b(dpa,dx),dxb=v_f32_ld_tnsr_b(dpb,dx);
                    int5 xpa={0,b*32,xrow_a,0,0},xpb={0,b*32,xrow_b,0,0};
                    int256 a0={0},a1={0},a2={0},b0={0},b1={0},b2={0};
                    int5 at={((group*blocks+b)*16*5+t0)*256,erow,0,0,0};
                    for(int c=0;c<32;c+=2) {
                        const char256 xa0=v_i8_ld_tnsr_b(xpa,x);xpa[1]+=1;
                        const char256 xa1=v_i8_ld_tnsr_b(xpa,x);xpa[1]+=1;
                        const char256 xb0=v_i8_ld_tnsr_b(xpb,x);xpb[1]+=1;
                        const char256 xb1=v_i8_ld_tnsr_b(xpb,x);xpb[1]+=1;
                        int5 tp=at;
                        #define STEP2(A,B) { \
                            const uchar256 v=v_u8_ld_tnsr_b(tp,W);tp[0]+=256; \
                            const char256 w0=(char256)v_u8_shuffle_b(lut,v|(unsigned char)0x80,0,lut); \
                            const char256 w1=(char256)v_u8_shuffle_b(lut,(v>>4)|(unsigned char)0x80,0,lut); \
                            A=v_i8_mac_b(w1,xa1,v_i8_mac_b(w0,xa0,A,0,1,0),0,1,0); \
                            B=v_i8_mac_b(w1,xb1,v_i8_mac_b(w0,xb0,B,0,1,0),0,1,0); }
                        STEP2(a0,b0) STEP2(a1,b1)
                        if(nt==3) STEP2(a2,b2)
                        #undef STEP2
                        at[0]+=5*256;
                    }
                    int5 sp={codes_bytes+((group*blocks+b)*5+t0)*1024,erow,0,0,0};
                    #define SCALE2(T,A,B) { \
                        const float64 s0=(float64)v_u8_ld_tnsr_b(sp,W);sp[0]+=256;const float64 s1=(float64)v_u8_ld_tnsr_b(sp,W);sp[0]+=256; \
                        const float64 s2=(float64)v_u8_ld_tnsr_b(sp,W);sp[0]+=256;const float64 s3=(float64)v_u8_ld_tnsr_b(sp,W);sp[0]+=256; \
                        pa[T][0]=v_f32_mac_b(v_convert_i32_to_f32_b(A.v1,0)*s0,dxa,pa[T][0]); \
                        pa[T][1]=v_f32_mac_b(v_convert_i32_to_f32_b(A.v2,0)*s1,dxa,pa[T][1]); \
                        pa[T][2]=v_f32_mac_b(v_convert_i32_to_f32_b(A.v3,0)*s2,dxa,pa[T][2]); \
                        pa[T][3]=v_f32_mac_b(v_convert_i32_to_f32_b(A.v4,0)*s3,dxa,pa[T][3]); \
                        pb[T][0]=v_f32_mac_b(v_convert_i32_to_f32_b(B.v1,0)*s0,dxb,pb[T][0]); \
                        pb[T][1]=v_f32_mac_b(v_convert_i32_to_f32_b(B.v2,0)*s1,dxb,pb[T][1]); \
                        pb[T][2]=v_f32_mac_b(v_convert_i32_to_f32_b(B.v3,0)*s2,dxb,pb[T][2]); \
                        pb[T][3]=v_f32_mac_b(v_convert_i32_to_f32_b(B.v4,0)*s3,dxb,pb[T][3]); }
                    SCALE2(0,a0,b0) SCALE2(1,a1,b1)
                    if(nt==3) SCALE2(2,a2,b2)
                    #undef SCALE2
                }
                for(int t=0;t<nt;++t) for(int k=0;k<4;++k) {
                    int5 oa={(group*5+t0+t)*256+64*k,slot_a,part,0,0},ob={(group*5+t0+t)*256+64*k,slot_b,part,0,0};
                    v_f32_st_tnsr(oa,out,pa[t][k]);v_f32_st_tnsr(ob,out,pb[t][k]);
                }
            }
        }
    }
}
