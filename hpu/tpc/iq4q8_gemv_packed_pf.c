// STRATA_HPU_GEMV_PF: iq4q8_gemv_packed.c with its software-pipelined load branch selected. The next column pair's
// five code vectors are issued before this pair's MACs consume theirs, so one member keeps twice as many global
// loads in flight - the lever at the decode shape, where eleven slots leave one member per TPC and the rate is
// latency-bound, not bandwidth-bound. MAC order (c ascending, v0..v4, w0 then w1) is identical to the plain loop,
// so the outputs are bit-identical to iq4q8_gemv_packed.c.
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
        const bool is_shared=slot%period==period-1;
        const int W=is_shared?5:4,erow=is_shared?0:id,xrow=slot/x_group;
        for(int t=0;t<5;++t) for(int k=0;k<4;++k) partial[t][k]=0;
        for(int b=part*per_part;b<(part+1)*per_part;++b) {
            int5 dp={0,b,xrow,0,0};
            const float64 dxv=v_f32_ld_tnsr_b(dp,dx);
            int5 xp={0,2*b,xrow,0,0};
            const uchar256 xa=(uchar256)v_i8_ld_tnsr_b(xp,x);xp[1]+=1;
            const uchar256 xb=(uchar256)v_i8_ld_tnsr_b(xp,x);
            int256 acc0={0},acc1={0},acc2={0},acc3={0},acc4={0};
            const int base=((group*blocks+b)*16*5)*256;
            // Software pipeline: issue the following column pair's five loads before consuming this pair's.
            int5 at={base,erow,0,0,0};
            uchar256 n0=v_u8_ld_tnsr_b(at,W);at[0]+=256;
            uchar256 n1=v_u8_ld_tnsr_b(at,W);at[0]+=256;
            uchar256 n2=v_u8_ld_tnsr_b(at,W);at[0]+=256;
            uchar256 n3=v_u8_ld_tnsr_b(at,W);at[0]+=256;
            uchar256 n4=v_u8_ld_tnsr_b(at,W);at[0]+=256;
            for(int c=0;c<32;c+=2) {
                const uchar256 v0=n0,v1=n1,v2=n2,v3=n3,v4=n4;
                if(c+2<32) {
                    n0=v_u8_ld_tnsr_b(at,W);at[0]+=256;
                    n1=v_u8_ld_tnsr_b(at,W);at[0]+=256;
                    n2=v_u8_ld_tnsr_b(at,W);at[0]+=256;
                    n3=v_u8_ld_tnsr_b(at,W);at[0]+=256;
                    n4=v_u8_ld_tnsr_b(at,W);at[0]+=256;
                }
                const uchar256 xs=c<16?xa:xb,i0=(unsigned char)(0x80|(c&15)),i1=(unsigned char)(0x80|((c+1)&15));
                const char256 x0=(char256)v_u8_shuffle_b(xs,i0,0,xs);
                const char256 x1=(char256)v_u8_shuffle_b(xs,i1,0,xs);
                #define STEP(ACC,V) { \
                    const char256 w0=(char256)v_u8_shuffle_b(lut,V|(unsigned char)0x80,0,lut); \
                    const char256 w1=(char256)v_u8_shuffle_b(lut,(V>>4)|(unsigned char)0x80,0,lut); \
                    ACC=v_i8_mac_b(w1,x1,v_i8_mac_b(w0,x0,ACC,0,1,0),0,1,0); }
                STEP(acc0,v0) STEP(acc1,v1) STEP(acc2,v2) STEP(acc3,v3) STEP(acc4,v4)
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
