// Grouped pairs of matching experts share each weight decode across two rows.
// Three independent output tiles at a time; a boundary/odd slot uses the
// production packed loop. Every slot retains its original arithmetic order.
__local__ float64 partial[5][4],pa[3][4],pb[3][4];
void main(tensor x,tensor dx,tensor table,tensor ids,tensor w,tensor shared,tensor order,tensor out,
          int columns,int rows,int x_group,int parts,int slots,int period) {
    const int5 begin=get_index_space_offset(),end=begin+get_index_space_size();
    const int pairs=(slots+1)/2,blocks=columns/32,per_part=blocks/parts,codes_bytes=rows/2*columns;
    int5 zero={0,0,0,0,0};const uchar256 lut=v_u8_ld_tnsr_b(zero,table);
    for(int member=begin[0];member<end[0];++member) {
        const int group=member/(parts*pairs),rest=member-group*parts*pairs,part=rest/pairs,pair=rest-part*pairs;
        int5 oa={2*pair,0,0,0,0},ob={2*pair+1,0,0,0,0};
        const int slot_a=(int)s_f32_ld_g((__global__ float*)gen_addr(oa,order));
        const bool has_b=2*pair+1<slots;
        int slot_b=slot_a;if(has_b) slot_b=(int)s_f32_ld_g((__global__ float*)gen_addr(ob,order));
        int5 ia={slot_a,0,0,0,0},ib={slot_b,0,0,0,0};
        const int ida=s_i32_ld_g((__global__ int*)gen_addr(ia,ids)),idb=s_i32_ld_g((__global__ int*)gen_addr(ib,ids));
        const bool sha=slot_a%period==period-1,shb=slot_b%period==period-1;
        const bool same=has_b && sha==shb && (sha || ida==idb);
        const int W=sha?5:4,erow=sha?0:ida;
        const int xrow_a=slot_a/x_group,xrow_b=slot_b/x_group;
        if(!same) {
            for(int side=0;side<(has_b?2:1);++side) {
                const int slot=side?slot_b:slot_a,xrow=slot/x_group;
                const bool sh=slot%period==period-1;
                const int W=sh?5:4,erow=sh?0:(side?idb:ida);
        for(int t=0;t<5;++t) for(int k=0;k<4;++k) partial[t][k]=0;
        for(int b=part*per_part;b<(part+1)*per_part;++b) {
            int5 dp={0,b,xrow,0,0};
            const float64 dxv=v_f32_ld_tnsr_b(dp,dx);
            int5 xp={0,2*b,xrow,0,0};
            const uchar256 xa=(uchar256)v_i8_ld_tnsr_b(xp,x);xp[1]+=1;
            const uchar256 xb=(uchar256)v_i8_ld_tnsr_b(xp,x);
            int256 acc0={0},acc1={0},acc2={0},acc3={0},acc4={0};
            const int base=((group*blocks+b)*16*5)*256;
                for(int c=0;c<32;c+=2) {
                    int5 at={base+(c/2)*5*256,erow,0,0,0};
                    const uchar256 xs=c<16?xa:xb,i0=(unsigned char)(0x80|(c&15)),i1=(unsigned char)(0x80|((c+1)&15));
                    const char256 x0=(char256)v_u8_shuffle_b(xs,i0,0,xs);
                    const char256 x1=(char256)v_u8_shuffle_b(xs,i1,0,xs);
                    const uchar256 v0=v_u8_ld_tnsr_b(at,W);at[0]+=256;
                    const uchar256 v1=v_u8_ld_tnsr_b(at,W);at[0]+=256;
                    const uchar256 v2=v_u8_ld_tnsr_b(at,W);at[0]+=256;
                    const uchar256 v3=v_u8_ld_tnsr_b(at,W);at[0]+=256;
                    const uchar256 v4=v_u8_ld_tnsr_b(at,W);
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
            continue;
        }
            for(int pass=0;pass<2;++pass) {
                const int t0=pass*3,nt=pass?2:3;
                for(int t=0;t<3;++t) for(int k=0;k<4;++k) {pa[t][k]=0;pb[t][k]=0;}
                for(int b=part*per_part;b<(part+1)*per_part;++b) {
                    int5 dpa={0,b,xrow_a,0,0},dpb={0,b,xrow_b,0,0};
                    const float64 dxa=v_f32_ld_tnsr_b(dpa,dx),dxb=v_f32_ld_tnsr_b(dpb,dx);
                    int5 xpa={0,2*b,xrow_a,0,0},xpb={0,2*b,xrow_b,0,0};
                    const uchar256 xa=(uchar256)v_i8_ld_tnsr_b(xpa,x);xpa[1]+=1;
                    const uchar256 xb=(uchar256)v_i8_ld_tnsr_b(xpa,x);
                    const uchar256 ya=(uchar256)v_i8_ld_tnsr_b(xpb,x);xpb[1]+=1;
                    const uchar256 yb=(uchar256)v_i8_ld_tnsr_b(xpb,x);
                    int256 a0={0},a1={0},a2={0},b0={0},b1={0},b2={0};
                    int5 at={((group*blocks+b)*16*5+t0)*256,erow,0,0,0};
                    for(int c=0;c<32;c+=2) {
                        const uchar256 ax=c<16?xa:xb,bx=c<16?ya:yb;
                        const uchar256 i0=(unsigned char)(0x80|(c&15)),i1=(unsigned char)(0x80|((c+1)&15));
                        const char256 xa0=(char256)v_u8_shuffle_b(ax,i0,0,ax),xa1=(char256)v_u8_shuffle_b(ax,i1,0,ax);
                        const char256 xb0=(char256)v_u8_shuffle_b(bx,i0,0,bx),xb1=(char256)v_u8_shuffle_b(bx,i1,0,bx);
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
