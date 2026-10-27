// STRATA_HPU_GEMV_TILE: iq4q8_gemv_packed.c with ONE 256-row output tile per member instead of five.
//
// Why: the packed GEMV is latency bound, not issue bound. The five-tile member holds five int256 accumulators, the
// table, both activation halves and five in-flight weight vectors at once, which is the whole TPC register file, so
// the next column pair's loads cannot be issued until the previous pair's MACs have consumed their vectors
// (measured: 237 GB/s at the decode shape = 10% of HBM peak). One tile per member leaves one accumulator, so the
// eight loads of a column-pair batch are issued together and 2 KB per member is in flight instead of 1.28 KB.
//
// The layout is the packed one (strata/hpu/resident_experts.hpp): tile T of a group is (5g+t)*256 rows, its codes sit
// at (((g*blocks+b)*16 + c/2)*5 + t)*256 and its scales at codes_bytes + (((g*blocks+b)*5 + t)*256 + i)*4.
//
// Not a new arithmetic: for every output element the MAC order is still block ascending, then column pair ascending,
// with the same two shuffles per load, so the result is bit-identical to iq4q8_gemv_packed.c.
__local__ float64 partial[4];
void main(tensor x,tensor dx,tensor table,tensor ids,tensor w,tensor shared,tensor out,
          int columns,int rows,int x_group,int parts,int slots,int period) {
    const int5 begin=get_index_space_offset(),end=begin+get_index_space_size();
    const int tiles=rows/256,blocks=columns/32,per_part=blocks/parts;
    const int codes_bytes=rows/2*columns;
    int5 zero={0,0,0,0,0};
    const uchar256 lut=v_u8_ld_tnsr_b(zero,table);
    for(int member=begin[0];member<end[0];++member) {
        const int slot=member/(parts*tiles),rest=member-slot*parts*tiles,part=rest/tiles,tile=rest-part*tiles;
        const int group=tile/5,t=tile-group*5;
        int5 ip={slot,0,0,0,0};
        const int id=s_i32_ld_g((__global__ int*)gen_addr(ip,ids));
        const bool is_shared=slot%period==period-1;
        const int W=is_shared?5:4,erow=is_shared?0:id,xrow=slot/x_group;
        for(int k=0;k<4;++k) partial[k]=0;
        for(int b=part*per_part;b<(part+1)*per_part;++b) {
            int5 dp={0,b,xrow,0,0};
            const float64 dxv=v_f32_ld_tnsr_b(dp,dx);
            int5 xp={0,2*b,xrow,0,0};
            const uchar256 xa=(uchar256)v_i8_ld_tnsr_b(xp,x);xp[1]+=1;
            const uchar256 xb=(uchar256)v_i8_ld_tnsr_b(xp,x);
            int256 acc={0};
            const int base=((group*blocks+b)*16*5)*256+t*256;
            // A 32-column block is 16 column pairs held in two activation halves (columns 0-15, 16-31), so each half
            // is exactly one batch of eight pairs whose loads can all be issued before any MAC reads them.
            // (`half` is a TPC type name, so the loop variable is hb.)
            for(int hb=0;hb<2;++hb) {
                const uchar256 xs=hb?xb:xa;
                int5 at={base+hb*8*5*256,erow,0,0,0};
                const uchar256 v0=v_u8_ld_tnsr_b(at,W);at[0]+=1280;
                const uchar256 v1=v_u8_ld_tnsr_b(at,W);at[0]+=1280;
                const uchar256 v2=v_u8_ld_tnsr_b(at,W);at[0]+=1280;
                const uchar256 v3=v_u8_ld_tnsr_b(at,W);at[0]+=1280;
                const uchar256 v4=v_u8_ld_tnsr_b(at,W);at[0]+=1280;
                const uchar256 v5=v_u8_ld_tnsr_b(at,W);at[0]+=1280;
                const uchar256 v6=v_u8_ld_tnsr_b(at,W);at[0]+=1280;
                const uchar256 v7=v_u8_ld_tnsr_b(at,W);
                #define PAIR(P,V) { \
                    const char256 x0=(char256)v_u8_shuffle_b(xs,(unsigned char)(0x80|(2*(P))),0,xs); \
                    const char256 x1=(char256)v_u8_shuffle_b(xs,(unsigned char)(0x80|(2*(P)+1)),0,xs); \
                    const char256 w0=(char256)v_u8_shuffle_b(lut,V|(unsigned char)0x80,0,lut); \
                    const char256 w1=(char256)v_u8_shuffle_b(lut,(V>>4)|(unsigned char)0x80,0,lut); \
                    acc=v_i8_mac_b(w1,x1,v_i8_mac_b(w0,x0,acc,0,1,0),0,1,0); }
                PAIR(0,v0) PAIR(1,v1) PAIR(2,v2) PAIR(3,v3)
                PAIR(4,v4) PAIR(5,v5) PAIR(6,v6) PAIR(7,v7)
                #undef PAIR
            }
            int5 sp={codes_bytes+((group*blocks+b)*5+t)*1024,erow,0,0,0};
            #define SCALE(I,K) { \
                const float64 s=(float64)v_u8_ld_tnsr_b(sp,W);sp[0]+=256; \
                partial[I]=v_f32_mac_b(v_convert_i32_to_f32_b(acc.v##K,0)*s,dxv,partial[I]); }
            SCALE(0,1) SCALE(1,2) SCALE(2,3) SCALE(3,4)
            #undef SCALE
        }
        for(int k=0;k<4;++k) {
            int5 o={tile*256+64*k,slot,part,0,0};
            v_f32_st_tnsr(o,out,partial[k]);
        }
    }
}
