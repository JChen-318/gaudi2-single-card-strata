// STRATA_HPU_HC_WINDOW_TPC: hc_down_bf16.c for a verify window of `rows` tokens: the HC read's down and inject
// projections with the read's activation, on the TPC instead of the MME (a [rows, HC*H] x [HC*H, rank+HC] product
// with few rows and a narrow output that the MME tiles poorly). Every (output, row) value is computed as
// hc_down_bf16.c computes it for one token (same products, same accumulation order, same reduction).
//   x {columns, rows} BF16; w {columns, rank + streams} BF16; inverse_streams {1} F32 -> lo {rank, rows} BF16;
//   inject {streams, rows} F32. Members own two output rows for every token row, four token rows at a time: per 128
//   columns two weight loads serve four x loads and eight multiply-adds. rank % 2 == 0, so a member's two outputs
//   are both lo or both inject.
#include "gdn_math.h"
void main(tensor x,tensor w,tensor inverse_streams,tensor lo,tensor inject,int columns,int rank,int streams,int rows) {
    const int5 begin=get_index_space_offset(),end=begin+get_index_space_size();
    const int total=rank+streams;
    int5 z={0,0,0,0,0};
    const float scale=s_f32_ld_g((__global__ float*)gen_addr(z,inverse_streams));
    for(int m=begin[0];m<end[0];++m) {
        const int j0=m*2,j1=j0+1<total?j0+1:total-1;
        for(int t0=0;t0<rows;t0+=4) {
            // Token rows past the window repeat the last one (computed, never stored).
            const int t1=t0+1<rows?t0+1:rows-1,t2=t0+2<rows?t0+2:rows-1,t3=t0+3<rows?t0+3:rows-1;
            float128 a0={0},a1={0},a2={0},a3={0},b0={0},b1={0},b2={0},b3={0};
            // 256 columns per step, all twelve loads issued before the sixteen multiply-adds (the compiler does not
            // pipeline the loop; loads issued just before their use stall on the full latency). columns % 256 == 0.
            for(int c=0;c<columns;c+=256) {
                int5 xp={c,t0,0,0,0},xq={c+128,t0,0,0,0};
                const bfloat128 x0=v_bf16_ld_tnsr_b(xp,x),y0=v_bf16_ld_tnsr_b(xq,x);xp[1]=t1;xq[1]=t1;
                const bfloat128 x1=v_bf16_ld_tnsr_b(xp,x),y1=v_bf16_ld_tnsr_b(xq,x);xp[1]=t2;xq[1]=t2;
                const bfloat128 x2=v_bf16_ld_tnsr_b(xp,x),y2=v_bf16_ld_tnsr_b(xq,x);xp[1]=t3;xq[1]=t3;
                const bfloat128 x3=v_bf16_ld_tnsr_b(xp,x),y3=v_bf16_ld_tnsr_b(xq,x);
                int5 wp={c,j0,0,0,0},wq={c+128,j0,0,0,0};
                const bfloat128 w0=v_bf16_ld_tnsr_b(wp,w),v0=v_bf16_ld_tnsr_b(wq,w);wp[1]=j1;wq[1]=j1;
                const bfloat128 w1=v_bf16_ld_tnsr_b(wp,w),v1=v_bf16_ld_tnsr_b(wq,w);
                a0=v_bf16_mac_acc32_b(w0,x0,a0);a1=v_bf16_mac_acc32_b(w0,x1,a1);a2=v_bf16_mac_acc32_b(w0,x2,a2);a3=v_bf16_mac_acc32_b(w0,x3,a3);
                b0=v_bf16_mac_acc32_b(w1,x0,b0);b1=v_bf16_mac_acc32_b(w1,x1,b1);b2=v_bf16_mac_acc32_b(w1,x2,b2);b3=v_bf16_mac_acc32_b(w1,x3,b3);
                a0=v_bf16_mac_acc32_b(v0,y0,a0);a1=v_bf16_mac_acc32_b(v0,y1,a1);a2=v_bf16_mac_acc32_b(v0,y2,a2);a3=v_bf16_mac_acc32_b(v0,y3,a3);
                b0=v_bf16_mac_acc32_b(v1,y0,b0);b1=v_bf16_mac_acc32_b(v1,y1,b1);b2=v_bf16_mac_acc32_b(v1,y2,b2);b3=v_bf16_mac_acc32_b(v1,y3,b3);
            }
            // Per token row: lane 0 output j0, lane 1 output j1.
            #define ROW(A,B,T) { \
                float64 d=0; \
                d=v_f32_sel_eq_u32_b(V_LANE_ID_32,0u,v_f32_reduce_add(A.v1+A.v2),d); \
                d=v_f32_sel_eq_u32_b(V_LANE_ID_32,1u,v_f32_reduce_add(B.v1+B.v2),d); \
                if(t0+T<rows) { \
                    if(j0<rank) { \
                        const float64 s=d*scale; \
                        float128 wide;wide.v1=s*strata_sigmoid(s);wide.v2=0; \
                        int5 o={j0,t0+T,0,0,0};v_bf16_st_tnsr_partial(o,lo,v_convert_f32_to_bf16_all_b(wide,SW_LINEAR),1,0); \
                    } else { \
                        int5 o={j0-rank,t0+T,0,0,0};v_f32_st_tnsr_partial(o,inject,d,total-j0-1<1?total-j0-1:1,0); \
                    } \
                } }
            ROW(a0,b0,0) ROW(a1,b1,1) ROW(a2,b2,2) ROW(a3,b3,3)
            #undef ROW
        }
    }
}
