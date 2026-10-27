// STRATA_HPU_HC_DECODE_TPC: a decoded token's HC read down and inject projections on the TPC instead of the MME (which
// tiles a [1, HC*H] x [HC*H, rank+HC] product poorly), with the read's activation folded in:
//   d[j] = sum_c x[c] * w[j][c]   (BF16 products, F32 accumulation),
//   lo[j] = BF16(s * sigmoid(s)), s = d[j] * inverse_streams   for j < rank;   inject[j - rank] = d[j] for j >= rank.
//   x {columns, 1} BF16; w {columns, rank + streams} BF16; inverse_streams {1} F32 -> lo {rank, 1} BF16;
//   inject {streams, 1} F32. Members own eight output rows (glue: ceil((rank + streams) / 8); rank % 8 == 0, so a
//   member's rows are all lo or all inject). Per 128 columns one x load serves the eight rows' weight loads.
#include "gdn_math.h"
void main(tensor x,tensor w,tensor inverse_streams,tensor lo,tensor inject,int columns,int rank,int streams) {
    const int5 begin=get_index_space_offset(),end=begin+get_index_space_size();
    const int total=rank+streams;
    int5 z={0,0,0,0,0};
    const float scale=s_f32_ld_g((__global__ float*)gen_addr(z,inverse_streams));
    for(int m=begin[0];m<end[0];++m) {
        const int j0=m*8;
        // Rows past the end repeat the last one (computed, never stored).
        const int r0=j0,r1=j0+1<total?j0+1:total-1,r2=j0+2<total?j0+2:total-1,r3=j0+3<total?j0+3:total-1;
        const int r4=j0+4<total?j0+4:total-1,r5=j0+5<total?j0+5:total-1,r6=j0+6<total?j0+6:total-1,r7=j0+7<total?j0+7:total-1;
        float128 a0={0},a1={0},a2={0},a3={0},a4={0},a5={0},a6={0},a7={0};
        for(int c=0;c<columns;c+=128) {
            int5 xp={c,0,0,0,0};
            const bfloat128 xv=v_bf16_ld_tnsr_b(xp,x);
#define HC_ROW(A,R) {int5 p={c,R,0,0,0};A=v_bf16_mac_acc32_b(v_bf16_ld_tnsr_b(p,w),xv,A);}
            HC_ROW(a0,r0) HC_ROW(a1,r1) HC_ROW(a2,r2) HC_ROW(a3,r3) HC_ROW(a4,r4) HC_ROW(a5,r5) HC_ROW(a6,r6) HC_ROW(a7,r7)
#undef HC_ROW
        }
        float64 d=0;
#define HC_TOTAL(A,I) d=v_f32_sel_eq_u32_b(V_LANE_ID_32,(unsigned)I,v_f32_reduce_add(A.v1+A.v2),d);
        HC_TOTAL(a0,0) HC_TOTAL(a1,1) HC_TOTAL(a2,2) HC_TOTAL(a3,3) HC_TOTAL(a4,4) HC_TOTAL(a5,5) HC_TOTAL(a6,6) HC_TOTAL(a7,7)
#undef HC_TOTAL
        if(j0<rank) {
            const float64 s=d*scale;
            float128 wide;wide.v1=s*strata_sigmoid(s);wide.v2=0;
            int5 o={j0,0,0,0,0};v_bf16_st_tnsr_partial(o,lo,v_convert_f32_to_bf16_all_b(wide,SW_LINEAR),7,0);
        } else {
            int5 o={j0-rank,0,0,0,0};v_f32_st_tnsr_partial(o,inject,d,total-j0-1,0);
        }
    }
}
