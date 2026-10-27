// STRATA_HPU_GEMV_TPC: a one-row BF16 GEMM out[n] = sum_c x[c] * w[n][c] (F32 accumulation) on the TPC. At one row
// the MME streams the weights at 1.3-1.6 TB/s; a TPC member with eight independent weight loads in flight per
// 128 columns and x held in local memory reads near the 2.2 TB/s the read probe (tests/bw_read.cpp) measures.
//   x {columns, 1} BF16 (columns % 128 == 0, <= 8192); w {columns, rows} BF16 -> out {rows, 1}, F32 or (rounded = 1)
//   BF16 rounded to nearest even like the MME's BF16 output. Members own eight output rows (glue: ceil(rows / 8)).
__local__ bfloat128 xl[64];
void main(tensor x,tensor w,tensor out,int columns,int rows,int rounded) {
    const int5 begin=get_index_space_offset(),end=begin+get_index_space_size();
    for(int c=0;c<columns;c+=128) {int5 p={c,0,0,0,0};xl[c/128]=v_bf16_ld_tnsr_b(p,x);}
    for(int m=begin[0];m<end[0];++m) {
        const int j0=m*8;
        // Rows past the end repeat the last one (computed, never stored).
        const int r0=j0,r1=j0+1<rows?j0+1:rows-1,r2=j0+2<rows?j0+2:rows-1,r3=j0+3<rows?j0+3:rows-1;
        const int r4=j0+4<rows?j0+4:rows-1,r5=j0+5<rows?j0+5:rows-1,r6=j0+6<rows?j0+6:rows-1,r7=j0+7<rows?j0+7:rows-1;
        float128 a0={0},a1={0},a2={0},a3={0},a4={0},a5={0},a6={0},a7={0};
        for(int c=0;c<columns;c+=128) {
            const bfloat128 xv=xl[c/128];
#define GEMV_ROW(A,R) {int5 p={c,R,0,0,0};A=v_bf16_mac_acc32_b(v_bf16_ld_tnsr_b(p,w),xv,A);}
            GEMV_ROW(a0,r0) GEMV_ROW(a1,r1) GEMV_ROW(a2,r2) GEMV_ROW(a3,r3) GEMV_ROW(a4,r4) GEMV_ROW(a5,r5) GEMV_ROW(a6,r6) GEMV_ROW(a7,r7)
#undef GEMV_ROW
        }
        float64 d=0;
#define GEMV_TOTAL(A,I) d=v_f32_sel_eq_u32_b(V_LANE_ID_32,(unsigned)I,v_f32_reduce_add(A.v1+A.v2),d);
        GEMV_TOTAL(a0,0) GEMV_TOTAL(a1,1) GEMV_TOTAL(a2,2) GEMV_TOTAL(a3,3) GEMV_TOTAL(a4,4) GEMV_TOTAL(a5,5) GEMV_TOTAL(a6,6) GEMV_TOTAL(a7,7)
#undef GEMV_TOTAL
        const int count=rows-j0<8?rows-j0:8;
        int5 o={j0,0,0,0,0};
        if(rounded) {
            float128 wide;wide.v1=d;wide.v2=0;
            v_bf16_st_tnsr_partial(o,out,v_convert_f32_to_bf16_all_b(wide,SW_LINEAR),count-1,0);
        } else v_f32_st_tnsr_partial(o,out,d,count-1,0);
    }
}
