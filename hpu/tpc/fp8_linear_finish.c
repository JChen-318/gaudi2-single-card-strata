// STRATA_HPU_FP8_DENSE: the end of GraphBuilder::fp8_linear in one node. both {columns, 2*rows} F32 is the stacked
// FP8 GEMM (rows 0..rows-1 the hi products, rows..2*rows-1 the lo ones), x_scale {1, rows} and w_scale {columns, 1}
// the activation and weight row scales: out[r][n] = (hi + lo/16) * x_scale[r] * w_scale[n], in the SDK graph's order.
// One member per (row, 64 output columns).
void main(tensor both,tensor x_scale,tensor w_scale,tensor out,int columns,int rows) {
    const int5 begin=get_index_space_offset(),end=begin+get_index_space_size();
    const int chunks=(columns+63)/64;
    for(int member=begin[0];member<end[0];++member) {
        const int row=member/chunks,n=(member-row*chunks)*64;
        int5 sp={0,row,0,0,0};
        const float xs=s_f32_ld_g((__global__ float*)gen_addr(sp,x_scale));
        int5 hp={n,row,0,0,0},lp={n,rows+row,0,0,0},wp={n,0,0,0,0};
        const float64 hi=v_f32_ld_tnsr_b(hp,both),lo=v_f32_ld_tnsr_b(lp,both),ws=v_f32_ld_tnsr_b(wp,w_scale);
        const float64 product=hi+lo*(1.0f/16.0f);
        const float64 result=(product*xs)*ws;
        const int count=columns-n<64?columns-n:64;
        v_f32_st_tnsr_partial(hp,out,result,count-1,0);
    }
}
