// STRATA_HPU_FP8_DENSE: fp8_encode.c with the hi and lo codes of row r written as rows r and rows+r of one
// stacked {columns, 2*rows} tensor, so a single FP8 GEMM multiplies both against the weights (read once), and
// fp8_linear_finish.c combines the two halves. Same scale and code arithmetic as fp8_encode.c.
void main(tensor input,tensor stacked,tensor row_scale,int columns,int rows) {
    const int5 begin=get_index_space_offset(),end=begin+get_index_space_size();
    for(int row=begin[0];row<end[0] && row<rows;++row) {
        float64 peak=0;
        for(int col=0;col<columns;col+=64) {
            int5 p={col,row,0,0,0};
            const float64 x=v_f32_ld_tnsr_b(p,input);
            peak=v_f32_max_b(peak,v_f32_sel_less_f32_b(x,0.0f,-x,x));
        }
        peak=v_f32_reduce_max(peak);
        const float64 scale=peak*(1.0f/240.0f)+1e-30f;
        int5 sc={0,row,0,0,0};v_f32_st_tnsr(sc,row_scale,scale);
        for(int col=0;col<columns;col+=256) {
            int5 p={col,row,0,0,0};
            float256 x;
            x.v1=v_f32_ld_tnsr_b(p,input)/scale;p[0]+=64;
            x.v2=v_f32_ld_tnsr_b(p,input)/scale;p[0]+=64;
            x.v3=v_f32_ld_tnsr_b(p,input)/scale;p[0]+=64;
            x.v4=v_f32_ld_tnsr_b(p,input)/scale;
            const minifloat256 hi=v_convert_f32_to_f8_all_b(x,SW_LINEAR);
            const float256 restored=v_convert_f8_to_f32_all_b(hi,SW_LINEAR);
            float256 remainder;
            remainder.v1=(x.v1-restored.v1)*16.0f;remainder.v2=(x.v2-restored.v2)*16.0f;
            remainder.v3=(x.v3-restored.v3)*16.0f;remainder.v4=(x.v4-restored.v4)*16.0f;
            const minifloat256 lo=v_convert_f32_to_f8_all_b(remainder,SW_LINEAR);
            const int count=columns-col<256?columns-col:256;
            int5 h={col,row,0,0,0},l={col,rows+row,0,0,0};
            v_f8_st_tnsr_partial(h,stacked,hi,count-1,0);
            v_f8_st_tnsr_partial(l,stacked,lo,count-1,0);
        }
    }
}
