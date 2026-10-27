// Opt-in SIMD RMS: 64 independent compensated accumulators followed by a tree reduction.
// The scalar reference remains strata_row_norm_f32; the reduction order differs here.
void main(tensor input,tensor gamma,tensor output,int width,int rows,int shared_gamma,float epsilon) {
    const int5 start=get_index_space_offset(),end=start+get_index_space_size();
    for(int row=start[0];row<end[0] && row<rows;++row) {
        float64 maximum=1.0f;
        for(int col=0;col<width;col+=64) {
            int5 p={col,row,0,0,0};
            const float64 x=v_f32_ld_tnsr_b(p,input);
            const float64 absolute=v_f32_sel_less_f32_b(x,0.0f,-x,x);
            maximum=v_f32_max_b(maximum,absolute);
        }
        maximum=v_f32_reduce_max(maximum);
        const float64 im=1.0f/maximum;
        float64 sum=0,correction=0;
        for(int col=0;col<width;col+=64) {
            int5 p={col,row,0,0,0};
            const float64 x=v_f32_ld_tnsr_b(p,input)*im;
            const float64 adjusted=x*x-correction,next=sum+adjusted;
            correction=(next-sum)-adjusted;sum=next;
        }
        sum=v_f32_reduce_add(sum);
        const float64 mean=sum*(1.0f/width)+(epsilon*im)*im;
        const float64 inverse=v_rsqrt_f32(mean)*im;
        const int gamma_row=shared_gamma==1?0:(shared_gamma<0?row%(-shared_gamma):row);
        for(int col=0;col<width;col+=64) {
            int5 p={col,row,0,0,0},g={col,gamma_row,0,0,0};
            v_f32_st_tnsr(p,output,(v_f32_ld_tnsr_b(p,input)*inverse)*v_f32_ld_tnsr_b(g,gamma));
        }
    }
}
