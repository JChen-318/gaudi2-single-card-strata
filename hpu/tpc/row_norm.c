// Per-row weighted RMS. Width is a multiple of 64. Scale the sum to avoid
// overflow; this is FP32 TPC arithmetic, not the CPU FP64 reduction contract.
void main(tensor input,tensor gamma,tensor output,int width,int rows,int shared_gamma,float epsilon) {
    const int5 start=get_index_space_offset(),end=start+get_index_space_size();
    for(int row=start[0];row<end[0] && row<rows;++row) {
        float maximum=1;
        for(int col=0;col<width;++col) {
            int5 p={col,row,0,0,0};float x=s_f32_ld_g((__global__ float*)gen_addr(p,input));
            x=x<0?-x:x;if(x>maximum) maximum=x;
        }
        const float im=1.0f/maximum;float sum=0,correction=0;
        for(int col=0;col<width;++col) {
            int5 p={col,row,0,0,0};float x=s_f32_ld_g((__global__ float*)gen_addr(p,input))*im;
            const float adjusted=x*x-correction,next=sum+adjusted;
            correction=(next-sum)-adjusted;sum=next;
        }
        float64 mean=sum/width+(epsilon*im)*im;
        const float64 inverse=v_rsqrt_f32(mean)*im;
        for(int col=0;col<width;col+=64) {
            // Legacy 0: independent rows; 1: one gamma. Negative values reuse
            // that many stream gammas across a token-major window.
            const int gamma_row=shared_gamma==1?0:(shared_gamma<0?row%(-shared_gamma):row);
            int5 p={col,row,0,0,0},g={col,gamma_row,0,0,0};
            v_f32_st_tnsr(p,output,(v_f32_ld_tnsr_b(p,input)*inverse)*v_f32_ld_tnsr_b(g,gamma));
        }
    }
}
