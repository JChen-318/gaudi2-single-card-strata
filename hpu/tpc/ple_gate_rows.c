#include "gdn_math.h"
// A whole window's PLE gate in one node: ple_gate_simd.c for each of rows = window rows * streams key/query rows, the
// value row being that window row's (row / streams). Same arithmetic per row, so the same bits as one ple_gate_simd
// node per window row. key/query {width, rows}, value {width, rows / streams} -> output {width, rows}.
void main(tensor key,tensor query,tensor value,tensor output,int width,int rows,int streams,float inverse_sqrt_width) {
    const int5 start=get_index_space_offset(),end=start+get_index_space_size();
    for(int row=start[0];row<end[0] && row<rows;++row) {
        float64 dot=0,correction=0;
        for(int col=0;col<width;col+=64) {
            int5 p={col,row,0,0,0};
            const float64 product=v_f32_ld_tnsr_b(p,key)*v_f32_ld_tnsr_b(p,query);
            const float64 adjusted=product-correction,next=dot+adjusted;
            correction=(next-dot)-adjusted;dot=next;
        }
        const float64 score=v_f32_reduce_add(dot)*inverse_sqrt_width;
        const float64 absolute=v_f32_sel_less_f32_b(score,0.0f,-score,score);
        const float64 magnitude=v_f32_max_b(absolute,1e-6f);
        float64 sign=v_f32_sel_grt_f32_b(score,0.0f,1.0f,0.0f);
        sign=v_f32_sel_less_f32_b(score,0.0f,-1.0f,sign);
        const float64 gate=strata_sigmoid(sign*(magnitude*v_rsqrt_f32(magnitude)));
        for(int col=0;col<width;col+=64) {
            int5 p={col,row,0,0,0},v={col,row/streams,0,0,0};
            v_f32_st_tnsr(p,output,v_f32_ld_tnsr_b(v,value)*gate);
        }
    }
}
