#include "gdn_math.h"
// STRATA_HPU_PLE_GATE_SIMD: ple_gate.c with the key.query dot product over 64 lanes (a compensated sum per lane, then
// one lane reduction) instead of one scalar load pair per column. Same gate formula; the dot's summation order
// differs, so the gate can differ in the last bits. key/query {streams, width}, value {width}, output {streams, width}.
void main(tensor key,tensor query,tensor value,tensor output,int width,int streams,float inverse_sqrt_width) {
    const int5 start=get_index_space_offset(),end=start+get_index_space_size();
    for(int row=start[0];row<end[0] && row<streams;++row) {
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
            int5 p={col,row,0,0,0},v={col,0,0,0,0};
            v_f32_st_tnsr(p,output,v_f32_ld_tnsr_b(v,value)*gate);
        }
    }
}
