#include "gdn_math.h"
void main(tensor key,tensor query,tensor value,tensor output,int width,int streams,float inverse_sqrt_width) {
    const int5 start=get_index_space_offset(),end=start+get_index_space_size();
    for(int row=start[0];row<end[0] && row<streams;++row) {
        float dot=0,correction=0;
        for(int col=0;col<width;++col) {
            int5 p={col,row,0,0,0};
            const float product=s_f32_ld_g((__global__ float*)gen_addr(p,key))*s_f32_ld_g((__global__ float*)gen_addr(p,query));
            const float adjusted=product-correction,next=dot+adjusted;
            correction=(next-dot)-adjusted;dot=next;
        }
        const float score=dot*inverse_sqrt_width;
        float magnitude=score<0?-score:score;if(magnitude<1e-6f) magnitude=1e-6f;
        float64 m=magnitude,sign=score>0?1.0f:(score<0?-1.0f:0.0f);
        const float64 gate=strata_sigmoid(sign*(m*v_rsqrt_f32(m)));
        for(int col=0;col<width;col+=64) {
            int5 p={col,row,0,0,0},v={col,0,0,0,0};
            v_f32_st_tnsr(p,output,v_f32_ld_tnsr_b(v,value)*gate);
        }
    }
}
