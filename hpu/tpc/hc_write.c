#include "gdn_math.h"
void main(tensor residual,tensor block,tensor inject,tensor output,int width,int streams) {
    const int5 start=get_index_space_offset(),end=start+get_index_space_size();
    for(int row=start[1];row<end[1] && row<streams;++row) {
        int5 ic={row,0,0,0,0};
        float64 scalar=s_f32_ld_g((__global__ float*)gen_addr(ic,inject))*(1.0f/streams);
        float64 weight=strata_sigmoid(scalar)*2.0f;
        for(int tile=start[0];tile<end[0] && tile*64<width;++tile) {
            int5 p={tile*64,row,0,0,0},b={tile*64,0,0,0,0};
            v_f32_st_tnsr(p,output,v_f32_ld_tnsr_b(p,residual)+v_f32_ld_tnsr_b(b,block)*weight);
        }
    }
}
