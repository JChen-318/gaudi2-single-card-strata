#include "gdn_math.h"
void main(tensor normalized,tensor gate,tensor output,int width,int streams) {
    const int5 start=get_index_space_offset(),end=start+get_index_space_size();
    for(int tile=start[0];tile<end[0] && tile*64<width;++tile) {
        float64 sum=0;
        for(int row=0;row<streams;++row) {
            int5 p={tile*64,row,0,0,0};
            sum+=v_f32_ld_tnsr_b(p,normalized)*strata_sigmoid(v_f32_ld_tnsr_b(p,gate));
        }
        int5 p={tile*64,0,0,0,0};v_f32_st_tnsr(p,output,sum*(1.0f/streams));
    }
}
