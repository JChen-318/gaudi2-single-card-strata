#include "gdn_math.h"
// State is time-major [history,channels], oldest first. It stores normalized
// gated values. Both state and weights must be transposed from GGUF/native state.
void main(tensor history,tensor normalized,tensor weights,tensor hidden,tensor gated,
          tensor next,tensor output,int channels,int taps,int dilation) {
    const int5 start=get_index_space_offset(),end=start+get_index_space_size();
    const int rows=(taps-1)*dilation;
    for(int tile=start[0];tile<end[0] && tile*64<channels;++tile) {
        int5 p={tile*64,0,0,0,0};const float64 current=v_f32_ld_tnsr_b(p,normalized);
        float64 sum=0;
        for(int tap=0;tap<taps;++tap) {
            int5 w={tile*64,tap,0,0,0},h={tile*64,tap*dilation,0,0,0};
            const float64 input=tap==taps-1?current:v_f32_ld_tnsr_b(h,history);
            sum+=input*v_f32_ld_tnsr_b(w,weights);
        }
        const float64 result=(v_f32_ld_tnsr_b(p,hidden)+v_f32_ld_tnsr_b(p,gated))+sum*strata_sigmoid(sum);
        v_f32_st_tnsr(p,output,result);
        for(int row=0;row<rows;++row) {
            int5 dst={tile*64,row,0,0,0},src={tile*64,row+1,0,0,0};
            v_f32_st_tnsr(dst,next,row==rows-1?current:v_f32_ld_tnsr_b(src,history));
        }
    }
}
