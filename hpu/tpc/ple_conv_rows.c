#include "gdn_math.h"
// A whole window's PLE causal conv in one node (ple_conv.c row after row, without the row loop). With the history H
// ((taps-1)*dilation rows, oldest first) followed by the window's normalized rows X as one sequence E = [H; X], row
// r's taps read E[r + tap*dilation] (the last tap is X[r] itself), so every output row is independent; the next
// history is E's last (taps-1)*dilation rows. Same products and sums in the same order: the same bits as ple_conv.c.
//   history {channels, depth}, normalized/hidden/gated {channels, rows}, weights {channels, taps}
//   -> next {channels, depth}, output {channels, rows}. Members: (64-channel tile, block of 64 rows).
void main(tensor history,tensor normalized,tensor weights,tensor hidden,tensor gated,
          tensor next,tensor output,int channels,int taps,int dilation,int rows) {
    const int5 start=get_index_space_offset(),end=start+get_index_space_size();
    const int depth=(taps-1)*dilation;
    for(int tile=start[0];tile<end[0] && tile*64<channels;++tile) {
        for(int block=start[1];block<end[1] && block*64<rows;++block) {
            const int last=block*64+64<rows?block*64+64:rows;
            for(int row=block*64;row<last;++row) {
                float64 sum=0;
                for(int tap=0;tap<taps;++tap) {
                    const int at=row+tap*dilation; // index into E = [history; normalized]
                    int5 w={tile*64,tap,0,0,0},h={tile*64,at,0,0,0},x={tile*64,at-depth,0,0,0};
                    const float64 input=at<depth?v_f32_ld_tnsr_b(h,history):v_f32_ld_tnsr_b(x,normalized);
                    sum+=input*v_f32_ld_tnsr_b(w,weights);
                }
                int5 p={tile*64,row,0,0,0};
                const float64 result=(v_f32_ld_tnsr_b(p,hidden)+v_f32_ld_tnsr_b(p,gated))+sum*strata_sigmoid(sum);
                v_f32_st_tnsr(p,output,result);
            }
            if(block==0) {
                // The next history: E[rows .. rows + depth).
                for(int i=0;i<depth;++i) {
                    const int at=rows+i;
                    int5 h={tile*64,at,0,0,0},x={tile*64,at-depth,0,0,0},dst={tile*64,i,0,0,0};
                    v_f32_st_tnsr(dst,next,at<depth?v_f32_ld_tnsr_b(h,history):v_f32_ld_tnsr_b(x,normalized));
                }
            }
        }
    }
}
