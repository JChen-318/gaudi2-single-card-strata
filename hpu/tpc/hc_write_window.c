#include "gdn_math.h"
// Members are (8 64-lane column tiles, stream, token): the stream gate's scalar load and sigmoid once per member, and
// the eight tiles' loads in straight-line code so they overlap. out = residual + block * 2 sigmoid(inject / streams).
void main(tensor residual,tensor block,tensor inject,tensor output,int width,int streams,int rows) {
    const int5 start=get_index_space_offset(),end=start+get_index_space_size();
    const int tiles=width/64;
    for(int token=start[2];token<end[2] && token<rows;++token)
    for(int stream=start[1];stream<end[1] && stream<streams;++stream) {
        int5 ic={stream,token,0,0,0};
        float64 scalar=s_f32_ld_g((__global__ float*)gen_addr(ic,inject))*(1.0f/streams);
        float64 weight=strata_sigmoid(scalar)*2.0f;
        for(int group=start[0];group<end[0] && group*8<tiles;++group) {
            if(group*8+8<=tiles) {
                float64 r[8],b[8];
                STRATA_UNROLL
                for(int i=0;i<8;++i) {
                    int5 p={(group*8+i)*64,stream,token,0,0},q={(group*8+i)*64,token,0,0,0};
                    r[i]=v_f32_ld_tnsr_b(p,residual);b[i]=v_f32_ld_tnsr_b(q,block);
                }
                STRATA_UNROLL
                for(int i=0;i<8;++i) {
                    int5 p={(group*8+i)*64,stream,token,0,0};
                    v_f32_st_tnsr(p,output,r[i]+b[i]*weight);
                }
            } else
            for(int tile=group*8;tile<tiles;++tile) {
                int5 p={tile*64,stream,token,0,0},b={tile*64,token,0,0,0};
                v_f32_st_tnsr(p,output,v_f32_ld_tnsr_b(p,residual)+v_f32_ld_tnsr_b(b,block)*weight);
            }
        }
    }
}
