// STRATA_HPU_MOE_TAIL_FUSED (verify windows): the resident MoE's route weighting and per-row expert sum folded into the
// HC write: block[t] = sum_j experts[t*per_row+j] * route[t*per_row+j] (j in order), then hc_write_window.c's
// out = residual + block * 2 sigmoid(inject / streams) for every stream. Replaces the mult, reduce_sum and HC write
// nodes; the sum order may differ from reduce_sum_fwd_f32 in the last bits.
//   residual, output {rows, streams, width}; experts {rows*per_row, width}; route {rows*per_row}; inject {rows, streams}.
// Members are (8 64-lane column tiles, token): the token's route weights and stream gates are scalar loads, so they are
// read (and the gates' sigmoids computed) once per member, not once per tile.
#include "gdn_math.h"
__local__ float weights[64];
__local__ float64 doubled[32];
void main(tensor residual,tensor experts,tensor route,tensor inject,tensor output,int width,int streams,int rows,int per_row) {
    const int5 start=get_index_space_offset(),end=start+get_index_space_size();
    const int tiles=width/64;
    for(int token=start[1];token<end[1] && token<rows;++token) {
        for(int j=0;j<per_row;++j) {
            int5 rp={token*per_row+j,0,0,0,0};
            weights[j]=s_f32_ld_g((__global__ float*)gen_addr(rp,route));
        }
        // The stream gates 2 sigmoid(inject / streams) depend only on the token: computed once per member.
        for(int stream=0;stream<streams;++stream) {
            int5 ic={stream,token,0,0,0};
            float64 scalar=s_f32_ld_g((__global__ float*)gen_addr(ic,inject))*(1.0f/streams);
            doubled[stream]=strata_sigmoid(scalar)*2.0f;
        }
        // Four tiles' expert sums at a time: four independent add chains (each tile's in j order, as before), so the
        // loads and adds of one tile hide the others' latency.
        for(int group=start[0];group<end[0];++group)
        for(int t0=group*8;t0<group*8+8 && t0<tiles;t0+=4) {
            if(t0+4<=tiles) {
                float64 block[4]={0,0,0,0};
                for(int j=0;j<per_row;++j) {
                    const float weight=weights[j];
                    #pragma unroll
                    for(int i=0;i<4;++i) {int5 ep={(t0+i)*64,token*per_row+j,0,0,0};block[i]+=v_f32_ld_tnsr_b(ep,experts)*weight;}
                }
                for(int stream=0;stream<streams;++stream) {
                    const float64 gate=doubled[stream];
                    #pragma unroll
                    for(int i=0;i<4;++i) {
                        int5 p={(t0+i)*64,stream,token,0,0};
                        v_f32_st_tnsr(p,output,v_f32_ld_tnsr_b(p,residual)+block[i]*gate);
                    }
                }
            } else
            for(int tile=t0;tile<tiles;++tile) {
                float64 block=0;
                for(int j=0;j<per_row;++j) {
                    int5 ep={tile*64,token*per_row+j,0,0,0};
                    block+=v_f32_ld_tnsr_b(ep,experts)*weights[j];
                }
                for(int stream=0;stream<streams;++stream) {
                    int5 p={tile*64,stream,token,0,0};
                    v_f32_st_tnsr(p,output,v_f32_ld_tnsr_b(p,residual)+block*doubled[stream]);
                }
            }
        }
    }
}
