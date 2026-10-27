// STRATA_HPU_MOE_MME + MOE_TAIL_FUSED: moe_mme_join_perm.c folded into hc_write_moe.c. The expert rows are read
// where the grouped GEMMs left them (BF16 in variant-4 column order: routed rows by `place`, the shared expert's row
// by token) instead of from a joined F32 copy; the sum over the token's slots and the HC write are hc_write_moe.c's
// in the same order, so the same bits. residual, output {width, streams, rows} F32; routed {width, tiles*rows} BF16;
// shared {width, rows} BF16; place {rows*per_row} I32 (-1: no row, zeros); route {rows*per_row} F32;
// inject {streams, rows} F32. The shared slot is the last of each token's per_row slots. Members are
// (two 256-column groups, token).
#include "gdn_math.h"
__local__ float weights[64];
__local__ int sources[64];
__local__ float64 doubled[32];
void main(tensor residual,tensor routed,tensor shared,tensor place,tensor route,tensor inject,tensor output,
          int width,int streams,int rows,int per_row) {
    const int5 start=get_index_space_offset(),end=start+get_index_space_size();
    const int groups=width/256;
    for(int token=start[1];token<end[1] && token<rows;++token) {
        for(int j=0;j<per_row;++j) {
            int5 rp={token*per_row+j,0,0,0,0};
            weights[j]=s_f32_ld_g((__global__ float*)gen_addr(rp,route));
            sources[j]=j==per_row-1?-2:s_i32_ld_g((__global__ int*)gen_addr(rp,place));
        }
        for(int stream=0;stream<streams;++stream) {
            int5 ic={stream,token,0,0,0};
            float64 scalar=s_f32_ld_g((__global__ float*)gen_addr(ic,inject))*(1.0f/streams);
            doubled[stream]=strata_sigmoid(scalar)*2.0f;
        }
        for(int pair=start[0];pair<end[0];++pair)
        for(int group=pair*2;group<pair*2+2 && group<groups;++group) {
            const int base=group*256;
            // Natural 64-column chunks base + 64c: a 128-lane BF16 load at base + 128h converts to chunks h and 2+h.
            float64 block[4]={0,0,0,0};
            for(int j=0;j<per_row;++j) {
                const int source=sources[j];
                const float weight=weights[j];
                float128 f0,f1;
                if(source==-1) {f0.v1=0;f0.v2=0;f1.v1=0;f1.v2=0;}
                else {
                    int5 a={base,source<0?token:source,0,0,0},b={base+128,source<0?token:source,0,0,0};
                    f0=v_convert_bf16_to_f32_all_b(source<0?v_bf16_ld_tnsr_b(a,shared):v_bf16_ld_tnsr_b(a,routed),0);
                    f1=v_convert_bf16_to_f32_all_b(source<0?v_bf16_ld_tnsr_b(b,shared):v_bf16_ld_tnsr_b(b,routed),0);
                }
                block[0]+=f0.v1*weight;block[1]+=f1.v1*weight;block[2]+=f0.v2*weight;block[3]+=f1.v2*weight;
            }
            for(int stream=0;stream<streams;++stream) {
                const float64 gate=doubled[stream];
                #pragma unroll
                for(int c=0;c<4;++c) {
                    int5 p={base+64*c,stream,token,0,0};
                    v_f32_st_tnsr(p,output,v_f32_ld_tnsr_b(p,residual)+block[c]*gate);
                }
            }
        }
    }
}
