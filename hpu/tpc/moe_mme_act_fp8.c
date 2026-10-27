// STRATA_HPU_MOE_FP8: moe_mme_act.c for the FP8 gate/up product: BF16 in the resident row order (moe_mme_fp8.h),
// the true product being gu[r][n] * a_scale[r] * s_gu[n] (s_gu in natural order). SiLU(gate) * up of row r is then
// encoded for the down GEMM as fp8(act / scale_r), scale_r = max|act| / 240.
// gu {2*ff, rows, batch} BF16, s_gu {2*ff, batch} F32, a_scale {64, rows, batch} F32 (lane-replicated), fill {tiles} I32
// -> act {ff, rows, batch} F8 (natural order) and act_scale {64, rows, batch} F32 (lane-replicated). Rows from
// fill[first+b] on are padding and are skipped. Members are (8 rows, batch). 2*ff % 256 == 0, ff <= 640.
#include "gdn_math.h"
#include "moe_mme_fp8.h"
#define ACT_CHUNKS 10
__local__ float64 natural[2*ACT_CHUNKS];
void main(tensor gu,tensor s_gu,tensor a_scale,tensor fill,tensor act,tensor act_scale,int ff,int use_totals,int first) {
    const int5 begin=get_index_space_offset(),end=begin+get_index_space_size();
    const int chunks=ff/64,rows=get_dim_size(gu,1);
    for(int b=begin[1];b<end[1];++b) {
        int limit=rows;
        if(use_totals) {
            int5 tp={first+b,0,0,0,0};
            const int total=s_i32_ld_g((__global__ int*)gen_addr(tp,fill));
            limit=total<limit?total:limit;
        }
        for(int group=begin[0];group<end[0];++group) {
            const int stop=8*group+8<limit?8*group+8:limit;
            for(int row=8*group;row<stop;++row) {
                int5 ap={0,row,b,0,0};
                const float64 a=v_f32_ld_tnsr_b(ap,a_scale);
                // Natural chunks 0..chunks-1 are the gate, chunks..2*chunks-1 the up projection.
                #pragma unroll
                for(int t=0;t<ACT_CHUNKS/2;++t) {
                    if(t>=chunks/2) break;
                    int5 tp={256*t,row,b,0,0};
                    float64 c0,c1,c2,c3;
                    TILE4(gu,tp,c0,c1,c2,c3)
                    natural[4*t]=c0;natural[4*t+1]=c1;natural[4*t+2]=c2;natural[4*t+3]=c3;
                }
                float64 v[ACT_CHUNKS];float64 peak=0;
                #pragma unroll
                for(int c=0;c<ACT_CHUNKS;++c) {
                    if(c>=chunks) break;
                    int5 sg={c*64,b,0,0,0},su={ff+c*64,b,0,0,0};
                    const float64 g=natural[c]*(v_f32_ld_tnsr_b(sg,s_gu)*a),u=natural[chunks+c]*(v_f32_ld_tnsr_b(su,s_gu)*a);
                    v[c]=g*strata_sigmoid(g)*u;
                    peak=v_f32_max_b(peak,v_f32_sel_less_f32_b(v[c],0.0f,-v[c],v[c]));
                }
                const float64 scale=v_f32_reduce_max(peak)*(1.0f/240.0f)+1e-30f,inverse=1.0f/scale;
                int5 so={0,row,b,0,0};v_f32_st_tnsr(so,act_scale,scale);
                #pragma unroll
                for(int c=0;c<ACT_CHUNKS;c+=4) {
                    if(c>=chunks) break;
                    float256 x;
                    x.v1=v[c]*inverse;x.v2=v[c+1]*inverse;
                    x.v3=c+2<chunks?v[c+2]*inverse:0;x.v4=c+3<chunks?v[c+3]*inverse:0;
                    int5 o={c*64,row,b,0,0};
                    const minifloat256 codes=v_convert_f32_to_f8_all_b(x,SW_LINEAR);
                    if(c+4<=chunks) v_f8_st_tnsr(o,act,codes); else v_f8_st_tnsr_partial(o,act,codes,(chunks-c)*64-1,0);
                }
            }
        }
    }
}
