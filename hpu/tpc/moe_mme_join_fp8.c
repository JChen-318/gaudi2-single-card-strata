// STRATA_HPU_MOE_FP8: moe_mme_join_perm.c with routed rows from the FP8 down GEMM (BF16 in the resident row order,
// moe_mme_fp8.h; the shared rows keep variant-4 order); the true routed row is routed[n] * act_scale[row] * s_down[tile][n] (tile = row / rows, s_down in
// natural order). routed {width, tiles*rows} BF16, shared {width, window rows} BF16,
// place {slots} I32, s_down {width, tiles} F32, act_scale {64, tiles*rows} F32 (lane-replicated) -> expert_rows
// {width, slots} F32 in natural column order. Members are slots.
#include "moe_mme_fp8.h"
void main(tensor routed,tensor shared,tensor place,tensor s_down,tensor act_scale,tensor out,int per_row,int top_k,int bucket_len,int rows) {
    const int5 begin=get_index_space_offset(),end=begin+get_index_space_size();
    (void)bucket_len;
    const int width=get_dim_size(out,0);
    for(int slot=begin[0];slot<end[0];++slot) {
        const int is_shared=slot%per_row==top_k;
        int5 ip={slot,0,0,0,0};
        const int source=is_shared?slot/per_row:s_i32_ld_g((__global__ int*)gen_addr(ip,place));
        if(is_shared) {
            for(int tile=0;tile<width;tile+=256) for(int h=0;h<2;++h) {
                int5 o0={tile+64*h,slot,0,0,0},o1={tile+128+64*h,slot,0,0,0};
                int5 rp={tile+128*h,source,0,0,0};
                const float128 f=v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(rp,shared),0);
                v_f32_st_tnsr(o0,out,f.v1);v_f32_st_tnsr(o1,out,f.v2);
            }
            continue;
        }
        if(source<0) {
            for(int n=0;n<width;n+=64) {int5 o={n,slot,0,0,0};v_f32_st_tnsr(o,out,(float64)0.0f);}
            continue;
        }
        const int tile=source/rows;
        int5 ap={0,source,0,0,0};
        const float64 a=v_f32_ld_tnsr_b(ap,act_scale);
        for(int base=0;base<width;base+=256) {
            int5 rp={base,source,0,0,0};
            float64 c[4];
            TILE4(routed,rp,c[0],c[1],c[2],c[3])
            #pragma unroll
            for(int k=0;k<4;++k) {
                int5 sp={base+64*k,tile,0,0,0},o={base+64*k,slot,0,0,0};
                v_f32_st_tnsr(o,out,c[k]*(v_f32_ld_tnsr_b(sp,s_down)*a));
            }
        }
    }
}
