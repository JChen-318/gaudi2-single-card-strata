// STRATA_HPU_MOE_MME (expand variant 4): SiLU(gate) * up of the gate/up GEMM's BF16 output, whose columns follow
// iq4_expand_bf16.c variant 4's row order. Within a 256-row tile position 128*h + 2*i + p holds row 64*h + 128*p + i,
// so the BF16 -> F32 convert of the 128 positions of half h returns rows 64*h + i in v1 and 128 + 64*h + i in v2:
// natural 64-row chunks. Chunk c (rows 64c..64c+63) is in tile c/4, half (c%4)%2, convert vector (c%4)/2.
// gu {2*ff, rows, batch} BF16 + totals {experts} I32 -> act {ff, rows, batch} BF16 in natural column order. With
// use_totals, rows from totals[first+b] on are bucket padding and are skipped (their down GEMM rows are never
// read). Members are (8 rows, batch).
#include "gdn_math.h"
// The F32 chunk C of row `row`, batch member b (a macro: tensors cannot be passed to functions).
#define CHUNK(OUT,C) { \
    const int k_=(C)%4; \
    int5 p_={((C)/4)*256+128*(k_%2),row,b,0,0}; \
    const float128 f_=v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(p_,gu),0); \
    OUT=k_/2?f_.v2:f_.v1; }
void main(tensor gu,tensor totals,tensor act,int ff,int use_totals,int first) {
    const int5 begin=get_index_space_offset(),end=begin+get_index_space_size();
    const int chunks=ff/64,rows=get_dim_size(gu,1);
    for(int b=begin[1];b<end[1];++b) {
        int limit=rows;
        if(use_totals) {
            int5 tp={first+b,0,0,0,0};
            const int total=s_i32_ld_g((__global__ int*)gen_addr(tp,totals));
            limit=total<limit?total:limit;
        }
        for(int group=begin[0];group<end[0];++group) {
            const int stop=8*group+8<limit?8*group+8:limit;
            for(int row=8*group;row<stop;++row) {
                for(int c=0;c<chunks;c+=2) {
                    float64 g0,u0,g1,u1;
                    CHUNK(g0,c) CHUNK(u0,chunks+c) CHUNK(g1,c+1) CHUNK(u1,chunks+c+1)
                    float128 v;v.v1=g0*strata_sigmoid(g0)*u0;v.v2=g1*strata_sigmoid(g1)*u1;
                    int5 o={c*64,row,b,0,0};
                    v_bf16_st_tnsr(o,act,v_convert_f32_to_bf16_all_b(v,SW_LINEAR));
                }
            }
        }
    }
}
