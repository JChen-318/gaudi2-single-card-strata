// STRATA_HPU_MOE_MME (expand variant 4): moe_mme_join.c over BF16 down outputs whose columns follow
// iq4_expand_bf16.c variant 4's row order (see moe_mme_act.c): natural 64-column chunk c of a row is half
// (c%4)%2 of tile c/4, convert vector (c%4)/2, so one BF16 load and convert yields two natural chunks.
// An overflowed slot is zero as in moe_mme_join.c. routed {width, experts*rows} BF16, shared {width, window rows} BF16, place {slots} I32 -> expert_rows
// {width, slots} F32 in natural column order. Members are slots.
void main(tensor routed,tensor shared,tensor place,tensor out,int per_row,int top_k,int bucket_len) {
    const int5 begin=get_index_space_offset(),end=begin+get_index_space_size();
    (void)bucket_len;
    const int width=get_dim_size(out,0);
    for(int slot=begin[0];slot<end[0];++slot) {
        const int is_shared=slot%per_row==top_k;
        int5 ip={slot,0,0,0,0};
        const int source=is_shared?slot/per_row:s_i32_ld_g((__global__ int*)gen_addr(ip,place));
        for(int tile=0;tile<width;tile+=256) for(int h=0;h<2;++h) {
            int5 o0={tile+64*h,slot,0,0,0},o1={tile+128+64*h,slot,0,0,0};
            if(source<0) {v_f32_st_tnsr(o0,out,(float64)0.0f);v_f32_st_tnsr(o1,out,(float64)0.0f);continue;}
            int5 rp={tile+128*h,source,0,0,0};
            const bfloat128 raw=is_shared?v_bf16_ld_tnsr_b(rp,shared):v_bf16_ld_tnsr_b(rp,routed);
            const float128 f=v_convert_bf16_to_f32_all_b(raw,0);
            v_f32_st_tnsr(o0,out,f.v1);v_f32_st_tnsr(o1,out,f.v2);
        }
    }
}
