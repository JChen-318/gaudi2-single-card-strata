// STRATA_HPU_MOE_MME: the grouped MME's outputs back in slot order. routed is the down GEMM {width, experts*rows}
// (bucket row order), shared the shared expert's {width, window rows}. A shared slot takes its window row, a routed
// slot the bucket row moe_mme_scatter.c placed it in (place[slot]), and a slot that overflowed its expert's bucket
// (place -1) is zero; the bucketing's overflow counts let the host replay such a window. No route weight: the
// consumer applies it. Members are slots.
void main(tensor routed,tensor shared,tensor place,tensor out,int per_row,int top_k,int bucket_len) {
    const int5 begin=get_index_space_offset(),end=begin+get_index_space_size();
    (void)bucket_len;
    const int width=get_dim_size(out,0);
    for(int slot=begin[0];slot<end[0];++slot) {
        const int is_shared=slot%per_row==top_k;
        int5 ip={slot,0,0,0,0};
        const int source=is_shared?slot/per_row:s_i32_ld_g((__global__ int*)gen_addr(ip,place));
        for(int c=0;c<width;c+=64) {
            int5 op={c,slot,0,0,0};
            if(source<0) {v_f32_st_tnsr(op,out,(float64)0.0f);continue;}
            int5 rp={c,source,0,0,0};
            v_f32_st_tnsr(op,out,is_shared?v_f32_ld_tnsr_b(rp,shared):v_f32_ld_tnsr_b(rp,routed));
        }
    }
}
