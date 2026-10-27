// STRATA_HPU_RESIDENT_U4: the two MME chains write different slots of one routed-rows tensor, so this join
// produces the dense per-slot rows the route weighting reads: a shared-expert slot takes the shared chain's row,
// a routed slot takes the grouped chain's row when this launch bucketed it, and every other slot is zero (the
// window overflowed the bucket for that slot). place/bucket are u4_moe_bucket.c's inverse map and buckets.
// Each member copies one 64-lane column block of one slot.
void main(tensor routed,tensor shared,tensor bucket,tensor place,tensor out,int per_row,int top_k,int bucket_len) {
    const int5 begin=get_index_space_offset(),end=begin+get_index_space_size();
    for(int slot=begin[1];slot<end[1];++slot) {
        const int is_shared=slot%per_row==top_k;
        int source=-1;
        if(is_shared) source=slot;
        else {
            int5 ip={slot,0,0,0,0};
            const int at=s_i32_ld_g((__global__ int*)gen_addr(ip,place));
            if(at>=0 && at<bucket_len) {
                int5 bp={at,0,0,0,0};
                if(s_i32_ld_g((__global__ int*)gen_addr(bp,bucket))==slot) source=slot;
            }
        }
        for(int block=begin[0];block<end[0];++block) {
            int5 op={block*64,slot,0,0,0};
            if(source<0) {v_f32_st_tnsr(op,out,(float64)0.0f);continue;}
            int5 rp={block*64,source,0,0,0};
            if(is_shared) v_f32_st_tnsr(op,out,v_f32_ld_tnsr_b(rp,shared));
            else v_f32_st_tnsr(op,out,v_f32_ld_tnsr_b(rp,routed));
        }
    }
}
