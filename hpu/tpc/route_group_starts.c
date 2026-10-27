// STRATA_HPU_MOE_GROUP: where every expert's run begins in the grouped slot order (route_group_order.c).
// Member = expert e in [0, experts]: start[e] is the number of slots whose routed id is below e, so the run of e
// is order[start[e] .. start[e+1]); member `experts` writes `slots` and closes the last run. The shared-expert
// slots carry id 0 (route_topk leaves that lane at zero), so they join expert 0's run - the GEMV picks their
// weights by the slot, not by the id.
// The scan is one 64-slot vector per step (v_f32_sel_less counts the lanes below e); the slots past the last
// whole vector are counted by scalar loads, so no vector load crosses the end of `ids`. Slot numbers stay exact
// in F32 (slots <= 4096).
void main(tensor ids,tensor starts,int experts,int slots) {
    const int5 begin=get_index_space_offset(),end=begin+get_index_space_size();
    const int whole=slots&~63;
    for(int e=begin[0];e<end[0];++e) {
        float64 count=0;int5 p={0,0,0,0,0};
        for(int block=0;block<whole;block+=64) {
            const float64 values=v_convert_i32_to_f32_b(v_i32_ld_tnsr_b(p,ids),0);
            count=count+v_f32_reduce_add(v_f32_sel_less_f32_b(values,(float)e,1.0f,0.0f));
            p[0]+=64;
        }
        for(int slot=whole;slot<slots;++slot) {
            int5 q={slot,0,0,0,0};
            if(s_i32_ld_g((__global__ int*)gen_addr(q,ids))<e) count=count+1.0f;
        }
        int5 o={e,0,0,0,0};
        v_f32_st_tnsr_partial(o,starts,count,0,0);
    }
}
