// STRATA_HPU_RESIDENT_U4: the window's routed slots bucketed by expert, the row index of the grouped MME.
// `starts`/`order` are route_group_starts/order: the run of expert e is order[start[e] .. start[e+1]). Member e
// places the first `rows` routed slots of its run in bucket[e*rows+m], and the rest in overflow[e]: a window that
// ever holds more than `rows` slots of one expert is reported (the host checks it under STRATA_HPU_U4_OVERFLOW)
// instead of being dropped silently. Shared-expert slots (the last slot of every row, id 0 in the routing) are
// skipped here: their weights are the tail expert slab, gathered in row order by the shared chain.
// `place` is the inverse map: place[slot] = e*rows+m for every bucketed slot, so the join can tell a routed slot
// from one an earlier launch routed and this one did not (bucket[place[slot]] != slot).
void main(tensor starts,tensor order,tensor bucket,tensor overflow,tensor place,
          int experts,int rows,int slots,int per_row,int top_k) {
    const int5 begin=get_index_space_offset(),end=begin+get_index_space_size();
    (void)slots;
    for(int e=begin[0];e<end[0]&&e<experts;++e) {
        int5 a={e,0,0,0,0};
        const int from=(int)s_f32_ld_g((__global__ float*)gen_addr(a,starts));
        a[0]=e+1;
        const int to=(int)s_f32_ld_g((__global__ float*)gen_addr(a,starts));
        int m=0,spill=0;
        for(int at=from;at<to;++at) {
            int5 op={at,0,0,0,0};
            const int slot=(int)s_f32_ld_g((__global__ float*)gen_addr(op,order));
            if(slot%per_row==top_k) continue;
            if(m<rows) {
                int5 bp={e*rows+m,0,0,0,0};
                s_i32_st_g((__global__ int*)gen_addr(bp,bucket),slot);
                int5 pp={slot,0,0,0,0};
                s_i32_st_g((__global__ int*)gen_addr(pp,place),e*rows+m);
                ++m;
            } else ++spill;
        }
        for(;m<rows;++m) {
            int5 bp={e*rows+m,0,0,0,0};
            s_i32_st_g((__global__ int*)gen_addr(bp,bucket),-1);
        }
        int5 fp={e,0,0,0,0};
        s_i32_st_g((__global__ int*)gen_addr(fp,overflow),spill);
    }
}
