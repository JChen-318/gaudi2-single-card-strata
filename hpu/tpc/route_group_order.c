// STRATA_HPU_MOE_GROUP: the grouped slot order the resident MoE GEMV reads (iq4q8_gemv_group.c). Member = expert
// e: it writes every slot whose routed id is e at start[e] + k, in slot order, so `order` is a permutation of
// 0..slots-1 whose runs are one expert each (the shared slots, id 0, form one long run). The GEMV's consecutive
// members then read the same expert's weight rows instead of one copy per slot.
// The ids scan is uniform scalar work - the whole array is 11 KB and cached after the first member - and only the
// matching slots are stored, one scalar each, so `order` carries slot numbers as F32 (exact below 2^24).
// Params are abi::RouteGroup{experts, slots}: `experts` bounds the member loop, `slots` the scan.
void main(tensor ids,tensor starts,tensor order,int experts,int slots) {
    const int5 begin=get_index_space_offset(),end=begin+get_index_space_size();
    for(int e=begin[0];e<end[0]&&e<experts;++e) {
        int5 sp={e,0,0,0,0};
        int at=(int)s_f32_ld_g((__global__ float*)gen_addr(sp,starts));
        for(int slot=0;slot<slots;++slot) {
            int5 p={slot,0,0,0,0};
            const int id=s_i32_ld_g((__global__ int*)gen_addr(p,ids));
            if(id==e) {
                int5 o={at,0,0,0,0};
                s_f32_st_g((__global__ float*)gen_addr(o,order),(float)slot);
                ++at;
            }
        }
    }
}
