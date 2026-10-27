// Stable grouped slot order, skipping 64-slot blocks with no matching id.
// starts comes from route_group_starts; counts from u4_route_blocks. No fixed
// bucket, padding or overflow: every input slot is written exactly once.
void main(tensor ids,tensor starts,tensor counts,tensor order,int experts,int slots) {
    const int5 begin=get_index_space_offset(),end=begin+get_index_space_size();
    const int blocks=(slots+63)/64;
    for(int e=begin[0];e<end[0] && e<experts;++e) {
        int5 sp={e,0,0,0,0};
        int at=(int)s_f32_ld_g((__global__ float*)gen_addr(sp,starts));
        for(int block=0;block<blocks;++block) {
            int5 cp={block,e,0,0,0};
            if(s_f32_ld_g((__global__ float*)gen_addr(cp,counts))==0) continue;
            const int limit=(block+1)*64<slots?(block+1)*64:slots;
            for(int slot=block*64;slot<limit;++slot) {
                int5 p={slot,0,0,0,0};
                if(s_i32_ld_g((__global__ int*)gen_addr(p,ids))==e) {
                    int5 o={at,0,0,0,0};s_f32_st_g((__global__ float*)gen_addr(o,order),(float)slot);++at;
                }
            }
        }
    }
}
