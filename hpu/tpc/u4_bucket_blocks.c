// Stable direct buckets: visit matching 64-slot blocks in increasing
// slot order. Every current slot initializes place, including overflow
// and shared slots, so stale entries cannot survive a replay.
void main(tensor ids,tensor counts,tensor bucket,tensor overflow,tensor place,
          int experts,int rows,int slots,int per_row,int top_k) {
    const int5 begin=get_index_space_offset(),end=begin+get_index_space_size();
    const int blocks=(slots+63)/64;
    for(int e=begin[0];e<end[0] && e<experts;++e) {
        int m=0,spill=0;
        for(int block=0;block<blocks;++block) {
            int5 cp={block,e,0,0,0};
            if(s_f32_ld_g((__global__ float*)gen_addr(cp,counts))==0) continue;
            const int endslot=(block+1)*64<slots?(block+1)*64:slots;
            for(int slot=block*64;slot<endslot;++slot) {
                int5 ip={slot,0,0,0,0};
                if(s_i32_ld_g((__global__ int*)gen_addr(ip,ids))!=e) continue;
                int pos=-1;
                if(slot%per_row!=top_k) {
                    if(m<rows) {
                        pos=e*rows+m;int5 bp={pos,0,0,0,0};
                        s_i32_st_g((__global__ int*)gen_addr(bp,bucket),slot);++m;
                    } else ++spill;
                }
                s_i32_st_g((__global__ int*)gen_addr(ip,place),pos);
            }
        }
        for(;m<rows;++m) {int5 bp={e*rows+m,0,0,0,0};s_i32_st_g((__global__ int*)gen_addr(bp,bucket),-1);}
        int5 fp={e,0,0,0,0};s_i32_st_g((__global__ int*)gen_addr(fp,overflow),spill);
    }
}
