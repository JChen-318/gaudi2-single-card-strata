// STRATA_HPU_MOE_MME bucketing, step 3: every routed slot's tile row. Slot s of block b routed to expert e is row
// p = offsets[e][b] + (earlier slots of block b routed to e) of its expert, so it lands in tile start[e] + p / rows at
// row p % rows: each expert's rows fill its tiles in slot order, and no slot is ever dropped (an expert has
// ceil(total / rows) tiles). ids {slots} I32 + offsets {experts, blocks} F32 + start {experts} F32 -> bucket
// {tiles*rows} I32 (only real rows are written: readers stop at tile_rows) and place {slots} I32 (tile*rows + row, or
// -1 for a row's shared slot; every slot is written). Members are 64-slot blocks.
// The scalar global loads are the cost (each waits a memory round trip), so a block runs in batches of eight slots
// whose loads are issued together; the earlier-slot count is a small hash table of the block's experts (slots in
// order), and a power-of-two `rows` divides by shifts. A full block's places go out as one vector.
__local__ int placed[64];
__local__ int keys[64],counts[64];
void main(tensor ids,tensor offsets,tensor start,tensor bucket,tensor place,int experts,int slots,int per_row,int top_k,int rows) {
    const int5 begin=get_index_space_offset(),end=begin+get_index_space_size();
    (void)experts;
    int shift=0;while((1<<shift)<rows) ++shift;
    const int pow2=(1<<shift)==rows;
    for(int block=begin[0];block<end[0];++block) {
        const int first=block*64,last=first+64<slots?first+64:slots;
        for(int i=0;i<64;++i) keys[i]=-1;
        for(int b0=first;b0<last;b0+=8) {
            int id[8];float base[8],tile0[8];
            #pragma unroll
            for(int i=0;i<8;++i) {
                const int slot=b0+i<last?b0+i:last-1;
                int5 ip={slot,0,0,0,0};
                id[i]=slot%per_row==top_k || b0+i>=last?-1:s_i32_ld_g((__global__ int*)gen_addr(ip,ids));
            }
            #pragma unroll
            for(int i=0;i<8;++i) {
                const int e=id[i]<0?0:id[i];
                int5 op={e,block,0,0,0},sp={e,0,0,0,0};
                base[i]=s_f32_ld_g((__global__ float*)gen_addr(op,offsets));
                tile0[i]=s_f32_ld_g((__global__ float*)gen_addr(sp,start));
            }
            for(int i=0;i<8 && b0+i<last;++i) {
                const int slot=b0+i;
                int at=-1;
                if(id[i]>=0) {
                    // Open addressing over 64 entries: at most 64 distinct experts in a block.
                    int h=id[i]&63;
                    while(keys[h]!=id[i] && keys[h]>=0) h=(h+1)&63;
                    if(keys[h]<0) {keys[h]=id[i];counts[h]=0;}
                    const int earlier=counts[h]++;
                    const int position=(int)base[i]+earlier;
                    const int tile=(int)tile0[i]+(pow2?position>>shift:position/rows);
                    at=tile*rows+(pow2?position&(rows-1):position%rows);
                    int5 bp={at,0,0,0,0};
                    s_i32_st_g((__global__ int*)gen_addr(bp,bucket),slot);
                }
                if(last-first<64) {int5 ip={slot,0,0,0,0};s_i32_st_g((__global__ int*)gen_addr(ip,place),at);}
                else placed[slot-first]=at;
            }
        }
        if(last-first==64) {
            const int64 lane=(int64)V_LANE_ID_32;int64 row=-1;
            for(int j=0;j<64;++j) row=v_i32_sel_eq_i32_b(lane,j,placed[j],row);
            int5 pp={first,0,0,0,0};v_i32_st_tnsr(pp,place,row);
        }
    }
}
