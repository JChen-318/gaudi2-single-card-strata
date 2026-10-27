// STRATA_HPU_MOE_MME tiles: tile j of the window belongs to the expert e with start[e] <= j < end[e] (end = inclusive
// prefix of the experts' tile counts, start = end - tiles); its rows are that expert's rows (j - start[e]) * rows ..
// + rows, of which tile_rows[j] are real. Tiles past the last expert's end are unused: expert -1, 0 rows.
// end/start {experts} F32, totals {experts} I32 -> tile_expert {tiles} I32, tile_rows {tiles} I32. Members are tiles.
void main(tensor end_,tensor start,tensor totals,tensor tile_expert,tensor tile_rows,int experts,int tiles,int rows) {
    const int5 begin=get_index_space_offset(),end=begin+get_index_space_size();
    (void)tiles;
    for(int j=begin[0];j<end[0];++j) {
        // The first expert whose end exceeds j (binary search over the non-decreasing ends).
        int lo=0,hi=experts;
        while(lo<hi) {
            const int mid=(lo+hi)/2;int5 mp={mid,0,0,0,0};
            if((int)s_f32_ld_g((__global__ float*)gen_addr(mp,end_))>j) hi=mid; else lo=mid+1;
        }
        int expert=-1,count=0;
        if(lo<experts) {
            int5 ep={lo,0,0,0,0};
            const int first=(int)s_f32_ld_g((__global__ float*)gen_addr(ep,start));
            const int total=s_i32_ld_g((__global__ int*)gen_addr(ep,totals));
            const int left=total-(j-first)*rows;
            expert=lo;count=left<rows?left:rows;
        }
        int5 tp={j,0,0,0,0};
        s_i32_st_g((__global__ int*)gen_addr(tp,tile_expert),expert);
        s_i32_st_g((__global__ int*)gen_addr(tp,tile_rows),count);
    }
}
