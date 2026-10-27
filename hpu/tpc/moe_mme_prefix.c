// STRATA_HPU_MOE_MME bucketing, step 2: per expert the exclusive running count over blocks (where the block's
// slots start in the expert's rows), the expert's total and the number of `rows`-row tiles it needs.
// counts {experts, blocks} F32 -> offsets {experts, blocks} F32, totals {experts} I32, tiles {experts} F32
// (ceil(total / rows)). Members are groups of 64 experts.
void main(tensor counts,tensor offsets,tensor totals,tensor tiles,int experts,int slots,int per_row,int top_k,int rows) {
    const int5 begin=get_index_space_offset(),end=begin+get_index_space_size();
    const int blocks=(slots+63)/64;
    (void)experts;(void)per_row;(void)top_k;
    for(int group=begin[0];group<end[0];++group) {
        float64 running=0;
        for(int block=0;block<blocks;++block) {
            int5 p={group*64,block,0,0,0};
            const float64 count=v_f32_ld_tnsr_b(p,counts);
            v_f32_st_tnsr(p,offsets,running);
            running+=count;
        }
        int5 q={group*64,0,0,0,0};
        v_i32_st_tnsr(q,totals,v_convert_f32_to_i32_b(running,0));
        // ceil(total / rows) of an exact integer in F32: floor((total + rows - 1/2) / rows).
        const float64 quotient=(running+((float)rows-0.5f))*(1.0f/(float)rows);
        const float64 tile_count=v_convert_i32_to_f32_b(v_convert_f32_to_i32_b(quotient,SW_RD),0);
        v_f32_st_tnsr(q,tiles,tile_count);
    }
}
