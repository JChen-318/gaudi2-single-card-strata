// STRATA_HPU_MOE_MME bucketing, step 1 of 3: how many routed slots of each 64-slot block go to each expert.
// ids {slots} I32 (route_topk; the shared slot of every row, slot % per_row == top_k, holds 0 and is not
// counted) -> counts {experts, blocks} F32, dim 0 the expert so the prefix step reads 64 experts per vector.
// Members are (block, 64 experts): the block's ids are loaded and masked once, then the 64 experts' compares and
// reductions run unrolled (one (expert, block) per member spent nearly all its time on member overhead).
void main(tensor ids,tensor counts,int experts,int slots,int per_row,int top_k,int rows) {
    const int5 begin=get_index_space_offset(),end=begin+get_index_space_size();
    (void)rows;
    const float64 lane=v_convert_i32_to_f32_b((int64)V_LANE_ID_32,0);
    for(int block=begin[0];block<end[0];++block) {
        int5 ip={block*64,0,0,0,0};
        const float64 values=v_convert_i32_to_f32_b(v_i32_ld_tnsr_b(ip,ids),0);
        // Lanes that count: not a shared slot (lanes L0, L0+per_row, ... with (block*64+L0) % per_row == top_k),
        // and inside the slots. A lane that does not count compares against -1, which no expert id equals.
        float64 id=values;
        const int first=(top_k-(block*64)%per_row+per_row)%per_row;
        for(int j=first;j<64;j+=per_row) id=v_f32_sel_eq_f32_b(lane,(float)j,-1.0f,id);
        id=v_f32_sel_less_f32_b(lane,(float)(slots-block*64),id,-1.0f);
        for(int group=begin[1];group<end[1] && group*64<experts;++group)
        for(int e0=group*64;e0<group*64+64;e0+=8) {
            float64 count[8];
            #pragma unroll
            for(int i=0;i<8;++i) count[i]=v_f32_reduce_add(v_f32_sel_eq_f32_b(id,(float)(e0+i),1.0f,0.0f));
            #pragma unroll
            for(int i=0;i<8;++i) {int5 op={e0+i,block,0,0,0};v_f32_st_tnsr_partial(op,counts,count[i],0,0);}
        }
    }
}
