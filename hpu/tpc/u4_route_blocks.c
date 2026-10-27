// One vector count per expert and 64-slot block. The following bucket
// kernel reads the counts as scalars and skips all nonmatching blocks.
void main(tensor ids,tensor counts,int experts,int slots) {
    const int5 begin=get_index_space_offset(),end=begin+get_index_space_size();
    const int blocks=(slots+63)/64;
    (void)experts;
    for(int member=begin[0];member<end[0];++member) {
        const int e=member/blocks,block=member-e*blocks;
        int5 ip={block*64,0,0,0,0};
        const float64 values=v_convert_i32_to_f32_b(v_i32_ld_tnsr_b(ip,ids),0);
        const float64 count=v_f32_reduce_add(v_f32_sel_eq_f32_b(values,(float)e,1.0f,0.0f));
        int5 op={block,e,0,0,0};v_f32_st_tnsr_partial(op,counts,count,0,0);
    }
}
