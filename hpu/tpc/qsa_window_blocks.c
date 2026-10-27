// STRATA_HPU_QSA_DECODE_SELECT, verify windows: tpc/qsa_decode_blocks.c for each row of a small window. scores
// {blocks, heads, rows} F32 -> per row the sum over heads of max(score, 0) per block, the blocks that are not complete
// committed blocks (id >= cell / block; `cell` {1} F32 is the window's first cell) -3e38.
//   scores {blocks, heads, rows} F32, cell {1} F32 -> out {blocks, rows} F32. blocks % 64 == 0.
//   Members are (64-block vector, row).
void main(tensor scores,tensor cell_t,tensor out,int blocks,int heads,int block,int rows) {
    const int5 begin=get_index_space_offset(),end=begin+get_index_space_size();
    (void)blocks;(void)rows;
    int5 z={0,0,0,0,0};
    const int cell=(int)s_f32_ld_g((__global__ float*)gen_addr(z,cell_t));
    const float complete=(float)(cell/block);
    const int64 lane=(int64)V_LANE_ID_32;
    for(int r=begin[1];r<end[1];++r)
    for(int v=begin[0];v<end[0];++v) {
        float64 s=0;
        for(int h=0;h<heads;++h) {int5 p={v*64,h,r,0,0};s+=v_f32_max_b(v_f32_ld_tnsr_b(p,scores),0.0f);}
        const float64 id=v_convert_i32_to_f32_b(lane+v*64,0);
        s=v_f32_sel_less_f32_b(id,complete,s,-3.0e38f);
        int5 o={v*64,r,0,0,0};v_f32_st_tnsr(o,out,s);
    }
}
