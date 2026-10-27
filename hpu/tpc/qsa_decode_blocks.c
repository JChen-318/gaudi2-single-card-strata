// STRATA_HPU_QSA_DECODE_SELECT: one decoded token's indexer block scores on the device. scores {blocks, heads} F32
// (the indexer query's heads against every pooled block key) -> sum over heads of max(score, 0) per block, the
// blocks the token cannot rank (its own unfinished block and every later one: id >= cell / block) -3e38, so a
// top-k over the result ranks exactly the complete blocks. `cell` {1} F32 is the token's position (runtime).
//   scores {blocks, heads} F32, cell {1} F32 -> out {blocks} F32. blocks % 64 == 0. Members are 64-block vectors.
void main(tensor scores,tensor cell_t,tensor out,int blocks,int heads,int block) {
    const int5 begin=get_index_space_offset(),end=begin+get_index_space_size();
    (void)blocks;
    int5 z={0,0,0,0,0};
    const int cell=(int)s_f32_ld_g((__global__ float*)gen_addr(z,cell_t));
    const float complete=(float)(cell/block);
    const int64 lane=(int64)V_LANE_ID_32;
    for(int v=begin[0];v<end[0];++v) {
        float64 s=0;
        for(int h=0;h<heads;++h) {int5 p={v*64,h,0,0,0};s+=v_f32_max_b(v_f32_ld_tnsr_b(p,scores),0.0f);}
        const float64 id=v_convert_i32_to_f32_b(lane+v*64,0);
        s=v_f32_sel_less_f32_b(id,complete,s,-3.0e38f);
        int5 o={v*64,0,0,0,0};v_f32_st_tnsr(o,out,s);
    }
}
