// STRATA_HPU_QSA_DEVICE_SELECT: the attention mask from tpc/qsa_select_mask.c's per-block counts. `kept` holds, at
// every key position (cell), how many cells of its block the row keeps from the block's first (the block counts
// broadcast to their positions by the graph); position p is selected when p % block < kept and p < size (the row's
// cell + 1; the forced block's later cells are the row's future). `start` {2} F32: the window's first cell and this
// chunk's first window row.
//   start {2} F32, kept {keys, rows} F32 -> mask {keys, 1, rows} F32 (0 selected, -3e38 not: finite, so the softmax's
//   max subtraction stays defined). keys % 64 == 0, block a power of two <= 64. Members are rows.
void main(tensor start_cell,tensor kept,tensor mask,int keys,int rows,int window,int width_sel,int block,int heads) {
    const int5 begin=get_index_space_offset(),end=begin+get_index_space_size();
    const int vectors=keys/64;
    const int64 lane=(int64)V_LANE_ID_32;
    const float minus_infinity=-3.0e38f;
    int5 z={0,0,0,0,0};
    const int start=(int)s_f32_ld_g((__global__ float*)gen_addr(z,start_cell));
    z[0]=1;const int first_row=(int)s_f32_ld_g((__global__ float*)gen_addr(z,start_cell));
    // A position's offset in its block depends only on the lane (64 is a multiple of block).
    const float64 offset=v_convert_i32_to_f32_b(lane&(int64)(block-1),0);
    for(int r=begin[0];r<end[0] && r<rows;++r) {
        const int size=start+first_row+r+1;
        for(int v=0;v<vectors;++v) {
            int5 p={v*64,r,0,0,0};const float64 n=v_f32_ld_tnsr_b(p,kept);
            const float64 position=v_convert_i32_to_f32_b(lane+v*64,0);
            const float64 m=v_f32_sel_less_f32_b(offset,n,v_f32_sel_less_f32_b(position,(float)size,0.0f,minus_infinity),minus_infinity);
            int5 o={v*64,0,r,0,0};v_f32_st_tnsr(o,mask,m);
        }
    }
}
