// STRATA_HPU_QSA_DECODE_SELECT, verify windows: tpc/qsa_gather_blocks_fp8.c for each row of a small window starting at
// cell `cell`. Rank r < top is row's r-th ranked complete block; rank `top` is the committed part of the block holding
// `cell` (its cells before `cell`). The window's own rows are attended separately (causally), so row `row` takes
// width_sel - 1 - row - (cell % block) cells from its ranked blocks, in rank, then id order.
//   keys, values {256, heads, cells} FP8; key/value scales {32, cells} F32; top_ids {top, rows} I32;
//   top_scores {top, rows} F32; cell {1} F32 -> key, value {256, n, heads, rows} FP8; key/value scales {n, heads, rows}
//   F32; mask {n, rows} F32 (0 kept, -3e38 not), n = (top + 1) * block. Members are (rank in [0, top], head, row).
void main(tensor keys,tensor values,tensor key_scale,tensor value_scale,tensor top_ids,tensor top_scores,tensor cell_t,
          tensor key,tensor value,tensor key_out_scale,tensor value_out_scale,tensor mask,
          int heads,int cells,int top,int block,int width_sel,int rows) {
    const int5 begin=get_index_space_offset(),end=begin+get_index_space_size();
    (void)heads;(void)rows;
    int5 z={0,0,0,0,0};
    const int cell=(int)s_f32_ld_g((__global__ float*)gen_addr(z,cell_t));
    const int own=cell/block,before=cell-own*block;
    const int64 lane=(int64)V_LANE_ID_32;
    const float minus_infinity=-3.0e38f;
    for(int row=begin[2];row<end[2];++row)
    for(int rank=begin[0];rank<end[0];++rank) {
        int id=own,count=before;
        if(rank<top) {
            int5 tp={rank,row,0,0,0};
            id=s_i32_ld_g((__global__ int*)gen_addr(tp,top_ids));
            const float score=s_f32_ld_g((__global__ float*)gen_addr(tp,top_scores));
            const int left=width_sel-1-row-before-rank*block;
            count=score>-1.0e38f && id>=0 && id<own?(left<block?(left>0?left:0):block):0;
        }
        for(int head=begin[1];head<end[1];++head) {
            // A dropped cell reads past the cache (cells), where a tensor load returns zeros: its codes and scale
            // come out zero with no branch. A scale is lane 0 of the vector read at slot `head` of its cell's row.
            // The common 4-cell block issues its four cells' loads before any store.
#define GATHER_LOAD(o) \
            const int at##o=o<count && id*block+o>=0 && id*block+o<cells?id*block+o:cells; \
            int5 src##o={0,head,at##o,0,0},sp##o={head,at##o,0,0,0}; \
            const uchar256 k##o=v_u8_ld_tnsr_b(src##o,keys),v##o=v_u8_ld_tnsr_b(src##o,values); \
            const float64 ks##o=v_f32_ld_tnsr_b(sp##o,key_scale),vs##o=v_f32_ld_tnsr_b(sp##o,value_scale);
#define GATHER_STORE(o) { \
            int5 dst={0,rank*block+o,head,row,0},so={rank*block+o,head,row,0,0}; \
            v_u8_st_tnsr(dst,key,k##o);v_u8_st_tnsr(dst,value,v##o); \
            v_f32_st_tnsr_partial(so,key_out_scale,ks##o,0,0);v_f32_st_tnsr_partial(so,value_out_scale,vs##o,0,0);}
            if(block==4) {
                GATHER_LOAD(0) GATHER_LOAD(1) GATHER_LOAD(2) GATHER_LOAD(3)
                GATHER_STORE(0) GATHER_STORE(1) GATHER_STORE(2) GATHER_STORE(3)
            } else for(int o=0;o<block;++o) {GATHER_LOAD(o) GATHER_STORE(o)}
#undef GATHER_LOAD
#undef GATHER_STORE
            if(head==0) {
                float64 m=minus_infinity;
                for(int o=0;o<block;++o) m=v_f32_sel_eq_i32_b(lane,(int64)o,o<count?0.0f:minus_infinity,m);
                int5 mp={rank*block,row,0,0,0};v_f32_st_tnsr_partial(mp,mask,m,block-1,0);
            }
        }
    }
}
