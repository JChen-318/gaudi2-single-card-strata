// STRATA_HPU_QSA_DEVICE_SELECT: the indexer key of every block of cells, one row per block, for a window of `rows`
// cells starting at the runtime cell `start` (start {1} F32). Attention key positions are the cells themselves (the
// window's rows are written to the cache before attention), so block j covers key positions [j * block, (j + 1) *
// block). Blocks wholly before `start` are the host's pooled rows (QsaState::pooled_rows); later blocks are pooled here
// as QsaState::append does: the mean of the block's raw index keys (the window's rows, or the up to block-1 raw keys
// before `start` in `tail`), an RMS norm with the index-key gamma, and NeoX RoPE at the block's first cell. A block
// the window does not complete is only ever a row's forced last block, so its value is not read for ranking (the
// mean of what exists). Blocks past the window are not written (every row masks them).
//   start {1} F32, pooled {width, >= blocks} F32, tail {width, block} F32, window {width, rows} F32, gamma {width},
//   rope {rotary, >= cells} (per cell) -> out {width, keys / block} F32. Members are blocks.
void main(tensor start_cell,tensor pooled,tensor tail,tensor window,tensor gamma,tensor rope,tensor out,
          int width,int rotary,float epsilon,int keys,int rows,int block) {
    const int5 begin=get_index_space_offset(),end=begin+get_index_space_size();
    const int pairs=rotary/2;
    int5 z={0,0,0,0,0};
    const int start=(int)s_f32_ld_g((__global__ float*)gen_addr(z,start_cell));
    const int cells=start+rows,host_blocks=start/block;
    for(int j=begin[0];j<end[0] && j*block<cells;++j) {
        const int first=j*block,last=first+block<cells?first+block:cells;
        if(j<host_blocks) {
            for(int col=0;col<width;col+=64) {
                int5 s={col,j,0,0,0},o={col,j,0,0,0};v_f32_st_tnsr(o,out,v_f32_ld_tnsr_b(s,pooled));
            }
            continue;
        }
        // The mean of the block's raw keys at column offset AT (64 lanes).
        #define MEAN(OUT,AT) { \
            float64 sum_=0; \
            for(int c_=first;c_<last;++c_) { \
                int5 q_={(AT),c_<start?c_-host_blocks*block:c_-start,0,0,0}; \
                sum_+=c_<start?v_f32_ld_tnsr_b(q_,tail):v_f32_ld_tnsr_b(q_,window); \
            } \
            OUT=sum_*(1.0f/(float)block); }
        float64 squares=0;
        for(int col=0;col<width;col+=64) {float64 m;MEAN(m,col) squares+=m*m;}
        const float64 inverse=1.0f/v_sqrt_f32(v_f32_reduce_add(squares)/(float)width+epsilon);
        int5 rp={0,first,0,0,0},rq={pairs,first,0,0,0};
        const float64 cosine=v_f32_ld_tnsr_b(rp,rope),sine=v_f32_ld_tnsr_b(rq,rope);
        for(int col=64;col<width;col+=64) {
            float64 m;MEAN(m,col)
            int5 pg={col,0,0,0,0};int5 o={col,j,0,0,0};v_f32_st_tnsr(o,out,(m*inverse)*v_f32_ld_tnsr_b(pg,gamma));
        }
        float64 ma,mb;MEAN(ma,0) MEAN(mb,pairs)
        int5 ga={0,0,0,0,0},gb={pairs,0,0,0,0};
        const float64 na=(ma*inverse)*v_f32_ld_tnsr_b(ga,gamma),nb=(mb*inverse)*v_f32_ld_tnsr_b(gb,gamma);
        const float64 left=na*cosine-nb*sine,right=na*sine+nb*cosine;
        const float64 head=v_f32_sel_less_u32_b(V_LANE_ID_32,(unsigned)pairs,left,na);
        int5 a={0,j,0,0,0},b={pairs,j,0,0,0};
        v_f32_st_tnsr(a,out,head);v_f32_st_tnsr_partial(b,out,right,pairs-1,0);
        #undef MEAN
    }
}
