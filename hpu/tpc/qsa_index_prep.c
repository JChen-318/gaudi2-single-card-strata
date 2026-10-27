// STRATA_HPU_QSA_DEVICE_SELECT: the indexer query rows of a window as QsaBlock::stage_rows prepares them on the host:
// per (row, head) an RMS norm with the index-query gamma (shared, or one row per head) and NeoX RoPE on the first
// `rotary` columns at the row's position. Sums run in F32 (the host sums in double).
//   raw {width, heads, rows} F32, gamma {width * gamma_heads} F32, rope {rotary, rows} (rotary/2 cosines then sines
//   of each row's position) -> out {width, heads, rows} F32. Members are rows.
void main(tensor raw,tensor gamma,tensor rope,tensor out,int width,int heads,int rotary,float epsilon,int rows,int gamma_heads) {
    const int5 begin=get_index_space_offset(),end=begin+get_index_space_size();
    const int pairs=rotary/2;
    for(int row=begin[0];row<end[0] && row<rows;++row) {
        int5 z={0,row,0,0,0},hp={pairs,row,0,0,0};
        const float64 cosine=v_f32_ld_tnsr_b(z,rope),sine=v_f32_ld_tnsr_b(hp,rope);
        for(int head=0;head<heads;++head) {
            const int g0=gamma_heads==1?0:head*width;
            float64 squares=0;
            for(int col=0;col<width;col+=64) {int5 p={col,head,row,0,0};const float64 x=v_f32_ld_tnsr_b(p,raw);squares+=x*x;}
            const float64 inverse=1.0f/v_sqrt_f32(v_f32_reduce_add(squares)/(float)width+epsilon);
            for(int col=64;col<width;col+=64) {
                int5 p={col,head,row,0,0},pg={g0+col,0,0,0,0};
                v_f32_st_tnsr(p,out,(v_f32_ld_tnsr_b(p,raw)*inverse)*v_f32_ld_tnsr_b(pg,gamma));
            }
            int5 a={0,head,row,0,0},b={pairs,head,row,0,0},ga={g0,0,0,0,0},gb={g0+pairs,0,0,0,0};
            const float64 na=(v_f32_ld_tnsr_b(a,raw)*inverse)*v_f32_ld_tnsr_b(ga,gamma);
            const float64 nb=(v_f32_ld_tnsr_b(b,raw)*inverse)*v_f32_ld_tnsr_b(gb,gamma);
            const float64 left=na*cosine-nb*sine,right=na*sine+nb*cosine;
            // Lanes [pairs, 64) of the first vector are columns past the rotary pairs' first halves: the second
            // halves are written by the partial store at `pairs` below, the rest stay normalized.
            v_f32_st_tnsr(a,out,v_f32_sel_less_u32_b(V_LANE_ID_32,(unsigned)pairs,left,na));
            v_f32_st_tnsr_partial(b,out,right,pairs-1,0);
        }
    }
}
