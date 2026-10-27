// STRATA_HPU_Q8_PACKED: q8_quant_shuffle.c's codes and scales, stored packed: per block of 32 columns two 256-byte
// vectors, vector k holding the codes of columns 16k..16k+15 in lanes 0..15 and repeated in every 16 lanes, so the
// GEMV broadcasts one column with one in-group shuffle (iq4q8_gemv_packed.c) instead of loading a broadcast vector
// per column. Same scale and code arithmetic, 16 times fewer code bytes written.
void main(tensor x,tensor xq,tensor dx,int columns,int rows) {
    const int5 begin=get_index_space_offset(),end=begin+get_index_space_size();
    const int blocks=columns/32;
    const int writes=SW_WR_LOWER_GROUP0|SW_WR_UPPER_GROUP0|SW_WR_LOWER_GROUP1|SW_WR_UPPER_GROUP1|
                     SW_WR_LOWER_GROUP2|SW_WR_UPPER_GROUP2|SW_WR_LOWER_GROUP3|SW_WR_UPPER_GROUP3;
    // Four members side by side (the last repeated where fewer are left, rewriting the same values): each member is
    // one dependent chain through a reduction, so interleaving them hides that latency.
    for(int first=begin[0];first<end[0];first+=4) {
        #pragma unroll
        for(int i=0;i<4;++i) {
        const int member=first+i<end[0]?first+i:end[0]-1;
        const int row=member/blocks,block=member-row*blocks;
        int5 p={block*32,row,0,0,0};
        const float64 value=v_f32_ld_tnsr_b(p,x);
        const float64 magnitude=v_f32_sel_less_u32_b(V_LANE_ID_32,32u,v_f32_sel_less_f32_b(value,0.0f,-value,value),0.0f);
        const float64 peak=v_f32_reduce_max(magnitude),d=peak/127.0f;
        const float64 ratio=v_f32_sel_grt_f32_b(peak,0.0f,value/v_f32_sel_grt_f32_b(peak,0.0f,d,1.0f),0.0f);
        const float64 even=v_convert_i32_to_f32_b(v_convert_f32_to_i32_b(ratio,0),0);
        const float64 sign=v_f32_sel_less_f32_b(ratio,0.0f,-1.0f,1.0f);
        float64 q=even+v_f32_sel_eq_f32_b((ratio-even)*sign,0.5f,sign,0.0f);
        q=v_f32_min_b(v_f32_max_b(q,-127.0f),127.0f);
        int5 dp={0,block,row,0,0};v_f32_st_tnsr(dp,dx,d);
        #pragma unroll
        for(int k=0;k<2;++k) {
            // Lanes 16*k..16*k+15 (one dual group of F32 lanes) into all four dual groups, then the four
            // vectors converted in linear order: byte lane L holds code 16*k + L%16.
            float64 source=0;
            if(k==0) source=v_f32_mov_dual_group_all_b(q,0xffffffff,0,0,0,0,writes,source);
            else source=v_f32_mov_dual_group_all_b(q,0xffffffff,1,1,1,1,writes,source);
            float256 wide;wide.v1=source;wide.v2=source;wide.v3=source;wide.v4=source;
            int5 xp={0,2*block+k,row,0,0};v_i8_st_tnsr(xp,xq,v_convert_f32_to_i8_all_b(wide,SW_LINEAR));
        }
        }
    }
}
