// STRATA_HPU_RESIDENT_IQ4 activation quantizer, one member per (row, 32-column block):
//   d = max|x| / 127 (F32), q = round-half-away-from-zero(x / d), 0 when the block is all zero -- the code law of
//   the CUDA Q8_1 quantizer; its scale stays F32 instead of being rounded to F16.
//   xq: U8 {256, columns, rows}, column c's int8 code in every lane (the GEMV's broadcast operand);
//   dx: F32 {64, columns/32, rows}, the block scale in every lane.
void main(tensor x,tensor xq,tensor dx,int columns,int rows) {
    const int5 begin=get_index_space_offset(),end=begin+get_index_space_size();
    const int blocks=columns/32;
    for(int member=begin[0];member<end[0];++member) {
        const int row=member/blocks,block=member-row*blocks;
        int5 p={block*32,row,0,0,0};
        const float64 value=v_f32_ld_tnsr_b(p,x);
        const float64 magnitude=v_f32_sel_less_u32_b(V_LANE_ID_32,32u,v_f32_sel_less_f32_b(value,0.0f,-value,value),0.0f);
        const float64 peak=v_f32_reduce_max(magnitude);
        const float64 d=peak/127.0f;
        const float64 ratio=v_f32_sel_grt_f32_b(peak,0.0f,value/v_f32_sel_grt_f32_b(peak,0.0f,d,1.0f),0.0f);
        // F32 -> I32 rounds half to even; move exact halves away from zero.
        const float64 even=v_convert_i32_to_f32_b(v_convert_f32_to_i32_b(ratio,0),0);
        const float64 sign=v_f32_sel_less_f32_b(ratio,0.0f,-1.0f,1.0f);
        float64 q=even+v_f32_sel_eq_f32_b((ratio-even)*sign,0.5f,sign,0.0f);
        q=v_f32_min_b(v_f32_max_b(q,-127.0f),127.0f);
        int5 dp={0,block,row,0,0};
        v_f32_st_tnsr(dp,dx,d);
        for(int i=0;i<32;++i) {
            const float64 lane=v_f32_reduce_max(v_f32_sel_eq_u32_b(V_LANE_ID_32,(unsigned)i,q,-1000.0f));
            float256 wide;wide.v1=lane;wide.v2=lane;wide.v3=lane;wide.v4=lane;
            int5 xp={0,block*32+i,row,0,0};
            v_i8_st_tnsr(xp,xq,v_convert_f32_to_i8_all_b(wide,0));
        }
    }
}
