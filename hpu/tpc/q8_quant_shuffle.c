// Same scale and code arithmetic as q8_quant.c. Broadcast a quantized lane by group moves/shuffle
// instead of 32 masked reduction trees. All 256 int8 lanes still contain the same column code.
void main(tensor x,tensor xq,tensor dx,int columns,int rows) {
    const int5 begin=get_index_space_offset(),end=begin+get_index_space_size();
    const int blocks=columns/32;
    const int writes=SW_WR_LOWER_GROUP0|SW_WR_UPPER_GROUP0|SW_WR_LOWER_GROUP1|SW_WR_UPPER_GROUP1|
                     SW_WR_LOWER_GROUP2|SW_WR_UPPER_GROUP2|SW_WR_LOWER_GROUP3|SW_WR_UPPER_GROUP3;
    for(int member=begin[0];member<end[0];++member) {
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
        for(int group=0;group<2;++group) {
            float64 source=0;
            if(group==0) source=v_f32_mov_dual_group_all_b(q,0xffffffff,0,0,0,0,writes,source);
            else source=v_f32_mov_dual_group_all_b(q,0xffffffff,1,1,1,1,writes,source);
            for(int j=0;j<16;++j) {
                const uchar256 index=(unsigned char)(128+(j&7)+((j&8)<<2));
                const float64 lane=v_f32_shuffle_b(source,index,0,0);
                float256 wide;wide.v1=lane;wide.v2=lane;wide.v3=lane;wide.v4=lane;
                int5 xp={0,block*32+group*16+j,row,0,0};v_i8_st_tnsr(xp,xq,v_convert_f32_to_i8_all_b(wide,0));
            }
        }
    }
}
