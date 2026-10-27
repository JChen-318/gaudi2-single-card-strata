// STRATA_HPU_QSA_SDPA_FOLD: tpc/qsa_select_expand.c writing the SDPA's mask directly: BF16, each row's mask repeated
// for the `group` query heads folded into the SDPA's rows (row r, head g at r * group + g), instead of an F32 mask that
// a cast and a broadcast then turn into that. Same 0 / -3e38 values (-3e38 rounds to the BF16 -3.0e38 grid point).
//   start {2} F32, kept {keys, rows} F32 -> mask {keys, group, rows} BF16. keys % 128 == 0. Members are rows.
void main(tensor start_cell,tensor kept,tensor mask,int keys,int rows,int window,int width_sel,int block,int heads,int group) {
    const int5 begin=get_index_space_offset(),end=begin+get_index_space_size();
    const int pairs=keys/128;
    const int64 lane=(int64)V_LANE_ID_32;
    const float minus_infinity=-3.0e38f;
    int5 z={0,0,0,0,0};
    const int start=(int)s_f32_ld_g((__global__ float*)gen_addr(z,start_cell));
    z[0]=1;const int first_row=(int)s_f32_ld_g((__global__ float*)gen_addr(z,start_cell));
    const float64 offset=v_convert_i32_to_f32_b(lane&(int64)(block-1),0);
    for(int r=begin[0];r<end[0] && r<rows;++r) {
        const int size=start+first_row+r+1;
        for(int v=0;v<pairs;++v) {
            float128 wide;
            #pragma unroll
            for(int half64=0;half64<2;++half64) {
                const int at=v*128+half64*64;
                int5 p={at,r,0,0,0};const float64 n=v_f32_ld_tnsr_b(p,kept);
                const float64 position=v_convert_i32_to_f32_b(lane+at,0);
                const float64 m=v_f32_sel_less_f32_b(offset,n,v_f32_sel_less_f32_b(position,(float)size,0.0f,minus_infinity),minus_infinity);
                if(half64) wide.v2=m; else wide.v1=m;
            }
            const bfloat128 b=v_convert_f32_to_bf16_all_b(wide,SW_LINEAR);
            for(int g=0;g<group;++g) {int5 o={v*128,g,r,0,0};v_bf16_st_tnsr(o,mask,b);}
        }
    }
}
