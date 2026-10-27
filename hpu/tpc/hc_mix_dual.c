// STRATA_HPU_HC_DECODE_TPC: hc_mix.c that also writes its result rounded to BF16 (the cast the next projection's
// input took) and quantized as q8_quant_packed.c would (the resident MoE's activation codes), so neither a cast nor a
// quantization node follows the read. Same arithmetic for the F32 output and the same codes and scales.
//   normalized, gate {width, streams} F32 -> output {width} F32, rounded {width, 1} BF16,
//   codes {256, width/16, 1} U8, scales {64, width/32, 1} F32 (with `codes`; else one-block placeholders, untouched).
//   Members own 64 columns: two 32-column Q8 blocks.
#include "gdn_math.h"
void main(tensor normalized,tensor gate,tensor output,tensor rounded,tensor xq,tensor dx,int width,int streams,int codes) {
    const int5 start=get_index_space_offset(),end=start+get_index_space_size();
    const int writes=SW_WR_LOWER_GROUP0|SW_WR_UPPER_GROUP0|SW_WR_LOWER_GROUP1|SW_WR_UPPER_GROUP1|
                     SW_WR_LOWER_GROUP2|SW_WR_UPPER_GROUP2|SW_WR_LOWER_GROUP3|SW_WR_UPPER_GROUP3;
    for(int tile=start[0];tile<end[0] && tile*64<width;++tile) {
        float64 sum=0;
        for(int row=0;row<streams;++row) {
            int5 p={tile*64,row,0,0,0};
            sum+=v_f32_ld_tnsr_b(p,normalized)*strata_sigmoid(v_f32_ld_tnsr_b(p,gate));
        }
        const float64 mixed=sum*(1.0f/streams);
        int5 p={tile*64,0,0,0,0};v_f32_st_tnsr(p,output,mixed);
        float128 wide;wide.v1=mixed;wide.v2=0;
        v_bf16_st_tnsr_partial(p,rounded,v_convert_f32_to_bf16_all_b(wide,SW_LINEAR),63,0);
        if(!codes) continue;
        // q8_quant_packed.c per 32-column block, its columns in lanes 0..31: the upper block's dual groups 2, 3 are
        // moved down first.
        #pragma unroll
        for(int upper=0;upper<2;++upper) {
            const int block=2*tile+upper;
            float64 value=mixed;
            if(upper) value=v_f32_mov_dual_group_all_b(mixed,0xffffffff,2,3,2,3,writes,value);
            const float64 magnitude=v_f32_sel_less_u32_b(V_LANE_ID_32,32u,v_f32_sel_less_f32_b(value,0.0f,-value,value),0.0f);
            const float64 peak=v_f32_reduce_max(magnitude),d=peak/127.0f;
            const float64 ratio=v_f32_sel_grt_f32_b(peak,0.0f,value/v_f32_sel_grt_f32_b(peak,0.0f,d,1.0f),0.0f);
            const float64 even=v_convert_i32_to_f32_b(v_convert_f32_to_i32_b(ratio,0),0);
            const float64 sign=v_f32_sel_less_f32_b(ratio,0.0f,-1.0f,1.0f);
            float64 q=even+v_f32_sel_eq_f32_b((ratio-even)*sign,0.5f,sign,0.0f);
            q=v_f32_min_b(v_f32_max_b(q,-127.0f),127.0f);
            int5 dp={0,block,0,0,0};v_f32_st_tnsr(dp,dx,d);
            #pragma unroll
            for(int k=0;k<2;++k) {
                float64 source=0;
                if(k==0) source=v_f32_mov_dual_group_all_b(q,0xffffffff,0,0,0,0,writes,source);
                else source=v_f32_mov_dual_group_all_b(q,0xffffffff,1,1,1,1,writes,source);
                float256 four;four.v1=source;four.v2=source;four.v3=source;four.v4=source;
                int5 xp={0,2*block+k,0,0,0};v_i8_st_tnsr(xp,xq,v_convert_f32_to_i8_all_b(four,SW_LINEAR));
            }
        }
    }
}
