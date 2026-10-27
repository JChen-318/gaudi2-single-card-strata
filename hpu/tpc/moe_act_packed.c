// STRATA_HPU_MOE_ACT_FUSED (with Q8_PACKED): the resident MoE's gate/up part sum, SiLU(gate) * up and the packed Q8
// quantization of the result in one node. parts {2*ff, slots, count} F32 are the gate/up GEMV's column parts, summed in
// part order; xq/dx are q8_quant_packed.c's outputs for activation [slots, ff], with its scale and code arithmetic.
// SiLU uses gdn_math.h's sigmoid, so the activation may differ from the SDK sigmoid in the last bits.
#include "gdn_math.h"
void main(tensor parts,tensor xq,tensor dx,int ff,int slots,int count) {
    const int5 begin=get_index_space_offset(),end=begin+get_index_space_size();
    const int blocks=ff/32;
    const int writes=SW_WR_LOWER_GROUP0|SW_WR_UPPER_GROUP0|SW_WR_LOWER_GROUP1|SW_WR_UPPER_GROUP1|
                     SW_WR_LOWER_GROUP2|SW_WR_UPPER_GROUP2|SW_WR_LOWER_GROUP3|SW_WR_UPPER_GROUP3;
    for(int member=begin[0];member<end[0];++member) {
        const int row=member/blocks,block=member-row*blocks;
        int5 gp={block*32,row,0,0,0},up={ff+block*32,row,0,0,0};
        float64 gate=v_f32_ld_tnsr_b(gp,parts),lift=v_f32_ld_tnsr_b(up,parts);
        for(int p=1;p<count;++p) {gp[2]=p;up[2]=p;gate+=v_f32_ld_tnsr_b(gp,parts);lift+=v_f32_ld_tnsr_b(up,parts);}
        const float64 value=(gate*strata_sigmoid(gate))*lift;
        const float64 magnitude=v_f32_sel_less_u32_b(V_LANE_ID_32,32u,v_f32_sel_less_f32_b(value,0.0f,-value,value),0.0f);
        const float64 peak=v_f32_reduce_max(magnitude),d=peak/127.0f;
        const float64 ratio=v_f32_sel_grt_f32_b(peak,0.0f,value/v_f32_sel_grt_f32_b(peak,0.0f,d,1.0f),0.0f);
        const float64 even=v_convert_i32_to_f32_b(v_convert_f32_to_i32_b(ratio,0),0);
        const float64 sign=v_f32_sel_less_f32_b(ratio,0.0f,-1.0f,1.0f);
        float64 q=even+v_f32_sel_eq_f32_b((ratio-even)*sign,0.5f,sign,0.0f);
        q=v_f32_min_b(v_f32_max_b(q,-127.0f),127.0f);
        int5 dp={0,block,row,0,0};v_f32_st_tnsr(dp,dx,d);
        for(int k=0;k<2;++k) {
            float64 source=0;
            if(k==0) source=v_f32_mov_dual_group_all_b(q,0xffffffff,0,0,0,0,writes,source);
            else source=v_f32_mov_dual_group_all_b(q,0xffffffff,1,1,1,1,writes,source);
            float256 wide;wide.v1=source;wide.v2=source;wide.v3=source;wide.v4=source;
            int5 xp={0,2*block+k,row,0,0};v_i8_st_tnsr(xp,xq,v_convert_f32_to_i8_all_b(wide,SW_LINEAR));
        }
    }
}
