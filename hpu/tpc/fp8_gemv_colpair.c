// One SIMD lane per output row. Adjacent input columns are interleaved in the resident codes:
// [rows/128][columns/2][128][2]. Within each 128-row block, rows are interleaved as
// 0,64,1,65,...,63,127 to match the BF16 MAC accumulator halves.
// Row scales retain their original order. No horizontal dot-product reduction is needed.
__local__ bfloat128 xc[20];
void main(tensor x,tensor w0,tensor w1,tensor w2,tensor w3,tensor w4,tensor w5,tensor w6,tensor w7,tensor w8,tensor w9,tensor w10,
          tensor out,int columns,int rows,int shared_x) {
    const int5 begin=get_index_space_offset(),end=begin+get_index_space_size();
    const int blocks=rows/128;int loaded=-1;
    for(int member=begin[0];member<end[0];++member) {
        const int e=member/blocks,block=member-e*blocks,W=1+e;
        if(e!=loaded) {
            loaded=e;
            for(int c=0;c<columns;c+=128) {
                int5 p={c,shared_x?0:e,0,0,0};
                xc[c/128]=v_bf16_ld_tnsr_b(p,x);
            }
        }
        const int r0=block*128;float128 a0={0},a1={0},a2={0},a3={0};
        for(int chunk=0;chunk<columns;chunk+=128) {
            const bfloat128 source=xc[chunk/128];
            #pragma clang loop unroll(disable)
            for(int group=0;group<4;++group) {
                bfloat128 wide=0;
                if(group==0) wide=v_bf16_mov_dual_group_all_b(source,0xffffffff,0,0,0,0,SW_WR_LOWER_GROUP0|SW_WR_UPPER_GROUP0|SW_WR_LOWER_GROUP1|SW_WR_UPPER_GROUP1|SW_WR_LOWER_GROUP2|SW_WR_UPPER_GROUP2|SW_WR_LOWER_GROUP3|SW_WR_UPPER_GROUP3,wide);
                if(group==1) wide=v_bf16_mov_dual_group_all_b(source,0xffffffff,1,1,1,1,SW_WR_LOWER_GROUP0|SW_WR_UPPER_GROUP0|SW_WR_LOWER_GROUP1|SW_WR_UPPER_GROUP1|SW_WR_LOWER_GROUP2|SW_WR_UPPER_GROUP2|SW_WR_LOWER_GROUP3|SW_WR_UPPER_GROUP3,wide);
                if(group==2) wide=v_bf16_mov_dual_group_all_b(source,0xffffffff,2,2,2,2,SW_WR_LOWER_GROUP0|SW_WR_UPPER_GROUP0|SW_WR_LOWER_GROUP1|SW_WR_UPPER_GROUP1|SW_WR_LOWER_GROUP2|SW_WR_UPPER_GROUP2|SW_WR_LOWER_GROUP3|SW_WR_UPPER_GROUP3,wide);
                if(group==3) wide=v_bf16_mov_dual_group_all_b(source,0xffffffff,3,3,3,3,SW_WR_LOWER_GROUP0|SW_WR_UPPER_GROUP0|SW_WR_LOWER_GROUP1|SW_WR_UPPER_GROUP1|SW_WR_LOWER_GROUP2|SW_WR_UPPER_GROUP2|SW_WR_LOWER_GROUP3|SW_WR_UPPER_GROUP3,wide);
                #pragma unroll(16)
                for(int j=0;j<32;j+=2) {
                    const int c=chunk+group*32+j;
                    int5 p={block*columns*128+(c/2)*256,0,0,0,0};
                    const bfloat256 q=v_convert_f8_to_bf16_all_b(v_f8_ld_tnsr_b(p,W),0);
                    // BF16 shuffle addresses 16-element groups: bit5 selects the other half;
                    // bit4 is unused. Four independent accumulators hide BF16 MAC latency.
                    const uchar256 index0=(unsigned char)(128+(j&15)+((j&16)<<1)),index1=(unsigned char)(129+(j&15)+((j&16)<<1));
                    const bfloat128 x0=v_bf16_shuffle_b(wide,index0,0,0),x1=v_bf16_shuffle_b(wide,index1,0,0);
                    if(j%4==0) {a0=v_bf16_mac_acc32_b(q.v1,x0,a0);a1=v_bf16_mac_acc32_b(q.v2,x1,a1);}
                    else {a2=v_bf16_mac_acc32_b(q.v1,x0,a2);a3=v_bf16_mac_acc32_b(q.v2,x1,a3);}
                }
            }
        }
        const float128 sum={((a0.v1+a1.v1)+a2.v1)+a3.v1,((a0.v2+a1.v2)+a2.v2)+a3.v2};
        int5 s0={rows*columns+r0*4,0,0,0,0},s1={rows*columns+(r0+64)*4,0,0,0,0};
        int5 o0={r0,e,0,0,0},o1={r0+64,e,0,0,0};
        v_f32_st_tnsr(o0,out,sum.v1*(float64)v_u8_ld_tnsr_b(s0,W));
        v_f32_st_tnsr(o1,out,sum.v2*(float64)v_u8_ld_tnsr_b(s1,W));
    }
}
