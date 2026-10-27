// Same dynamically scaled FP8 hi/lo activation representation as the MME path.
__local__ bfloat128 xl[20], ll[20]; // padded columns / 128, up to 2560 columns
void main(tensor x,tensor low,tensor xscale,tensor w0,tensor w1,tensor w2,tensor w3,tensor w4,tensor w5,tensor w6,tensor w7,tensor w8,tensor w9,tensor w10,
          tensor out,int columns,int rows,int shared_x) {
    const int5 begin=get_index_space_offset(),end=begin+get_index_space_size();
    const int padded=(columns+255)/256*256,full=columns/256*256;
    const unsigned char tail=(unsigned char)(columns-full);
    // One index-space member per (expert, block of 64 rows), flattened so the compiler spreads them evenly.
    const int blocks=(rows+63)/64;int loaded=-1;float64 activation_scale=1;
    for(int member=begin[0];member<end[0];++member) {
        const int e=member/blocks,block=member-e*blocks;
        if(e!=loaded) {
            const int xrow=shared_x?0:e;loaded=e;
            int5 sc={0,xrow,0,0,0};activation_scale=v_f32_reduce_max(v_f32_ld_tnsr_b(sc,xscale));
            for(int c=0;c<padded;c+=128) {int5 p={c,xrow,0,0,0};xl[c/128]=v_bf16_ld_tnsr_b(p,x);ll[c/128]=v_bf16_ld_tnsr_b(p,low);}
        }
        // Gaudi2 takes the tensor id of ld_tnsr from a register: one copy of the loop serves every expert.
        const int W=3+e;
        {
            const int r0=block*64;float64 result=0;
            for(int r=0;r<64 && r0+r<rows;r+=4) {
                const int b0=(r0+r+0<rows?r0+r+0:r0+r)*columns;float128 a0={0},d0={0};
                const int b1=(r0+r+1<rows?r0+r+1:r0+r)*columns;float128 a1={0},d1={0};
                const int b2=(r0+r+2<rows?r0+r+2:r0+r)*columns;float128 a2={0},d2={0};
                const int b3=(r0+r+3<rows?r0+r+3:r0+r)*columns;float128 a3={0},d3={0};
                for(int c=0;c<full;c+=256) {
                    const bfloat128 x0=xl[c/128],x1=xl[c/128+1],l0=ll[c/128],l1=ll[c/128+1];
                    {int5 p={b0+c,0,0,0,0};const bfloat256 q=v_convert_f8_to_bf16_all_b(v_f8_ld_tnsr_b(p,W),0);
                     a0=v_bf16_mac_acc32_b(q.v1,x0,a0);a0=v_bf16_mac_acc32_b(q.v2,x1,a0);
                     d0=v_bf16_mac_acc32_b(q.v1,l0,d0);d0=v_bf16_mac_acc32_b(q.v2,l1,d0);}
                    {int5 p={b1+c,0,0,0,0};const bfloat256 q=v_convert_f8_to_bf16_all_b(v_f8_ld_tnsr_b(p,W),0);
                     a1=v_bf16_mac_acc32_b(q.v1,x0,a1);a1=v_bf16_mac_acc32_b(q.v2,x1,a1);
                     d1=v_bf16_mac_acc32_b(q.v1,l0,d1);d1=v_bf16_mac_acc32_b(q.v2,l1,d1);}
                    {int5 p={b2+c,0,0,0,0};const bfloat256 q=v_convert_f8_to_bf16_all_b(v_f8_ld_tnsr_b(p,W),0);
                     a2=v_bf16_mac_acc32_b(q.v1,x0,a2);a2=v_bf16_mac_acc32_b(q.v2,x1,a2);
                     d2=v_bf16_mac_acc32_b(q.v1,l0,d2);d2=v_bf16_mac_acc32_b(q.v2,l1,d2);}
                    {int5 p={b3+c,0,0,0,0};const bfloat256 q=v_convert_f8_to_bf16_all_b(v_f8_ld_tnsr_b(p,W),0);
                     a3=v_bf16_mac_acc32_b(q.v1,x0,a3);a3=v_bf16_mac_acc32_b(q.v2,x1,a3);
                     d3=v_bf16_mac_acc32_b(q.v1,l0,d3);d3=v_bf16_mac_acc32_b(q.v2,l1,d3);}
                }
                if(tail) {
                    // Codes past the row end belong to the next row or to the scales: they become +0 before the
                    // conversion, so a NaN-looking byte cannot turn the zero padding of x into a NaN.
                    const bfloat128 x0=xl[full/128],x1=xl[full/128+1],l0=ll[full/128],l1=ll[full/128+1];const uchar256 lane=V_LANE_ID_8;
                    {int5 p={b0+full,0,0,0,0};const uchar256 raw=v_u8_sel_less_u8_b(lane,tail,v_u8_ld_tnsr_b(p,W),(unsigned char)0);
                     const bfloat256 q=v_convert_f8_to_bf16_all_b((minifloat256)raw,0);
                     a0=v_bf16_mac_acc32_b(q.v1,x0,a0);a0=v_bf16_mac_acc32_b(q.v2,x1,a0);
                     d0=v_bf16_mac_acc32_b(q.v1,l0,d0);d0=v_bf16_mac_acc32_b(q.v2,l1,d0);}
                    {int5 p={b1+full,0,0,0,0};const uchar256 raw=v_u8_sel_less_u8_b(lane,tail,v_u8_ld_tnsr_b(p,W),(unsigned char)0);
                     const bfloat256 q=v_convert_f8_to_bf16_all_b((minifloat256)raw,0);
                     a1=v_bf16_mac_acc32_b(q.v1,x0,a1);a1=v_bf16_mac_acc32_b(q.v2,x1,a1);
                     d1=v_bf16_mac_acc32_b(q.v1,l0,d1);d1=v_bf16_mac_acc32_b(q.v2,l1,d1);}
                    {int5 p={b2+full,0,0,0,0};const uchar256 raw=v_u8_sel_less_u8_b(lane,tail,v_u8_ld_tnsr_b(p,W),(unsigned char)0);
                     const bfloat256 q=v_convert_f8_to_bf16_all_b((minifloat256)raw,0);
                     a2=v_bf16_mac_acc32_b(q.v1,x0,a2);a2=v_bf16_mac_acc32_b(q.v2,x1,a2);
                     d2=v_bf16_mac_acc32_b(q.v1,l0,d2);d2=v_bf16_mac_acc32_b(q.v2,l1,d2);}
                    {int5 p={b3+full,0,0,0,0};const uchar256 raw=v_u8_sel_less_u8_b(lane,tail,v_u8_ld_tnsr_b(p,W),(unsigned char)0);
                     const bfloat256 q=v_convert_f8_to_bf16_all_b((minifloat256)raw,0);
                     a3=v_bf16_mac_acc32_b(q.v1,x0,a3);a3=v_bf16_mac_acc32_b(q.v2,x1,a3);
                     d3=v_bf16_mac_acc32_b(q.v1,l0,d3);d3=v_bf16_mac_acc32_b(q.v2,l1,d3);}
                }
                result=v_f32_sel_eq_u32_b(V_LANE_ID_32,(unsigned)(r+0),(v_f32_reduce_add(a0.v1+a0.v2)+v_f32_reduce_add(d0.v1+d0.v2)*(1.0f/16.0f))*activation_scale,result);
                result=v_f32_sel_eq_u32_b(V_LANE_ID_32,(unsigned)(r+1),(v_f32_reduce_add(a1.v1+a1.v2)+v_f32_reduce_add(d1.v1+d1.v2)*(1.0f/16.0f))*activation_scale,result);
                result=v_f32_sel_eq_u32_b(V_LANE_ID_32,(unsigned)(r+2),(v_f32_reduce_add(a2.v1+a2.v2)+v_f32_reduce_add(d2.v1+d2.v2)*(1.0f/16.0f))*activation_scale,result);
                result=v_f32_sel_eq_u32_b(V_LANE_ID_32,(unsigned)(r+3),(v_f32_reduce_add(a3.v1+a3.v2)+v_f32_reduce_add(d3.v1+d3.v2)*(1.0f/16.0f))*activation_scale,result);
            }
            int5 sp={rows*columns+r0*4,0,0,0,0};
            result=result*(float64)v_u8_ld_tnsr_b(sp,W); // the 64 per-row F32 scales, read as bytes
            int5 o={r0,e,0,0,0};const int count=rows-r0<64?rows-r0:64;
            v_f32_st_tnsr_partial(o,out,result,count-1,0);
        }
    }
}
