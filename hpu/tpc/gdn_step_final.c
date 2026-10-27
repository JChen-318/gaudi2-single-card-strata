// Full-window prefill: only final state is published. State stays in local vector memory between rows.
// One member owns 32 value lanes. Two key rows share a vector in local memory,
// keeping width128's matrix in exactly 16 KiB without increasing Gaudi2 VLM.
__local__ float64 state_cache[64];
float64 group16(float64 source,int group) {
    float64 wide=0;
    const int writes=SW_WR_LOWER_GROUP0|SW_WR_UPPER_GROUP0|SW_WR_LOWER_GROUP1|SW_WR_UPPER_GROUP1|
                     SW_WR_LOWER_GROUP2|SW_WR_UPPER_GROUP2|SW_WR_LOWER_GROUP3|SW_WR_UPPER_GROUP3;
    if(group==0) wide=v_f32_mov_dual_group_all_b(source,0xffffffff,0,0,0,0,writes,wide);
    if(group==1) wide=v_f32_mov_dual_group_all_b(source,0xffffffff,1,1,1,1,writes,wide);
    if(group==2) wide=v_f32_mov_dual_group_all_b(source,0xffffffff,2,2,2,2,writes,wide);
    if(group==3) wide=v_f32_mov_dual_group_all_b(source,0xffffffff,3,3,3,3,writes,wide);
    return wide;
}
float64 half32(float64 source,int half_index) {
    const int writes=SW_WR_LOWER_GROUP0|SW_WR_UPPER_GROUP0|SW_WR_LOWER_GROUP1|SW_WR_UPPER_GROUP1|
                     SW_WR_LOWER_GROUP2|SW_WR_UPPER_GROUP2|SW_WR_LOWER_GROUP3|SW_WR_UPPER_GROUP3;
    if(half_index==0) return v_f32_mov_dual_group_all_b(source,0xffffffff,0,1,0,1,writes,source);
    return v_f32_mov_dual_group_all_b(source,0xffffffff,2,3,2,3,writes,source);
}
float64 set_half(float64 packed,float64 source,int half_index) {
    const int lo=SW_WR_LOWER_GROUP0|SW_WR_UPPER_GROUP0|SW_WR_LOWER_GROUP1|SW_WR_UPPER_GROUP1;
    const int hi=SW_WR_LOWER_GROUP2|SW_WR_UPPER_GROUP2|SW_WR_LOWER_GROUP3|SW_WR_UPPER_GROUP3;
    if(half_index==0) return v_f32_mov_dual_group_all_b(source,0xffffffff,0,1,0,1,lo,packed);
    return v_f32_mov_dual_group_all_b(source,0xffffffff,0,1,0,1,hi,packed);
}
void main(tensor state_in,tensor q,tensor k,tensor v,tensor gate,tensor beta,tensor state_out,tensor output,int width,int key_heads,int rows) {
    const int5 start=get_index_space_offset(),end=start+get_index_space_size();
    for(int head=start[1];head<end[1];++head) for(int block=start[0];block<end[0];++block) {
        int5 sc={block*32,head,0,0,0};
        for(int i=0;i<width;i+=2) {
            sc[2]=i;const float64 a=v_f32_ld_tnsr_b(sc,state_in);
            sc[2]=i+1;const float64 b=v_f32_ld_tnsr_b(sc,state_in);
            state_cache[i/2]=set_half(half32(a,0),b,1);
        }
        for(int row=0;row<rows;++row) {
            int5 gc={head,row,0,0,0};
            const float gate_value=s_f32_ld_g((__global__ float*)gen_addr(gc,gate));
            const float strength=s_f32_ld_g((__global__ float*)gen_addr(gc,beta));
            float64 gate_vector=gate_value;const float64 decay=v_exp_f32(gate_vector);
            float64 contraction=0;
            for(int chunk=0;chunk<width;chunk+=64) {
                int5 p={chunk,head%key_heads,row,0,0};const float64 source=v_f32_ld_tnsr_b(p,k);
                for(int group=0;group<4;++group) {
                    const float64 wide=group16(source,group);
                    for(int j=0;j<16;++j) {
                        const int i=chunk+group*16+j;
                        const uchar256 index=(unsigned char)(128+(j&7)+((j&8)<<2));
                        const float64 ki=v_f32_shuffle_b(wide,index,0,0),old=half32(state_cache[i/2],i%2);
                        contraction+=(old*decay)*ki;
                    }
                }
            }
            int5 vc={block*32,head,row,0,0};
            const float64 value=v_f32_ld_tnsr_b(vc,v),delta=(value-contraction)*strength;float64 answer=0;
            for(int chunk=0;chunk<width;chunk+=64) {
                int5 p={chunk,head%key_heads,row,0,0};const float64 qs=v_f32_ld_tnsr_b(p,q),ks=v_f32_ld_tnsr_b(p,k);
                for(int group=0;group<4;++group) {
                    const float64 qw=group16(qs,group),kw=group16(ks,group);
                    for(int j=0;j<16;++j) {
                        const int i=chunk+group*16+j;
                        const uchar256 index=(unsigned char)(128+(j&7)+((j&8)<<2));
                        const float64 qi=v_f32_shuffle_b(qw,index,0,0),ki=v_f32_shuffle_b(kw,index,0,0);
                        const float64 next=half32(state_cache[i/2],i%2)*decay+delta*ki;
                        state_cache[i/2]=set_half(state_cache[i/2],next,i%2);answer+=next*qi;
                    }
                }
            }
            v_f32_st_tnsr_partial(vc,output,answer,31,0);
        }
        for(int i=0;i<width;++i) {sc[2]=i;v_f32_st_tnsr_partial(sc,state_out,half32(state_cache[i/2],i%2),31,0);}
    }
}
