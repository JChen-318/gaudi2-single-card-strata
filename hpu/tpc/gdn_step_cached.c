// Same i-order and per-lane arithmetic as gdn_step.c. q/k fit in vector local memory;
// group moves and shuffles broadcast each element without scalar global loads or expanded q/k tensors.
__local__ float64 qcache[4],kcache[4];
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
void main(tensor state_in,tensor q,tensor k,tensor v,tensor gate,tensor beta,tensor state_out,tensor output,int width,int key_heads) {
    const int5 start=get_index_space_offset(),end=start+get_index_space_size();
    for(int head=start[1];head<end[1];++head) {
        int5 scalar_coord={head,0,0,0,0};
        const float gate_value=s_f32_ld_g((__global__ float*)gen_addr(scalar_coord,gate));
        const float strength=s_f32_ld_g((__global__ float*)gen_addr(scalar_coord,beta));
        float64 gate_vector=gate_value;const float64 decay=v_exp_f32(gate_vector);
        for(int c=0;c<width;c+=64) {
            int5 p={c,head%key_heads,0,0,0};qcache[c/64]=v_f32_ld_tnsr_b(p,q);kcache[c/64]=v_f32_ld_tnsr_b(p,k);
        }
        for(int block=start[0];block<end[0];++block) {
            int5 state_coord={block*64,head,0,0,0};float64 contraction=0;
            for(int chunk=0;chunk<width;chunk+=64) {
                const float64 source=kcache[chunk/64];
                for(int group=0;group<4;++group) {
                    const float64 wide=group16(source,group);
                    for(int j=0;j<16;++j) {
                        state_coord[2]=chunk+group*16+j;
                        const uchar256 index=(unsigned char)(128+(j&7)+((j&8)<<2));
                        const float64 ki=v_f32_shuffle_b(wide,index,0,0),old=v_f32_ld_tnsr_b(state_coord,state_in);
                        contraction+=(old*decay)*ki;
                    }
                }
            }
            int5 value_coord={block*64,head,0,0,0};
            const float64 value=v_f32_ld_tnsr_b(value_coord,v),delta=(value-contraction)*strength;float64 answer=0;
            for(int chunk=0;chunk<width;chunk+=64) {
                const float64 qs=qcache[chunk/64],ks=kcache[chunk/64];
                for(int group=0;group<4;++group) {
                    const float64 qw=group16(qs,group),kw=group16(ks,group);
                    for(int j=0;j<16;++j) {
                        state_coord[2]=chunk+group*16+j;
                        const uchar256 index=(unsigned char)(128+(j&7)+((j&8)<<2));
                        const float64 qi=v_f32_shuffle_b(qw,index,0,0),ki=v_f32_shuffle_b(kw,index,0,0);
                        const float64 old=v_f32_ld_tnsr_b(state_coord,state_in),next=old*decay+delta*ki;
                        v_f32_st_tnsr(state_coord,state_out,next);answer+=next*qi;
                    }
                }
            }
            v_f32_st_tnsr(value_coord,output,answer);
        }
    }
}
