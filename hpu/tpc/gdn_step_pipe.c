// STRATA_HPU_GDN_STEP_PIPE: gdn_step_cached.c with the state loads issued sixteen at a time ahead of their use, so
// the loads of a group are in flight together instead of each one waiting for the previous multiply-add. Same tensors,
// index space, i-order and per-lane arithmetic as gdn_step_cached.c (only the load order moves), so the state and
// output are identical.
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
// One coordinate serves the sixteen loads (index registers are few); each load reads it when issued.
#define LD(J,BASE) (lc[2]=(BASE)+(J),v_f32_ld_tnsr_b(lc,state_in))
#define LOAD16(BASE) \
    int5 lc=state_coord; \
    const float64 o0=LD(0,BASE),o1=LD(1,BASE),o2=LD(2,BASE),o3=LD(3,BASE),o4=LD(4,BASE),o5=LD(5,BASE),o6=LD(6,BASE),o7=LD(7,BASE); \
    const float64 o8=LD(8,BASE),o9=LD(9,BASE),o10=LD(10,BASE),o11=LD(11,BASE),o12=LD(12,BASE),o13=LD(13,BASE),o14=LD(14,BASE),o15=LD(15,BASE);
#define IDX(J) ((unsigned char)(128+((J)&7)+(((J)&8)<<2)))
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
                    LOAD16(chunk+group*16)
                    #define C(J) contraction+=(o##J*decay)*v_f32_shuffle_b(wide,IDX(J),0,0);
                    C(0) C(1) C(2) C(3) C(4) C(5) C(6) C(7) C(8) C(9) C(10) C(11) C(12) C(13) C(14) C(15)
                    #undef C
                }
            }
            int5 value_coord={block*64,head,0,0,0};
            const float64 value=v_f32_ld_tnsr_b(value_coord,v),delta=(value-contraction)*strength;float64 answer=0;
            for(int chunk=0;chunk<width;chunk+=64) {
                const float64 qs=qcache[chunk/64],ks=kcache[chunk/64];
                for(int group=0;group<4;++group) {
                    const float64 qw=group16(qs,group),kw=group16(ks,group);
                    LOAD16(chunk+group*16)
                    #define U(J) { const float64 qi=v_f32_shuffle_b(qw,IDX(J),0,0),ki=v_f32_shuffle_b(kw,IDX(J),0,0); \
                                   const float64 next=o##J*decay+delta*ki;lc[2]=chunk+group*16+(J);v_f32_st_tnsr(lc,state_out,next);answer+=next*qi; }
                    U(0) U(1) U(2) U(3) U(4) U(5) U(6) U(7) U(8) U(9) U(10) U(11) U(12) U(13) U(14) U(15)
                    #undef U
                }
            }
            v_f32_st_tnsr(value_coord,output,answer);
        }
    }
}
