// STRATA_HPU_GDN_STEP_ROWS: the recurrent steps of a verify window's 2..4 rows in one kernel, with the state read
// twice instead of twice per row. The delta rule S_t = a_t S_{t-1} + k_t (x) d_t (a_t = exp(gate_t), d_t over the
// 64 lanes of a block) gives every row's contraction from the window's first state:
//   S_{t-1}.k_t = A_{t-1} (S_0.k_t) + sum_{u<t} (a_{u+1}..a_{t-1}) d_u (k_u.k_t),   A_{t-1} = a_1..a_{t-1},
// so pass 1 accumulates S_0.k_t for every row at once, the deltas follow from those and the k Gram values, and
// pass 2 walks S_0 -> S_1 -> ... in registers, storing each row's state and accumulating its output. Mathematically
// gdn_step_pipe.c applied row after row; the sums are associated differently, so not bit-identical to it.
// Both passes issue their state loads in groups ahead of use (gdn_step_pipe.c), and every row is computed: rows past
// the window read the last row's inputs and store into the graph's unused state tensors and past the output's end.
//   state_in, state_out_0..3 {s(i), hv, s(j)} as gdn_step; q, k {rows, hk, s}; v, output {rows, hv, s};
//   gate, beta {rows, hv}. One member per (block of 64 j lanes, head), as gdn_step.
__local__ float64 qcache[4][2],kcache[4][2];
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
#define IDX(J) ((unsigned char)(128+((J)&7)+(((J)&8)<<2)))
// One coordinate serves a group's loads (index registers are few); each load reads it when issued.
#define LD(J) (lc[2]=base+(J),v_f32_ld_tnsr_b(lc,state_in))
void main(tensor state_in,tensor q,tensor k,tensor v,tensor gate,tensor beta,
          tensor state_out_0,tensor state_out_1,tensor state_out_2,tensor state_out_3,tensor output,
          int width,int key_heads,int rows) {
    const int5 start=get_index_space_offset(),end=start+get_index_space_size();
    const int chunks=width/64;
    for(int head=start[1];head<end[1];++head) {
        const int r1=rows>1?1:rows-1,r2=rows>2?2:rows-1,r3=rows>3?3:rows-1;
        // gate/beta of this head: vector loads whose lane 0 is the head (scalar global loads wait ~0.5 us each),
        // lane 0 then broadcast to every lane.
        #define BCAST0(T,R) ({int5 c_={head,R,0,0,0};v_f32_shuffle_b(group16(v_f32_ld_tnsr_b(c_,T),0),IDX(0),0,0);})
        const float64 d0=v_exp_f32(BCAST0(gate,0)),d1=v_exp_f32(BCAST0(gate,r1)),d2=v_exp_f32(BCAST0(gate,r2)),d3=v_exp_f32(BCAST0(gate,r3));
        const float64 b0=BCAST0(beta,0),b1=BCAST0(beta,r1),b2=BCAST0(beta,r2),b3=BCAST0(beta,r3);
        #undef BCAST0
        #pragma unroll
        for(int t=0;t<4;++t) {
            const int row=t<rows?t:rows-1;
            #pragma unroll
            for(int c=0;c<2;++c) {
                int5 p={c*64,head%key_heads,row,0,0};qcache[t][c]=c<chunks?v_f32_ld_tnsr_b(p,q):0;kcache[t][c]=c<chunks?v_f32_ld_tnsr_b(p,k):0;
            }
        }
        // k_u . k_t for u < t (every lane holds the sum).
        #define GRAM(U,T) v_f32_reduce_add(kcache[U][0]*kcache[T][0]+kcache[U][1]*kcache[T][1])
        const float64 g01=GRAM(0,1),g02=GRAM(0,2),g03=GRAM(0,3),g12=GRAM(1,2),g13=GRAM(1,3),g23=GRAM(2,3);
        #undef GRAM
        for(int block=start[0];block<end[0];++block) {
            // Pass 1: S_0 . k_t for every row, eight state rows loaded ahead at a time.
            float64 dot0=0,dot1=0,dot2=0,dot3=0;
            for(int c=0;c<chunks;++c)
            for(int group=0;group<4;++group) {
                const float64 k0=group16(kcache[0][c],group),k1=group16(kcache[1][c],group);
                const float64 k2=group16(kcache[2][c],group),k3=group16(kcache[3][c],group);
                #pragma unroll
                for(int hh=0;hh<2;++hh) {
                    int5 lc={block*64,head,0,0,0};const int base=c*64+group*16+hh*8;
                    const float64 o0=LD(0),o1=LD(1),o2=LD(2),o3=LD(3),o4=LD(4),o5=LD(5),o6=LD(6),o7=LD(7);
                    #define P(J) {const unsigned char ix=IDX(hh*8+J);                         dot0+=o##J*v_f32_shuffle_b(k0,ix,0,0);dot1+=o##J*v_f32_shuffle_b(k1,ix,0,0);                         dot2+=o##J*v_f32_shuffle_b(k2,ix,0,0);dot3+=o##J*v_f32_shuffle_b(k3,ix,0,0);}
                    P(0) P(1) P(2) P(3) P(4) P(5) P(6) P(7)
                    #undef P
                }
            }
            // Deltas, row by row: contraction_t = a_t (S_{t-1} . k_t), the products associated left to right.
            int5 vc={block*64,head,0,0,0};
            const float64 v0=v_f32_ld_tnsr_b(vc,v);vc[2]=r1;const float64 v1=v_f32_ld_tnsr_b(vc,v);
            vc[2]=r2;const float64 v2=v_f32_ld_tnsr_b(vc,v);vc[2]=r3;const float64 v3=v_f32_ld_tnsr_b(vc,v);
            const float64 e0=(v0-dot0*d0)*b0;
            const float64 along1=dot1*d0+e0*g01;
            const float64 e1=(v1-along1*d1)*b1;
            const float64 along2=dot2*(d0*d1)+(e0*d1)*g02+e1*g12;
            const float64 e2=(v2-along2*d2)*b2;
            const float64 along3=dot3*((d0*d1)*d2)+(e0*(d1*d2))*g03+(e1*d2)*g13+e2*g23;
            const float64 e3=(v3-along3*d3)*b3;
            // Pass 2: S_0 -> S_1 -> ..., each row's state stored and its output accumulated; eight rows loaded ahead.
            float64 answer0=0,answer1=0,answer2=0,answer3=0;
            for(int c=0;c<chunks;++c)
            for(int group=0;group<4;++group) {
                const float64 k0=group16(kcache[0][c],group),k1=group16(kcache[1][c],group);
                const float64 k2=group16(kcache[2][c],group),k3=group16(kcache[3][c],group);
                const float64 q0=group16(qcache[0][c],group),q1=group16(qcache[1][c],group);
                const float64 q2=group16(qcache[2][c],group),q3=group16(qcache[3][c],group);
                #pragma unroll 1 // (unrolled, the scheduler hoists both halves and spills)
                for(int hh=0;hh<2;++hh) {
                    int5 lc={block*64,head,0,0,0};const int base=c*64+group*16+hh*8;
                    const float64 o0=LD(0),o1=LD(1),o2=LD(2),o3=LD(3),o4=LD(4),o5=LD(5),o6=LD(6),o7=LD(7);
                    #define U(J) {const unsigned char ix=IDX(hh*8+J);lc[2]=base+J;float64 o=o##J; \
                        o=o*d0+e0*v_f32_shuffle_b(k0,ix,0,0);v_f32_st_tnsr(lc,state_out_0,o);answer0+=o*v_f32_shuffle_b(q0,ix,0,0); \
                        o=o*d1+e1*v_f32_shuffle_b(k1,ix,0,0);v_f32_st_tnsr(lc,state_out_1,o);answer1+=o*v_f32_shuffle_b(q1,ix,0,0); \
                        o=o*d2+e2*v_f32_shuffle_b(k2,ix,0,0);v_f32_st_tnsr(lc,state_out_2,o);answer2+=o*v_f32_shuffle_b(q2,ix,0,0); \
                        o=o*d3+e3*v_f32_shuffle_b(k3,ix,0,0);v_f32_st_tnsr(lc,state_out_3,o);answer3+=o*v_f32_shuffle_b(q3,ix,0,0);}
                    U(0) U(1) U(2) U(3) U(4) U(5) U(6) U(7)
                    #undef U
                }
            }
            // Rows past the window land past the output's end (dropped stores).
            int5 oc={block*64,head,0,0,0};
            v_f32_st_tnsr(oc,output,answer0);oc[2]=1;v_f32_st_tnsr(oc,output,answer1);
            oc[2]=2;v_f32_st_tnsr(oc,output,answer2);oc[2]=3;v_f32_st_tnsr(oc,output,answer3);
        }
    }
}
