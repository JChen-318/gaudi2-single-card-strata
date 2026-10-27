// STRATA_HPU_QSA_DEVICE: the host steps of one decode token's QSA attention, on the device (QsaBlock::fused_attend
// without selection): weighted RMS norm of every query and key head row (gamma shared by the heads), NeoX RoPE on
// the first `rotary` columns, and the key/value rows rounded to FP16-representable F32 (the cache stores FP16).
// Rows 0..query_heads-1 are query heads, then kv_heads key rows (each also converts its value). A member owns up to four
// rows of one kind of one token (query heads in fours, then the key heads in fours; glue: tokens x groups), run side by
// side so their load / reduction / RoPE chains overlap; each row's operations are the one-row ones in their order.
//   query {width, query_heads} F32 out; k_slot/v_slot {width, kv_heads} (the cache row, bound at the cell) and
//   k_now/v_now {width, kv_heads} (the same values, for this graph's attention and the host commit).
// RoPE pairs column i with i + rotary/2 (i < rotary/2): both halves come from two loads offset by rotary/2 columns,
// so no lane shuffles are needed. rope is {rotary}: rotary/2 cosines then rotary/2 sines of this cell's position.
// The norm sums squares in F32 trees (the host sums in double), so this is its own arithmetic mode.
static inline float64 fp16_grid(float64 x) {
    // Values on the FP16 grid: 11 significant bits for normal magnitudes (Veltkamp split), multiples of 2^-24 below
    // 2^-14 (adding and removing 0.75, whose F32 ulp is 2^-24). Overflow is left to the host's FP16 append check.
    const float64 big=x*8193.0f;
    const float64 normal=big-(big-x);
    const float64 tiny=(x+0.75f)-0.75f;
    const float64 magnitude=v_f32_sel_less_f32_b(x,0.0f,-x,x);
    return v_f32_sel_less_f32_b(magnitude,6.103515625e-05f,tiny,normal);
}
void main(tensor q,tensor k,tensor v,tensor q_gamma,tensor k_gamma,tensor rope,
          tensor query,tensor k_slot,tensor v_slot,tensor k_now,tensor v_now,
          int width,int query_heads,int kv_heads,int rotary,float epsilon,int tokens) {
    const int5 begin=get_index_space_offset(),end=begin+get_index_space_size();
    const int pairs=rotary/2;
    // tokens > 1 (target verification): every tensor gains a token axis, rope holds one row per token.
    (void)tokens;
    const int query_groups=(query_heads+3)/4,key_groups=(kv_heads+3)/4,groups=query_groups+key_groups;
    for(int member=begin[0];member<end[0];++member) {
        const int token=member/groups,group=member-token*groups;
        const bool is_key=group>=query_groups;
        const int first=is_key?4*(group-query_groups):4*group,limit=is_key?kv_heads:query_heads;
        const int count=limit-first<4?limit-first:4;
        int5 z={0,token,0,0,0};
        int5 hp=z;hp[0]=pairs;
        const float64 cosine=v_f32_ld_tnsr_b(z,rope),sine=v_f32_ld_tnsr_b(hp,rope);
        int row[4];float64 squares[4],inverse[4];
        #pragma unroll
        for(int i=0;i<4;++i) {row[i]=first+(i<count?i:0);squares[i]=0;}
        for(int col=0;col<width;col+=64) {
            #pragma unroll
            for(int i=0;i<4;++i) {
                int5 p={col,row[i],token,0,0};
                const float64 x=is_key?v_f32_ld_tnsr_b(p,k):v_f32_ld_tnsr_b(p,q);
                squares[i]+=x*x;
            }
        }
        #pragma unroll
        for(int i=0;i<4;++i) {
            const float64 total=v_f32_reduce_add(squares[i]);
            inverse[i]=1.0f/v_sqrt_f32(total/(float)width+epsilon);
        }
        // Columns from 64 on (and the rotary tail inside the first 64 when rotary < 64 is not supported here).
        // The gammas are {width}, shared by every head: they are read at row 0 (a row coordinate would be out of
        // bounds and load zeros).
        for(int col=64;col<width;col+=64) {
            int5 pg={col,0,0,0,0};
            const float64 g=is_key?v_f32_ld_tnsr_b(pg,k_gamma):v_f32_ld_tnsr_b(pg,q_gamma);
            #pragma unroll
            for(int i=0;i<4;++i) {
                if(i>=count) break;
                int5 p={col,row[i],token,0,0};
                float64 x=is_key?v_f32_ld_tnsr_b(p,k):v_f32_ld_tnsr_b(p,q);
                x=(x*inverse[i])*g;
                if(is_key) {x=fp16_grid(x);v_f32_st_tnsr(p,k_slot,x);v_f32_st_tnsr(p,k_now,x);}
                else v_f32_st_tnsr(p,query,x);
            }
        }
        const int5 ga_at={0,0,0,0,0},gb_at={pairs,0,0,0,0};
        const float64 ga=is_key?v_f32_ld_tnsr_b(ga_at,k_gamma):v_f32_ld_tnsr_b(ga_at,q_gamma),gb=is_key?v_f32_ld_tnsr_b(gb_at,k_gamma):v_f32_ld_tnsr_b(gb_at,q_gamma);
        #pragma unroll
        for(int i=0;i<4;++i) {
            if(i>=count) break;
            int5 a={0,row[i],token,0,0},b={pairs,row[i],token,0,0};
            const float64 xa=is_key?v_f32_ld_tnsr_b(a,k):v_f32_ld_tnsr_b(a,q),xb=is_key?v_f32_ld_tnsr_b(b,k):v_f32_ld_tnsr_b(b,q);
            const float64 na=(xa*inverse[i])*ga,nb=(xb*inverse[i])*gb;
            float64 left=na*cosine-nb*sine,right=na*sine+nb*cosine;
            // Lanes [pairs, 64) of the first vector hold unrotated columns rotary..63 (none when rotary == 64).
            float64 head=v_f32_sel_less_u32_b(V_LANE_ID_32,(unsigned)pairs,left,na);
            if(is_key) {left=fp16_grid(head);right=fp16_grid(right);}
            else left=head;
            if(is_key) {
                v_f32_st_tnsr_partial(a,k_slot,left,63,0);v_f32_st_tnsr_partial(a,k_now,left,63,0);
                v_f32_st_tnsr_partial(b,k_slot,right,pairs-1,0);v_f32_st_tnsr_partial(b,k_now,right,pairs-1,0);
                for(int col=0;col<width;col+=64) {
                    int5 p={col,row[i],token,0,0};
                    const float64 value=fp16_grid(v_f32_ld_tnsr_b(p,v));
                    v_f32_st_tnsr(p,v_slot,value);v_f32_st_tnsr(p,v_now,value);
                }
            } else {
                v_f32_st_tnsr_partial(a,query,left,63,0);
                v_f32_st_tnsr_partial(b,query,right,pairs-1,0);
            }
        }
    }
}
