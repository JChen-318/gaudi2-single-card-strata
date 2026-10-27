// STRATA_HPU_HC_FUSED: row_norm_simd.c that also writes its result rounded to BF16 (the SDK cast that followed it),
// so the projection GEMM reads it without a cast node in between. Same arithmetic for the F32 output.
// Members own four rows (glue: ceil(rows / 4)): the four rows' loads, maxima and compensated sums are independent, so
// the compiler interleaves them; each row's operations are row_norm_simd.c's in the same order, so the same bits.
// Each pass reads its rows from HBM again (four rows do not fit the 16 KiB VLM; same values).
// Up to 32 rows (a decoded token's streams, a DFlash verify window) a member owns ONE row (glue: rows members), its
// four slots computing that row (one stored): a few rows would otherwise leave most TPCs idle.
#define RN_ROWS 4
void main(tensor input,tensor gamma,tensor output,tensor rounded,int width,int rows,int shared_gamma,float epsilon) {
    const int5 start=get_index_space_offset(),end=start+get_index_space_size();
    const int per=rows>32?RN_ROWS:1;
    for(int group=start[0];group<end[0] && group*per<rows;++group) {
        int row[RN_ROWS];float64 maximum[RN_ROWS],im[RN_ROWS],sum[RN_ROWS],correction[RN_ROWS],inverse[RN_ROWS];
        int gamma_row[RN_ROWS];
        #pragma unroll
        for(int i=0;i<RN_ROWS;++i) {
            const int own=group*per+(i<per?i:per-1);
            row[i]=own<rows?own:rows-1;maximum[i]=1.0f;sum[i]=0;correction[i]=0;
            gamma_row[i]=shared_gamma==1?0:(shared_gamma<0?row[i]%(-shared_gamma):row[i]);
        }
#ifdef STRATA_FAST_MATH
        // STRATA_HPU_ROW_NORM_FAST: two passes instead of three -- the plain sum of squares (no max pre-scale: it
        // overflows F32 only past |x| ~ 1e18; no compensation, which measured the same logits KL), then the
        // normalization. Not the same bits as the exact passes below (about an ulp in the inverse).
        (void)maximum;(void)correction;
        for(int col=0;col<width;col+=64) {
            #pragma unroll
            for(int i=0;i<RN_ROWS;++i) {
                int5 p={col,row[i],0,0,0};
                const float64 x=v_f32_ld_tnsr_b(p,input);
                sum[i]+=x*x;
            }
        }
        #pragma unroll
        for(int i=0;i<RN_ROWS;++i) {
            im[i]=1.0f;
            inverse[i]=v_rsqrt_f32(v_f32_reduce_add(sum[i])*(1.0f/width)+epsilon);
        }
#else
        for(int col=0;col<width;col+=64) {
            #pragma unroll
            for(int i=0;i<RN_ROWS;++i) {
                int5 p={col,row[i],0,0,0};
                const float64 x=v_f32_ld_tnsr_b(p,input);
                const float64 absolute=v_f32_sel_less_f32_b(x,0.0f,-x,x);
                maximum[i]=v_f32_max_b(maximum[i],absolute);
            }
        }
        #pragma unroll
        for(int i=0;i<RN_ROWS;++i) {maximum[i]=v_f32_reduce_max(maximum[i]);im[i]=1.0f/maximum[i];}
        for(int col=0;col<width;col+=64) {
            #pragma unroll
            for(int i=0;i<RN_ROWS;++i) {
                int5 p={col,row[i],0,0,0};
                const float64 x=v_f32_ld_tnsr_b(p,input)*im[i];
                const float64 adjusted=x*x-correction[i],next=sum[i]+adjusted;
                correction[i]=(next-sum[i])-adjusted;sum[i]=next;
            }
        }
        #pragma unroll
        for(int i=0;i<RN_ROWS;++i) {
            const float64 total=v_f32_reduce_add(sum[i]);
            const float64 mean=total*(1.0f/width)+(epsilon*im[i])*im[i];
            inverse[i]=v_rsqrt_f32(mean)*im[i];
        }
#endif
        for(int col=0;col<width;col+=128) {
            #pragma unroll
            for(int i=0;i<RN_ROWS;++i) {
                if(i>=per || group*per+i>=rows) continue;
                int5 p={col,row[i],0,0,0},g={col,gamma_row[i],0,0,0};
                float128 wide;
                wide.v1=(v_f32_ld_tnsr_b(p,input)*inverse[i])*v_f32_ld_tnsr_b(g,gamma);
                v_f32_st_tnsr(p,output,wide.v1);
                if(col+64<width) {
                    int5 q={col+64,row[i],0,0,0},h={col+64,gamma_row[i],0,0,0};
                    wide.v2=(v_f32_ld_tnsr_b(q,input)*inverse[i])*v_f32_ld_tnsr_b(h,gamma);
                    v_f32_st_tnsr(q,output,wide.v2);
                    v_bf16_st_tnsr(p,rounded,v_convert_f32_to_bf16_all_b(wide,SW_LINEAR));
                } else {
                    wide.v2=0;
                    v_bf16_st_tnsr_partial(p,rounded,v_convert_f32_to_bf16_all_b(wide,SW_LINEAR),63,0);
                }
            }
        }
    }
}
