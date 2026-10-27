// STRATA_HPU_QSA_VERIFY_BF16: qsa_gather_window_fp8.c with members of 16 ranks (the ranks' block ids and scores
// loaded first, so their scalar-load latencies overlap instead of two serial loads per rank) and the codes written
// widened to BF16 (exact) in the TPC's native conversion order, as qsa_gather_rows_fp8.c: a cell's 256 values as its
// even then its odd channels; the attention permutes the query to match and its context back. Same cells, scales and
// mask as qsa_gather_window_fp8.c.
//   keys, values {256, heads, cells} FP8; key/value scales {32, cells} F32; top_ids {top, rows} I32;
//   top_scores {top, rows} F32; cell {1} F32 -> key, value {256, n, heads, rows} BF16 (even/odd order); key/value
//   scales {n, heads, rows} F32; mask {n, rows} F32 (0 kept, -3e38 not), n = (top + 1) * 4.
//   Members are (16-rank group of [0, top], head, row). block == 4.
void main(tensor keys,tensor values,tensor key_scale,tensor value_scale,tensor top_ids,tensor top_scores,tensor cell_t,
          tensor key,tensor value,tensor key_out_scale,tensor value_out_scale,tensor mask,
          int heads,int cells,int top,int block,int width_sel,int rows) {
    const int5 begin=get_index_space_offset(),end=begin+get_index_space_size();
    (void)heads;(void)rows;(void)block;
    int5 z={0,0,0,0,0};
    const int cell=(int)s_f32_ld_g((__global__ float*)gen_addr(z,cell_t));
    const int own=cell/4,before=cell-own*4;
    const int64 lane=(int64)V_LANE_ID_32;
    for(int row=begin[2];row<end[2];++row)
    for(int head=begin[1];head<end[1];++head)
    for(int group=begin[0];group<end[0];++group) {
        const int first=group*16;
        int id[16];float score[16];
        #pragma unroll
        for(int k=0;k<16;++k) {
            int5 tp={first+k<top?first+k:top-1,row,0,0,0};
            id[k]=s_i32_ld_g((__global__ int*)gen_addr(tp,top_ids));
            score[k]=s_f32_ld_g((__global__ float*)gen_addr(tp,top_scores));
        }
        // Two ranks (eight cells) at a time: their sixteen code loads issued before any conversion or store.
        #pragma unroll
        for(int k=0;k<16;k+=2) {
            int at[8],count[2];
            #pragma unroll
            for(int u=0;u<2;++u) {
                const int rank=first+k+u;
                int bid=own,n=before;
                if(rank<top) {
                    bid=id[k+u];
                    const int left=width_sel-1-row-before-rank*4;
                    n=score[k+u]>-1.0e38f && bid>=0 && bid<own?(left<4?(left>0?left:0):4):0;
                }
                if(rank>top) n=0;
                count[u]=n;
                #pragma unroll
                for(int o=0;o<4;++o) {const int c=bid*4+o;at[u*4+o]=o<n && c>=0 && c<cells?c:cells;}
            }
            minifloat256 kc[8],vc[8];
            #pragma unroll
            for(int i=0;i<8;++i) {int5 src={0,head,at[i],0,0};kc[i]=v_f8_ld_tnsr_b(src,keys);vc[i]=v_f8_ld_tnsr_b(src,values);}
            #pragma unroll
            for(int i=0;i<8;++i) {
                const int rank=first+k+i/4;
                if(rank>top) break;
                int5 dst={0,rank*4+i%4,head,row,0};
                const bfloat256 kw=v_convert_f8_to_bf16_all_b(kc[i],0),vw=v_convert_f8_to_bf16_all_b(vc[i],0);
                v_bf16_st_tnsr(dst,key,kw.v1);v_bf16_st_tnsr(dst,value,vw.v1);dst[0]=128;
                v_bf16_st_tnsr(dst,key,kw.v2);v_bf16_st_tnsr(dst,value,vw.v2);
            }
            #pragma unroll
            for(int u=0;u<2;++u) {
                const int rank=first+k+u;
                if(rank>top) break;
                float64 m=-3.0e38f;
                #pragma unroll
                for(int o=0;o<4;++o) {
                    int5 sp={head,at[u*4+o],0,0,0},so={rank*4+o,head,row,0,0};
                    v_f32_st_tnsr_partial(so,key_out_scale,v_f32_ld_tnsr_b(sp,key_scale),0,0);
                    v_f32_st_tnsr_partial(so,value_out_scale,v_f32_ld_tnsr_b(sp,value_scale),0,0);
                    m=v_f32_sel_eq_i32_b(lane,(int64)o,o<count[u]?0.0f:-3.0e38f,m);
                }
                if(head==0) {int5 mp={rank*4,row,0,0,0};v_f32_st_tnsr_partial(mp,mask,m,3,0);}
            }
        }
    }
}
