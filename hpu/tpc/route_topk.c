// STRATA_HPU_RESIDENT_IQ4 device routing, one member: softmax over `experts` router logits, the `top_k` largest
// (ties to the lower id, the order route_reference sorts in), weights p_i / max(sum of selected p, 2^-14), and
// the shared expert gate appended as slot top_k. Selection compares logits, which order exactly like their
// probabilities unless two distinct logits round to one F32 probability on the host. Sums run in F32 here and
// in double on the host, so weights can differ in the last bits.
//   ids:     I32 {slots} (slot top_k reads the shared expert and holds 0);
//   weights: F32 {slots}.
// Up to 1024 experts, members own four rows (glue: ceil(rows / 4)): the rows' selection rounds are independent
// chains run side by side, each with the one-row operations in their order, so the same ids and weights.
__local__ float64 values[64];
#define TOPK_ROWS 4
void main(tensor logits,tensor shared_gate,tensor ids,tensor weights,int experts,int top_k,int slots,int rows) {
    const int5 begin=get_index_space_offset(),end=begin+get_index_space_size();
    const int vectors=experts/64;
    const float64 lane=v_convert_i32_to_f32_b((int64)V_LANE_ID_32,0);
    // Up to 24 rows (a decoded token, a DFlash verify window) take the one-row loop below, a row per member on its
    // own TPC: grouping four rows into one member would run them on a quarter of the cores (glue: rows <= 24).
    if(vectors*TOPK_ROWS<=64 && rows>24) {
        for(int group=begin[0];group<end[0] && group*TOPK_ROWS<rows;++group) {
            int row[TOPK_ROWS];float64 peak[TOPK_ROWS],total[TOPK_ROWS],chosen[TOPK_ROWS],exponent[TOPK_ROWS];
            #pragma unroll
            for(int i=0;i<TOPK_ROWS;++i) {
                row[i]=group*TOPK_ROWS+i<rows?group*TOPK_ROWS+i:rows-1;
                peak[i]=-3.0e38f;total[i]=0;chosen[i]=0;exponent[i]=0;
            }
            for(int v=0;v<vectors;++v) {
                #pragma unroll
                for(int i=0;i<TOPK_ROWS;++i) {
                    int5 p={v*64,row[i],0,0,0};values[i*vectors+v]=v_f32_ld_tnsr_b(p,logits);peak[i]=v_f32_max_b(peak[i],values[i*vectors+v]);
                }
            }
            #pragma unroll
            for(int i=0;i<TOPK_ROWS;++i) peak[i]=v_f32_reduce_max(peak[i]);
            for(int v=0;v<vectors;++v) {
                #pragma unroll
                for(int i=0;i<TOPK_ROWS;++i) total[i]+=v_exp_f32(values[i*vectors+v]-peak[i]);
            }
            #pragma unroll
            for(int i=0;i<TOPK_ROWS;++i) total[i]=v_f32_reduce_add(total[i]);
            for(int k=0;k<top_k;++k) {
                float64 best[TOPK_ROWS],first[TOPK_ROWS];
                #pragma unroll
                for(int i=0;i<TOPK_ROWS;++i) {best[i]=-3.0e38f;first[i]=1.0e9f;}
                for(int v=0;v<vectors;++v) {
                    #pragma unroll
                    for(int i=0;i<TOPK_ROWS;++i) best[i]=v_f32_max_b(best[i],values[i*vectors+v]);
                }
                #pragma unroll
                for(int i=0;i<TOPK_ROWS;++i) best[i]=v_f32_reduce_max(best[i]);
                for(int v=0;v<vectors;++v) {
                    #pragma unroll
                    for(int i=0;i<TOPK_ROWS;++i)
                        first[i]=v_f32_min_b(first[i],v_f32_sel_eq_f32_b(values[i*vectors+v],best[i],lane+(float)(64*v),1.0e9f));
                }
                #pragma unroll
                for(int i=0;i<TOPK_ROWS;++i) {
                    first[i]=v_f32_reduce_min(first[i]);
                    first[i]=v_f32_sel_grt_f32_b(first[i],(float)(experts-1),0.0f,first[i]); // never an address outside the layer
                }
                for(int v=0;v<vectors;++v) {
                    #pragma unroll
                    for(int i=0;i<TOPK_ROWS;++i)
                        values[i*vectors+v]=v_f32_sel_eq_f32_b(lane+(float)(64*v),first[i],-3.0e38f,values[i*vectors+v]);
                }
                #pragma unroll
                for(int i=0;i<TOPK_ROWS;++i) {
                    chosen[i]=v_f32_sel_eq_u32_b(V_LANE_ID_32,(unsigned)k,first[i],chosen[i]);
                    exponent[i]=v_f32_sel_eq_u32_b(V_LANE_ID_32,(unsigned)k,v_exp_f32(best[i]-peak[i]),exponent[i]);
                }
            }
            #pragma unroll
            for(int i=0;i<TOPK_ROWS;++i) {
                if(group*TOPK_ROWS+i>=rows) continue;
                const float64 probability=exponent[i]/total[i];
                const float64 selected=v_f32_reduce_add(probability);
                const float64 scale=v_f32_max_b(selected,6.103515625e-05f);
                float64 weight=probability/scale;
                int5 p={0,row[i],0,0,0};
                const float64 gate=v_f32_reduce_add(v_f32_sel_eq_u32_b(V_LANE_ID_32,0u,v_f32_ld_tnsr_b(p,shared_gate),0.0f));
                p[0]=row[i]*slots;p[1]=0;
                weight=v_f32_sel_eq_u32_b(V_LANE_ID_32,(unsigned)top_k,gate,weight);
                v_i32_st_tnsr_partial(p,ids,v_convert_f32_to_i32_b(chosen[i],0),slots-1,0);
                v_f32_st_tnsr_partial(p,weights,weight,slots-1,0);
            }
        }
        return;
    }
    for(int row=begin[0];row<end[0];++row) {
    int5 p={0,row,0,0,0};
    float64 peak=-3.0e38f;
    for(int v=0;v<vectors;++v) {p[0]=v*64;values[v]=v_f32_ld_tnsr_b(p,logits);peak=v_f32_max_b(peak,values[v]);}
    peak=v_f32_reduce_max(peak);
    float64 total=0;
    for(int v=0;v<vectors;++v) total+=v_exp_f32(values[v]-peak);
    total=v_f32_reduce_add(total);
    float64 chosen=0,exponent=0;
    for(int k=0;k<top_k;++k) {
        float64 best=-3.0e38f;
        for(int v=0;v<vectors;++v) best=v_f32_max_b(best,values[v]);
        best=v_f32_reduce_max(best);
        float64 first=1.0e9f;
        for(int v=0;v<vectors;++v) first=v_f32_min_b(first,v_f32_sel_eq_f32_b(values[v],best,lane+(float)(64*v),1.0e9f));
        first=v_f32_reduce_min(first);
        first=v_f32_sel_grt_f32_b(first,(float)(experts-1),0.0f,first); // never an address outside the layer
        for(int v=0;v<vectors;++v) values[v]=v_f32_sel_eq_f32_b(lane+(float)(64*v),first,-3.0e38f,values[v]);
        chosen=v_f32_sel_eq_u32_b(V_LANE_ID_32,(unsigned)k,first,chosen);
        exponent=v_f32_sel_eq_u32_b(V_LANE_ID_32,(unsigned)k,v_exp_f32(best-peak),exponent);
    }
    const float64 probability=exponent/total;
    const float64 selected=v_f32_reduce_add(probability);
    const float64 scale=v_f32_max_b(selected,6.103515625e-05f);
    float64 weight=probability/scale;
    p[0]=0;
    const float64 gate=v_f32_reduce_add(v_f32_sel_eq_u32_b(V_LANE_ID_32,0u,v_f32_ld_tnsr_b(p,shared_gate),0.0f));
    p[0]=row*slots;p[1]=0;
    weight=v_f32_sel_eq_u32_b(V_LANE_ID_32,(unsigned)top_k,gate,weight);
    v_i32_st_tnsr_partial(p,ids,v_convert_f32_to_i32_b(chosen,0),slots-1,0);
    v_f32_st_tnsr_partial(p,weights,weight,slots-1,0);
    }
}
