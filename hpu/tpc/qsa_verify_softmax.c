// STRATA_HPU_QSA_VERIFY_BF16: a verify window's attention probabilities in two kernels (this file, compiled twice:
// -DSTRATA_SOFTMAX_PARTIALS for the first), replacing the key-scale multiply, the score concat, the mask add (each a
// broadcast the graph compiler tiles), the softmax, the split and the value-scale multiply with its BF16 cast.
// Per score row b (window row b / heads, KV head) and query head g of the group:
//   s_w = scores_w + mask[0 .. rows)                            (the window's own rows, causal),
//   s_c = scores_c * key_scale + mask[rows .. rows + cells)     (the gathered cells),
//   p = softmax over [s_w, s_c];  p_w = p over the window rows (F32);  p16 = BF16(p_c * value_scale).
// Members own 256 cells of one score row (chunk 0 also the window's rows), so a row's work spreads over the TPCs:
// the first kernel writes each chunk's maximum and sum of exp(s - maximum) (parts {chunks, 2, group, B}: lane j of
// a row's two vectors is chunk j, chunks <= 64), the second combines a row's chunks and writes the probabilities.
//   scores_w {rows, group, B} F32; scores_c {cells, group, B} F32; key_scale, value_scale {cells, B} F32;
//   mask {rows + cells, rows} F32 -> p_w {rows, group, B} F32; p16 {cells, group, B} BF16, p_f32 the same unrounded. B = rows * heads,
//   rows <= 64. Members: (chunk, query head of the group, b).
#define LOW (-3.0e38f)
// The chunk's four score vectors, loads issued together; lanes past `cells` at LOW.
#define CHUNK_SCORES \
    float64 s[4]; \
    { \
        float64 sc[4],ks[4],ms[4]; \
        _Pragma("unroll") for(int i=0;i<4;++i) { \
            int5 sp={c0+64*i,g,b,0,0},kp={c0+64*i,b,0,0,0},cm={rows+c0+64*i,row,0,0,0}; \
            sc[i]=v_f32_ld_tnsr_b(sp,scores_c);ks[i]=v_f32_ld_tnsr_b(kp,key_scale);ms[i]=v_f32_ld_tnsr_b(cm,mask); \
        } \
        _Pragma("unroll") for(int i=0;i<4;++i) s[i]=v_f32_sel_less_i32_b(lane,(int64)(cells-c0-64*i),sc[i]*ks[i]+ms[i],LOW); \
    }
#ifdef STRATA_SOFTMAX_PARTIALS
void main(tensor scores_w,tensor scores_c,tensor key_scale,tensor mask,tensor parts,int rows,int heads,int group,int cells) {
#else
void main(tensor scores_w,tensor scores_c,tensor key_scale,tensor value_scale,tensor mask,tensor parts,tensor p_w,tensor p16,
          tensor p_f32,int rows,int heads,int group,int cells) {
#endif
    const int5 begin=get_index_space_offset(),end=begin+get_index_space_size();
    (void)group;
    const int64 lane=(int64)V_LANE_ID_32;
    const int chunks=(cells+255)/256;
    for(int b=begin[2];b<end[2];++b)
    for(int g=begin[1];g<end[1];++g)
    for(int k=begin[0];k<end[0];++k) {
        const int row=b/heads,c0=k*256;
        int5 wp={0,g,b,0,0},mp={0,row,0,0,0};
        const float64 w=v_f32_sel_less_i32_b(lane,(int64)rows,v_f32_ld_tnsr_b(wp,scores_w)+v_f32_ld_tnsr_b(mp,mask),LOW);
        CHUNK_SCORES
#ifdef STRATA_SOFTMAX_PARTIALS
        float64 peak=v_f32_max_b(v_f32_max_b(s[0],s[1]),v_f32_max_b(s[2],s[3]));
        if(k==0) peak=v_f32_max_b(peak,w);
        const float64 m=v_f32_reduce_max(peak);
        float64 sum=v_exp_f32(s[0]-m)+v_exp_f32(s[1]-m)+v_exp_f32(s[2]-m)+v_exp_f32(s[3]-m);
        if(k==0) sum+=v_exp_f32(w-m);
        int5 op={k,0,g,b,0};v_f32_st_tnsr_partial(op,parts,m,0,0);op[1]=1;v_f32_st_tnsr_partial(op,parts,v_f32_reduce_add(sum),0,0);
#else
        // The row's maximum and sum from its chunks' (lane j: chunk j; each chunk's sum is relative to its own maximum).
        int5 pp={0,0,g,b,0};const float64 maxes=v_f32_sel_less_i32_b(lane,(int64)chunks,v_f32_ld_tnsr_b(pp,parts),LOW);
        pp[1]=1;const float64 sums=v_f32_sel_less_i32_b(lane,(int64)chunks,v_f32_ld_tnsr_b(pp,parts),0.0f);
        const float64 m=v_f32_reduce_max(maxes),sum=v_f32_reduce_add(sums*v_exp_f32(maxes-m));
        const float64 inverse=1.0f/sum;
        if(k==0) v_f32_st_tnsr_partial(wp,p_w,v_exp_f32(w-m)*inverse,rows-1,0);
        float64 vs[4];
        _Pragma("unroll") for(int i=0;i<4;++i) {int5 kp={c0+64*i,b,0,0,0};vs[i]=v_f32_ld_tnsr_b(kp,value_scale);}
        // Lanes past `cells` land past the tensor's end (dropped stores).
        _Pragma("unroll") for(int i=0;i<4;i+=2) {
            float128 wide;wide.v1=v_exp_f32(s[i]-m)*inverse*vs[i];wide.v2=v_exp_f32(s[i+1]-m)*inverse*vs[i+1];
            int5 op={c0+64*i,g,b,0,0};v_bf16_st_tnsr(op,p16,v_convert_f32_to_bf16_all_b(wide,SW_LINEAR));
            // (p_f32: the same products unrounded, for STRATA_HPU_QSA_VERIFY_PLO's low part.)
            v_f32_st_tnsr(op,p_f32,wide.v1);op[0]+=64;v_f32_st_tnsr(op,p_f32,wide.v2);
        }
#endif
    }
}
