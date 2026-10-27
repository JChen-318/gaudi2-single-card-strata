// Sparse QSA prefill: every window row's selected blocks gathered from the FP8 cache, the block ids listed per row by
// the host from tpc/qsa_select_bits.c's bits (ids {top, rows} I32, -1 = no block). Rank k of row r becomes cells
// k*block.. of that row's keys. A block keeps all its cells but the row's cut block (info {2, rows}: its id and kept
// cells); a cell not kept, past the row's position (start[0] + start[1] + r: the row's future) or past the cache gets
// zero codes, a zero scale and a -3e38 mask.
// A member owns 16 ranks of one row and head: the 16 block ids are loaded first so their scalar-load latencies
// overlap, then each block's cells are copied (a dropped cell reads past the cache, where a tensor load gives zeros).
// The codes are written widened to BF16 (exact) in the TPC's native conversion order: a cell's 256 values as its even
// then its odd channels (c[0], c[2], .., c[254], c[1], c[3], ..); the attention permutes the query to match and its
// context back. The per-cell scales stay separate (the attention applies them to the scores and probabilities).
//   keys, values {256, heads, cells} FP8; scales {32, cells} F32 x2; ids {top, rows} I32; info {2, rows} F32;
//   start {2} F32 (the window's first cell, the chunk's first row)
//   -> key, value {256, top*block, heads, rows} BF16 (even/odd order); scales {top*block, heads, rows} F32 x2;
//   mask {top*block, rows} F32.
//   Members: (16-rank group, head, row). block == 4.
void main(tensor keys,tensor values,tensor key_scale,tensor value_scale,tensor ids,tensor info,tensor start_t,
          tensor key,tensor value,tensor key_out_scale,tensor value_out_scale,tensor mask,
          int heads,int cells,int top,int block,int rows) {
    const int5 begin=get_index_space_offset(),end=begin+get_index_space_size();
    (void)heads;(void)rows;(void)block;
    int5 z={0,0,0,0,0};
    const int cell=(int)s_f32_ld_g((__global__ float*)gen_addr(z,start_t));
    z[0]=1;const int first_row=(int)s_f32_ld_g((__global__ float*)gen_addr(z,start_t));
    const int64 lane=(int64)V_LANE_ID_32;
    for(int row=begin[2];row<end[2];++row)
    for(int head=begin[1];head<end[1];++head)
    for(int group=begin[0];group<end[0];++group) {
        const int first=group*16,position=cell+first_row+row;
        int5 cp={0,row,0,0,0};const int cut=(int)s_f32_ld_g((__global__ float*)gen_addr(cp,info));
        cp[0]=1;const int rest=(int)s_f32_ld_g((__global__ float*)gen_addr(cp,info));
        int id[16];
        #pragma unroll
        for(int k=0;k<16;++k) {
            int5 ip={first+k<top?first+k:top-1,row,0,0,0};
            id[k]=s_i32_ld_g((__global__ int*)gen_addr(ip,ids));
            if(first+k>=top) id[k]=-1;
        }
        #pragma unroll
        for(int k=0;k<16;++k) {
            const int rank=first+k;
            if(rank>=top) break;
            float64 m=-3.0e38f;
            const int count=id[k]==cut?rest:4;
            #pragma unroll
            for(int o=0;o<4;++o) {
                const int c=id[k]*4+o;
                const bool keep=id[k]>=0 && o<count && c<=position && c<cells;
                const int at=keep?c:cells;
                int5 src={0,head,at,0,0},sp={head,at,0,0,0},dst={0,rank*4+o,head,row,0},so={rank*4+o,head,row,0,0};
                const bfloat256 kw=v_convert_f8_to_bf16_all_b(v_f8_ld_tnsr_b(src,keys),0),vw=v_convert_f8_to_bf16_all_b(v_f8_ld_tnsr_b(src,values),0);
                v_bf16_st_tnsr(dst,key,kw.v1);v_bf16_st_tnsr(dst,value,vw.v1);dst[0]=128;
                v_bf16_st_tnsr(dst,key,kw.v2);v_bf16_st_tnsr(dst,value,vw.v2);
                v_f32_st_tnsr_partial(so,key_out_scale,v_f32_ld_tnsr_b(sp,key_scale),0,0);
                v_f32_st_tnsr_partial(so,value_out_scale,v_f32_ld_tnsr_b(sp,value_scale),0,0);
                m=v_f32_sel_eq_i32_b(lane,(int64)o,keep?0.0f:-3.0e38f,m);
            }
            if(head==0) {int5 mp={rank*4,row,0,0,0};v_f32_st_tnsr_partial(mp,mask,m,3,0);}
        }
    }
}
