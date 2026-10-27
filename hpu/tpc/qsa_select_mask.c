// STRATA_HPU_QSA_DEVICE_SELECT: QsaState::select for the rows of a window, at block granularity. Key positions are the
// cells (tpc/qsa_index_cells.c), block j holds cells [j * block, (j + 1) * block); `start` {2} F32 holds the window's
// first cell and this chunk's first window row -- runtime values, so one recipe serves every chunk of every window.
// Row r (cell start + first_row + r) sees size = that cell + 1 cells; up to `width_sel` cells it attends to all of
// them, past that to the `width_sel` cells of the highest block score (score desc, cell id asc), the row's own
// unfinished block forced first (+1e9) -- which is the host's block ranking expanded to cells and cut at width_sel,
// since a block's cells share its score and are taken in id order. A block's score is the sum over the indexer heads
// of max(q.k, 0) (F32; the host sums in double). The cut is found per row by a search on the scores' bits (all
// candidate scores are >= 0, so their bit patterns order like the values; cells counted as `block` per block, minus
// the forced block's missing cells) and, among blocks equal to the threshold, on the block id; both run a fixed
// number of 8-way passes in vector registers (every lane holds the same bounds), without a vector-to-scalar transfer.
// The result per block is how many of its cells (from its first) the row keeps, 0..block; tpc/qsa_select_expand.c
// turns that (broadcast to the block's positions) and the causal bound into the attention mask.
//   start {2} F32, scores {keys / block, heads, rows} F32 -> scratch {keys / block, rows} F32 (the summed scores, -1
//   where the block holds none of the row's cells) and kept {keys / block, rows} F32. keys / block % 256 == 0, block a
//   power of two. Members are rows.
// STRATA_SELECT_BITS (tpc/qsa_select_bits.c, the sparse prefill): the same selection written as bits for the host to
// list -- bits {keys / block / 16, rows} F32, word w holding blocks 16w..16w+15 kept (any cell) as 2^(b % 16), exact in
// F32 -- and info {2, rows} F32: the cut block (the one block kept partly, -1 for none) and its kept cells.
#ifdef STRATA_SELECT_BITS
// The kept flags of 64 blocks (FLAG > 0.5) as four 16-bit words at word V*4.
#define EMIT_BITS(V,FLAG) {     const float64 pw=v_convert_i32_to_f32_b(((int64)1)<<(lane&(int64)15),0);float64 w=0;     _Pragma("unroll") for(int k=0;k<4;++k) {         const float64 part=v_f32_reduce_add(v_f32_sel_eq_i32_b(lane>>4,(int64)k,v_f32_sel_grt_f32_b(FLAG,0.5f,pw,0.0f),0.0f));         w=v_f32_sel_eq_i32_b(lane,(int64)k,part,w);}     int5 o={(V)*4,r,0,0,0};v_f32_st_tnsr_partial(o,bits_out,w,3,0);}
void main(tensor start_cell,tensor scores,tensor scratch,tensor bits_out,tensor info,int keys,int rows,int window,int width_sel,int block,int heads) {
#else
void main(tensor start_cell,tensor scores,tensor scratch,tensor kept,int keys,int rows,int window,int width_sel,int block,int heads) {
#endif
    const int5 begin=get_index_space_offset(),end=begin+get_index_space_size();
    const int blocks=keys/block,vectors=blocks/64;
    const int64 lane=(int64)V_LANE_ID_32;
    int5 z={0,0,0,0,0};
    const int start=(int)s_f32_ld_g((__global__ float*)gen_addr(z,start_cell));
    z[0]=1;const int first_row=(int)s_f32_ld_g((__global__ float*)gen_addr(z,start_cell));
    // Bits of 1e9f: every probe at or below it counts the forced block (its score is >= 1e9).
    const int64 forced_bits=0x4E6E6B28;
    for(int r=begin[0];r<end[0] && r<rows;++r) {
        const int size=start+first_row+r+1;
        // The first cell of each block (exact in F32 below 2^24).
        #define FIRST(V) v_convert_i32_to_f32_b((lane+(V)*64)*(int64)block,0)
        if(size<=width_sel) {
#ifdef STRATA_SELECT_BITS
            for(int v=0;v<vectors;++v) EMIT_BITS(v,v_f32_sel_less_f32_b(FIRST(v),(float)size,1.0f,0.0f))
            {float64 none=v_f32_sel_eq_i32_b(lane,(int64)0,-1.0f,0.0f);int5 o={0,r,0,0,0};v_f32_st_tnsr_partial(o,info,none,1,0);}
#else
            for(int v=0;v<vectors;++v) {
                int5 o={v*64,r,0,0,0};v_f32_st_tnsr(o,kept,v_f32_sel_less_f32_b(FIRST(v),(float)size,(float)block,0.0f));
            }
#endif
            continue;
        }
        const int forced=size%block?size-size%block:size;
        // The forced block counts `block` cells in the block counts below but holds only size - forced of the row's.
        const int64 missing=size%block?(int64)(block-size%block):(int64)0;
        for(int v=0;v<vectors;++v) {
            float64 s=0;
            for(int h=0;h<heads;++h) {int5 p={v*64,h,r,0,0};s+=v_f32_max_b(v_f32_ld_tnsr_b(p,scores),0.0f);}
            const float64 first=FIRST(v);
            s=v_f32_sel_less_f32_b(first,(float)forced,s,s+1.0e9f);
            s=v_f32_sel_less_f32_b(first,(float)size,s,-1.0f);
            int5 o={v*64,r,0,0,0};v_f32_st_tnsr(o,scratch,s);
        }
        // The row's cells in blocks scoring >= PROBE, from the block count N.
        #define CELLS(N,PROBE) (v_f32_reduce_add(v_convert_i32_to_f32_b(N,0))*(float)block- \
            v_convert_i32_to_f32_b(v_i32_sel_less_i32_b(forced_bits,(PROBE),(int64)0,missing),0))
        // The largest bit pattern T with at least width_sel cells in blocks >= T: an 8-way search, seven probes per
        // pass over the row (11 passes cover [0, 0x7F800000]), four vectors a step so the loads overlap.
        int64 lo=0,hi=0x7F800000;
        for(int pass=0;pass<11;++pass) {
            const int64 span=hi-lo+1,stride=(span+7)>>3;
            int64 probe[7],count[7];
            #pragma unroll
            for(int k=0;k<7;++k) {probe[k]=lo+stride*(k+1);count[k]=0;}
            for(int v=0;v<vectors;v+=4) {
                #pragma unroll
                for(int u=0;u<4;++u) {
                    int5 p={(v+u)*64,r,0,0,0};const int64 bits=as_int64(v_f32_ld_tnsr_b(p,scratch));
                    #pragma unroll
                    for(int k=0;k<7;++k) count[k]+=v_i32_sel_less_i32_b(bits,probe[k],(int64)0,(int64)1);
                }
            }
            // The highest probe still holding width_sel cells raises lo; the lowest failing one caps hi.
            int64 next_lo=lo,next_hi=hi;
            #pragma unroll
            for(int k=0;k<7;++k) {
                const float64 total=CELLS(count[k],probe[k]);
                const int64 enough=v_convert_f32_to_i32_b(v_f32_sel_less_f32_b(total,(float)width_sel,0.0f,1.0f),0);
                const int64 valid=v_i32_sel_less_i32_b(hi,probe[k],(int64)0,(int64)1); // probe inside [lo, hi]
                next_lo=v_i32_sel_eq_i32_b(enough*valid,1,probe[k],next_lo);
                const int64 fail=v_i32_sel_eq_i32_b(enough,1,(int64)0,valid);
                next_hi=v_i32_sel_eq_i32_b(fail,1,v_i32_sel_less_i32_b(probe[k]-1,next_hi,probe[k]-1,next_hi),next_hi);
            }
            lo=next_lo;hi=next_hi;
        }
        const int64 threshold=lo;
        int64 above=0;
        for(int v=0;v<vectors;++v) {
            int5 p={v*64,r,0,0,0};const int64 bits=as_int64(v_f32_ld_tnsr_b(p,scratch));
            above+=v_i32_sel_less_i32_b(bits,threshold+1,(int64)0,(int64)1);
        }
        // Cells still to take from the threshold-equal blocks (full blocks: the forced one is above any threshold).
        const float64 need=(float)width_sel-CELLS(above,threshold+1);
        // The smallest block id I whose threshold-equal blocks at ids <= I hold `need` cells: an 8-way search over
        // [0, blocks-1] (6 passes cover 2^18 blocks).
        int64 ilo=0,ihi=blocks-1;
        for(int pass=0;pass<6;++pass) {
            const int64 span=ihi-ilo+1,stride=(span+7)>>3;
            int64 probe[7],count[7];
            #pragma unroll
            for(int k=0;k<7;++k) {probe[k]=ilo+stride*(k+1)-1;count[k]=0;}
            for(int v=0;v<vectors;v+=4) {
                #pragma unroll
                for(int u=0;u<4;++u) {
                    int5 p={(v+u)*64,r,0,0,0};const int64 bits=as_int64(v_f32_ld_tnsr_b(p,scratch));
                    const int64 ids=lane+(v+u)*64;
                    const int64 tie=v_i32_sel_eq_i32_b(bits,threshold,(int64)1,(int64)0);
                    #pragma unroll
                    for(int k=0;k<7;++k) count[k]+=v_i32_sel_less_i32_b(ids,probe[k]+1,tie,(int64)0);
                }
            }
            // The lowest probe already holding `need` cells caps ihi; the highest one short of it raises ilo.
            int64 next_lo=ilo,next_hi=ihi;
            #pragma unroll
            for(int k=6;k>=0;--k) {
                const float64 total=v_f32_reduce_add(v_convert_i32_to_f32_b(count[k],0))*(float)block;
                const int64 enough=v_convert_f32_to_i32_b(v_f32_sel_less_f32_b(total,need,0.0f,1.0f),0);
                const int64 valid=v_i32_sel_less_i32_b(ihi,probe[k],(int64)0,(int64)1);
                next_hi=v_i32_sel_eq_i32_b(enough*valid,1,probe[k],next_hi);
            }
            #pragma unroll
            for(int k=0;k<7;++k) {
                const float64 total=v_f32_reduce_add(v_convert_i32_to_f32_b(count[k],0))*(float)block;
                const int64 short_=v_convert_f32_to_i32_b(v_f32_sel_less_f32_b(total,need,1.0f,0.0f),0);
                const int64 valid=v_i32_sel_less_i32_b(ihi,probe[k],(int64)0,(int64)1);
                next_lo=v_i32_sel_eq_i32_b(short_*valid,1,probe[k]+1,next_lo);
            }
            ilo=next_lo;ihi=next_hi;
        }
        const int64 last_tie=ilo;
        // The cut block keeps `need` minus the cells of the threshold-equal blocks before it.
        int64 before=0;
        for(int v=0;v<vectors;++v) {
            int5 p={v*64,r,0,0,0};const int64 bits=as_int64(v_f32_ld_tnsr_b(p,scratch));
            const int64 ids=lane+v*64;
            before+=v_i32_sel_eq_i32_b(bits,threshold,v_i32_sel_less_i32_b(ids,last_tie,(int64)1,(int64)0),(int64)0);
        }
        const float64 rest=need-v_f32_reduce_add(v_convert_i32_to_f32_b(before,0))*(float)block;
        for(int v=0;v<vectors;++v) {
            int5 p={v*64,r,0,0,0};const int64 bits=as_int64(v_f32_ld_tnsr_b(p,scratch));
            const int64 ids=lane+v*64;
            const int64 tie=v_i32_sel_eq_i32_b(bits,threshold,(int64)1,(int64)0);
            // above the threshold: all; a tie before the cut: all; the cut block: rest; else none.
            const float64 tie_kept=v_f32_sel_less_f32_b(v_convert_i32_to_f32_b(ids,0),v_convert_i32_to_f32_b(last_tie,0),(float)block,
                                   v_f32_sel_eq_f32_b(v_convert_i32_to_f32_b(ids,0),v_convert_i32_to_f32_b(last_tie,0),rest,0.0f));
            const float64 n=v_f32_sel_eq_f32_b(v_convert_i32_to_f32_b(tie,0),1.0f,tie_kept,0.0f);
            const float64 out=v_f32_sel_less_i32_b(threshold,bits,(float)block,n);
#ifdef STRATA_SELECT_BITS
            EMIT_BITS(v,out)
#else
            int5 o={v*64,r,0,0,0};v_f32_st_tnsr(o,kept,out);
#endif
        }
#ifdef STRATA_SELECT_BITS
        {
            // The cut block keeps `rest` of its cells (a whole block when rest == block: then no block is partial).
            const float64 cut=v_f32_sel_less_f32_b(rest,(float)block,v_convert_i32_to_f32_b(last_tie,0),-1.0f);
            float64 pair=v_f32_sel_eq_i32_b(lane,(int64)0,cut,rest);int5 o={0,r,0,0,0};v_f32_st_tnsr_partial(o,info,pair,1,0);
        }
#endif
        #undef CELLS
    }
    #undef FIRST
}
