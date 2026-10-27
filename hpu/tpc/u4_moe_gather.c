// STRATA_HPU_RESIDENT_U4: the grouped MME's activation. b = e*groups+g is one batch member; its row m holds
// x[g*group .. g*group+group) of the slot the map gives (the routed bucket, or the shared expert's static
// row-order map). A row of a padding entry (map < 0) is zeroed, and so is its sum.
// a is BF16 {batch, rows, group} - the MME's A. sumx is F32 {batch*rows}: the affine epilogue multiplies each
// channel's offset by it, because w = scale*code + offset and the MME only returns sum_k code*x.
void main(tensor x,tensor map,tensor a,tensor sumx,int groups,int group,int rows,int per_row) {
    const int5 begin=get_index_space_offset(),end=begin+get_index_space_size();
    for(int b=begin[0];b<end[0];++b) {
        const int e=b/groups,g=b-groups*e;
        for(int m=0;m<rows;++m) {
            int5 mp={e*rows+m,0,0,0,0};
            const int slot=s_i32_ld_g((__global__ int*)gen_addr(mp,map));
            int5 sp={b*rows+m,0,0,0,0};
            if(slot<0) {
                const bfloat128 zeros={0};
                for(int k=0;k<group;k+=64) {
                    int5 ap={k,m,b,0,0};
                    v_bf16_st_tnsr_partial(ap,a,zeros,63,0);
                }
                v_f32_st_tnsr_partial(sp,sumx,(float64)0.0f,0,0);
                continue;
            }
            const int xrow=slot/per_row;
            float64 total=0;
            for(int k=0;k<group;k+=64) {
                int5 xp={g*group+k,xrow,0,0,0};
                const float64 value=v_f32_ld_tnsr_b(xp,x);
                float128 wide;wide.v1=value;wide.v2=0;
                const bfloat128 rounded=v_convert_f32_to_bf16_all_b(wide,SW_LINEAR);
                const float128 restored=convert_bfloat128_to_float128(rounded,0);
                // The high-level conversion distributes the linear BF16 lanes
                // across both F32 vectors; wide.v2 contributed only zeros.
                total+=restored.v1+restored.v2;
                int5 ap={k,m,b,0,0};
                v_bf16_st_tnsr_partial(ap,a,rounded,63,0);
            }
            v_f32_st_tnsr_partial(sp,sumx,v_f32_reduce_add(total),0,0);
        }
    }
}
