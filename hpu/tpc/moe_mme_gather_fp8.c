// STRATA_HPU_MOE_FP8: moe_mme_gather.c into FP8 E4M3 with one scale per row: a[r] = fp8(x / scale_r), scale_r =
// max|x| / 240. x {hidden, window rows} F32, bucket {tiles*rows} I32, tile_rows {tiles} I32 -> a {hidden, rows, count}
// F8 and a_scale {64, rows, count} F32 (the row's scale in all 64 lanes, so readers take it with one vector load).
// Padding rows are not written. Members are (8 tile rows, local tile): the 8 rows' bucket entries are loaded first,
// and each row is read from HBM once into VLM (hidden <= 4096) for its maximum, then scaled from there.
__local__ float64 cached[64];
void main(tensor x,tensor bucket,tensor tile_rows,tensor a,tensor a_scale,int hidden,int rows,int per_row,int first) {
    const int5 begin=get_index_space_offset(),end=begin+get_index_space_size();
    for(int l=begin[1];l<end[1];++l) {
        const int tile=first+l;
        int5 tp={tile,0,0,0,0};
        const int total=s_i32_ld_g((__global__ int*)gen_addr(tp,tile_rows));
        for(int chunk=begin[0];chunk<end[0];++chunk) {
            const int stop=8*chunk+8<total?8*chunk+8:total;
            int xrows[8];
            #pragma unroll
            for(int i=0;i<8;++i) {
                const int r=8*chunk+i<stop?8*chunk+i:8*chunk;
                int5 bp={tile*rows+r,0,0,0,0};
                xrows[i]=s_i32_ld_g((__global__ int*)gen_addr(bp,bucket))/per_row;
            }
            for(int r=8*chunk;r<stop;++r) {
                const int xrow=xrows[r-8*chunk];
                float64 peak=0;
                // Fixed trip counts (hidden <= 4096) so both passes unroll into straight-line code.
                #pragma unroll
                for(int k=0;k<4096;k+=256) {
                    if(k>=hidden) break;
                    int5 xp={k,xrow,0,0,0};
                    float64 v[4];
                    #pragma unroll
                    for(int i=0;i<4;++i) {v[i]=v_f32_ld_tnsr_b(xp,x);xp[0]+=64;cached[k/64+i]=v[i];}
                    #pragma unroll
                    for(int i=0;i<4;++i) peak=v_f32_max_b(peak,v_f32_sel_less_f32_b(v[i],0.0f,-v[i],v[i]));
                }
                const float64 scale=v_f32_reduce_max(peak)*(1.0f/240.0f)+1e-30f,inverse=1.0f/scale;
                int5 sp={0,r,l,0,0};v_f32_st_tnsr(sp,a_scale,scale);
                #pragma unroll
                for(int k=0;k<4096;k+=256) {
                    if(k>=hidden) break;
                    float256 v;
                    v.v1=cached[k/64]*inverse;v.v2=cached[k/64+1]*inverse;v.v3=cached[k/64+2]*inverse;v.v4=cached[k/64+3]*inverse;
                    int5 ap={k,r,l,0,0};
                    v_f8_st_tnsr(ap,a,v_convert_f32_to_f8_all_b(v,SW_LINEAR));
                }
            }
        }
    }
}
