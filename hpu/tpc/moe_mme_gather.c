// STRATA_HPU_MOE_MME: the routed GEMM's A operand for tiles [first, first+count). Row r of local tile l is
// x[bucket[(first+l)*rows+r] / per_row] in BF16 for r < tile_rows[first+l]; later rows are padding and are not
// written (their GEMM rows are never read back).
// x {hidden, window rows} F32, bucket {tiles*rows} I32, tile_rows {tiles} I32 -> a {hidden, rows, count} BF16.
// Members are (8 tile rows, local tile). The member's eight bucket loads are issued together (each scalar global load
// waits a memory round trip), and a row's columns go 512 at a time so the vector loads overlap.
void main(tensor x,tensor bucket,tensor tile_rows,tensor a,int hidden,int rows,int per_row,int first) {
    const int5 begin=get_index_space_offset(),end=begin+get_index_space_size();
    const int wide=hidden-hidden%512;
    for(int l=begin[1];l<end[1];++l) {
        const int tile=first+l;
        int5 tp={tile,0,0,0,0};
        const int total=s_i32_ld_g((__global__ int*)gen_addr(tp,tile_rows));
        for(int chunk=begin[0];chunk<end[0];++chunk) {
            const int stop=8*chunk+8<total?8*chunk+8:total;
            if(8*chunk>=stop) continue;
            // Rows past `stop` read a bucket entry of the tile that is unused (clamped to the tile's last row).
            int source[8];
            #pragma unroll
            for(int i=0;i<8;++i) {
                int5 bp={tile*rows+(8*chunk+i<rows?8*chunk+i:rows-1),0,0,0,0};source[i]=s_i32_ld_g((__global__ int*)gen_addr(bp,bucket));
            }
            for(int i=0;i<8 && 8*chunk+i<stop;++i) {
                const int r=8*chunk+i,xrow=source[i]/per_row;
                int k=0;
                for(;k<wide;k+=512) {
                    int5 x0={k,xrow,0,0,0},x1={k+64,xrow,0,0,0},x2={k+128,xrow,0,0,0},x3={k+192,xrow,0,0,0};
                    int5 x4={k+256,xrow,0,0,0},x5={k+320,xrow,0,0,0},x6={k+384,xrow,0,0,0},x7={k+448,xrow,0,0,0};
                    float128 v0,v1,v2,v3;
                    v0.v1=v_f32_ld_tnsr_b(x0,x);v0.v2=v_f32_ld_tnsr_b(x1,x);v1.v1=v_f32_ld_tnsr_b(x2,x);v1.v2=v_f32_ld_tnsr_b(x3,x);
                    v2.v1=v_f32_ld_tnsr_b(x4,x);v2.v2=v_f32_ld_tnsr_b(x5,x);v3.v1=v_f32_ld_tnsr_b(x6,x);v3.v2=v_f32_ld_tnsr_b(x7,x);
                    int5 a0={k,r,l,0,0},a1={k+128,r,l,0,0},a2={k+256,r,l,0,0},a3={k+384,r,l,0,0};
                    v_bf16_st_tnsr(a0,a,v_convert_f32_to_bf16_all_b(v0,SW_LINEAR));v_bf16_st_tnsr(a1,a,v_convert_f32_to_bf16_all_b(v1,SW_LINEAR));
                    v_bf16_st_tnsr(a2,a,v_convert_f32_to_bf16_all_b(v2,SW_LINEAR));v_bf16_st_tnsr(a3,a,v_convert_f32_to_bf16_all_b(v3,SW_LINEAR));
                }
                for(;k<hidden;k+=128) {
                    int5 xp={k,xrow,0,0,0};
                    float128 v;v.v1=v_f32_ld_tnsr_b(xp,x);xp[0]+=64;v.v2=v_f32_ld_tnsr_b(xp,x);
                    int5 ap={k,r,l,0,0};
                    v_bf16_st_tnsr(ap,a,v_convert_f32_to_bf16_all_b(v,SW_LINEAR));
                }
            }
        }
    }
}
