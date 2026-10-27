// STRATA_HPU_HC_FUSED: hc_mix_window.c that also writes the mix rounded to BF16 (the SDK cast that followed it).
#include "gdn_math.h"
void main(tensor normalized,tensor gate,tensor output,tensor rounded,int width,int streams,int rows) {
    const int5 start=get_index_space_offset(),end=start+get_index_space_size();
    // Members own 256 columns of one token: two 128-column tiles (two 64-lane halves each; `half` is a TPC type name,
    // so the BF16 copy is one full 128-lane store), unrolled so the tiles' loads and sigmoids overlap.
    for(int token=start[1];token<end[1] && token<rows;++token)
    for(int pair=start[0];pair<end[0] && pair*256<width;++pair)
    #pragma unroll
    for(int sub=0;sub<2;++sub) {
        const int tile=pair*2+sub;
        if(tile*128>=width) break;
        float128 wide;wide.v2=0;
        const int halves=tile*128+64<width?2:1;
        if(streams==4) {
            // The model's four streams in straight-line code: all sixteen loads and eight sigmoids are independent,
            // so the compiler overlaps them. Each half sums its streams in order, as the loop below.
            float64 n[2][4],g[2][4];
            #pragma unroll
            for(int hb=0;hb<2;++hb)
            #pragma unroll
            for(int stream=0;stream<4;++stream) {
                int5 p={tile*128+(hb<halves?hb:0)*64,stream,token,0,0};
                n[hb][stream]=v_f32_ld_tnsr_b(p,normalized);g[hb][stream]=v_f32_ld_tnsr_b(p,gate);
            }
            #pragma unroll
            for(int hb=0;hb<2;++hb) {
                float64 sum=0;
                #pragma unroll
                for(int stream=0;stream<4;++stream) sum+=n[hb][stream]*strata_sigmoid(g[hb][stream]);
                const float64 y=sum*(1.0f/streams);
                if(hb<halves) {int5 p={tile*128+hb*64,token,0,0,0};v_f32_st_tnsr(p,output,y);}
                if(hb) wide.v2=halves>1?y:0; else wide.v1=y;
            }
        } else
        for(int hb=0;hb<halves;++hb) {
            float64 sum=0;
            for(int stream=0;stream<streams;++stream) {
                int5 p={tile*128+hb*64,stream,token,0,0};
                sum+=v_f32_ld_tnsr_b(p,normalized)*strata_sigmoid(v_f32_ld_tnsr_b(p,gate));
            }
            int5 p={tile*128+hb*64,token,0,0,0};
            const float64 y=sum*(1.0f/streams);v_f32_st_tnsr(p,output,y);
            if(hb) wide.v2=y; else wide.v1=y;
        }
        int5 p={tile*128,token,0,0,0};
        if(halves>1) v_bf16_st_tnsr(p,rounded,v_convert_f32_to_bf16_all_b(wide,SW_LINEAR));
        else v_bf16_st_tnsr_partial(p,rounded,v_convert_f32_to_bf16_all_b(wide,SW_LINEAR),63,0);
    }
}
