// U4 gate/up affine epilogue. One member owns a complete down K-group
// and up to eight rows, so each activation sum is reduced once for all
// down output channels. Invalid bucket rows never read the GEMM output.
#include "gdn_math.h"
__local__ float64 gate[8],lift[8],totals[8];
void main(tensor out_gu,tensor scale,tensor offset,tensor map,tensor sumx,tensor a_down,tensor partial,
          int experts,int groups,int ff,int rows,int width,int down_groups) {
    const int5 begin=get_index_space_offset(),end=begin+get_index_space_size();
    (void)experts;
    const int chunks=(rows+7)/8,parts=width/64;
    for(int member=begin[0];member<end[0];++member) {
        const int e=member/(down_groups*chunks),rest=member-e*down_groups*chunks;
        const int dg=rest/chunks,chunk=rest-dg*chunks,bd=e*down_groups+dg;
        const int first=chunk*8,count=rows-first<8?rows-first:8;
        int valid[8],active=0;
        for(int i=0;i<8;++i) {
            valid[i]=0;totals[i]=0;
            if(i>=count) continue;
            int5 mp={e*rows+first+i,0,0,0,0};
            valid[i]=s_i32_ld_g((__global__ int*)gen_addr(mp,map))>=0;
            active+=valid[i];
        }
        for(int part=0;part<parts;++part) {
            const int column=dg*width+part*64;
            for(int i=0;i<8;++i) {gate[i]=0;lift[i]=0;}
            for(int g=0;g<groups && active;++g) {
                const int b=e*groups+g;
                int5 np={column,b,0,0,0};
                const float64 s=v_f32_ld_tnsr_b(np,scale),o=v_f32_ld_tnsr_b(np,offset);
                np[0]=ff+column;
                const float64 su=v_f32_ld_tnsr_b(np,scale),ou=v_f32_ld_tnsr_b(np,offset);
                for(int i=0;i<count;++i) {
                    if(!valid[i]) continue;
                    int5 qp={b*rows+first+i,0,0,0,0};
                    const float64 sum=(float64)s_f32_ld_g((__global__ float*)gen_addr(qp,sumx));
                    int5 gp={column,first+i,b,0,0};
                    gate[i]+=v_f32_ld_tnsr_b(gp,out_gu)*s+o*sum;
                    gp[0]=ff+column;
                    lift[i]+=v_f32_ld_tnsr_b(gp,out_gu)*su+ou*sum;
                }
            }
            for(int i=0;i<count;++i) {
                int5 ap={part*64,first+i,bd,0,0};
                if(!valid[i]) {
                    const bfloat128 zeros={0};v_bf16_st_tnsr_partial(ap,a_down,zeros,63,0);
                    continue;
                }
                const float64 value=(gate[i]*strata_sigmoid(gate[i]))*lift[i];
                float128 wide;wide.v1=value;wide.v2=0;
                const bfloat128 rounded=v_convert_f32_to_bf16_all_b(wide,SW_LINEAR);
                const float128 restored=convert_bfloat128_to_float128(rounded,0);
                v_bf16_st_tnsr_partial(ap,a_down,rounded,63,0);
                totals[i]+=v_f32_reduce_add(restored.v1+restored.v2);
            }
        }
        for(int i=0;i<count;++i) {
            int5 pp={bd*rows+first+i,0,0,0,0};
            v_f32_st_tnsr_partial(pp,partial,totals[i],0,0);
        }
    }
}
