// U4 down affine epilogue. Bucket entries are a dense valid prefix
// followed by -1 padding (the shared map is entirely valid). A member
// owns an expert and 64 output channels, stopping at the first pad row.
// The GU epilogue already reduced the sum of each down K-group.
__local__ float64 value[8];
void main(tensor out_down,tensor scale,tensor offset,tensor map,tensor partial,tensor route,tensor expert_rows,
          int experts,int groups,int n,int rows,int width,int apply_route) {
    const int5 begin=get_index_space_offset(),end=begin+get_index_space_size();
    (void)experts;(void)width;
    const int blocks=n/64;
    for(int member=begin[0];member<end[0];++member) {
        const int e=member/blocks,block=member-e*blocks;
        for(int first=0;first<rows;first+=8) {
            const int count=rows-first<8?rows-first:8;
            int slot[8],active=0;
            for(int i=0;i<8;++i) {slot[i]=-1;value[i]=0;}
            for(int i=0;i<count;++i) {
                int5 mp={e*rows+first+i,0,0,0,0};
                slot[i]=s_i32_ld_g((__global__ int*)gen_addr(mp,map));
                if(slot[i]<0) break;
                ++active;
            }
            if(!active) break;
            for(int g=0;g<groups;++g) {
                const int b=e*groups+g;
                int5 np={block*64,b,0,0,0};
                const float64 s=v_f32_ld_tnsr_b(np,scale),o=v_f32_ld_tnsr_b(np,offset);
                for(int i=0;i<active;++i) {
                    int5 qp={b*rows+first+i,0,0,0,0};
                    const float sum=s_f32_ld_g((__global__ float*)gen_addr(qp,partial));
                    int5 dp={block*64,first+i,b,0,0};
                    value[i]+=v_f32_ld_tnsr_b(dp,out_down)*s+o*(float64)sum;
                }
            }
            for(int i=0;i<active;++i) {
                int5 rp={slot[i],0,0,0,0};
                const float64 weight=apply_route?(float64)s_f32_ld_g((__global__ float*)gen_addr(rp,route)):1.0f;
                int5 ep={block*64,slot[i],0,0,0};v_f32_st_tnsr(ep,expert_rows,value[i]*weight);
            }
            if(active<count) break;
        }
    }
}
