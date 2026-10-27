// Copy host-selected cells, preserving their order and every FP32 bit. Source
// cache is {width,heads,cells}; output merges row/head as attention's batch axis.
// Invalid/padded ids produce zero K/V and -inf mask, never an out-of-range load.
void main(tensor keys,tensor values,tensor ids,tensor key,tensor value,tensor mask,
          int width,int heads,int cells,int bucket,int rows,int mask_heads) {
    const int5 begin=get_index_space_offset(),end=begin+get_index_space_size();
    (void)rows;
    for(int member=begin[0];member<end[0];++member) {
        const int item=member%bucket,batch=member/bucket,head=batch%heads,row=batch/heads;
        int5 ip={item,row,0,0,0};
        const int id=(int)s_f32_ld_g((__global__ float*)gen_addr(ip,ids));
        const bool valid=id>=0 && id<cells;
        for(int col=0;col<width;col+=64) {
            int5 src={col,head,valid?id:0,0,0},dst={col,item,batch,0,0};
            float64 k=0,v=0;
            if(valid) {k=v_f32_ld_tnsr_b(src,keys);v=v_f32_ld_tnsr_b(src,values);}
            v_f32_st_tnsr(dst,key,k);v_f32_st_tnsr(dst,value,v);
        }
        if(head<mask_heads) {
            int5 mp={item,0,mask_heads==1?row:batch,0,0};
            // -inf is exact IEEE754; no exp/normalization or K/V arithmetic here.
            const float64 m=valid?0.0f:(float64)((uint64)0xff800000u);
            v_f32_st_tnsr_partial(mp,mask,m,0,0);
        }
    }
}
