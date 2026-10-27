// Virtual prefix for one canonical 32-row attention chunk in a larger window.
// Past rows retain their cache positions; prior rows of this window occupy the
// next positions; padding remains before this chunk, exactly as a standalone
// 32-row verification. Copies preserve every FP32 bit and cannot read future rows.
void main(tensor cache_k,tensor cache_v,tensor now_k,tensor now_v,tensor base_cell,tensor key,tensor value,
          int width,int heads,int cache_cells,int tokens,int first,int count,int bucket) {
    const int5 begin=get_index_space_offset(),end=begin+get_index_space_size();
    int5 zero={0,0,0,0,0};const int cell=(int)s_f32_ld_g((__global__ float*)gen_addr(zero,base_cell));
    const int keys=bucket+count;
    for(int member=begin[0];member<end[0];++member) {
        const int head=member/keys,item=member%keys;
        int source=-1;int K=0,V=1;
        if(cell>=0 && cell<=cache_cells) {
            if(item<cell) source=item;
            else if(item<cell+first) {source=item-cell;K=2;V=3;}
            else if(item>=bucket) {source=first+item-bucket;K=2;V=3;}
        }
        const bool valid=source>=0 && (K==0?source<cache_cells:source<tokens);
        for(int col=0;col<width;col+=64) {
            int5 src={col,head,valid?source:0,0,0},dst={col,item,head,0,0};float64 k=0,v=0;
            if(valid) {k=v_f32_ld_tnsr_b(src,K);v=v_f32_ld_tnsr_b(src,V);}
            v_f32_st_tnsr(dst,key,k);v_f32_st_tnsr(dst,value,v);
        }
    }
}
