// Own a complete output row: load its scale once, then stream all
// columns. A member per 64 columns repeats scalar addressing millions
// of times at production batch sizes.
void main(tensor both,tensor scale,tensor out,int columns,int rows,int terms) {
    const int5 begin=get_index_space_offset(),end=begin+get_index_space_size();
    for(int batch=begin[0];batch<end[0];++batch) {
      for(int row=0;row<rows;++row) {
        int5 sc={0,row,batch,0,0};const float xs=s_f32_ld_g((__global__ float*)gen_addr(sc,scale));
        for(int n=0;n<columns;n+=64) {
            int5 hp={n,row,batch,0,0};float64 value=v_f32_ld_tnsr_b(hp,both);
            if(terms==2) {int5 lp={n,rows+row,batch,0,0};value+=v_f32_ld_tnsr_b(lp,both)*(1.0f/16.0f);}
            const int count=columns-n<64?columns-n:64;v_f32_st_tnsr_partial(hp,out,value*xs,count-1,0);
        }
      }
    }
}
