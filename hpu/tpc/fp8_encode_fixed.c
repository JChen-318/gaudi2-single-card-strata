// Fixed MME bias11: raw FP8 bits encode x*16, no dynamic scale or high/low residual.
// Saturation is deliberate and must be checked by end-to-end fixed-input quality measurements.
void main(tensor input,tensor output,int columns,int rows) {
 const int5 begin=get_index_space_offset(),end=begin+get_index_space_size();
 const int chunks=(columns+255)/256;
 for(int member=begin[0];member<end[0];++member) {
  const int row=member/chunks,col=(member-row*chunks)*256;
  int5 p={col,row,0,0,0};float256 x;
  x.v1=v_f32_min_b(v_f32_max_b(v_f32_ld_tnsr_b(p,input)*16.0f,-240.0f),240.0f);p[0]+=64;
  x.v2=v_f32_min_b(v_f32_max_b(v_f32_ld_tnsr_b(p,input)*16.0f,-240.0f),240.0f);p[0]+=64;
  x.v3=v_f32_min_b(v_f32_max_b(v_f32_ld_tnsr_b(p,input)*16.0f,-240.0f),240.0f);p[0]+=64;
  x.v4=v_f32_min_b(v_f32_max_b(v_f32_ld_tnsr_b(p,input)*16.0f,-240.0f),240.0f);
  int5 o={col,row,0,0,0};const int count=columns-col<256?columns-col:256;
  v_f8_st_tnsr_partial(o,output,v_convert_f32_to_f8_all_b(x,SW_LINEAR),count-1,0);
 }
}
