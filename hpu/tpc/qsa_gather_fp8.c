// STRATA_HPU_QSA_KV_FP8: tpc/qsa_gather.c for an FP8 cache. Copies the selected cells' FP8 code rows byte for byte
// (one 256-byte vector per head row: width == 256) and their per-(cell, head) F32 scales; the graph dequantizes
// (cast to F32 times the scale), so no FP8 conversion runs here. Invalid/padded ids produce zero codes, a zero scale
// and a -inf mask, never an out-of-range load.
//   keys, values {width, heads, cells} FP8; key_scale, value_scale {32, cells} F32 (slot = head); ids {bucket, rows} F32 ->
//   key, value {width, bucket, rows*heads} FP8; key/value scales {bucket, rows*heads} F32; mask as qsa_gather.c.
void main(tensor keys,tensor values,tensor key_scale,tensor value_scale,tensor ids,
          tensor key,tensor value,tensor key_out_scale,tensor value_out_scale,tensor mask,
          int width,int heads,int cells,int bucket,int rows,int mask_heads) {
    const int5 begin=get_index_space_offset(),end=begin+get_index_space_size();
    (void)rows;(void)width;
    for(int member=begin[0];member<end[0];++member) {
        const int item=member%bucket,batch=member/bucket,head=batch%heads,row=batch/heads;
        int5 ip={item,row,0,0,0};
        const int id=(int)s_f32_ld_g((__global__ float*)gen_addr(ip,ids));
        const bool valid=id>=0 && id<cells;
        int5 src={0,head,valid?id:0,0,0},dst={0,item,batch,0,0};
        uchar256 k=0,v=0;
        if(valid) {k=v_u8_ld_tnsr_b(src,keys);v=v_u8_ld_tnsr_b(src,values);}
        v_u8_st_tnsr(dst,key,k);v_u8_st_tnsr(dst,value,v);
        int5 sp={head,valid?id:0,0,0,0};
        const float ks=valid?s_f32_ld_g((__global__ float*)gen_addr(sp,key_scale)):0.0f;
        const float vs=valid?s_f32_ld_g((__global__ float*)gen_addr(sp,value_scale)):0.0f;
        int5 so={item,batch,0,0,0};
        const float64 kw=ks,vw=vs;
        v_f32_st_tnsr_partial(so,key_out_scale,kw,0,0);v_f32_st_tnsr_partial(so,value_out_scale,vw,0,0);
        if(head<mask_heads) {
            int5 mp={item,0,mask_heads==1?row:batch,0,0};
            const float64 m=valid?0.0f:(float64)((uint64)0xff800000u);
            v_f32_st_tnsr_partial(mp,mask,m,0,0);
        }
    }
}
