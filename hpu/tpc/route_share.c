// STRATA_HPU_DFLASH verify windows (multi-row resident MoE): which slots use the same expert, so its weights are
// read once for two of them (iq4q8_gemv_rows.c). One member per slot. The key of a routed slot is its expert id;
// every shared-expert slot (the last of each row's `period` slots) has key -1, slots past `slots` -2. The slots of
// one key pair up in slot order: the first of each pair computes both, the second nothing, an unpaired last one
// itself, so every member of the GEMV does at most one pair:
//   share[slot][0] = 2 (this slot and its partner), 1 (this slot alone) or 0 (its partner computes it);
//   share[slot][1] = the partner slot when share[slot][0] == 2.
// All arithmetic is F32 lane work (slot numbers and ids < 2^24); the slot's own key is spread to every lane by a
// masked sum, as route_topk.c spreads the shared gate.
//   ids:   I32 {slots} (slots <= 128);   share: I32 {16, slots}.
void main(tensor ids,tensor share,int slots,int period) {
    const int5 begin=get_index_space_offset(),end=begin+get_index_space_size();
    const float64 lane=v_convert_i32_to_f32_b((int64)V_LANE_ID_32,0),upper=lane+64.0f;
    int5 p0={0,0,0,0,0},p1={64,0,0,0,0};
    float64 key0=v_convert_i32_to_f32_b(v_i32_ld_tnsr_b(p0,ids),0),key1=v_convert_i32_to_f32_b(v_i32_ld_tnsr_b(p1,ids),0);
    for(int s=period-1;s<slots;s+=period) {key0=v_f32_sel_eq_f32_b(lane,(float)s,-1.0f,key0);key1=v_f32_sel_eq_f32_b(upper,(float)s,-1.0f,key1);}
    key0=v_f32_sel_less_f32_b(lane,(float)slots,key0,-2.0f);key1=v_f32_sel_less_f32_b(upper,(float)slots,key1,-2.0f);
    for(int slot=begin[0];slot<end[0];++slot) {
        const float position=(float)slot;
        const float64 own=v_f32_reduce_add(slot<64?v_f32_sel_eq_f32_b(lane,position,key0,0.0f):v_f32_sel_eq_f32_b(upper,position,key1,0.0f));
        // Lanes of this key hold their slot number, the others 1e9.
        const float64 hit0=v_f32_sel_eq_f32_b(key0,own,lane,1.0e9f),hit1=v_f32_sel_eq_f32_b(key1,own,upper,1.0e9f);
        const float64 rank=v_f32_reduce_add(v_f32_sel_less_f32_b(hit0,position,1.0f,0.0f)+v_f32_sel_less_f32_b(hit1,position,1.0f,0.0f));
        const float64 partner=v_f32_reduce_min(v_f32_min_b(v_f32_sel_grt_f32_b(hit0,position,hit0,1.0e9f),v_f32_sel_grt_f32_b(hit1,position,hit1,1.0e9f)));
        // rank even: this slot leads a pair (or stands alone); rank odd: the slot before it computes it.
        const float64 halved=rank*0.5f;
        const float64 even=v_f32_sel_eq_f32_b(v_convert_i32_to_f32_b(v_convert_f32_to_i32_b(halved,0),0),halved,1.0f,0.0f);
        const float64 role=v_f32_sel_eq_f32_b(even,1.0f,v_f32_sel_less_f32_b(partner,1.0e8f,2.0f,1.0f),0.0f);
        float64 row=v_f32_sel_eq_f32_b(lane,0.0f,role,0.0f);
        row=v_f32_sel_eq_f32_b(lane,1.0f,v_f32_sel_less_f32_b(partner,1.0e8f,partner,0.0f),row);
        int5 o={0,slot,0,0,0};
        v_i32_st_tnsr_partial(o,share,v_convert_f32_to_i32_b(row,0),15,0);
    }
}
