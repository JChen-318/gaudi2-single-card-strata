// Same recurrence and the same per-lane arithmetic order as gdn_step.c, with q and k read from broadcast
// tensors (qb/kb[key head][i][64 lanes], every lane holding q[i]/k[i]) instead of one scalar global load per i.
// Original-state layout is Strata's (S, value_heads, S), last axis fastest.
void main(tensor state_in, tensor qb, tensor kb, tensor v, tensor gate,
          tensor beta, tensor state_out, tensor output, int width, int key_heads) {
    const int5 start = get_index_space_offset();
    const int5 end = start + get_index_space_size();
    for (int head = start[1]; head < end[1]; ++head) {
        int5 scalar_coord = {head, 0, 0, 0, 0};
        const float gate_value = s_f32_ld_g((__global__ float*)gen_addr(scalar_coord, gate));
        const float strength = s_f32_ld_g((__global__ float*)gen_addr(scalar_coord, beta));
        float64 gate_vector = gate_value;
        const float64 decay = v_exp_f32(gate_vector);
        for (int block = start[0]; block < end[0]; ++block) {
            int5 state_coord = {block * 64, head, 0, 0, 0};
            int5 qk_coord = {0, 0, head % key_heads, 0, 0};
            float64 contraction = 0;
            for (int i = 0; i < width; ++i) {
                state_coord[2] = i;
                qk_coord[1] = i;
                const float64 ki = v_f32_ld_tnsr_b(qk_coord, kb);
                const float64 old = v_f32_ld_tnsr_b(state_coord, state_in);
                contraction += (old * decay) * ki;
            }
            const int5 value_coord = {block * 64, head, 0, 0, 0};
            const float64 value = v_f32_ld_tnsr_b(value_coord, v);
            const float64 delta = (value - contraction) * strength;
            float64 answer = 0;
            for (int i = 0; i < width; ++i) {
                state_coord[2] = i;
                qk_coord[1] = i;
                const float64 ki = v_f32_ld_tnsr_b(qk_coord, kb);
                const float64 qi = v_f32_ld_tnsr_b(qk_coord, qb);
                const float64 old = v_f32_ld_tnsr_b(state_coord, state_in);
                const float64 next = old * decay + delta * ki;
                v_f32_st_tnsr(state_coord, state_out, next);
                answer += next * qi;
            }
            v_f32_st_tnsr(value_coord, output, answer);
        }
    }
}
