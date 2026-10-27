// STRATA_HPU_GDN_WINDOW: a prefill window's recurrent steps in one launch. A member owns one (64 value lanes,
// head) state slice outright, so it walks the window row by row instead of the host chaining one node per row;
// the slice is small enough to stay in L2 between rows.
// Per row the arithmetic is gdn_step.c's: same vector width, same load/multiply/add order, and every row reads the
// state the previous row wrote (the running state lives in state_out, copied from state_in first). Every row's
// output and the final state are therefore bit-identical to the per-row chain.
//   state_in, state_out {s(i), hv, s(j)} as gdn_step.c; q, k {rows, hk, s}; v, output {rows, hv, s};
//   gate, beta {rows, hv}.
void main(tensor state_in, tensor q, tensor k, tensor v, tensor gate,
          tensor beta, tensor state_out, tensor output, int width, int key_heads, int rows) {
    const int5 start = get_index_space_offset();
    const int5 end = start + get_index_space_size();
    for (int head = start[1]; head < end[1]; ++head) {
        for (int block = start[0]; block < end[0]; ++block) {
            int5 state_coord = {block * 64, head, 0, 0, 0};
            for (int i = 0; i < width; ++i) {
                state_coord[2] = i;
                v_f32_st_tnsr(state_coord, state_out, v_f32_ld_tnsr_b(state_coord, state_in));
            }
            for (int row = 0; row < rows; ++row) {
                // Tensor vector stores are not coherent with younger vector loads.
                // Retire the initial copy / previous row before reading the running state.
                aso();
                int5 scalar_coord = {head, row, 0, 0, 0};
                const float gate_value = s_f32_ld_g((__global__ float*)gen_addr(scalar_coord, gate));
                const float strength = s_f32_ld_g((__global__ float*)gen_addr(scalar_coord, beta));
                float64 gate_vector = gate_value;
                const float64 decay = v_exp_f32(gate_vector);
                float64 contraction = 0;
                for (int i = 0; i < width; ++i) {
                    state_coord[2] = i;
                    int5 qk_coord = {i, head % key_heads, row, 0, 0};
                    const float ki = s_f32_ld_g((__global__ float*)gen_addr(qk_coord, k));
                    const float64 old = v_f32_ld_tnsr_b(state_coord, state_out);
                    contraction += (old * decay) * ki;
                }
                int5 value_coord = {block * 64, head, row, 0, 0};
                const float64 value = v_f32_ld_tnsr_b(value_coord, v);
                const float64 delta = (value - contraction) * strength;
                float64 answer = 0;
                for (int i = 0; i < width; ++i) {
                    state_coord[2] = i;
                    int5 qk_coord = {i, head % key_heads, row, 0, 0};
                    const float ki = s_f32_ld_g((__global__ float*)gen_addr(qk_coord, k));
                    const float qi = s_f32_ld_g((__global__ float*)gen_addr(qk_coord, q));
                    const float64 old = v_f32_ld_tnsr_b(state_coord, state_out);
                    const float64 next = old * decay + delta * ki;
                    v_f32_st_tnsr(state_coord, state_out, next);
                    answer += next * qi;
                }
                v_f32_st_tnsr(value_coord, output, answer);
            }
        }
    }
}
