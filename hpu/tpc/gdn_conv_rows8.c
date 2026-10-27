#include "gdn_math.h"
// STRATA_HPU_DFLASH verify windows: gdn_conv.c over `rows` consecutive inputs in one kernel. Each member keeps its
// 64 channels' tap window in registers and slides it row by row: row r's output and its conv state (next_state<r>,
// the state after row r) are the ones gdn_conv.c gives when it is run row after row, in the same operation order.
//   state {taps-1, c}; input {rows, c}; weights {taps, c}; next_state0..7 {taps-1, c} (rows of them written);
//   output {rows, c}. taps <= 4, rows <= 8.
void main(tensor state, tensor input, tensor weights, tensor next0, tensor next1, tensor next2, tensor next3, tensor next4, tensor next5, tensor next6, tensor next7,
          tensor output, int channels, int taps, int rows) {
    const int5 start = get_index_space_offset();
    const int5 end = start + get_index_space_size();
    for (int block = start[0]; block < end[0] && block * 64 < channels; ++block) {
        int5 sc = {block * 64, 0, 0, 0, 0};
        // w0..w2: gdn_conv.c's state rows 0..taps-2 (w0 the oldest input).
        float64 w0 = 0, w1 = 0, w2 = 0;
        sc[1] = 0; if (taps > 1) w0 = v_f32_ld_tnsr_b(sc, state);
        sc[1] = 1; if (taps > 2) w1 = v_f32_ld_tnsr_b(sc, state);
        sc[1] = 2; if (taps > 3) w2 = v_f32_ld_tnsr_b(sc, state);
        sc[1] = 0; const float64 k0 = v_f32_ld_tnsr_b(sc, weights);
        sc[1] = 1; const float64 k1 = v_f32_ld_tnsr_b(sc, weights);
        sc[1] = 2; const float64 k2 = taps > 2 ? v_f32_ld_tnsr_b(sc, weights) : 0;
        sc[1] = 3; const float64 k3 = taps > 3 ? v_f32_ld_tnsr_b(sc, weights) : 0;
        for (int row = 0; row < rows; ++row) {
            int5 vc = {block * 64, row, 0, 0, 0};
            const float64 value = v_f32_ld_tnsr_b(vc, input);
            // gdn_conv.c: sum over tap 0..taps-2 of old[tap] * weight[tap], then value * weight[taps-1].
            float64 sum = 0;
            sum += w0 * k0;
            if (taps > 2) sum += w1 * k1;
            if (taps > 3) sum += w2 * k2;
            const float64 last = taps == 2 ? k1 : taps == 3 ? k2 : k3;
            sum += value * last;
            // The state after this row: the window shifted by one, this row's input newest.
            if (taps == 2) {w0 = value;}
            else if (taps == 3) {w0 = w1; w1 = value;}
            else {w0 = w1; w1 = w2; w2 = value;}
            const int target = 3 + row; // next<row> is tensor 3 + row
            int5 nc = {block * 64, 0, 0, 0, 0};
            v_f32_st_tnsr(nc, target, w0);
            if (taps > 2) {nc[1] = 1; v_f32_st_tnsr(nc, target, w1);}
            if (taps > 3) {nc[1] = 2; v_f32_st_tnsr(nc, target, w2);}
            v_f32_st_tnsr(vc, output, sum * strata_sigmoid(sum));
        }
    }
}
