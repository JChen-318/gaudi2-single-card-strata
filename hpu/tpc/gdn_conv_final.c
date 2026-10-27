#include "gdn_math.h"
// Full-window prefill: gdn_conv.c over `rows` consecutive inputs in one kernel. Each member keeps its
// 64 channels' tap window in registers and slides it row by row: row r's output and its conv state (next_state<r>,
// only the final state is published). Arithmetic follows gdn_conv.c in the same operation order.
//   state {taps-1, c}; input {rows, c}; weights {taps, c}; next {taps-1, c} (final row only);
//   output {rows, c}. taps <= 4, rows <= 256.
void main(tensor state, tensor input, tensor weights, tensor next,
          tensor output, int channels, int taps, int rows) {
    const int5 start = get_index_space_offset();
    const int5 end = start + get_index_space_size();
    // Members are (64 channels, 64 rows): a member rebuilds its tap window from the three rows before its first
    // (the input, or the carried state for rows before 0), so every row's sum is the same as in one serial pass.
    for (int block = start[0]; block < end[0] && block * 64 < channels; ++block)
    for (int part = start[1]; part < end[1] && part * 64 < rows; ++part) {
        const int first = part * 64, stop = first + 64 < rows ? first + 64 : rows;
        int5 sc = {block * 64, 0, 0, 0, 0};
        // w0..w2: the taps-1 inputs before `first`, w0 the oldest; virtual row j < 0 is state row j + taps - 1.
        float64 w0 = 0, w1 = 0, w2 = 0;
        for (int j = 0; j < taps - 1; ++j) {
            const int src = first - (taps - 1) + j;
            int5 p = {block * 64, src < 0 ? src + taps - 1 : src, 0, 0, 0};
            const float64 old = src < 0 ? v_f32_ld_tnsr_b(p, state) : v_f32_ld_tnsr_b(p, input);
            if (j == 0) w0 = old; else if (j == 1) w1 = old; else w2 = old;
        }
        sc[1] = 0; const float64 k0 = v_f32_ld_tnsr_b(sc, weights);
        sc[1] = 1; const float64 k1 = v_f32_ld_tnsr_b(sc, weights);
        sc[1] = 2; const float64 k2 = taps > 2 ? v_f32_ld_tnsr_b(sc, weights) : 0;
        sc[1] = 3; const float64 k3 = taps > 3 ? v_f32_ld_tnsr_b(sc, weights) : 0;
        int row = first;
        if (taps == 4) {
            // The model's four taps, four rows per step in straight-line code: the four input loads and sigmoids
            // are independent, so the compiler overlaps them. Each row's sum is the loop below's, in its order.
            for (; row + 4 <= stop; row += 4) {
                int5 c0 = {block * 64, row, 0, 0, 0}, c1 = {block * 64, row + 1, 0, 0, 0};
                int5 c2 = {block * 64, row + 2, 0, 0, 0}, c3 = {block * 64, row + 3, 0, 0, 0};
                const float64 v0 = v_f32_ld_tnsr_b(c0, input), v1 = v_f32_ld_tnsr_b(c1, input);
                const float64 v2 = v_f32_ld_tnsr_b(c2, input), v3 = v_f32_ld_tnsr_b(c3, input);
                float64 s0 = 0; s0 += w0 * k0; s0 += w1 * k1; s0 += w2 * k2; s0 += v0 * k3;
                float64 s1 = 0; s1 += w1 * k0; s1 += w2 * k1; s1 += v0 * k2; s1 += v1 * k3;
                float64 s2 = 0; s2 += w2 * k0; s2 += v0 * k1; s2 += v1 * k2; s2 += v2 * k3;
                float64 s3 = 0; s3 += v0 * k0; s3 += v1 * k1; s3 += v2 * k2; s3 += v3 * k3;
                w0 = v1; w1 = v2; w2 = v3;
                v_f32_st_tnsr(c0, output, s0 * strata_sigmoid(s0));
                v_f32_st_tnsr(c1, output, s1 * strata_sigmoid(s1));
                v_f32_st_tnsr(c2, output, s2 * strata_sigmoid(s2));
                v_f32_st_tnsr(c3, output, s3 * strata_sigmoid(s3));
            }
            if (row == rows) {
                const int target = next;
                int5 nc = {block * 64, 0, 0, 0, 0};
                v_f32_st_tnsr(nc, target, w0);
                nc[1] = 1; v_f32_st_tnsr(nc, target, w1);
                nc[1] = 2; v_f32_st_tnsr(nc, target, w2);
            }
        }
        for (; row < stop; ++row) {
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
            const int target = next;
            if(row+1==rows) {
                int5 nc = {block * 64, 0, 0, 0, 0};
                v_f32_st_tnsr(nc, target, w0);
                if (taps > 2) {nc[1] = 1; v_f32_st_tnsr(nc, target, w1);}
                if (taps > 3) {nc[1] = 2; v_f32_st_tnsr(nc, target, w2);}
            }
            v_f32_st_tnsr(vc, output, sum * strata_sigmoid(sum));
        }
    }
}
