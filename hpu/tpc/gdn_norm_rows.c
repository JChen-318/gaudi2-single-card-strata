#include "gdn_math.h"
// STRATA_HPU_DFLASH verify windows: gdn_norm_simd.c for `rows` rows in one kernel (a row loop around its body; the index space
// splits heads and rows,
// unchanged per row).   input {rows, hv, s}; z {rows, hv * s}; norm {s}; output {rows, hv, s}.
// z_col: z may be a column block of a wider F32 tensor (the merged projection).
void main(tensor input, tensor z, tensor norm, tensor output,
          int width, int value_heads, float epsilon, int rows, int z_col) {
    const int5 start = get_index_space_offset();
    const int5 end = start + get_index_space_size();
    for (int head = start[0]; head < end[0] && head < value_heads; ++head) {
        for (int row = start[1]; row < end[1] && row < rows; ++row) {
            // The row's (at most four) input vectors, gate and weights are loaded once and kept in registers: the
            // same operations in the same order as the three-pass loop, without reloading the input twice.
            float64 x[4],zv[4],wv[4];
            const int count = width / 64;
            for (int i = 0; i < count; ++i) {
                int5 pos = {i * 64, head, row, 0, 0};int5 nc = {i * 64, 0, 0, 0, 0};
                int5 zp = {z_col + head * width + i * 64, row, 0, 0, 0};
                x[i] = v_f32_ld_tnsr_b(pos, input);zv[i] = v_f32_ld_tnsr_b(zp, z);wv[i] = v_f32_ld_tnsr_b(nc, norm);
            }
            float64 peak = 0;
            for (int i = 0; i < count; ++i) peak = v_f32_max_b(peak, v_f32_sel_less_f32_b(x[i], 0.0f, -x[i], x[i]));
            const float64 maximum = v_f32_max_b(v_f32_reduce_max(peak), 1.0f);
            const float64 inverse_max = 1.0f / maximum;
            float64 squares = 0;
            for (int i = 0; i < count; ++i) {const float64 value = x[i] * inverse_max; squares += value * value;}
            const float64 sum = v_f32_reduce_add(squares);
            float64 mean = sum / width + (epsilon * inverse_max) * inverse_max;
            const float64 inverse = v_rsqrt_f32(mean) * inverse_max;
            for (int i = 0; i < count; ++i) {
                int5 pos = {i * 64, head, row, 0, 0};
                v_f32_st_tnsr(pos, output, ((x[i] * inverse) * wv[i]) * strata_sigmoid(zv[i]));
            }
        }
    }
}
