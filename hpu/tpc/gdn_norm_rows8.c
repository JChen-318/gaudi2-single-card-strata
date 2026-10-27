#include "gdn_math.h"
// gdn_norm_rows_bf16.c for 128-wide heads, R rows per member in unrolled straight-line code, so the compiler can
// interleave the rows' serial chains (reductions, divides, rsqrt). Same operations per row, so the same bits.
//   input {rows, hv, 128} F32; z {rows, hv * 128} F32 (z_col: column block of a wider tensor); norm {128};
//   output {rows, hv, 128} BF16. Members are (head, R rows).
#define R 4
void main(tensor input, tensor z, tensor norm, tensor output,
          int width, int value_heads, float epsilon, int rows, int z_col) {
    const int5 start = get_index_space_offset();
    const int5 end = start + get_index_space_size();
    int5 n0 = {0, 0, 0, 0, 0}, n1 = {64, 0, 0, 0, 0};
    const float64 w0 = v_f32_ld_tnsr_b(n0, norm), w1 = v_f32_ld_tnsr_b(n1, norm);
    for (int head = start[0]; head < end[0] && head < value_heads; ++head)
    for (int group = start[1]; group < end[1] && group * R < rows; ++group) {
        float64 x0[R], x1[R], g0[R], g1[R];
        #pragma unroll
        for (int i = 0; i < R; ++i) {
            const int row = group * R + i < rows ? group * R + i : rows - 1;
            int5 p0 = {0, head, row, 0, 0}, p1 = {64, head, row, 0, 0};
            int5 q0 = {z_col + head * width, row, 0, 0, 0}, q1 = {z_col + head * width + 64, row, 0, 0, 0};
            x0[i] = v_f32_ld_tnsr_b(p0, input); x1[i] = v_f32_ld_tnsr_b(p1, input);
            g0[i] = v_f32_ld_tnsr_b(q0, z); g1[i] = v_f32_ld_tnsr_b(q1, z);
        }
        #pragma unroll
        for (int i = 0; i < R; ++i) {
            const int row = group * R + i;
            float64 peak = 0;
            peak = v_f32_max_b(peak, v_f32_sel_less_f32_b(x0[i], 0.0f, -x0[i], x0[i]));
            peak = v_f32_max_b(peak, v_f32_sel_less_f32_b(x1[i], 0.0f, -x1[i], x1[i]));
            const float64 maximum = v_f32_max_b(v_f32_reduce_max(peak), 1.0f);
            const float64 inverse_max = 1.0f / maximum;
            float64 squares = 0;
            {const float64 value = x0[i] * inverse_max; squares += value * value;}
            {const float64 value = x1[i] * inverse_max; squares += value * value;}
            const float64 sum = v_f32_reduce_add(squares);
            float64 mean = sum / width + (epsilon * inverse_max) * inverse_max;
            const float64 inverse = v_rsqrt_f32(mean) * inverse_max;
            float128 wide;
            wide.v1 = ((x0[i] * inverse) * w0) * strata_sigmoid(g0[i]);
            wide.v2 = ((x1[i] * inverse) * w1) * strata_sigmoid(g1[i]);
            int5 p = {0, head, row, 0, 0};
            if (row < rows) v_bf16_st_tnsr(p, output, v_convert_f32_to_bf16_all_b(wide, SW_LINEAR));
        }
    }
}
