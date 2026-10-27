#include "gdn_math.h"
// gdn_norm_rows8.c with cheaper math: no max pre-scale (a sum of 128 squares overflows F32 only past |x| ~ 1e18),
// the mean as a multiply by 1 / 128 (exact for a power of two), and the compiler library sigmoid. Not the same bits
// as gdn_norm_rows_bf16: each F32 step rounds differently by about an ulp, before the BF16 rounding.
//   input {rows, hv, 128} F32; z {rows, hv * 128} F32 (z_col: column block of a wider tensor); norm {128};
//   output {rows, hv, 128} BF16. Members are (head, R = 8 rows): more independent rows for the scheduler to overlap.
// A small verify window (<= 8 rows) takes one row per member instead (glue): more members and no row computed for
// nothing; each row's operations are the same either way.
#define R 8
static inline void norm_row(int head, int row, tensor input, tensor z, tensor output, float64 w0, float64 w1,
                            int width, float epsilon, float inverse_width, int z_col) {
    int5 p0 = {0, head, row, 0, 0}, p1 = {64, head, row, 0, 0};
    int5 q0 = {z_col + head * width, row, 0, 0, 0}, q1 = {z_col + head * width + 64, row, 0, 0, 0};
    const float64 x0 = v_f32_ld_tnsr_b(p0, input), x1 = v_f32_ld_tnsr_b(p1, input);
    const float64 g0 = v_f32_ld_tnsr_b(q0, z), g1 = v_f32_ld_tnsr_b(q1, z);
    const float64 sum = v_f32_reduce_add(x0 * x0 + x1 * x1);
    const float64 inverse = v_rsqrt_f32(sum * inverse_width + epsilon);
    float128 wide;
    wide.v1 = ((x0 * inverse) * w0) * v_sigmoid_f32(g0);
    wide.v2 = ((x1 * inverse) * w1) * v_sigmoid_f32(g1);
    v_bf16_st_tnsr(p0, output, v_convert_f32_to_bf16_all_b(wide, SW_LINEAR));
}
void main(tensor input, tensor z, tensor norm, tensor output,
          int width, int value_heads, float epsilon, int rows, int z_col) {
    const int5 start = get_index_space_offset();
    const int5 end = start + get_index_space_size();
    int5 n0 = {0, 0, 0, 0, 0}, n1 = {64, 0, 0, 0, 0};
    const float64 w0 = v_f32_ld_tnsr_b(n0, norm), w1 = v_f32_ld_tnsr_b(n1, norm);
    const float inverse_width = 1.0f / width;
    if (rows <= 8) {
        for (int head = start[0]; head < end[0] && head < value_heads; ++head)
        for (int row = start[1]; row < end[1] && row < rows; ++row)
            norm_row(head, row, input, z, output, w0, w1, width, epsilon, inverse_width, z_col);
        return;
    }
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
            const float64 sum = v_f32_reduce_add(x0[i] * x0[i] + x1[i] * x1[i]);
            const float64 inverse = v_rsqrt_f32(sum * inverse_width + epsilon);
            float128 wide;
            wide.v1 = ((x0[i] * inverse) * w0) * v_sigmoid_f32(g0[i]);
            wide.v2 = ((x1[i] * inverse) * w1) * v_sigmoid_f32(g1[i]);
            int5 p = {0, head, row, 0, 0};
            if (row < rows) v_bf16_st_tnsr(p, output, v_convert_f32_to_bf16_all_b(wide, SW_LINEAR));
        }
    }
}
