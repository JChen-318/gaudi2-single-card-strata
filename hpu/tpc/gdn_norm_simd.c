#include "gdn_math.h"
// STRATA_HPU_SIMD_GDN variant of gdn_norm.c: per-head RMS norm, shared norm[width], sigmoid(z), with the max and
// the sum of squares reduced as vector trees. This deliberately departs from gdn_norm.c's column-order accumulation
// contract, so it is opt-in and the disk-state configuration records it.
void main(tensor input, tensor z, tensor norm, tensor output,
          int width, int value_heads, float epsilon) {
    const int5 start = get_index_space_offset();
    const int5 end = start + get_index_space_size();
    for (int head = start[0]; head < end[0] && head < value_heads; ++head) {
        float64 peak = 0;
        for (int col = 0; col < width; col += 64) {
            int5 pos = {col, head, 0, 0, 0};
            const float64 value = v_f32_ld_tnsr_b(pos, input);
            peak = v_f32_max_b(peak, v_f32_sel_less_f32_b(value, 0.0f, -value, value));
        }
        const float64 maximum = v_f32_max_b(v_f32_reduce_max(peak), 1.0f);
        const float64 inverse_max = 1.0f / maximum;
        float64 squares = 0;
        for (int col = 0; col < width; col += 64) {
            int5 pos = {col, head, 0, 0, 0};
            const float64 value = v_f32_ld_tnsr_b(pos, input) * inverse_max;
            squares += value * value;
        }
        const float64 sum = v_f32_reduce_add(squares);
        float64 mean = sum / width + (epsilon * inverse_max) * inverse_max;
        const float64 inverse = v_rsqrt_f32(mean) * inverse_max;
        for (int col = 0; col < width; col += 64) {
            int5 pos = {col, head, 0, 0, 0};
            int5 nc = {col, 0, 0, 0, 0};
            float64 value = v_f32_ld_tnsr_b(pos, input), weight = v_f32_ld_tnsr_b(nc, norm);
            float64 gate = v_f32_ld_tnsr_b(pos, z);
            v_f32_st_tnsr(pos, output, ((value * inverse) * weight) * strata_sigmoid(gate));
        }
    }
}
