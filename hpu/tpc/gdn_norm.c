#include "gdn_math.h"
// Per-head RMS norm, shared norm[width], sigmoid(z). This is not SiLU(z).
//
// ACCUMULATION CONTRACT (declared, not accidental): both reductions below are a PLAIN float running sum in COLUMN
// order, with NO compensated (Kahan) accumulation and no pairwise tree. That is the arithmetic this kernel is written
// against; upstream's CUDA kernel reduces the same values in its own order and precision, so whether the two agree
// bit-for-bit is an OPEN NUMERIC QUESTION that only the device can answer. Do not add compensation here on its own:
// the GPU-side reference this port compares against would move at the same time, and a silent change of the
// accumulation is exactly what this comment exists to prevent.
// The `float64` locals below are the deliberate exception: the scale and the reciprocal square root are computed wide
// so the normalization does not lose the whole mantissa.
void main(tensor input, tensor z, tensor norm, tensor output,
          int width, int value_heads, float epsilon) {
    const int5 start = get_index_space_offset();
    const int5 end = start + get_index_space_size();
    for (int head = start[0]; head < end[0] && head < value_heads; ++head) {
        float maximum = 1.0f;
        for (int col = 0; col < width; ++col) {
            int5 pos = {col, head, 0, 0, 0};
            float value = s_f32_ld_g((__global__ float*)gen_addr(pos, input));
            value = value < 0 ? -value : value;
            if (value > maximum) maximum = value;
        }
        const float inverse_max = 1.0f / maximum;
        float sum = 0;
        for (int col = 0; col < width; ++col) {
            int5 pos = {col, head, 0, 0, 0};
            float value = s_f32_ld_g((__global__ float*)gen_addr(pos, input));
            value *= inverse_max;
            sum += value * value;
        }
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
