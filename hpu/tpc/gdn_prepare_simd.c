#include "gdn_math.h"
// STRATA_HPU_SIMD_GDN variant of gdn_prepare.c: the same scaled FP32 sum-of-squares normalization with the
// max and the sum reduced as vector trees instead of column-order scalar loops (different rounding order).
void main(tensor activated, tensor alpha, tensor dt, tensor ssm_a, tensor raw_beta,
          tensor q, tensor k, tensor v, tensor gate, tensor beta,
          int width, int key_heads, int value_heads, float epsilon) {
    const int5 start = get_index_space_offset();
    const int5 end = start + get_index_space_size();
    const int key_size = key_heads * width;
    for (int head = start[0]; head < end[0] && head < value_heads; ++head) {
        if (head == 0) for (int block = 0; block < value_heads; block += 64) {
            int5 vc = {block, 0, 0, 0, 0};
            float64 a = v_f32_ld_tnsr_b(vc, alpha), bias = v_f32_ld_tnsr_b(vc, dt);
            float64 multiplier = v_f32_ld_tnsr_b(vc, ssm_a), raw = v_f32_ld_tnsr_b(vc, raw_beta);
            v_f32_st_tnsr(vc, gate, multiplier * strata_softplus(a + bias));
            v_f32_st_tnsr(vc, beta, strata_sigmoid(raw));
        }
        for (int col = 0; col < width; col += 64) {
            int5 source = {2 * key_size + head * width + col, 0, 0, 0, 0};
            int5 dest = {col, head, 0, 0, 0};
            float64 value = v_f32_ld_tnsr_b(source, activated);
            v_f32_st_tnsr(dest, v, value);
        }
        if (head < key_heads) {
            // Vector max |x| and sum of squares, reduced as trees (SIMD variant of the column-order scalar loops).
            float64 qpeak = 0, kpeak = 0;
            for (int col = 0; col < width; col += 64) {
                int5 qc = {head * width + col, 0, 0, 0, 0};
                int5 kc = {key_size + head * width + col, 0, 0, 0, 0};
                const float64 qvalue = v_f32_ld_tnsr_b(qc, activated), kvalue = v_f32_ld_tnsr_b(kc, activated);
                qpeak = v_f32_max_b(qpeak, v_f32_sel_less_f32_b(qvalue, 0.0f, -qvalue, qvalue));
                kpeak = v_f32_max_b(kpeak, v_f32_sel_less_f32_b(kvalue, 0.0f, -kvalue, kvalue));
            }
            const float64 qmax = v_f32_max_b(v_f32_reduce_max(qpeak), 1.0f), kmax = v_f32_max_b(v_f32_reduce_max(kpeak), 1.0f);
            const float64 qi = 1.0f / qmax, ki = 1.0f / kmax;
            float64 qsquares = 0, ksquares = 0;
            for (int col = 0; col < width; col += 64) {
                int5 qc = {head * width + col, 0, 0, 0, 0};
                int5 kc = {key_size + head * width + col, 0, 0, 0, 0};
                const float64 qvalue = v_f32_ld_tnsr_b(qc, activated) * qi, kvalue = v_f32_ld_tnsr_b(kc, activated) * ki;
                qsquares += qvalue * qvalue;
                ksquares += kvalue * kvalue;
            }
            const float64 qsum = v_f32_reduce_add(qsquares), ksum = v_f32_reduce_add(ksquares);
            float64 qnorm = qsum + (epsilon * qi) * qi, knorm = ksum + (epsilon * ki) * ki, dimension = width;
            const float64 qinv = v_rsqrt_f32(qnorm) * qi, kinv = v_rsqrt_f32(knorm) * ki, scale = v_rsqrt_f32(dimension);
            for (int col = 0; col < width; col += 64) {
                int5 qc = {head * width + col, 0, 0, 0, 0};
                int5 kc = {key_size + head * width + col, 0, 0, 0, 0};
                int5 dest = {col, head, 0, 0, 0};
                float64 qvalue = v_f32_ld_tnsr_b(qc, activated), kvalue = v_f32_ld_tnsr_b(kc, activated);
                v_f32_st_tnsr(dest, q, (qvalue * qinv) * scale);
                v_f32_st_tnsr(dest, k, kvalue * kinv);
            }
        }
    }
}
