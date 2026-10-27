#include "gdn_math.h"
// Scaled FP32 sum-of-squares path. Its numerical error against the upstream
// FP64 normalization oracle must be measured; this is not bit-exact FP64 math.
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
            float qmax = 1.0f, kmax = 1.0f;
            for (int col = 0; col < width; ++col) {
                int5 qc = {head * width + col, 0, 0, 0, 0};
                int5 kc = {key_size + head * width + col, 0, 0, 0, 0};
                float qvalue = s_f32_ld_g((__global__ float*)gen_addr(qc, activated));
                float kvalue = s_f32_ld_g((__global__ float*)gen_addr(kc, activated));
                qvalue = qvalue < 0 ? -qvalue : qvalue;
                kvalue = kvalue < 0 ? -kvalue : kvalue;
                if (qvalue > qmax) qmax = qvalue;
                if (kvalue > kmax) kmax = kvalue;
            }
            const float qi = 1.0f / qmax, ki = 1.0f / kmax;
            float qsum = 0, ksum = 0;
            for (int col = 0; col < width; ++col) {
                int5 qc = {head * width + col, 0, 0, 0, 0};
                int5 kc = {key_size + head * width + col, 0, 0, 0, 0};
                float qvalue = s_f32_ld_g((__global__ float*)gen_addr(qc, activated));
                float kvalue = s_f32_ld_g((__global__ float*)gen_addr(kc, activated));
                qvalue *= qi; kvalue *= ki;
                qsum += qvalue * qvalue;
                ksum += kvalue * kvalue;
            }
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
