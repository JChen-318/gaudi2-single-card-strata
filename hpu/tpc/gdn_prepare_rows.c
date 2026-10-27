#include "gdn_math.h"
// STRATA_HPU_DFLASH verify windows: gdn_prepare_simd.c for `rows` rows in one kernel (a row loop around its body; the index space
// splits heads and rows,
// unchanged per row). 128-wide heads run four rows per member (glue: ceil(rows / 4)).
//   activated {rows, c}; alpha, raw_beta {rows, hv}; dt, ssm_a {hv};
//   q, k {rows, hk, s}; v {rows, hv, s}; gate, beta {rows, hv}.
// alpha_col/beta_col: alpha/raw_beta may be column blocks of one wider F32 tensor (the merged projection).
void main(tensor activated, tensor alpha, tensor dt, tensor ssm_a, tensor raw_beta,
          tensor q, tensor k, tensor v, tensor gate, tensor beta,
          int width, int key_heads, int value_heads, float epsilon, int rows,
          int alpha_col, int beta_col) {
    const int5 start = get_index_space_offset();
    const int5 end = start + get_index_space_size();
    const int key_size = key_heads * width;
    if (width == 128 && rows > 8) { // (a verify window of <= 8 rows takes one row per member below; same bits)
        // 128-wide heads: members are (head, 4 rows) in straight-line code, each q/k row read once into registers
        // (the passes below reload the same values). Same operations per row in the same order, so the same bits.
        for (int head = start[0]; head < end[0] && head < value_heads; ++head)
        for (int group = start[1]; group < end[1] && group * 4 < rows; ++group) {
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int row = group * 4 + i;
                if (row >= rows) continue;
                if (head == 0) for (int block = 0; block < value_heads; block += 64) {
                    int5 vc = {block, row, 0, 0, 0}, sc = {block, 0, 0, 0, 0};
                    int5 ac = {alpha_col + block, row, 0, 0, 0}, bc = {beta_col + block, row, 0, 0, 0};
                    float64 a = v_f32_ld_tnsr_b(ac, alpha), bias = v_f32_ld_tnsr_b(sc, dt);
                    float64 multiplier = v_f32_ld_tnsr_b(sc, ssm_a), raw = v_f32_ld_tnsr_b(bc, raw_beta);
                    v_f32_st_tnsr(vc, gate, multiplier * strata_softplus(a + bias));
                    v_f32_st_tnsr(vc, beta, strata_sigmoid(raw));
                }
                int5 s0 = {2 * key_size + head * 128, row, 0, 0, 0}, s1 = {2 * key_size + head * 128 + 64, row, 0, 0, 0};
                int5 d0 = {0, head, row, 0, 0}, d1 = {64, head, row, 0, 0};
                v_f32_st_tnsr(d0, v, v_f32_ld_tnsr_b(s0, activated));
                v_f32_st_tnsr(d1, v, v_f32_ld_tnsr_b(s1, activated));
            }
            if (head >= key_heads) continue;
            float64 q0[4], q1[4], k0[4], k1[4];
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int row = group * 4 + i < rows ? group * 4 + i : rows - 1;
                int5 a0 = {head * 128, row, 0, 0, 0}, a1 = {head * 128 + 64, row, 0, 0, 0};
                int5 b0 = {key_size + head * 128, row, 0, 0, 0}, b1 = {key_size + head * 128 + 64, row, 0, 0, 0};
                q0[i] = v_f32_ld_tnsr_b(a0, activated); q1[i] = v_f32_ld_tnsr_b(a1, activated);
                k0[i] = v_f32_ld_tnsr_b(b0, activated); k1[i] = v_f32_ld_tnsr_b(b1, activated);
            }
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int row = group * 4 + i;
                float64 qpeak = 0, kpeak = 0;
                qpeak = v_f32_max_b(qpeak, v_f32_sel_less_f32_b(q0[i], 0.0f, -q0[i], q0[i]));
                kpeak = v_f32_max_b(kpeak, v_f32_sel_less_f32_b(k0[i], 0.0f, -k0[i], k0[i]));
                qpeak = v_f32_max_b(qpeak, v_f32_sel_less_f32_b(q1[i], 0.0f, -q1[i], q1[i]));
                kpeak = v_f32_max_b(kpeak, v_f32_sel_less_f32_b(k1[i], 0.0f, -k1[i], k1[i]));
                const float64 qmax = v_f32_max_b(v_f32_reduce_max(qpeak), 1.0f), kmax = v_f32_max_b(v_f32_reduce_max(kpeak), 1.0f);
                const float64 qi = 1.0f / qmax, ki = 1.0f / kmax;
                float64 qsquares = 0, ksquares = 0;
                {const float64 qvalue = q0[i] * qi, kvalue = k0[i] * ki; qsquares += qvalue * qvalue; ksquares += kvalue * kvalue;}
                {const float64 qvalue = q1[i] * qi, kvalue = k1[i] * ki; qsquares += qvalue * qvalue; ksquares += kvalue * kvalue;}
                const float64 qsum = v_f32_reduce_add(qsquares), ksum = v_f32_reduce_add(ksquares);
                float64 qnorm = qsum + (epsilon * qi) * qi, knorm = ksum + (epsilon * ki) * ki, dimension = width;
                const float64 qinv = v_rsqrt_f32(qnorm) * qi, kinv = v_rsqrt_f32(knorm) * ki, scale = v_rsqrt_f32(dimension);
                int5 d0 = {0, head, row, 0, 0}, d1 = {64, head, row, 0, 0};
                if (row < rows) {
                    v_f32_st_tnsr(d0, q, (q0[i] * qinv) * scale); v_f32_st_tnsr(d0, k, k0[i] * kinv);
                    v_f32_st_tnsr(d1, q, (q1[i] * qinv) * scale); v_f32_st_tnsr(d1, k, k1[i] * kinv);
                }
            }
        }
        return;
    }
    for (int head = start[0]; head < end[0] && head < value_heads; ++head) {
        for (int row = start[1]; row < end[1] && row < rows; ++row) {
            if (head == 0) for (int block = 0; block < value_heads; block += 64) {
                int5 vc = {block, row, 0, 0, 0}, sc = {block, 0, 0, 0, 0};
                int5 ac = {alpha_col + block, row, 0, 0, 0}, bc = {beta_col + block, row, 0, 0, 0};
                float64 a = v_f32_ld_tnsr_b(ac, alpha), bias = v_f32_ld_tnsr_b(sc, dt);
                float64 multiplier = v_f32_ld_tnsr_b(sc, ssm_a), raw = v_f32_ld_tnsr_b(bc, raw_beta);
                v_f32_st_tnsr(vc, gate, multiplier * strata_softplus(a + bias));
                v_f32_st_tnsr(vc, beta, strata_sigmoid(raw));
            }
            #pragma unroll(2)
            for (int col = 0; col < width; col += 64) {
                int5 source = {2 * key_size + head * width + col, row, 0, 0, 0};
                int5 dest = {col, head, row, 0, 0};
                float64 value = v_f32_ld_tnsr_b(source, activated);
                v_f32_st_tnsr(dest, v, value);
            }
            if (head < key_heads) {
                float64 qpeak = 0, kpeak = 0;
                #pragma unroll(2)
                for (int col = 0; col < width; col += 64) {
                    int5 qc = {head * width + col, row, 0, 0, 0};
                    int5 kc = {key_size + head * width + col, row, 0, 0, 0};
                    const float64 qvalue = v_f32_ld_tnsr_b(qc, activated), kvalue = v_f32_ld_tnsr_b(kc, activated);
                    qpeak = v_f32_max_b(qpeak, v_f32_sel_less_f32_b(qvalue, 0.0f, -qvalue, qvalue));
                    kpeak = v_f32_max_b(kpeak, v_f32_sel_less_f32_b(kvalue, 0.0f, -kvalue, kvalue));
                }
                const float64 qmax = v_f32_max_b(v_f32_reduce_max(qpeak), 1.0f), kmax = v_f32_max_b(v_f32_reduce_max(kpeak), 1.0f);
                const float64 qi = 1.0f / qmax, ki = 1.0f / kmax;
                float64 qsquares = 0, ksquares = 0;
                #pragma unroll(2)
                for (int col = 0; col < width; col += 64) {
                    int5 qc = {head * width + col, row, 0, 0, 0};
                    int5 kc = {key_size + head * width + col, row, 0, 0, 0};
                    const float64 qvalue = v_f32_ld_tnsr_b(qc, activated) * qi, kvalue = v_f32_ld_tnsr_b(kc, activated) * ki;
                    qsquares += qvalue * qvalue;
                    ksquares += kvalue * kvalue;
                }
                const float64 qsum = v_f32_reduce_add(qsquares), ksum = v_f32_reduce_add(ksquares);
                float64 qnorm = qsum + (epsilon * qi) * qi, knorm = ksum + (epsilon * ki) * ki, dimension = width;
                const float64 qinv = v_rsqrt_f32(qnorm) * qi, kinv = v_rsqrt_f32(knorm) * ki, scale = v_rsqrt_f32(dimension);
                #pragma unroll(2)
                for (int col = 0; col < width; col += 64) {
                    int5 qc = {head * width + col, row, 0, 0, 0};
                    int5 kc = {key_size + head * width + col, row, 0, 0, 0};
                    int5 dest = {col, head, row, 0, 0};
                    float64 qvalue = v_f32_ld_tnsr_b(qc, activated), kvalue = v_f32_ld_tnsr_b(kc, activated);
                    v_f32_st_tnsr(dest, q, (qvalue * qinv) * scale);
                    v_f32_st_tnsr(dest, k, kvalue * kinv);
                }
            }
        }
    }
}
