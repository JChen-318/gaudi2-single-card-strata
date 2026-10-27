#include "gdn_math.h"
// HPU conv state [taps-1,channels], channels contiguous, oldest first.
// HPU conv weights [taps,channels]: transpose the original GGUF [channels,taps].
// Separate next-state avoids cross-invocation read/write aliasing.
void main(tensor state, tensor input, tensor weights, tensor next_state,
          tensor output, int channels, int taps) {
    const int5 start = get_index_space_offset();
    const int5 end = start + get_index_space_size();
    for (int block = start[0]; block < end[0] && block * 64 < channels; ++block) {
        int5 sc = {block * 64, 0, 0, 0, 0};
        int5 vc = {block * 64, 0, 0, 0, 0};
        const float64 value = v_f32_ld_tnsr_b(vc, input);
        float64 sum = 0;
        for (int tap = 0; tap < taps - 1; ++tap) {
            sc[1] = tap;
            const float64 old = v_f32_ld_tnsr_b(sc, state);
            const float64 weight = v_f32_ld_tnsr_b(sc, weights);
            sum += old * weight;
            sc[1] = tap + 1;
            const float64 shifted = tap == taps - 2 ? value : v_f32_ld_tnsr_b(sc, state);
            sc[1] = tap;
            v_f32_st_tnsr(sc, next_state, shifted);
        }
        sc[1] = taps - 1;
        sum += value * v_f32_ld_tnsr_b(sc, weights);
        v_f32_st_tnsr(vc, output, sum * strata_sigmoid(sum));
    }
}
