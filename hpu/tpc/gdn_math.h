#pragma once
// Full unroll of a constant-count loop. The host semantic emulator (tests/tpc_emulator.hpp) defines it empty.
#ifndef STRATA_UNROLL
#define STRATA_UNROLL _Pragma("unroll")
#endif
// SIMD math helpers. No vector-to-scalar register transfers are required.
#ifdef STRATA_FAST_MATH
// Fast-math twins (CMake builds selected kernels a second time with -DSTRATA_FAST_MATH): the compiler library sigmoid.
static inline float64 strata_sigmoid(float64 x) {return v_sigmoid_f32(x);}
#else
static inline float64 strata_sigmoid(float64 x) {
    float64 absolute = v_f32_sel_less_f32_b(x, 0.0f, -x, x);
    float64 exponential = v_exp_f32(-absolute);
    float64 denominator = 1.0f + exponential;
    return v_f32_sel_less_f32_b(x, 0.0f, exponential / denominator, 1.0f / denominator);
}
#endif
static inline float64 strata_softplus(float64 x) {
    float64 capped = v_f32_sel_less_f32_b(x, 20.0f, x, 20.0f);
    float64 exponential = v_exp_f32(capped);
    float64 rounded = 1.0f + exponential;
    float64 difference = rounded - 1.0f;
    float64 safe = v_f32_sel_less_f32_b(difference, 1e-30f, 1.0f, difference);
    float64 corrected = v_log_f32(rounded) * (exponential / safe);
    float64 small = v_f32_sel_less_f32_b(difference, 1e-30f, exponential, corrected);
    return v_f32_sel_less_f32_b(x, 20.0f, small, x);
}
