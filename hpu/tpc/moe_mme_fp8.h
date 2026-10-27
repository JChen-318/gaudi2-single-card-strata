#pragma once
// STRATA_HPU_MOE_FP8: a routed FP8 GEMM output tile of 256 BF16 columns is in the resident row order (column L is
// natural row 64*(L%4) + L/4, tpc/iq4_expand_tiles_fp8.c). TILE4 returns its four natural 64-row chunks as F32:
// the default BF16 -> F32 convert splits even / odd columns, a linear F32 -> BF16 pack (exact: the values are BF16)
// joins the two halves' even (odd) parts, and a second split separates chunks 0 / 2 (1 / 3). A macro, as tensors
// cannot be passed to functions. T: tensor; P: int5 of column 0 of the tile.
#define TILE4(T,P,C0,C1,C2,C3) { \
    int5 q_=P; \
    const float128 a_=v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(q_,T),0); \
    q_[0]+=128; \
    const float128 b_=v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(q_,T),0); \
    float128 e_,o_;e_.v1=a_.v1;e_.v2=b_.v1;o_.v1=a_.v2;o_.v2=b_.v2; \
    const float128 even_=v_convert_bf16_to_f32_all_b(v_convert_f32_to_bf16_all_b(e_,SW_LINEAR),0); \
    const float128 odd_=v_convert_bf16_to_f32_all_b(v_convert_f32_to_bf16_all_b(o_,SW_LINEAR),0); \
    C0=even_.v1;C2=even_.v2;C1=odd_.v1;C3=odd_.v2; }
