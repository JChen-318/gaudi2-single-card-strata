#pragma once
#include "strata/hpu/tensor.hpp"
#include <span>

namespace strata::hpu {
enum class PackedActivation : uint32_t { Q8_1=0, RoutedS2=1, RoutedQ8=2 };
bool packed_supported(uint32_t type);
// RoutedQ8 has Q8_1 activation bytes/metadata, but Q4_0 centers the weight
// codes before its integer dot (iq_kernels.cu). Shared/head Q4_0 instead
// subtracts the raw-input sum (native_mmvq.cu). Q2 routed uses its own S2 mode.
// The 256-value K superblocks (Q3_K/Q4_K/Q5_K) take their own superblock dot
// under either q8_1 contract: upstream native_mmvq.cu q3_q8_dot_impl,
// q4_q8_dot_impl and iq_kernels.cu vec_dot_q4_K_q8_1_impl_vmmq all multiply the
// group terms by the sum of the QUANTIZED activation codes. Q3_K has one signed
// super-block d, a -32 scale bias and no learned minimum. IQ4_NL(20) is a 32-value
// nonlinear codebook block: small_q8_dot(IQ4NLBlock) multiplies the codebook dot by
// one half d and the activation d8, so it takes the same q8_1 integer contracts.
// IQ4_XS(23) keeps the 256-value superblock shape but replaces the affine group scale
// with (6-bit raw - 32) and the codebook values, like upstream iq4_xs_q8_dot. IQ2_XXS(16)
// is a 256-value codebook superblock whose group scale the pinned CUDA applies as the
// integer sumi*(2*sc+1)/8, so it takes the same q8_1 integer contracts. IQ3_XXS(18) is
// its 3-bit sibling: four magnitudes per grid byte and the pinned CUDA scale
// (sc*sumi + sumi/2)/2, again under both q8_1 contracts. IQ2_XS(17) packs a 9-bit
// index and a 7-bit sign field into one uint16 per 8 values and splits its two 4-bit
// nibble scales over the halves of a 32-value group, as the pinned CUDA integer.
// IQ3_S(21) carries its 9th index bit in a separate qh field, its plain 8-bit signs in
// a signs array, and the integer multiplier (1 + 2*sc) with no truncation at all.
// IQ2_S(22) keeps the 10th index bit in qh too, signs each 8-value lane with a plain
// sign byte, and truncates through the CUDA integer
// (sumi0*ls0 + sumi1*ls1 + (sumi0+sumi1)/2)/4 over the two nibble scales.
// IQ1_M(29) assembles its fp16 scale from nibbles of the block's own scale bytes, keeps
// index bits 8..10 and a per-lane +/-0.125 delta bit in qh, and sums the ternary grid as
// (sumi + delta*sumy)*(2*field+1), which is the decoder's law term by term.
// NVFP4(40) is a 64-value block with four UE4M3 sub-block scales and doubled e2m1
// nibble magnitudes; the pinned helper's closing half makes the dot integer. Upstream
// native_mmvq does NOT admit it, so tests/packed_coverage.cpp records that divergence.
uint32_t packed_values_per_block(uint32_t type);
inline bool packed_layout_supported(uint32_t type,uint64_t columns) {
    if(!columns || columns>65536 || !packed_supported(type)) return false;
    return columns%packed_values_per_block(type)==0;
}
// Literal type sets keep this predicate free of out-of-line calls: the test
// graph interpreter uses it without linking the packed translation unit.
inline bool packed_activation_supported(uint32_t type,PackedActivation mode) {
    const bool integer=type==2 || type==3 || type==6 || type==7 || type==8 || type==11 || type==12 || type==13 || type==14 || type==20 || type==23 || type==16 || type==17 || type==18 || type==21 || type==22 || type==29 || type==40;
    return (mode==PackedActivation::Q8_1 && (integer || type==42)) ||
        (mode==PackedActivation::RoutedS2 && type==42) ||
        (mode==PackedActivation::RoutedQ8 && integer);
}
// Stable GGML IDs: Q4_0=2, Q4_1=3, Q5_0=6, Q5_1=7, Q8_0=8, Q3_K=11, Q4_K=12,
// Q5_K=13, Q6_K=14, IQ4_NL=20, IQ4_XS=23, IQ2_XXS=16, IQ2_XS=17, IQ3_XXS=18, IQ3_S=21, IQ2_S=22, IQ1_M=29, Strata Q2_0=42, NVFP4=40. Raw row layouts are kept. Affine3/7 use the quantized-code
// sum for their learned minimum; symmetric2/6 use Q8_1's original-input sum;
// K-quant group minima use the quantized-code sum with separate d and dmin.
uint32_t packed_block_bytes(uint32_t type);
uint32_t packed_codebook_slot_bytes(); // generated TPC layout, without exposing its source-tree path
uint64_t packed_row_bytes(uint32_t type,uint64_t columns);
struct Q8Activation {
    std::vector<uint8_t> codes; // signed int8 bits, row-major
    std::vector<float> metadata; // [rows,columns/32,2]: scale and correction (mode-specific)
};
// Original CUDA Q8_1 contract: raw-input XOR sum, away-from-zero codes,
// finite-half clamp, then RN-even FP16 scale/sum. Host staging is intentional;
// this function does not claim accelerator quantization or CUDA dot reduction.
Q8Activation quantize_q8_activation(std::span<const float>,uint32_t columns);
// Routed Q2 uses CPU ActQ's FP32 scale, reciprocal multiply + signed .5
// truncation, and FP32 scale*sum(codes). It is NOT the head/shared Q8_1 contract.
Q8Activation quantize_s2_activation(std::span<const float>,uint32_t columns);
// The runtime's host-staging selector. RoutedQ8 uses Q8_1 too; only RoutedS2
// may use CPU ActQ's FP32 scale/correction. Keep this outside device code so
// the choice used by ModelHead can be exercised without acquiring an HPU.
inline Q8Activation quantize_packed_activation(std::span<const float> input,uint32_t columns,PackedActivation mode) {
    if(mode==PackedActivation::RoutedS2) return quantize_s2_activation(input,columns);
    if(mode==PackedActivation::Q8_1 || mode==PackedActivation::RoutedQ8) return quantize_q8_activation(input,columns);
    throw std::invalid_argument("unknown packed activation quantizer contract");
}
void validate_packed_rows(std::span<const uint8_t>,uint32_t type,uint64_t columns,uint64_t rows);
}
