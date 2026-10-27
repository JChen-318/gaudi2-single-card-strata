#pragma once
#include "strata/hpu/packed_quant.hpp"
#include <algorithm>
#include <array>
namespace strata::hpu {
struct ExpertFormat {
    std::array<uint32_t,3> types{30,30,30};
    uint64_t bytes=0;
    PackedActivation activation=PackedActivation::Q8_1;
    bool packed=false;
    // --fp8-experts: every role is stored as FP8 codes [rows,columns] followed by its F32 per-row scales
    // (strata/hpu/fp8.hpp) and runs as an MME FP8 GEMM; `types`/`activation` keep their defaults.
    bool fp8=false;
    bool joined_gate_up=false; // opt-in: gate owns [gate codes,up codes,gate scales,up scales]; up is empty
    bool fp8_column_pairs=false;
    bool operator==(const ExpertFormat&) const=default;
};
inline uint64_t fp8_role_bytes(uint64_t rows,uint64_t columns) {return checked_product(rows,columns)+checked_product(rows,4);}
inline ExpertFormat fp8_expert_format(uint64_t hidden,uint64_t intermediate) {
    // The scales follow the codes in one buffer and are bound at that offset, which must stay 128-aligned.
    if(!hidden || !intermediate || hidden>65536 || intermediate>65536 || (hidden*intermediate)%128)
        throw std::invalid_argument("invalid FP8 expert geometry");
    ExpertFormat out;out.fp8=true;
    out.bytes=2*fp8_role_bytes(intermediate,hidden)+fp8_role_bytes(hidden,intermediate);
    return out;
}
inline ExpertFormat expert_format(uint64_t hidden,uint64_t intermediate,int32_t expert,std::array<uint32_t,3> source) {
    if(!hidden || !intermediate || hidden>65536 || intermediate>65536 || expert< -1) throw std::invalid_argument("invalid expert format geometry/ID");
    ExpertFormat out;
    // Every role must fit one packed layout (block size divides its length).
    // K-quant superblocks are accepted wherever the layout fits, matching
    // upstream native_mmvq (shared/head) and iq_kernels (routed); a K-quant down
    // projection at n_ff 640 falls back to decoded BF16.
    out.packed=packed_layout_supported(source[0],hidden) && packed_layout_supported(source[1],hidden)
        && packed_layout_supported(source[2],intermediate);
    if(!out.packed) {out.bytes=checked_product(checked_product(hidden,intermediate),6);return out;}
    out.types=source;out.activation=expert==-1?PackedActivation::Q8_1:
        std::all_of(source.begin(),source.end(),[](auto t){return t==42;})?PackedActivation::RoutedS2:PackedActivation::RoutedQ8;
    const auto gate=checked_product(packed_row_bytes(source[0],hidden),intermediate),up=checked_product(packed_row_bytes(source[1],hidden),intermediate);
    const auto down=checked_product(packed_row_bytes(source[2],intermediate),hidden);
    out.bytes=gate+up+down;return out;
}
inline PackedActivation expert_projection_activation(const ExpertFormat& format,size_t role) {
    if(role>=format.types.size() || !format.packed) throw std::invalid_argument("invalid packed expert projection policy");
    // Mixed Q2/Q4/Q5/Q8 experts preserve Q2's CPU ActQ contract per projection.
    const auto mode=format.activation==PackedActivation::RoutedQ8 && format.types[role]==42?PackedActivation::RoutedS2:format.activation;
    if(!packed_activation_supported(format.types[role],mode)) throw std::invalid_argument("unsupported packed expert projection contract");
    return mode;
}
// Three projection slots: two [codes(H)+metadata(H/16)+F32(FF)] and
// one [codes(FF)+metadata(FF/16)+F32(H)]. Both activation contracts may
// coexist for buckets 1/8/32/128. Three contexts cover shared Q8_1, routed
// all-S2 and routed Q8/mixed-S2. Pinned copies fit under the separate cap.
inline uint64_t packed_expert_mutable_bytes(uint64_t hidden,uint64_t intermediate,uint64_t layers) {
    if(!hidden || !intermediate || hidden>65536 || intermediate>65536 || !layers) throw std::invalid_argument("invalid expert execution reserve geometry");
    const auto per_row=(26*hidden+37*intermediate+3)/4;
    // Every projection has its own persistent codebook, even for a format
    // that does not read it. Shared Q8_1 also owns an FP32 device-quantizer
    // input; routed Q8/S2 currently use host staging. HostRecipe allocates
    // both host and device copies of all these slots, including graph outputs.
    const auto codebooks=checked_product(packed_codebook_slot_bytes(),3*3*4);
    const auto quantizer_inputs=checked_product(8*hidden+4*intermediate,169);
    const auto slots=checked_product(per_row,169*3)+codebooks+quantizer_inputs;
    return checked_product(slots,layers);
}
inline uint64_t scaled_expert_mutable_bytes(uint64_t hidden,uint64_t intermediate) {
    if(!hidden || !intermediate || hidden>65536 || intermediate>65536) throw std::invalid_argument("invalid scaled expert reserve geometry");
    // Two BF16(H)->F32(FF) slots and one BF16(FF)->F32(H), all four buckets.
    return checked_product(8*hidden+10*intermediate,169);
}
}
