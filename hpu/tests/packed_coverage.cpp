// Packed-format coverage audit: every GGML type the pinned upstream admits for a native
// integer dot is accounted for either as an implemented packed dot here or as a named
// decoded format, so no format can be dropped silently. Three authorities must agree for
// each packed type: this file's transcription of native_mmvq.cu:2516 (block values and
// bytes), ggml's own runtime type traits, and the HPU policy. Below that, the whole-surface
// audit classifies EVERY id the pinned ggml build knows (0..GGML_TYPE_COUNT-1) as packed,
// decoded to the reference path, or refused by the loader - and with which refusal - so a
// format cannot be silently unhandled, mislabelled, or charged for bytes it does not own.
// Host only; no device.
#include "strata/hpu/packed_quant.hpp"
#include "strata/hpu/expert_format.hpp"
#include "strata/artifact/gguf_reader.hpp"
#include "../src/weight_format.hpp"
#include <ggml.h>
#include <algorithm>
#include <array>
#include <cstddef>
#include <iostream>
#include <string>
#include <vector>
using namespace strata::hpu;
namespace {
void require(bool ok,const std::string& why){if(!ok) throw std::runtime_error(why);}
struct Authority { uint32_t type,values,bytes; const char* name; };
// Transcribed from src/kernels/cuda/native_mmvq.cu:2516 (native_mmvq_weight_bytes). Every id here
// is a stable GGML id in the pinned build - 42 is GGML_TYPE_Q2_0, which this backend also decodes
// with its own row helper rather than ggml's (see packed_authority).
constexpr std::array<Authority,19> pinned{{
    {2,32,18,"Q4_0"},{3,32,20,"Q4_1"},{6,32,22,"Q5_0"},{7,32,24,"Q5_1"},{8,32,34,"Q8_0"},
    {11,256,110,"Q3_K"},{12,256,144,"Q4_K"},{13,256,176,"Q5_K"},{14,256,210,"Q6_K"},
    {20,32,18,"IQ4_NL"},{23,256,136,"IQ4_XS"},{16,256,66,"IQ2_XXS"},{17,256,74,"IQ2_XS"},{18,256,98,"IQ3_XXS"},{21,256,110,"IQ3_S"},{22,256,82,"IQ2_S"},{29,256,56,"IQ1_M"},{42,64,18,"Strata Q2_0"},{40,64,36,"NVFP4"}}};
// Nothing the pinned upstream admits for a native dot is left decoded.
// The admission list itself is transcribed from native_mmvq_supported
// (native_mmvq.cu:2509), and the two ids this backend packs BEYOND it are named here
// rather than left to be inferred: Q4_1(3), whose affine dot predates the mmvq route, and
// NVFP4(40), which upstream never sends through native_mmvq at all (its row bytes come
// from iq_row_bytes, iq_kernels.cu:2892). ggml's traits stay the byte authority for both.
constexpr std::array<uint32_t,17> mmvq_admitted{{2,6,7,8,11,12,13,14,16,17,18,20,21,22,23,29,42}};
constexpr std::array<uint32_t,2> beyond_mmvq{{3,40}};
struct Missing { uint32_t type; const char* name; };
constexpr std::array<Missing,0> decoded{};
uint32_t checked=0;
void packed_authority(const Authority& a) {
    require(packed_supported(a.type),(a.name+std::string(" is not accepted as packed")).c_str());
    require(packed_values_per_block(a.type)==a.values,(a.name+std::string(" block length differs from the pinned authority")).c_str());
    require(packed_block_bytes(a.type)==a.bytes,(a.name+std::string(" block bytes differ from the pinned authority")).c_str());
    // ggml's own traits must agree for every packed id, including 42: the pinned build registers
    // GGML_TYPE_Q2_0 with a 64/18 block and a row decoder, and this backend's loader gate keeps the
    // same geometry while deliberately NOT taking ggml's decoder (hpu/src/weight_format.hpp:17
    // returns no decode function for 42, because Q2_0 stays on Strata's own S2 row contract).
    const auto* traits=ggml_get_type_traits(static_cast<ggml_type>(a.type));
    require(traits && traits->blck_size==int(a.values) && traits->type_size==std::size_t(a.bytes),
        "ggml traits disagree with the packed block geometry");
    require(traits->to_float,"ggml has no row decoder for a packed type, so the layout has no authority");
    // Every packed type must divide the projection lengths the model actually uses, and a
    // 256-value superblock must still refuse n_ff 640.
    require(packed_layout_supported(a.type,2560) && packed_layout_supported(a.type,6144)
        && packed_layout_supported(a.type,512),"a packed type stopped dividing a real projection length");
    require(packed_layout_supported(a.type,640)==(a.values<=64),(a.name+std::string(" superblock rejection at n_ff 640 changed")).c_str());
    // Shared/head and routed q8_1 serve every integer dot; only Strata Q2_0 keeps the
    // separate CPU-ActQ S2 contract, and it never takes the routed q8_1 dot.
    require(packed_activation_supported(a.type,PackedActivation::Q8_1),std::string(a.name)+" lost the head q8_1 contract");
    require(packed_activation_supported(a.type,PackedActivation::RoutedQ8)==(a.type!=42),std::string(a.name)+" routed contract drifted");
    require(packed_activation_supported(a.type,PackedActivation::RoutedS2)==(a.type==42),std::string(a.name)+" S2 contract drifted");
    // Admission is explicit: every authority id is either in the pinned list or named as a
    // declared extension, and nothing declared an extension may appear in the pinned list.
    const bool admitted=std::find(mmvq_admitted.begin(),mmvq_admitted.end(),a.type)!=mmvq_admitted.end();
    const bool declared=std::find(beyond_mmvq.begin(),beyond_mmvq.end(),a.type)!=beyond_mmvq.end();
    require(admitted||declared,std::string(a.name)+" is packed without an upstream admission or a declared extension");
    require(!(admitted&&declared),std::string(a.name)+" is both upstream-admitted and declared an extension");
    ++checked;
}
// ---- whole-surface classification ----
// These lists were measured, not assumed: work/validation/ggml_surface_probe.cpp (its v2 run is
// work/validation/type-surface-probe-v2.txt) asked the pinned ggml build, the upstream artifact
// reader's own name/geometry tables (include/strata/artifact/gguf_reader.hpp:56 and :125) and this
// backend's loader gate (hpu/src/weight_format.hpp, used by hpu/src/weights.cpp:42 and :68) what
// each says about every id in 0..GGML_TYPE_COUNT-1 - 43 in this build - and about ids past the end.
// The audit requires the live code to still answer with exactly these sets, so a ggml type that is
// added, renamed, re-geometried or re-classified fails here instead of being silently packed,
// decoded, or refused by accident.
constexpr std::array<uint32_t,9> decoded_surface{{0,1,10,19,30,34,35,39,41}};   // F32 F16 Q2_K IQ1_S BF16 TQ1_0 TQ2_0 MXFP4 Q1_0
constexpr std::array<uint32_t,8> refused_no_geometry{{4,5,31,32,33,36,37,38}};  // Q4_2/Q4_3 and the removed *_4_4/_4_8/_8_8 ids
constexpr std::array<uint32_t,7> refused_no_decoder{{9,15,24,25,26,27,28}};    // Q8_1 Q8_K I8 I16 I32 I64 F64
// IQ1_S, TQ1_0, TQ2_0 and Q1_0 have a ggml row decoder, so the loader takes them through the
// reference path, while the upstream artifact table has no bracket for them at all - it counts them
// as "types unknown" and only brackets the formats it knows. That divergence is a fact to pin: a
// fifth id appearing there means the two authorities moved apart again.
constexpr std::array<uint32_t,4> loader_wider{{19,34,35,41}};
// Q8_1 (the activation block) and I8 (the FP8 PLE table bytes, read by hpu/tests/ple_input's own
// path) have upstream geometry but are not weight formats, so the loader refuses them.
constexpr std::array<uint32_t,2> artifact_wider{{9,24}};
// The ids the upstream artifact NAME table does not cover. IQ1_M(29) is in it: this backend packs
// 29 from ggml's traits and its geometry, so a missing name is cosmetic, but the list is pinned so
// the gap cannot quietly widen to a format that is packed or decoded under an anonymous id.
constexpr std::array<uint32_t,13> artifact_unnamed{{4,5,25,26,27,28,29,31,32,33,36,37,38}};
uint32_t surface_rows=0;
template<std::size_t Size> bool listed(const std::array<uint32_t,Size>& ids,uint32_t type) {
    return std::find(ids.begin(),ids.end(),type)!=ids.end();
}
template<std::size_t Size> std::vector<uint32_t> sorted_ids(const std::array<uint32_t,Size>& ids) {
    std::vector<uint32_t> out(ids.begin(),ids.end());std::sort(out.begin(),out.end());return out;
}
std::string upper(std::string text){for(auto& c:text) c=char(c>='a'&&c<='z'?c-32:c);return text;}
// Both lengths are multiples of 256 so a superblock role is never refused for its column count
// while the surface is being classified; n_ff 640 is covered by packed_authority above.
constexpr uint64_t surface_hidden=2560,surface_intermediate=512;
bool authority_row(uint32_t type) {
    return std::any_of(pinned.begin(),pinned.end(),[&](const Authority& a){return a.type==type;});
}
void surface_row(uint32_t type) {
    const std::string id="type "+std::to_string(type);
    const auto* traits=ggml_get_type_traits(static_cast<ggml_type>(type));
    require(traits,(id+": ggml has no traits row for an in-range id").c_str());
    const auto loaded=detail::format(type);
    require(loaded.elements==uint64_t(traits->blck_size) && loaded.bytes==traits->type_size,
        (id+": the weight loader gate reinterprets ggml's block geometry").c_str());
    int geometry_elements=0,geometry_bytes=0;
    const bool artifact=strata::block_geometry(type,geometry_elements,geometry_bytes);
    if(artifact) require(geometry_elements==traits->blck_size && geometry_bytes==int(traits->type_size),
        (id+": the upstream artifact geometry table disagrees with ggml's traits").c_str());
    const std::string artifact_name(strata::ggml_type_name(type));
    if(artifact_name=="?") require(listed(artifact_unnamed,type),(id+": the artifact name table stopped naming a format").c_str());
    else require(traits->type_name && upper(traits->type_name)==artifact_name,
        (id+": the artifact name table and ggml's traits name disagree").c_str());
    const auto format=expert_format(surface_hidden,surface_intermediate,-1,{type,type,type});
    if(packed_supported(type)) {
        require(authority_row(type),(id+": packed by policy with no authority row above").c_str());
        require(traits->is_quantized && traits->to_float && artifact,
            (id+": a packed format lost ggml's decoder or the artifact geometry bracket").c_str());
        require(format.packed && format.bytes==2*packed_row_bytes(type,surface_hidden)*surface_intermediate
            +packed_row_bytes(type,surface_intermediate)*surface_hidden,
            (id+": the expert charge is not the sum of its three packed rows").c_str());
    } else if(detail::convertible(type)) {
        require(listed(decoded_surface,type),(id+": decoded without being listed in the surface").c_str());
        require(!traits->is_quantized || traits->to_float,
            (id+": a decoded quantized format has no ggml row decoder").c_str());
        // The whole expert must fall back to decoded BF16 at the full six-bytes-per-weight charge,
        // never a half-packed mix or a charge for packed bytes that were never produced.
        require(!format.packed && format.bytes==surface_hidden*surface_intermediate*6,
            (id+": decoded expert is mischarged").c_str());
    } else {
        require(!packed_supported(type),(id+": refused by the loader but claimed packed").c_str());
        if(listed(refused_no_geometry,type))
            require(loaded.elements==0 && loaded.bytes==0,
                (id+": pinned as having no block geometry, so payload() throws 'unknown GGUF type' - it now has one").c_str());
        else {
            require(listed(refused_no_decoder,type),(id+": refused without a named reason").c_str());
            require(loaded.elements && !loaded.decode,
                (id+": pinned as having no row decoder, so slice() throws 'GGUF conversion not supported' - it now has one").c_str());
        }
    }
    ++surface_rows;
}
void surface_audit() {
    std::vector<uint32_t> packed_by_policy,decoded_by_policy,refused_by_policy;
    for(uint32_t type=0;type<uint32_t(GGML_TYPE_COUNT);++type)
        (packed_supported(type)?packed_by_policy:detail::convertible(type)?decoded_by_policy:refused_by_policy).push_back(type);
    std::vector<uint32_t> authority;for(const auto& a:pinned) authority.push_back(a.type);
    std::sort(authority.begin(),authority.end());
    require(packed_by_policy==authority,"the packed part of the type surface drifted from the authority list");
    require(decoded_by_policy==sorted_ids(decoded_surface),"the decoded part of the type surface drifted");
    auto refused_expected=sorted_ids(refused_no_geometry);
    for(const auto type:refused_no_decoder) refused_expected.push_back(type);
    std::sort(refused_expected.begin(),refused_expected.end());
    require(refused_by_policy==refused_expected,"the refused part of the type surface drifted");
    require(packed_by_policy.size()+decoded_by_policy.size()+refused_by_policy.size()==uint64_t(GGML_TYPE_COUNT),
        "a ggml id is classified by none of the three surface lists");
    for(uint32_t type=0;type<uint32_t(GGML_TYPE_COUNT);++type) surface_row(type);
    std::vector<uint32_t> wider,narrower;
    for(uint32_t type=0;type<uint32_t(GGML_TYPE_COUNT);++type) {
        int elements=0,bytes=0;
        const bool artifact=strata::block_geometry(type,elements,bytes);
        if(detail::convertible(type) && !artifact) wider.push_back(type);
        if(artifact && !detail::convertible(type)) narrower.push_back(type);
    }
    require(wider==sorted_ids(loader_wider),
        "the loader accepts a format the upstream artifact geometry table cannot bracket");
    require(narrower==sorted_ids(artifact_wider),
        "the artifact geometry table covers a format the loader no longer refuses");
    // Past the end of the enum there is no traits row, so both gates must refuse and no charge may
    // be computed: ggml_get_type_traits would index out of its own table.
    for(const uint32_t type:{uint32_t(GGML_TYPE_COUNT),uint32_t(GGML_TYPE_COUNT)+1,64u}) {
        require(!detail::convertible(type) && !packed_supported(type) && detail::format(type).elements==0,
            "an id past GGML_TYPE_COUNT is not refused by both gates");
        ++surface_rows;
    }
}
}
int main(){try {
    for(const auto& a:pinned) packed_authority(a);
    surface_audit();
    for(const auto& d:decoded) {
        const std::string name(d.name);
        require(!packed_supported(d.type),name+" is claimed packed without a dot");
        const auto* traits=ggml_get_type_traits(static_cast<ggml_type>(d.type));
        require(traits && traits->is_quantized && traits->to_float && traits->blck_size && traits->type_size,name+" has no ggml block authority");
        // The whole expert must fall back to decoded BF16, never a half-packed mix.
        const std::array<uint32_t,3> roles{d.type,d.type,d.type};
        const auto shared=expert_format(2560,640,-1,roles);
        require(!shared.packed && shared.bytes==uint64_t(2560)*640*6,name+" decoded expert is mischarged");
        ++checked;
    }
    // The packed set may not grow without an authority row above: enumerate what policy
    // accepts and require it to be exactly the pinned list.
    std::vector<uint32_t> accepted;for(uint32_t type=0;type<64;++type) if(packed_supported(type)) accepted.push_back(type);
    std::vector<uint32_t> expected;for(const auto& a:pinned) expected.push_back(a.type);
    std::sort(expected.begin(),expected.end());   // the authority table is grouped by family, not by id
    require(accepted==expected,"the packed type set drifted from the audited authority list");
    // Row charges must be the block count times the block size for a real projection.
    require(packed_row_bytes(11,2560)==1100 && packed_row_bytes(20,2560)==1440 && packed_row_bytes(22,2560)==820 && packed_row_bytes(29,2560)==560 && packed_row_bytes(23,2560)==1360
        && packed_row_bytes(14,2560)==2100 && packed_row_bytes(42,640)==180 && packed_row_bytes(40,2560)==1440
        && packed_row_bytes(40,64)==36,"packed row charge drifted");
    std::cout<<"PASS packed coverage audit "<<checked<<" authorities + "<<surface_rows<<" surface rows: "
        <<pinned.size()<<" integer-dot types agree with ggml's runtime traits AND the HPU policy AND the artifact geometry table, "
        <<uint32_t(mmvq_admitted.size())<<" of them with native_mmvq's admission and "<<uint32_t(beyond_mmvq.size())
        <<" declared extensions (Q4_1, NVFP4), with no upstream-admitted format left decoded ("<<decoded.size()<<"); "
        <<"every one of the "<<int(GGML_TYPE_COUNT)<<" ggml ids plus 3 past the end is classified as packed, decoded to the reference path ("
        <<decoded_surface.size()<<": F32 F16 Q2_K IQ1_S BF16 TQ1_0 TQ2_0 MXFP4 Q1_0) or refused with a named reason ("
        <<refused_no_geometry.size()<<" without any block geometry, "<<refused_no_decoder.size()<<" with geometry but no row decoder), "
        <<loader_wider.size()<<" loader-vs-artifact coverage divergences and "<<artifact_wider.size()<<" the other way are pinned as exact sets, "
        <<"superblock rejection at n_ff 640 holds, q8_1 contracts hold and the accepted type set has not drifted; no device\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
