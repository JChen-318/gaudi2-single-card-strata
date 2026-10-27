// Native IQ2_S (GGML type 22) 256-value codebook superblock dot for the shared/head and
// routed q8_1 contracts. IQ2_S shares IQ2_XS's truncating CUDA scale law - the pinned dot
// (src/kernels/cuda/iq_kernels.cu:155) applies the two nibble scales as the INTEGER
// (sumi0*ls0 + sumi1*ls1 + (sumi0+sumi1)/2)/4 while the unmodified pinned decoder
// (ggml/src/ggml-quants.c:2543) multiplies each 16-value half by the float (0.5+ls)*0.25 -
// so the kernel is compared EXACTLY against an independent transcription of the CUDA
// integer form and against the decoder inside the band that the two truncations provably
// introduce (at most 1 + 1/8 integer units of d*d8 per 32-value group). IQ2_S differs from
// IQ2_XS in the fields: a 10-bit index split over a grid byte plus two bits of a separate qh
// byte, and signs that are PLAIN bit-for-value with no popcount parity fold. All 1024 grid
// entries and both rules are recovered through the decoder, so nothing trusts the table
// compiled into the kernel. Host only: no ISA execution, no device, no performance claim.
#include "strata/hpu/packed_quant.hpp"
#include "strata/hpu/expert_format.hpp"
#include "strata/hpu/packed_expert_flow.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "packed_emulator.hpp"
#include <ggml.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <span>
#include <vector>
using namespace strata::hpu;
namespace {
void require(bool ok,const char* why){if(!ok) throw std::runtime_error(why);}
template<class F> void rejects(F f){bool bad=false;try{f();}catch(const std::exception&){bad=true;}require(bad,"invalid IQ2_S payload or policy accepted");}
int signed_code(uint8_t b){return b<128?b:int(b)-256;}
constexpr uint32_t super=256,groups=8,codes=32,lane_values=8,xs_bytes=82,grid_entries=1024,q8_bytes=34;
constexpr uint32_t qs_at=2,signs_at=34,qh_at=66,scales_at=74;
uint32_t blocks(uint32_t columns){return columns/super;}
struct Oracle { long double value,magnitude,bound; };
struct Row {
    uint32_t columns;
    std::vector<uint16_t> index;                 // [blocks][32] 10-bit grid index, one per 8-value lane
    std::vector<uint8_t> signs;                  // [blocks][32] plain sign bits for the same lanes
    std::vector<uint8_t> scale;                  // [blocks][8] two 4-bit nibble scales per byte
    std::vector<float> d;                        // [blocks] signed
};
const ggml_type_traits* traits() {
    const auto* t=ggml_get_type_traits(GGML_TYPE_IQ2_S);
    require(t && t->blck_size==int(super) && t->type_size==xs_bytes && t->to_float,"pinned ggml lacks the IQ2_S row decoder or its block size differs");
    return t;
}
std::vector<float> decode_row(const std::vector<uint8_t>& bytes,uint32_t columns,uint32_t row) {
    std::vector<float> weights(columns);
    traits()->to_float(bytes.data()+uint64_t(row)*packed_row_bytes(22,columns),weights.data(),int64_t(columns));
    return weights;
}
// The measured map: half d at 0, 32 grid bytes at 2 whose low 8 bits are the index, the 32
// sign bytes at 34, 8 qh bytes at 66 carrying the 9th and 10th index bits of their four
// lanes, and 8 scale bytes at 74. qh byte g holds lane l's two high bits at 2l and 2l+1.
uint32_t index_of(const Row& w,uint32_t sb,uint32_t code) {return w.index[uint64_t(sb)*codes+code];}
void store_index(Row& w,uint32_t sb,uint32_t code,uint32_t value) {w.index[uint64_t(sb)*codes+code]=uint16_t(value&(grid_entries-1));}
std::vector<uint8_t> x_encode(const std::vector<Row>& rows) {
    const auto columns=rows.front().columns;
    std::vector<uint8_t> bytes(uint64_t(rows.size())*blocks(columns)*xs_bytes,0);
    const auto put=[&](size_t at,float value){const auto bits=strata::kernels::f16_from_f32(value);bytes[at]=uint8_t(bits);bytes[at+1]=uint8_t(bits>>8);};
    for(size_t row=0;row<rows.size();++row) {const auto& w=rows[row];
        for(uint32_t sb=0;sb<blocks(columns);++sb) {
            const size_t at=(uint64_t(row)*blocks(columns)+sb)*xs_bytes;
            for(uint32_t c=0;c<codes;++c) {
                bytes[at+qs_at+c]=uint8_t(index_of(w,sb,c)&255u);
                bytes[at+signs_at+c]=w.signs[uint64_t(sb)*codes+c];
                bytes[at+qh_at+c/4]|=uint8_t((index_of(w,sb,c)>>8)<<(2*(c%4)));
            }
            for(uint32_t g=0;g<groups;++g) bytes[at+scales_at+g]=w.scale[uint64_t(sb)*groups+g];
            put(at,w.d[sb]);
        }
    }
    return bytes;
}
// Eight magnitudes of one 10-bit grid index, read back through the decoder: d=1, code 0 of
// group 0 carries the index (its high two bits in qh[0] bits 0 and 1), every other code stays
// 0 and a scale nibble of 0 gives the factor (0.5+0)*0.25 = 1/8 exactly.
std::array<int,8> magnitudes_of(uint32_t index) {
    std::vector<uint8_t> bytes(xs_bytes,0);
    bytes[0]=0x00;bytes[1]=0x3c;bytes[qs_at]=uint8_t(index&255u);bytes[qh_at]=uint8_t(index>>8);
    const auto weights=decode_row(bytes,super,0);
    std::array<int,8> out{};
    for(uint32_t j=0;j<lane_values;++j) {
        const double value=double(weights[j])*8.0;
        require(std::abs(value-std::round(value))<1e-3,"IQ2_S grid magnitude is not integral");
        out[j]=int(std::round(value));
    }
    return out;
}
const std::array<std::array<int,8>,grid_entries>& grid_table() {
    static const std::array<std::array<int,8>,grid_entries> table=[]{
        std::array<std::array<int,8>,grid_entries> built{};
        for(uint32_t index=0;index<grid_entries;++index) built[index]=magnitudes_of(index);
        return built;
    }();
    return table;
}
uint32_t grid_pins=0,table_bytes=0;
void pin_grid() {
    std::vector<std::array<int,8>> seen(grid_entries);
    for(uint32_t index=0;index<grid_entries;++index) {
        const auto m=grid_table()[index];
        require(std::all_of(m.begin(),m.end(),[](int value){return value==8||value==25||value==43;}),"an IQ2_S grid byte is not a pinned magnitude");
        seen[index]=m;++grid_pins;
    }
    // The codebook compiled into the TPC kernel must agree byte for byte with what the decoder
    // reports for the same index: a dot only reaches the entries its inputs happen to select.
    for(uint32_t index=0;index<grid_entries;++index) {
        const auto m=grid_table()[index];
        const unsigned long long compiled=packed_emulator::packed_iq2s_grid[index];
        for(uint32_t j=0;j<lane_values;++j) {
            require(int((compiled>>(8*j))&255ull)==m[j],"the IQ2_S codebook compiled into the TPC kernel differs from the pinned decoder");
            ++table_bytes;
        }
    }
    uint32_t collisions=0;
    for(uint32_t a=0;a<grid_entries;++a) for(uint32_t b=a+1;b<grid_entries;++b) if(seen[a]==seen[b]) ++collisions;
    require(collisions==0,"IQ2_S grid entries are not distinguishable through the decoder");
    // The two qh bits must be readable: index 0 and index 512 are different entries, and each
    // adjacent pair differs somewhere, so a wrong index cannot hide.
    require(grid_table()[0]!=grid_table()[512] && grid_table()[0]!=grid_table()[256],"the IQ2_S high index bits are not observable through the decoder");
    uint32_t neighbours=0;
    for(uint32_t index=0;index+1<grid_entries;++index) if(grid_table()[index]!=grid_table()[index+1]) ++neighbours;
    require(neighbours==grid_entries-1,"an IQ2_S grid index has an indistinguishable neighbour");
}
uint32_t sign_checks=0;
void sign_rule() {
    // IQ2_S signs are a PLAIN 8-bit field: bit b of the sign byte that owns values
    // 8c..8c+7 negates exactly value 8c+b, with no popcount parity fold as IQ2_XXS/IQ3_XXS
    // use. Verified for every pattern of two lanes.
    for(uint32_t lane=0;lane<2;++lane) for(uint32_t bits=0;bits<256;++bits) {
        std::vector<uint8_t> bytes(xs_bytes,0);
        bytes[0]=0x00;bytes[1]=0x3c;bytes[signs_at+lane]=uint8_t(bits);
        const auto weights=decode_row(bytes,super,0);
        for(uint32_t j=0;j<lane_values;++j) {
            const float expected=((bits&(1u<<j))?-1.0f:1.0f)*float(grid_table()[0][j])/8.0f;
            require(std::abs(double(weights[lane*8+j])-double(expected))<1e-4,"an IQ2_S sign bit did not map bit-for-value as measured");
            ++sign_checks;
        }
    }
}
Row x_logical(uint32_t columns,uint32_t row) {
    const auto count=blocks(columns);
    Row w{columns,std::vector<uint16_t>(uint64_t(count)*codes),std::vector<uint8_t>(uint64_t(count)*codes),
        std::vector<uint8_t>(uint64_t(count)*groups),std::vector<float>(count)};
    for(uint32_t sb=0;sb<count;++sb) {
        w.d[sb]=float(int((sb*7+row*3+4)%11)-5)/8.f;
        for(uint32_t c=0;c<codes;++c) {
            store_index(w,sb,c,uint32_t((c*37+sb*11+row*5)%grid_entries));
            w.signs[uint64_t(sb)*codes+c]=uint8_t((c*13+sb*5+row)%256);
        }
        for(uint32_t g=0;g<groups;++g) w.scale[uint64_t(sb)*groups+g]=uint8_t(((g*3+row)%16) | (uint32_t((g*5+sb*2+row)%16)<<4));
    }
    return w;
}
// Independent transcription of the pinned CUDA integer form (the dot starts at
// iq_kernels.cu:155 and sums at :183): two 16-value half-dots scaled by the two nibbles,
// summed with the truncating (sumi0+sumi1)/2 and /4.
Oracle x_oracle(const Row& w,const Q8Activation& q,uint32_t token) {
    Oracle out{0,0,0};
    for(uint32_t sb=0;sb<blocks(w.columns);++sb) for(uint32_t ib=0;ib<groups;++ib) {
        const int ls0=int(w.scale[uint64_t(sb)*groups+ib]&15u),ls1=int(w.scale[uint64_t(sb)*groups+ib]>>4);
        int sumi0=0,sumi1=0;
        for(uint32_t lane=0;lane<4;++lane) {
            const uint32_t code=ib*4+lane;
            const auto m=grid_table()[index_of(w,sb,code)];
            const uint8_t signs=w.signs[uint64_t(sb)*codes+code];
            int lane_sum=0;
            for(uint32_t j=0;j<lane_values;++j)
                lane_sum+=((signs&(1u<<j))?-m[j]:m[j])*signed_code(q.codes[uint64_t(token)*w.columns+sb*super+ib*32+lane*8+j]);
            if(lane<2) sumi0+=lane_sum;else sumi1+=lane_sum;
        }
        const int sumi=(sumi0*ls0+sumi1*ls1+(sumi0+sumi1)/2)/4;               // C truncation, as upstream
        const double factor=double(w.d[sb])*q.metadata[(uint64_t(token)*w.columns/32+sb*groups+ib)*2];
        out.value+=static_cast<long double>(factor)*sumi;
        out.magnitude+=std::abs(static_cast<long double>(factor)*sumi);
        out.bound+=std::abs(static_cast<long double>(factor));                // one integer unit per group
    }
    return out;
}
Oracle x_decoded(const std::vector<uint8_t>& bytes,uint32_t columns,uint32_t row,const Q8Activation& q,uint32_t token) {
    const auto weights=decode_row(bytes,columns,row);
    const size_t base=uint64_t(row)*packed_row_bytes(22,columns);
    Oracle out{0,0,0};
    for(uint32_t block=0;block<columns/32;++block) {
        const long double d8=q.metadata[(uint64_t(token)*columns/32+block)*2];long double part=0;
        for(uint32_t v=0;v<32;++v) part+=static_cast<long double>(weights[block*32+v])*signed_code(q.codes[uint64_t(token)*columns+block*32+v]);
        const long double term=d8*part;out.value+=term;out.magnitude+=std::abs(term);
        const size_t at=base+uint64_t(block/8)*xs_bytes;
        const float d=strata::kernels::f32_from_f16(uint16_t(bytes[at])|(uint16_t(bytes[at+1])<<8));
        out.bound+=std::abs(static_cast<long double>(d)*d8);
    }
    return out;
}
// The kernel's own accumulation, term by term in float32, so a flow can tell a cancellation
// band apart from an arithmetic disagreement. The pinned dot sums in float32 as well.
float x_accumulated(const Row& w,const Q8Activation& q,uint32_t token) {
    float total=0;
    for(uint32_t sb=0;sb<blocks(w.columns);++sb) for(uint32_t ib=0;ib<groups;++ib) {
        const int ls0=int(w.scale[uint64_t(sb)*groups+ib]&15u),ls1=int(w.scale[uint64_t(sb)*groups+ib]>>4);
        int sumi0=0,sumi1=0;
        for(uint32_t lane=0;lane<4;++lane) {
            const uint32_t code=ib*4+lane;
            const auto m=grid_table()[index_of(w,sb,code)];
            const uint8_t signs=w.signs[uint64_t(sb)*codes+code];
            int lane_sum=0;
            for(uint32_t j=0;j<lane_values;++j)
                lane_sum+=((signs&(1u<<j))?-m[j]:m[j])*signed_code(q.codes[uint64_t(token)*w.columns+sb*super+ib*32+lane*8+j]);
            if(lane<2) sumi0+=lane_sum;else sumi1+=lane_sum;
        }
        const int sumi=(sumi0*ls0+sumi1*ls1+(sumi0+sumi1)/2)/4;
        const float d=w.d[sb],d8=q.metadata[(uint64_t(token)*w.columns/32+sb*groups+ib)*2];
        total+=(d*d8)*(float)sumi;
    }
    return total;
}
std::vector<float> tpc(const std::vector<uint8_t>& bytes,uint32_t type,uint32_t columns,uint32_t rows,const Q8Activation& q,uint32_t batch,PackedActivation mode) {
    std::vector<float> result(uint64_t(rows)*batch,NAN);
    using namespace packed_emulator;
    auto& w_bytes=const_cast<std::vector<uint8_t>&>(bytes);auto& codes=const_cast<std::vector<uint8_t>&>(q.codes);auto& meta=const_cast<std::vector<float>&>(q.metadata);
    tensor w{w_bytes.data(),{packed_row_bytes(type,columns),rows},1},a{codes.data(),{columns,batch},1},
        m{meta.data(),{2,columns/32,batch},4},o{result.data(),{rows,batch},4};
    for(uint32_t token=0;token<batch;++token) for(uint32_t row=0;row<rows;++row) {
        offset={int(row),int(token),0,0,0};extent={1,1,1,1,1};
        packed_kernel(w,a,m,o,int(columns),int(rows),int(batch),int(type),int(mode));
    }
    return result;
}
bool close_exact(float actual,long double expected) {
    return std::isfinite(actual) && std::abs(static_cast<long double>(actual)-expected)<=std::max<long double>(2e-3L,std::abs(expected)*2e-5L);
}
// Truncating (sumi0+sumi1)/2 loses at most 1/2 and the outer /4 loses at most 1 unit of the
// numerator, so the CUDA integer form is at most 1 + 1/8 integer units away from the
// decoder's float law per 32-value group; 1.13 plus float-accumulation slack is provable.
bool within_bound(float actual,const Oracle& decoder) {
    const auto difference=std::abs(static_cast<long double>(actual)-decoder.value);
    return std::isfinite(actual)
        && difference<=decoder.bound*1.13L+std::max<long double>(0.02L,decoder.magnitude*2e-5L);
}
std::vector<float> source(uint64_t values,uint32_t seed) {
    std::vector<float> out(values);for(uint64_t i=0;i<values;++i) out[i]=float(int((i*23+seed)%127)-63)*.03125f+.0140625f;return out;
}
uint32_t cases_count=0;
void cases(uint32_t columns,uint32_t batch) {
    constexpr uint32_t rows=3;std::vector<Row> logical_rows;
    for(uint32_t row=0;row<rows;++row) logical_rows.push_back(x_logical(columns,row));
    const auto bytes=x_encode(logical_rows);
    validate_packed_rows(bytes,22,columns,rows);
    const auto values=source(uint64_t(batch)*columns,batch);
    const auto q=quantize_q8_activation(values,columns);
    const auto actual=tpc(bytes,22,columns,rows,q,batch,PackedActivation::Q8_1);
    const auto routed=tpc(bytes,22,columns,rows,q,batch,PackedActivation::RoutedQ8);
    for(uint32_t token=0;token<batch;++token) for(uint32_t row=0;row<rows;++row) {
        const auto index=uint64_t(token)*rows+row;
        require(close_exact(actual[index],x_oracle(logical_rows[row],q,token).value),
            "IQ2_S TPC differs from the independent CUDA integer-form oracle");
        require(within_bound(actual[index],x_decoded(bytes,columns,row,q,token)),
            "IQ2_S TPC differs from the pinned decoder by more than the derived truncation bound");
        require(routed[index]==actual[index],"IQ2_S dot differs between the routed and shared/head q8_1 contracts");
        std::vector<float> one(values.begin()+uint64_t(token)*columns,values.begin()+uint64_t(token)*columns+columns);
        const auto isolated=tpc(bytes,22,columns,rows,quantize_q8_activation(one,columns),1,PackedActivation::Q8_1);
        require(isolated[row]==actual[index],"batched IQ2_S changed single-row arithmetic or crossed tokens");
        ++cases_count;
    }
}
void counterexamples() {
    constexpr uint32_t columns=512;
    const auto w=x_logical(columns,0);const auto bytes=x_encode({w});
    const auto values=source(columns,7);const auto q=quantize_q8_activation(values,columns);
    const auto before=tpc(bytes,22,columns,1,q,1,PackedActivation::Q8_1)[0];
    require(close_exact(before,x_oracle(w,q,0).value) && within_bound(before,x_decoded(bytes,columns,0,q,0)),"IQ2_S baseline disagrees with its authorities");
    auto raw=q;for(uint32_t block=0;block<columns/32;++block) const_cast<Q8Activation&>(raw).metadata[block*2+1]=1234.5f;
    require(tpc(bytes,22,columns,1,raw,1,PackedActivation::Q8_1)[0]==before,"IQ2_S output moved when only raw sums changed");
    uint32_t moved=0;
    for(uint32_t v=0;v<columns && moved<4;++v) {
        if(!signed_code(q.codes[v]) || !q.metadata[(v/32)*2]) continue;
        ++moved;
        auto altered=w;const uint32_t sb=v/super,code=(v%super)/lane_values;
        store_index(altered,sb,code,index_of(altered,sb,code)+1);
        const auto altered_bytes=x_encode({altered});validate_packed_rows(altered_bytes,22,columns,1);
        const auto result=tpc(altered_bytes,22,columns,1,q,1,PackedActivation::Q8_1)[0];
        require(result!=before,"an IQ2_S grid index change was absorbed by the group scale");
        require(close_exact(result,x_oracle(altered,q,0).value) && within_bound(result,x_decoded(altered_bytes,columns,0,q,0)),
            "the changed IQ2_S index lost its oracle or decoder match");
    }
    require(moved==4,"too few observable IQ2_S lanes to pin a single grid index");
    // The 9th and 10th index bits live in a separate qh field: flipping both of one lane's
    // bits selects a different grid entry, so it must stay on the independent oracle and move
    // the dot somewhere.
    uint32_t lifted_lanes=0;
    for(uint32_t code=0;code<codes && lifted_lanes<2;++code) {
        auto lifted=w;lifted.index[code]^=uint16_t(0x300u);
        const auto lifted_bytes=x_encode({lifted});validate_packed_rows(lifted_bytes,22,columns,1);
        const auto result=tpc(lifted_bytes,22,columns,1,q,1,PackedActivation::Q8_1)[0];
        require(close_exact(result,x_oracle(lifted,q,0).value),"flipping an IQ2_S qh bit pair left the independent oracle");
        require(within_bound(result,x_decoded(lifted_bytes,columns,0,q,0)),"flipping an IQ2_S qh bit pair left the pinned decoder band");
        if(result!=before) ++lifted_lanes;
    }
    require(lifted_lanes>=1,"flipping IQ2_S qh bit pairs changed nothing, so the high index bits are unread");
    // Signs are per value and plain: flipping one bit must move the dot and stay authority-consistent.
    auto flipped=w;flipped.signs[0]^=1u;
    const auto flipped_bytes=x_encode({flipped});validate_packed_rows(flipped_bytes,22,columns,1);
    const auto flipped_result=tpc(flipped_bytes,22,columns,1,q,1,PackedActivation::Q8_1)[0];
    require(flipped_result!=before && close_exact(flipped_result,x_oracle(flipped,q,0).value)
        && within_bound(flipped_result,x_decoded(flipped_bytes,columns,0,q,0)),
        "flipping an IQ2_S sign bit did nothing or lost an authority");
    // Parity is NOT folded: two sign bytes with the same popcount parity must differ.
    auto odd=w;odd.signs[0]=0x01u;auto other=w;other.signs[0]=0x02u;
    require(tpc(x_encode({odd}),22,columns,1,q,1,PackedActivation::Q8_1)[0]!=tpc(x_encode({other}),22,columns,1,q,1,PackedActivation::Q8_1)[0],
        "IQ2_S signs behaved like a parity fold, so the plain bit field was misread");
    // The two nibbles of one scale byte are independent: change only the second half.
    auto high=w;high.scale[0]=uint8_t(high.scale[0]^0x10u);
    const auto high_bytes=x_encode({high});validate_packed_rows(high_bytes,22,columns,1);
    require(close_exact(tpc(high_bytes,22,columns,1,q,1,PackedActivation::Q8_1)[0],x_oracle(high,q,0).value),
        "the IQ2_S high nibble scale did not follow the independent oracle");
    // Zeroing every scale leaves only the (sumi0+sumi1)/2 half-offset term, which is a
    // different truncating integer from the scaled dot.
    auto zeroed=w;std::fill(zeroed.scale.begin(),zeroed.scale.end(),uint8_t(0));
    const auto zeroed_bytes=x_encode({zeroed});validate_packed_rows(zeroed_bytes,22,columns,1);
    const auto zeroed_result=tpc(zeroed_bytes,22,columns,1,q,1,PackedActivation::Q8_1)[0];
    require(zeroed_result!=before && close_exact(zeroed_result,x_oracle(zeroed,q,0).value)
        && within_bound(zeroed_result,x_decoded(zeroed_bytes,columns,0,q,0)),
        "zeroing the IQ2_S scales left the dot, the oracle or the decoder inconsistent");
    auto negated=w;for(auto& value:negated.d) value=-value;
    const auto neg_bytes=x_encode({negated});validate_packed_rows(neg_bytes,22,columns,1);
    require(tpc(neg_bytes,22,columns,1,q,1,PackedActivation::Q8_1)[0]==-before,"negating the IQ2_S super-block d did not negate the dot");
}
// The packed type set is asserted as a whole: adding a format cannot leave a stale
// negative assertion behind in an older test.
void policy_types() {
    for(uint32_t type:{2,3,6,7,8,11,12,13,14,16,17,18,20,21,22,23,29,40,42}) require(packed_supported(type),"a packed GGML type with a working dot was rejected");
    for(uint32_t type:{19,30,31}) require(!packed_supported(type),"a still-decoded GGML type was accepted as packed");
}
void guards() {
    policy_types();
    const auto w=x_logical(256,0);auto bytes=x_encode({w});
    bytes[0]=0;bytes[1]=0x7c;rejects([&]{validate_packed_rows(bytes,22,256,1);});
    bytes[0]=0;bytes[1]=0x3c;validate_packed_rows(bytes,22,256,1);
    for(uint32_t at:{qs_at,signs_at,qh_at,scales_at}) bytes[at]=0xff;
    validate_packed_rows(bytes,22,256,1);
    rejects([&]{(void)packed_row_bytes(22,640);});rejects([&]{(void)packed_row_bytes(22,300);});
    require(packed_supported(22) && packed_values_per_block(22)==super && packed_block_bytes(22)==xs_bytes
        && packed_row_bytes(22,2560)==820 && packed_row_bytes(22,256)==xs_bytes,"IQ2_S geometry wrong");
    require(packed_activation_supported(22,PackedActivation::Q8_1) && packed_activation_supported(22,PackedActivation::RoutedQ8)
        && !packed_activation_supported(22,PackedActivation::RoutedS2),"IQ2_S policy wrong");
    const auto routed=expert_format(2560,640,0,{22,22,8});
    require(routed.packed && routed.activation==PackedActivation::RoutedQ8
        && routed.bytes==uint64_t(820)*640*2+uint64_t(34)*20*2560,"routed IQ2_S expert charge wrong");
    const auto shared=expert_format(2560,640,-1,{22,21,8});
    require(shared.packed && shared.activation==PackedActivation::Q8_1
        && shared.bytes==uint64_t(820)*640+uint64_t(1100)*640+uint64_t(34)*20*2560,"shared IQ2_S expert was not packed or is mischarged");
    const auto down=expert_format(2560,640,0,{8,8,22});
    require(!down.packed && down.bytes==uint64_t(2560)*640*6,"IQ2_S down projection with 640 columns was not rejected by layout");
}
struct Flow {
    uint32_t type,columns,rows;
    std::vector<uint8_t> bytes;
    std::vector<Row> xs;
    std::vector<int> q8_codes;
    std::vector<float> q8_d;
};
Flow flow_role(uint32_t type,uint32_t columns,uint32_t rows,uint32_t seed) {
    Flow r{type,columns,rows,{},{},{},{}};
    if(type==8) {
        r.q8_codes.resize(uint64_t(rows)*columns);r.q8_d.resize(uint64_t(rows)*columns/32);
        r.bytes.assign(uint64_t(rows)*(columns/32)*q8_bytes,0);
        for(uint32_t row=0;row<rows;++row) for(uint32_t b=0;b<columns/32;++b) {
            const size_t at=(uint64_t(row)*columns/32+b)*q8_bytes;
            const float d=float(int((b*3+row*5+seed)%9)-4)/16.f;
            const auto bits=strata::kernels::f16_from_f32(d);
            r.bytes[at]=uint8_t(bits);r.bytes[at+1]=uint8_t(bits>>8);r.q8_d[uint64_t(row)*columns/32+b]=d;
            for(uint32_t j=0;j<32;++j) {
                const int code=int((b*32+j)*13+row*11+seed*7+3)%249-124;
                r.q8_codes[uint64_t(row)*columns+b*32+j]=code;r.bytes[at+2+j]=uint8_t(code);
            }
        }
        validate_packed_rows(r.bytes,8,columns,rows);return r;
    }
    for(uint32_t row=0;row<rows;++row) r.xs.push_back(x_logical(columns,row+seed*1000));
    r.bytes=x_encode(r.xs);validate_packed_rows(r.bytes,22,columns,rows);return r;
}
std::vector<Oracle> flow_oracle(const Flow& r,const Q8Activation& q,uint32_t batch) {
    std::vector<Oracle> out(uint64_t(r.rows)*batch);
    for(uint32_t token=0;token<batch;++token) for(uint32_t row=0;row<r.rows;++row) {
        if(r.type==8) {
            long double total=0,magnitude=0;
            for(uint32_t b=0;b<r.columns/32;++b) {
                long double dot=0;
                for(uint32_t j=0;j<32;++j) dot+=static_cast<long double>(r.q8_codes[uint64_t(row)*r.columns+b*32+j])*signed_code(q.codes[uint64_t(token)*r.columns+b*32+j]);
                const long double term=r.q8_d[uint64_t(row)*r.columns/32+b]*q.metadata[(uint64_t(token)*r.columns/32+b)*2]*dot;
                total+=term;magnitude+=std::abs(term);
            }
            out[uint64_t(token)*r.rows+row]=Oracle{total,magnitude,0};continue;
        }
        out[uint64_t(token)*r.rows+row]=x_oracle(r.xs[row],q,token);
    }
    return out;
}
uint32_t flows=0;
void shared_flows() {
    constexpr uint32_t hidden=512,intermediate=256;
    // These rows are IQ2_S-shaped, so only IQ2_S and Q8_0 roles can appear here; experts that
    // mix several packed formats are exercised by tests/packed_expert.cpp's real layout source.
    for(auto types:{std::array<uint32_t,3>{22,22,22},std::array<uint32_t,3>{22,22,8},std::array<uint32_t,3>{22,8,22},std::array<uint32_t,3>{8,22,22}})
    for(uint32_t batch:{1u,3u}) {
        const auto format=expert_format(hidden,intermediate,-1,types);
        require(format.packed && format.activation==PackedActivation::Q8_1,"shared IQ2_S expert lost the head contract");
        std::array<Flow,3> role{flow_role(types[0],hidden,intermediate,0),flow_role(types[1],hidden,intermediate,1),flow_role(types[2],intermediate,hidden,2)};
        const uint64_t charge=packed_row_bytes(types[0],hidden)*intermediate+packed_row_bytes(types[1],hidden)*intermediate
            +packed_row_bytes(types[2],intermediate)*hidden;
        require(format.bytes==charge,"shared IQ2_S expert payload mischarged");
        std::vector<float> input(uint64_t(batch)*hidden);
        for(size_t i=0;i<input.size();++i) input[i]=float(int((i*29+batch)%113)-56)*.0625f;
        std::array<uint32_t,3> calls{0,0,0};
        auto run=[&](const std::array<Flow,3>& roles)->std::vector<float> {
            calls={0,0,0};
            return execute_packed_expert(std::span<const float>(input),hidden,intermediate,[&](uint32_t which,std::span<const float> values) {
                ++calls[which];
                const auto mode=expert_projection_activation(format,which);
                require(mode==PackedActivation::Q8_1,"shared IQ2_S projection inherited a routed activation contract");
                const auto tokens=uint32_t(values.size()/roles[which].columns);
                const auto q=quantize_q8_activation(values,roles[which].columns);
                const auto expected=flow_oracle(roles[which],q,tokens);
                const auto actual=tpc(roles[which].bytes,roles[which].type,roles[which].columns,roles[which].rows,q,tokens,mode);
                for(size_t i=0;i<actual.size();++i) {
                    require(std::isfinite(actual[i]),"non-finite shared IQ2_S projection");
                    // A flow feeds the down projection far larger intermediates than the fixed
                    // inputs of cases(), so the band must follow the size of the summed terms,
                    // not the size of the cancelled result.
                    const auto tolerance=std::max({.02,std::abs(static_cast<double>(expected[i].value))*2e-3,
                        static_cast<double>(expected[i].magnitude)*8e-7});
                    require(std::abs(double(actual[i])-double(expected[i].value))<=tolerance,
                        "shared expert IQ2_S projection differs from the independent oracle");
                    if(roles[which].type==22) {
                        const uint32_t row=uint32_t(i%roles[which].rows),token=uint32_t(i/roles[which].rows);
                        require(std::abs(double(actual[i])-double(x_accumulated(roles[which].xs[row],q,token)))
                            <=std::max(1e-6,double(std::abs(actual[i]))*1e-6),
                            "the shared IQ2_S projection did not follow the documented float32 term order");
                    }
                }
                return actual;
            });
        };
        const auto output=run(role);
        require(calls==std::array<uint32_t,3>{1,1,1},"shared IQ2_S expert flow did not run each projection once");
        require(output.size()==input.size(),"shared IQ2_S expert flow output shape");
        for(float value:output) require(std::isfinite(value),"non-finite shared IQ2_S expert output");
        if(types[0]==types[1] && types[0]==22) {
            auto swapped=role;std::swap(swapped[0],swapped[1]);
            const auto other=run(swapped);bool differs=false;
            for(size_t i=0;i<output.size();++i) differs=differs||output[i]!=other[i];
            require(differs,"exchanging gate and up IQ2_S weights left the shared expert output unchanged");
        }
        ++flows;
    }
}
}
int main(){try {
    pin_grid();sign_rule();
    for(uint32_t columns:{256u,512u,2560u}) for(uint32_t batch:{1u,3u,8u}) cases(columns,batch);
    counterexamples();guards();shared_flows();
    std::cout<<"PASS IQ2_S TPC "<<cases_count<<" cases against the independent CUDA integer-form oracle ((sumi0*ls0+sumi1*ls1+(sumi0+sumi1)/2)/4) AND the unmodified pinned ggml IQ2_S row decoder inside the derived 1+1/8-unit truncation band, "
        <<grid_pins<<" 10-bit grid magnitudes recovered through the decoder and shown distinguishable including every neighbour pair, "
        <<table_bytes<<" codebook bytes compiled into the kernel matched against those decoder readings, "
        <<sign_checks<<" plain-sign checks, "<<flows<<" production shared-expert flows, index/high-bit/sign/nibble-independence counterexamples and policy guards; no device\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
