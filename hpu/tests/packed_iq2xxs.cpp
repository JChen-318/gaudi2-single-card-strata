// Native IQ2_XXS (GGML type 16) 256-value codebook superblock dot for the shared/head and
// routed q8_1 contracts. IQ2_XXS needs two authorities that disagree by construction: the
// pinned CUDA dot applies its group scale as the INTEGER sumi*(2*sc+1)/8 (iq_kernels.cu:98)
// while ggml's row decoder multiplies the float (0.5+sc)*0.25. So the kernel is compared
// EXACTLY against an independent transcription of the CUDA integer formula and against the
// pinned decoder within the bound that the truncation provably introduces - never against a
// loosened tolerance. The grid magnitudes themselves are recovered through the decoder, so
// this test never trusts the table the kernel was compiled with. Host only: no ISA, no
// device, no performance claim.
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
template<class F> void rejects(F f){bool bad=false;try{f();}catch(const std::exception&){bad=true;}require(bad,"invalid IQ2_XXS payload or policy accepted");}
int signed_code(uint8_t b){return b<128?b:int(b)-256;}
constexpr uint32_t super=256,groups=8,lane_values=8,xs_bytes=66,q8_bytes=34;
uint32_t blocks(uint32_t columns){return columns/super;}
struct Oracle { long double value,magnitude,bound; };
const ggml_type_traits* traits() {
    const auto* t=ggml_get_type_traits(GGML_TYPE_IQ2_XXS);
    require(t && t->blck_size==int(super) && t->type_size==xs_bytes && t->to_float,"pinned ggml lacks the IQ2_XXS row decoder or its block size differs");
    return t;
}
std::vector<float> decode_row(const std::vector<uint8_t>& bytes,uint32_t columns,uint32_t row) {
    std::vector<float> weights(columns);
    traits()->to_float(bytes.data()+uint64_t(row)*packed_row_bytes(16,columns),weights.data(),int64_t(columns));
    return weights;
}
// The eight magnitudes of one grid index, read back through the decoder: d=1, scale 0 so
// db = 1*(0.5+0)*0.25 = 0.125, signs clear, and lane 0 of group 0 carries the index under
// test. Nothing here consults the table baked into the kernel.
std::array<int,8> magnitudes(uint32_t index) {
    std::vector<uint8_t> bytes(xs_bytes,0);
    bytes[0]=0x00;bytes[1]=0x3c;                 // half 1.0 at offset 0
    bytes[2]=uint8_t(index);                     // group 0, lane 0
    const auto weights=decode_row(bytes,super,0);
    std::array<int,8> out{};
    for(uint32_t j=0;j<lane_values;++j) {
        require(std::abs(double(weights[j]*8.0f)-std::round(double(weights[j]*8.0f)))<1e-3,"IQ2_XXS grid magnitude is not integral");
        out[j]=int(std::round(double(weights[j]*8.0f)));
    }
    return out;
}
// The decoder-derived table, built once: the kernel carries its own compiled copy, so the
// two agreeing on all 2048 magnitudes is a real cross-authority check, not a restatement.
const std::array<std::array<int,8>,256>& grid_table() {
    static const std::array<std::array<int,8>,256> table=[]{
        std::array<std::array<int,8>,256> built{};
        for(uint32_t index=0;index<256;++index) built[index]=magnitudes(index);
        return built;
    }();
    return table;
}
uint32_t grid_pins=0;
void pin_grid() {
    std::vector<std::array<int,8>> seen(256);
    for(uint32_t index=0;index<256;++index) {
        const auto m=grid_table()[index];
        require(std::all_of(m.begin(),m.end(),[](int value){return value==8||value==25||value==43;}),"an IQ2_XXS grid byte is not one of the pinned magnitudes");
        seen[index]=m;
        ++grid_pins;
    }
    // Two distinct indices must not be indistinguishable in every lane, or a wrong table
    // lookup could never show up in a value.
    uint32_t collisions=0;
    for(uint32_t a=0;a<256;++a) for(uint32_t b=a+1;b<256;++b) if(seen[a]==seen[b]) ++collisions;
    require(collisions==0,"IQ2_XXS grid entries are not distinguishable through the decoder");
}
uint32_t sign_rule_checks=0;
void sign_rule() {
    // Verify the parity-fold rule the kernel uses against the decoder for every 7-bit
    // pattern: value j is negative exactly when bits ^ ((popcount(bits)&1)<<7) has bit j.
    for(uint32_t bits=0;bits<128;++bits) {
        uint32_t parity=0;for(uint32_t bit=0;bit<7;++bit) if(bits&(1u<<bit)) ++parity;
        const uint32_t pattern=bits^((parity&1u)<<7);
        std::vector<uint8_t> bytes(xs_bytes,0);
        bytes[0]=0x00;bytes[1]=0x3c;bytes[2]=0x01;   // lane 0 grid index 1: 43,8,8,8,8,8,8,8
        bytes[6]=uint8_t(bits);                       // group 0 sign/scale word, byte 0 = bits 0..6
        const auto weights=decode_row(bytes,super,0);
        const auto m=grid_table()[1];
        for(uint32_t j=0;j<lane_values;++j) {
            const float expected=0.125f*float(m[j])*((pattern&(1u<<j))?-1.0f:1.0f);
            require(std::abs(double(weights[j])-double(expected))<1e-4,"the IQ2_XXS sign parity rule does not match the decoder");
            ++sign_rule_checks;
        }
    }
}
struct XRow {
    uint32_t columns;
    std::vector<uint8_t> index;                     // [blocks][groups][4]
    std::vector<uint8_t> bits;                      // [blocks][groups][4] seven-bit sign field
    std::vector<uint8_t> scale;                     // [blocks][groups] 4-bit scale
    std::vector<float> d;                           // [blocks] signed
};
XRow x_logical(uint32_t columns,uint32_t row) {
    const auto count=blocks(columns);
    XRow w{columns,std::vector<uint8_t>(uint64_t(count)*groups*4),std::vector<uint8_t>(uint64_t(count)*groups*4),
        std::vector<uint8_t>(uint64_t(count)*groups),std::vector<float>(count)};
    for(uint32_t sb=0;sb<count;++sb) {
        w.d[sb]=float(int((sb*7+row*3+4)%11)-5)/8.f;                                     // both signs
        for(uint32_t ib=0;ib<groups;++ib) {
            w.scale[sb*groups+ib]=uint8_t((sb*5+ib*3+row)%16);
            for(uint32_t lane=0;lane<4;++lane) {
                const size_t at=(uint64_t(sb)*groups+ib)*4+lane;
                w.index[at]=uint8_t((at*37+sb*11+row*5)%256);
                w.bits[at]=uint8_t((at*13+sb*7+row)%128);
            }
        }
    }
    return w;
}
// The measured map: half d at 0, then eight 8-byte groups at 2+8*ib whose first four bytes
// are grid indices and whose last four are the little-endian sign/scale word.
std::vector<uint8_t> x_encode(const std::vector<XRow>& rows) {
    const auto columns=rows.front().columns;
    std::vector<uint8_t> bytes(uint64_t(rows.size())*blocks(columns)*xs_bytes,0);
    const auto put=[&](size_t at,float value){const auto bits=strata::kernels::f16_from_f32(value);bytes[at]=uint8_t(bits);bytes[at+1]=uint8_t(bits>>8);};
    for(size_t row=0;row<rows.size();++row) {const auto& w=rows[row];
        for(uint32_t sb=0;sb<blocks(columns);++sb) {
            const size_t at=(uint64_t(row)*blocks(columns)+sb)*xs_bytes;
            for(uint32_t ib=0;ib<groups;++ib) {
                const size_t word_at=at+2+ib*4*2;
                uint32_t word=uint32_t(w.scale[sb*groups+ib])<<28;
                for(uint32_t lane=0;lane<4;++lane) {
                    const size_t index_at=(uint64_t(sb)*groups+ib)*4+lane;
                    bytes[word_at+lane]=w.index[index_at];
                    word|=uint32_t(w.bits[index_at]&127u)<<(7*lane);
                }
                bytes[word_at+4]=uint8_t(word);bytes[word_at+5]=uint8_t(word>>8);
                bytes[word_at+6]=uint8_t(word>>16);bytes[word_at+7]=uint8_t(word>>24);
            }
            put(at,w.d[sb]);
        }
    }
    return bytes;
}
uint32_t popcount7(uint32_t value) {uint32_t count=0;for(uint32_t bit=0;bit<7;++bit) if(value&(1u<<bit)) ++count;return count;}
// Independent transcription of the pinned CUDA integer form.
Oracle x_oracle(const XRow& w,const Q8Activation& q,uint32_t token) {
    Oracle out{0,0,0};
    for(uint32_t sb=0;sb<blocks(w.columns);++sb) for(uint32_t ib=0;ib<groups;++ib) {
        int sumi=0;
        for(uint32_t lane=0;lane<4;++lane) {
            const size_t at=(uint64_t(sb)*groups+ib)*4+lane;
            const auto m=grid_table()[w.index[at]];
            const uint32_t pattern=w.bits[at]&127u;
            const uint32_t parity=popcount7(pattern);
            const uint32_t signs=pattern^((parity&1u)<<7);
            for(uint32_t j=0;j<lane_values;++j)
                sumi+=((signs&(1u<<j))?-m[j]:m[j])*signed_code(q.codes[uint64_t(token)*w.columns+sb*super+ib*lane_values*4+lane*lane_values+j]);
        }
        const int ls=2*int(w.scale[sb*groups+ib])+1;
        const int scaled=sumi*ls/8;                                   // C truncation, as upstream
        const double factor=double(w.d[sb])*q.metadata[(uint64_t(token)*w.columns/32+sb*groups+ib)*2];
        out.value+=static_cast<long double>(factor)*scaled;
        out.magnitude+=std::abs(static_cast<long double>(factor)*scaled);
        out.bound+=std::abs(static_cast<long double>(factor));        // at most one integer unit per group
    }
    return out;
}
Oracle x_decoded(const std::vector<uint8_t>& bytes,uint32_t columns,uint32_t row,const Q8Activation& q,uint32_t token) {
    const auto weights=decode_row(bytes,columns,row);
    Oracle out{0,0,0};
    for(uint32_t block=0;block<columns/32;++block) {
        const long double d8=q.metadata[(uint64_t(token)*columns/32+block)*2];long double part=0;
        for(uint32_t v=0;v<32;++v) part+=static_cast<long double>(weights[block*32+v])*signed_code(q.codes[uint64_t(token)*columns+block*32+v]);
        const long double term=d8*part;out.value+=term;out.magnitude+=std::abs(term);
        // At most one integer unit of (d * d8) is lost to the CUDA truncation per group.
        const size_t row_base=uint64_t(row)*packed_row_bytes(16,columns);
        const size_t block_at=row_base+uint64_t(block/8)*xs_bytes;
        const float d=strata::kernels::f32_from_f16(uint16_t(bytes[block_at])|(uint16_t(bytes[block_at+1])<<8));
        out.bound+=std::abs(static_cast<long double>(d)*d8);    }
    return out;
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
bool within_bound(float actual,const Oracle& decoder) {
    const auto difference=std::abs(static_cast<long double>(actual)-decoder.value);
    return std::isfinite(actual) && difference<=decoder.bound*1.02L+0.02L;
}
std::vector<float> source(uint64_t values,uint32_t seed) {
    std::vector<float> out(values);for(uint64_t i=0;i<values;++i) out[i]=float(int((i*23+seed)%127)-63)*.03125f+.0140625f;return out;
}
uint32_t cases_count=0;
void cases(uint32_t columns,uint32_t batch) {
    constexpr uint32_t rows=3;std::vector<XRow> logical_rows;
    for(uint32_t row=0;row<rows;++row) logical_rows.push_back(x_logical(columns,row));
    const auto bytes=x_encode(logical_rows);
    validate_packed_rows(bytes,16,columns,rows);
    const auto values=source(uint64_t(batch)*columns,batch);
    const auto q=quantize_q8_activation(values,columns);
    const auto actual=tpc(bytes,16,columns,rows,q,batch,PackedActivation::Q8_1);
    const auto routed=tpc(bytes,16,columns,rows,q,batch,PackedActivation::RoutedQ8);
    for(uint32_t token=0;token<batch;++token) for(uint32_t row=0;row<rows;++row) {
        const auto index=uint64_t(token)*rows+row;
        require(close_exact(actual[index],x_oracle(logical_rows[row],q,token).value),
            "IQ2_XXS TPC differs from the independent CUDA integer-form oracle");
        require(within_bound(actual[index],x_decoded(bytes,columns,row,q,token)),
            "IQ2_XXS TPC differs from the pinned decoder by more than the derived truncation bound");
        require(routed[index]==actual[index],"IQ2_XXS dot differs between the routed and shared/head q8_1 contracts");
        std::vector<float> one(values.begin()+uint64_t(token)*columns,values.begin()+uint64_t(token)*columns+columns);
        const auto isolated=tpc(bytes,16,columns,rows,quantize_q8_activation(one,columns),1,PackedActivation::Q8_1);
        require(isolated[row]==actual[index],"batched IQ2_XXS changed single-row arithmetic or crossed tokens");
        ++cases_count;
    }
}
void counterexamples() {
    constexpr uint32_t columns=512;
    const auto w=x_logical(columns,0);const auto bytes=x_encode({w});
    const auto values=source(columns,7);const auto q=quantize_q8_activation(values,columns);
    const auto before=tpc(bytes,16,columns,1,q,1,PackedActivation::Q8_1)[0];
    require(close_exact(before,x_oracle(w,q,0).value) && within_bound(before,x_decoded(bytes,columns,0,q,0)),"IQ2_XXS baseline disagrees with its authorities");
    auto raw=q;for(uint32_t block=0;block<columns/32;++block) const_cast<Q8Activation&>(raw).metadata[block*2+1]=1234.5f;
    require(tpc(bytes,16,columns,1,raw,1,PackedActivation::Q8_1)[0]==before,"IQ2_XXS output moved when only raw sums changed");
    // A single grid index change must be visible wherever the lane is observable.
    uint32_t moved=0;
    for(uint32_t v=0;v<columns && moved<4;++v) {
        if(!signed_code(q.codes[v]) || !q.metadata[(v/32)*2]) continue;
        ++moved;
        auto altered=w;
        const size_t slot=((uint64_t(v/super)*groups)+((v%super)/32))*4+((v%32)/lane_values);
        altered.index[slot]=uint8_t((altered.index[slot]+1)%256);
        const auto altered_bytes=x_encode({altered});validate_packed_rows(altered_bytes,16,columns,1);
        const auto result=tpc(altered_bytes,16,columns,1,q,1,PackedActivation::Q8_1)[0];
        require(result!=before,"an IQ2_XXS grid index change was absorbed by the group scale");
        require(close_exact(result,x_oracle(altered,q,0).value) && within_bound(result,x_decoded(altered_bytes,columns,0,q,0)),
            "the changed IQ2_XXS index lost its oracle or decoder match");
    }
    require(moved==4,"too few observable IQ2_XXS lanes to pin a single grid index");
    // The parity fold is real data, not decoration: a pattern whose popcount is odd must
    // negate lane value 7 as well, which a plain bit test would miss.
    auto parity=w;const size_t slot=(uint64_t(0)*groups+0)*4+0;
    parity.bits[slot]=uint8_t(parity.bits[slot]|(parity.bits[slot]&1?0u:1u));
    const auto parity_bytes=x_encode({parity});validate_packed_rows(parity_bytes,16,columns,1);
    require(close_exact(tpc(parity_bytes,16,columns,1,q,1,PackedActivation::Q8_1)[0],x_oracle(parity,q,0).value),
        "an odd-popcount IQ2_XXS sign field stopped matching the parity rule");
    // Clearing every sign field must leave all values positive and decoder-consistent.
    auto cleared=w;std::fill(cleared.bits.begin(),cleared.bits.end(),uint8_t(0));
    const auto cleared_bytes=x_encode({cleared});validate_packed_rows(cleared_bytes,16,columns,1);
    require(cleared_bytes!=bytes && close_exact(tpc(cleared_bytes,16,columns,1,q,1,PackedActivation::Q8_1)[0],x_oracle(cleared,q,0).value),
        "clearing the IQ2_XXS sign fields changed nothing or lost the oracle");
    // Scale 0 and scale 15 must differ, and negating d negates the dot.
    auto minimal=w;std::fill(minimal.scale.begin(),minimal.scale.end(),uint8_t(0));
    const auto minimal_bytes=x_encode({minimal});validate_packed_rows(minimal_bytes,16,columns,1);
    require(tpc(minimal_bytes,16,columns,1,q,1,PackedActivation::Q8_1)[0]!=before,"flattening the IQ2_XXS group scales did not change the dot");
    auto negated=w;for(auto& value:negated.d) value=-value;
    const auto neg_bytes=x_encode({negated});validate_packed_rows(neg_bytes,16,columns,1);
    require(tpc(neg_bytes,16,columns,1,q,1,PackedActivation::Q8_1)[0]==-before,"negating the IQ2_XXS super-block d did not negate the dot");
}
// The packed type set is asserted as a whole: adding a format cannot leave a stale
// negative assertion behind in an older test. The remaining IQ families and BF16 must stay
// on the decoded side until each has its own dot and dual-authority test.
void policy_types() {
    for(uint32_t type:{2,3,6,7,8,11,12,13,14,16,17,18,20,21,22,23,29,40,42}) require(packed_supported(type),"a packed GGML type with a working dot was rejected");
    for(uint32_t type:{19,30,31}) require(!packed_supported(type),"a still-decoded GGML type was accepted as packed");
}
void guards() {
    policy_types();
    const auto w=x_logical(256,0);auto bytes=x_encode({w});
    bytes[0]=0;bytes[1]=0x7c;rejects([&]{validate_packed_rows(bytes,16,256,1);});
    bytes[0]=0;bytes[1]=0x3c;validate_packed_rows(bytes,16,256,1);
    // Grid indices, sign fields and scales are all code data: arbitrary bytes are legal.
    bytes[2]=0xff;bytes[6]=0xff;bytes[9]=0xff;validate_packed_rows(bytes,16,256,1);
    rejects([&]{(void)packed_row_bytes(16,640);});rejects([&]{(void)packed_row_bytes(16,300);});
    require(packed_supported(16) && packed_values_per_block(16)==super && packed_block_bytes(16)==xs_bytes
        && packed_row_bytes(16,2560)==660 && packed_row_bytes(16,256)==xs_bytes,"IQ2_XXS geometry wrong");
    require(packed_activation_supported(16,PackedActivation::Q8_1) && packed_activation_supported(16,PackedActivation::RoutedQ8)
        && !packed_activation_supported(16,PackedActivation::RoutedS2),"IQ2_XXS policy wrong");
    const auto routed=expert_format(2560,640,0,{16,16,8});
    require(routed.packed && routed.activation==PackedActivation::RoutedQ8
        && routed.bytes==uint64_t(660)*640*2+uint64_t(34)*20*2560,"routed IQ2_XXS expert charge wrong");
    const auto shared=expert_format(2560,640,-1,{16,23,8});
    require(shared.packed && shared.activation==PackedActivation::Q8_1
        && shared.bytes==uint64_t(660)*640+uint64_t(1360)*640+uint64_t(34)*20*2560,"shared IQ2_XXS expert was not packed or is mischarged");
    const auto down=expert_format(2560,640,0,{8,8,16});
    require(!down.packed && down.bytes==uint64_t(2560)*640*6,"IQ2_XXS down projection with 640 columns was not rejected by layout");
}
struct Flow {
    uint32_t type,columns,rows;
    std::vector<uint8_t> bytes;
    std::vector<XRow> xs;
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
    r.bytes=x_encode(r.xs);validate_packed_rows(r.bytes,16,columns,rows);return r;
}
std::vector<float> flow_oracle(const Flow& r,const Q8Activation& q,uint32_t batch) {
    std::vector<float> out(uint64_t(r.rows)*batch);
    for(uint32_t token=0;token<batch;++token) for(uint32_t row=0;row<r.rows;++row) {
        if(r.type==8) {
            long double total=0;
            for(uint32_t b=0;b<r.columns/32;++b) {
                long double dot=0;
                for(uint32_t j=0;j<32;++j) dot+=static_cast<long double>(r.q8_codes[uint64_t(row)*r.columns+b*32+j])*signed_code(q.codes[uint64_t(token)*r.columns+b*32+j]);
                total+=r.q8_d[uint64_t(row)*r.columns/32+b]*q.metadata[(uint64_t(token)*r.columns/32+b)*2]*dot;
            }
            out[uint64_t(token)*r.rows+row]=float(total);continue;
        }
        out[uint64_t(token)*r.rows+row]=float(x_oracle(r.xs[row],q,token).value);
    }
    return out;
}
uint32_t flows=0;
void shared_flows() {
    constexpr uint32_t hidden=512,intermediate=256;
    for(auto types:{std::array<uint32_t,3>{16,16,16},std::array<uint32_t,3>{16,16,8},std::array<uint32_t,3>{16,8,16},std::array<uint32_t,3>{8,16,16}})
    for(uint32_t batch:{1u,3u}) {
        const auto format=expert_format(hidden,intermediate,-1,types);
        require(format.packed && format.activation==PackedActivation::Q8_1,"shared IQ2_XXS expert lost the head contract");
        std::array<Flow,3> role{flow_role(types[0],hidden,intermediate,0),flow_role(types[1],hidden,intermediate,1),flow_role(types[2],intermediate,hidden,2)};
        const uint64_t charge=packed_row_bytes(types[0],hidden)*intermediate+packed_row_bytes(types[1],hidden)*intermediate
            +packed_row_bytes(types[2],intermediate)*hidden;
        require(format.bytes==charge,"shared IQ2_XXS expert payload mischarged");
        std::vector<float> input(uint64_t(batch)*hidden);
        for(size_t i=0;i<input.size();++i) input[i]=float(int((i*29+batch)%113)-56)*.0625f;
        std::array<uint32_t,3> calls{0,0,0};
        auto run=[&](const std::array<Flow,3>& roles)->std::vector<float> {
            calls={0,0,0};
            return execute_packed_expert(std::span<const float>(input),hidden,intermediate,[&](uint32_t which,std::span<const float> values) {
                ++calls[which];
                const auto mode=expert_projection_activation(format,which);
                require(mode==PackedActivation::Q8_1,"shared IQ2_XXS projection inherited a routed activation contract");
                const auto tokens=uint32_t(values.size()/roles[which].columns);
                const auto q=quantize_q8_activation(values,roles[which].columns);
                const auto expected=flow_oracle(roles[which],q,tokens);
                const auto actual=tpc(roles[which].bytes,roles[which].type,roles[which].columns,roles[which].rows,q,tokens,mode);
                for(size_t i=0;i<actual.size();++i) {
                    require(std::isfinite(actual[i]),"non-finite shared IQ2_XXS projection");
                    require(std::abs(double(actual[i])-double(expected[i]))<=std::max(.02,double(std::abs(expected[i]))*2e-3),
                        "shared expert IQ2_XXS projection differs from the independent oracle");
                }
                return actual;
            });
        };
        const auto output=run(role);
        require(calls==std::array<uint32_t,3>{1,1,1},"shared IQ2_XXS expert flow did not run each projection once");
        require(output.size()==input.size(),"shared IQ2_XXS expert flow output shape");
        for(float value:output) require(std::isfinite(value),"non-finite shared IQ2_XXS expert output");
        if(types[0]==types[1] && types[0]==16) {
            auto swapped=role;std::swap(swapped[0],swapped[1]);
            const auto other=run(swapped);bool differs=false;
            for(size_t i=0;i<output.size();++i) differs=differs||output[i]!=other[i];
            require(differs,"exchanging gate and up IQ2_XXS weights left the shared expert output unchanged");
        }
        ++flows;
    }
}
}
int main(){try {
    pin_grid();sign_rule();
    for(uint32_t columns:{256u,512u,2560u}) for(uint32_t batch:{1u,3u,8u}) cases(columns,batch);
    counterexamples();guards();shared_flows();
    std::cout<<"PASS IQ2_XXS TPC "<<cases_count<<" cases exact against the pinned CUDA integer form (sumi*(2*sc+1)/8), within the derived one-unit truncation bound of the unmodified pinned ggml decoder, "
        <<grid_pins<<" grid magnitudes recovered through the decoder and shown distinguishable, "<<sign_rule_checks<<" sign-parity checks over all 128 fields, "
        <<flows<<" production shared-expert flows, raw-sum insensitivity and policy guards; no device\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
