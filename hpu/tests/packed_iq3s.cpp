// Native IQ3_S (GGML type 21) 256-value codebook superblock dot for the shared/head and
// routed q8_1 contracts. IQ3_S is the first grid family whose pinned CUDA scale is a plain
// INTEGER multiplier (1 + 2*sc) with no division, so the CUDA form and the pinned ggml row
// decoder agree in value and differ only in float accumulation order; the test therefore
// demands an exact match against an independent transcription of the CUDA form AND a close
// match against the decoder, with no truncation band to widen. All 512 grid entries and the
// plain 8-bit sign fields are recovered through the decoder, so nothing trusts the table
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
template<class F> void rejects(F f){bool bad=false;try{f();}catch(const std::exception&){bad=true;}require(bad,"invalid IQ3_S payload or policy accepted");}
int signed_code(uint8_t b){return b<128?b:int(b)-256;}
constexpr uint32_t super=256,groups=8,codes=64,lanes=4,mags=4,xs_bytes=110,grid_entries=512,q8_bytes=34;
constexpr uint32_t qs_at=2,qh_at=66,signs_at=74,scales_at=106;
uint32_t blocks(uint32_t columns){return columns/super;}
struct Oracle { long double value,magnitude; };
struct Row {
    uint32_t columns;
    std::vector<uint8_t> qs;                     // [blocks][64] low index bits
    std::vector<uint8_t> qh;                     // [blocks][8] one bit per code, the 9th index bit
    std::vector<uint8_t> signs;                  // [blocks][32] plain sign byte per 8-value lane
    std::vector<uint8_t> scale;                  // [blocks][4] two group nibbles per byte
    std::vector<float> d;                        // [blocks] signed
};
const ggml_type_traits* traits() {
    const auto* t=ggml_get_type_traits(GGML_TYPE_IQ3_S);
    require(t && t->blck_size==int(super) && t->type_size==xs_bytes && t->to_float,"pinned ggml lacks the IQ3_S row decoder or its block size differs");
    return t;
}
std::vector<float> decode_row(const std::vector<uint8_t>& bytes,uint32_t columns,uint32_t row) {
    std::vector<float> weights(columns);
    traits()->to_float(bytes.data()+uint64_t(row)*packed_row_bytes(21,columns),weights.data(),int64_t(columns));
    return weights;
}
// Four magnitudes of one 9-bit grid index, recovered through the decoder: d=1, scale nibble
// 0 so the factor is exactly (1+2*0) = 1, signs clear, and code 0 of group 0 carries the
// index with qh[0] bit 0 supplying its ninth bit.
std::array<int,4> magnitudes_of(uint32_t index) {
    std::vector<uint8_t> bytes(xs_bytes,0);
    bytes[0]=0x00;bytes[1]=0x3c;
    bytes[qs_at]=uint8_t(index&255u);bytes[qh_at]=uint8_t(index>>8);
    const auto weights=decode_row(bytes,super,0);
    std::array<int,4> out{};
    for(uint32_t j=0;j<mags;++j) {
        require(std::abs(double(weights[j])-std::round(double(weights[j])))<1e-3,"IQ3_S grid magnitude is not integral");
        out[j]=int(std::round(double(weights[j])));
    }
    return out;
}
const std::array<std::array<int,4>,grid_entries>& grid_table() {
    static const std::array<std::array<int,4>,grid_entries> table=[]{
        std::array<std::array<int,4>,grid_entries> built{};
        for(uint32_t index=0;index<grid_entries;++index) built[index]=magnitudes_of(index);
        return built;
    }();
    return table;
}
uint32_t grid_pins=0;
void pin_grid() {
    std::vector<std::array<int,4>> seen(grid_entries);
    for(uint32_t index=0;index<grid_entries;++index) {
        const auto m=grid_table()[index];
        require(std::all_of(m.begin(),m.end(),[](int value){return value>=1 && value<=15;}),"an IQ3_S grid byte is outside the pinned magnitude range");
        seen[index]=m;++grid_pins;
    }
    uint32_t collisions=0;
    for(uint32_t a=0;a<grid_entries;++a) for(uint32_t b=a+1;b<grid_entries;++b) if(seen[a]==seen[b]) ++collisions;
    require(collisions==0,"IQ3_S grid entries are not distinguishable through the decoder");
    // The 9th bit must be readable: index 0 and index 256 are different entries.
    require(grid_table()[0]!=grid_table()[256],"the IQ3_S ninth index bit is not observable through the decoder");
}
uint32_t sign_checks=0;
void sign_rule() {
    // IQ3_S signs are a plain 8-bit field: bit b of signs[4g+l] negates exactly value
    // 32g+8l+b, with no popcount parity fold. Verified for every bit of two lanes.
    for(uint32_t lane=0;lane<2;++lane) for(uint32_t bit=0;bit<8;++bit) {
        std::vector<uint8_t> bytes(xs_bytes,0);
        bytes[0]=0x00;bytes[1]=0x3c;bytes[signs_at+lane]=uint8_t(1u<<bit);
        const auto weights=decode_row(bytes,super,0);
        // Grid index 0 holds four magnitudes of 1, so every value is +1 except the one
        // this sign bit governs: value lane*8+bit of the lane that owns values
        // 8*lane..8*lane+7.
        for(uint32_t v=lane*8;v<lane*8+8;++v) {
            const float expected=v==lane*8+bit?-1.0f:1.0f;
            require(std::abs(double(weights[v])-double(expected))<1e-4,"an IQ3_S sign bit did not map bit-for-value as measured");
            ++sign_checks;
        }
    }
}
Row x_logical(uint32_t columns,uint32_t row) {
    const auto count=blocks(columns);
    Row w{columns,std::vector<uint8_t>(uint64_t(count)*codes),std::vector<uint8_t>(uint64_t(count)*groups),
        std::vector<uint8_t>(uint64_t(count)*groups*lanes),std::vector<uint8_t>(uint64_t(count)*4),std::vector<float>(count)};
    for(uint32_t sb=0;sb<count;++sb) {
        w.d[sb]=float(int((sb*7+row*3+4)%11)-5)/8.f;
        for(uint32_t c=0;c<codes;++c) {
            const uint32_t index=uint32_t((c*37+sb*11+row*5)%grid_entries);
            w.qs[uint64_t(sb)*codes+c]=uint8_t(index&255u);
            w.qh[uint64_t(sb)*groups+c/8]|=uint8_t((index>>8)<<(c%8));
        }
        for(uint32_t s=0;s<groups*lanes;++s) w.signs[uint64_t(sb)*groups*lanes+s]=uint8_t((s*13+sb*5+row)%256);
        for(uint32_t b=0;b<4;++b) w.scale[uint64_t(sb)*4+b]=uint8_t(((b*3+row)%16) | (uint32_t((b*7+sb)%16)<<4));
    }
    return w;
}
std::vector<uint8_t> x_encode(const std::vector<Row>& rows) {
    const auto columns=rows.front().columns;
    std::vector<uint8_t> bytes(uint64_t(rows.size())*blocks(columns)*xs_bytes,0);
    const auto put=[&](size_t at,float value){const auto bits=strata::kernels::f16_from_f32(value);bytes[at]=uint8_t(bits);bytes[at+1]=uint8_t(bits>>8);};
    for(size_t row=0;row<rows.size();++row) {const auto& w=rows[row];
        for(uint32_t sb=0;sb<blocks(columns);++sb) {
            const size_t at=(uint64_t(row)*blocks(columns)+sb)*xs_bytes;
            for(uint32_t c=0;c<codes;++c) bytes[at+qs_at+c]=w.qs[uint64_t(sb)*codes+c];
            for(uint32_t g=0;g<groups;++g) bytes[at+qh_at+g]=w.qh[uint64_t(sb)*groups+g];
            for(uint32_t s=0;s<groups*lanes;++s) bytes[at+signs_at+s]=w.signs[uint64_t(sb)*groups*lanes+s];
            for(uint32_t b=0;b<4;++b) bytes[at+scales_at+b]=w.scale[uint64_t(sb)*4+b];
            put(at,w.d[sb]);
        }
    }
    return bytes;
}
// Independent transcription of the pinned CUDA form (iq_kernels.cu:215): an integer dot per
// 32-value group multiplied by the odd integer (1 + 2*sc).
Oracle x_oracle(const Row& w,const Q8Activation& q,uint32_t token) {
    Oracle out{0,0};
    for(uint32_t sb=0;sb<blocks(w.columns);++sb) for(uint32_t g=0;g<groups;++g) {
        const int sc=int((w.scale[uint64_t(sb)*4+g/2]>>(4*(g%2)))&15u);
        int dot=0;
        for(uint32_t lane=0;lane<lanes;++lane) {
            const uint8_t signs=w.signs[uint64_t(sb)*groups*lanes+g*lanes+lane];
            for(uint32_t half=0;half<2;++half) {
                const uint32_t code=g*8+lane*2+half;
                const uint32_t index=w.qs[uint64_t(sb)*codes+code]|uint32_t(((w.qh[uint64_t(sb)*groups+g]>>(lane*2+half))&1u)<<8);
                const auto m=grid_table()[index];
                for(uint32_t j=0;j<mags;++j)
                    dot+=((signs&(1u<<(half*4+j)))?-m[j]:m[j])*signed_code(q.codes[uint64_t(token)*w.columns+sb*super+g*32+lane*8+half*4+j]);
            }
        }
        const double factor=double(w.d[sb])*q.metadata[(uint64_t(token)*w.columns/32+sb*groups+g)*2];
        const long double term=static_cast<long double>(factor)*(1+2*sc)*dot;
        out.value+=term;out.magnitude+=std::abs(term);
    }
    return out;
}
Oracle x_decoded(const std::vector<uint8_t>& bytes,uint32_t columns,uint32_t row,const Q8Activation& q,uint32_t token) {
    const auto weights=decode_row(bytes,columns,row);
    Oracle out{0,0};
    for(uint32_t block=0;block<columns/32;++block) {
        const long double d8=q.metadata[(uint64_t(token)*columns/32+block)*2];long double part=0;
        for(uint32_t v=0;v<32;++v) part+=static_cast<long double>(weights[block*32+v])*signed_code(q.codes[uint64_t(token)*columns+block*32+v]);
        const long double term=d8*part;out.value+=term;out.magnitude+=std::abs(term);
    }
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
bool close(float actual,const Oracle& expected) {
    if(!std::isfinite(actual)) return false;
    return std::abs(static_cast<long double>(actual)-expected.value)<=std::max<long double>(2e-3L,expected.magnitude*2e-4L);
}
std::vector<float> source(uint64_t values,uint32_t seed) {
    std::vector<float> out(values);for(uint64_t i=0;i<values;++i) out[i]=float(int((i*23+seed)%127)-63)*.03125f+.0140625f;return out;
}
uint32_t cases_count=0;
void cases(uint32_t columns,uint32_t batch) {
    constexpr uint32_t rows=3;std::vector<Row> logical_rows;
    for(uint32_t row=0;row<rows;++row) logical_rows.push_back(x_logical(columns,row));
    const auto bytes=x_encode(logical_rows);
    validate_packed_rows(bytes,21,columns,rows);
    const auto values=source(uint64_t(batch)*columns,batch);
    const auto q=quantize_q8_activation(values,columns);
    const auto actual=tpc(bytes,21,columns,rows,q,batch,PackedActivation::Q8_1);
    const auto routed=tpc(bytes,21,columns,rows,q,batch,PackedActivation::RoutedQ8);
    for(uint32_t token=0;token<batch;++token) for(uint32_t row=0;row<rows;++row) {
        const auto index=uint64_t(token)*rows+row;
        require(close(actual[index],x_oracle(logical_rows[row],q,token)),"IQ3_S TPC differs from the independent CUDA integer-form oracle");
        require(close(actual[index],x_decoded(bytes,columns,row,q,token)),"IQ3_S TPC differs from the unmodified pinned ggml decoder");
        require(routed[index]==actual[index],"IQ3_S dot differs between the routed and shared/head q8_1 contracts");
        std::vector<float> one(values.begin()+uint64_t(token)*columns,values.begin()+uint64_t(token)*columns+columns);
        const auto isolated=tpc(bytes,21,columns,rows,quantize_q8_activation(one,columns),1,PackedActivation::Q8_1);
        require(isolated[row]==actual[index],"batched IQ3_S changed single-row arithmetic or crossed tokens");
        ++cases_count;
    }
}
void counterexamples() {
    constexpr uint32_t columns=512;
    const auto w=x_logical(columns,0);const auto bytes=x_encode({w});
    const auto values=source(columns,7);const auto q=quantize_q8_activation(values,columns);
    const auto before=tpc(bytes,21,columns,1,q,1,PackedActivation::Q8_1)[0];
    require(close(before,x_oracle(w,q,0)) && close(before,x_decoded(bytes,columns,0,q,0)),"IQ3_S baseline disagrees with its authorities");
    auto raw=q;for(uint32_t block=0;block<columns/32;++block) const_cast<Q8Activation&>(raw).metadata[block*2+1]=1234.5f;
    require(tpc(bytes,21,columns,1,raw,1,PackedActivation::Q8_1)[0]==before,"IQ3_S output moved when only raw sums changed");
    uint32_t moved=0;
    for(uint32_t v=0;v<columns && moved<4;++v) {
        if(!signed_code(q.codes[v]) || !q.metadata[(v/32)*2]) continue;
        ++moved;
        auto altered=w;const size_t code=(v/256)*codes+(v%256)/4;
        altered.qs[code]=uint8_t(altered.qs[code]+1);
        const auto altered_bytes=x_encode({altered});validate_packed_rows(altered_bytes,21,columns,1);
        const auto result=tpc(altered_bytes,21,columns,1,q,1,PackedActivation::Q8_1)[0];
        require(result!=before,"an IQ3_S grid index change was absorbed by the group scale");
        require(close(result,x_oracle(altered,q,0)) && close(result,x_decoded(altered_bytes,columns,0,q,0)),
            "the changed IQ3_S index lost its oracle or decoder match");
    }
    require(moved==4,"too few observable IQ3_S lanes to pin a single grid index");
    // The 9th index bit lives in a separate field: flipping one qh bit must move the dot.
    auto raised=w;raised.qh[0]^=1u;
    const auto raised_bytes=x_encode({raised});validate_packed_rows(raised_bytes,21,columns,1);
    require(tpc(raised_bytes,21,columns,1,q,1,PackedActivation::Q8_1)[0]!=before,"flipping an IQ3_S qh bit changed nothing, so the 9th index bit is unread");
    // Signs are per value: flipping one bit must move the dot and stay authority-consistent.
    auto flipped=w;flipped.signs[0]^=1u;
    const auto flipped_bytes=x_encode({flipped});validate_packed_rows(flipped_bytes,21,columns,1);
    const auto flipped_result=tpc(flipped_bytes,21,columns,1,q,1,PackedActivation::Q8_1)[0];
    require(flipped_result!=before && close(flipped_result,x_oracle(flipped,q,0)) && close(flipped_result,x_decoded(flipped_bytes,columns,0,q,0)),
        "flipping an IQ3_S sign bit did nothing or lost an authority");
    // The two nibbles of one scale byte are independent: change only the odd group.
    auto odd=w;odd.scale[0]=uint8_t(odd.scale[0]^0x10u);
    const auto odd_bytes=x_encode({odd});validate_packed_rows(odd_bytes,21,columns,1);
    require(close(tpc(odd_bytes,21,columns,1,q,1,PackedActivation::Q8_1)[0],x_oracle(odd,q,0)),"the IQ3_S high nibble scale did not follow the independent oracle");
    // The integer multiplier (1+2*sc) never reaches 0 and is odd: zeroing scales changes
    // the dot by exactly the factor 13 -> 1 on group 0.
    auto zeroed=w;std::fill(zeroed.scale.begin(),zeroed.scale.end(),uint8_t(0));
    const auto zeroed_bytes=x_encode({zeroed});validate_packed_rows(zeroed_bytes,21,columns,1);
    require(tpc(zeroed_bytes,21,columns,1,q,1,PackedActivation::Q8_1)[0]!=before,"zeroing the IQ3_S scales did not change the dot");
    auto negated=w;for(auto& value:negated.d) value=-value;
    const auto neg_bytes=x_encode({negated});validate_packed_rows(neg_bytes,21,columns,1);
    require(tpc(neg_bytes,21,columns,1,q,1,PackedActivation::Q8_1)[0]==-before,"negating the IQ3_S super-block d did not negate the dot");
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
    bytes[0]=0;bytes[1]=0x7c;rejects([&]{validate_packed_rows(bytes,21,256,1);});
    bytes[0]=0;bytes[1]=0x3c;validate_packed_rows(bytes,21,256,1);
    for(uint32_t at:{qs_at,qh_at,signs_at,scales_at}) bytes[at]=0xff;
    validate_packed_rows(bytes,21,256,1);
    rejects([&]{(void)packed_row_bytes(21,640);});rejects([&]{(void)packed_row_bytes(21,300);});
    require(packed_supported(21) && packed_values_per_block(21)==super && packed_block_bytes(21)==xs_bytes
        && packed_row_bytes(21,2560)==1100 && packed_row_bytes(21,256)==xs_bytes,"IQ3_S geometry wrong");
    require(packed_activation_supported(21,PackedActivation::Q8_1) && packed_activation_supported(21,PackedActivation::RoutedQ8)
        && !packed_activation_supported(21,PackedActivation::RoutedS2),"IQ3_S policy wrong");
    const auto routed=expert_format(2560,640,0,{21,21,8});
    require(routed.packed && routed.activation==PackedActivation::RoutedQ8
        && routed.bytes==uint64_t(1100)*640*2+uint64_t(34)*20*2560,"routed IQ3_S expert charge wrong");
    const auto shared=expert_format(2560,640,-1,{21,17,8});
    require(shared.packed && shared.activation==PackedActivation::Q8_1
        && shared.bytes==uint64_t(1100)*640+uint64_t(740)*640+uint64_t(34)*20*2560,"shared IQ3_S expert was not packed or is mischarged");
    const auto down=expert_format(2560,640,0,{8,8,21});
    require(!down.packed && down.bytes==uint64_t(2560)*640*6,"IQ3_S down projection with 640 columns was not rejected by layout");
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
    r.bytes=x_encode(r.xs);validate_packed_rows(r.bytes,21,columns,rows);return r;
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
    for(auto types:{std::array<uint32_t,3>{21,21,21},std::array<uint32_t,3>{21,21,8},std::array<uint32_t,3>{21,8,21},std::array<uint32_t,3>{8,21,21}})
    for(uint32_t batch:{1u,3u}) {
        const auto format=expert_format(hidden,intermediate,-1,types);
        require(format.packed && format.activation==PackedActivation::Q8_1,"shared IQ3_S expert lost the head contract");
        std::array<Flow,3> role{flow_role(types[0],hidden,intermediate,0),flow_role(types[1],hidden,intermediate,1),flow_role(types[2],intermediate,hidden,2)};
        const uint64_t charge=packed_row_bytes(types[0],hidden)*intermediate+packed_row_bytes(types[1],hidden)*intermediate
            +packed_row_bytes(types[2],intermediate)*hidden;
        require(format.bytes==charge,"shared IQ3_S expert payload mischarged");
        std::vector<float> input(uint64_t(batch)*hidden);
        for(size_t i=0;i<input.size();++i) input[i]=float(int((i*29+batch)%113)-56)*.0625f;
        std::array<uint32_t,3> calls{0,0,0};
        auto run=[&](const std::array<Flow,3>& roles)->std::vector<float> {
            calls={0,0,0};
            return execute_packed_expert(std::span<const float>(input),hidden,intermediate,[&](uint32_t which,std::span<const float> values) {
                ++calls[which];
                const auto mode=expert_projection_activation(format,which);
                require(mode==PackedActivation::Q8_1,"shared IQ3_S projection inherited a routed activation contract");
                const auto tokens=uint32_t(values.size()/roles[which].columns);
                const auto q=quantize_q8_activation(values,roles[which].columns);
                const auto expected=flow_oracle(roles[which],q,tokens);
                const auto actual=tpc(roles[which].bytes,roles[which].type,roles[which].columns,roles[which].rows,q,tokens,mode);
                for(size_t i=0;i<actual.size();++i) {
                    require(std::isfinite(actual[i]),"non-finite shared IQ3_S projection");
                    require(std::abs(double(actual[i])-double(expected[i]))<=std::max(.02,double(std::abs(expected[i]))*2e-3),
                        "shared expert IQ3_S projection differs from the independent oracle");
                }
                return actual;
            });
        };
        const auto output=run(role);
        require(calls==std::array<uint32_t,3>{1,1,1},"shared IQ3_S expert flow did not run each projection once");
        require(output.size()==input.size(),"shared IQ3_S expert flow output shape");
        for(float value:output) require(std::isfinite(value),"non-finite shared IQ3_S expert output");
        if(types[0]==types[1] && types[0]==21) {
            auto swapped=role;std::swap(swapped[0],swapped[1]);
            const auto other=run(swapped);bool differs=false;
            for(size_t i=0;i<output.size();++i) differs=differs||output[i]!=other[i];
            require(differs,"exchanging gate and up IQ3_S weights left the shared expert output unchanged");
        }
        ++flows;
    }
}
}
int main(){try {
    pin_grid();sign_rule();
    for(uint32_t columns:{256u,512u,2560u}) for(uint32_t batch:{1u,3u,8u}) cases(columns,batch);
    counterexamples();guards();shared_flows();
    std::cout<<"PASS IQ3_S TPC "<<cases_count<<" cases against the independent CUDA integer-form oracle (1+2*sc, no truncation) AND the unmodified pinned ggml IQ3_S row decoder, "
        <<grid_pins<<" 9-bit grid magnitudes recovered through the decoder and shown distinguishable, "<<sign_checks<<" plain-sign checks, "
        <<flows<<" production shared-expert flows, qh/sign/nibble-independence counterexamples and policy guards; no device\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
