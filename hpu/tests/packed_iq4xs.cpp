// Native IQ4_XS (GGML type 23) 256-value superblock packed dot: eight 32-value
// sub-blocks, each with a 6-bit scale used as (raw-32) and its own activation d8, and
// codebook values rather than affine codes. Three independent authorities are used: a
// logical-codebook long-double oracle, a separately written 6-bit scale reader, and the
// UNMODIFIED pinned ggml row decoder, which decides every bit position. Host only: no
// ISA execution, no device, no performance claim.
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
template<class F> void rejects(F f){bool bad=false;try{f();}catch(const std::exception&){bad=true;}require(bad,"invalid IQ4_XS payload or policy accepted");}
int signed_code(uint8_t b){return b<128?b:int(b)-256;}
constexpr uint32_t super=256,subs=8,sub_values=32,xs_bytes=136,q8_bytes=34;
// Pinned ggml src/ggml-common.h:1120 and native_mmvq.cu:449 agree on these values.
constexpr std::array<int,16> codebook{{-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113}};
uint32_t blocks(uint32_t columns){return columns/super;}
struct Oracle { long double value,magnitude; };
struct XSRow {
    uint32_t columns;
    std::vector<int> codes;                     // [columns] 0..15 codebook indices
    std::vector<int> raw;                       // [blocks*8] 6-bit scale, 0..63
    std::vector<float> d;                       // [blocks] signed
};
XSRow xs_logical(uint32_t columns,uint32_t row) {
    const auto count=blocks(columns);
    XSRow w{columns,std::vector<int>(columns),std::vector<int>(uint64_t(count)*subs),std::vector<float>(count)};
    for(uint32_t sb=0;sb<count;++sb) {
        w.d[sb]=float(int((sb*7+row*3+4)%11)-5)/8.f;                                             // both signs
        // raw is never 32, so every group scale (raw-32) is non-zero; every third group
        // takes the positive side and the rest the negative side.
        for(uint32_t s=0;s<subs;++s) w.raw[sb*subs+s]=int(3+(sb*11+s*5+row*2)%21)+((s%3)?0:32);
    }
    for(uint32_t v=0;v<columns;++v) w.codes[v]=int((v*13+(v>>4)*7+(v/super)*5+row*3)%16);
    require(std::any_of(w.codes.begin(),w.codes.end(),[](int c){return c==15;}),"IQ4_XS fixture never reached index 15");
    for(uint32_t sb=0;sb<count;++sb) for(uint32_t s=0;s<subs;++s)
        require(w.raw[sb*subs+s]!=32,"IQ4_XS fixture produced a zero (raw-32) group");
    for(uint32_t v=0;v+16<columns;++v) if(v%sub_values<16) require(w.codes[v]!=w.codes[v+16],"IQ4_XS fixture gave a nibble pair identical indices");
    return w;
}
// The measured field map (work/validation/iq4xs_layout_probe_v2.cpp): half d at 0,
// uint16 scales_h at 2, scales_l[4] at 4 and qs[128] at 8. Sub-block s takes its low
// nibble from scales_l[s/2] at shift 4*(s%2) and its top two bits from scales_h at
// shift 2*s. Byte 8+16*s+j carries value 32*s+j in the low nibble and 32*s+16+j in the
// high nibble - a per-sub-block byte run, not the Q3_K/Q4_K bit-plane layout.
std::vector<uint8_t> xs_encode(const std::vector<XSRow>& rows) {
    const auto columns=rows.front().columns;
    std::vector<uint8_t> bytes(uint64_t(rows.size())*blocks(columns)*xs_bytes,0);
    const auto put=[&](size_t at,float value){const auto bits=strata::kernels::f16_from_f32(value);bytes[at]=uint8_t(bits);bytes[at+1]=uint8_t(bits>>8);};
    for(size_t row=0;row<rows.size();++row) {const auto& w=rows[row];
        for(uint32_t sb=0;sb<blocks(columns);++sb) {
            const size_t at=(uint64_t(row)*blocks(columns)+sb)*xs_bytes;
            uint16_t high=0;
            for(uint32_t s=0;s<subs;++s) {
                const int raw=w.raw[sb*subs+s];
                bytes[at+4+s/2]|=uint8_t(uint32_t(raw&15)<<(4*(s%2)));
                high|=uint16_t(uint32_t((raw>>4)&3)<<(2*s));
            }
            bytes[at+2]=uint8_t(high);bytes[at+3]=uint8_t(high>>8);
            for(uint32_t s=0;s<subs;++s) for(uint32_t j=0;j<16;++j) {
                bytes[at+8+s*16+j]|=uint8_t(uint32_t(w.codes[sb*super+s*sub_values+j])&15u);
                bytes[at+8+s*16+j]|=uint8_t(uint32_t(w.codes[sb*super+s*sub_values+j+16]&15u)<<4);
            }
            put(at,w.d[sb]);
        }
    }
    return bytes;
}
// Independent scale reader: the whole block's scale fields are read as one 32-bit word
// pair and split by arithmetic, which is not how the kernel indexes individual bytes.
int xs_scale_at(const std::vector<uint8_t>& bytes,size_t at,uint32_t sub) {
    const uint32_t nibbles=uint32_t(bytes[at+4])|(uint32_t(bytes[at+5])<<8)|(uint32_t(bytes[at+6])<<16)|(uint32_t(bytes[at+7])<<24);
    const uint32_t pairs=uint32_t(bytes[at+2])|(uint32_t(bytes[at+3])<<8);
    const uint32_t low=(nibbles>>(4*(sub%2)+8*(sub/2)))&15u;
    const uint32_t high=(pairs>>(2*sub))&3u;
    return int(low|(high<<4))-32;
}
// Logical-codebook oracle in the upstream shape: sum += d8*(dot*(raw-32)) over the eight
// 32-value sub-blocks, then result += d*sum.
Oracle xs_oracle(const std::vector<uint8_t>& bytes,const XSRow& w,uint32_t row,const Q8Activation& q,uint32_t token,bool check_fixture=true,int only=-1) {
    Oracle out{0,0};
    const size_t base=uint64_t(row)*packed_row_bytes(23,w.columns);
    for(uint32_t sb=0;sb<blocks(w.columns);++sb) {
        const size_t at=base+uint64_t(sb)*xs_bytes;
        const float d=strata::kernels::f32_from_f16(uint16_t(bytes[at])|(uint16_t(bytes[at+1])<<8));
        require(std::abs(double(d)-double(w.d[sb]))<=1e-9,"IQ4_XS fixture d disagrees with its own encoded half");
        long double sum=0;
        for(uint32_t s=0;s<subs;++s) {
            if(only>=0 && int(s)!=only) continue;
            const int sc=xs_scale_at(bytes,at,s);
            // The scale used by the oracle always comes from the bytes through the
            // independent reader; the fixture cross-check only applies to an untouched
            // encoding, because scale_fields deliberately rewrites one field.
            if(check_fixture) require(sc==w.raw[sb*subs+s]-32,"IQ4_XS independent scale reader disagrees with the fixture's (raw-32)");
            int dot=0;
            for(uint32_t v=0;v<sub_values;++v) {
                dot+=codebook[w.codes[sb*super+s*sub_values+v]]*signed_code(q.codes[uint64_t(token)*w.columns+sb*super+s*sub_values+v]);
            }
            const long double d8=q.metadata[(uint64_t(token)*w.columns/32+sb*subs+s)*2];
            sum+=d8*static_cast<long double>(dot)*sc;
        }
        const long double term=d*sum;out.value+=term;out.magnitude+=std::abs(term);
    }
    return out;
}
// Byte-layout authority: decode with the unmodified pinned ggml row decoder, then dot
// with the values the q8_1 activation represents.
Oracle xs_decoded(const std::vector<uint8_t>& bytes,uint32_t columns,uint32_t row,const Q8Activation& q,uint32_t token) {
    const auto* traits=ggml_get_type_traits(GGML_TYPE_IQ4_XS);
    require(traits && traits->blck_size==int(super) && traits->to_float,"pinned ggml lacks the IQ4_XS row decoder");
    require(traits->type_size==xs_bytes,"pinned ggml IQ4_XS block size differs from the packed charge");
    std::vector<float> weights(columns);
    traits->to_float(bytes.data()+uint64_t(row)*packed_row_bytes(23,columns),weights.data(),int64_t(columns));
    Oracle out{0,0};
    for(uint32_t sb=0;sb<blocks(columns);++sb) for(uint32_t s=0;s<subs;++s) {
        const long double d8=q.metadata[(uint64_t(token)*columns/32+sb*subs+s)*2];long double part=0;
        for(uint32_t v=0;v<sub_values;++v) part+=static_cast<long double>(weights[sb*super+s*sub_values+v])*signed_code(q.codes[uint64_t(token)*columns+sb*super+s*sub_values+v]);
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
    return std::abs(static_cast<long double>(actual)-expected.value)<=std::max<long double>(.02L,expected.magnitude*2e-4L);
}
std::vector<float> source(uint64_t values,uint32_t seed) {
    std::vector<float> out(values);for(uint64_t i=0;i<values;++i) out[i]=float(int((i*23+seed)%127)-63)*.03125f+.0140625f;return out;
}
void cases(uint32_t columns,uint32_t batch,uint32_t& count) {
    constexpr uint32_t rows=3;std::vector<XSRow> logical_rows;
    for(uint32_t row=0;row<rows;++row) logical_rows.push_back(xs_logical(columns,row));
    const auto bytes=xs_encode(logical_rows);
    validate_packed_rows(bytes,23,columns,rows);
    const auto values=source(uint64_t(batch)*columns,batch);
    const auto q=quantize_q8_activation(values,columns);
    const auto actual=tpc(bytes,23,columns,rows,q,batch,PackedActivation::Q8_1);
    const auto routed=tpc(bytes,23,columns,rows,q,batch,PackedActivation::RoutedQ8);
    for(uint32_t token=0;token<batch;++token) for(uint32_t row=0;row<rows;++row) {
        const auto index=uint64_t(token)*rows+row;
        require(close(actual[index],xs_oracle(bytes,logical_rows[row],row,q,token)),"IQ4_XS TPC differs from the independent logical-codebook oracle");
        require(close(actual[index],xs_decoded(bytes,columns,row,q,token)),"IQ4_XS bytes disagree with the unmodified pinned ggml decoder");
        require(routed[index]==actual[index],"IQ4_XS dot differs between the routed and shared/head q8_1 contracts");
        std::vector<float> one(values.begin()+uint64_t(token)*columns,values.begin()+uint64_t(token)*columns+columns);
        const auto isolated=tpc(bytes,23,columns,rows,quantize_q8_activation(one,columns),1,PackedActivation::Q8_1);
        require(isolated[row]==actual[index],"batched IQ4_XS changed single-row arithmetic or crossed tokens");
        ++count;
    }
}
// The packed type set is asserted as a whole: adding a format cannot leave a stale
// negative assertion behind in an older test. The remaining IQ families and BF16 must
// stay on the decoded side until each has its own dot and dual-authority test.
void policy_types() {
    for(uint32_t type:{2,3,6,7,8,11,12,13,14,16,17,18,20,21,22,23,29,40,42}) require(packed_supported(type),"a packed GGML type with a working dot was rejected");
    for(uint32_t type:{19,30,31}) require(!packed_supported(type),"a still-decoded GGML type was accepted as packed");
}
// Each of the eight 6-bit scale fields must govern exactly its own 32 values: muting
// one field removes precisely that sub-block's contribution and nothing else.
void scale_fields(uint32_t& count) {
    constexpr uint32_t columns=256,rows=1;
    const auto w=xs_logical(columns,0);const auto bytes=xs_encode({w});
    const auto values=source(columns,9);const auto q=quantize_q8_activation(values,columns);
    const auto whole=xs_oracle(bytes,w,0,q,0);
    require(std::abs(double(whole.value))>1.0,"IQ4_XS scale-field fixture cancels itself");
    for(uint32_t s=0;s<subs;++s) {
        const auto part=xs_oracle(bytes,w,0,q,0,true,int(s));
        require(std::abs(double(part.value))>1e-3,"an IQ4_XS sub-block contributes nothing, so its scale field is unobservable");
        auto muted=bytes;
        // raw 32 means (raw-32) == 0: rewrite only this group's six bits.
        auto& nibble=muted[4+s/2];const uint32_t shift=4*(s%2);
        nibble=uint8_t(nibble&~(15u<<shift));
        uint16_t pairs=uint16_t(muted[2])|(uint16_t(muted[3])<<8);
        pairs=uint16_t(pairs&~(3u<<(2*s)));pairs=uint16_t(pairs|(2u<<(2*s)));
        muted[2]=uint8_t(pairs);muted[3]=uint8_t(pairs>>8);
        validate_packed_rows(muted,23,columns,rows);
        const auto result=tpc(muted,23,columns,rows,q,1,PackedActivation::Q8_1)[0];
        require(close(result,{whole.value-part.value,whole.magnitude}),"muting one IQ4_XS scale field did not remove exactly that sub-block");
        require(close(result,xs_oracle(muted,w,0,q,0,false)),"the muted IQ4_XS field disagrees with the independent scale reader");
        require(close(result,xs_decoded(muted,columns,0,q,0)),"muting one IQ4_XS scale field broke the pinned decoder match");
        ++count;
    }
}
void counterexamples() {
    const auto w=xs_logical(512,0);const auto bytes=xs_encode({w});
    const auto values=source(512,7);const auto q=quantize_q8_activation(values,512);
    const auto before=tpc(bytes,23,512,1,q,1,PackedActivation::Q8_1)[0];
    require(close(before,xs_oracle(bytes,w,0,q,0)) && close(before,xs_decoded(bytes,512,0,q,0)),"IQ4_XS baseline disagrees with its oracles");
    auto raw=q;for(uint32_t b=0;b<512/32;++b) const_cast<Q8Activation&>(raw).metadata[b*2+1]=1234.5f;
    require(tpc(bytes,23,512,1,raw,1,PackedActivation::Q8_1)[0]==before,"IQ4_XS output moved when only raw sums changed");
    // A codebook index step is visible wherever the lane can be observed.
    uint32_t stepped=0;
    for(uint32_t v=0;v<512 && stepped<6;++v) {
        if(!signed_code(q.codes[v]) || !q.metadata[(v/sub_values)*2]) continue;
        ++stepped;
        auto advanced=w;advanced.codes[v]=(advanced.codes[v]+1)%16;
        const auto advanced_bytes=xs_encode({advanced});validate_packed_rows(advanced_bytes,23,512,1);
        const auto moved=tpc(advanced_bytes,23,512,1,q,1,PackedActivation::Q8_1)[0];
        require(moved!=before,"an IQ4_XS codebook index step was absorbed by the group scale");
        require(close(moved,xs_oracle(advanced_bytes,advanced,0,q,0)) && close(moved,xs_decoded(advanced_bytes,512,0,q,0)),
            "the stepped IQ4_XS index lost its oracle or decoder match");
    }
    require(stepped==6,"too few observable IQ4_XS lanes to pin a single index step");
    // The low nibble of a code byte and the high nibble are 16 values apart: exchanging
    // them inside one byte must change the row.
    auto exchanged=w;std::swap(exchanged.codes[2],exchanged.codes[18]);
    require(exchanged.codes[2]!=exchanged.codes[18],"the fixture put identical indices in a nibble pair");
    const auto exchange_bytes=xs_encode({exchanged});validate_packed_rows(exchange_bytes,23,512,1);
    require(tpc(exchange_bytes,23,512,1,q,1,PackedActivation::Q8_1)[0]!=before,"exchanging an IQ4_XS nibble pair changed nothing");
    // (raw-32) is signed: mirroring every group about 32 negates the whole dot.
    auto mirrored=w;for(auto& value:mirrored.raw) value=64-value;
    const auto mirrored_bytes=xs_encode({mirrored});validate_packed_rows(mirrored_bytes,23,512,1);
    require(tpc(mirrored_bytes,23,512,1,q,1,PackedActivation::Q8_1)[0]==-before,"mirroring the IQ4_XS 6-bit scales about 32 did not negate the dot");
    require(close(tpc(mirrored_bytes,23,512,1,q,1,PackedActivation::Q8_1)[0],xs_decoded(mirrored_bytes,512,0,q,0)),
        "mirrored IQ4_XS scales stopped matching the pinned decoder");
    // Swapping two adjacent group scales must matter: the two halves have different dots.
    auto exchange_scales=w;std::swap(exchange_scales.raw[0],exchange_scales.raw[1]);
    const auto scale_bytes=xs_encode({exchange_scales});validate_packed_rows(scale_bytes,23,512,1);
    require(tpc(scale_bytes,23,512,1,q,1,PackedActivation::Q8_1)[0]!=before,"exchanging two adjacent IQ4_XS group scales did not change the dot");
    // Negating d negates the result; the scales are unchanged.
    auto negated=w;for(auto& value:negated.d) value=-value;
    const auto neg_bytes=xs_encode({negated});validate_packed_rows(neg_bytes,23,512,1);
    require(tpc(neg_bytes,23,512,1,q,1,PackedActivation::Q8_1)[0]==-before,"negating the IQ4_XS super-block d did not negate the dot");
    // Clearing scales_h entirely is legal data: the top two bits of every group vanish,
    // and the pinned decoder must still agree with the kernel.
    auto no_high=bytes;no_high[2]=0;no_high[3]=0;no_high[136+2]=0;no_high[136+3]=0;
    validate_packed_rows(no_high,23,512,1);
    const auto lowered=tpc(no_high,23,512,1,q,1,PackedActivation::Q8_1)[0];
    require(lowered!=before && close(lowered,xs_decoded(no_high,512,0,q,0)),"dropping the IQ4_XS high scale bits changed nothing or left the decoder");
}
void guards() {
    policy_types();
    const auto w=xs_logical(256,0);auto bytes=xs_encode({w});
    bytes[0]=0;bytes[1]=0x7c;rejects([&]{validate_packed_rows(bytes,23,256,1);});
    bytes[0]=0;bytes[1]=0x3c;validate_packed_rows(bytes,23,256,1);
    // The scale fields are code data: arbitrary bytes stay acceptable.
    bytes[2]=0xff;bytes[4]=0xff;validate_packed_rows(bytes,23,256,1);
    rejects([&]{(void)packed_row_bytes(23,640);});rejects([&]{(void)packed_row_bytes(23,300);});
    require(packed_supported(23) && packed_values_per_block(23)==super && packed_block_bytes(23)==xs_bytes
        && packed_row_bytes(23,2560)==1360 && packed_row_bytes(23,256)==xs_bytes,"IQ4_XS geometry wrong");
    require(packed_activation_supported(23,PackedActivation::Q8_1) && packed_activation_supported(23,PackedActivation::RoutedQ8)
        && !packed_activation_supported(23,PackedActivation::RoutedS2),"IQ4_XS policy wrong");
    const auto routed=expert_format(2560,640,0,{23,23,8});
    require(routed.packed && routed.activation==PackedActivation::RoutedQ8
        && routed.bytes==uint64_t(1360)*640*2+uint64_t(34)*20*2560,"routed IQ4_XS expert charge wrong");
    const auto shared=expert_format(2560,640,-1,{23,14,8});
    require(shared.packed && shared.activation==PackedActivation::Q8_1
        && shared.bytes==uint64_t(1360)*640+uint64_t(2100)*640+uint64_t(34)*20*2560,"shared IQ4_XS expert was not packed or is mischarged");
    const auto down=expert_format(2560,640,0,{8,8,23});
    require(!down.packed && down.bytes==uint64_t(2560)*640*6,"IQ4_XS down projection with 640 columns was not rejected by layout");
    require(expert_projection_activation(routed,0)==PackedActivation::RoutedQ8,"IQ4_XS routed role inherited the wrong activation contract");
}
struct Flow {
    uint32_t type,columns,rows;
    std::vector<uint8_t> bytes;
    std::vector<XSRow> xs;
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
    for(uint32_t row=0;row<rows;++row) r.xs.push_back(xs_logical(columns,row+seed*1000));
    r.bytes=xs_encode(r.xs);validate_packed_rows(r.bytes,23,columns,rows);return r;
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
        out[uint64_t(token)*r.rows+row]=float(xs_oracle(r.bytes,r.xs[row],row,q,token).value);
    }
    return out;
}
void shared_flows(uint32_t& count) {
    constexpr uint32_t hidden=512,intermediate=256;
    for(auto types:{std::array<uint32_t,3>{23,23,23},std::array<uint32_t,3>{23,23,8},std::array<uint32_t,3>{23,8,23},std::array<uint32_t,3>{8,23,23}})
    for(uint32_t batch:{1u,3u}) {
        const auto format=expert_format(hidden,intermediate,-1,types);
        require(format.packed && format.activation==PackedActivation::Q8_1,"shared IQ4_XS expert lost the head contract");
        std::array<Flow,3> role{flow_role(types[0],hidden,intermediate,0),flow_role(types[1],hidden,intermediate,1),flow_role(types[2],intermediate,hidden,2)};
        const uint64_t charge=packed_row_bytes(types[0],hidden)*intermediate+packed_row_bytes(types[1],hidden)*intermediate
            +packed_row_bytes(types[2],intermediate)*hidden;
        require(format.bytes==charge,"shared IQ4_XS expert payload mischarged");
        std::vector<float> input(uint64_t(batch)*hidden);
        for(size_t i=0;i<input.size();++i) input[i]=float(int((i*29+batch)%113)-56)*.0625f;
        std::array<uint32_t,3> calls{0,0,0};
        auto run=[&](const std::array<Flow,3>& roles)->std::vector<float> {
            calls={0,0,0};
            return execute_packed_expert(std::span<const float>(input),hidden,intermediate,[&](uint32_t which,std::span<const float> values) {
                ++calls[which];
                const auto mode=expert_projection_activation(format,which);
                require(mode==PackedActivation::Q8_1,"shared IQ4_XS projection inherited a routed activation contract");
                const auto tokens=uint32_t(values.size()/roles[which].columns);
                const auto q=quantize_q8_activation(values,roles[which].columns);
                const auto expected=flow_oracle(roles[which],q,tokens);
                const auto actual=tpc(roles[which].bytes,roles[which].type,roles[which].columns,roles[which].rows,q,tokens,mode);
                for(size_t i=0;i<actual.size();++i) {
                    require(std::isfinite(actual[i]),"non-finite shared IQ4_XS projection");
                    require(std::abs(double(actual[i])-double(expected[i]))<=std::max(.02,double(std::abs(expected[i]))*2e-3),
                        "shared expert IQ4_XS projection differs from the independent oracle");
                }
                return actual;
            });
        };
        const auto output=run(role);
        require(calls==std::array<uint32_t,3>{1,1,1},"shared IQ4_XS expert flow did not run each projection once");
        require(output.size()==input.size(),"shared IQ4_XS expert flow output shape");
        for(float value:output) require(std::isfinite(value),"non-finite shared IQ4_XS expert output");
        if(types[0]==types[1] && types[0]==23) {
            auto swapped=role;std::swap(swapped[0],swapped[1]);
            const auto other=run(swapped);bool differs=false;
            for(size_t i=0;i<output.size();++i) differs=differs||output[i]!=other[i];
            require(differs,"exchanging gate and up IQ4_XS weights left the shared expert output unchanged");
        }
        ++count;
    }
}
}
int main(){try {uint32_t count=0,fields=0,flows=0;
    for(uint32_t columns:{256u,512u,2560u}) for(uint32_t batch:{1u,3u,8u}) cases(columns,batch,count);
    scale_fields(fields);counterexamples();guards();shared_flows(flows);
    std::cout<<"PASS IQ4_XS TPC "<<count<<" cases against an independent logical-codebook oracle, a separately written 6-bit scale reader and the unmodified pinned ggml IQ4_XS row decoder, plus "<<fields<<" per-group scale-field checks and "<<flows<<" production shared-expert flows, signed (raw-32) group scales, codebook values, per-sub-block d8, raw-sum insensitivity and policy guards; no device\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
