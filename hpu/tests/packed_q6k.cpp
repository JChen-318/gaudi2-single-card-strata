// Native Q6_K (GGML type 14) 256-value superblock packed dot for the shared/head
// and routed q8_1 contracts. As with Q3_K, two independent authorities are used: a
// logical-code long-double oracle and the UNMODIFIED pinned ggml row decoder, which
// decides every bit position here. Host only: no ISA execution, no device, no
// performance claim.
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
template<class F> void rejects(F f){bool bad=false;try{f();}catch(const std::exception&){bad=true;}require(bad,"invalid Q6_K payload or policy accepted");}
int signed_code(uint8_t b){return b<128?b:int(b)-256;}
constexpr uint32_t super=256,subs=16,sub_size=16,q6_bytes=210,q8_bytes=34;
uint32_t blocks(uint32_t columns){return columns/super;}
struct Oracle { long double value,magnitude; };

// Logical Q6_K row: six-bit codes, one SIGNED int8 scale per 16-value sub-block and
// one signed super-block d (no learned minimum).
struct Q6Row {
    uint32_t columns;
    std::vector<int> codes;                     // [columns] 0..63
    std::vector<int> scale;                     // [blocks*16] -32..31
    std::vector<float> d;                       // [blocks] signed
};
Q6Row q6_logical(uint32_t columns,uint32_t row) {
    const auto count=blocks(columns);
    Q6Row w{columns,std::vector<int>(columns),std::vector<int>(uint64_t(count)*subs),std::vector<float>(count)};
    for(uint32_t sb=0;sb<count;++sb) {
        w.d[sb]=float(int((sb*7+row*3+4)%11)-5)/8.f;                       // both signs occur
        for(uint32_t s=0;s<subs;++s) w.scale[sb*subs+s]=int((sb*5+s*13+row*11+7)%17)-8;  // -8..8
    }
    for(uint32_t v=0;v<columns;++v) w.codes[v]=int((v*37+(v/super)*5+row*7)%64);
    return w;
}
// The measured field map (see work/validation/q6k_layout_probe.cpp): ql[128] at 0
// keeps the low nibble of value v in byte 64*(v/128)+(v%128)%64 at shift
// 4*((v%128)/64); qh[64] at 128 keeps the two high bits in byte 32*(v/128)+
// (v%128)%32 at shift 2*((v%128)/32); the SIGNED sub-block scale sits at 192+s and
// the half d at 208.
std::vector<uint8_t> q6_encode(const std::vector<Q6Row>& rows) {
    const auto columns=rows.front().columns;
    std::vector<uint8_t> bytes(uint64_t(rows.size())*blocks(columns)*q6_bytes,0);
    const auto put=[&](size_t at,float value){const auto bits=strata::kernels::f16_from_f32(value);bytes[at]=uint8_t(bits);bytes[at+1]=uint8_t(bits>>8);};
    for(size_t row=0;row<rows.size();++row) {const auto& w=rows[row];
        for(uint32_t sb=0;sb<blocks(columns);++sb) {
            const size_t at=(uint64_t(row)*blocks(columns)+sb)*q6_bytes;
            for(uint32_t v=0;v<super;++v) {
                const int code=w.codes[sb*super+v],j=v/128,i=v%128;
                bytes[at+64*j+(i%64)]|=uint8_t(uint32_t(code&15)<<(4*(i/64)));
                bytes[at+128+32*j+(i%32)]|=uint8_t(uint32_t((code>>4)&3)<<(2*(i/32)));
            }
            for(uint32_t s=0;s<subs;++s) bytes[at+192+s]=uint8_t(int8_t(w.scale[sb*subs+s]));
            put(at+208,w.d[sb]);
        }
    }
    return bytes;
}
// Logical-code oracle in the upstream accumulation shape: per half-superblock
// sum += d8*(dot*sc) over the eight 16-value sub-blocks, then result += d*sum.
Oracle q6_oracle(const std::vector<uint8_t>& bytes,const Q6Row& w,uint32_t row,const Q8Activation& q,uint32_t token) {
    Oracle out{0,0};
    const size_t base=uint64_t(row)*packed_row_bytes(14,w.columns);
    for(uint32_t sb=0;sb<blocks(w.columns);++sb) {
        const size_t at=base+uint64_t(sb)*q6_bytes;
        const float d=strata::kernels::f32_from_f16(uint16_t(bytes[at+208])|(uint16_t(bytes[at+209])<<8));
        require(std::abs(double(d)-double(w.d[sb]))<=1e-9,"Q6_K fixture d disagrees with its own encoded half");
        long double first=0,second=0;
        for(uint32_t s=0;s<subs;++s) {
            const int sc=int(int8_t(bytes[at+192+s]));
            require(sc==w.scale[sb*subs+s],"Q6_K fixture scale disagrees with the signed byte read back");
            int dot=0;
            for(uint32_t l=0;l<sub_size;++l) {const uint32_t v=s*sub_size+l;
                dot+=(w.codes[sb*super+v]-32)*signed_code(q.codes[uint64_t(token)*w.columns+sb*super+v]);}
            const long double d8=q.metadata[(uint64_t(token)*w.columns/32+sb*8+s/2)*2];
            const long double term=d8*static_cast<long double>(dot)*sc;
            if(s<8) first+=term;else second+=term;
        }
        const long double a=d*first,b=d*second;
        out.value+=a+b;out.magnitude+=std::abs(a)+std::abs(b);
    }
    return out;
}
// Byte-layout authority: decode with the unmodified pinned ggml row decoder, then
// dot with the values q8_1 represents (d8 times the int8 code).
Oracle q6_decoded(const std::vector<uint8_t>& bytes,uint32_t columns,uint32_t row,const Q8Activation& q,uint32_t token) {
    const auto* traits=ggml_get_type_traits(GGML_TYPE_Q6_K);
    require(traits && traits->blck_size==int(super) && traits->to_float,"pinned ggml lacks the Q6_K row decoder");
    require(traits->type_size==q6_bytes,"pinned ggml Q6_K block size differs from the packed charge");
    std::vector<float> weights(columns);
    traits->to_float(bytes.data()+uint64_t(row)*packed_row_bytes(14,columns),weights.data(),int64_t(columns));
    Oracle out{0,0};
    for(uint32_t sb=0;sb<blocks(columns);++sb) for(uint32_t s=0;s<subs;++s) {
        const long double d8=q.metadata[(uint64_t(token)*columns/32+sb*8+s/2)*2];long double part=0;
        for(uint32_t l=0;l<sub_size;++l) part+=static_cast<long double>(weights[sb*super+s*sub_size+l])*signed_code(q.codes[uint64_t(token)*columns+sb*super+s*sub_size+l]);
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
    constexpr uint32_t rows=3;std::vector<Q6Row> logical_rows;
    for(uint32_t row=0;row<rows;++row) logical_rows.push_back(q6_logical(columns,row));
    const auto bytes=q6_encode(logical_rows);
    validate_packed_rows(bytes,14,columns,rows);
    const auto values=source(uint64_t(batch)*columns,batch);
    const auto q=quantize_q8_activation(values,columns);
    const auto actual=tpc(bytes,14,columns,rows,q,batch,PackedActivation::Q8_1);
    const auto routed=tpc(bytes,14,columns,rows,q,batch,PackedActivation::RoutedQ8);
    for(uint32_t token=0;token<batch;++token) for(uint32_t row=0;row<rows;++row) {
        const auto index=uint64_t(token)*rows+row;
        require(close(actual[index],q6_oracle(bytes,logical_rows[row],row,q,token)),"Q6_K TPC differs from the independent logical-code oracle");
        require(close(actual[index],q6_decoded(bytes,columns,row,q,token)),"Q6_K bytes disagree with the unmodified pinned ggml decoder");
        require(routed[index]==actual[index],"Q6_K dot differs between the routed and shared/head q8_1 contracts");
        std::vector<float> one(values.begin()+uint64_t(token)*columns,values.begin()+uint64_t(token)*columns+columns);
        const auto isolated=tpc(bytes,14,columns,rows,quantize_q8_activation(one,columns),1,PackedActivation::Q8_1);
        require(isolated[row]==actual[index],"batched Q6_K changed single-row arithmetic or crossed tokens");
        ++count;
    }
}
void counterexamples() {
    const auto w=q6_logical(512,0);const auto bytes=q6_encode({w});
    const auto values=source(512,7);const auto q=quantize_q8_activation(values,512);
    const auto before=tpc(bytes,14,512,1,q,1,PackedActivation::Q8_1)[0];
    require(close(before,q6_oracle(bytes,w,0,q,0)) && close(before,q6_decoded(bytes,512,0,q,0)),"Q6_K baseline disagrees with its oracles");
    auto raw=q;for(uint32_t g=0;g<512/32;++g) const_cast<Q8Activation&>(raw).metadata[g*2+1]=1234.5f;
    require(tpc(bytes,14,512,1,raw,1,PackedActivation::Q8_1)[0]==before,"Q6_K output moved when only raw sums changed");
    require(std::any_of(w.codes.begin(),w.codes.end(),[](int c){return c>=16;}),"fixture never produced a code with the high-bit pair set");
    // The two high bits live in qh: clearing that whole field must change the row
    // and stay decoder-consistent.
    auto no_hi=bytes;for(uint32_t i=128;i<192;++i) no_hi[i]=0;
    validate_packed_rows(no_hi,14,512,1);
    const auto hi=tpc(no_hi,14,512,1,q,1,PackedActivation::Q8_1)[0];
    require(hi!=before && close(hi,q6_decoded(no_hi,512,0,q,0)),"dropping the Q6_K high-bit pair changed neither the dot nor the decoder match");
    // All-zero signed scales must annihilate the dot; this also proves the scale is
    // read as a signed byte rather than as 0..255.
    auto zeroed=bytes;for(uint32_t sb=0;sb<2;++sb) for(uint32_t s=0;s<subs;++s) zeroed[sb*q6_bytes+192+s]=0;
    validate_packed_rows(zeroed,14,512,1);
    require(tpc(zeroed,14,512,1,q,1,PackedActivation::Q8_1)[0]==0.f,"zeroing the Q6_K sub-block scales did not zero the dot");
    // Negating every scale must negate the result (signed int8, no learned minimum).
    auto neg_scales=bytes;for(uint32_t sb=0;sb<2;++sb) for(uint32_t s=0;s<subs;++s) {
        const size_t at=sb*q6_bytes+192+s;neg_scales[at]=uint8_t(uint8_t(256u-uint8_t(neg_scales[at])));
    }
    validate_packed_rows(neg_scales,14,512,1);
    require(close(tpc(neg_scales,14,512,1,q,1,PackedActivation::Q8_1)[0],q6_decoded(neg_scales,512,0,q,0)),"negated scale bytes stopped matching the decoder");
    // One code change moves the result and both authorities agree. A change is only
    // observable on a lane whose SIGNED sub-block scale and activation int8 code are
    // both non-zero, so search for such a lane rather than assuming a fixed one.
    uint32_t moved_value=0;bool found=false;
    for(uint32_t v=0;v<512 && !found;++v) if(w.scale[v/16]!=0 && signed_code(q.codes[v])!=0) {moved_value=v;found=true;}
    require(found,"no Q6_K lane had both a non-zero signed sub-block scale and a non-zero activation code");
    auto shifted=w;shifted.codes[moved_value]=(shifted.codes[moved_value]+1)%64;
    const auto shifted_bytes=q6_encode({shifted});validate_packed_rows(shifted_bytes,14,512,1);
    const auto moved=tpc(shifted_bytes,14,512,1,q,1,PackedActivation::Q8_1)[0];
    require(moved!=before,"a single Q6_K code change was absorbed by the sub-block scale");
    require(close(moved,q6_oracle(shifted_bytes,shifted,0,q,0)) && close(moved,q6_decoded(shifted_bytes,512,0,q,0)),"changed Q6_K codes lost their oracle or decoder match");
    // Flipping d's sign flips the result.
    auto negated=w;for(auto& value:negated.d) value=-value;
    const auto neg_bytes=q6_encode({negated});validate_packed_rows(neg_bytes,14,512,1);
    require(tpc(neg_bytes,14,512,1,q,1,PackedActivation::Q8_1)[0]==-before,"negating the single Q6_K d did not negate the dot");
    // Sub-block scales are per 16 values: swapping two of them must matter.
    auto exchange=w;std::swap(exchange.scale[0],exchange.scale[1]);
    const auto ex_bytes=q6_encode({exchange});validate_packed_rows(ex_bytes,14,512,1);
    require(tpc(ex_bytes,14,512,1,q,1,PackedActivation::Q8_1)[0]!=before,"exchanging two Q6_K sub-block scales did not change the dot");
}
// The packed type set is asserted as a whole: adding a format cannot leave a stale
// negative assertion behind in an older test. The IQ families and BF16 must stay on
// the decoded side until each has its own dot and dual-authority test.
void policy_types() {
    for(uint32_t type:{2,3,6,7,8,11,12,13,14,16,17,18,20,21,22,23,29,40,42}) require(packed_supported(type),"a packed GGML type with a working dot was rejected");
    for(uint32_t type:{19,30,31}) require(!packed_supported(type),"a still-decoded GGML type was accepted as packed");
}
void guards() {
    policy_types();
    const auto w=q6_logical(256,0);auto bytes=q6_encode({w});
    bytes[208]=0;bytes[209]=0x7c;rejects([&]{validate_packed_rows(bytes,14,256,1);});
    bytes[208]=0;bytes[209]=0x3c;validate_packed_rows(bytes,14,256,1);
    // Scale bytes are signed code data: an arbitrary 0xff must stay acceptable.
    bytes[192]=0xff;validate_packed_rows(bytes,14,256,1);
    rejects([&]{(void)packed_row_bytes(14,640);});rejects([&]{(void)packed_row_bytes(14,300);});
    require(packed_supported(14) && packed_values_per_block(14)==super && packed_block_bytes(14)==q6_bytes
        && packed_row_bytes(14,2560)==2100 && packed_row_bytes(14,256)==q6_bytes,"Q6_K geometry wrong");
    require(packed_activation_supported(14,PackedActivation::Q8_1) && packed_activation_supported(14,PackedActivation::RoutedQ8)
        && !packed_activation_supported(14,PackedActivation::RoutedS2),
        "Q6_K policy or unsupported format set wrong");
    const auto routed=expert_format(2560,640,0,{14,14,8});
    require(routed.packed && routed.activation==PackedActivation::RoutedQ8
        && routed.bytes==uint64_t(2100)*640*2+uint64_t(34)*20*2560,"routed Q6_K expert charge wrong");
    const auto shared=expert_format(2560,640,-1,{14,11,8});
    require(shared.packed && shared.activation==PackedActivation::Q8_1
        && shared.bytes==uint64_t(2100)*640+uint64_t(1100)*640+uint64_t(34)*20*2560,"shared Q6_K expert was not packed or is mischarged");
    const auto down=expert_format(2560,640,0,{8,8,14});
    require(!down.packed && down.bytes==uint64_t(2560)*640*6,"Q6_K down projection with 640 columns was not rejected by layout");
    rejects([&]{(void)expert_projection_activation(expert_format(2560,640,0,{8,8,14}),0);});
    rejects([&]{(void)expert_projection_activation(expert_format(2560,640,0,{14,14,14}),2);});
}
// Production shared-expert flow with Q6_K roles on the head/shared q8_1 contract.
struct Flow {
    uint32_t type,columns,rows;
    std::vector<uint8_t> bytes;
    std::vector<Q6Row> q6;
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
    for(uint32_t row=0;row<rows;++row) r.q6.push_back(q6_logical(columns,row+seed*1000));
    r.bytes=q6_encode(r.q6);validate_packed_rows(r.bytes,14,columns,rows);return r;
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
        out[uint64_t(token)*r.rows+row]=float(q6_oracle(r.bytes,r.q6[row],row,q,token).value);
    }
    return out;
}
void shared_flows(uint32_t& count) {
    constexpr uint32_t hidden=512,intermediate=256;
    for(auto types:{std::array<uint32_t,3>{14,14,14},std::array<uint32_t,3>{14,14,8},std::array<uint32_t,3>{14,8,14},std::array<uint32_t,3>{8,14,14}})
    for(uint32_t batch:{1u,3u}) {
        const auto format=expert_format(hidden,intermediate,-1,types);
        require(format.packed && format.activation==PackedActivation::Q8_1,"shared Q6_K expert lost the head contract");
        std::array<Flow,3> role{flow_role(types[0],hidden,intermediate,0),flow_role(types[1],hidden,intermediate,1),flow_role(types[2],intermediate,hidden,2)};
        const uint64_t charge=packed_row_bytes(types[0],hidden)*intermediate+packed_row_bytes(types[1],hidden)*intermediate
            +packed_row_bytes(types[2],intermediate)*hidden;
        require(format.bytes==charge,"shared Q6_K expert payload mischarged");
        std::vector<float> input(uint64_t(batch)*hidden);
        for(size_t i=0;i<input.size();++i) input[i]=float(int((i*29+batch)%113)-56)*.0625f;
        std::array<uint32_t,3> calls{0,0,0};
        auto run=[&](const std::array<Flow,3>& roles)->std::vector<float> {
            calls={0,0,0};
            return execute_packed_expert(std::span<const float>(input),hidden,intermediate,[&](uint32_t which,std::span<const float> values) {
                ++calls[which];
                const auto mode=expert_projection_activation(format,which);
                require(mode==PackedActivation::Q8_1,"shared Q6_K projection inherited a routed activation contract");
                const auto tokens=uint32_t(values.size()/roles[which].columns);
                const auto q=quantize_q8_activation(values,roles[which].columns);
                const auto expected=flow_oracle(roles[which],q,tokens);
                const auto actual=tpc(roles[which].bytes,roles[which].type,roles[which].columns,roles[which].rows,q,tokens,mode);
                for(size_t i=0;i<actual.size();++i) {
                    require(std::isfinite(actual[i]),"non-finite shared Q6_K projection");
                    require(std::abs(double(actual[i])-double(expected[i]))<=std::max(.02,double(std::abs(expected[i]))*2e-3),
                        "shared expert Q6_K projection differs from the independent oracle");
                }
                return actual;
            });
        };
        const auto output=run(role);
        require(calls==std::array<uint32_t,3>{1,1,1},"shared Q6_K expert flow did not run each projection once");
        require(output.size()==input.size(),"shared Q6_K expert flow output shape");
        for(float value:output) require(std::isfinite(value),"non-finite shared Q6_K expert output");
        if(types[0]==types[1] && types[0]==14) {
            auto swapped=role;std::swap(swapped[0],swapped[1]);
            const auto other=run(swapped);bool differs=false;
            for(size_t i=0;i<output.size();++i) differs=differs||output[i]!=other[i];
            require(differs,"exchanging gate and up Q6_K weights left the shared expert output unchanged");
        }
        ++count;
    }
}
}
int main(){try {uint32_t count=0,flows=0;
    for(uint32_t columns:{256u,512u,2560u}) for(uint32_t batch:{1u,3u,8u}) cases(columns,batch,count);
    counterexamples();guards();shared_flows(flows);
    std::cout<<"PASS Q6_K TPC "<<count<<" cases against an independent logical-code oracle and the unmodified pinned ggml Q6_K row decoder, plus "<<flows<<" production shared-expert flows, signed int8 sub-block scales, 4+2-bit ql/qh split, signed super-block d, raw-sum insensitivity and policy guards; no device\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
