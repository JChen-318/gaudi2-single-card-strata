// Routed K-quant packed arithmetic: independent logical codes, the ACTUAL
// production TPC C source, and the unmodified pinned ggml decoder as the
// authoritative byte-layout check. Host only; no ISA, device or performance claim.
#include "strata/hpu/packed_quant.hpp"
#include "strata/hpu/expert_format.hpp"
#include "strata/hpu/packed_expert_flow.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "packed_emulator.hpp"
#include <ggml.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <span>
#include <iostream>
#include <vector>
using namespace strata::hpu;
namespace {
void require(bool ok,const char* why){if(!ok) throw std::runtime_error(why);}
template<class F> void rejects(F f){bool bad=false;try{f();}catch(const std::exception&){bad=true;}require(bad,"invalid K-quant payload or policy accepted");}
int signed_code(uint8_t b){return b<128?b:int(b)-256;}
constexpr uint32_t super=256,groups=8;
uint32_t blocks(uint32_t columns){return columns/super;}
// One row: logical codes per value plus per-superblock d/dmin and the eight
// 32-value group scales and minima in the ranges GGML stores (0..63).
struct KRow {
    uint32_t type,columns;
    std::vector<int> codes;
    std::vector<float> d,dmin;
    std::vector<int> scale,minimum;
};
KRow logical(uint32_t type,uint32_t columns,uint32_t row) {
    const auto count=blocks(columns);KRow w{type,columns,{},{},{},{},{}};
    w.codes.resize(uint64_t(columns));w.d.resize(count);w.dmin.resize(count);
    w.scale.resize(uint64_t(count)*groups);w.minimum.resize(uint64_t(count)*groups);
    const int limit=type==13?32:16;
    for(uint32_t sb=0;sb<count;++sb) {
        w.d[sb]=float(int((sb*7+row*3+3)%9)-4)/16.f;
        w.dmin[sb]=float(int((sb*5+row+1)%7))/64.f;
        for(uint32_t g=0;g<groups;++g) {
            // Values above 15 push two bits into the separate high field, which
            // only the 6-bit combine path can carry.
            w.scale[sb*groups+g]=int((sb*5+g*13+row*11+11)%64);
            w.minimum[sb*groups+g]=int((sb*3+g*17+row*5+5)%64);
        }
    }
    // Codes vary per value, so every superblock holds a distinct pattern.
    for(uint32_t v=0;v<columns;++v) w.codes[v]=int((v*19+(v/super)*5+row*7)%(uint32_t)limit);
    return w;
}
// GGML's own packing rule (quantize_row_q4_K_ref with get_scale_min_k4, checked
// by dequantize_row_q4_K/q5_K): chunk c bytes 32c..32c+31 hold the low nibbles
// of values 64c..64c+31 (group 2c) and the high nibbles of values 64c+32..64c+63
// (group 2c+1). Q5_K stores the fifth bit of both halves in qh[l] bits 4c, 4c+1.
std::vector<uint8_t> encode(const std::vector<KRow>& rows,uint32_t type,uint32_t columns) {
    const uint32_t size=type==12?144:176,qs=type==12?16:48;
    std::vector<uint8_t> bytes(uint64_t(rows.size())*blocks(columns)*size,0);
    for(size_t row=0;row<rows.size();++row) {const auto& w=rows[row];
        for(uint32_t sb=0;sb<blocks(columns);++sb) {
            const size_t at=(uint64_t(row)*blocks(columns)+sb)*size;
            const auto put=[&](size_t offset,float value){const auto bits=strata::kernels::f16_from_f32(value);bytes[offset]=uint8_t(bits);bytes[offset+1]=uint8_t(bits>>8);};
            put(at,w.d[sb]);put(at+2,w.dmin[sb]);
            const uint32_t base=sb*groups;
            for(uint32_t g=0;g<4;++g) {bytes[at+4+g]=uint8_t(w.scale[base+g]&63);bytes[at+8+g]=uint8_t(w.minimum[base+g]&63);}
            for(uint32_t g=0;g<4;++g) {
                const uint32_t j=base+4+g;
                bytes[at+12+g]=uint8_t(uint32_t(w.scale[j]&15u)|(uint32_t(w.minimum[j]&15u)<<4));
                bytes[at+4+g]|=uint8_t(uint32_t(w.scale[j]>>4)<<6);
                bytes[at+8+g]|=uint8_t(uint32_t(w.minimum[j]>>4)<<6);
            }
            for(uint32_t chunk=0;chunk<4;++chunk) for(uint32_t l=0;l<32;++l) {
                const int low=w.codes[sb*super+chunk*64+l],high=w.codes[sb*super+chunk*64+32+l];
                bytes[at+qs+chunk*32+l]=uint8_t(uint32_t(low&15u)|(uint32_t(high&15u)<<4));
                if(type==13) {
                    if(low>=16) bytes[at+16+l]|=uint8_t(1u<<(chunk*2));
                    if(high>=16) bytes[at+16+l]|=uint8_t(1u<<(chunk*2+1));
                }
            }
        }
    }
    return bytes;
}
// One launch per (row,token) index-space cell: exercises partial launches and
// leaves untouched cells as sentinels.
std::vector<float> tpc(const std::vector<uint8_t>& bytes,uint32_t type,uint32_t columns,uint32_t rows,const Q8Activation& q,uint32_t batch,PackedActivation mode=PackedActivation::RoutedQ8) {
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
Q8Activation activate(std::span<const float> values,uint32_t columns) {return quantize_q8_activation(values,columns);}
// Logical-code oracle: per superblock d*sum(d8*sc*dot) - dmin*sum(d8*m*codeSum),
// where codeSum is the sum of the QUANTIZED activation int8 codes. magnitude is
// the sum of the absolute term sizes, which bounds float32 accumulation error
// when the two large terms cancel.
struct Oracle { long double value,magnitude; };
Oracle oracle(const KRow& w,const Q8Activation& q,uint32_t token) {
    Oracle out{0,0};
    for(uint32_t sb=0;sb<blocks(w.columns);++sb) {
        long double sum_d=0,sum_m=0;
        for(uint32_t g=0;g<groups;++g) {
            int dot=0,code_sum=0;
            for(uint32_t l=0;l<32;++l) {const auto a=signed_code(q.codes[uint64_t(token)*w.columns+sb*super+g*32+l]);
                dot+=w.codes[sb*super+g*32+l]*a;code_sum+=a;}
            const long double d8=q.metadata[(uint64_t(token)*w.columns/32+sb*groups+g)*2];
            sum_d+=d8*static_cast<long double>(dot)*w.scale[sb*groups+g];sum_m+=d8*static_cast<long double>(code_sum)*w.minimum[sb*groups+g];
        }
        const long double positive=w.d[sb]*sum_d,negative=w.dmin[sb]*sum_m;
        out.value+=positive-negative;out.magnitude+=std::abs(positive)+std::abs(negative);
    }
    return out;
}
// Routed or shared gate/up plus one Q8_0 role under the shared/head Q8_1
// contract: a 34-byte block with one half d and 32 int8 codes, no raw-input and
// no learned-minimum correction (pinned native_mmvq q8_q8_dot).
struct Q8Row {
    uint32_t columns;
    std::vector<int> codes;
    std::vector<float> d;
};
Q8Row q8_logical(uint32_t columns,uint32_t row) {
    Q8Row w{columns,{},std::vector<float>(columns/32)};w.codes.resize(columns);
    for(uint32_t b=0;b<w.d.size();++b) w.d[b]=float(int((b*3+row*5+2)%9)-4)/16.f;
    for(uint32_t v=0;v<columns;++v) w.codes[v]=int((v*13+(v/32)*7+row*11+3)%249)-124;
    return w;
}
std::vector<uint8_t> encode_q8(const std::vector<Q8Row>& rows) {
    const auto blocks=rows.front().d.size();
    std::vector<uint8_t> bytes(uint64_t(rows.size())*blocks*packed_block_bytes(8),0);
    for(size_t row=0;row<rows.size();++row) for(uint32_t b=0;b<blocks;++b) {
        const size_t at=(uint64_t(row)*blocks+b)*packed_block_bytes(8);
        const auto bits=strata::kernels::f16_from_f32(rows[row].d[b]);
        bytes[at]=uint8_t(bits);bytes[at+1]=uint8_t(bits>>8);
        for(uint32_t j=0;j<32;++j) bytes[at+2+j]=uint8_t(rows[row].codes[b*32+j]);
    }
    return bytes;
}
Oracle q8_oracle(const Q8Row& w,const Q8Activation& q,uint32_t token) {
    Oracle out{0,0};
    for(uint32_t b=0;b<w.d.size();++b) {
        long double dot=0;
        for(uint32_t j=0;j<32;++j) dot+=static_cast<long double>(w.codes[b*32+j])*signed_code(q.codes[uint64_t(token)*w.columns+b*32+j]);
        const long double term=w.d[b]*q.metadata[(uint64_t(token)*w.d.size()+b)*2]*dot;
        out.value+=term;out.magnitude+=std::abs(term);
    }
    return out;
}
// One projection role: the raw GGML bytes uploaded to HBM plus the logical
// weights and scales that those bytes must represent.
struct Role {
    uint32_t type,columns,rows;
    std::vector<uint8_t> bytes;
    std::vector<KRow> k;
    std::vector<Q8Row> q8;
};
// Each role gets its own logical fixture so a gate/up swap cannot alias.
Role k_role(uint32_t type,uint32_t columns,uint32_t rows,uint32_t offset) {
    Role r{type,columns,rows,{},{},{}};
    for(uint32_t row=0;row<rows;++row) r.k.push_back(logical(type,columns,row+offset));
    r.bytes=encode(r.k,type,columns);validate_packed_rows(r.bytes,type,columns,rows);return r;
}
Role q8_role(uint32_t columns,uint32_t rows,uint32_t offset) {
    Role r{8,columns,rows,{},{},{}};
    for(uint32_t row=0;row<rows;++row) r.q8.push_back(q8_logical(columns,row+offset));
    r.bytes=encode_q8(r.q8);validate_packed_rows(r.bytes,8,columns,rows);return r;
}
// Both branches consume the same q8_1 activation bytes, so the comparison
// isolates integer-dot arithmetic instead of quantization boundary flips.
std::pair<std::vector<float>,std::vector<float>> role_logical(const Role& role,const Q8Activation& q,uint32_t batch) {
    std::vector<float> values(uint64_t(role.rows)*batch),magnitudes(values.size());
    for(uint32_t token=0;token<batch;++token) for(uint32_t row=0;row<role.rows;++row) {
        const auto expected=role.type==8?q8_oracle(role.q8[row],q,token):oracle(role.k[row],q,token);
        values[uint64_t(token)*role.rows+row]=float(expected.value);magnitudes[uint64_t(token)*role.rows+row]=float(expected.magnitude);
    }
    return {values,magnitudes};
}
std::vector<float> role_tpc(const Role& role,const Q8Activation& q,uint32_t batch,PackedActivation mode) {
    return tpc(role.bytes,role.type,role.columns,role.rows,q,batch,mode);
}
// Production shared-expert flow: gate/up -> SiLU*up -> down, every role on the
// head/shared Q8_1 contract with K-quant superblocks kept as raw bytes.
void shared_flows(uint32_t& count) {
    constexpr uint32_t hidden=512,intermediate=256;
    for(auto types:{std::array<uint32_t,3>{12,12,12},std::array<uint32_t,3>{12,13,12},std::array<uint32_t,3>{13,12,13},
                   std::array<uint32_t,3>{13,13,12},std::array<uint32_t,3>{12,13,8},std::array<uint32_t,3>{13,12,8}})
    for(uint32_t batch:{1u,3u}) {
        const auto format=expert_format(hidden,intermediate,-1,types);
        require(format.packed && format.activation==PackedActivation::Q8_1,"shared K-quant expert lost the head contract");
        std::array<Role,3> role{k_role(types[0],hidden,intermediate,0),k_role(types[1],hidden,intermediate,64),
            types[2]==8?q8_role(intermediate,hidden,128):k_role(types[2],intermediate,hidden,128)};
        const uint64_t charge=packed_row_bytes(types[0],hidden)*intermediate+packed_row_bytes(types[1],hidden)*intermediate
            +packed_row_bytes(types[2],intermediate)*hidden;
        require(format.bytes==charge,"shared K-quant expert payload mischarged");
        std::vector<float> input(uint64_t(batch)*hidden);
        for(size_t i=0;i<input.size();++i) input[i]=float(int((i*29+batch)%113)-56)*.0625f;
        std::array<uint32_t,3> calls{0,0,0};
        std::array<std::vector<float>,2> previous{};
        bool saw_down_input=false;
        auto projection=[&](uint32_t which,std::span<const float> values,const std::array<Role,3>& roles) {
            const auto mode=expert_projection_activation(format,which);
            require(mode==PackedActivation::Q8_1,"shared projection inherited a routed activation contract");
            const auto tokens=uint32_t(values.size()/roles[which].columns);
            const auto q=quantize_q8_activation(values,roles[which].columns);
            const auto expected=role_logical(roles[which],q,tokens);
            const auto actual=role_tpc(roles[which],q,tokens,mode);
            for(size_t i=0;i<actual.size();++i) {
                require(std::isfinite(actual[i]),"non-finite shared K-quant projection");
                require(std::abs(double(actual[i])-double(expected.first[i]))<=std::max(.01,expected.second[i]*2e-4),
                    "shared expert projection differs from the independent logical-code oracle");
            }
            return actual;
        };
        auto run=[&](const std::array<Role,3>& roles) {
            calls={0,0,0};saw_down_input=false;previous={};
            return execute_packed_expert(std::span<const float>(input),hidden,intermediate,[&](uint32_t which,std::span<const float> values) {
                ++calls[which];
                if(which==2 && !saw_down_input) {
                    // The down projection must consume the flow's own SiLU*up product.
                    require(values.size()==previous[0].size() && previous[1].size()==values.size(),"down projection lost the gated intermediate");
                    for(size_t i=0;i<values.size();++i) {
                        const float gated=(previous[0][i]/(1.0f+std::exp(-previous[0][i])))*previous[1][i];
                        require(std::abs(values[i]-gated)<=1e-4f*std::max(1.0f,std::abs(gated)),"SiLU gate contract changed");
                    }
                    saw_down_input=true;
                }
                auto result=projection(which,values,roles);
                if(which<2) previous[which]=result;
                return result;
            });
        };
        const auto output=run(role);
        require(calls==std::array<uint32_t,3>{1,1,1},"shared expert flow did not run each projection once");
        require(saw_down_input && output.size()==input.size(),"shared expert flow output shape");
        for(float value:output) require(std::isfinite(value),"non-finite shared K-quant expert output");
        // The same rows executed one at a time must reproduce the batched result.
        for(uint32_t token=0;token<batch;++token) {
            std::vector<float> one(input.begin()+uint64_t(token)*hidden,input.begin()+uint64_t(token+1)*hidden);
            std::array<std::vector<float>,2> single{};
            const auto result=execute_packed_expert(std::span<const float>(one),hidden,intermediate,[&](uint32_t which,std::span<const float> values) {
                auto partial=projection(which,values,role);
                if(which<2) single[which]=partial;
                return partial;
            });
            require(!single[0].empty() && !single[1].empty(),"single-row shared expert lost an intermediate");
            require(!std::memcmp(result.data(),output.data()+uint64_t(token)*hidden,hidden*sizeof(float)),"batched shared expert crossed token rows");
        }
        if(types[0]==types[1]) {
            // A layout-identical gate/up swap must change the expert output.
            auto swapped=role;std::swap(swapped[0],swapped[1]);
            const auto other=run(swapped);
            bool differs=false;
            for(size_t i=0;i<output.size();++i) differs=differs||output[i]!=other[i];
            require(differs,"exchanging gate and up weights left the shared expert output unchanged");
            require(calls==std::array<uint32_t,3>{1,1,1},"swapped shared expert flow skipped a projection");
        }
        ++count;
    }
}
// Authoritative layout check: decode with the unmodified pinned ggml row
// decoder, then dot with the values q8_1 actually represents (d8 times code).
Oracle decoded(const std::vector<uint8_t>& bytes,uint32_t type,uint32_t columns,uint32_t row,const Q8Activation& q,uint32_t token) {
    const auto* traits=ggml_get_type_traits(type==12?GGML_TYPE_Q4_K:GGML_TYPE_Q5_K);
    require(traits && traits->blck_size==int(super) && traits->to_float,"pinned ggml lacks the K-quant row decoder");
    require(traits->type_size==(type==12?144u:176u),"pinned ggml K-quant block size differs from the packed charge");
    std::vector<float> weights(columns);
    traits->to_float(bytes.data()+uint64_t(row)*packed_row_bytes(type,columns),weights.data(),int64_t(columns));
    Oracle out{0,0};
    for(uint32_t sb=0;sb<blocks(columns);++sb) for(uint32_t g=0;g<groups;++g) {
        const long double d8=q.metadata[(uint64_t(token)*columns/32+sb*groups+g)*2];long double part=0;
        for(uint32_t l=0;l<32;++l) part+=static_cast<long double>(weights[sb*super+g*32+l])*signed_code(q.codes[uint64_t(token)*columns+sb*super+g*32+l]);
        const long double term=d8*part;out.value+=term;out.magnitude+=std::abs(term);
    }
    return out;
}
// The TPC accumulates in float32; bound its error by the magnitude of the terms
// it cancels, not by the (possibly much smaller) result.
bool close(float actual,const Oracle& expected) {
    if(!std::isfinite(actual)) return false;
    return std::abs(static_cast<long double>(actual)-expected.value)<=std::max<long double>(.01L,expected.magnitude*2e-4L);
}
std::vector<float> source(uint64_t values,uint32_t seed) {
    std::vector<float> out(values);for(uint64_t i=0;i<values;++i) out[i]=float(int((i*23+seed)%127)-63)*.03125f+.0140625f;return out;
}
void cases(uint32_t type,uint32_t columns,uint32_t batch,uint32_t& count) {
    constexpr uint32_t rows=3;std::vector<KRow> logical_rows;
    for(uint32_t row=0;row<rows;++row) logical_rows.push_back(logical(type,columns,row));
    const auto bytes=encode(logical_rows,type,columns);
    validate_packed_rows(bytes,type,columns,rows);
    const auto values=source(uint64_t(batch)*columns,batch);
    auto q=activate(values,columns);
    const auto actual=tpc(bytes,type,columns,rows,q,batch);
    for(uint32_t token=0;token<batch;++token) for(uint32_t row=0;row<rows;++row) {
        const auto result=actual[uint64_t(token)*rows+row];
        require(close(result,oracle(logical_rows[row],q,token)),"routed K-quant TPC differs from the independent logical-code oracle");
        require(close(result,decoded(bytes,type,columns,row,q,token)),"routed K-quant bytes disagree with the unmodified pinned ggml decoder");
        // Shared/head (Q8_1) and routed (RoutedQ8) graphs consume identical q8_1
        // activation bytes, so the superblock dot must give the same value.
        const auto other=tpc(bytes,type,columns,rows,q,batch,PackedActivation::Q8_1);
        require(other[uint64_t(token)*rows+row]==result,"K-quant dot differs between the routed and shared/head q8_1 contracts");
        // The same row against one-token activations must be bit-identical.
        std::vector<float> one(values.begin()+uint64_t(token)*columns,values.begin()+uint64_t(token)*columns+columns);
        const auto isolated=tpc(bytes,type,columns,rows,activate(one,columns),1);
        require(isolated[row]==result,"batched routed K-quant changed single-row arithmetic or crossed tokens");
        ++count;
    }
}
void counterexamples() {
    const auto w=logical(12,super,0);const auto bytes=encode({w},12,super);
    const auto values=source(super,7);auto q=activate(values,super);
    const auto before=tpc(bytes,12,super,1,q,1)[0];
    require(close(before,oracle(w,q,0)),"K-quant baseline disagrees with its oracle");
    const auto raw=q;for(uint32_t g=0;g<groups;++g) const_cast<Q8Activation&>(raw).metadata[g*2+1]=1234.5f;   // Q8_1 raw sums
    require(tpc(bytes,12,super,1,q,1)[0]==before,"routed K-quant output moved when only raw sums changed");
    require(std::any_of(w.scale.begin(),w.scale.end(),[](int v){return v>15;}) && std::any_of(w.minimum.begin(),w.minimum.end(),[](int v){return v>15;}),
        "fixture never exercised the two high bits of a 6-bit K-quant field");
    auto swapped=w;std::swap(swapped.scale,swapped.minimum);
    require(tpc(encode({swapped},12,super),12,super,1,q,1)[0]!=before,"exchanging scales and minima did not change the result");
    auto shifted=w;shifted.codes[9]=(shifted.codes[9]+1)%16;
    require(tpc(encode({shifted},12,super),12,super,1,q,1)[0]!=before,"a single logical code change was absorbed by the group scale");
    auto zero_min=w;std::fill(zero_min.minimum.begin(),zero_min.minimum.end(),0);
    const auto zeroed=tpc(encode({zero_min},12,super),12,super,1,q,1)[0];
    require(close(zeroed,oracle(zero_min,q,0)) && zeroed!=before,"zeroing the learned minima did not remove exactly the min term");
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
    const auto w=logical(12,super,0);auto bytes=encode({w},12,super);
    bytes[2]=0;bytes[3]=0x7c;rejects([&]{validate_packed_rows(bytes,12,super,1);});
    bytes[2]=0;bytes[3]=0x3c;bytes[0]=0;bytes[1]=0x7c;rejects([&]{validate_packed_rows(bytes,12,super,1);});
    bytes[0]=0;bytes[1]=0x3c;validate_packed_rows(bytes,12,super,1);
    rejects([&]{(void)packed_row_bytes(12,640);});rejects([&]{(void)packed_row_bytes(13,300);});rejects([&]{(void)packed_values_per_block(30);});
    // The K family is now complete in policy: Q3_K(11), Q4_K(12), Q5_K(13) and
    // Q6_K(14) all take the 256-value integer dot. Only the IQ tables are missing.
    require(packed_activation_supported(12,PackedActivation::Q8_1) && packed_activation_supported(13,PackedActivation::Q8_1)
        && packed_activation_supported(12,PackedActivation::RoutedQ8) && !packed_activation_supported(13,PackedActivation::RoutedS2)
        && !packed_activation_supported(42,PackedActivation::RoutedQ8) && packed_activation_supported(42,PackedActivation::Q8_1)
        && packed_supported(2) && packed_supported(11) && packed_supported(12) && packed_supported(13) && packed_supported(14)
       ,"K-quant policy or still-missing IQ format set wrong");
    const auto routed=expert_format(2560,640,0,{12,12,8});
    require(routed.packed && routed.activation==PackedActivation::RoutedQ8 && routed.bytes==uint64_t(1440)*640*2+uint64_t(34)*20*2560,"routed K-quant charge wrong");
    const auto shared=expert_format(2560,640,-1,{12,13,8}),down=expert_format(2560,640,0,{2,2,12});
    require(shared.packed && shared.activation==PackedActivation::Q8_1
        && shared.bytes==uint64_t(1440)*640+uint64_t(1760)*640+uint64_t(34)*20*2560,"shared K-quant expert was not packed or is mischarged");
    require(expert_projection_activation(shared,0)==PackedActivation::Q8_1 && expert_projection_activation(shared,1)==PackedActivation::Q8_1
        && expert_projection_activation(shared,2)==PackedActivation::Q8_1,"shared expert inherited a routed projection policy");
    require(!down.packed && down.bytes==uint64_t(2560)*640*6,"K-quant down projection with 640 columns was not rejected by layout");
    const auto mixed=expert_format(2560,640,0,{42,12,8});
    require(mixed.packed && mixed.activation==PackedActivation::RoutedQ8,"mixed Q2/K routed expert lost packed arithmetic");
    require(expert_projection_activation(mixed,0)==PackedActivation::RoutedS2 && expert_projection_activation(mixed,1)==PackedActivation::RoutedQ8
        && expert_projection_activation(mixed,2)==PackedActivation::RoutedQ8,"mixed projection policy assigned the wrong quantizer");
    require(packed_row_bytes(12,2560)==1440 && packed_row_bytes(13,2560)==1760 && packed_row_bytes(12,256)==144 && packed_row_bytes(13,256)==176
        && packed_block_bytes(12)==144 && packed_block_bytes(13)==176
        && packed_values_per_block(12)==super && packed_values_per_block(42)==64 && packed_values_per_block(2)==32,"packed K-quant geometry wrong");
    rejects([&]{(void)expert_projection_activation(expert_format(2560,640,0,{2,2,12}),0);});
    rejects([&]{(void)expert_projection_activation(expert_format(2560,640,-1,{12,12,12}),2);});
}
}
int main(){try {uint32_t count=0;
    for(uint32_t type:{12u,13u}) for(uint32_t columns:{256u,512u,2560u}) for(uint32_t batch:{1u,3u,8u}) cases(type,columns,batch,count);
    uint32_t flows=0;shared_flows(flows);
    counterexamples();guards();
    std::cout<<"PASS routed K-quant TPC "<<count<<" Q4_K/Q5_K cases against independent logical-code and unmodified pinned ggml decoder oracles, 6-bit scale/min combine, qh fifth-bit layout, code-sum minima counterexamples, finite half guards, shared/head contract equivalence and "<<flows<<" production shared-expert K-quant flows\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
